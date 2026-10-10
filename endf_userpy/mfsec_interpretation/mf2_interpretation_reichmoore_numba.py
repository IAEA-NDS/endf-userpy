"""Numba-compiled Reich-Moore kernel for
:mod:`mf2_interpretation_reichmoore`.

Same three-backend pattern as :mod:`mf2_interpretation_mlbw_numba`:
numpy / JAX share one implementation through the array-namespace
adapter, and this module provides a dedicated ``@njit`` fused
per-energy kernel for the ``'numba'`` backend.

Scope
-----

- Elastic + capture + up to 2 fission channels per J·π group,
  matching the sketch's :class:`RMData` layout.
- Arbitrary non-negative L: closed forms for L in 0..5 stay
  inline for hot-path speed; L >= 6 goes through the Newton
  recurrence in :func:`pnt_shf_any_L` /
  :func:`phase_any_L` (:mod:`mf2_interpretation_factors_numba`),
  matching the numpy / jax path via
  :func:`mf2_interpretation_factors.newton_step_pnt_shf` /
  :func:`newton_step_phase`.
- Same physics as the numpy path: no level shift on the R-matrix
  denominator, matching ENDF-6 R-M (``L̃_c = i P_c(E)``) and NJOY
  reconr's csrmat.
- Hand-coded 1×1 / 2×2 / 3×3 solves for ``(I - i R P) X = R``.
  Faster than calling numba's ``np.linalg.solve`` per (E, group)
  under ``parallel=True`` (which serialises through LAPACK) and
  keeps the whole kernel in numba-native arithmetic.

The scalar factor helpers ``low_L_pnt_shf`` and ``low_L_phase``
live in :mod:`mf2_interpretation_factors_numba`, shared with the
MLBW numba path. ``@njit(inline='always')`` fuses each call into
the per-energy kernel body at compile time, so importing them
from a shared module costs nothing at runtime.
"""
from __future__ import annotations

import math

import numpy as np

from ..primitives import array_ns
from ..primitives import tab1 as tab1_mod
from .mf2_interpretation_factors_numba import (
    pnt_shf_any_L as _pnt_shf,
    phase_any_L as _phase,
)


# Newton-recurrence step (pure Python; identical formula to
# :func:`mf2_interpretation_factors.newton_step_pnt_shf`). Used
# by the wrapper's ``_pnt_shf_scalar`` for L >= 6 so the wrapper
# does not require numba to be installed at import time.
def _newton_step_pnt_shf_py(p_prev, s_prev, r2, L):
    sdif = L - s_prev
    ratio = r2 / (sdif * sdif + p_prev * p_prev)
    return ratio * p_prev, ratio * sdif - L

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*a, **kw):
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        return lambda f: f

    prange = range


_EPS = 1e-38


@njit(cache=True, inline='always')
def accumulate_r_matrix(E, er, hg, a00, a01, a11, a02, a12, a22, nch):
    """Upper triangle of one J-group's symmetric R-matrix at energy ``E``.

    ``R_cd = sum_r a_cd[r] / (er[r] - E - i hg[r])`` with the width
    products ``a_cd = gamma_c * gamma_d`` and the half radiation width
    ``hg = Gamma_gamma / 2`` precomputed per resonance; the arrays are
    the group's contiguous slices. ``nch`` (1..3) is the number of
    channels; entries of absent channels are returned as 0.

    The sums are carried as real / imaginary scalars with
    ``1 / (dr - i h) = (dr + i h) / (dr**2 + h**2)`` and loops over
    ``range(slice length)``, which lets LLVM vectorise them; complex
    accumulators or ``range(start, end)`` over the full arrays do not
    vectorise (~4-6x slower). The reciprocal is written as a power
    so the per-division zero check of numba's default error model
    does not block vectorisation; a zero denominator (``E == E_r``
    with a zero radiation width) is counted instead and returned as
    ``nzero`` so the caller can flag those energies (NaN) outside the
    parallel loop.
    """
    r00r = 0.0
    r00i = 0.0
    r01r = 0.0
    r01i = 0.0
    r11r = 0.0
    r11i = 0.0
    r02r = 0.0
    r02i = 0.0
    r12r = 0.0
    r12i = 0.0
    r22r = 0.0
    r22i = 0.0
    nzero = 0.0
    n = er.shape[0]
    if nch == 1:
        for r in range(n):
            dr = er[r] - E
            h = hg[r]
            d = dr * dr + h * h
            nzero += 1.0 if d == 0.0 else 0.0
            q = d ** -1.0
            cr = dr * q
            ci = h * q
            r00r += a00[r] * cr
            r00i += a00[r] * ci
    elif nch == 2:
        for r in range(n):
            dr = er[r] - E
            h = hg[r]
            d = dr * dr + h * h
            nzero += 1.0 if d == 0.0 else 0.0
            q = d ** -1.0
            cr = dr * q
            ci = h * q
            r00r += a00[r] * cr
            r00i += a00[r] * ci
            r01r += a01[r] * cr
            r01i += a01[r] * ci
            r11r += a11[r] * cr
            r11i += a11[r] * ci
    else:
        for r in range(n):
            dr = er[r] - E
            h = hg[r]
            d = dr * dr + h * h
            nzero += 1.0 if d == 0.0 else 0.0
            q = d ** -1.0
            cr = dr * q
            ci = h * q
            r00r += a00[r] * cr
            r00i += a00[r] * ci
            r01r += a01[r] * cr
            r01i += a01[r] * ci
            r11r += a11[r] * cr
            r11i += a11[r] * ci
            r02r += a02[r] * cr
            r02i += a02[r] * ci
            r12r += a12[r] * cr
            r12i += a12[r] * ci
            r22r += a22[r] * cr
            r22i += a22[r] * ci
    return (complex(r00r, r00i), complex(r01r, r01i), complex(r11r, r11i),
            complex(r02r, r02i), complex(r12r, r12i), complex(r22r, r22i),
            nzero)


@njit(cache=True, inline='always')
def capture_deficit_row(w0, w1, w2, R00, R01, R11, R02, R12, R22):
    """``w Im(R) w^H`` for the incident row ``w = (w0, w1, w2)`` of
    ``W⁻¹`` (unused channels zero), ``R`` symmetric. Times
    ``4 P_incident`` this is ``1 - Σ_c |U_{inc,c}|²`` without the
    cancelling subtraction; see
    :func:`mf2_interpretation_reichmoore.capture_unitarity_deficit`."""
    a00 = w0.real * w0.real + w0.imag * w0.imag
    a11 = w1.real * w1.real + w1.imag * w1.imag
    a22 = w2.real * w2.real + w2.imag * w2.imag
    a01 = w0.real * w1.real + w0.imag * w1.imag      # Re(w0 conj(w1))
    a02 = w0.real * w2.real + w0.imag * w2.imag
    a12 = w1.real * w2.real + w1.imag * w2.imag
    return (a00 * R00.imag + a11 * R11.imag + a22 * R22.imag
            + 2.0 * (a01 * R01.imag + a02 * R02.imag + a12 * R12.imag))


@njit(cache=True, parallel=True, fastmath=True)
def _reconstruct_kernel(
    e,                     # (ne,) float64
    group_r_a,             # (ngroups,) channel radius per group (scalar)
    group_r_ap,            # (ngroups,) scattering radius per group (scalar)
    abn, ki,
    group_l,               # (ngroups,) int
    group_g,               # (ngroups,) float
    group_nfis,            # (ngroups,) int (0, 1, or 2)
    group_res_start,       # (ngroups,) int  -- contiguous slice into res_*
    group_res_end,         # (ngroups,) int
    res_er,                # (nres,) float
    res_hg,                # (nres,) float  -- Gamma_gamma / 2
    a00, a01, a11,         # (nres,) float  -- reduced-width products
    a02, a12, a22,         #   gamma_c * gamma_d (n = 0, f1 = 1, f2 = 2)
):
    ne = e.shape[0]
    ngroups = group_l.shape[0]

    sct = np.zeros(ne)
    cap = np.zeros(ne)
    fis = np.zeros(ne)
    pot = np.zeros(ne)
    nzero = np.zeros(ne)

    # --- Parallel per-energy loop. Each thread computes all groups
    # at one energy. Per-group R-matrix and U-matrix stay in
    # thread-local complex scalars; no (ne, ..., ...) tensors are
    # materialised.
    for i in prange(ne):
        E = e[i]
        E_safe = E if E > 0.0 else 0.0
        if E > 0.0:
            k2 = (ki * ki) * E_safe
            pi_k2 = abn * math.pi / k2
        else:
            pi_k2 = 0.0

        sct_sum = 0.0
        cap_sum = 0.0
        fis_sum = 0.0
        pot_sum = 0.0

        sqrt_E = math.sqrt(E_safe)
        nzero_i = 0.0

        for g in range(ngroups):
            L = group_l[g]
            gJ = group_g[g]
            nfis = group_nfis[g]

            # Per-group radii: each L may have a different APL
            # override. Compute rho_a / rho_ap here per group
            # rather than at the outer per-energy level.
            rho_a_i = ki * sqrt_E * group_r_a[g]
            rho_ap_i = ki * sqrt_E * group_r_ap[g]

            # Elastic-channel factors at E. R-matrix penetration uses
            # the channel radius (rho_a); the hard-sphere phase in Ω
            # uses the scattering radius (rho_ap). These differ only
            # for NAPS=2 evaluations but the physics is identical.
            # No shift is used: ENDF-6 R-M defines L̃_c = i P_c(E)
            # (pure imaginary), so the S(E) - S(|E_r|) level-shift
            # correction that appears in MLBW does NOT appear here;
            # NJOY reconr's csrmat confirms `diff = er - e` with no
            # shift term. The (_, shf_e) discard on _pnt_shf keeps the
            # numba kernel signature stable.
            p_e, _shf_e = _pnt_shf(rho_a_i, L)
            phi_e = _phase(rho_ap_i, L)

            # Potential scattering contribution: 4π/k² Σ_J g_J sin²(φ_L).
            sin_phi = math.sin(phi_e)
            pot_sum += gJ * sin_phi * sin_phi

            # --- Build R-matrix contributions from this group's resonances. ---
            # R is (nch × nch) complex, symmetric. We only carry the
            # 6 upper-triangle entries R_00, R_01, R_02, R_11, R_12, R_22
            # as scalar accumulators; unused entries for nch<3 stay 0.
            # No shift correction: ENDF-6 R-M denominator is
            # ``E_r - E - i Γ_γ / 2`` (matches NJOY reconr csrmat
            # line 3350: `diff = er - e`). Reduced-width amplitudes
            # are already computed from Γ_n(|E_r|) in the wrapper.
            r0 = group_res_start[g]
            r1 = group_res_end[g]
            R00, R01, R11, R02, R12, R22, nz = accumulate_r_matrix(
                E_safe, res_er[r0:r1], res_hg[r0:r1],
                a00[r0:r1], a01[r0:r1], a11[r0:r1],
                a02[r0:r1], a12[r0:r1], a22[r0:r1], nfis + 1,
            )
            nzero_i += nz

            # --- Solve (I - i R P) X = R for row 0 of X. ---
            # P = diag(p_e, 1, 1) with fission channels having P=1.
            # Small-matrix branches, hand-coded for nch=1, 2, 3.
            omega1 = complex(math.cos(-phi_e), math.sin(-phi_e))
            omega2 = omega1 * omega1
            sqrt_pe = math.sqrt(p_e) if p_e > 0.0 else 0.0

            if nfis == 0:
                # nch = 1: scalar inverse.
                W00 = 1.0 - 1j * R00 * p_e
                X00 = R00 / W00
                U00 = omega2 * (1.0 + 2j * p_e * X00)
                sct_g = pi_k2 * gJ * ((1.0 - U00.real) ** 2 + U00.imag ** 2)
                fis_g = 0.0
                cap_g = 4.0 * p_e * pi_k2 * gJ * capture_deficit_row(
                    1.0 / W00, 0j, 0j, R00, R01, R11, R02, R12, R22)
            elif nfis == 1:
                # nch = 2: 2×2 inverse via adjugate.
                W00 = 1.0 - 1j * R00 * p_e
                W01 = -1j * R01
                W10 = -1j * R01 * p_e       # R symmetric: R_10 = R_01
                W11 = 1.0 - 1j * R11
                det = W00 * W11 - W01 * W10
                X00 = (W11 * R00 - W01 * R01) / det
                X01 = (W11 * R01 - W01 * R11) / det
                U00 = omega2 * (1.0 + 2j * p_e * X00)
                U01 = omega1 * (2j * sqrt_pe * X01)
                sct_g = pi_k2 * gJ * ((1.0 - U00.real) ** 2 + U00.imag ** 2)
                s01 = U01.real * U01.real + U01.imag * U01.imag
                fis_g = pi_k2 * gJ * s01
                # Capture = (π/k²) g (1 - Σ_c|U_0c|²), evaluated without
                # the subtraction; row 0 of W⁻¹ = (W11, -W01) / det.
                cap_g = 4.0 * p_e * pi_k2 * gJ * capture_deficit_row(
                    W11 / det, -W01 / det, 0j, R00, R01, R11, R02, R12, R22)
            else:
                # nch = 3: 3×3 inverse via cofactor expansion.
                W00 = 1.0 - 1j * R00 * p_e
                W01 = -1j * R01
                W02 = -1j * R02
                W10 = -1j * R01 * p_e
                W11 = 1.0 - 1j * R11
                W12 = -1j * R12
                W20 = -1j * R02 * p_e
                W21 = -1j * R12
                W22 = 1.0 - 1j * R22
                m00 = W11 * W22 - W12 * W21
                m01 = W10 * W22 - W12 * W20
                m02 = W10 * W21 - W11 * W20
                det = W00 * m00 - W01 * m01 + W02 * m02
                # Row 0 of W^{-1}: [C00, C10, C20] / det with
                # C_ij = (-1)^{i+j} M_{ij}.
                Winv00 = m00 / det
                Winv01 = -(W01 * W22 - W02 * W21) / det
                Winv02 = (W01 * W12 - W02 * W11) / det
                # X_row0 = Winv_row0 @ R  (R symmetric).
                X00 = Winv00 * R00 + Winv01 * R01 + Winv02 * R02
                X01 = Winv00 * R01 + Winv01 * R11 + Winv02 * R12
                X02 = Winv00 * R02 + Winv01 * R12 + Winv02 * R22
                U00 = omega2 * (1.0 + 2j * p_e * X00)
                U01 = omega1 * (2j * sqrt_pe * X01)
                U02 = omega1 * (2j * sqrt_pe * X02)
                sct_g = pi_k2 * gJ * ((1.0 - U00.real) ** 2 + U00.imag ** 2)
                s01 = U01.real * U01.real + U01.imag * U01.imag
                s02 = U02.real * U02.real + U02.imag * U02.imag
                fis_g = pi_k2 * gJ * (s01 + s02)
                cap_g = 4.0 * p_e * pi_k2 * gJ * capture_deficit_row(
                    Winv00, Winv01, Winv02, R00, R01, R11, R02, R12, R22)

            sct_sum += sct_g
            cap_sum += cap_g
            fis_sum += fis_g

        sct[i] = sct_sum
        cap[i] = cap_sum
        fis[i] = fis_sum
        pot[i] = 4.0 * pi_k2 * pot_sum
        nzero[i] = nzero_i

    return sct, cap, fis, pot, nzero


def reconstruct(data, energies_in):
    """Reich-Moore cross sections via the numba-compiled kernel.

    Signature-compatible with
    :func:`mf2_interpretation_reichmoore.reconstruct` minus the
    backend argument (implicit here).

    The wrapper's job:

    - Pre-interpolate ``r_a`` and ``r_ap`` on the query grid (numba
      can't call the TAB1 helper directly).
    - Sort resonances by group so each group's resonances are
      contiguous, then compute per-group start / end indices.
    - Pre-compute signed reduced-width amplitudes ``γ_n``, ``γ_{f,1}``,
      ``γ_{f,2}``: elastic amplitudes need per-resonance
      ``P_L(|E_r|)`` which requires ``r_a`` interpolated at ``|E_r|``.
    - Validate L range, then dispatch.
    """
    if not HAS_NUMBA:
        raise RuntimeError(
            "MF2 Reich-Moore numba backend requested but `numba` is "
            "not installed. `pip install numba` or use "
            "`array_ns.get_backend('numpy')`."
        )

    e = np.asarray(energies_in, dtype=np.float64)
    er = np.asarray(data.res_er, dtype=np.float64)
    gg = np.asarray(data.res_gg, dtype=np.float64)
    gn = np.asarray(data.res_gn, dtype=np.float64)
    gf1 = np.asarray(data.res_gf1, dtype=np.float64)
    gf2 = np.asarray(data.res_gf2, dtype=np.float64)
    res_group = np.asarray(data.res_group, dtype=np.int64)

    group_l = np.asarray(data.group_l, dtype=np.int64)
    group_g = np.asarray(data.group_g, dtype=np.float64)
    group_nfis = np.asarray(data.group_nfis, dtype=np.int64)
    ngroups = group_l.shape[0]

    # --- Per-group scalar radii. If the preproc supplied per-L
    # (data.group_r_a / data.group_r_ap), use them directly.
    # Otherwise fall back to interpolating the range-level TAB1
    # at a single reference energy (this is the old behaviour
    # and is only correct when every L shares the same AP).
    xp = array_ns.get_backend('numpy')
    if getattr(data, 'group_r_a', None) is not None:
        group_r_a_arr = np.asarray(data.group_r_a, dtype=np.float64)
    else:
        # Fallback: constant TAB1 at any energy -> the same scalar
        # for every group.
        ref_val = float(tab1_mod.interp(
            data.r_a, np.array([1.0]), xp,
        )[0])
        group_r_a_arr = np.full(ngroups, ref_val, dtype=np.float64)
    if getattr(data, 'group_r_ap', None) is not None:
        group_r_ap_arr = np.asarray(data.group_r_ap, dtype=np.float64)
    else:
        ref_val = float(tab1_mod.interp(
            data.r_ap, np.array([1.0]), xp,
        )[0])
        group_r_ap_arr = np.full(ngroups, ref_val, dtype=np.float64)

    # Per-resonance channel radius at |E_r|: each resonance uses
    # its OWN group's r_a (constant per L). Bug fix vs the
    # previous "single r_a for all resonances" behaviour.
    res_group_np = np.asarray(data.res_group, dtype=np.int64)
    r_a_at_er = group_r_a_arr[res_group_np]

    # --- Sort resonances by group so each group is contiguous. ---
    order = np.argsort(res_group, kind='stable')
    er_s = er[order]
    gg_s = gg[order]
    gn_s = gn[order]
    gf1_s = gf1[order]
    gf2_s = gf2[order]
    res_group_s = res_group[order]
    r_a_at_er_s = r_a_at_er[order]

    # --- Per-group start / end into the sorted arrays. ---
    group_res_start = np.zeros(ngroups, dtype=np.int64)
    group_res_end = np.zeros(ngroups, dtype=np.int64)
    if res_group_s.size:
        for g in range(ngroups):
            group_res_start[g] = int(np.searchsorted(res_group_s, g, side='left'))
            group_res_end[g] = int(np.searchsorted(res_group_s, g, side='right'))

    # --- Precompute reduced-width amplitudes. ---
    # γ_n = sign(GN) * sqrt(|GN| / (2 * P_L(|E_r|)))
    # γ_{f,c} = sign(GF_c) * sqrt(|GF_c| / 2)   (P=1 for fission)
    # No shift factor is needed: ENDF-6 R-M does not apply a level
    # shift on the R-matrix denominator (NJOY reconr csrmat matches).
    nres = er_s.shape[0]
    gamma_n = np.zeros(nres, dtype=np.float64)
    gamma_f1 = np.zeros(nres, dtype=np.float64)
    gamma_f2 = np.zeros(nres, dtype=np.float64)
    for r in range(nres):
        g_idx = int(res_group_s[r])
        L = int(group_l[g_idx])
        rho_at_er = data.ki * math.sqrt(abs(er_s[r])) * r_a_at_er_s[r]
        p_r, _s_r = _pnt_shf_scalar(rho_at_er, L)
        if p_r > _EPS:
            gamma_n[r] = math.copysign(
                math.sqrt(abs(gn_s[r]) / (2.0 * p_r)), gn_s[r],
            )
        if gf1_s[r] != 0.0:
            gamma_f1[r] = math.copysign(
                math.sqrt(abs(gf1_s[r]) / 2.0), gf1_s[r],
            )
        if gf2_s[r] != 0.0:
            gamma_f2[r] = math.copysign(
                math.sqrt(abs(gf2_s[r]) / 2.0), gf2_s[r],
            )

    sct, cap, fis, pot, nzero = _reconstruct_kernel(
        e, group_r_a_arr, group_r_ap_arr,
        float(data.abn), float(data.ki),
        group_l, group_g, group_nfis,
        group_res_start, group_res_end,
        er_s, 0.5 * gg_s,
        gamma_n * gamma_n, gamma_n * gamma_f1, gamma_f1 * gamma_f1,
        gamma_n * gamma_f2, gamma_f1 * gamma_f2, gamma_f2 * gamma_f2,
    )
    # A zero R-matrix denominator (E == E_r with a zero eliminated
    # width) makes R infinite; return NaN there, as the numpy / jax
    # backends do, instead of fastmath's arbitrary value.
    bad = nzero > 0.0
    if bad.any():
        sct[bad] = np.nan
        cap[bad] = np.nan
        fis[bad] = np.nan
    tot = sct + cap + fis
    return {'sct': sct, 'cap': cap, 'fis': fis, 'pot': pot, 'tot': tot}


# Pure-Python scalar helper used only in the wrapper (not in the
# hot kernel); avoids depending on the `@njit` version at wrapper
# import time when numba isn't installed.
def _pnt_shf_scalar(rho, L):
    r2 = rho * rho
    if L == 0:
        return rho, 0.0
    if L == 1:
        d = 1.0 + r2
        return rho * r2 / d, -1.0 / d
    if L == 2:
        d = 9.0 + r2 * (3.0 + r2)
        return rho * r2 * r2 / d, -(18.0 + 3.0 * r2) / d
    if L == 3:
        d = 225.0 + r2 * (45.0 + r2 * (6.0 + r2))
        return (rho * r2 ** 3 / d,
                -(675.0 + r2 * (90.0 + 6.0 * r2)) / d)
    if L == 4:
        d = 11025.0 + r2 * (1575.0 + r2 * (135.0 + r2 * (10.0 + r2)))
        return (rho * r2 ** 4 / d,
                -(44100.0 + r2 * (4725.0 + r2 * (270.0 + 10.0 * r2))) / d)
    if L == 5:
        d = 893025.0 + r2 * (
            99225.0 + r2 * (6300.0 + r2 * (315.0 + r2 * (15.0 + r2)))
        )
        return (rho * r2 ** 5 / d,
                -(4465125.0 + r2 * (
                    396900.0 + r2 * (18900.0 + r2 * (630.0 + 15.0 * r2))
                )) / d)
    # L >= 6: Newton recurrence from L=5 upward. Matches
    # :func:`mf2_interpretation_factors.newton_step_pnt_shf` step
    # by step, so this stays numerically equivalent to the numpy
    # / jax path at arbitrary L.
    r2 = rho * rho
    # Seed at L=5 from the closed form we just fell through.
    d5 = 893025.0 + r2 * (
        99225.0 + r2 * (6300.0 + r2 * (315.0 + r2 * (15.0 + r2)))
    )
    p_prev = rho * r2 ** 5 / d5
    s_prev = -(4465125.0 + r2 * (
        396900.0 + r2 * (18900.0 + r2 * (630.0 + 15.0 * r2))
    )) / d5
    for LL in range(6, L + 1):
        p_prev, s_prev = _newton_step_pnt_shf_py(p_prev, s_prev, r2, LL)
    return p_prev, s_prev
