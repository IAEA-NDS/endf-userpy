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
- L in 0..5 (closed forms for P_L, S_L, φ_L); higher L is
  rejected at the wrapper with a clear message pointing at the
  numpy / jax backends.
- Same physics as the numpy path: LSSF=0 handled via the
  ``E_r - γ_n^2 (S_L(E) - S_L(|E_r|))`` level shift on the R-matrix
  denominator (matches SAMMY convention). ``shf_r`` is
  precomputed per resonance in the wrapper and passed in.
- Hand-coded 1×1 / 2×2 / 3×3 solves for ``(I - i R P) X = R``.
  Faster than calling numba's ``np.linalg.solve`` per (E, group)
  under ``parallel=True`` (which serialises through LAPACK) and
  keeps the whole kernel in numba-native arithmetic.

The scalar factor helpers ``_low_L_pnt_shf`` and ``_low_L_phase``
are numerically identical to the ones in
:mod:`mf2_interpretation_mlbw_numba`. They are duplicated here
rather than shared to keep the two numba modules independent
until a third formalism (SLBW, RML) shows up and motivates a
dedicated ``mf2_interpretation_factors_numba`` module.
"""
from __future__ import annotations

import math

import numpy as np

from ..primitives import array_ns
from ..primitives import tab1 as tab1_mod

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
def _low_L_pnt_shf(rho, L):
    """Scalar closed-form (P_L, S_L) for L in 0..5.

    Duplicated from :func:`mf2_interpretation_mlbw_numba._low_L_pnt_shf`;
    keep the two in sync.
    """
    r2 = rho * rho
    if L == 0:
        return rho, 0.0
    elif L == 1:
        d = 1.0 + r2
        return rho * r2 / d, -1.0 / d
    elif L == 2:
        d = 9.0 + r2 * (3.0 + r2)
        return rho * r2 * r2 / d, -(18.0 + 3.0 * r2) / d
    elif L == 3:
        d = 225.0 + r2 * (45.0 + r2 * (6.0 + r2))
        return (rho * r2 * r2 * r2 / d,
                -(675.0 + r2 * (90.0 + 6.0 * r2)) / d)
    elif L == 4:
        d = 11025.0 + r2 * (1575.0 + r2 * (135.0 + r2 * (10.0 + r2)))
        return (rho * r2 ** 4 / d,
                -(44100.0 + r2 * (4725.0 + r2 * (270.0 + 10.0 * r2))) / d)
    elif L == 5:
        d = 893025.0 + r2 * (
            99225.0 + r2 * (6300.0 + r2 * (315.0 + r2 * (15.0 + r2)))
        )
        p = rho * r2 ** 5 / d
        s = -(4465125.0 + r2 * (
            396900.0 + r2 * (18900.0 + r2 * (630.0 + 15.0 * r2))
        )) / d
        return p, s
    else:
        return float('nan'), float('nan')


@njit(cache=True, inline='always')
def _low_L_phase(rho, L):
    """Scalar hard-sphere phase φ_L for L in 0..5.

    Duplicated from :func:`mf2_interpretation_mlbw_numba._low_L_phase`.
    """
    r2 = rho * rho
    if L == 0:
        return rho
    elif L == 1:
        return rho - math.atan(rho)
    elif L == 2:
        return rho - math.atan2(3.0 * rho, 3.0 - r2)
    elif L == 3:
        return rho - math.atan2(rho * (15.0 - r2), 15.0 - 6.0 * r2)
    elif L == 4:
        return rho - math.atan2(
            rho * (105.0 - 10.0 * r2), 105.0 - r2 * (45.0 - r2),
        )
    elif L == 5:
        return rho - math.atan2(
            rho * (945.0 - r2 * (105.0 - r2)),
            945.0 - r2 * (420.0 - 15.0 * r2),
        )
    else:
        return float('nan')


@njit(cache=True, parallel=True, fastmath=True)
def _reconstruct_kernel(
    e,                     # (ne,) float64
    r_a_e,                 # (ne,) channel radius interpolated at E
    r_ap_e,                # (ne,) scattering radius interpolated at E
    abn, ki,
    group_l,               # (ngroups,) int
    group_g,               # (ngroups,) float
    group_nfis,            # (ngroups,) int (0, 1, or 2)
    group_res_start,       # (ngroups,) int  -- contiguous slice into res_*
    group_res_end,         # (ngroups,) int
    res_er,                # (nres,) float
    res_gg,                # (nres,) float
    gamma_n,               # (nres,) float  -- signed reduced-width amps, precomputed
    gamma_f1,              # (nres,) float
    gamma_f2,              # (nres,) float
    shf_r,                 # (nres,) float  -- S_L(rho(|E_r|)) per resonance
):
    ne = e.shape[0]
    ngroups = group_l.shape[0]

    sct = np.zeros(ne)
    cap = np.zeros(ne)
    fis = np.zeros(ne)
    pot = np.zeros(ne)

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
        rho_a_i = ki * sqrt_E * r_a_e[i]
        rho_ap_i = ki * sqrt_E * r_ap_e[i]

        for g in range(ngroups):
            L = group_l[g]
            gJ = group_g[g]
            nfis = group_nfis[g]

            # Elastic-channel factors at E. R-matrix penetration uses
            # the channel radius (rho_a); the hard-sphere phase in Ω
            # uses the scattering radius (rho_ap). These differ only
            # for NAPS=2 evaluations but the physics is identical.
            # `shf_e` is used below to build the S(E) - S(|E_r|) level
            # shift correction on the R-matrix denominator (LSSF=0
            # handling, matches the numpy path).
            p_e, shf_e = _low_L_pnt_shf(rho_a_i, L)
            phi_e = _low_L_phase(rho_ap_i, L)

            # Potential scattering contribution: 4π/k² Σ_J g_J sin²(φ_L).
            sin_phi = math.sin(phi_e)
            pot_sum += gJ * sin_phi * sin_phi

            # --- Build R-matrix contributions from this group's resonances. ---
            # R is (nch × nch) complex, symmetric. We only carry the
            # 6 upper-triangle entries R_00, R_01, R_02, R_11, R_12, R_22
            # as scalar accumulators; unused entries for nch<3 stay 0.
            R00 = 0.0 + 0.0j
            R01 = 0.0 + 0.0j
            R02 = 0.0 + 0.0j
            R11 = 0.0 + 0.0j
            R12 = 0.0 + 0.0j
            R22 = 0.0 + 0.0j

            r_start = group_res_start[g]
            r_end = group_res_end[g]
            for r in range(r_start, r_end):
                # Level shift on the effective resonance energy:
                #   E_r^eff = E_r - γ_n^2 (S_L(E) - S_L(|E_r|))
                # Only the elastic channel contributes (S=0 for fission).
                # For s-wave the shift is identically 0.
                gn_r = gamma_n[r]
                delta_r = -(gn_r * gn_r) * (shf_e - shf_r[r])
                er_eff = res_er[r] + delta_r
                # 1 / (E_r^eff - E - i Γ_γ / 2)
                denom = complex(er_eff - E_safe, -0.5 * res_gg[r])
                inv_d = 1.0 / denom

                R00 = R00 + (gn_r * gn_r) * inv_d
                if nfis >= 1:
                    gf1_r = gamma_f1[r]
                    R01 = R01 + (gn_r * gf1_r) * inv_d
                    R11 = R11 + (gf1_r * gf1_r) * inv_d
                if nfis >= 2:
                    gf2_r = gamma_f2[r]
                    R02 = R02 + (gn_r * gf2_r) * inv_d
                    R12 = R12 + (gamma_f1[r] * gf2_r) * inv_d
                    R22 = R22 + (gf2_r * gf2_r) * inv_d

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
                sumsq = U00.real * U00.real + U00.imag * U00.imag
                abs_g = pi_k2 * gJ * (1.0 - sumsq)
                fis_g = 0.0
                cap_g = abs_g
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
                sumsq = U00.real * U00.real + U00.imag * U00.imag + s01
                abs_g = pi_k2 * gJ * (1.0 - sumsq)
                fis_g = pi_k2 * gJ * s01
                cap_g = abs_g   # `abs_` is already σ_cap in R-M (see numpy path)
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
                sumsq = U00.real * U00.real + U00.imag * U00.imag + s01 + s02
                abs_g = pi_k2 * gJ * (1.0 - sumsq)
                fis_g = pi_k2 * gJ * (s01 + s02)
                cap_g = abs_g   # `abs_` is already σ_cap in R-M (see numpy path)

            sct_sum += sct_g
            cap_sum += cap_g
            fis_sum += fis_g

        sct[i] = sct_sum
        cap[i] = cap_sum
        fis[i] = fis_sum
        pot[i] = 4.0 * pi_k2 * pot_sum

    return sct, cap, fis, pot


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

    max_L = int(group_l.max()) if group_l.size else 0
    if max_L > 5:
        raise NotImplementedError(
            f"RM numba kernel supports L<=5; got L={max_L}. "
            f"Use the numpy or jax backend for higher L."
        )

    # --- Pre-interpolate radii on the query grid. ---
    xp = array_ns.get_backend('numpy')
    e_safe = np.maximum(e, 0.0)
    r_a_e = np.asarray(tab1_mod.interp(data.r_a, e_safe, xp))
    r_ap_e = np.asarray(tab1_mod.interp(data.r_ap, e_safe, xp))
    r_a_at_er = np.asarray(tab1_mod.interp(data.r_a, np.abs(er), xp))

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
    nres = er_s.shape[0]
    gamma_n = np.zeros(nres, dtype=np.float64)
    gamma_f1 = np.zeros(nres, dtype=np.float64)
    gamma_f2 = np.zeros(nres, dtype=np.float64)
    shf_r = np.zeros(nres, dtype=np.float64)
    for r in range(nres):
        g_idx = int(res_group_s[r])
        L = int(group_l[g_idx])
        rho_at_er = data.ki * math.sqrt(abs(er_s[r])) * r_a_at_er_s[r]
        p_r, s_r = _pnt_shf_scalar(rho_at_er, L)
        shf_r[r] = s_r
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

    sct, cap, fis, pot = _reconstruct_kernel(
        e, r_a_e, r_ap_e,
        float(data.abn), float(data.ki),
        group_l, group_g, group_nfis,
        group_res_start, group_res_end,
        er_s, gg_s, gamma_n, gamma_f1, gamma_f2, shf_r,
    )
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
    raise NotImplementedError(f'RM numba wrapper: L<=5 only, got L={L}')
