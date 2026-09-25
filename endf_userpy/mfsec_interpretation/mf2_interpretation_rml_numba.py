"""Numba-compiled KRM=3 (Reich-Moore) kernel for
:mod:`mf2_interpretation_rml`.

Third module of the LRF=7 R-Matrix Limited arc (#228 PR 3). Same
three-backend pattern as MLBW / R-M: the numpy and JAX code path
in :mod:`mf2_interpretation_rml` shares one implementation through
:mod:`endf_userpy.primitives.array_ns`; this module supplies a
hand-fused ``@njit`` kernel for the ``'numba'`` backend.

Scope (matches :mod:`mf2_interpretation_rml`):

- KRM=3 Reich-Moore approximation. Gamma channels eliminated;
  their widths sum into ``Gamma_gamma_r`` in the R-matrix denominator.
- Up to 3 particle channels per J-group (all real LRF=7 corpus
  files fit — Pu-239's largest group is 3 particle channels).
  Wrapper raises for groups with more than 3 particle channels.
- Per-channel L, APE, APT. Multi-L within a group is fine
  (Rh-103 group 3, Cu-63 groups 3/4 exercise this).
- Elastic slot is reordered to index 0 within particle channels by
  the wrapper; the kernel assumes slot 0 is the incident channel.

Hand-coded 1x1 / 2x2 / 3x3 solves for ``(I - i R P) X = R`` avoid
the numba LAPACK serialisation cost inside the ``parallel=True``
per-energy loop, matching the pattern in
:mod:`mf2_interpretation_reichmoore_numba`.
"""
from __future__ import annotations

import math

import numpy as np

from .mf2_interpretation_factors_numba import (
    pnt_shf_any_L as _pnt_shf,
    phase_any_L as _phase,
)
from .mf2_interpretation_reichmoore_numba import (
    _pnt_shf_scalar,
)

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
_MAX_NPART_KERNEL = 3


@njit(cache=True, parallel=True, fastmath=True)
def _reconstruct_kernel(
    e,                        # (ne,) float64 -- incident energies
    abn, ki,
    group_npart,              # (ngroups,) int64 -- particle channels per group (1..3)
    group_gJ,                 # (ngroups,) float64
    slot_L,                   # (ngroups, 3) int64 -- L per particle slot (padded)
    slot_ape,                 # (ngroups, 3) float64 -- APE per slot
    slot_apt,                 # (ngroups, 3) float64 -- APT per slot
    slot_has_pnt,             # (ngroups, 3) int64 -- 1 if compute P_L, else 0 (P=1)
    slot_has_phase,           # (ngroups, 3) int64 -- 1 if compute phi_L, else 0
    slot_is_fis,              # (ngroups, 3) int64 -- 1 if fission slot for sum
    group_res_start,          # (ngroups,) int64
    group_res_end,            # (ngroups,) int64
    res_er,                   # (nres,) float64
    res_gg_eff,               # (nres,) float64 -- summed eliminated widths per resonance
    res_gamma,                # (nres, 3) float64 -- signed reduced-width amps per particle slot
):
    ne = e.shape[0]
    ngroups = group_npart.shape[0]

    sct = np.zeros(ne)
    cap = np.zeros(ne)
    fis = np.zeros(ne)
    pot = np.zeros(ne)

    for i in prange(ne):
        E = e[i]
        if E > 0.0:
            k2 = (ki * ki) * E
            pi_k2 = abn * math.pi / k2
        else:
            pi_k2 = 0.0

        sct_sum = 0.0
        cap_sum = 0.0
        fis_sum = 0.0
        pot_sum = 0.0

        sqrt_E = math.sqrt(E) if E > 0.0 else 0.0

        for g in range(ngroups):
            npart = group_npart[g]
            gJ = group_gJ[g]

            # Per-slot penetration and phase at E.
            # Slot 0 = elastic (incident). Higher slots may be fission or
            # additional elastic-like channels.
            p0 = 1.0
            p1 = 1.0
            p2 = 1.0
            phi0 = 0.0
            phi1 = 0.0
            phi2 = 0.0

            # Slot 0
            L0 = slot_L[g, 0]
            rho_apt0 = ki * sqrt_E * slot_apt[g, 0]
            rho_ape0 = ki * sqrt_E * slot_ape[g, 0]
            if slot_has_pnt[g, 0] == 1:
                p0, _ = _pnt_shf(rho_apt0, L0)
            if slot_has_phase[g, 0] == 1:
                phi0 = _phase(rho_ape0, L0)

            if npart >= 2:
                L1 = slot_L[g, 1]
                rho_apt1 = ki * sqrt_E * slot_apt[g, 1]
                rho_ape1 = ki * sqrt_E * slot_ape[g, 1]
                if slot_has_pnt[g, 1] == 1:
                    p1, _ = _pnt_shf(rho_apt1, L1)
                if slot_has_phase[g, 1] == 1:
                    phi1 = _phase(rho_ape1, L1)

            if npart >= 3:
                L2 = slot_L[g, 2]
                rho_apt2 = ki * sqrt_E * slot_apt[g, 2]
                rho_ape2 = ki * sqrt_E * slot_ape[g, 2]
                if slot_has_pnt[g, 2] == 1:
                    p2, _ = _pnt_shf(rho_apt2, L2)
                if slot_has_phase[g, 2] == 1:
                    phi2 = _phase(rho_ape2, L2)

            # Potential-scattering contribution from the elastic slot only.
            sin_phi0 = math.sin(phi0)
            pot_sum += gJ * sin_phi0 * sin_phi0

            # --- R-matrix accumulators (upper triangle, complex). ---
            R00 = 0.0 + 0.0j
            R01 = 0.0 + 0.0j
            R02 = 0.0 + 0.0j
            R11 = 0.0 + 0.0j
            R12 = 0.0 + 0.0j
            R22 = 0.0 + 0.0j

            r0 = group_res_start[g]
            r1 = group_res_end[g]
            for r in range(r0, r1):
                gam0 = res_gamma[r, 0]
                denom = complex(res_er[r] - E, -0.5 * res_gg_eff[r])
                inv_d = 1.0 / denom
                R00 = R00 + (gam0 * gam0) * inv_d
                if npart >= 2:
                    gam1 = res_gamma[r, 1]
                    R01 = R01 + (gam0 * gam1) * inv_d
                    R11 = R11 + (gam1 * gam1) * inv_d
                if npart >= 3:
                    gam2 = res_gamma[r, 2]
                    R02 = R02 + (gam0 * gam2) * inv_d
                    R12 = R12 + (gam1 * gam2) * inv_d
                    R22 = R22 + (gam2 * gam2) * inv_d

            # --- Solve (I - i R P) X = R for row 0 of X. ---
            omega0 = complex(math.cos(-phi0), math.sin(-phi0))
            omega1 = complex(math.cos(-phi1), math.sin(-phi1)) if npart >= 2 else complex(1.0, 0.0)
            omega2 = complex(math.cos(-phi2), math.sin(-phi2)) if npart >= 3 else complex(1.0, 0.0)
            sqrt_p0 = math.sqrt(p0) if p0 > 0.0 else 0.0
            sqrt_p1 = math.sqrt(p1) if npart >= 2 and p1 > 0.0 else (1.0 if npart >= 2 else 0.0)
            sqrt_p2 = math.sqrt(p2) if npart >= 3 and p2 > 0.0 else (1.0 if npart >= 3 else 0.0)

            if npart == 1:
                W00 = 1.0 - 1j * R00 * p0
                X00 = R00 / W00
                U00 = omega0 * omega0 * (1.0 + 2j * p0 * X00)
                sumsq0 = U00.real * U00.real + U00.imag * U00.imag
                sumsq = sumsq0
                sct_g = pi_k2 * gJ * ((1.0 - U00.real) ** 2 + U00.imag ** 2)
                fis_g = 0.0
                cap_g = pi_k2 * gJ * (1.0 - sumsq)
            elif npart == 2:
                W00 = 1.0 - 1j * R00 * p0
                W01 = -1j * R01 * p1
                W10 = -1j * R01 * p0
                W11 = 1.0 - 1j * R11 * p1
                det = W00 * W11 - W01 * W10
                X00 = (W11 * R00 - W01 * R01) / det
                X01 = (W11 * R01 - W01 * R11) / det
                U00 = omega0 * omega0 * (1.0 + 2j * sqrt_p0 * sqrt_p0 * X00)
                U01 = omega0 * omega1 * (2j * sqrt_p0 * sqrt_p1 * X01)
                sumsq0 = U00.real * U00.real + U00.imag * U00.imag
                sumsq1 = U01.real * U01.real + U01.imag * U01.imag
                sumsq = sumsq0 + sumsq1
                sct_g = pi_k2 * gJ * ((1.0 - U00.real) ** 2 + U00.imag ** 2)
                fis_slot1 = sumsq1 if slot_is_fis[g, 1] == 1 else 0.0
                fis_g = pi_k2 * gJ * fis_slot1
                cap_g = pi_k2 * gJ * (1.0 - sumsq)
            else:   # npart == 3
                W00 = 1.0 - 1j * R00 * p0
                W01 = -1j * R01 * p1
                W02 = -1j * R02 * p2
                W10 = -1j * R01 * p0
                W11 = 1.0 - 1j * R11 * p1
                W12 = -1j * R12 * p2
                W20 = -1j * R02 * p0
                W21 = -1j * R12 * p1
                W22 = 1.0 - 1j * R22 * p2
                m00 = W11 * W22 - W12 * W21
                m01 = W10 * W22 - W12 * W20
                m02 = W10 * W21 - W11 * W20
                det = W00 * m00 - W01 * m01 + W02 * m02
                Winv00 = m00 / det
                Winv01 = -(W01 * W22 - W02 * W21) / det
                Winv02 = (W01 * W12 - W02 * W11) / det
                X00 = Winv00 * R00 + Winv01 * R01 + Winv02 * R02
                X01 = Winv00 * R01 + Winv01 * R11 + Winv02 * R12
                X02 = Winv00 * R02 + Winv01 * R12 + Winv02 * R22
                U00 = omega0 * omega0 * (1.0 + 2j * sqrt_p0 * sqrt_p0 * X00)
                U01 = omega0 * omega1 * (2j * sqrt_p0 * sqrt_p1 * X01)
                U02 = omega0 * omega2 * (2j * sqrt_p0 * sqrt_p2 * X02)
                sumsq0 = U00.real * U00.real + U00.imag * U00.imag
                sumsq1 = U01.real * U01.real + U01.imag * U01.imag
                sumsq2 = U02.real * U02.real + U02.imag * U02.imag
                sumsq = sumsq0 + sumsq1 + sumsq2
                sct_g = pi_k2 * gJ * ((1.0 - U00.real) ** 2 + U00.imag ** 2)
                fis_slot1 = sumsq1 if slot_is_fis[g, 1] == 1 else 0.0
                fis_slot2 = sumsq2 if slot_is_fis[g, 2] == 1 else 0.0
                fis_g = pi_k2 * gJ * (fis_slot1 + fis_slot2)
                cap_g = pi_k2 * gJ * (1.0 - sumsq)

            sct_sum += sct_g
            cap_sum += cap_g
            fis_sum += fis_g

        sct[i] = sct_sum
        cap[i] = cap_sum
        fis[i] = fis_sum
        pot[i] = 4.0 * pi_k2 * pot_sum

    return sct, cap, fis, pot


def _classify_slots(data, g):
    """Return per-slot metadata for one J-group, with the elastic
    slot moved to index 0. Mirrors ``_classify_group_channels`` in
    :mod:`mf2_interpretation_rml`, but returns the numeric arrays
    the numba kernel consumes.
    """
    from . import mf2_interpretation_rml as _rml
    kinds, particle, gamma_slots, elastic_local = (
        _rml._classify_group_channels(data, g)
    )
    npart = len(particle)
    if npart > _MAX_NPART_KERNEL:
        # Wide-group case is rejected here so the caller's error
        # includes the true npart in the message. The numba kernel
        # arrays are sized to _MAX_NPART_KERNEL and can't take more.
        raise NotImplementedError(
            f'LRF=7 numba kernel supports at most '
            f'{_MAX_NPART_KERNEL} particle channels per group; '
            f'group {g} has {npart}. Use the numpy backend for '
            f'files that exceed the kernel limit.'
        )
    # Reorder so elastic is at local index 0.
    reord = [particle[elastic_local]] + [
        p for i, p in enumerate(particle) if i != elastic_local
    ]
    slot_L = np.zeros(_MAX_NPART_KERNEL, dtype=np.int64)
    slot_ape = np.zeros(_MAX_NPART_KERNEL, dtype=np.float64)
    slot_apt = np.zeros(_MAX_NPART_KERNEL, dtype=np.float64)
    slot_has_pnt = np.zeros(_MAX_NPART_KERNEL, dtype=np.int64)
    slot_has_phase = np.zeros(_MAX_NPART_KERNEL, dtype=np.int64)
    slot_is_fis = np.zeros(_MAX_NPART_KERNEL, dtype=np.int64)
    reord_orig_ch = []
    for i, ch in enumerate(reord):
        ppi = int(round(float(data.ch_ppi[g, ch])))
        slot_L[i] = int(round(float(data.ch_l[g, ch])))
        slot_ape[i] = float(data.ch_ape[g, ch])
        slot_apt[i] = float(data.ch_apt[g, ch])
        slot_has_pnt[i] = 1 if _rml._pair_uses_penetration(data, ppi) else 0
        slot_has_phase[i] = 1 if kinds[ch] == 'elastic' else 0
        slot_is_fis[i] = 1 if kinds[ch] == 'fission' else 0
        reord_orig_ch.append(ch)
    return npart, slot_L, slot_ape, slot_apt, slot_has_pnt, slot_has_phase, slot_is_fis, reord_orig_ch, gamma_slots


def reconstruct(data, energies_in):
    """KRM=3 reconstruction via the numba-compiled kernel.

    Signature-compatible with
    :func:`mf2_interpretation_rml.reconstruct` minus the backend
    argument (implicit here).
    """
    if not HAS_NUMBA:
        raise RuntimeError(
            'MF2 LRF=7 KRM=3 numba backend requested but `numba` is '
            "not installed. `pip install numba` or use "
            "`array_ns.get_backend('numpy')`."
        )
    if data.krm != 3:
        raise NotImplementedError(
            f'LRF=7 KRM={data.krm} not supported; only KRM=3 in this arc.'
        )

    e = np.asarray(energies_in, dtype=np.float64)

    ngroups = data.n_groups()
    group_npart = np.zeros(ngroups, dtype=np.int64)
    group_gJ = np.asarray(data.group_g, dtype=np.float64)
    slot_L = np.zeros((ngroups, _MAX_NPART_KERNEL), dtype=np.int64)
    slot_ape = np.zeros((ngroups, _MAX_NPART_KERNEL), dtype=np.float64)
    slot_apt = np.zeros((ngroups, _MAX_NPART_KERNEL), dtype=np.float64)
    slot_has_pnt = np.zeros((ngroups, _MAX_NPART_KERNEL), dtype=np.int64)
    slot_has_phase = np.zeros((ngroups, _MAX_NPART_KERNEL), dtype=np.int64)
    slot_is_fis = np.zeros((ngroups, _MAX_NPART_KERNEL), dtype=np.int64)
    reord_per_group = []
    gamma_slots_per_group = []
    for g in range(ngroups):
        npart_g, sL, sAPE, sAPT, sPNT, sPHASE, sFIS, reord, gsl = _classify_slots(data, g)
        if npart_g > _MAX_NPART_KERNEL:
            raise NotImplementedError(
                f'LRF=7 numba kernel supports at most '
                f'{_MAX_NPART_KERNEL} particle channels per group; '
                f'group {g} has {npart_g}. Use the numpy backend for '
                f'files that exceed the kernel limit.'
            )
        group_npart[g] = npart_g
        slot_L[g] = sL
        slot_ape[g] = sAPE
        slot_apt[g] = sAPT
        slot_has_pnt[g] = sPNT
        slot_has_phase[g] = sPHASE
        slot_is_fis[g] = sFIS
        reord_per_group.append(reord)
        gamma_slots_per_group.append(gsl)

    # --- Sort resonances by group so each group's slice is contiguous. ---
    res_group_np = np.asarray(data.res_group, dtype=np.int64)
    order = np.argsort(res_group_np, kind='stable')
    er_s = np.asarray(data.res_er, dtype=np.float64)[order]
    gam_full_s = np.asarray(data.res_gam, dtype=np.float64)[order]
    res_group_s = res_group_np[order]

    nres = er_s.shape[0]
    group_res_start = np.zeros(ngroups, dtype=np.int64)
    group_res_end = np.zeros(ngroups, dtype=np.int64)
    if nres > 0:
        for g in range(ngroups):
            group_res_start[g] = int(np.searchsorted(res_group_s, g, side='left'))
            group_res_end[g] = int(np.searchsorted(res_group_s, g, side='right'))

    # --- Precompute per-resonance effective Gamma_gamma (sum of |GAM|
    # over eliminated / gamma slots) and per-particle-slot reduced-
    # width amplitudes gamma_{r,c}. ---
    res_gg_eff = np.zeros(nres, dtype=np.float64)
    res_gamma = np.zeros((nres, _MAX_NPART_KERNEL), dtype=np.float64)
    ki = float(data.ki)
    for r in range(nres):
        g_idx = int(res_group_s[r])
        # Eliminated (gamma) contribution.
        for c in gamma_slots_per_group[g_idx]:
            res_gg_eff[r] += abs(gam_full_s[r, c])
        # Particle-slot reduced-width amplitudes.
        for i, ch in enumerate(reord_per_group[g_idx]):
            gam_raw = float(gam_full_s[r, ch])
            if group_npart[g_idx] < 1:
                continue
            if slot_has_pnt[g_idx, i] == 1:
                L = int(slot_L[g_idx, i])
                rho_er = ki * math.sqrt(abs(float(er_s[r]))) * float(slot_apt[g_idx, i])
                p_r, _ = _pnt_shf_scalar(rho_er, L)
                if p_r > _EPS:
                    res_gamma[r, i] = math.copysign(
                        math.sqrt(abs(gam_raw) / (2.0 * p_r)), gam_raw,
                    )
            else:
                # P=1 for eliminated / fission / phase-shift-free slots.
                if gam_raw != 0.0:
                    res_gamma[r, i] = math.copysign(
                        math.sqrt(abs(gam_raw) / 2.0), gam_raw,
                    )

    sct, cap, fis, pot = _reconstruct_kernel(
        e, float(data.abn), ki,
        group_npart, group_gJ,
        slot_L, slot_ape, slot_apt,
        slot_has_pnt, slot_has_phase, slot_is_fis,
        group_res_start, group_res_end,
        er_s, res_gg_eff, res_gamma,
    )
    tot = sct + cap + fis
    return {'sct': sct, 'cap': cap, 'fis': fis, 'pot': pot, 'tot': tot}
