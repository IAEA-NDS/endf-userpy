"""Numba-compiled MLBW kernel for :mod:`mf2_interpretation_mlbw`.

Backend specialisation: the numpy/JAX physics in
:mod:`mf2_interpretation_mlbw` runs through the array-namespace adapter
(:mod:`endf_userpy.primitives.array_ns`), which is elegant but cannot
be consumed by numba's ``@njit`` (which needs plain typed arrays and
scalar Python control flow). This module holds a hand-written
``@njit(parallel=True, fastmath=True)`` fused per-energy kernel that
implements the same physics.

Same inputs (:class:`~mf2_interpretation_mlbw.MLBWData`), same outputs
(``dict`` of partial cross sections). The public entry point is
:func:`reconstruct`; :func:`mf2_interpretation_mlbw.reconstruct`
dispatches here when called with a ``NumbaBackend`` adapter.

Benchmark on Nb-93 (NE=200 000, 202 resonances, 7 channels): ~50x
over the numpy adapter path and ~200x over the JAX prototype. That is
what numba is good at: fusing per-element math (sqrt, atan, polynomial
evaluation, division) into a tight, thread-parallel loop without
materialising the ``(ne, nres)`` intermediate matrices that the
vectorised numpy path pays for.

Supports arbitrary non-negative L. Closed forms for L in 0..5 stay
in the inline fast path; L >= 6 goes through
:func:`pnt_shf_any_L` / :func:`phase_any_L` in
:mod:`mf2_interpretation_factors_numba` which run the Newton
recurrence from L=5 up to the target L, matching
:mod:`mf2_interpretation_factors`'s numpy / JAX path.
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

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Fallbacks so the module still imports cleanly without numba;
    # callers that hit :func:`reconstruct` get a runtime error.
    def njit(*a, **kw):
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        return lambda f: f

    prange = range


_EPS = 1e-38


@njit(cache=True, parallel=True, fastmath=True)
def _reconstruct_kernel(
    e,             # (ne,) query energies, float64
    r_a_e,         # (ne,)  channel radius interpolated at e
    r_ap_e,        # (ne,)  scattering radius interpolated at e
    r_a_at_er,     # (nres,) r_a at |E_r|
    r_a_at_er_qx,  # (nres,) r_a at max(E_r + qx, 0)
    abn, spi, ki, qx,
    ch_l, ch_g,    # (nch,) int, (nch,) float
    ch_start,      # (nch,) int -- resonances sorted by channel; channel
    ch_end,        # (nch,) int    c owns the slice [ch_start[c], ch_end[c])
    res_l,         # (nres,) int
    res_er,        # (nres,)
    res_gn, res_gg, res_gf, res_gx,
):
    ne = e.shape[0]
    nch = ch_l.shape[0]
    nres = res_er.shape[0]

    # --- Per-resonance precompute (small loop over nres). ---
    # Nothing here parallelises usefully at nres ~ hundreds; keep it
    # serial in the main thread.
    pnt_r = np.zeros(nres)
    shf_r = np.zeros(nres)
    pntx_r = np.zeros(nres)
    for r in range(nres):
        er_abs = abs(res_er[r])
        rho_r = ki * math.sqrt(er_abs) * r_a_at_er[r]
        p, s = _pnt_shf(rho_r, res_l[r])
        pnt_r[r] = p
        shf_r[r] = s

        # Competitive channel: rho at max(E_r + qx, 0) with L = |L-2|
        # for spi==0 targets (SAMMY / NJOY convention).
        e_x = max(res_er[r] + qx, 0.0)
        rho_xr = ki * math.sqrt(e_x) * r_a_at_er_qx[r]
        L = res_l[r]
        lx = L if spi != 0.0 else abs(L - 2)
        px, _ = _pnt_shf(rho_xr, lx)
        pntx_r[r] = px

    gn0 = np.zeros(nres)
    gx0 = np.zeros(nres)
    for r in range(nres):
        if pnt_r[r] > _EPS:
            gn0[r] = res_gn[r] / pnt_r[r]
        if pntx_r[r] > _EPS:
            gx0[r] = res_gx[r] / pntx_r[r]

    # --- Outputs. ---
    sct = np.zeros(ne)
    cap = np.zeros(ne)
    fis = np.zeros(ne)
    pot = np.zeros(ne)
    rxx = np.zeros(ne)

    # --- Main parallel per-energy loop. Each thread owns one energy
    # index; per-channel accumulators are small locals that stay in
    # cache. This is where numba beats the vectorised numpy path:
    # no (ne, nres) intermediates get materialised.
    for i in prange(ne):
        E = e[i]
        E_safe = E if E > 0.0 else 0.0
        if E > 0.0:
            k_e = ki * math.sqrt(E_safe)
            inv_k2 = math.pi / (k_e * k_e)
        else:
            inv_k2 = 0.0
        pi_k2 = abn * inv_k2
        twopi_k2 = 2.0 * pi_k2

        r_a_i = r_a_e[i]
        r_ap_i = r_ap_e[i]
        rho_e = ki * math.sqrt(E_safe) * r_a_i
        e_x_scalar = E_safe + qx if (E + qx) > 0.0 else 0.0
        rho_xe = ki * math.sqrt(e_x_scalar) * r_a_i
        rho_s = ki * math.sqrt(E_safe) * r_ap_i

        sct_sum = 0.0
        pot_sum = 0.0
        cap_sum = 0.0
        fis_sum = 0.0
        rxx_sum = 0.0
        for c in range(nch):
            # Every resonance of channel c has L = ch_l[c], so the
            # energy-dependent factors are evaluated once per channel;
            # the resonance loop below is plain arithmetic over the
            # channel's contiguous slice, which LLVM vectorises.
            L = ch_l[c]
            p_e, s_e = _pnt_shf(rho_e, L)
            lx = L if spi != 0.0 else abs(L - 2)
            px_e, _ = _pnt_shf(rho_xe, lx)

            r0 = ch_start[c]
            r1 = ch_end[c]
            er_c = res_er[r0:r1]
            shf_c = shf_r[r0:r1]
            gn0_c = gn0[r0:r1]
            gx0_c = gx0[r0:r1]
            gg_c = res_gg[r0:r1]
            gf_c = res_gf[r0:r1]
            A_c = 0.0
            B_c = 0.0
            cap_c = 0.0
            fis_c = 0.0
            rxx_c = 0.0
            for r in range(er_c.shape[0]):
                gn_e = p_e * gn0_c[r]
                gx_e = px_e * gx0_c[r]
                gt = gn_e + gg_c[r] + gf_c[r] + gx_e
                erp = er_c[r] + 0.5 * (shf_c[r] - s_e) * gn0_c[r]
                de = 2.0 * (E_safe - erp)
                denom = gt * gt + de * de
                ratio = 2.0 * gn_e / denom if denom > _EPS else 0.0
                A_c += ratio * gt
                B_c += ratio * de
                cap_c += ratio * gg_c[r]
                fis_c += ratio * gf_c[r]
                rxx_c += ratio * gx_e

            phi2 = 2.0 * _phase(rho_s, L)
            pot_c = 1.0 - math.cos(phi2)
            sin_phi2 = math.sin(phi2)
            sct_c = (pot_c - A_c) ** 2 + (sin_phi2 + B_c) ** 2

            g = ch_g[c]
            sct_sum += sct_c * g
            pot_sum += pot_c * g
            cap_sum += cap_c * g
            fis_sum += fis_c * g
            rxx_sum += rxx_c * g

        sct[i] = sct_sum * pi_k2
        cap[i] = cap_sum * twopi_k2
        fis[i] = fis_sum * twopi_k2
        pot[i] = pot_sum * twopi_k2
        rxx[i] = rxx_sum * twopi_k2

    return sct, cap, fis, pot, rxx


def reconstruct(data, energies_in):
    """MLBW cross sections via the numba-compiled kernel.

    Signature-compatible with
    :func:`mf2_interpretation_mlbw.reconstruct` minus the backend
    argument (implicit here). The wrapper's job is to pre-interpolate
    the radii on the query grid and at the resonance energies
    (numba can't call the TAB1 helper directly), validate the L
    range, and dispatch.

    First call incurs the numba compile cost (typically a few
    seconds); subsequent calls hit the cached compilation.
    """
    if not HAS_NUMBA:
        raise RuntimeError(
            "MLBW numba backend requested but `numba` is not installed. "
            "`pip install numba` or use `array_ns.get_backend('numpy')`."
        )

    e = np.asarray(energies_in, dtype=np.float64)
    er = np.asarray(data.res_er, dtype=np.float64)
    e_safe = np.maximum(e, 0.0)

    # Pre-interpolate radii on the query grid and at resonance energies.
    xp = array_ns.get_backend('numpy')
    r_a_e = np.asarray(tab1_mod.interp(data.r_a, e_safe, xp))
    r_ap_e = np.asarray(tab1_mod.interp(data.r_ap, e_safe, xp))
    r_a_at_er = np.asarray(tab1_mod.interp(data.r_a, np.abs(er), xp))
    e_x = np.maximum(er + data.qx, 0.0)
    r_a_at_er_qx = np.asarray(tab1_mod.interp(data.r_a, e_x, xp))

    # The kernel walks each channel's resonances as one contiguous
    # slice and evaluates P_L / S_L once per channel, which relies on
    # the MLBWData invariant res_l == ch_l[res_channel].
    ch_l = np.asarray(data.ch_l, dtype=np.int64)
    res_channel = np.asarray(data.res_channel, dtype=np.int64)
    res_l = np.asarray(data.res_l, dtype=np.int64)
    if not np.array_equal(res_l, ch_l[res_channel]):
        raise ValueError(
            'MLBWData.res_l must equal ch_l[res_channel] for every '
            'resonance (each channel has a single L).'
        )
    order = np.argsort(res_channel, kind='stable')
    channels = np.arange(ch_l.shape[0])
    ch_start = np.searchsorted(res_channel[order], channels, side='left')
    ch_end = np.searchsorted(res_channel[order], channels, side='right')

    def _sorted(a):
        return np.asarray(a, dtype=np.float64)[order]

    sct, cap, fis, pot, rxx = _reconstruct_kernel(
        e, r_a_e, r_ap_e, r_a_at_er[order], r_a_at_er_qx[order],
        float(data.abn), float(data.spi), float(data.ki), float(data.qx),
        ch_l,
        np.asarray(data.ch_g, dtype=np.float64),
        ch_start.astype(np.int64), ch_end.astype(np.int64),
        res_l[order],
        er[order],
        _sorted(data.res_gn), _sorted(data.res_gg),
        _sorted(data.res_gf), _sorted(data.res_gx),
    )
    tot = sct + cap + fis + rxx
    return {'sct': sct, 'cap': cap, 'fis': fis, 'pot': pot,
            'rxx': rxx, 'tot': tot}
