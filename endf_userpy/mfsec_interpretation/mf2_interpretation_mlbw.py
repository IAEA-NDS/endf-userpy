"""Multi-Level Breit-Wigner (MLBW) reconstruction for MF2/MT151.

Port of the JAX MLBW kernel (``physics.py:mlbw`` on branch
``feature_resonance``) to a pure-numpy natural-size implementation
that runs unchanged on the JAX backend via the
:mod:`endf_userpy.primitives.array_ns` adapter. Deliberately
avoids the JAX-only design pressures:

- No pre-allocated padding (``NRS_SIZE`` / ``MAX_CHN`` / ``NP_TAB1``).
  Resonance and channel arrays are their natural sizes.
- No integer-key channel encoding (``CHN_STRIDE * L + CHN_OFFSET
  + 2 * J``). Channels are indexed 0..nch-1 directly; the
  resonance-to-channel map is a plain integer array
  ``res_channel[nres]``.
- No ``lax.switch`` for per-element branching. Vectorised
  ``xp.select`` handles interpolation-law dispatch;
  :func:`factors.pnt_shf` handles the L-dependence via a
  Newton recurrence.

Formalism (ENDF-6 D.1.3.4, MLBW):

    sigma_sct(E) = pi / k^2 * sum_c g_c ((1 - cos 2phi_c - A_c)^2
                                         + (sin 2phi_c + B_c)^2)
    sigma_cap(E) = 2 pi / k^2 * sum_c g_c C_c
    sigma_fis(E) = 2 pi / k^2 * sum_c g_c F_c
    sigma_pot(E) = 2 pi / k^2 * sum_c g_c (1 - cos 2phi_c)
    sigma_rxx(E) = 2 pi / k^2 * sum_c g_c X_c    (competitive)

Per-resonance ratio ``r_r = 2 Gamma_n / ((E - E_r')^2 + Gamma^2 / 4)``
with shifted resonance energy ``E_r' = E_r + (S_r - S_e) Gamma_n^0 /
2``. Channel contributions ``A_c / B_c / C_c / F_c / X_c`` accumulate
resonance ratios weighted by widths, summed via segment-sum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..primitives import tab1
from . import mf2_interpretation_factors as factors


@dataclass
class MLBWData:
    """Natural-size MLBW input for one isotope / one energy range.

    Fields:
        abn:  isotopic abundance in the material.
        spi:  target spin I (used for statistical weight and lx).
        ki:   wavenumber coefficient: k(E) = ki * sqrt(E).
        qx:   Q-value for the competitive width (0 if absent).
        r_a:  channel radius as TAB1 vs E (or single-panel constant).
        r_ap: scattering radius as TAB1 vs E.
        ch_l:      (nch,) L per channel.
        ch_g:      (nch,) statistical weight g(J) per channel.
        res_channel: (nres,) channel index per resonance.
        res_l:  (nres,) L per resonance (== ch_l[res_channel]; kept
                explicit for clarity and to avoid one indirection
                inside vectorised loops).
        res_er, res_gn, res_gg, res_gf, res_gx: (nres,)
                resonance energy, neutron / gamma / fission /
                competitive widths.
    """
    abn: float
    spi: float
    ki: float
    qx: float
    r_a: tab1.TAB1
    r_ap: tab1.TAB1
    ch_l: np.ndarray
    ch_g: np.ndarray
    res_channel: np.ndarray
    res_l: np.ndarray
    res_er: np.ndarray
    res_gn: np.ndarray
    res_gg: np.ndarray
    res_gf: np.ndarray
    res_gx: np.ndarray


# Machine-eps guards for the "resonance width is zero" and
# "penetration factor is zero" edge cases. Both branches would
# blow up as 0/0; we clamp to zero via `where` masks below.
_EPS = 1e-38


def _lx_values(l_vec, spi, xp):
    """Angular-momentum values for the competitive width.

    Approximation used by all major reconstruction codes (SAMMY,
    NJOY, PREPRO): for spin-zero targets the competitive channel
    is assumed to be l = |L - 2|; otherwise the same L.
    """
    return l_vec + (xp.abs(l_vec - 2) - l_vec) * (spi == 0)


def _rho(e, ki, r_tab, xp):
    """Channel radius parameter rho = k(|E|) * radius(|E|)."""
    ee = xp.abs(e)
    return ki * xp.sqrt(ee) * tab1.interp(r_tab, ee, xp)


def _rho_competitive(e, qx, ki, r_a_tab, xp):
    """Rho for the competitive channel: uses E + Q shifted argument."""
    ee = xp.maximum(e + qx, 0.0)
    return ki * xp.sqrt(ee) * tab1.interp(r_a_tab, ee, xp)


def reconstruct(data: MLBWData, energies_in, xp, nl_max: int = 8):
    """MLBW cross sections at ``energies_in`` (any 1D array).

    Returns a dict with keys sct, cap, fis, pot, rxx, tot, each of
    shape ``(len(energies_in),)``, in barn (with the ki convention).

    Parameters
    ----------
    data : MLBWData
        Preprocessed MLBW input for the range (one isotope, one
        energy range). See :class:`MLBWData`.
    energies_in : array_like
        Incident-neutron energies. Cast to the backend's float64
        via ``xp.asarray``.
    xp : backend
        As returned by :func:`array_ns.get_backend`.
    nl_max : int, optional
        Cap on the L-value Newton recurrence for penetration and
        shift factors. Default 8 matches the JAX prototype. Files
        with higher L in their resonance table (rare) need this
        raised.
    """
    e = xp.asarray(energies_in, dtype=xp.float64)
    ne = e.shape[0]

    er = xp.asarray(data.res_er, dtype=xp.float64)
    gn = xp.asarray(data.res_gn, dtype=xp.float64)
    gg = xp.asarray(data.res_gg, dtype=xp.float64)
    gf = xp.asarray(data.res_gf, dtype=xp.float64)
    gx = xp.asarray(data.res_gx, dtype=xp.float64)
    lr = xp.asarray(data.res_l, dtype=xp.int32)
    ich = xp.asarray(data.res_channel, dtype=xp.int32)
    ch_l = xp.asarray(data.ch_l, dtype=xp.int32)
    ch_g = xp.asarray(data.ch_g, dtype=xp.float64)
    nch = ch_l.shape[0]

    # --- Per-resonance quantities that depend only on E_r. ---
    #
    # Penetration and shift factors P_L(rho(E_r)) and S_L(rho(E_r));
    # the reduced widths gamma^0_n / gamma^0_x are (Gamma / P), used
    # below to build the energy-dependent widths P(E) * gamma^0.

    rho_r = _rho(er, data.ki, data.r_a, xp)                # (nres,)
    pnt_r, shf_r = factors.pnt_shf(rho_r, lr, xp, nl_max)  # (nres,)

    lx = _lx_values(lr, data.spi, xp)
    rho_xr = _rho_competitive(er, data.qx, data.ki, data.r_a, xp)
    pntx_r, _ = factors.pnt_shf(rho_xr, lx, xp, nl_max)     # (nres,)

    gn0 = xp.where(pnt_r > _EPS, gn / pnt_r, 0.0)
    gx0 = xp.where(pntx_r > _EPS, gx / pntx_r, 0.0)

    # --- Per-energy factors (broadcast to (ne, nres) below). ---

    e_safe = xp.maximum(e, 0.0)
    e_pos = e > 0.0                                          # (ne,)
    k_e = data.ki * xp.sqrt(xp.where(e_pos, e_safe, 1.0))    # (ne,)
    inv_k2 = xp.where(e_pos, xp.pi / (k_e * k_e), 0.0)       # (ne,)
    pi_k2 = data.abn * inv_k2                                # (ne,)
    twopi_k2 = 2.0 * pi_k2                                   # (ne,)

    # Penetration / shift at the query energy, per resonance's L:
    # rho_e is a per-energy scalar; L varies per resonance. Broadcast
    # rho_e to (ne, 1), lr to (1, nres); output (ne, nres).
    rho_e = _rho(e_safe, data.ki, data.r_a, xp)              # (ne,)
    pnt_e, shf_e = factors.pnt_shf(
        rho_e.reshape(-1, 1), lr.reshape(1, -1), xp, nl_max,
    )   # (ne, nres)
    rho_xe = _rho_competitive(e_safe, data.qx, data.ki, data.r_a, xp)   # (ne,)
    pntx_e, _ = factors.pnt_shf(
        rho_xe.reshape(-1, 1), lx.reshape(1, -1), xp, nl_max,
    )   # (ne, nres)

    # Widths at the query energy.
    gn_e = pnt_e * gn0.reshape(1, -1)                       # (ne, nres)
    gx_e = pntx_e * gx0.reshape(1, -1)                      # (ne, nres)
    gg_e = gg.reshape(1, -1)                                 # (ne, nres)
    gf_e = gf.reshape(1, -1)                                 # (ne, nres)
    gt = gn_e + gg_e + gf_e + gx_e

    # Shifted resonance energy.
    erp = er.reshape(1, -1) + 0.5 * (
        shf_r.reshape(1, -1) - shf_e
    ) * gn0.reshape(1, -1)

    de = 2.0 * (e_safe.reshape(-1, 1) - erp)
    denom = gt ** 2 + de ** 2
    ratio = xp.where(denom > _EPS, 2.0 * gn_e / denom, 0.0)

    A_r = ratio * gt
    B_r = ratio * de
    cap_r = ratio * gg_e
    fis_r = ratio * gf_e
    rxx_r = ratio * gx_e

    # --- Sum per-resonance contributions into per-channel arrays. ---

    def _sum_by_channel_per_energy(sig_ne_nres):
        """(ne, nres) -> (ne, nch); per-row segment_sum."""
        out = xp.zeros((ne, nch), dtype=sig_ne_nres.dtype)
        for i in range(ne):
            out = _write_row(out, i, xp.segment_sum(sig_ne_nres[i], ich, nch), xp)
        return out

    A_ch = _sum_by_channel_per_energy(A_r)
    B_ch = _sum_by_channel_per_energy(B_r)
    cap_ch = _sum_by_channel_per_energy(cap_r)
    fis_ch = _sum_by_channel_per_energy(fis_r)
    rxx_ch = _sum_by_channel_per_energy(rxx_r)

    # --- Hard-sphere potential-scattering phase per channel. ---
    rho_s = _rho(e_safe, data.ki, data.r_ap, xp)            # (ne,)
    phi2 = 2.0 * factors.phase(
        rho_s.reshape(-1, 1), ch_l.reshape(1, -1), xp, nl_max,
    )   # (ne, nch)
    pot_ch = 1.0 - xp.cos(phi2)                             # (ne, nch)

    # --- Assemble by channel and reduce over channels. ---
    sct_ch = (pot_ch - A_ch) ** 2 + (xp.sin(phi2) + B_ch) ** 2   # (ne, nch)
    gmask = ch_g.reshape(1, -1)                             # (1, nch)

    sct = xp.sum(sct_ch * gmask, axis=1) * pi_k2
    cap = xp.sum(cap_ch * gmask, axis=1) * twopi_k2
    fis = xp.sum(fis_ch * gmask, axis=1) * twopi_k2
    pot = xp.sum(pot_ch * gmask, axis=1) * twopi_k2
    rxx = xp.sum(rxx_ch * gmask, axis=1) * twopi_k2
    tot = sct + cap + fis + rxx

    return {'sct': sct, 'cap': cap, 'fis': fis, 'pot': pot,
            'rxx': rxx, 'tot': tot}


def _write_row(mat, i, row, xp):
    """`mat[i] = row` in a backend-friendly way.

    NumPy: plain in-place assignment. JAX: functional update via
    `mat.at[i].set(row)`. Kept out of the main formula for clarity.
    """
    if xp.name == 'jax':
        return mat.at[i].set(row)
    mat = mat.copy()
    mat[i] = row
    return mat
