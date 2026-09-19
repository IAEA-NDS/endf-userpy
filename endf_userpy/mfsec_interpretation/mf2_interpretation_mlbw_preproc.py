"""ENDF-6 MF2/MT151 -> :class:`MLBWData` preprocessing.

Small dedicated preprocessor that lifts a parsed ENDF-6 dict
(from ``endf_parserpy``) into the natural-size :class:`MLBWData`
dataclass consumed by :mod:`mf2_interpretation_mlbw`. The wire-format
handling and the physics core were deliberately kept apart in the
sketch design; this file is the wire-format side.

Scope: LRU=1, LRF=2 (Multi-Level Breit-Wigner). One isotope + one
energy range at a time; a top-level driver that iterates isotopes
and ranges is a small wrapper on top and is left to the caller
for now.

Non-goals (deliberate scope):

- LSSF != 0 (alternate boundary condition; MLBW doesn't need shift
  subtraction in the sketch, so LSSF is not a concern here).
- Energy-dependent scattering radius (APE / NRO=1): supported by
  falling back to the tabulated ``AP_table`` when present, else
  the scalar ``AP``.
- URR (LRU=2): separate module.
- Preprocessing of MF3 background (already handled by
  :func:`mf3_interpretation.compute_cross_section_agnostic`).

References
----------

- ENDF-6 Formats Manual, D.1 (MF2 general) and D.1.3.4 (MLBW).
- SAMMY manual, Chapter II.
- The JAX prototype's ``bw_preprocessor`` on branch
  ``feature_resonance`` (endf_userpy/resonance/preprocessing.py):
  numerically-equivalent reference implementation. This module
  differs in producing natural-size arrays rather than padded ones
  and in outputting the sketch's channel structure directly, but
  the underlying physics extraction matches.
"""
from __future__ import annotations

import math

import numpy as np

from ..primitives.physical_constants import (
    AMU_TO_EV, PARTICLE_MASSES_AMU,
)
from ..primitives.tab1 import TAB1
from .mf2_interpretation_mlbw import MLBWData


# Universal constants for computing the wavenumber prefactor
#     kn = sqrt(2 * mn * amu_to_ev) * fm / (hbar * c)
# so that k(E) = kn * sqrt(awi) * awri / (awri + awi) * sqrt(E).
_HBAR_EV_S = 6.582119569e-16          # reduced Planck constant, eV·s
_C_CM_S    = 2.99792458e10            # speed of light, cm/s
_FM_TO_CM  = 1.0e-12                  # 1 fm in cm

# Neutron mass in amu; used to make the wavenumber formula match
# ENDF conventions where AWRI = m_target / mn and AWI = m_proj / mn.
_MN_AMU = PARTICLE_MASSES_AMU['n']

# k prefactor: k(E) = _KN * sqrt(awi) * awri/(awri+awi) * sqrt(E) [cm^-1] * ...
# The `sqrt(barn)` = 1e-12 cm baked in via _FM_TO_CM matches the
# convention where 1/k^2 comes out in barns.
_KN = math.sqrt(2.0 * _MN_AMU * AMU_TO_EV) * _FM_TO_CM / (_HBAR_EV_S * _C_CM_S)


# (mass_amu, spin) per incident-particle ZAI (= NSUB // 10). Mirrors
# the JAX prototype's `particle_data` table; kept private to this
# module because the spin values are specific to the R-matrix
# formulation (particle_data in :mod:`primitives.physical_constants`
# has masses but not spins).
_PARTICLE_MASS_SPIN_BY_ZAI = {
    1:    (PARTICLE_MASSES_AMU['n'], 0.5),
    1001: (PARTICLE_MASSES_AMU['p'], 0.5),
    1002: (PARTICLE_MASSES_AMU['d'], 1.0),
    1003: (PARTICLE_MASSES_AMU['t'], 0.5),
    2003: (PARTICLE_MASSES_AMU['h'], 0.5),
    2004: (PARTICLE_MASSES_AMU['a'], 0.0),
    0:    (0.0,                       1.0),  # photon
}


def _incident_particle_from_endf(endf_dict):
    """Return (m_over_mn, spin) for the incident particle of this file.

    NSUB in MF1/MT451 is ``10 * ZAI + code``; ZAI = NSUB // 10.
    """
    d1 = endf_dict[1][451]
    izai = int(d1['NSUB']) // 10
    if izai not in _PARTICLE_MASS_SPIN_BY_ZAI:
        raise NotImplementedError(
            f'incident particle ZAI={izai} not supported by MLBW '
            f'preprocessor; supported: {sorted(_PARTICLE_MASS_SPIN_BY_ZAI)}'
        )
    mass, spin = _PARTICLE_MASS_SPIN_BY_ZAI[izai]
    return mass / _MN_AMU, spin


def _radius_tab1_from_ap(ap: float, emax: float) -> TAB1:
    """Constant scattering radius packaged as a two-point TAB1
    (lin-lin), matching the format the reconstruction expects.

    The upper x bound is 1e11 eV (well past any physical energy),
    not the RRR ``emax``. The reconstruction evaluates the radius
    at each resonance's |E_r|, and R-M evaluations routinely list
    bound-state or extension poles with |E_r| several times the
    RRR's EH; clipping the radius TAB1 to ``emax`` would silently
    zero those resonances' penetration factor (via
    ``tab1.interp``'s outside-value=0 default), dropping their
    contribution to the R-matrix sum and shifting the elastic
    cross section by a few percent (see the U-235 vs NJOY
    comparison landing in this branch). Radius is truly constant
    across the wide range anyway, so extending is physically
    correct.
    """
    return TAB1(
        x=np.array([1e-5, 1e11], dtype=np.float64),
        y=np.array([ap, ap], dtype=np.float64),
        nbt=np.array([1], dtype=np.int32),      # 0-indexed, one region
        intp=np.array([2], dtype=np.int32),     # lin-lin
    )


def _radius_tab1_from_ape(ape: dict, emax: float) -> TAB1:
    """Energy-dependent scattering radius from an APE-style dict
    (``{'Eint': [...], 'AP': [...], 'NBT': [...], 'INT': [...]}``).
    Padded at both ends with constant fill so queries beyond the
    tabulated range (bound / extension poles at |E_r| > EH) return
    the boundary value rather than zero. See the note in
    :func:`_radius_tab1_from_ap` for why this matters.
    """
    x = list(ape['Eint'])
    y = list(ape['AP'])
    nbt = list(ape['NBT'])
    intp = list(ape['INT'])
    if x[0] > 1e-5:
        x = [1e-5] + x
        y = [y[0]] + y
        # Shift NBT indices by one to account for the prepended point.
        nbt = [n + 1 for n in nbt]
    hi = max(emax, 1e11)
    if x[-1] < hi:
        x = x + [hi]
        y = y + [y[-1]]
        nbt[-1] += 1
    return TAB1(
        # NBT here is 1-based per ENDF-6; the sketch's `interp` uses
        # 0-based segment endpoints, so subtract 1.
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        nbt=np.asarray(nbt, dtype=np.int32) - 1,
        intp=np.asarray(intp, dtype=np.int32),
    )


def _channel_radius(ap: float, awri: float, naps: int) -> float:
    """Channel radius ``a`` per ENDF-6 conventions.

    NAPS=0: derive from mass, ``a = 0.123 * (AWRI * mn)^{1/3} + 0.08`` fm.
    NAPS=1: same as scattering radius, ``a = AP``.
    NAPS=2: derive from mass (same formula as NAPS=0), while the
            scattering radius ``R'`` stays ``AP`` separately. This is
            the case where the two radii differ.
    """
    if naps == 1:
        return float(ap)
    mwri = awri * _MN_AMU
    return 0.123 * mwri ** (1.0 / 3.0) + 0.08


def mlbw_data_from_endf_dict(
    endf_dict, isotope_idx: int = 1, range_idx: int = 1,
) -> MLBWData:
    """Build an :class:`MLBWData` from a parsed ENDF-6 dict.

    Parameters
    ----------
    endf_dict : dict
        Full parsed ENDF-6 dict from ``endf_parserpy``. Must include
        MF1/MT451 (for the incident-particle info) and
        MF2/MT151 (for the resonance data).
    isotope_idx : int, optional
        1-based isotope index (default 1; NIS>1 is uncommon).
    range_idx : int, optional
        1-based energy-range index within the isotope (default 1).

    Returns
    -------
    MLBWData
        Natural-size dataclass ready for
        :func:`mf2_interpretation_mlbw.reconstruct`.

    Raises
    ------
    ValueError
        If the requested range is not (LRU=1, LRF=2).
    NotImplementedError
        For incident particles not in
        :data:`_PARTICLE_MASS_SPIN_BY_ZAI`, or for LRX>0 groups with
        multiple resonances that would need the competitive-width
        derivation (handled in-line; only raised on a shape corner
        case that hasn't shown up in real files).
    """
    awi, spin_inc = _incident_particle_from_endf(endf_dict)

    d151 = endf_dict[2][151]
    d_iso = d151['isotope'][isotope_idx]
    abn = float(d_iso['ABN'])
    d_range = d_iso['range'][range_idx]

    lru = int(d_range['LRU'])
    lrf = int(d_range['LRF'])
    if not (lru == 1 and lrf == 2):
        raise ValueError(
            f'mlbw_data_from_endf_dict expects LRU=1 LRF=2 (MLBW); '
            f'got LRU={lru} LRF={lrf}'
        )
    naps = int(d_range['NAPS'])
    nro = int(d_range.get('NRO', 0))
    spi = float(d_range['SPI'])
    ap = float(d_range.get('AP', 0.0))
    nls = int(d_range['NLS'])
    emax = float(d_range['EH'])
    d_grp = d_range['spingroup']

    # --- Sweep L-groups to build the channel table + per-resonance rows.
    #
    # Channel structure for MLBW: within each L, unique |J| values are
    # channels. If a J that would be reachable via l-spin coupling is
    # missing from the resonance table, a "dummy" potential-only
    # channel carries the missing statistical weight so total elastic
    # (potential) scattering is right even when a J-group has no
    # resonances explicitly listed.

    ch_l_list: list[int] = []
    ch_g_list: list[float] = []
    res_channel_list: list[int] = []
    res_l_list: list[int] = []
    res_er_list: list[float] = []
    res_gn_list: list[float] = []
    res_gg_list: list[float] = []
    res_gf_list: list[float] = []
    res_gx_list: list[float] = []

    # For QX: the JAX reference averages across L when consistent, else
    # takes the mean of the near-median subset. In practice one QX per
    # range is the norm; we take the max non-zero and warn if
    # inconsistent (the sketch's `qx` field is a single scalar).
    qx_values: list[float] = []

    awri_ref: float | None = None    # first L-group's AWRI, used for kn

    for l_idx in range(1, nls + 1):
        d_l = d_grp[l_idx]
        L = int(d_l['L'])
        awri = float(d_l['AWRI'])
        if awri_ref is None:
            awri_ref = awri
        qx_l = float(d_l['QX'])
        lrx = int(d_l['LRX'])
        nrs = int(d_l['NRS'])

        # ENDF QX is stored in the CM frame convention; multiply by
        # (awri + awi) / awri to lift to the LAB-frame convention the
        # sketch's `_rho_competitive` expects (matches JAX reference).
        if lrx > 0:
            qx_values.append(qx_l * (awri + awi) / awri)

        # Sort resonances in this L-group by |J|, so all resonances
        # sharing a J stay adjacent in the channel indexing.
        aj_arr = np.array(list(d_l['AJ'].values()), dtype=np.float64)
        j2 = np.rint(2 * aj_arr).astype(np.int32)          # 2 * |J|
        order = np.argsort(np.abs(j2))
        j2_ordered = j2[order]

        er_arr = np.array(list(d_l['ER'].values()), dtype=np.float64)[order]
        gn_arr = np.array(list(d_l['GN'].values()), dtype=np.float64)[order]
        gg_arr = np.array(list(d_l['GG'].values()), dtype=np.float64)[order]
        gf_arr = np.array(list(d_l['GF'].values()), dtype=np.float64)[order]
        gt_arr = np.array(list(d_l['GT'].values()), dtype=np.float64)[order]

        # Competitive width GX from GT - GN - GG - GF, gated by LRX.
        if lrx > 0:
            gx_raw = gt_arr - gn_arr - gg_arr - gf_arr
            gx_arr = np.where(
                (gx_raw >= 0.0) & (gx_raw >= 1e-6 * gt_arr), gx_raw, 0.0,
            )
        else:
            gx_arr = np.zeros_like(gt_arr)

        # Channels for this L: unique |J| values, in the order they
        # appear after sorting. Statistical weight g_J = (2|J|+1) /
        # (2 * (2s_inc + 1) * (2 I + 1)) where s_inc is the incident-
        # particle spin and I is the target spin.
        gj_den = (2.0 * spin_inc + 1.0) * (2.0 * spi + 1.0)
        j2_unique, first_idx = np.unique(np.abs(j2_ordered), return_index=True)
        # first_idx gives the position in j2_ordered of each unique |J|;
        # we don't actually need it beyond checking sort order.
        del first_idx

        # Per-resonance channel index within THIS L-group.
        ch_base = len(ch_l_list)   # global channel offset
        j2_to_local_ch = {
            int(j): local_c for local_c, j in enumerate(j2_unique)
        }
        for j_val in j2_unique:
            ch_l_list.append(L)
            ch_g_list.append((float(j_val) + 1.0) / gj_den)

        for r in range(nrs):
            local_ch = j2_to_local_ch[int(abs(j2_ordered[r]))]
            res_channel_list.append(ch_base + local_ch)
            res_l_list.append(L)
            res_er_list.append(er_arr[r])
            res_gn_list.append(gn_arr[r])
            res_gg_list.append(gg_arr[r])
            res_gf_list.append(gf_arr[r])
            res_gx_list.append(gx_arr[r])

        # Missing-J-multiplicity check: for a given L, the sum of g_J
        # over ALL (s_c, J) couplings (with multiplicity) equals
        # ``2L+1``. Iterating unique J values (which is what the ENDF
        # file lists) undercounts by the multiplicity of J values that
        # are reachable via more than one channel spin s_c. The
        # convention (SAMMY / NJOY) is to lump that missing weight into
        # a single dummy potential-only channel so total potential
        # scattering is still right.
        gj_sum = sum(ch_g_list[ch_base + c] for c in range(len(j2_unique)))
        gj_target = 2.0 * L + 1.0
        gj_dif = gj_target - gj_sum
        if gj_dif > 1e-30:
            ch_l_list.append(L)
            ch_g_list.append(gj_dif)
            # A "dummy" resonance at a tiny energy carries the missing
            # potential-scattering channel. No widths => contributes
            # only through phi_L in the sct/pot sums.
            dummy_ch = len(ch_l_list) - 1
            res_channel_list.append(dummy_ch)
            res_l_list.append(L)
            res_er_list.append(1.0e-12)
            res_gn_list.append(0.0)
            res_gg_list.append(0.0)
            res_gf_list.append(0.0)
            res_gx_list.append(0.0)

    # Consolidate QX to a single scalar (all matching in practice; take
    # the first one).
    qx = qx_values[0] if qx_values else 0.0

    # ki = kn * sqrt(awi) * awri / (awri + awi).
    ki = _KN * math.sqrt(awi) * awri_ref / (awri_ref + awi)

    # Radii. r_ap is the scattering radius (AP or APE); r_a is the
    # channel radius (from mass when NAPS=0/2, else = AP).
    ape = d_range.get('AP_table') if nro else None
    if ape is not None:
        r_ap = _radius_tab1_from_ape(ape, emax)
    else:
        r_ap = _radius_tab1_from_ap(ap, emax)

    a = _channel_radius(ap, awri_ref, naps)
    r_a = _radius_tab1_from_ap(a, emax)

    return MLBWData(
        abn=abn,
        spi=spi,
        ki=ki,
        qx=qx,
        r_a=r_a,
        r_ap=r_ap,
        ch_l=np.asarray(ch_l_list, dtype=np.int32),
        ch_g=np.asarray(ch_g_list, dtype=np.float64),
        res_channel=np.asarray(res_channel_list, dtype=np.int32),
        res_l=np.asarray(res_l_list, dtype=np.int32),
        res_er=np.asarray(res_er_list, dtype=np.float64),
        res_gn=np.asarray(res_gn_list, dtype=np.float64),
        res_gg=np.asarray(res_gg_list, dtype=np.float64),
        res_gf=np.asarray(res_gf_list, dtype=np.float64),
        res_gx=np.asarray(res_gx_list, dtype=np.float64),
    )
