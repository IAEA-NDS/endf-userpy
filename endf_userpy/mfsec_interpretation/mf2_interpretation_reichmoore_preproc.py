"""ENDF-6 MF2/MT151 -> :class:`RMData` preprocessing.

Reich-Moore counterpart of :mod:`mf2_interpretation_mlbw_preproc`.
Lifts a parsed ENDF-6 dict (from ``endf_parserpy``) with a LRU=1
LRF=3 range into the natural-size :class:`RMData` dataclass that
:mod:`mf2_interpretation_reichmoore` consumes.

Scope: LRU=1, LRF=3. One isotope + one energy range at a time.

Sharing with :mod:`mf2_interpretation_mlbw_preproc`
--------------------------------------------------

Constants (wavenumber prefactor, particle table), radius helpers
and the incident-particle lookup are the same as MLBW's; imported
directly from that module rather than duplicated. The two
preprocessors remain independent public entry points but share the
low-level physical-constants / TAB1 plumbing.

Non-goals (deliberate scope):

- Alternate ``B_c`` boundary condition on the R-matrix. The R-M
  reconstruction uses SAMMY / NJOY-reconr's shift-eliminated
  convention (``B_c = S_c(|E_r|)``, equivalently ``L̃_c(E) = i P_c(E)``
  after the level shift is absorbed into ``E_r``). Every real
  LRU=1 ENDF-6 evaluation follows this convention; the format
  does not carry a boundary-condition flag on LRU=1 range
  records. Earlier drafts of this docstring called this
  "``LSSF != 0``", which is a URR-only flag (see LRU=2 below) and
  a misnomer here.
- URR (LRU=2): separate module. When implemented, LSSF=1 URR
  (MF3 already carries the average XS) is a no-op; LSSF=0 URR
  needs actual URR reconstruction.
"""
from __future__ import annotations

import math

import numpy as np

from .mf2_interpretation_reichmoore import RMData
from .mf2_interpretation_mlbw_preproc import (
    _KN,
    _channel_radius,
    _incident_particle_from_endf,
    _radius_tab1_from_ap,
    _radius_tab1_from_ape,
)


def rm_data_from_endf_dict(
    endf_dict, isotope_idx: int = 1, range_idx: int = 1,
) -> RMData:
    """Build an :class:`RMData` from a parsed ENDF-6 dict.

    Reich-Moore lays resonances out by L-group in the file, with each
    resonance carrying its own signed ``AJ``. Physically the reaction
    channels group by (L, |J|), so this preprocessor collates each
    L-group's resonances into (L, |J|)-keyed buckets, one J·π group
    per unique key.

    Parameters
    ----------
    endf_dict : dict
        Full parsed ENDF-6 dict from ``endf_parserpy``. Must include
        MF1/MT451 (for the incident-particle info) and
        MF2/MT151 (for the resonance data).
    isotope_idx : int, optional
        1-based isotope index (default 1).
    range_idx : int, optional
        1-based energy-range index within the isotope (default 1).

    Returns
    -------
    RMData
        Natural-size dataclass ready for
        :func:`mf2_interpretation_reichmoore.reconstruct`.

    Raises
    ------
    ValueError
        If the requested range is not (LRU=1, LRF=3).
    """
    awi, spin_inc = _incident_particle_from_endf(endf_dict)

    d151 = endf_dict[2][151]
    d_iso = d151['isotope'][isotope_idx]
    abn = float(d_iso['ABN'])
    d_range = d_iso['range'][range_idx]

    lru = int(d_range['LRU'])
    lrf = int(d_range['LRF'])
    if not (lru == 1 and lrf == 3):
        raise ValueError(
            f'rm_data_from_endf_dict expects LRU=1 LRF=3 (Reich-Moore); '
            f'got LRU={lru} LRF={lrf}'
        )
    naps = int(d_range['NAPS'])
    nro = int(d_range.get('NRO', 0))
    spi = float(d_range['SPI'])
    ap = float(d_range.get('AP', 0.0))
    nls = int(d_range['NLS'])
    emax = float(d_range['EH'])
    d_grp = d_range['spingroup']

    # Collate resonances by (L, |J|). endf_parserpy renders LRF=3
    # spingroups one-per-L; individual resonances inside carry their
    # own signed AJ. Sign of AJ marks spin-group parity but does NOT
    # add channels beyond what |J| identifies (the fine-structure
    # coupling is spin-averaged in this convention).
    per_JPi: dict[tuple[int, int], list[tuple[float, float, float, float, float]]] = {}

    awri_ref: float | None = None    # first L-group's AWRI, used for ki
    apl_ref: float | None = None      # first L-group's APL if given

    for l_idx in range(1, nls + 1):
        d_l = d_grp[l_idx]
        L = int(d_l['L'])
        awri = float(d_l['AWRI'])
        apl = float(d_l.get('APL', 0.0))
        if awri_ref is None:
            awri_ref = awri
            apl_ref = apl if apl > 0 else None

        aj_arr = list(d_l['AJ'].values())
        er_arr = list(d_l['ER'].values())
        gn_arr = list(d_l['GN'].values())
        gg_arr = list(d_l['GG'].values())
        gfa_arr = list(d_l.get('GFA', {}).values()) or [0.0] * len(aj_arr)
        gfb_arr = list(d_l.get('GFB', {}).values()) or [0.0] * len(aj_arr)

        for i in range(len(aj_arr)):
            j2 = int(round(abs(float(aj_arr[i])) * 2))
            key = (L, j2)
            per_JPi.setdefault(key, []).append((
                float(er_arr[i]),
                float(gn_arr[i]),
                float(gg_arr[i]),
                float(gfa_arr[i]),
                float(gfb_arr[i]),
            ))

    # Order groups deterministically: by (L, |J|). Reconstruction is
    # invariant to ordering (it iterates ngroups), but a stable order
    # keeps snapshot tests reproducible.
    group_keys = sorted(per_JPi.keys())
    ngroups = len(group_keys)

    group_l = np.zeros(ngroups, dtype=np.int32)
    group_g = np.zeros(ngroups, dtype=np.float64)
    group_nfis = np.zeros(ngroups, dtype=np.int32)

    res_group_list: list[int] = []
    res_er_list: list[float] = []
    res_gn_list: list[float] = []
    res_gg_list: list[float] = []
    res_gf1_list: list[float] = []
    res_gf2_list: list[float] = []

    gj_den = (2.0 * spin_inc + 1.0) * (2.0 * spi + 1.0)

    for g, key in enumerate(group_keys):
        L, j2 = key
        group_l[g] = L
        group_g[g] = (float(j2) + 1.0) / gj_den

        # nfis: 0 if all GFA and GFB are zero for every resonance in
        # this group; 1 if any GFA is non-zero; 2 if any GFB is
        # non-zero (the second fission channel is only present when
        # the file lists a non-zero GFB somewhere in the group).
        has_gfa = any(abs(r[3]) > 0 for r in per_JPi[key])
        has_gfb = any(abs(r[4]) > 0 for r in per_JPi[key])
        if has_gfb:
            group_nfis[g] = 2
        elif has_gfa:
            group_nfis[g] = 1
        else:
            group_nfis[g] = 0

        for er, gn, gg, gfa, gfb in per_JPi[key]:
            res_group_list.append(g)
            res_er_list.append(er)
            res_gn_list.append(gn)
            res_gg_list.append(gg)
            res_gf1_list.append(gfa)
            res_gf2_list.append(gfb)

    # ki = kn * sqrt(awi) * awri / (awri + awi).
    ki = _KN * math.sqrt(awi) * awri_ref / (awri_ref + awi)

    # Scattering radius: APL (per L, first-non-zero) takes precedence
    # over range-level AP if the file provides it; otherwise AP.
    r_ap_val = apl_ref if apl_ref else ap

    ape = d_range.get('AP_table') if nro else None
    if ape is not None:
        r_ap = _radius_tab1_from_ape(ape, emax)
    else:
        r_ap = _radius_tab1_from_ap(r_ap_val, emax)

    a = _channel_radius(r_ap_val, awri_ref, naps)
    r_a = _radius_tab1_from_ap(a, emax)

    return RMData(
        abn=abn,
        spi=spi,
        ki=ki,
        r_a=r_a,
        r_ap=r_ap,
        group_l=group_l,
        group_g=group_g,
        group_nfis=group_nfis,
        res_group=np.asarray(res_group_list, dtype=np.int32),
        res_er=np.asarray(res_er_list, dtype=np.float64),
        res_gn=np.asarray(res_gn_list, dtype=np.float64),
        res_gg=np.asarray(res_gg_list, dtype=np.float64),
        res_gf1=np.asarray(res_gf1_list, dtype=np.float64),
        res_gf2=np.asarray(res_gf2_list, dtype=np.float64),
    )
