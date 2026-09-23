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

from ..primitives import array_ns
from ..primitives.helpers import dict2array
from .mf2_interpretation_reichmoore import RMData
from .mf2_interpretation_mlbw_preproc import (
    _KN,
    _channel_radius,
    _get_l_group,
    _incident_particle_from_endf,
    _radius_tab1_from_ap,
    _radius_tab1_from_ape,
)


def rm_data_from_endf_dict(
    endf_dict, isotope_idx: int = 1, range_idx: int = 1, xp=None,
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

    ``xp`` : optional backend adapter (``array_ns.get_backend(name)``).
        ``xp=None`` (default) is numpy and preserves the pre-port
        behaviour bit-for-bit. Passing a JAX adapter routes the
        per-resonance width and energy marshaling (``ER``, ``GN``,
        ``GG``) through :func:`~primitives.helpers.dict2array`'s
        xp-aware code path so JAX tracers stored at those dict
        leaves propagate through the dataclass into the R-matrix
        reconstruction (issue #159, dict-first parity with MF6
        LAW=2). ``AJ`` (channel-spin sign marker + |J| bucketing),
        ``SPI`` / ``AWRI`` (channel radius, g_J denominator), and
        the fission-channel widths ``GFA`` / ``GFB`` (used in the
        ``has_gfa`` / ``has_gfb`` Python-side ``group_nfis``
        classification) stay concrete numpy on purpose.

    Raises
    ------
    ValueError
        If the requested range is not (LRU=1, LRF=3).
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    awi, spin_inc = _incident_particle_from_endf(endf_dict)

    d151 = endf_dict[2][151]
    d_iso = d151['isotope'][isotope_idx]
    abn = np.asarray(d_iso['ABN'], dtype=np.float64)
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
    spi = np.asarray(d_range['SPI'], dtype=np.float64)
    ap = np.asarray(d_range.get('AP', 0.0), dtype=np.float64)
    nls = int(d_range['NLS'])
    emax = np.asarray(d_range['EH'], dtype=np.float64)
    d_grp = _get_l_group(d_range)

    spin_inc_val = float(spin_inc)
    spi_val = float(spi)

    def _n_chan_spin(L: int, j2: int) -> int:
        """Count channel spins S in {|I-i|, ..., I+i} that admit
        the coupling |L - S| <= |J| <= L + S. For I=0 there is a
        single channel spin (S=1/2 for neutron scattering); for
        I>0 there are two."""
        # 2*S iterated as an integer to avoid float drift.
        two_s_lo = int(round(abs(spi_val - spin_inc_val) * 2))
        two_s_hi = int(round((spi_val + spin_inc_val) * 2))
        n = 0
        for two_s in range(two_s_lo, two_s_hi + 1, 2):
            two_s_val = two_s
            # |L - S|*2 <= j2 <= (L+S)*2
            two_L = 2 * L
            if abs(two_L - two_s_val) <= j2 <= two_L + two_s_val:
                n += 1
        return n

    # Collate resonances by (L, |J|, channel-spin marker). ENDF-6
    # LRF=3 encodes the channel-spin ambiguity via the sign of AJ:
    # for target-spin I where (L, |J|) can couple through more than
    # one channel spin S = I ± 1/2 (e.g. K-39 L=1 with I=3/2, where
    # J=1 and J=2 each admit S=1 and S=2), the evaluator marks
    # AJ > 0 for one channel spin and AJ < 0 for the other. Merging
    # AJ=+J and AJ=-J into a single group is wrong: (i) it mixes
    # resonance R-matrix contributions from disjoint channel-spin
    # blocks, and (ii) it drops one channel spin's g_J from the
    # potential sum (Σ_group g_J at that L would fall short of the
    # physical 2L+1). The channel-spin marker is 0 when AJ = 0
    # (unambiguous J=0 case), +1 for AJ > 0, -1 for AJ < 0. AJ = 0
    # never coexists with another sign at the same (L, |J|) in
    # real files.
    per_JPi: dict[tuple[int, int, int], list[tuple[float, float, float, float, float]]] = {}

    awri_ref: float | None = None    # first L-group's AWRI, used for ki
    apl_by_L: dict[int, float] = {}   # L -> APL if provided per L, 0 else

    for l_idx in range(1, nls + 1):
        d_l = d_grp[l_idx]
        L = int(d_l['L'])
        awri = np.asarray(d_l['AWRI'], dtype=np.float64)
        apl = np.asarray(d_l.get('APL', 0.0), dtype=np.float64)
        apl_by_L[L] = float(apl)
        if awri_ref is None:
            awri_ref = awri

        aj_arr = list(d_l['AJ'].values())
        # ER / GN / GG are the leaves users trace for autodiff; route
        # through dict2array's xp-aware path so JAX tracers survive.
        er_arr = dict2array(d_l['ER'], dtype=float, xp=xp)
        gn_arr = dict2array(d_l['GN'], dtype=float, xp=xp)
        gg_arr = dict2array(d_l['GG'], dtype=float, xp=xp)
        # GFA / GFB stay concrete: they feed the has_gfa / has_gfb
        # Python-side test that fixes group_nfis (a static int per
        # group), and they are not typical fitting targets.
        gfa_arr = list(d_l.get('GFA', {}).values()) or [0.0] * len(aj_arr)
        gfb_arr = list(d_l.get('GFB', {}).values()) or [0.0] * len(aj_arr)

        for i in range(len(aj_arr)):
            aj = float(aj_arr[i])
            j2 = int(round(abs(aj) * 2))
            if aj > 0:
                spin_mark = +1
            elif aj < 0:
                spin_mark = -1
            else:
                spin_mark = 0
            key = (L, j2, spin_mark)
            per_JPi.setdefault(key, []).append((
                er_arr[i],
                gn_arr[i],
                gg_arr[i],
                float(gfa_arr[i]),
                float(gfb_arr[i]),
            ))

    # Phantom groups for missing channel spins: for each L in the
    # file, walk every physically-allowed |J| coupling and count
    # how many channel spins the file listed resonances for at
    # that (L, |J|). If fewer than the physics admits (interior J
    # values on a target with I > 0 typically admit two S=I±1/2
    # channels), add one phantom group per missing channel spin
    # so the reconstruction emits its potential-only U = Ω²
    # contribution to elastic. This mirrors NJOY reconr's kkkkkk=2
    # branch in csrmat (reconr.f90 lines 3378-3487), which adds
    # `termn += 2*gj*(1 - cos(2φ)) = 4*gj*sin²(φ)` for each missing
    # channel spin — the same quantity a phantom group produces
    # via `sct_g = (π/k²) g_J |1 - Ω²|² = 4*(π/k²) g_J sin²(φ)`.
    # Phantom sign markers start at +2 and count up so they never
    # collide with the real +1 / -1 / 0 markers.
    Ls_in_file = sorted({key[0] for key in per_JPi.keys()})
    phantom_mark = 2
    for L in Ls_in_file:
        two_s_lo = int(round(abs(spi_val - spin_inc_val) * 2))
        two_s_hi = int(round((spi_val + spin_inc_val) * 2))
        allowed_j2 = set()
        for two_s in range(two_s_lo, two_s_hi + 1, 2):
            j2_lo = abs(2 * L - two_s)
            j2_hi = 2 * L + two_s
            for j2 in range(j2_lo, j2_hi + 1, 2):
                allowed_j2.add(j2)
        for j2 in allowed_j2:
            n_present = sum(
                1 for (Lf, jf, _sm) in per_JPi.keys()
                if Lf == L and jf == j2
            )
            n_chan = _n_chan_spin(L, j2)
            for _ in range(n_chan - n_present):
                per_JPi[(L, j2, phantom_mark)] = []
                phantom_mark += 1

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
        L, j2, _spin_mark = key
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

    # Per-group scattering / channel radius. For each spin group,
    # look up its L; use APL[L] if the file provides a per-L
    # override (non-zero), else the range-level AP. Then compute
    # the channel radius via _channel_radius(this-group's r_ap,
    # AWRI, NAPS). Applies per group so the reconstruction uses
    # the correct r_a / r_ap for the group's L. See JEFF-4.0
    # Fe-56 for a real file where per-L APL differs.
    def _r_ap_for_L(L: int) -> float:
        apl_val = apl_by_L.get(L, 0.0)
        return apl_val if apl_val > 0 else float(ap)

    group_r_ap_arr = np.zeros(ngroups, dtype=np.float64)
    group_r_a_arr = np.zeros(ngroups, dtype=np.float64)
    for g, (L, _j2, _spin) in enumerate(group_keys):
        r_ap_g = _r_ap_for_L(L)
        group_r_ap_arr[g] = r_ap_g
        group_r_a_arr[g] = _channel_radius(r_ap_g, awri_ref, naps)

    # Range-level r_a / r_ap TAB1s: kept for backward compat and
    # NRO=1 (energy-dependent scattering radius) support. Fill
    # from the first L-group's radius; reconstruction prefers
    # the per-group arrays above and falls back to these TAB1s
    # only if the per-group arrays are absent.
    r_ap_val_range = _r_ap_for_L(int(group_l[0])) if ngroups else float(ap)
    ape = d_range.get('AP_table') if nro else None
    if ape is not None:
        r_ap = _radius_tab1_from_ape(ape, emax)
    else:
        r_ap = _radius_tab1_from_ap(r_ap_val_range, emax)

    a = _channel_radius(r_ap_val_range, awri_ref, naps)
    r_a = _radius_tab1_from_ap(a, emax)

    # Integer-typed group / channel bookkeeping stays on numpy.
    # Per-resonance ER / GN / GG lists may contain JAX tracer
    # scalars; route through xp so they survive into the dataclass.
    # GFA / GFB kept on numpy (see the per-resonance loop above).
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
        res_er=xp.asarray(res_er_list, dtype=xp.float64),
        res_gn=xp.asarray(res_gn_list, dtype=xp.float64),
        res_gg=xp.asarray(res_gg_list, dtype=xp.float64),
        res_gf1=np.asarray(res_gf1_list, dtype=np.float64),
        res_gf2=np.asarray(res_gf2_list, dtype=np.float64),
        group_r_a=group_r_a_arr,
        group_r_ap=group_r_ap_arr,
    )
