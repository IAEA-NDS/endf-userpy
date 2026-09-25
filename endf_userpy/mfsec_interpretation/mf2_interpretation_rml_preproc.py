"""ENDF-6 MF2/MT151 -> :class:`RMLData` preprocessing.

R-Matrix Limited (LRF=7) counterpart of
:mod:`mf2_interpretation_reichmoore_preproc`. Lifts a parsed
ENDF-6 dict (from ``endf_parserpy``) with an ``LRU=1 LRF=7`` range
into the natural-size :class:`RMLData` dataclass that the (upcoming)
:mod:`mf2_interpretation_rml` reconstruction consumes.

Scope: ``LRU=1, LRF=7, KRM=3, KRL=0, IFG=0, NRO=0, KBK=0, KPS=0``.
See the module docstring in :mod:`mf2_interpretation_rml` for the
full arc plan (#228). One isotope + one energy range at a time.

Sharing with :mod:`mf2_interpretation_mlbw_preproc`
--------------------------------------------------

Universal constants, radius helpers, and the incident-particle
lookup are imported from the MLBW preprocessor rather than
duplicated. The two preprocessors remain independent public entry
points but share the low-level physical-constants and TAB1
plumbing.
"""
from __future__ import annotations

import numpy as np

from ..primitives import array_ns
from ..primitives.helpers import dict2array
from .mf2_interpretation_rml import RMLData
from .mf2_interpretation_mlbw_preproc import (
    _KN,
    _incident_particle_from_endf,
    _radius_tab1_from_ap,
)


def _statistical_weight(aj: float, spi: float, spin_inc: float) -> float:
    """Statistical weight ``g_J = (2|J| + 1) / ((2I + 1)(2i + 1))``
    for a J-group with target spin ``I`` and incident spin ``i``.

    LRF=7 files carry ``AJ`` signed (sign encodes parity for
    KPS=0). ``|AJ|`` is the physical ``J``.
    """
    return (2.0 * abs(aj) + 1.0) / ((2.0 * spi + 1.0) * (2.0 * spin_inc + 1.0))


def _identify_incident_pair(pp_mt: np.ndarray, pp_ma: np.ndarray) -> int:
    """Return the 1-based particle-pair index of the incident
    (elastic) pair.

    LRF=7 elastic is stored as MT=2 with the neutron (mass 1 in
    neutron units) as particle A of the pair. Every LRU=1 range
    has exactly one such pair; the preprocessor asserts on that.
    """
    for i in range(pp_mt.shape[0]):
        if int(round(float(pp_mt[i]))) == 2 and float(pp_ma[i]) > 0.5:
            return i + 1
    raise ValueError(
        'LRF=7 range does not declare an incident (MT=2, neutron '
        'particle A) elastic pair; every LRU=1 range must have '
        'exactly one.'
    )


def rml_data_from_endf_dict(
    endf_dict, isotope_idx: int = 1, range_idx: int = 1, xp=None,
) -> RMLData:
    """Build an :class:`RMLData` from a parsed ENDF-6 dict.

    Parameters
    ----------
    endf_dict : dict
        Full parsed ENDF-6 dict from ``endf_parserpy``. Must
        include MF1/MT451 (incident particle) and MF2/MT151
        (resonance data).
    isotope_idx : int, optional
        1-based isotope index (default 1).
    range_idx : int, optional
        1-based energy-range index within the isotope (default 1).
    xp : optional backend adapter (``array_ns.get_backend(name)``).
        ``xp=None`` (default) is numpy. Passing a JAX adapter
        routes the per-resonance-parameter marshaling through
        :func:`~primitives.helpers.dict2array`'s xp-aware code
        path so JAX tracers stored at file-side leaves
        (``ER``, ``GAM``) propagate through the dataclass into
        the reconstruction. Values that steer the channel
        bookkeeping (``AJ``, ``NCH``, ``PPI``, ``L``, ``SCH``,
        ``AP``) stay concrete on purpose; tracing them would
        require rewriting the Python channel loop with static
        shape masking.

    Returns
    -------
    RMLData

    Raises
    ------
    ValueError
        If the requested range is not ``LRU=1, LRF=7``.
    NotImplementedError
        For unsupported ``KRM`` / ``KRL`` / ``IFG`` / ``NRO`` /
        ``KBK`` / ``KPS`` values (see the arc's initial scope).
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    awi, spin_inc = _incident_particle_from_endf(endf_dict)

    d151 = endf_dict[2][151]
    d_iso = d151['isotope'][isotope_idx]
    abn = float(np.asarray(d_iso['ABN'], dtype=np.float64))
    d_range = d_iso['range'][range_idx]

    if int(d_range['LRU']) != 1 or int(d_range['LRF']) != 7:
        raise ValueError(
            f'rml_data_from_endf_dict requires an LRU=1, LRF=7 '
            f'range; got LRU={d_range["LRU"]}, LRF={d_range["LRF"]}.'
        )

    krm = int(d_range['KRM'])
    krl = int(d_range['KRL'])
    ifg = int(d_range['IFG'])
    nro = int(d_range['NRO'])
    if krm != 3:
        raise NotImplementedError(
            f'LRF=7 KRM={krm} not supported by this arc; only '
            f'KRM=3 (Reich-Moore) is implemented. See #228.'
        )
    if krl != 0:
        raise NotImplementedError(
            f'LRF=7 KRL={krl} (relativistic) not supported; only '
            f'KRL=0 (non-relativistic) is implemented.'
        )
    if ifg != 0:
        raise NotImplementedError(
            f'LRF=7 IFG={ifg} (reduced-width amplitudes) not '
            f'supported; only IFG=0 (widths) is implemented in the '
            f'initial scope. Conversion at ingest is a follow-up.'
        )
    if nro != 0:
        raise NotImplementedError(
            f'LRF=7 NRO={nro} (energy-dependent scattering radius) '
            f'not supported; only NRO=0 in the initial scope.'
        )
    naps = int(d_range['NAPS'])

    npp = int(d_range['NPP'])
    pp_ma = dict2array(d_range['MA'], dtype=float, xp=xp)
    pp_mb = dict2array(d_range['MB'], dtype=float, xp=xp)
    pp_za = dict2array(d_range['ZA'], dtype=float, xp=xp)
    pp_zb = dict2array(d_range['ZB'], dtype=float, xp=xp)
    pp_ia = dict2array(d_range['IA'], dtype=float, xp=xp)
    pp_ib = dict2array(d_range['IB'], dtype=float, xp=xp)
    pp_q  = dict2array(d_range['Q'],  dtype=float, xp=xp)
    pp_pnt = dict2array(d_range['PNT'], dtype=float, xp=xp)
    pp_shf = dict2array(d_range['SHF'], dtype=float, xp=xp)
    pp_mt = dict2array(d_range['MT'], dtype=float, xp=xp)
    pp_pa = dict2array(d_range['PA'], dtype=float, xp=xp)
    pp_pb = dict2array(d_range['PB'], dtype=float, xp=xp)
    if int(pp_ma.shape[0]) != npp:
        raise ValueError(
            f'LRF=7 range declares NPP={npp} but MA has '
            f'{pp_ma.shape[0]} entries.'
        )

    incident_idx = _identify_incident_pair(pp_mt, pp_ma)

    # LRF=7 does NOT carry an explicit ``SPI`` scalar on the range
    # record (unlike LRF=1/2/3). Target spin is embedded in the
    # elastic pair's ``IB`` value; take the absolute value since IB
    # carries a sign that encodes the target's z-projection ordering
    # convention in ENDF (which is orthogonal to the g_J denominator).
    spi = float(abs(float(pp_ib[incident_idx - 1])))

    # ki for the elastic pair. The R-M / MLBW convention is
    #     k(E) = _KN * sqrt(awi) * mb / (mb + awi) * sqrt(E)
    # with awi the incident mass in mn units and mb the target mass
    # in mn units. For LRF=7 elastic, awi is the neutron mass (~1)
    # and mb is the target mass, i.e. pp_mb[elastic_idx - 1].
    mb_inc = float(pp_mb[incident_idx - 1])
    ki = float(_KN * np.sqrt(awi) * mb_inc / (mb_inc + awi))

    # Range-level scattering radius. NRO=0 -> constant AP over the
    # whole range; wrap in a TAB1 to keep the reconstruction
    # backend-uniform with LRF=2 / LRF=3.
    ap = float(d_range.get('AP', 0.0))
    emax = float(d_range['EH'])
    r_ap = _radius_tab1_from_ap(ap, emax)

    # ---- Per J-group + per-channel + per-resonance extraction ----
    sg = d_range.get('spingroup') or d_range.get('l_group')
    if sg is None:
        raise KeyError(
            'LRF=7 range has neither "spingroup" nor "l_group" — '
            'parser output shape unrecognised.'
        )
    njs = int(d_range['NJS'])
    if len(sg) != njs:
        raise ValueError(
            f'LRF=7 range declares NJS={njs} but spingroup has '
            f'{len(sg)} entries.'
        )

    group_aj = np.zeros(njs, dtype=np.float64)
    group_pj = np.zeros(njs, dtype=np.float64)
    group_g = np.zeros(njs, dtype=np.float64)
    group_nch = np.zeros(njs, dtype=np.int32)

    per_group_channels = []   # list of dicts of arrays
    per_group_resonances = []  # list of (er_array, gam_matrix)

    for gi, jkey in enumerate(sorted(sg.keys())):
        g = sg[jkey]
        if int(g.get('KBK', 0)) != 0:
            raise NotImplementedError(
                f'LRF=7 group {jkey} has KBK={g["KBK"]} '
                f'(background R-matrix) — not supported in the '
                f'initial scope.'
            )
        if int(g.get('KPS', 0)) != 0:
            raise NotImplementedError(
                f'LRF=7 group {jkey} has KPS={g["KPS"]} '
                f'(additional phase shift) — not supported in the '
                f'initial scope.'
            )
        aj = float(g['AJ'])
        pj = float(g.get('PJ', 0.0))
        nch = int(g['NCH'])
        group_aj[gi] = aj
        group_pj[gi] = pj
        group_nch[gi] = nch
        group_g[gi] = _statistical_weight(aj, spi, spin_inc)

        # Per-channel scalars for this group.
        ppi = dict2array(g['PPI'], dtype=float)
        ll  = dict2array(g['L'],   dtype=float)
        sch = dict2array(g['SCH'], dtype=float)
        bnd = dict2array(g['BND'], dtype=float)
        ape = dict2array(g['APE'], dtype=float)
        apt = dict2array(g['APT'], dtype=float)
        if not (ppi.shape[0] == ll.shape[0] == sch.shape[0] ==
                bnd.shape[0] == ape.shape[0] == apt.shape[0] == nch):
            raise ValueError(
                f'LRF=7 group {jkey} declares NCH={nch} but per-'
                f'channel arrays have inconsistent lengths: '
                f'PPI={ppi.shape[0]}, L={ll.shape[0]}, '
                f'SCH={sch.shape[0]}, BND={bnd.shape[0]}, '
                f'APE={ape.shape[0]}, APT={apt.shape[0]}.'
            )
        per_group_channels.append({
            'PPI': ppi, 'L': ll, 'SCH': sch,
            'BND': bnd, 'APE': ape, 'APT': apt,
        })

        # Per-resonance parameters.
        nrs = int(g['NRS'])
        if nrs == 0:
            er = np.zeros(0, dtype=np.float64)
            gam = np.zeros((0, nch), dtype=np.float64)
        else:
            er = dict2array(g['ER'], dtype=float, xp=xp)
            if int(er.shape[0]) != nrs:
                raise ValueError(
                    f'LRF=7 group {jkey} declares NRS={nrs} but ER '
                    f'has {er.shape[0]} entries.'
                )
            gam_dict = g['GAM']
            if len(gam_dict) != nch:
                raise ValueError(
                    f'LRF=7 group {jkey} declares NCH={nch} but '
                    f'GAM has {len(gam_dict)} channel rows.'
                )
            gam_columns = []
            for cix in sorted(gam_dict.keys()):
                col = dict2array(gam_dict[cix], dtype=float, xp=xp)
                if int(col.shape[0]) != nrs:
                    raise ValueError(
                        f'LRF=7 group {jkey} channel {cix} has '
                        f'{col.shape[0]} widths, expected NRS={nrs}.'
                    )
                gam_columns.append(col)
            # Stack to (nrs, nch) using xp so tracers on GAM leaves
            # survive.
            gam = xp.stack(gam_columns, axis=1)
        per_group_resonances.append((er, gam))

    max_nch = int(group_nch.max()) if njs > 0 else 0

    # Pack per-(group, channel) arrays with padding.
    ch_ppi = np.zeros((njs, max_nch), dtype=np.float64)
    ch_l = np.zeros((njs, max_nch), dtype=np.float64)
    ch_sch = np.zeros((njs, max_nch), dtype=np.float64)
    ch_bnd = np.zeros((njs, max_nch), dtype=np.float64)
    ch_ape = np.zeros((njs, max_nch), dtype=np.float64)
    ch_apt = np.zeros((njs, max_nch), dtype=np.float64)
    ch_active = np.zeros((njs, max_nch), dtype=bool)
    for gi, ch in enumerate(per_group_channels):
        nch = int(group_nch[gi])
        ch_ppi[gi, :nch] = np.asarray(ch['PPI'])
        ch_l[gi, :nch] = np.asarray(ch['L'])
        ch_sch[gi, :nch] = np.asarray(ch['SCH'])
        ch_bnd[gi, :nch] = np.asarray(ch['BND'])
        ch_ape[gi, :nch] = np.asarray(ch['APE'])
        ch_apt[gi, :nch] = np.asarray(ch['APT'])
        ch_active[gi, :nch] = True

    # Pack per-(resonance) and per-(resonance, channel) arrays.
    total_nres = sum(int(er.shape[0]) for er, _ in per_group_resonances)
    res_group = np.zeros(total_nres, dtype=np.int32)
    # ``res_er`` and ``res_gam`` are xp arrays so tracer file-side
    # widths survive.
    res_er_pieces = []
    res_gam_pieces = []
    cursor = 0
    for gi, (er, gam) in enumerate(per_group_resonances):
        n = int(er.shape[0])
        if n == 0:
            continue
        res_group[cursor:cursor + n] = gi
        res_er_pieces.append(er)
        # Pad ``gam`` from (n, group_nch[gi]) to (n, max_nch).
        nch = int(group_nch[gi])
        if nch < max_nch:
            zeros = xp.zeros((n, max_nch - nch), dtype=gam.dtype)
            gam_padded = xp.concatenate([gam, zeros], axis=1)
        else:
            gam_padded = gam
        res_gam_pieces.append(gam_padded)
        cursor += n

    if total_nres > 0:
        res_er = xp.concatenate(res_er_pieces, axis=0)
        res_gam = xp.concatenate(res_gam_pieces, axis=0)
    else:
        res_er = xp.zeros(0, dtype=xp.float64)
        res_gam = xp.zeros((0, max_nch), dtype=xp.float64)

    return RMLData(
        abn=abn,
        spi=spi,
        ki=ki,
        r_ap=r_ap,
        krm=krm,
        ifg=ifg,
        krl=krl,
        naps=naps,
        pp_ma=pp_ma,
        pp_mb=pp_mb,
        pp_za=pp_za,
        pp_zb=pp_zb,
        pp_ia=pp_ia,
        pp_ib=pp_ib,
        pp_q=pp_q,
        pp_pnt=pp_pnt,
        pp_shf=pp_shf,
        pp_mt=pp_mt,
        pp_pa=pp_pa,
        pp_pb=pp_pb,
        pp_incident_idx=incident_idx,
        group_aj=group_aj,
        group_pj=group_pj,
        group_g=group_g,
        group_nch=group_nch,
        ch_ppi=ch_ppi,
        ch_l=ch_l,
        ch_sch=ch_sch,
        ch_bnd=ch_bnd,
        ch_ape=ch_ape,
        ch_apt=ch_apt,
        ch_active=ch_active,
        res_group=res_group,
        res_er=res_er,
        res_gam=res_gam,
    )
