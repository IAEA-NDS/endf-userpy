"""ENDF-6 MF2/MT151 (LRU=2, LRF=2) -> :class:`URRData` preprocessing.

Case C (LRF=2, energy-dependent widths) unresolved-resonance
range. Modern actinide URR ranges (ENDF/B-VIII.1, TENDL-2021,
JEFF-4.0, JENDL-5 U/Pu/Th) all follow this format.

Shares low-level plumbing (wavenumber prefactor, incident-particle
lookup, channel-radius helper, TAB1-radius extenders) with
:mod:`mf2_interpretation_mlbw_preproc` and imports them from
there; the URR-specific code is what shapes the (nJ, NE) width
tables and pulls per-J-group scalars into their own arrays.

Scope
-----

- LRU=2, LRF=2. One isotope + one range at a time.
- Constant NE per J-group within a range (all real URR files
  the corpus has seen fix NE at the range level, so every
  J-group shares the same energy grid; if a future file has
  variable NE, the preproc will raise a clear error rather
  than silently corrupt the (nJ, NE) rectangular table).
- LSSF=0 and LSSF=1 both accepted; LSSF is a composition-layer
  concern, not a preprocessing one. This function returns the
  same URRData whether the caller intends to compose it with
  MF3 or leave MF3 as the average XS.
"""
from __future__ import annotations

import math

import numpy as np

from ..primitives import array_ns
from ..primitives.helpers import dict2array
from .mf2_interpretation_mlbw_preproc import (
    _KN,
    _channel_radius,
    _get_l_group,
    _incident_particle_from_endf,
    _radius_tab1_from_ap,
    _radius_tab1_from_ape,
)
from .mf2_interpretation_urr import URRData


def _get_j_group(d_l: dict) -> dict:
    """Return the per-J subsection table from a URR L-group dict.

    Same pattern as :func:`_get_l_group`: ``endf_parserpy`` names
    this table ``subsec`` from 0.17 onwards; older releases used
    ``j_group`` for the same content. Try the new name first and
    fall back to the old one so both parser versions work without
    pinning.
    """
    grp = d_l.get('subsec')
    if grp is None:
        grp = d_l.get('j_group')
    if grp is None:
        raise KeyError(
            "URR L-group has neither 'subsec' (endf_parserpy >= 0.17) "
            "nor 'j_group' (older) key; parser output shape unrecognised"
        )
    return grp


def urr_data_from_endf_dict(
    endf_dict, isotope_idx: int = 1, range_idx: int = 2, xp=None,
) -> URRData:
    """Build a :class:`URRData` from a parsed ENDF-6 dict.

    Parameters
    ----------
    endf_dict : dict
        Full parsed ENDF-6 dict from ``endf_parserpy``. Must include
        MF1/MT451 (for the incident-particle info) and MF2/MT151.
    isotope_idx : int, optional
        1-based isotope index (default 1).
    range_idx : int, optional
        1-based energy-range index within the isotope. Default is
        **2** because URR ranges follow the RRR range in the
        standard ENDF-6 layout (LRU=1 at range 1, LRU=2 at range
        2). Pass the actual index if the file's layout differs.

    Returns
    -------
    URRData
        Natural-size dataclass ready for
        :func:`mf2_interpretation_urr.reconstruct`.

    Raises
    ------
    ValueError
        If the requested range is not (LRU=2, LRF=2), or if the
        J-groups within the range do not share a common NE (the
        rectangular (nJ, NE) table shape would otherwise be
        ambiguous).
    NotImplementedError
        For incident particles the MLBW preproc's particle table
        does not cover.

    Notes
    -----
    ``xp`` (optional backend adapter, default numpy): passing a JAX
    adapter routes the per-J-group width-table marshaling through
    :func:`~primitives.helpers.dict2array`'s xp-aware code path so
    JAX tracers stored at ``d_j['ES']`` / ``d_j['D']`` /
    ``d_j['GN0']`` / ``d_j['GG']`` / ``d_j['GF']`` / ``d_j['GX']``
    survive into ``URRData`` and through the URR reconstruction
    end-to-end (issue #159). Scalars that steer channel bookkeeping
    (``AJ``, ``AMU*``, ``INT``, ``NAPS``, ``AWRI``, ``SPI``, ``AP``)
    stay concrete numpy on purpose.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    awi, _spin_inc = _incident_particle_from_endf(endf_dict)

    d151 = endf_dict[2][151]
    d_iso = d151['isotope'][isotope_idx]
    abn = np.asarray(d_iso['ABN'], dtype=np.float64)
    d_range = d_iso['range'][range_idx]

    lru = int(d_range['LRU'])
    lrf = int(d_range['LRF'])
    if not (lru == 2 and lrf == 2):
        raise ValueError(
            f'urr_data_from_endf_dict expects LRU=2 LRF=2 '
            f'(unresolved-resonance region, Case C); got '
            f'LRU={lru} LRF={lrf}'
        )
    naps = int(d_range['NAPS'])
    nro = int(d_range.get('NRO', 0))
    spi = np.asarray(d_range['SPI'], dtype=np.float64)
    ap = np.asarray(d_range.get('AP', 0.0), dtype=np.float64)
    emax = np.asarray(d_range['EH'], dtype=np.float64)
    nls = int(d_range['NLS'])
    d_grp = _get_l_group(d_range)

    # Walk L-groups -> J-groups, collecting per-J scalars and
    # width tables in flat order. Every real file has a single
    # AWRI shared across L-groups; we take it from L=1 and check
    # the others match.
    awri_ref: float | None = None
    ne_ref: int | None = None

    group_l: list[int] = []
    group_j2: list[int] = []
    group_amun: list[float] = []
    group_amug: list[float] = []
    group_amuf: list[float] = []
    group_amux: list[float] = []
    group_int: list[int] = []

    table_es_rows: list[np.ndarray] = []
    table_d_rows: list[np.ndarray] = []
    table_gn0_rows: list[np.ndarray] = []
    table_gg_rows: list[np.ndarray] = []
    table_gf_rows: list[np.ndarray] = []
    table_gx_rows: list[np.ndarray] = []

    for l_idx in range(1, nls + 1):
        d_l = d_grp[l_idx]
        L = int(d_l['L'])
        awri = float(d_l['AWRI'])
        if awri_ref is None:
            awri_ref = awri
        elif abs(awri - awri_ref) > 1e-9 * awri_ref:
            raise ValueError(
                f'URR range has AWRI={awri} at L-group {l_idx} but '
                f'first L-group carried AWRI={awri_ref}; the URRData '
                f'schema assumes a single AWRI per range. File this '
                f'as a limitation if you hit it.'
            )

        j_group = _get_j_group(d_l)
        for _j_idx, d_j in sorted(j_group.items()):
            aj = float(d_j['AJ'])
            j2 = int(round(abs(aj) * 2))
            ne = int(d_j['NE'])
            intp = int(d_j['INT'])
            if ne_ref is None:
                ne_ref = ne
            elif ne != ne_ref:
                raise ValueError(
                    f'URR J-group has NE={ne} but a previous group '
                    f'had NE={ne_ref}; the URRData schema uses a '
                    f'rectangular (nJ, NE) width table and requires '
                    f'a common NE across groups within a range. Pad '
                    f'or split as a follow-up if a real file needs it.'
                )

            group_l.append(L)
            group_j2.append(j2)
            group_amun.append(float(d_j['AMUN']))
            group_amug.append(float(d_j['AMUG']))
            group_amuf.append(float(d_j['AMUF']))
            group_amux.append(float(d_j['AMUX']))
            group_int.append(intp)

            def _as_row(key):
                # endf_parserpy renders these as 1-indexed dicts;
                # dict2array preserves insertion order AND, when
                # xp=jax, keeps any tracer scalars alive so
                # ``jax.grad`` reaches back to the width leaves.
                return dict2array(d_j[key], dtype=float, xp=xp)

            table_es_rows.append(_as_row('ES'))
            table_d_rows.append(_as_row('D'))
            table_gn0_rows.append(_as_row('GN0'))
            table_gg_rows.append(_as_row('GG'))
            table_gf_rows.append(_as_row('GF'))
            table_gx_rows.append(_as_row('GX'))

    if awri_ref is None or ne_ref is None:
        raise ValueError(
            'URR range has no L-groups / J-groups; expected at '
            'least one J-group per L-group per the LRF=2 layout.'
        )

    ki_val = _KN * math.sqrt(awi) * awri_ref / (awri_ref + awi)

    # Radii. Scattering radius: energy-dependent via APE if NRO=1,
    # else the scalar AP. Channel radius: derived from mass unless
    # NAPS=1.
    ape = d_range.get('AP_table') if nro else None
    if ape is not None:
        r_ap = _radius_tab1_from_ape(ape, float(emax))
    else:
        r_ap = _radius_tab1_from_ap(float(ap), float(emax))

    a = _channel_radius(float(ap), awri_ref, naps)
    r_a = _radius_tab1_from_ap(a, float(emax))

    # Statistical weight per group.
    j2_arr = np.asarray(group_j2, dtype=np.int32)
    g_denom = 2.0 * (2.0 * float(spi) + 1.0)
    group_g = (j2_arr.astype(np.float64) + 1.0) / g_denom

    # Width tables may contain JAX tracer scalars from the dict
    # leaves; stack via xp so they survive into the dataclass. All
    # per-J-group scalar arrays (integer counts, degrees of freedom,
    # statistical weight) stay on numpy: they steer bookkeeping.
    return URRData(
        abn=abn,
        spi=spi,
        ap=ap,
        awri=np.asarray(awri_ref, dtype=np.float64),
        ki=np.asarray(ki_val, dtype=np.float64),
        naps=naps,
        group_l=np.asarray(group_l, dtype=np.int32),
        group_j2=j2_arr,
        group_g=group_g,
        group_amun=np.asarray(group_amun, dtype=np.float64),
        group_amug=np.asarray(group_amug, dtype=np.float64),
        group_amuf=np.asarray(group_amuf, dtype=np.float64),
        group_amux=np.asarray(group_amux, dtype=np.float64),
        group_int=np.asarray(group_int, dtype=np.int32),
        table_es=xp.stack(table_es_rows, axis=0),
        table_d=xp.stack(table_d_rows, axis=0),
        table_gn0=xp.stack(table_gn0_rows, axis=0),
        table_gg=xp.stack(table_gg_rows, axis=0),
        table_gf=xp.stack(table_gf_rows, axis=0),
        table_gx=xp.stack(table_gx_rows, axis=0),
        r_a=r_a,
        r_ap=r_ap,
    )
