"""ENDF-6 MF6 LAW=2 (discrete two-body angular distribution)
preprocessing.

Lifts a parsed ENDF-6 subsection dict into a natural-size
:class:`MF6Law2Data` dataclass suitable for consumption by the
backend-agnostic reconstruction kernel in
:mod:`mf6_interpretation_subsecs` (function
``_law2_reconstruct_from_data``). Mirrors the split already used
for MF2 (see :mod:`mf2_interpretation_mlbw_preproc`), which the
issue #47 easy-tier port sequence adopted as the standard shape
for backend-agnostic reconstructions.

**Why a dataclass:** callers that want JAX autodiff wrt file-stored
Legendre coefficients (or tabulated ``f`` values) build the data
via :func:`mf6_law2_data_from_endf_dict`, then substitute a JAX
tracer into the dataclass field of interest with
:func:`dataclasses.replace`, and call the reconstruction with
``xp=array_ns.get_backend('jax')``. Gradients flow end-to-end
through the reconstruction. Issue #154 describes the broader
context.

Callers that only want to evaluate the distribution on a numpy
backend continue to use the dict-facing entry point
``get_angdist_from_subsec_law2(endf_dict, ...)`` which composes
this preproc with the kernel internally and is bit-identical to
the pre-refactor implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from ..primitives import array_ns
from ..primitives.helpers import dict2array
from ..primitives.properties import get_AWI, get_AWR, get_QI


@dataclass
class MF6Law2Data:
    """Natural-size input for the MF6 LAW=2 angular-distribution
    reconstruction kernel.

    Fields:

    - ``awi, awr, awp, q``: physical constants for the reaction
      (target mass ratio, incident particle, outgoing particle,
      Q value from MF3). Scalars; can be replaced with 0-d JAX
      arrays to enable autodiff wrt any of them.
    - ``lct``: original reference system flag (1 = LAB, 2 = CM,
      3 = LAB for AWP>4 else CM).
    - ``lang``: representation flag. 0 = Legendre (``coeffs``
      populated); 12 = tabulated INT=2 lin-lin in mu; 14 =
      tabulated INT=4 log-lin in mu (``records`` populated).
    - ``zap``: ejectile ZAP (Z*1000 + A, or 0 for photons). The
      dict-facing entry point short-circuits to zero on photons
      pending the massless-ejectile kinematics story (issue #78);
      the kernel itself does not enforce this.
    - ``ei_mesh``: (n_panels,) incident-energy mesh knots.
    - ``int_arr, nbt_arr``: ENDF interpolation-region descriptors
      for the outer (incident-energy) axis.
    - ``coeffs``: (n_panels, max_L+1) Legendre coefficient array
      with ``a_0 = 1`` prepended AND the ``(L + 0.5)`` ENDF
      normalisation factor already applied, so
      :func:`evaluate_interp_legendre_polynomials` evaluates
      ``sum_L coeffs[..., L] P_L(mu)`` directly. Populated only
      when ``lang == 0``.
    - ``records``: list of TAB1-shaped dicts
      ``{'mu', 'f', 'INT', 'NBT'}`` per panel. Populated only
      when ``lang in (12, 14)``.
    """
    awi: Any
    awr: Any
    awp: Any
    q: Any
    lct: int
    lang: int
    zap: float
    ei_mesh: np.ndarray
    int_arr: np.ndarray
    nbt_arr: np.ndarray
    coeffs: Optional[Any] = None
    records: Optional[List[dict]] = field(default=None)


def _law2_legendre_coeffs_array(subsec, xp=None):
    """Build the full ``(n_panels, max_L+1)`` Legendre coefficient
    array from ``subsec['A']``.

    ENDF-6 stores per-panel coefficients ``a_1, a_2, ..., a_{NL}``
    (``a_0 = 1`` implied by normalisation, not written). We prepend
    ``a_0 = 1`` and apply the ``(L + 0.5)`` factor per ENDF
    convention (``f(mu) = sum_L (L+0.5) a_L P_L(mu)``) so the
    caller can feed the result directly into
    :func:`evaluate_interp_legendre_polynomials`. Ragged rows are
    zero-padded to the max NL (higher-degree zeros contribute
    nothing to the Legendre sum).

    Backend-agnostic: ``xp=None`` (default) is numpy. Passing a
    JAX adapter routes ``dict2array`` and the a_0 prepend +
    (L + 0.5) scaling through xp so JAX tracers stored at any
    dict leaf of ``subsec['A']`` propagate through (issue #154).
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    a_arr = dict2array(
        subsec['A'], dtype=float, order='C', fill_value=0.0, xp=xp,
    )
    n_panels = a_arr.shape[0]
    # Prepend the a_0 = 1 column functionally so JAX-traced a_arr
    # doesn't need in-place assignment. Ragged panels are already
    # zero-padded by dict2array's fill_value, so higher-degree
    # zeros beyond each panel's NL contribute nothing.
    ones_col = xp.ones((n_panels, 1), dtype=xp.float64)
    coeffs = xp.concatenate([ones_col, a_arr], axis=-1)
    L_indices = xp.arange(coeffs.shape[1]).reshape(1, -1)
    return coeffs * (L_indices + 0.5)


def _law2_tab1_records(subsec, lang: int, xp=None) -> List[dict]:
    """Build a list of TAB1-shaped record dicts from LANG=12/14
    tabulated data in ``subsec['A']``.

    ENDF-6 stores each panel as ``[u_1, p_1, u_2, p_2, ...]``.
    We unpack to separate ``mu`` and ``f`` arrays and mark the
    per-panel INT (``lang - 10``: LANG=12 -> lin-lin INT=2,
    LANG=14 -> log-lin INT=4). Format matches
    :func:`interp_tab2`.

    Backend-agnostic: ``xp=None`` (default) is numpy. Passing a
    JAX adapter routes ``dict2array`` through xp so tracers stored
    in ``subsec['A']`` propagate to the returned records.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    a_arr = dict2array(
        subsec['A'], dtype=float, order='C', fill_value=0.0, xp=xp,
    )
    nl_arr = np.array(list(subsec['NL'].values()), dtype=int)
    inner_int = lang - 10
    records = []
    for p in range(a_arr.shape[0]):
        nl = int(nl_arr[p])
        pairs = a_arr[p, :2 * nl]
        mu_p = pairs[0::2]
        f_p = pairs[1::2]
        records.append({
            'mu': mu_p, 'f': f_p,
            'INT': np.array([inner_int], dtype=int),
            'NBT': np.array([nl], dtype=int),
        })
    return records


def mf6_law2_data_from_endf_dict(endf_dict, mt: int, subsec_num: int,
                                   xp=None) -> MF6Law2Data:
    """Build an :class:`MF6Law2Data` from a parsed ENDF-6 dict.

    Callers who want JAX autodiff wrt tabulated coefficients have
    two entry paths:

    1. **Dataclass-first (recommended):** call with ``xp=None``,
       then replace the appropriate field of the returned dataclass
       with a JAX tracer via :func:`dataclasses.replace` before
       calling the reconstruction kernel. See
       ``tests/test_mf6_law2_autodiff_from_coeffs.py``.
    2. **Dict-first:** store JAX tracers in
       ``endf_dict[6][mt]['subsection'][sn]['A']`` values and pass
       ``xp=array_ns.get_backend('jax')`` so the internal
       ``dict2array`` preserves them.
    """
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    awi = get_AWI(endf_dict)
    awr = get_AWR(endf_dict)
    awp = subsec['AWP']
    q = get_QI(endf_dict, mt)
    lct = int(sec['LCT'])
    lang = int(subsec['LANG'])
    zap = float(subsec.get('ZAP', 0.0))
    ei_mesh = dict2array(subsec['E'], dtype=float)
    int_arr = np.array(subsec['INT'], dtype=int)
    nbt_arr = np.array(subsec['NBT'], dtype=int)
    coeffs = None
    records = None
    if lang == 0:
        coeffs = _law2_legendre_coeffs_array(subsec, xp=xp)
    elif lang in (12, 14):
        records = _law2_tab1_records(subsec, lang, xp=xp)
    else:
        raise NotImplementedError(
            f'MF6 LAW=2 LANG={lang} not supported (only 0, 12, 14).'
        )
    return MF6Law2Data(
        awi=awi, awr=awr, awp=awp, q=q,
        lct=lct, lang=lang, zap=zap,
        ei_mesh=ei_mesh, int_arr=int_arr, nbt_arr=nbt_arr,
        coeffs=coeffs, records=records,
    )
