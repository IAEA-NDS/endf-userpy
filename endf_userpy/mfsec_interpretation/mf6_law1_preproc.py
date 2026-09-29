"""ENDF-6 MF6 LAW=1 (continuum energy-angle distribution)
preprocessing.

Lifts one parsed MF6 LAW=1 subsection dict into a natural-size
:class:`MF6Law1Data` dataclass suitable for consumption by the
backend-agnostic reconstruction kernel in
:mod:`mf6_interpretation_subsecs` (function
``_law1_reconstruct_from_data``). Mirrors the ``MF6Law2Data`` /
``mf6_law2_preproc`` pattern (see :mod:`mf6_law2_preproc`).

**Why a dataclass:** callers that want JAX autodiff wrt file-stored
LAW=1 coefficients (Legendre / Kalbach-Mann / tabulated ``b`` rows)
build the data via :func:`mf6_law1_data_from_endf_dict`, then
substitute a JAX tracer into the ``b_panels`` field of interest
with :func:`dataclasses.replace`, and call the reconstruction with
``xp=array_ns.get_backend('jax')``. Gradients flow end-to-end
through the LAW=1 continuum reconstruction.

Per-panel data (``ep_panels``, ``b_panels``) is padded to the max
shape across the section's panels; the per-panel ``nep`` and ``na``
counts are kept explicit so the kernel knows which slots are real.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..primitives import array_ns
from ..primitives.helpers import dict2array
from ..primitives.properties import (
    get_AWI, get_AWR, get_QI, get_ZA, get_ZAI,
)
from ..primitives.static_dict import StaticEndfDict


# Preproc-cache key namespace for MF6 LAW=1. Kept as a constant so
# other MF sections that later share the ``_preproc_cache`` dict on
# a :class:`StaticEndfDict` can use their own disjoint namespace.
_CACHE_TAG = 'mf6_law1'


@dataclass
class MF6Law1Data:
    """Natural-size input for the MF6 LAW=1 continuum-part
    reconstruction kernel.

    Scalars (physical constants, dispatch flags):
      awi, awr, awp, q, za, zai, zap: physical constants
      lct: reference-frame flag from the MT section (1 = LAB,
           2 = CM, 3 = LAB for AWP>4 else CM)
      lang: angular representation flag (1 = Legendre, 2 = Kalbach,
            11..15 = tabulated)
      lep:  interpolation scheme for the outgoing-energy axis inside
            a panel
    Per-panel arrays:
      ei_mesh (n_panels,)     : incident-energy panel knots
      int_arr, nbt_arr (nR,)  : ENDF interpolation-region descriptors
                                for the outer E axis
      nd_arr, na_arr (n_panels,) : discrete-lines count and angular
                                    parameter count per panel
      nep_arr (n_panels,)     : total (discrete+continuum) Ep count
                                per panel
      ep_panels (n_panels, max_nep) : per-panel Ep values, zero-padded
                                       beyond nep_arr[p]
      b_panels (n_panels, max_nep, max_na_plus_one) : per-panel
                                       angular parameters at each Ep,
                                       zero-padded beyond
                                       (nep_arr[p], na_arr[p]+1)
    """
    awi: Any
    awr: Any
    awp: Any
    q: Any
    za: Any
    zai: Any
    zap: Any
    lct: int
    lang: int
    lep: int
    ei_mesh: np.ndarray
    int_arr: np.ndarray
    nbt_arr: np.ndarray
    nd_arr: np.ndarray
    na_arr: np.ndarray
    nep_arr: np.ndarray
    ep_panels: Any
    b_panels: Any
    # Discrete-line dedup extension (issue #290, Phase 2). Discrete
    # lines with coincident Ep values within a panel are summed into
    # a single (Ep, b) row so downstream can iterate a fixed-shape
    # (max_nd_ded,) axis without doing dedup inside a jit-traced
    # kernel. ``nd_ded_arr[p]`` gives the true dedup'd count per
    # panel; slots beyond it in ``ep_disc_ded`` / ``b_disc_ded``
    # are zero-padded.
    #
    # ``ep_disc_ded``: shape ``(n_panels, max_nd_ded)`` (numpy).
    # ``b_disc_ded``:  shape ``(n_panels, max_nd_ded, max_na + 1)``
    #                  (xp-native, so ``b`` autodiff flows through).
    # ``nd_ded_arr``:  shape ``(n_panels,)`` (numpy int).
    ep_disc_ded: np.ndarray
    b_disc_ded: Any
    nd_ded_arr: np.ndarray


def mf6_law1_data_from_endf_dict(endf_dict, mt: int, subsec_num: int,
                                   xp=None) -> MF6Law1Data:
    """Build an :class:`MF6Law1Data` from a parsed ENDF-6 dict.

    Per-panel ``Ep`` and ``b`` arrays are padded to the max shape
    across the section so the reconstruction kernel can work on
    fixed-shape stacked arrays (needed for JAX). Padding entries
    are zero; the kernel uses per-panel ``nep`` and ``na`` counts
    to mask them out.

    Backend-agnostic: ``xp=None`` (default) is numpy. Passing a
    JAX adapter routes the panel-marshaling through
    :func:`primitives.helpers.dict2array`'s xp-aware code path so
    JAX tracers stored in ``subsec['b']`` values propagate through.

    When ``endf_dict`` is a :class:`StaticEndfDict` (produced via
    :func:`endf_userpy.primitives.static_dict.wrap_endf_dict`), the
    numpy build is cached on the wrapper's instance-attached
    ``_preproc_cache`` dict, keyed by ``(mt, subsec_num)`` under
    the ``'mf6_law1'`` namespace. Adaptive-Simpson integrators that
    walk the same subsection per quadrature point then pay the
    ``dict2array`` / ``pad_nested_ragged_lists`` overhead exactly
    once per (wrapper, MT, subsec). A raw ``dict`` argument gets
    no caching (safe fallback: no id-reuse risk, no stale reads
    from in-place mutation). JAX is not cached either, so tracer
    identities on file leaves stay confined to the call that
    injected them.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    use_cache = xp.name == 'numpy' and isinstance(endf_dict, StaticEndfDict)
    if use_cache:
        cache = endf_dict._preproc_cache
        key = (_CACHE_TAG, int(mt), int(subsec_num))
        cached = cache.get(key)
        if cached is not None:
            return cached
        data = _mf6_law1_data_build(endf_dict, mt, subsec_num, xp)
        cache[key] = data
        return data
    return _mf6_law1_data_build(endf_dict, mt, subsec_num, xp)


def _mf6_law1_data_build(endf_dict, mt: int, subsec_num: int, xp) -> MF6Law1Data:
    """Uncached build. ``mf6_law1_data_from_endf_dict`` is the
    public entry point and adds the numpy-side LRU cache on top."""
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    if subsec['LAW'] != 1:
        raise ValueError(
            f'MT={mt} subsec_num={subsec_num} is LAW={subsec["LAW"]}, '
            'not LAW=1'
        )
    awi = get_AWI(endf_dict)
    awr = get_AWR(endf_dict)
    awp = subsec['AWP']
    q = get_QI(endf_dict, mt)
    za = get_ZA(endf_dict)
    zai = get_ZAI(endf_dict)
    zap = subsec['ZAP']
    lct = int(sec['LCT'])
    lang = int(subsec['LANG'])
    lep = int(subsec['LEP'])
    # Route ``ei_mesh`` through xp so a JAX tracer at
    # ``subsec['E'][idx]`` (mesh-knot autodiff) survives the
    # kernel's default-path panel loop. Under xp=numpy this
    # remains a numpy build. Region-descriptor arrays and
    # per-panel counts stay integer.
    ei_mesh = dict2array(subsec['E'], dtype=float, xp=xp)
    int_arr = np.array(subsec['INT'], dtype=int)
    nbt_arr = np.array(subsec['NBT'], dtype=int)
    nd_arr = dict2array(subsec['ND'], dtype=int)
    na_arr = dict2array(subsec['NA'], dtype=int)

    # Determine padded shapes.
    n_panels = ei_mesh.shape[0]
    ep_lens = []
    for p in range(n_panels):
        ep_lens.append(len(subsec['Ep'][p + 1]))
    nep_arr = np.asarray(ep_lens, dtype=int)
    max_nep = int(nep_arr.max()) if nep_arr.size else 0
    max_na_plus_one = int(na_arr.max()) + 1 if na_arr.size else 1

    # Stack panel data into (n_panels, max_nep) and (n_panels,
    # max_nep, max_na+1). Use numpy for the panel-outer index
    # (Python-side data), and let the leaf `b` values route through
    # `dict2array` with xp so tracers propagate.
    #
    # Padding invariant (issue #166): slots beyond nep_arr[p] in
    # ep_panels replicate the last valid value (ep_panels[p, nep-1])
    # rather than zero. Zero padding would create spurious kinks at
    # 0 that the tracer-panel-index kernel would try to sort into the
    # kink set and produce non-monotonic results. Replicating the
    # last value keeps the panel's Ep row monotonic and gives
    # zero-width subpanels for the padded slots when they collapse
    # in the sort. Safe for the existing single-panel numpy kernel
    # because every slice into ep_panels is bounded by nep_arr[p].
    ep_panels = np.zeros((n_panels, max_nep), dtype=float)
    b_rows = []
    for p in range(n_panels):
        ep_panel = dict2array(subsec['Ep'][p + 1], dtype=float)
        b_panel_full = dict2array(
            subsec['b'][p + 1], dtype=float, fill_value=0.0, xp=xp,
        )
        nep_p = ep_panel.shape[0]
        ep_panels[p, :nep_p] = ep_panel
        if nep_p < max_nep:
            # Replicate the last valid Ep value into the padded slots.
            ep_panels[p, nep_p:] = ep_panel[-1] if nep_p > 0 else 0.0
        # xp-native for the b values: pad to (max_nep, max_na+1).
        rows_this_panel = int(b_panel_full.shape[0])
        cols_this_panel = int(b_panel_full.shape[1])
        pad_rows_before = 0
        pad_rows_after = max_nep - rows_this_panel
        pad_cols_after = max_na_plus_one - cols_this_panel
        if pad_rows_after < 0 or pad_cols_after < 0:
            raise ValueError(
                f'panel {p + 1}: b shape {b_panel_full.shape} exceeds '
                f'padded target ({max_nep}, {max_na_plus_one})'
            )
        b_padded = xp.pad(
            b_panel_full,
            ((pad_rows_before, pad_rows_after),
             (0, pad_cols_after)),
        )
        b_rows.append(b_padded)
    b_panels = xp.stack(b_rows, axis=0)

    # Route ``ei_mesh`` and ``ep_panels`` through the backend so
    # tracer values at those leaves flow through the kernel.
    # Under xp=numpy, ``xp.asarray`` on an already-numpy array is a
    # trivial no-op (measured at ~0.1 us on the two arrays combined
    # for Al-27 MT=91, i.e. 0.01% of the preproc call), so the
    # earlier ``if xp.name != 'numpy':`` guard was not worth the
    # asymmetry. ``nep_arr`` / ``nd_arr`` / ``na_arr`` stay numpy
    # unconditionally: they are integer descriptors that (i) drive
    # Python-side control flow in the reconstruction fallback path,
    # and (ii) still work as-is in the tracer-panel-index kernel,
    # since ``xp.take(numpy_arr, tracer_index)`` promotes the numpy
    # array to xp-native on-the-fly. Keeping them numpy means
    # ``jax.jit(loss)`` and ``jax.jit(jax.grad(loss))``, where the
    # whole preproc runs inside a JAX trace, do not turn integer
    # counts into tracers whose ``int(...)`` would raise.
    # Keep a numpy view of ep_panels for the dedup pass below; the
    # xp.asarray promotion turns it into a tracer under jit, which
    # then can't be materialised for the Python-side dedup lookup.
    ep_panels_np = ep_panels
    ei_mesh = xp.asarray(ei_mesh)
    ep_panels = xp.asarray(ep_panels)

    # Discrete-line dedup pass (issue #290, Phase 2). For each panel
    # compute the (ep, b) rows after summing coincident-Ep discrete
    # slots. Padded to a common ``max_nd_ded`` across panels so the
    # downstream traced kernel can index a fixed-shape axis. Uses
    # the sibling implementation from ``mf6_interpretation_subsecs``
    # to keep dedup semantics in one place.
    #
    # Dedup requires materialising the ``b`` rows to numpy so it
    # can group by ``ep`` value. If the caller injected a jax tracer
    # into ``subsec['b']`` for file-side autodiff (or the whole
    # preproc runs inside a jit trace), materialisation would raise;
    # in that case skip dedup and expose the raw padded ``(nd_max,)``-
    # wide arrays instead. The traced downstream kernel handles both
    # shapes uniformly, and files with genuine coincident discrete
    # lines are rare (verified: U-233 (n,g) nd_ded == nd across all
    # 118 panels).
    from ..mfsec_interpretation.mf6_law1_kernel import _is_jax_tracer
    from . import mf6_interpretation_subsecs as _subsecs
    max_nd = int(nd_arr.max()) if nd_arr.size else 0
    if xp.name == 'jax' and _is_jax_tracer(b_panels):
        # Skip dedup; expose the raw first ``nd_arr[p]`` slots of
        # each panel through the same interface. ``ep_panels_np`` is
        # the pre-tracer numpy view captured above.
        ep_disc_ded = np.zeros((n_panels, max_nd), dtype=float)
        for p in range(n_panels):
            nd_p = int(nd_arr[p])
            if nd_p > 0:
                ep_disc_ded[p, :nd_p] = ep_panels_np[p, :nd_p]
        # b_disc_ded slices ``b_panels`` xp-native so tracer identity
        # is preserved. Take first ``max_nd`` rows; per-panel valid
        # count is ``nd_arr[p]`` (== nd_ded_arr[p] in this branch).
        b_disc_ded = b_panels[:, :max_nd, :]
        nd_ded_arr = nd_arr.copy()
    else:
        per_panel_ded = []
        for p in range(n_panels):
            ep_panel_np = ep_panels_np[p]
            b_panel_np = np.asarray(b_panels[p])
            nd_p = int(nd_arr[p])
            ep_ded, b_ded, nd_ded = _subsecs._dedup_discrete_lines(
                ep_panel_np, b_panel_np, nd_p,
            )
            per_panel_ded.append((ep_ded, b_ded, nd_ded))
        nd_ded_arr = np.asarray([d[2] for d in per_panel_ded], dtype=int)
        max_nd_ded = int(nd_ded_arr.max()) if nd_ded_arr.size else 0
        ep_disc_ded = np.zeros((n_panels, max_nd_ded), dtype=float)
        b_disc_ded_np = np.zeros(
            (n_panels, max_nd_ded, max_na_plus_one), dtype=float,
        )
        for p, (ep_ded, b_ded, nd_ded) in enumerate(per_panel_ded):
            if nd_ded == 0:
                continue
            ep_disc_ded[p, :nd_ded] = ep_ded
            b_disc_ded_np[p, :nd_ded, :] = b_ded
        b_disc_ded = xp.asarray(b_disc_ded_np)

    return MF6Law1Data(
        awi=awi, awr=awr, awp=awp, q=q,
        za=za, zai=zai, zap=zap,
        lct=lct, lang=lang, lep=lep,
        ei_mesh=ei_mesh, int_arr=int_arr, nbt_arr=nbt_arr,
        nd_arr=nd_arr, na_arr=na_arr, nep_arr=nep_arr,
        ep_panels=ep_panels, b_panels=b_panels,
        ep_disc_ded=ep_disc_ded, b_disc_ded=b_disc_ded,
        nd_ded_arr=nd_ded_arr,
    )
