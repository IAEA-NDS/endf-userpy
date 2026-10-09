"""Numba per-point kernel for :func:`interpolation.endf_interp1d`.

Companion to :mod:`endf_userpy.primitives.interpolation` in the same
way :mod:`tab1_numba` is to :mod:`tab1`. ``endf_interp1d`` on the numpy
path splits the query into one array per interpolation region,
interpolates each with its own panel lookup, and restores the original
order with an ``argsort`` over all query points; per MF6 yield table
on a 1M-point mesh that is several full-mesh passes plus a sort. When
the backend asks for numba, ``endf_interp1d`` instead hands the
(deduplicated) mesh and a per-panel law table to
:func:`_endf_interp1d_kernel`, which does the lookup and the
arithmetic per point under ``prange``.

Semantics are those of the numpy path, not of :mod:`tab1_numba`:

- the mesh is the ``treat_duplicates`` output (doubled x-values moved
  apart by a relative 1e-8), and the panel of ``x`` is
  :func:`helpers.find_interval`'s: ``searchsorted(mesh, x, 'right') - 1``
  with ``x == mesh[-1]`` (and NaN) in the last panel;
- each panel uses the INT code of the region the numpy region loop
  assigns it to (a region's last row belongs to the next region);
- the scheme formulas, including the ``1e-38`` clamps of ``x1 == 0`` /
  ``y1 == 0`` before taking logs, are the ``interp_*`` ones term by
  term; no ``fastmath``, so NaN queries propagate as in numpy.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        if args and callable(args[0]):
            return args[0]

        def decorator(fn):
            return fn

        return decorator

    prange = range  # type: ignore[misc, assignment]


_INTERP_LOG_SMALL = 1.0e-38


def panel_laws(int_arr, nbt_arr, n_mesh):
    """Per-panel INT codes of a TAB1 with ``n_mesh`` points, assigned
    the way :func:`interpolation.endf_interp1d`'s region loop assigns
    query points (region ``i`` covers panels
    ``[NBT[i-1] - 1, NBT[i] - 1)``, the last region up to ``NBT[-1]``).

    Returns ``None`` when the regions do not cover every panel exactly
    once or use an INT code outside 1..5; callers then keep the numpy
    path, which handles (or reports) those cases itself.
    """
    n_panels = n_mesh - 1
    laws = np.zeros(max(n_panels, 0), dtype=np.int64)
    cover = np.zeros(max(n_panels, 0), dtype=np.int64)
    first_idx = 0
    nregions = len(int_arr)
    for i in range(nregions):
        last_idx = int(nbt_arr[i])
        upper = last_idx if i == nregions - 1 else last_idx - 1
        lo = max(first_idx, 0)
        hi = min(upper, n_panels)
        if hi > lo:
            law = int(int_arr[i])
            if law < 1 or law > 5:
                return None
            laws[lo:hi] = law
            cover[lo:hi] += 1
        first_idx = last_idx - 1
    if n_panels < 1 or not np.all(cover == 1):
        return None
    return laws


@njit(cache=True, parallel=True, error_model='numpy')
def _endf_interp1d_kernel(x, mesh, fp, laws, outside_value):
    """Interpolate ``fp`` on the deduplicated ``mesh`` at every ``x``.

    Points outside ``[mesh[0], mesh[-1]]`` get ``outside_value`` and
    are counted; the second return value is that count (the caller
    raises when no ``outside_value`` was given).
    """
    n = x.shape[0]
    n_mesh = mesh.shape[0]
    lo_x = mesh[0]
    hi_x = mesh[n_mesh - 1]
    y = np.empty(n, dtype=np.float64)
    outside = np.zeros(n, dtype=np.int64)
    for p in prange(n):
        xq = x[p]
        if xq < lo_x or xq > hi_x:
            y[p] = outside_value
            outside[p] = 1
            continue
        # searchsorted(mesh, xq, 'right'): first index with mesh > xq.
        # NaN sorts last in numpy, i.e. lands at n_mesh like x == mesh[-1]:
        # both use the last panel.
        if xq != xq:
            lo = n_mesh
        else:
            lo = 0
            hi = n_mesh
            while lo < hi:
                mid = (lo + hi) // 2
                if mesh[mid] <= xq:
                    lo = mid + 1
                else:
                    hi = mid
        i = lo - 1
        if lo == n_mesh:
            i = n_mesh - 2
        x1 = mesh[i]
        x2 = mesh[i + 1]
        y1 = fp[i]
        y2 = fp[i + 1]
        law = laws[i]
        if law == 1:
            y[p] = y1
        elif law == 2:
            y[p] = y1 + (xq - x1) * (y2 - y1) / (x2 - x1)
        elif law == 3:
            if x1 == 0.0:
                x1 = _INTERP_LOG_SMALL
            y[p] = y1 + math.log(xq / x1) * (y2 - y1) / math.log(x2 / x1)
        elif law == 4:
            if y1 == 0.0:
                y1 = _INTERP_LOG_SMALL
            y[p] = y1 * math.exp((xq - x1) * math.log(y2 / y1) / (x2 - x1))
        else:
            if x1 == 0.0:
                x1 = _INTERP_LOG_SMALL
            if y1 == 0.0:
                y1 = _INTERP_LOG_SMALL
            y[p] = y1 * math.exp(
                math.log(xq / x1) * math.log(y2 / y1) / math.log(x2 / x1)
            )
    return y, outside.sum()
