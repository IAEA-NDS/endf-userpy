"""Numba-accelerated per-point TAB1 interpolation (issue #349).

Companion to :mod:`endf_userpy.primitives.tab1`'s backend-agnostic
``interp`` entry point. When the caller runs on the numpy backend
AND numba is importable, ``tab1.interp`` routes through
:func:`_apply_law_numba` here; otherwise it falls back to the
pre-#349 ``xp``-vectorised path (which is what the JAX and
pure-numpy-no-numba configurations need anyway).

Design: the binary-search / segment-lookup work (which is already
cheap under numpy's C ``searchsorted``) stays on the numpy side.
This kernel receives per-query arrays of ``(x, x1, x2, y1, y2,
law)`` and runs a per-point law dispatch under
``@njit(cache=True, parallel=True)`` ``prange``. The win on wide
dense Ein meshes with mixed interpolation laws:

1. **Per-point dispatch skips dead branches.** The pre-#349
   ``_apply_law_vectorised`` computed all five candidate laws
   (``lin_lin``, ``lin_log``, ``log_lin``, ``log_log``, ``const``)
   on every query point and used ``xp.select`` to pick. So a
   point that lands in a lin-lin segment still paid for four
   wasted ``log`` and ``exp`` evaluations per branch. The numba
   kernel branches per point and only evaluates the law that
   applies. ``log`` and ``exp`` dominate numpy wall time on
   10 M-point calls because they are not SIMD-friendly, so this
   is the main saving.

2. **``prange`` over query points scales to all cores.** The
   per-point dispatch is embarrassingly parallel.

The numpy searchsorted is kept because numpy ships a C
implementation that does one SIMD pass over the sorted ``tab_x``
and that is cheaper than numba's inline binary search per point.
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


@njit(cache=True, parallel=True, fastmath=True)
def _interp_full_numba(x, tab_x, tab_y, tab_nbt, tab_intp,
                       outside_value, side_is_left):
    """End-to-end TAB1 interpolation under ``prange`` parallelism.

    All per-point work (panel binary search, segment lookup, law
    dispatch, arithmetic) lives inside the ``prange`` loop so numba
    can schedule each query point to a worker thread. This is the
    full-parallelism path invoked when the numpy preprocessing
    (searchsorted + indexing + where) would otherwise be the
    serial bottleneck on wide dense Ein meshes (issue #349).

    Correctness matches :func:`endf_userpy.primitives.tab1.interp`'s
    numpy-vectorised fallback bit-for-bit on all five standard laws
    (``INT`` in 1..5) and on degenerate-panel / out-of-range queries;
    ``tests/test_tab1_interp_discontinuity.py`` pins both doubled-x
    side conventions.
    """
    n = x.shape[0]
    np_tab = tab_x.shape[0]
    nr = tab_nbt.shape[0]
    y = np.empty(n, dtype=np.float64)
    for p in prange(n):
        xq = x[p]
        # Binary search on tab_x. side='right' / 'left' control
        # endpoint selection at doubled-x discontinuities.
        if side_is_left:
            lo = 0
            hi = np_tab
            while lo < hi:
                mid = (lo + hi) // 2
                if tab_x[mid] < xq:
                    lo = mid + 1
                else:
                    hi = mid
            i = lo
        else:
            lo = 0
            hi = np_tab
            while lo < hi:
                mid = (lo + hi) // 2
                if tab_x[mid] <= xq:
                    lo = mid + 1
                else:
                    hi = mid
            i = lo
        if i < 1:
            i = 1
        elif i > np_tab - 1:
            i = np_tab - 1
        x1 = tab_x[i - 1]
        x2 = tab_x[i]
        y1 = tab_y[i - 1]
        y2 = tab_y[i]
        if xq < x1 or xq > x2:
            y[p] = outside_value
            continue
        if x1 == x2 or y1 == y2:
            y[p] = y1 if side_is_left else y2
            continue
        # Resolve segment and law.
        klo = 0
        khi = nr
        while klo < khi:
            mid = (klo + khi) // 2
            if tab_nbt[mid] < i:
                klo = mid + 1
            else:
                khi = mid
        k = klo
        if k >= nr:
            k = nr - 1
        law = tab_intp[k] % 10
        if law == 1:
            y[p] = y1
        elif law == 2:
            y[p] = y1 + (xq - x1) * (y2 - y1) / (x2 - x1)
        elif law == 3:
            y[p] = y1 + math.log(xq / x1) * (y2 - y1) / math.log(x2 / x1)
        elif law == 4:
            y[p] = y1 * math.exp(
                (xq - x1) * math.log(y2 / y1) / (x2 - x1),
            )
        elif law == 5:
            y[p] = y1 * math.exp(
                math.log(xq / x1) * math.log(y2 / y1)
                / math.log(x2 / x1),
            )
        else:
            y[p] = outside_value
    return y


@njit(cache=True, parallel=True, fastmath=True)
def _apply_law_numba(law, x, x1, x2, y1, y2, outside_value):
    """Per-point ENDF-6 TAB1 law dispatch.

    All inputs are 1-D float64 / int32 arrays of the same length
    (one entry per query point). Caller is responsible for the
    panel lookup (``searchsorted`` on ``tab_x`` and ``tab_nbt``)
    and for encoding degenerate panels / out-of-range queries into
    the ``law`` array (``0`` -> outside, ``1`` -> constant ``y1``,
    ``6`` -> constant ``y2`` for ``side='right'`` doubled-x).

    Returns a (n,) float64 array of interpolated values.
    """
    n = x.shape[0]
    y = np.empty(n, dtype=np.float64)
    for p in prange(n):
        code = law[p]
        if code == 0:
            y[p] = outside_value
        elif code == 1:
            y[p] = y1[p]
        elif code == 6:
            y[p] = y2[p]
        elif code == 2:
            dx = x2[p] - x1[p]
            y[p] = y1[p] + (x[p] - x1[p]) * (y2[p] - y1[p]) / dx
        elif code == 3:
            y[p] = y1[p] + (
                math.log(x[p] / x1[p]) * (y2[p] - y1[p])
                / math.log(x2[p] / x1[p])
            )
        elif code == 4:
            dx = x2[p] - x1[p]
            y[p] = y1[p] * math.exp(
                (x[p] - x1[p]) * math.log(y2[p] / y1[p]) / dx,
            )
        elif code == 5:
            y[p] = y1[p] * math.exp(
                math.log(x[p] / x1[p]) * math.log(y2[p] / y1[p])
                / math.log(x2[p] / x1[p]),
            )
        else:
            y[p] = outside_value
    return y
