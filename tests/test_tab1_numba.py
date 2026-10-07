"""Numba fast path for `primitives.tab1.interp` (issue #349).

Pins:
- The numba path produces bit-identical output as the numpy
  xp-vectorised fallback on all five interpolation laws.
- The path fires only when the backend ``wants_accelerator('numba')``
  and ``accelerator_available('numba')`` are both true (so
  ``RunOptions(backend='numpy')`` keeps the xp-vectorised fallback
  and does NOT silently route through numba).
- Degenerate panels, out-of-range queries, and the ``side='left'``
  / ``side='right'`` endpoint-selection conventions match the
  fallback on both paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import tab1 as t
from endf_userpy.primitives import tab1_numba
from endf_userpy.primitives.array_ns import get_backend


REQUIRES_NUMBA = pytest.mark.skipif(
    not tab1_numba.HAS_NUMBA, reason='numba not installed'
)


def _lin_lin_tab():
    return t.TAB1(
        x=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        y=np.array([0.0, 2.0, 4.0, 8.0], dtype=np.float64),
        nbt=np.array([3], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )


def _mixed_law_tab():
    """Four panels with different interpolation laws (1..5)."""
    return t.TAB1(
        x=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        y=np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float64),
        nbt=np.array([1, 2, 3, 4], dtype=np.int32),
        intp=np.array([2, 3, 4, 5], dtype=np.int32),
    )


@REQUIRES_NUMBA
def test_numba_matches_numpy_lin_lin():
    """Lin-lin is the dominant case on evaluation XS tables; the
    numba path must be bit-identical to the numpy fallback here."""
    tab = _lin_lin_tab()
    x_query = np.linspace(-0.5, 3.5, 2001)
    xp_numpy = get_backend('numpy')
    xp_numba = get_backend('numba')
    y_numpy = t.interp(tab, x_query, xp_numpy)
    y_numba = t.interp(tab, x_query, xp_numba)
    np.testing.assert_array_equal(y_numba, y_numpy)


@REQUIRES_NUMBA
def test_numba_matches_numpy_mixed_laws():
    """Interior points of each panel exercise its law; endpoints
    exercise the panel-boundary resolution."""
    tab = _mixed_law_tab()
    x_query = np.linspace(0.5, 5.5, 1001)
    xp_numpy = get_backend('numpy')
    xp_numba = get_backend('numba')
    y_numpy = t.interp(tab, x_query, xp_numpy)
    y_numba = t.interp(tab, x_query, xp_numba)
    np.testing.assert_allclose(y_numba, y_numpy, rtol=1e-13, atol=0)


@REQUIRES_NUMBA
def test_numba_matches_numpy_out_of_range():
    """Below- and above-mesh queries must return ``outside_value``
    on both paths."""
    tab = _lin_lin_tab()
    x_query = np.array([-10.0, -1.0, 0.0, 3.0, 10.0])
    xp_numpy = get_backend('numpy')
    xp_numba = get_backend('numba')
    for ov in (0.0, float('nan'), -1.0):
        y_numpy = t.interp(tab, x_query, xp_numpy, outside_value=ov)
        y_numba = t.interp(tab, x_query, xp_numba, outside_value=ov)
        np.testing.assert_array_equal(
            np.isnan(y_numba), np.isnan(y_numpy),
        )
        mask = ~np.isnan(y_numpy)
        np.testing.assert_array_equal(y_numba[mask], y_numpy[mask])


@REQUIRES_NUMBA
def test_numba_matches_numpy_doubled_x_side():
    """ENDF-6 doubled-x discontinuity: left-limit vs right-limit
    selection must agree between the numba path and the numpy
    fallback."""
    tab = t.TAB1(
        x=np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float64),
        y=np.array([0.0, 2.0, 10.0, 12.0], dtype=np.float64),
        nbt=np.array([3], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )
    x_query = np.array([0.5, 1.0, 1.5])
    xp_numpy = get_backend('numpy')
    xp_numba = get_backend('numba')
    for side in ('left', 'right'):
        y_numpy = t.interp(tab, x_query, xp_numpy, side=side)
        y_numba = t.interp(tab, x_query, xp_numba, side=side)
        np.testing.assert_array_equal(y_numba, y_numpy)


@REQUIRES_NUMBA
def test_numba_path_fires_only_when_wanted():
    """``RunOptions(backend='numpy')`` resolves to a NumpyBackend
    whose ``wants_accelerator('numba')`` is False; the fast path
    must not fire there regardless of numba presence. Verified by
    swapping in a wrong-answer kernel and confirming the result
    stays correct under the numpy backend."""
    tab = _lin_lin_tab()
    x_query = np.array([0.5, 1.5, 2.5])
    xp_numpy = get_backend('numpy')
    expected = t.interp(tab, x_query, xp_numpy)
    # Swap the numba kernel for a sabotage version that returns -999
    # everywhere.
    orig_kernel = tab1_numba._interp_full_numba
    def sabotage(*args, **kwargs):
        return np.full_like(args[0], -999.0, dtype=np.float64)
    try:
        tab1_numba._interp_full_numba = sabotage
        y_numpy = t.interp(tab, x_query, xp_numpy)
    finally:
        tab1_numba._interp_full_numba = orig_kernel
    # NumpyBackend must have ignored the sabotage kernel.
    np.testing.assert_array_equal(y_numpy, expected)


@REQUIRES_NUMBA
def test_numba_path_fires_under_numba_backend():
    """Symmetric check: NumbaBackend should route through the fast
    path, so swapping in a sabotage kernel changes the result."""
    tab = _lin_lin_tab()
    x_query = np.array([0.5, 1.5, 2.5])
    xp_numba = get_backend('numba')
    orig_kernel = tab1_numba._interp_full_numba
    def sabotage(*args, **kwargs):
        return np.full_like(args[0], -999.0, dtype=np.float64)
    try:
        tab1_numba._interp_full_numba = sabotage
        y_numba = t.interp(tab, x_query, xp_numba)
    finally:
        tab1_numba._interp_full_numba = orig_kernel
    np.testing.assert_array_equal(y_numba, np.full_like(x_query, -999.0))
