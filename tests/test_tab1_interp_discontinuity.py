"""Tests for `primitives.tab1.interp` endpoint-side selection at
doubled-x discontinuities (issue #136).

ENDF-6 encodes a step discontinuity in a TAB1 by placing two
adjacent x-values equal with different y-values. A query at
exactly that shared x is ambiguous: the left-limit is the
y before the step, the right-limit is the y after. NJOY reconr's
own TAB1 lookups use the right-limit; endf-userpy's
`interpolation.endf_interp1d` already matches that via
`find_interval(side='right')`, but `primitives.tab1.interp` used
to default to the left-limit, giving a silent discrepancy between
the two paths whenever a caller sat right on a discontinuity.

This test file pins the new `interp(..., side='right' | 'left')`
parameter and confirms:

- Default (`side='right'`) matches NJOY: right-limit at a doubled x.
- `side='left'` returns the value before the discontinuity.
- Away from the duplicate, both settings agree.
- Both settings agree with `endf_interp1d` at the same query
  when the appropriate `side` is picked (agnostic-path parity
  with the non-agnostic path).
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import tab1 as tab1_mod
from endf_userpy.primitives import interpolation as interp_mod
from endf_userpy.primitives import array_ns


def _step_tab1():
    """TAB1 with a step discontinuity: y drops from 5.0 to 1.0
    at exactly x=100. Values around it exercise both the
    left panel [50, 100] and the right panel [100, 200]."""
    x = np.array([0.0, 50.0, 100.0, 100.0, 200.0, 300.0], dtype=np.float64)
    y = np.array([10.0, 5.0, 5.0, 1.0, 1.0, 0.5], dtype=np.float64)
    nbt = np.array([len(x) - 1], dtype=np.int32)
    intp = np.array([2], dtype=np.int32)   # lin-lin
    return tab1_mod.TAB1(x=x, y=y, nbt=nbt, intp=intp)


def test_default_side_returns_right_limit_at_discontinuity():
    """Default `side='right'` picks the right-limit at exactly
    the doubled-x point. Matches NJOY reconr's convention (via
    find_interval side='right')."""
    tab1 = _step_tab1()
    xp = array_ns.get_backend('numpy')
    x_query = np.array([99.9, 100.0, 100.1])
    y = np.asarray(tab1_mod.interp(tab1, x_query, xp))
    # Left panel [50, 100] is lin-lin flat at y=5 -> just below the
    # step returns 5. Right panel [100, 200] is lin-lin flat at
    # y=1 -> just above returns 1. At exactly x=100 the default
    # 'right' picks the right-limit = 1.0.
    np.testing.assert_allclose(y, [5.0, 1.0, 1.0], rtol=1e-12)


def test_side_left_returns_left_limit_at_discontinuity():
    """Explicit `side='left'` picks the value from the panel
    BEFORE the doubled-x point."""
    tab1 = _step_tab1()
    xp = array_ns.get_backend('numpy')
    x_query = np.array([99.9, 100.0, 100.1])
    y = np.asarray(tab1_mod.interp(tab1, x_query, xp, side='left'))
    # At x=100 the left panel's right endpoint gives y=5 (the value
    # before the step).
    np.testing.assert_allclose(y, [5.0, 5.0, 1.0], rtol=1e-12)


def test_side_only_matters_at_doubled_x():
    """Away from the doubled-x point, `side='right'` and
    `side='left'` produce identical numbers on every query."""
    tab1 = _step_tab1()
    xp = array_ns.get_backend('numpy')
    x_query = np.array([25.0, 50.0, 75.0, 150.0, 200.0, 250.0])
    y_right = np.asarray(tab1_mod.interp(tab1, x_query, xp, side='right'))
    y_left = np.asarray(tab1_mod.interp(tab1, x_query, xp, side='left'))
    np.testing.assert_allclose(y_right, y_left, rtol=1e-12)


def test_default_agrees_with_endf_interp1d_at_discontinuity():
    """Cross-path parity: the default `tab1.interp` (agnostic
    path) and `endf_interp1d` (non-agnostic path) return the
    same value at a doubled x. Before this fix the two paths
    disagreed by the full step size at exactly the discontinuity."""
    tab1 = _step_tab1()
    xp = array_ns.get_backend('numpy')
    x_query = np.array([99.9, 100.0, 100.1])
    y_agnostic = np.asarray(tab1_mod.interp(tab1, x_query, xp))
    # Rebuild the tab1 into the endf_interp1d input shape.
    xp_np = tab1.x
    fp_np = tab1.y
    int_arr = tab1.intp
    nbt_arr = tab1.nbt + 1   # endf_interp1d expects 1-indexed NBT
    y_endf1d = np.asarray(interp_mod.endf_interp1d(
        x_query, xp_np, fp_np, int_arr, nbt_arr,
    ))
    np.testing.assert_allclose(y_agnostic, y_endf1d, rtol=1e-12)


def test_invalid_side_raises():
    """`side` must be either 'right' or 'left'; anything else
    fails loudly at call time so a typo doesn't silently pick a
    convention."""
    tab1 = _step_tab1()
    xp = array_ns.get_backend('numpy')
    with pytest.raises(ValueError, match="side must be 'right' or 'left'"):
        tab1_mod.interp(tab1, np.array([100.0]), xp, side='middle')


def test_end_of_mesh_duplicated_x_returns_final_y_for_right_side():
    """Edge case: the last two mesh points are equal (a step at
    the end of the tabulation). `side='right'` should give the
    final y value; `side='left'` should give the value before the
    step. The panel is degenerate (x1 == x2) so both routes go
    through the constant collapse; the fix picks y2 for 'right'
    and y1 for 'left'."""
    x = np.array([0.0, 50.0, 100.0, 100.0], dtype=np.float64)
    y = np.array([10.0, 5.0, 5.0, 1.0], dtype=np.float64)
    tab1 = tab1_mod.TAB1(
        x=x, y=y,
        nbt=np.array([3], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )
    xp = array_ns.get_backend('numpy')
    y_right = float(np.asarray(tab1_mod.interp(tab1, np.array([100.0]), xp))[0])
    y_left = float(np.asarray(
        tab1_mod.interp(tab1, np.array([100.0]), xp, side='left')
    )[0])
    assert y_right == pytest.approx(1.0)
    assert y_left == pytest.approx(5.0)
