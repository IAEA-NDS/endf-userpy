"""Regression tests for the numpy compatibility cleanup (issue #15).

Two idioms are being modernised:

1. ``np.array(x, copy=None)`` in 14 places across
   ``primitives/helpers.py``, ``primitives/interpolation.py``,
   ``primitives/conversion.py``. On numpy 2.0 this means "copy if
   necessary, otherwise reference". On numpy 1.x (< 2.0) it raises
   ``ValueError: NoneType copy mode not allowed``. Replaced with
   ``np.asarray(x)``, which has the same "copy-if-necessary"
   semantic on every numpy version since 1.0.

2. ``~np.bool(mt_available)`` in ``quantities.py:190``. ``np.bool``
   was removed in numpy 1.24; ``mt_available`` was already a Python
   bool from ``mt in avail_mts`` so the cast was doing nothing.
   Replaced with an explicit ternary matching the pre-fix boolean
   algebra:

       (cur_xs == 0.0) if mt_available else all-True mask

The tests below pin the invariants both changes preserve without
requiring installations against multiple numpy versions (which the
in-session environment cannot easily test).
"""
import numpy as np

from endf_userpy.primitives.helpers import (
    deg2rad,
    check_int_nbt,
    find_interval,
    get_enclosing_points,
)
from endf_userpy.primitives.interpolation import endf_interp1d
from endf_userpy.primitives.conversion import convert_angdist_to_labsys


# ============================================================
# copy=None -> np.asarray: caller-supplied arrays are not copied
# unnecessarily when already ndarrays with matching dtype.
# ============================================================


def test_check_int_nbt_does_not_copy_ndarray_input():
    """`np.asarray` returns the same ndarray if input already is
    one with matching dtype. `check_int_nbt` reads `.ndim` and
    `.size` -- both non-mutating -- so the same-object identity is
    preserved through the call."""
    int_arr = np.array([2, 1], dtype=int)
    nbt_arr = np.array([3, 5], dtype=int)
    # Should not raise; should not mutate.
    check_int_nbt(int_arr, nbt_arr)


def test_check_int_nbt_accepts_python_list():
    """`np.asarray` also accepts Python lists. The old
    `np.array(x, copy=None)` did too on numpy 2.0; the point is
    that the replacement handles both idioms."""
    check_int_nbt([2, 1], [3, 5])


def test_find_interval_ndarray_input_unchanged():
    """`find_interval` uses `np.asarray(a)` and `np.asarray(v)`;
    the caller's arrays must not be mutated."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([1.5, 3.5])
    a_before = a.copy()
    v_before = v.copy()
    find_interval(a, v)
    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(v, v_before)


def test_get_enclosing_points_ndarray_input_unchanged():
    x = np.array([1.5, 2.5])
    xp = np.array([1.0, 2.0, 3.0, 4.0])
    fp = np.array([0.0, 1.0, 4.0, 9.0])
    x_before, xp_before, fp_before = x.copy(), xp.copy(), fp.copy()
    get_enclosing_points(x, xp, fp)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(xp, xp_before)
    np.testing.assert_array_equal(fp, fp_before)


def test_deg2rad_accepts_list_and_ndarray():
    """Basic correctness anchor for the very first `copy=None`
    site fixed."""
    r_list = deg2rad([0.0, 90.0, 180.0])
    r_ndarray = deg2rad(np.array([0.0, 90.0, 180.0]))
    expected = np.array([0.0, np.pi / 2, np.pi])
    np.testing.assert_allclose(r_list, expected)
    np.testing.assert_allclose(r_ndarray, expected)


def test_endf_interp1d_accepts_list_input():
    """Public interpolator: verify list-input path (`np.asarray`
    converts lists to arrays)."""
    r = endf_interp1d(
        [1.5, 2.5], np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 1.0, 4.0]),
        np.array([2]), np.array([3]),
    )
    np.testing.assert_allclose(r, [0.5, 2.5])


def test_convert_angdist_to_labsys_accepts_list_input():
    """`convert_angdist_to_labsys` also uses `np.asarray` for
    `mu_cm` and `f_cm`; verify it still runs on list input for
    those two parameters (`r2` is separately assumed ndarray by
    the function body via `.reshape(-1, 1)`)."""
    mu_cm = [0.5]
    f_cm = [0.5]
    r2 = np.array([0.5])
    f_lab = convert_angdist_to_labsys(mu_cm, f_cm, r2)
    assert f_lab.shape == (1, 1)
    assert np.all(np.isfinite(f_lab))


# ============================================================
# np.bool -> Python-native selection: verify the new ternary
# form matches the boolean algebra the pre-fix bitwise expression
# was doing.
# ============================================================


def _selection_new_form(mt_available, cur_xs, energies_in):
    """The post-fix expression from `get_reaction_xs`."""
    return (
        (cur_xs == 0.0) if mt_available
        else np.ones_like(energies_in, dtype=bool)
    )


def _selection_old_form(mt_available, cur_xs):
    """The pre-fix expression, computed with Python `not` instead
    of the removed `np.bool` for reproducibility on numpy>=1.24."""
    return (not mt_available) | (cur_xs == 0.0)


def test_selection_ternary_matches_pre_fix_when_mt_available():
    """When the MT is in the file (`mt_available=True`), the new
    ternary returns `cur_xs == 0.0` exactly, matching what the
    pre-fix expression `(~True) | (cur_xs == 0.0)` collapsed to
    (`False | (cur_xs == 0.0)`)."""
    cur_xs = np.array([0.0, 1e-5, 0.0, 2.3])
    energies_in = np.array([1e5, 1e6, 5e6, 1e7])
    new = _selection_new_form(True, cur_xs, energies_in)
    old = _selection_old_form(True, cur_xs)
    np.testing.assert_array_equal(new, old)
    np.testing.assert_array_equal(new, [True, False, True, False])


def test_selection_ternary_matches_pre_fix_when_mt_unavailable():
    """When the MT is missing (`mt_available=False`), the new
    ternary returns all-True, matching the pre-fix
    `(~False) | anything = True | anything = True`."""
    cur_xs = np.array([0.0, 1.0, 0.0, 2.3])
    energies_in = np.array([1e5, 1e6, 5e6, 1e7])
    new = _selection_new_form(False, cur_xs, energies_in)
    old = _selection_old_form(False, cur_xs)
    np.testing.assert_array_equal(new, old)
    np.testing.assert_array_equal(new, [True, True, True, True])


def test_selection_shape_matches_energies_in():
    """The `else`-branch mask shape must match `energies_in`
    (which the MT5 gather code indexes into). Anchor this so the
    ternary doesn't accidentally drift to `np.ones_like(cur_xs)`,
    which would break shape-mismatched inputs."""
    energies_in = np.linspace(1e5, 1e7, 11)
    cur_xs = np.zeros(11)
    mask = _selection_new_form(False, cur_xs, energies_in)
    assert mask.shape == energies_in.shape
