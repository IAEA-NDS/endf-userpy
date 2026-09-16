"""Tests for `primitives.interpolation.endf_interp1d` xp-mutation
guarantee (issue #49).

Before this fix, `endf_interp1d` called
`treat_duplicates(xp, inplace=True)` at the top, which perturbs
repeated mesh values by a relative epsilon so that `searchsorted`
can distinguish them. Doing that in place meant any caller that
passed a long-lived array (e.g. one cached from an ENDF dict) got
its mesh silently modified, and a second call on the same array
perturbed the same values again -- compounding the drift.

Today every caller happens to construct `xp` fresh
(`interp_tab1` does `np.array(tab1[xp_name], dtype=float)` for
every call), so nothing was visibly broken. It becomes a latent
bug the moment someone caches the converted meshes, which the
performance issue #46 proposes.

The fix rebinds `xp` inside `endf_interp1d` to the deduplicated
copy returned by `treat_duplicates` (default `inplace=False`), so
the caller's array is untouched.
"""
import numpy as np

from endf_userpy.primitives.interpolation import endf_interp1d
from endf_userpy.primitives.helpers import treat_duplicates


def _flat_mesh_with_step():
    """A monotone mesh with a duplicated value at one break point
    (the shape ENDF-6 tab1 uses for step-function representations).
    Two identical y values so the perturbation from treat_duplicates
    doesn't change the interpolated result."""
    xp = np.array([1.0, 2.0, 3.0, 3.0, 5.0], dtype=float)
    fp = np.array([0.0, 1.0, 2.0, 2.0, 4.0], dtype=float)
    int_arr = np.array([2], dtype=int)
    nbt_arr = np.array([5], dtype=int)
    return xp, fp, int_arr, nbt_arr


def test_endf_interp1d_does_not_mutate_xp():
    """The primary defect: `xp` array must be unchanged after the
    call. Before the fix, the duplicated value at `xp[3]` was
    perturbed by ~1e-8 relative and the caller's array carried that
    perturbation home."""
    xp, fp, int_arr, nbt_arr = _flat_mesh_with_step()
    xp_before = xp.copy()

    endf_interp1d(np.array([1.5, 2.5]), xp, fp, int_arr, nbt_arr)

    np.testing.assert_array_equal(xp, xp_before)


def test_endf_interp1d_repeated_calls_stay_identical():
    """Second-order consequence of the pre-fix in-place mutation:
    a second call on the same `xp` array would perturb the same
    duplicate again. Both interpolation results and the array itself
    stayed identical after the fix; the drift is bounded to the
    single per-call copy `treat_duplicates` now makes internally."""
    xp, fp, int_arr, nbt_arr = _flat_mesh_with_step()
    xp_before = xp.copy()
    x = np.array([1.5, 2.5, 4.0])

    r1 = endf_interp1d(x, xp, fp, int_arr, nbt_arr)
    r2 = endf_interp1d(x, xp, fp, int_arr, nbt_arr)
    r3 = endf_interp1d(x, xp, fp, int_arr, nbt_arr)

    np.testing.assert_array_equal(xp, xp_before)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(r2, r3)


def test_endf_interp1d_unchanged_result_on_mesh_without_duplicates():
    """Regression: the fix must not perturb the returned values on
    meshes without duplicates, since `treat_duplicates` is then a
    no-op copy."""
    xp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fp = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    int_arr = np.array([2])
    nbt_arr = np.array([5])
    x = np.array([1.5, 2.5, 3.5])
    r = endf_interp1d(x, xp, fp, int_arr, nbt_arr)
    np.testing.assert_allclose(r, [0.5, 2.5, 6.5])


def test_endf_interp1d_still_disambiguates_duplicates():
    """Regression: `treat_duplicates` was there for a reason -- with
    a genuinely-different y value at the duplicated x, the
    step-function interpolation must still resolve to the second
    y (i.e. the search-side of the duplicated x). The fix moved
    the dedup to a local copy but didn't remove it."""
    xp = np.array([1.0, 2.0, 2.0, 3.0])
    fp = np.array([0.0, 1.0, 10.0, 20.0])
    int_arr = np.array([1])  # histogram, so f is left-constant
    nbt_arr = np.array([4])
    # Just inside the second interval [2, 3]: expect f=10
    r = endf_interp1d(np.array([2.1]), xp, fp, int_arr, nbt_arr)
    assert r[0] == 10.0


def test_treat_duplicates_returns_copy_by_default():
    """Anchor the underlying `treat_duplicates(inplace=False)`
    contract that the fix depends on: default is copy, not
    mutation."""
    arr = np.array([1.0, 2.0, 2.0, 3.0])
    arr_before = arr.copy()
    result = treat_duplicates(arr)
    np.testing.assert_array_equal(arr, arr_before)
    # The returned copy has the perturbation applied.
    assert result is not arr
    assert result[1] != 2.0 or result[2] != 2.0
