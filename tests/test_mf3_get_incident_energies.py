"""Tests for `mf3_interpretation.get_incident_energies` bounds
(issue #96).

Two bugs in the pre-fix code:

1. Off-by-one on the trailing bracketing zero. The line

       last_idx = idcs[-1]+1 if idcs[-1]+1 < len(idcs) else idcs[-1]

   compared the last-nonzero index against ``len(idcs)`` (the count
   of nonzero points) instead of ``len(xs)`` (the mesh length). The
   trailing bracketing zero was retained only when there were no
   zeros before the last nonzero point. Any MT whose XS returns to
   zero at high Ein lost its final Ein point.

2. Empty ``idcs`` crash. All-zero cross sections (some libraries
   carry placeholder MT entries for MF6 or MF8 book-keeping with a
   zeroed MF3) hit ``idcs[0]`` and raised ``IndexError``.

Fix: cap `last_idx` against `len(xs) - 1`, and return an empty
mesh for all-zero XS. `get_incident_energy_range` raises
``ValueError`` on empty (min/max are undefined).
"""
import numpy as np
import pytest

from endf_userpy.mfsec_interpretation.mf3_interpretation import (
    get_incident_energies,
    get_incident_energy_range,
)


def _synthetic(mt, E, xs):
    return {3: {mt: {'xstable': {'E': list(E), 'xs': list(xs)}}}}


# ============================================================
# Bug 1: trailing bracketing zero must be kept.
# ============================================================


def test_trailing_zero_kept_symmetric_with_leading_zero():
    """The exact repro from issue #96. XS = [0, 0, 1, 1, 0]:
    nonzero block is at indices 2..3; the mesh should return
    indices 1..4 (one bracketing zero on each side)."""
    d = _synthetic(16, [1e6, 2e6, 3e6, 4e6, 5e6],
                       [0.0, 0.0, 1.0, 1.0, 0.0])
    result = get_incident_energies(d, 16)
    np.testing.assert_array_equal(result, [2e6, 3e6, 4e6, 5e6])


def test_nonzero_block_touching_right_edge_keeps_last_point():
    """XS = [0, 1, 1, 1, 1]: nonzero block runs to the end of the
    mesh; result should include the whole tail with the leading
    zero bracketed."""
    d = _synthetic(16, [1e6, 2e6, 3e6, 4e6, 5e6],
                       [0.0, 1.0, 1.0, 1.0, 1.0])
    result = get_incident_energies(d, 16)
    np.testing.assert_array_equal(result, [1e6, 2e6, 3e6, 4e6, 5e6])


def test_nonzero_block_touching_left_edge_keeps_first_point():
    """XS = [1, 1, 1, 0, 0]: nonzero from index 0 to 2; result
    should keep index 0 and add the trailing zero at index 3."""
    d = _synthetic(16, [1e6, 2e6, 3e6, 4e6, 5e6],
                       [1.0, 1.0, 1.0, 0.0, 0.0])
    result = get_incident_energies(d, 16)
    np.testing.assert_array_equal(result, [1e6, 2e6, 3e6, 4e6])


def test_nonzero_block_covers_whole_mesh():
    """XS all nonzero: result is the whole mesh (no bracketing
    zeros exist to add)."""
    d = _synthetic(16, [1e6, 2e6, 3e6],
                       [1.0, 2.0, 3.0])
    result = get_incident_energies(d, 16)
    np.testing.assert_array_equal(result, [1e6, 2e6, 3e6])


def test_single_nonzero_point_gets_bracketed_both_sides():
    """XS = [0, 0, 1, 0, 0]: single nonzero at index 2; result
    should include indices 1..3."""
    d = _synthetic(16, [1e6, 2e6, 3e6, 4e6, 5e6],
                       [0.0, 0.0, 1.0, 0.0, 0.0])
    result = get_incident_energies(d, 16)
    np.testing.assert_array_equal(result, [2e6, 3e6, 4e6])


# ============================================================
# Bug 2: all-zero XS returns empty rather than crashing.
# ============================================================


def test_all_zero_xs_returns_empty_mesh():
    d = _synthetic(16, [1e6, 2e6, 3e6, 4e6, 5e6],
                       [0.0, 0.0, 0.0, 0.0, 0.0])
    result = get_incident_energies(d, 16)
    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    assert result.size == 0


def test_all_zero_xs_range_raises_valueerror():
    d = _synthetic(16, [1e6, 2e6, 3e6],
                       [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match='all-zero cross section'):
        get_incident_energy_range(d, 16)


# ============================================================
# Non-regression: normal range case still works.
# ============================================================


def test_range_returns_first_and_last_of_trimmed_mesh():
    d = _synthetic(16, [1e6, 2e6, 3e6, 4e6, 5e6],
                       [0.0, 0.0, 1.0, 1.0, 0.0])
    lo, hi = get_incident_energy_range(d, 16)
    assert lo == 2e6
    assert hi == 5e6
