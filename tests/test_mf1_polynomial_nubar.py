"""Tests for MF1 LNU=1 polynomial nubar (issue #42).

`_compute_yields_from_polynomial` used to do

    coefs * (energies_in**(np.arange(len(coefs))))

which is elementwise multiplication (not the polynomial sum) and
broadcasts only when ``n_ein == len(coefs)``. Every realistic
call from `compute_yields_from_mt452` / `compute_yields_from_mt455`
therefore raised
``ValueError: operands could not be broadcast together with shapes
(n_ein,) (NC,)``.

`compute_yields_from_mt456` (prompt nubar) handles ``LNU=1`` as a
constant separately, so the prompt-nubar path used by
`compute_yields` for MT=18 was unaffected -- only total (MT452)
and delayed (MT455) nubar with polynomial representation were
broken.

Fix uses `np.polynomial.polynomial.polyval` (coefficients in
ascending order, matching the ENDF-6 MF1 LNU=1 convention).

No corpus file exercises MF1 MT452/455/456 with LNU=1 (issue #48
tracks the coverage gap), so these tests use synthetic dicts.
"""
import numpy as np
import pytest

from endf_userpy.mfsec_interpretation.mf1_interpretation import (
    _compute_yields_from_polynomial,
    compute_yields_from_mt452,
    compute_yields_from_mt455,
    compute_yields,
)


# ============================================================
# Helper-level: the polynomial evaluator.
# ============================================================


def test_polynomial_evaluates_correct_sum():
    """Direct check of the arithmetic: nu(E) = sum_k C_k * E^k.
    Pick coefficients c = (2.0, 3.0, 0.5) so nu(E) = 2 + 3E + 0.5E^2
    and sample at E = 0, 1, 2, 10."""
    coefs = np.array([2.0, 3.0, 0.5])
    E = np.array([0.0, 1.0, 2.0, 10.0])
    result = _compute_yields_from_polynomial(coefs, E)
    expected = 2.0 + 3.0 * E + 0.5 * E**2
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_polynomial_shape_survives_unequal_n_ein_ncoefs():
    """The primary defect: pre-fix code raised for `n_ein != NC`.
    After the fix the return shape is ``(n_ein,)`` regardless of
    the number of polynomial coefficients."""
    # 3 coefficients, 7 energy grid points -- would have raised
    # ValueError from the elementwise broadcast pre-fix.
    coefs = np.array([2.4, 1e-7, -1e-16])
    E = np.linspace(1e6, 2e7, 7)
    r = _compute_yields_from_polynomial(coefs, E)
    assert r.shape == (7,)
    assert np.all(np.isfinite(r))


def test_polynomial_matches_realistic_nubar_shape():
    """Realistic MF1 LNU=1 coefficients from the issue body:
    (2.4, 1e-7). At E = 1 MeV nu ~= 2.4 + 1e-7 * 1e6 = 2.5;
    at 10 MeV nu ~= 2.4 + 1.0 = 3.4."""
    coefs = np.array([2.4, 1e-7])
    E = np.array([1e6, 1e7])
    r = _compute_yields_from_polynomial(coefs, E)
    np.testing.assert_allclose(r, [2.5, 3.4], rtol=1e-12)


def test_polynomial_single_coefficient_is_constant():
    """Edge case: a single coefficient is a constant nubar."""
    coefs = np.array([2.5])
    E = np.linspace(0.0, 1e7, 5)
    r = _compute_yields_from_polynomial(coefs, E)
    np.testing.assert_allclose(r, np.full_like(E, 2.5))


# ============================================================
# End-to-end: compute_yields_from_mt452/mt455/mt456 through the
# public dispatcher on a synthetic ENDF dict.
# ============================================================


def _mf1_mt_polynomial(mt, coefs_field_name, coefs_values):
    """Build a synthetic endf_dict[1][mt] section for LNU=1
    (polynomial). The coefficient field name differs between MTs:
    'C' for MT452, 'nubar_d' for MT455."""
    return {
        1: {
            mt: {
                'LNU': 1,
                coefs_field_name: {i + 1: v for i, v in enumerate(coefs_values)},
            }
        }
    }


def test_mt452_lnu1_polynomial_end_to_end():
    """MT 452 total nubar, LNU=1. compute_yields_from_mt452 should
    return nu(E) = sum_k C_k E^k evaluated on the user grid."""
    d = _mf1_mt_polynomial(452, 'C', [2.4, 1e-7])
    E = np.array([1e6, 5e6, 1e7])
    r = compute_yields_from_mt452(d, E)
    expected = 2.4 + 1e-7 * E
    np.testing.assert_allclose(r, expected, rtol=1e-12)


def test_mt455_lnu1_polynomial_end_to_end():
    """MT 455 delayed nubar, LNU=1. Coefficient field is
    'nubar_d' per the mf1 reader."""
    d = _mf1_mt_polynomial(455, 'nubar_d', [0.02, 5e-9])
    E = np.array([1e6, 5e6, 1e7])
    r = compute_yields_from_mt455(d, E)
    expected = 0.02 + 5e-9 * E
    np.testing.assert_allclose(r, expected, rtol=1e-12)


def test_dispatcher_selects_polynomial_path_for_mt452():
    """Regression: the top-level `compute_yields` dispatcher must
    still route MT=452 through `compute_yields_from_mt452`, which
    (with the fix) now handles LNU=1 correctly."""
    d = _mf1_mt_polynomial(452, 'C', [2.4, 1e-7])
    E = np.array([1e6, 1e7])
    r = compute_yields(d, 452, E)
    np.testing.assert_allclose(r, [2.5, 3.4], rtol=1e-12)


# ============================================================
# Anchor: MT 456 prompt-nubar LNU=1 path is a separate constant
# constructor (not this polynomial evaluator). Nothing to fix,
# nothing to regress, but pin its shape so it doesn't drift.
# ============================================================


def test_mt456_lnu1_constant_still_works():
    """MT 456 LNU=1 returns a constant `nubar_p` -- unaffected by
    this fix but worth pinning here so nobody switches it to the
    polynomial path by mistake."""
    from endf_userpy.mfsec_interpretation.mf1_interpretation import (
        compute_yields_from_mt456,
    )
    d = {1: {456: {'LNU': 1, 'nubar_p': 2.42}}}
    E = np.linspace(1e5, 1e7, 5)
    r = compute_yields_from_mt456(d, E)
    np.testing.assert_allclose(r, np.full_like(E, 2.42))


# ============================================================
# Anchor: unsupported LNU values raise a legible ValueError.
# The issue body flagged missing f-prefixes in the error strings;
# checking current source shows the f-prefixes are already present
# so nothing changes on that front.
# ============================================================


def test_unknown_lnu_raises_value_error_with_number_in_message():
    d = {1: {452: {'LNU': 3}}}
    with pytest.raises(ValueError, match=r'LNU=3'):
        compute_yields_from_mt452(d, np.array([1e6]))
