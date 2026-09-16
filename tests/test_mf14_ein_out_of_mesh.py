"""Tests for `evaluate_interp_legendre_polynomials`'s per-line
Ein-out-of-mesh handling in the MF14 Legendre path (issue #81).

MF14 stores per-photon-line Legendre coefficients over an
Ein mesh that is specific to each line (`mtsec['E'][line_idx]`).
The user-supplied `energies_in` grid to
`get_particle_production_dxs_dmu` and
`get_particle_production_ddxs` frequently spans a wider range than
any single line's tabulation -- e.g. JENDL-5 C-12 gamma lines are
tabulated only above the (n,g) threshold while the user passes an
Ein grid starting at 100 keV.

Previously `endf_interp1d` raised
``ValueError: some 'x' value outside mesh given by 'xp'`` whenever
any `energies_in` point fell outside the line's own mesh, which
made gamma `dxs/dmu` unusable on JENDL-5 C-12 even on modest input
grids. Now the leaf helpers accept `outside_value=0.0` and the
MF14 caller passes that value, so out-of-mesh Ein contributes zero
to that line (the physically correct answer -- the line simply
wasn't tabulated there).

The MF4 callers of `evaluate_interp_legendre_polynomials` are
guarded by the `pad_outside_*` decorator upstream, so they never
hit this path; behaviour unchanged.

Tests pin:

1. Public API on JENDL-5 C-12 returns finite arrays.
2. Direct leaf: `evaluate_interp_legendre_polynomials` with
   `outside_value=0.0` produces zeros outside `xp` and matches the
   default in-range case bit-for-bit inside `xp`.
3. Non-regression: MF4 callers (via mf4_interpretation) still
   work unchanged.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_dxs_dmu,
    get_particle_production_ddxs,
)
from endf_userpy.primitives.interpolation import (
    evaluate_interp_legendre_polynomials,
)


ADHOC = Path(__file__).resolve().parent / 'data_law1_adhoc'


def _load(fn_name):
    fn = ADHOC / fn_name
    if not fn.exists():
        pytest.skip(
            f'{fn_name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh`'
        )
    return EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    ).parsefile(fn)


@pytest.fixture(scope='module')
def c12_jendl5():
    return _load('jendl5_n_C-12.endf')


# ============================================================
# Public API regression: JENDL-5 C-12 (n,total)+g.
# ============================================================


def test_c12_gamma_dxs_dmu_finite(c12_jendl5):
    einc = np.linspace(1e5, 1.4e7, 6)
    mus = np.linspace(-0.9, 0.9, 5)
    r = get_particle_production_dxs_dmu(
        c12_jendl5, '(n,total)', 'g', einc, mus,
    )
    assert r.shape == (6, 5)
    assert np.all(np.isfinite(r))
    assert np.all(r >= 0.0)


def test_c12_gamma_ddx_broadened_finite(c12_jendl5):
    einc = np.linspace(1e5, 1.4e7, 6)
    eouts = np.linspace(1e5, 1.4e7, 20)
    mus = np.linspace(-0.9, 0.9, 5)
    r = get_particle_production_ddxs(
        c12_jendl5, '(n,total)', 'g', einc, eouts, mus, broadening=3e4,
    )
    assert r is not None
    assert r.shape == (6, 20, 5)
    assert np.all(np.isfinite(r))


# ============================================================
# Direct leaf: outside_value=0.0 zeroes out-of-xp cells; matches
# the default in-range case bit-for-bit inside xp.
# ============================================================


def test_evaluate_interp_legendre_outside_zero():
    """Synthetic 2-degree Legendre coefficients on `xp=[1e6, 5e6]`
    with `outside_value=0.0`: for `x` outside `[1e6, 5e6]` the
    interpolated coefficients are zero, so the Legendre evaluation
    is zero at those `x`."""
    xp = np.array([1e6, 5e6], dtype=float)
    # 2-degree (constant + linear-in-mu) with slightly different
    # coefficients at each xp
    coeffs = np.array([[1.0, 0.5], [1.0, -0.5]], dtype=float)
    int_arr = np.array([2], dtype=int)
    nbt_arr = np.array([2], dtype=int)
    mu = np.array([0.0, 0.5, -0.5], dtype=float)
    # Some x inside, some outside
    x = np.array([5e5, 2e6, 4e6, 6e6], dtype=float)
    f = evaluate_interp_legendre_polynomials(
        x, mu, xp, coeffs, int_arr, nbt_arr, outside_value=0.0,
    )
    assert f.shape == (4, 3)
    # First (x=5e5) and last (x=6e6) rows: outside xp -> all zero
    np.testing.assert_array_equal(f[0], np.zeros(3))
    np.testing.assert_array_equal(f[3], np.zeros(3))
    # Middle rows: inside xp -> non-trivial
    assert not np.all(f[1] == 0)
    assert not np.all(f[2] == 0)


def test_evaluate_interp_legendre_default_still_raises_outside():
    """Regression guard: the default `outside_value=None` path
    must still raise for out-of-mesh `x`, so MF4 callers that
    rely on the raise (they pre-filter via `pad_outside_*`
    decorators) don't silently start swallowing bugs."""
    xp = np.array([1e6, 5e6], dtype=float)
    coeffs = np.array([[1.0], [1.0]], dtype=float)
    int_arr = np.array([2], dtype=int)
    nbt_arr = np.array([2], dtype=int)
    mu = np.array([0.0], dtype=float)
    x = np.array([2e6, 6e6], dtype=float)  # 6e6 is outside
    with pytest.raises(ValueError, match='outside mesh'):
        evaluate_interp_legendre_polynomials(
            x, mu, xp, coeffs, int_arr, nbt_arr,
        )


def test_evaluate_interp_legendre_outside_matches_default_inside():
    """Inside `xp` the `outside_value=0.0` and default paths must
    return identical numeric values."""
    xp = np.array([1e6, 5e6, 1e7], dtype=float)
    coeffs = np.array([[1.0, 0.3, 0.1], [0.8, 0.4, -0.1], [1.2, 0.1, 0.05]])
    int_arr = np.array([2], dtype=int)
    nbt_arr = np.array([3], dtype=int)
    mu = np.array([-0.5, 0.0, 0.5], dtype=float)
    x = np.array([2e6, 5e6, 8e6], dtype=float)  # all inside
    a = evaluate_interp_legendre_polynomials(x, mu, xp, coeffs, int_arr, nbt_arr)
    b = evaluate_interp_legendre_polynomials(
        x, mu, xp, coeffs, int_arr, nbt_arr, outside_value=0.0,
    )
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


# ============================================================
# MF4 non-regression: Legendre-based MF4 evaluators (which go
# through the same helper) still work.
# ============================================================


def test_mf4_legendre_still_works():
    """The MF4 evaluator's Legendre branch also calls
    `evaluate_interp_legendre_polynomials` (with the default
    `outside_value=None`). Guarded upstream by the
    `pad_outside_angdist_values` decorator, so this call must
    still work on any file with MF4 elastic."""
    endf = _load('endfb81_n_Al-27.endf')
    einc = np.linspace(1e5, 1.4e7, 6)
    mus = np.linspace(-0.9, 0.9, 5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        r = get_particle_production_dxs_dmu(
            endf, '(n,total)', 'n', einc, mus,
        )
    assert r.shape == (6, 5)
    assert np.all(np.isfinite(r))
    assert float(r.max()) > 0.0
