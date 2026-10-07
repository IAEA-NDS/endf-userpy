"""Tests for the above-range XS policy (issue #28).

Naohiko's original report: querying `get_residual_production_xs`
past the ENDF file's upper Ein boundary returned `sigma=0` silently.
That conflates "physically zero" (sub-threshold, or reaction
closed) with "we do not know" (above the evaluation's tabulated
range). The fix introduces a user-configurable policy exposed as
an `above_range` kwarg on every top-level `get_*` XS API in
`endf_userpy.quantities` and inherited by the leaf
`compute_cross_section` readers in `mf3` and `mf10` via a
`contextvars.ContextVar`.

Five policies:

- `'warn_nan'` (default): fill above-range points with NaN; emit
  ONE summary UserWarning per top-level call listing every affected
  MT and its exceeded mesh limit.
- `'nan'`: fill with NaN, silent.
- `'warn_zero'`: fill with 0, summary UserWarning.
- `'zero'`: fill with 0, silent (pre-issue-#28 behaviour).
- `'raise'`: raise ValueError on the first MT with an above-range
  hit.

Below the mesh (typically sub-threshold) always returns 0
regardless of `above_range`, matching physical intuition and
`interp_tab1`'s default fill.
"""
import warnings
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf3_interpretation as mf3
from endf_userpy.mfsec_interpretation.mf3_interpretation import (
    _ABOVE_RANGE_POLICIES,
)
from endf_userpy.run_options import RunOptions
from endf_userpy.quantities import (
    get_reaction_xs,
    get_particle_production_xs,
    get_residual_production_xs,
    get_particle_production_dxs_dE,
    get_particle_production_dxs_dmu,
    get_particle_production_ddxs,
)


ADHOC_DATA_DIR = Path(__file__).resolve().parent / 'data_law1_adhoc'


def _load(fn_name):
    fn = ADHOC_DATA_DIR / fn_name
    if not fn.exists():
        pytest.skip(
            f'{fn_name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh` to populate the corpus'
        )
    parser = EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    )
    return parser.parsefile(fn)


@pytest.fixture(scope='module')
def al27():
    """Al-27 ENDF/B-VIII.1 has max Ein = 150 MeV across all MTs,
    matching the file Naohiko reported on. Queries above 150 MeV
    exercise the above-range policy."""
    return _load('endfb81_n_Al-27.endf')


def _synthetic_mf3(mt=102, e_max=1e7):
    """Minimal synthetic endf_dict with one MF3/MT section spanning
    [1e5, e_max]. Used for isolated policy tests."""
    return {3: {mt: {'xstable': {
        'E': [1e5, 1e6, e_max], 'xs': [1.0, 0.5, 0.1],
        'INT': [2], 'NBT': [3],
    }}}}


# ============================================================
# Leaf-level: each of the five policies behaves as documented.
# ============================================================


def test_leaf_default_warn_nan_returns_nan_above_and_finite_below():
    d = _synthetic_mf3(e_max=1e7)
    E = np.array([1e5, 5e6, 1e7, 1.5e7, 2e7])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = mf3.compute_cross_section(d, 102, E)
    # First three points inside the mesh: finite, non-NaN.
    assert np.all(np.isfinite(r[:3]))
    # Last two points above the mesh: NaN.
    assert np.all(np.isnan(r[3:]))
    # Leaf call (no ctx active) emits one warning.
    assert len(w) == 1
    assert 'MT=102' in str(w[0].message)


def test_leaf_nan_silent():
    d = _synthetic_mf3(e_max=1e7)
    E = np.array([1e5, 2e7])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = mf3.compute_cross_section(d, 102, E, above_range='nan')
    assert np.isfinite(r[0])
    assert np.isnan(r[1])
    assert len(w) == 0


def test_leaf_warn_zero_matches_prefix_numeric():
    d = _synthetic_mf3(e_max=1e7)
    E = np.array([1e5, 2e7])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = mf3.compute_cross_section(d, 102, E, above_range='warn_zero')
    np.testing.assert_array_equal(r[1:], 0.0)
    assert len(w) == 1
    assert '0' in str(w[0].message) and 'NaN' not in str(w[0].message)


def test_leaf_zero_silent_backwards_compat():
    """Pre-issue-#28 behaviour: exact numeric of the old default."""
    d = _synthetic_mf3(e_max=1e7)
    E = np.array([1e5, 5e6, 2e7])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = mf3.compute_cross_section(d, 102, E, above_range='zero')
    assert np.all(np.isfinite(r))
    assert r[-1] == 0.0
    assert len(w) == 0


def test_leaf_raise_stops_the_call():
    d = _synthetic_mf3(e_max=1e7)
    E = np.array([1e5, 2e7])
    with pytest.raises(ValueError, match=r'MT=102.*upper mesh energy'):
        mf3.compute_cross_section(d, 102, E, above_range='raise')


def test_leaf_invalid_policy_raises():
    d = _synthetic_mf3()
    with pytest.raises(ValueError, match=r'above_range must be one of'):
        mf3.compute_cross_section(
            d, 102, np.array([1e5]), above_range='not_a_policy',
        )


def test_leaf_below_mesh_always_returns_zero():
    """Below the file's lowest tabulated Ein, `interp_tab1` returns
    0 regardless of `above_range`. Physically this matches sub-
    threshold sigma=0; the fix does not touch that branch."""
    d = _synthetic_mf3(e_max=1e7)
    # 1e4 is below the mesh's lowest point (1e5)
    E = np.array([1e4, 1e5])
    for policy in ('warn_nan', 'nan', 'warn_zero', 'zero', 'raise'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = mf3.compute_cross_section(d, 102, E, above_range=policy)
        assert r[0] == 0.0, f'{policy}: below-mesh must be 0'


def test_leaf_all_inside_mesh_no_signal_no_change():
    """If all `energies_in` are inside the mesh, all policies
    produce identical numeric output and no warnings / no raise."""
    d = _synthetic_mf3(e_max=1e7)
    E = np.array([1e5, 5e6, 1e7])
    results = {}
    for policy in ('warn_nan', 'nan', 'warn_zero', 'zero', 'raise'):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            results[policy] = mf3.compute_cross_section(
                d, 102, E, above_range=policy,
            )
        assert len(w) == 0, f'{policy}: unexpected warning'
    baseline = results['zero']
    for policy, xs in results.items():
        np.testing.assert_allclose(xs, baseline, err_msg=policy)


# ============================================================
# One-warning-per-top-level-call summary property (issue #143
# moved this from ctx-manager-driven to RunOptions-driven via
# the private _warning_summary_ctx installed by every top-level
# entry point).
# ============================================================


def test_summary_warning_lists_every_mt(al27):
    """The primary UX property: one summary UserWarning per top-
    level call listing every affected MT and the exceeded mesh
    limit -- not one warning per MT (~150 for a fully-summed
    file)."""
    E = np.array([1e6, 2e8])  # 200 MeV > Al-27's 150 MeV cap
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        xs = get_reaction_xs(al27, '(n,total)', E)
    # Exactly one summary warning (not ~150).
    assert len(w) == 1
    msg = str(w[0].message)
    assert 'MT=' in msg
    assert '1.5e+08' in msg
    assert np.isnan(xs[-1])
    assert np.isfinite(xs[0])


def test_no_warning_when_all_inside(al27):
    """When every incident energy is inside the mesh (Al-27 max is
    150 MeV; user queries up to 20 MeV), no summary warning fires."""
    E = np.linspace(1e6, 2e7, 5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        xs = get_reaction_xs(al27, '(n,total)', E)
    assert len(w) == 0
    assert np.all(np.isfinite(xs))


# ============================================================
# End-to-end: each top-level get_* API forwards the policy.
# ============================================================


def test_get_reaction_xs_zero_silent(al27):
    """Backwards-compat opt-in: `above_range='zero'` returns 0 in
    the tail, no warnings. Matches Naohiko's report pre-fix."""
    E = np.array([1e6, 2e8])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        xs = get_reaction_xs(al27, '(n,total)', E, options=RunOptions(above_range='zero'))
    assert xs[-1] == 0.0
    assert len(w) == 0


def test_get_reaction_xs_raise_propagates(al27):
    E = np.array([1e6, 2e8])
    with pytest.raises(ValueError, match=r'MT=.*upper mesh energy'):
        get_reaction_xs(al27, '(n,total)', E, options=RunOptions(above_range='raise'))


def test_get_particle_production_xs_takes_above_range(al27):
    """Confirm the kwarg is on the particle-production XS API too."""
    E = np.array([1e6, 2e8])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs_nan = get_particle_production_xs(al27, '(n,total)', 'n', E)
        xs_zero = get_particle_production_xs(al27, '(n,total)', 'n', E, options=RunOptions(above_range='zero'))
    if xs_nan is not None:
        assert np.isnan(xs_nan[-1]) or xs_nan[-1] == 0.0
    if xs_zero is not None:
        assert xs_zero[-1] == 0.0


def test_get_residual_production_xs_takes_above_range(al27):
    """Naohiko's exact API path. Same MT 2 elastic on Al-27 stays
    Al-27; use Al-27 as the residual so we get a positive result
    in the mesh interior. Pass ``include_target=True`` so MT 2
    counts toward the Al-27 residual (the #137 default
    ``include_target=False`` drops target-conserving channels and
    the diagonal case would return zero everywhere, defeating the
    above-range test)."""
    E = np.array([1e6, 2e8])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs_nan = get_residual_production_xs(
            al27, 'Al-27', E,
            options=RunOptions(include_target=True),
        )
        xs_zero = get_residual_production_xs(
            al27, 'Al-27', E,
            options=RunOptions(include_target=True, above_range='zero'),
        )
    assert np.isnan(xs_nan[-1])
    assert xs_zero[-1] == 0.0
    # Below-mesh point unchanged.
    np.testing.assert_allclose(xs_nan[0], xs_zero[0], atol=1e-12)


def test_differential_apis_accept_run_options_with_above_range():
    """The three differential entry points route `above_range`
    through RunOptions, same as the XS APIs. Signature check: each
    accepts ``options=``; RunOptions default matches
    ``'warn_nan'``. End-to-end behaviour exercised in the
    broadening-path tests below."""
    import inspect
    for func in (
        get_particle_production_dxs_dE,
        get_particle_production_dxs_dmu,
        get_particle_production_ddxs,
    ):
        sig = inspect.signature(func)
        assert 'options' in sig.parameters, (
            f'{func.__name__} missing `options` kwarg'
        )
    assert RunOptions().above_range == 'warn_nan'


# ============================================================
# Issue #303: policy propagates through broadening folders.
# ============================================================
#
# The seven direct `mf3_interp.compute_cross_section` call sites
# inside `ddx_broadening.py` used to apply the leaf's default
# policy regardless of what the top-level user requested. These
# tests pin the fix: a top-level `above_range='raise'` must raise
# when a broadened query touches above-range Ein. Before #303
# these calls would silently return NaN-filled arrays instead.


def test_broadened_dxs_dE_above_range_raise_propagates(al27):
    """`get_particle_production_dxs_dE(..., broadening=sigma)` with
    `above_range='raise'` must raise on Ein above the file mesh.
    Pins the invariant end-to-end on the 1D path; the DDX
    sibling tests below provide the stricter per-folder coverage
    (dxs_dE's continuous folder was already threaded in #301, so a
    stashed-fix run of this test still raises via that folder before
    the mf12-discrete folder is reached)."""
    E = np.array([1e6, 2e8])     # 200 MeV is above Al-27's 150 MeV mesh
    Eo = np.linspace(1e5, 1e7, 32)
    opts = RunOptions(above_range='raise')
    with pytest.raises(ValueError, match=r'exceed.*upper mesh energy'):
        get_particle_production_dxs_dE(
            al27, '(n,g)', 'g', E, Eo, broadening=1e3, options=opts,
        )


def test_broadened_ddx_above_range_raise_propagates(al27):
    """`get_particle_production_ddxs(..., broadening=sigma)` with
    `above_range='raise'` must raise on Ein above the file mesh.
    Exercises `compute_ddx_continuous_broadened`, `_mf12_discrete_`,
    and `_mf15_continuum_` folders in a single query."""
    E = np.array([1e6, 2e8])
    Eo = np.linspace(1e5, 1e7, 16)
    mu = np.linspace(-1.0, 1.0, 8)
    opts = RunOptions(above_range='raise')
    with pytest.raises(ValueError, match=r'exceed.*upper mesh energy'):
        get_particle_production_ddxs(
            al27, '(n,g)', 'g', E, Eo, mu, broadening=1e3, options=opts,
        )


def test_broadened_neutron_ddx_above_range_raise_propagates(al27):
    """Neutron-production DDX hits `compute_ddx_continuous_broadened`
    (and `_summed` if multiple MTs admit) via a neutron query. Same
    `above_range='raise'` propagation invariant."""
    E = np.array([1e6, 2e8])
    Eo = np.linspace(1e5, 1e7, 16)
    mu = np.linspace(-1.0, 1.0, 8)
    opts = RunOptions(above_range='raise')
    with pytest.raises(ValueError, match=r'exceed.*upper mesh energy'):
        get_particle_production_ddxs(
            al27, '(n,n)', 'n', E, Eo, mu, broadening=1e3, options=opts,
        )


# ============================================================
# Design invariants.
# ============================================================


def test_five_policies_registered():
    """Fixed set; nothing else."""
    assert set(_ABOVE_RANGE_POLICIES) == {
        'warn_nan', 'nan', 'warn_zero', 'zero', 'raise',
    }


def test_leaf_default_matches_ctx_default():
    """Without an active ctx and without an explicit kwarg, the
    leaf uses `'warn_nan'` -- matches `above_range_ctx`'s stated
    default."""
    d = _synthetic_mf3(e_max=1e7)
    E = np.array([1e5, 2e7])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = mf3.compute_cross_section(d, 102, E)
    assert np.isnan(r[-1])
    assert len(w) == 1  # warn_nan emits one warning at leaf
