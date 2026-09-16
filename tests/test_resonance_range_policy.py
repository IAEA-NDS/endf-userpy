"""Tests for the resonance-range policy in
`mf3_interpretation.compute_cross_section` (issue #84).

ENDF-6 stores MF3 inside the file's resolved-resonance region
(LRU=1 in MF2/MT151) as a subtractive **background** cross section
that must be added to the resonance reconstruction from MF2. This
library does not reconstruct MF2 resonances (documented in README's
"Known limitations"), so raw MF3 in the RRR is not the physical
cross section and can even be negative -- JENDL-5 Cu-63 MT1 and
MT2 tabulate `-0.9 barn` at thermal energies.

The audit (issue #84) flagged this as a silent numeric-loss bug at
the top-level API. The fix is a new `resonance_range=` kwarg on
every XS-emitting `get_*` entry point in `endf_userpy.quantities`,
mirroring the `above_range=` pattern from issue #28:

- ``'warn'`` (default): return raw MF3 background, one summary
  UserWarning per top-level query.
- ``'warn_nan'``: replace in-RRR values with NaN plus warning.
- ``'nan'``: same but silent.
- ``'raise'``: hard error on any Ein inside the RRR.

The `'clip'` / `'warn_clip'` variants were considered and dropped:
the MF3 background is a subtractive residual, not a physical XS on
its own, so clipping to zero produces nothing meaningful.

Files without MF2 (photonuclear, some light-nuclide evaluations)
or with only LRU=0 / LRU=2 ranges are unaffected -- the policy
layer is a silent no-op there.

These tests pin the numeric behaviour of each policy on the
JENDL-5 Cu-63 file that motivated the issue, the silent-no-op
behaviour on H-1 (LRU=0 only), the summary-warning shape, and
non-regression for the pre-issue-#84 default numeric answer above
the RRR.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_reaction_xs,
    get_particle_production_xs,
    get_particle_production_dxs_dE,
)
from endf_userpy.mfsec_interpretation.mf3_interpretation import (
    get_resolved_resonance_ranges,
    compute_cross_section,
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
def cu63_jendl5():
    """RRR: LRU=1 EL=1e-5 EH=99.5 keV. MT1/MT2 tabulate -0.9 barn
    at thermal energies."""
    return _load('jendl5_n_Cu-63.endf')


@pytest.fixture(scope='module')
def h1_endfb81():
    """LRU=0 only: no resolved-resonance range to check against."""
    return _load('endfb81_n_H-1.endf')


@pytest.fixture(scope='module')
def fe56_tendl21():
    """RRR present but MF3 background is positive at thermal; used
    to confirm the warning fires whenever the query touches the RRR
    (not only when values are negative)."""
    return _load('tendl21_n_Fe-56.endf')


# ============================================================
# Bounds helper (no MF2 -> empty, LRU=0 -> empty, LRU=1 -> tuple).
# ============================================================


def test_get_resolved_resonance_ranges_cu63(cu63_jendl5):
    ranges = get_resolved_resonance_ranges(cu63_jendl5)
    assert len(ranges) == 1
    el, eh = ranges[0]
    assert el == pytest.approx(1e-5, rel=1e-9)
    assert eh == pytest.approx(9.95e4, rel=1e-9)


def test_get_resolved_resonance_ranges_h1_lru0_returns_empty(h1_endfb81):
    """H-1 has LRU=0 (scattering radius only, no resonance params).
    The helper must skip LRU=0 ranges so downstream policy is a
    silent no-op."""
    assert get_resolved_resonance_ranges(h1_endfb81) == []


# ============================================================
# Policy behaviour on Cu-63 at thermal energies (in-RRR).
# ============================================================


IN_RRR_EINC = np.array([0.001, 1.0, 100.0, 1e4], dtype=float)
ABOVE_RRR_EINC = np.array([2e6, 1e7, 1.4e7], dtype=float)


def _get_warnings(recorded, needle='resonance'):
    """Grab both the summary form ('resonance_range: N in-RRR
    points ...') emitted by `resonance_range_ctx.__exit__` and the
    per-call fallback form ('... resolved-resonance region ...')
    emitted by the leaf helper when no context is active. Match on
    'resonance' which appears in both."""
    return [w for w in recorded if needle in str(w.message)]


def test_default_warn_returns_raw_and_warns(cu63_jendl5):
    """Default policy: numeric answer unchanged from pre-fix
    behaviour; one summary UserWarning fires."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        xs = get_reaction_xs(cu63_jendl5, '(n,total)', IN_RRR_EINC)
    assert xs.shape == IN_RRR_EINC.shape
    # Cu-63 MT1/MT2 both give -0.9 barn at thermal; the summed
    # (n,total) result stays negative here (below the resonance
    # region proper -- no reconstruction adds anything back).
    assert np.all(np.isfinite(xs))
    assert xs[0] < 0.0
    hits = _get_warnings(recorded)
    assert len(hits) == 1
    msg = str(hits[0].message)
    assert 'RRR' in msg
    assert 'MT=' in msg
    assert 'issue' not in msg or '#84' not in msg  # message doesn't self-cite


def test_warn_nan_replaces_in_rrr_with_nan(cu63_jendl5):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        xs = get_reaction_xs(
            cu63_jendl5, '(n,total)', IN_RRR_EINC,
            resonance_range='warn_nan',
        )
    assert np.all(np.isnan(xs))
    assert len(_get_warnings(recorded)) == 1


def test_nan_replaces_in_rrr_silently(cu63_jendl5):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        xs = get_reaction_xs(
            cu63_jendl5, '(n,total)', IN_RRR_EINC,
            resonance_range='nan',
        )
    assert np.all(np.isnan(xs))
    assert _get_warnings(recorded) == []


def test_raise_hard_errors_on_any_in_rrr_point(cu63_jendl5):
    with pytest.raises(ValueError, match='resolved-resonance region'):
        get_reaction_xs(
            cu63_jendl5, '(n,total)', IN_RRR_EINC,
            resonance_range='raise',
        )


def test_invalid_policy_rejected(cu63_jendl5):
    with pytest.raises(ValueError, match='resonance_range must be one of'):
        get_reaction_xs(
            cu63_jendl5, '(n,total)', IN_RRR_EINC,
            resonance_range='clip',
        )
    with pytest.raises(ValueError, match='resonance_range must be one of'):
        get_reaction_xs(
            cu63_jendl5, '(n,total)', IN_RRR_EINC,
            resonance_range='passthrough',
        )


# ============================================================
# Non-regression: above-RRR queries are unaffected. RRR-free
# files are silent no-ops.
# ============================================================


def test_above_rrr_query_unchanged_and_no_warning(cu63_jendl5):
    """Queries entirely above the file's RRR (Ein > 99.5 keV for
    Cu-63) must not fire a resonance-range warning and must return
    the same numeric answer under all four policies."""
    xs_ref = get_reaction_xs(
        cu63_jendl5, '(n,total)', ABOVE_RRR_EINC,
        resonance_range='nan',  # cheapest silent policy
    )
    assert np.all(np.isfinite(xs_ref))
    assert np.all(xs_ref > 0.0)
    for pol in ('warn', 'warn_nan', 'raise'):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            xs = get_reaction_xs(
                cu63_jendl5, '(n,total)', ABOVE_RRR_EINC,
                resonance_range=pol,
            )
        np.testing.assert_array_equal(xs, xs_ref)
        assert _get_warnings(recorded) == [], (
            f'policy={pol}: unexpected resonance_range warning above RRR'
        )


def test_lru0_only_file_is_silent_noop(h1_endfb81):
    """H-1 has LRU=0 only; the policy layer must not fire under
    any policy including 'raise' (nothing to check against)."""
    for pol in ('warn', 'warn_nan', 'nan', 'raise'):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            xs = get_reaction_xs(
                h1_endfb81, '(n,total)', IN_RRR_EINC,
                resonance_range=pol,
            )
        assert np.all(np.isfinite(xs))
        assert _get_warnings(recorded) == [], (
            f'H-1 policy={pol}: unexpected resonance_range warning'
        )


# ============================================================
# Fe-56 (positive background): warning still fires when the
# query touches the RRR even though the values look normal.
# ============================================================


def test_positive_background_still_warns(fe56_tendl21):
    """Fe-56 has an LRU=1 range but the MF3 background is positive
    at thermal (no negative-value symptom). The warning still fires
    because the underlying physics limitation applies: MF3 there
    is background-only, so the returned value is not the physical
    cross section regardless of its sign."""
    if not get_resolved_resonance_ranges(fe56_tendl21):
        pytest.skip('Fe-56 in this file has no LRU=1 range')
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        xs = get_reaction_xs(fe56_tendl21, '(n,total)', IN_RRR_EINC)
    assert np.all(np.isfinite(xs))
    assert len(_get_warnings(recorded)) == 1


# ============================================================
# The other XS-emitting APIs also route through the policy.
# ============================================================


def test_particle_production_xs_carries_policy(cu63_jendl5):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        xs = get_particle_production_xs(
            cu63_jendl5, '(n,total)', 'n', IN_RRR_EINC,
            resonance_range='nan',
        )
    # xs is a per-MT-summed particle-production. Every contributor
    # comes from MF3-scaled evaluators, so in-RRR points are NaN.
    assert np.all(np.isnan(xs) | (xs == 0.0))  # some MTs contribute 0
    assert _get_warnings(recorded) == []


def test_particle_production_dxs_dE_carries_policy(cu63_jendl5):
    einc_mixed = np.array([1.0, 100.0, 2e6, 1e7], dtype=float)
    eouts = np.linspace(1e5, 1.4e7, 15)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        r = get_particle_production_dxs_dE(
            cu63_jendl5, '(n,total)', 'n', einc_mixed, eouts,
            resonance_range='warn',
        )
    assert r is not None
    # Above-RRR rows should have finite non-negative content
    above_rows = r[2:]
    assert np.all(np.isfinite(above_rows))
    assert len(_get_warnings(recorded)) >= 1


# ============================================================
# Leaf helper works outside the top-level context.
# ============================================================


def test_compute_cross_section_direct_call_with_policy(cu63_jendl5):
    """When someone bypasses the top-level API and calls
    `compute_cross_section` directly, the explicit
    `resonance_range=` kwarg takes precedence and works without
    an active context."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        xs = compute_cross_section(
            cu63_jendl5, 2, IN_RRR_EINC, resonance_range='nan',
        )
    assert np.all(np.isnan(xs))
    assert _get_warnings(recorded) == []


def test_compute_cross_section_direct_call_per_call_warning(cu63_jendl5):
    """Outside `resonance_range_ctx`, the leaf falls back to a
    per-call warning so the caller still sees a signal."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        compute_cross_section(
            cu63_jendl5, 2, IN_RRR_EINC, resonance_range='warn',
        )
    assert len(_get_warnings(recorded)) == 1
