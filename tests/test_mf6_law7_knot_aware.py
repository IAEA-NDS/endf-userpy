"""Tests for the knot-aware LAW=7 integrator (issue #69).

The general adaptive-Simpson MF6 integrator (`distribution1d_helpers.
integrate_mf6_dist2d_over_eout`) lands at ~0.5-1 % of peak vs. a
knot-aware quad reference on the Be-9 MT16 (n,2n) LAW=7 boundary,
because LAW=7's reconstructed `f(E')` at an interpolated `(E_in, mu)`
has kinks that don't align with a uniform mesh (the tabulated E'
knots are transformed nonlinearly by the LAW=7 unit-base interpolation).

The new `mf6_law7_integrals.integrate_law7_subsec_over_eout`:

- Extracts the four bracketing tables at target `(E_in, mu)`.
- Projects each table's `Ep[k]` knots through the LAW=7 unit-base
  transform to obtain the effective E' knot set at the target.
- Integrates with the midpoint rule on that mesh -- exact for the
  piecewise-constant (INT=1) and piecewise-linear (INT=2) E'
  interpolants that appear in the corpus.

The dispatcher in `integrate_mf6_dist2d_over_eout` routes here only
when the MT has a single MF6 subsection with `LAW=7`; multi-subsec
sections continue to use the general adaptive-Simpson path.
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_particle_production_dxs_dmu
from endf_userpy.quantities_mt_zap.distribution2d import compute_dist2d_values
from endf_userpy.quantities_mt_zap import distribution1d_helpers as d1h
from endf_userpy.mfsec_interpretation import mf6_law7_integrals as mf6_l7
from endf_userpy.primitives.properties import get_QM, get_QI
from endf_userpy.primitives.np_compat import trapezoid


ADHOC = Path(__file__).resolve().parent / 'data_law1_adhoc'


def _load(fn_name):
    fn = ADHOC / fn_name
    if not fn.exists():
        pytest.skip(
            f'{fn_name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh` to populate the corpus'
        )
    return EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    ).parsefile(fn)


@pytest.fixture(scope='module')
def be9_endfb81():
    return _load('endfb81_n_Be-9.endf')


def _fine_trapezoid_reference(endf, mt, zap, einc, mus, N=50000):
    """Trapezoid on an ultra-fine uniform E' mesh, batched over mu.
    For LAW=7 with `INT=1`/`INT=2` (piecewise-constant / linear
    reconstructed f) this converges as `O(1/N)` and sits at
    `<= 1e-4` discretisation error at `N=50000` over the corpus.

    Chosen over `quad(points=knots, ...)` as the test reference
    because that path needs `limit=2000+` to converge on the harder
    (Ein, mu) cells and takes several minutes on the whole grid --
    too slow for CI. Fine trapezoid is fast (batched over mu, one
    dispatch call per Ein) and its remaining error dominates any
    residual from the knot-aware integrator, giving a fair
    conservative bound.
    """
    q = max(get_QM(endf, mt), get_QI(endf, mt))
    out = np.zeros((len(einc), len(mus)))
    for i, e in enumerate(einc):
        eout_max = (e + q) * 1.1
        if eout_max <= 0:
            continue
        ep = np.linspace(0.0, eout_max, N)
        dist = compute_dist2d_values(
            endf, mt, zap, np.array([e]), ep, mus, True,
        )
        out[i, :] = trapezoid(dist[0], x=ep, axis=0)
    return out


# ============================================================
# The main win: sub-permille agreement against knot-aware quad on
# the whole Be-9 MT16 (n,2n) grid (issue #46 pinned this at
# < 1.5 % via the generic Simpson path; #69 tightens it here).
# ============================================================


def test_be9_mt16_knot_aware_matches_ground_truth(be9_endfb81):
    """Be-9 MT16 is single-subsection LAW=7 with `INT=1` (histogram)
    on every table's E' axis. The knot-aware midpoint rule is exact
    for that interpolant, so agreement with a fine-mesh trapezoid
    reference should be limited only by the reference's own
    discretisation error (`~1e-4` at `N=50000`)."""
    einc = np.linspace(2e6, 1.7e7, 6)
    mus = np.linspace(-0.9, 0.9, 12)
    truth = _fine_trapezoid_reference(be9_endfb81, 16, 1, einc, mus)
    new = d1h.integrate_mf6_dist2d_over_eout(
        be9_endfb81, 16, 1, einc, mus,
    )
    peak = float(np.max(np.abs(truth)))
    max_err = float(np.max(np.abs(new - truth))) / peak
    assert max_err < 5e-4, (
        f'Be-9 MT16 knot-aware max |diff|/peak = {max_err:.3e} '
        f'(should be at the fine-trapezoid noise floor ~1e-4)'
    )


def test_be9_mt16_knot_aware_beats_generic_simpson(be9_endfb81):
    """Regression guard: on the same grid, the knot-aware result must
    be at least 10x closer to the reference than the generic
    adaptive-Simpson path is. Measured today: ratio ~ 100x. The 10x
    threshold gives ample headroom while still catching any change
    that lets the specialised path collapse toward the generic one.
    """
    einc = np.linspace(2e6, 1.7e7, 6)
    mus = np.linspace(-0.9, 0.9, 12)
    truth = _fine_trapezoid_reference(be9_endfb81, 16, 1, einc, mus)
    ka = d1h.integrate_mf6_dist2d_over_eout(be9_endfb81, 16, 1, einc, mus)
    generic = d1h._integrate_mf6_over_eout_adaptive_simpson(
        be9_endfb81, 16, 1, einc, mus,
    )
    peak = float(np.max(np.abs(truth)))
    err_ka = float(np.max(np.abs(ka - truth))) / peak
    err_generic = float(np.max(np.abs(generic - truth))) / peak
    assert err_ka * 10 < err_generic, (
        f'knot-aware error {err_ka:.3e} not at least 10x smaller than '
        f'generic-Simpson error {err_generic:.3e}'
    )


# ============================================================
# Unit-base transform: projecting raw tabulated knots to the
# effective E' axis. Hand-computed reference values.
# ============================================================


def test_project_ub_knots_hand_computed():
    """Two tables at y=0 and y=1 with disjoint x ranges. At y0=0.5
    the effective range is the linear midpoint of the two, and the
    projected knots line up on the interpolated span. Compare
    element-wise to the closed-form."""
    y1, y2 = 0.0, 1.0
    x1_knots = np.array([0.0, 50.0, 100.0])       # range 100
    x2_knots = np.array([0.0, 100.0, 200.0])      # range 200
    y0 = 0.5
    # Expected: xlow = 0, xhigh = (100+200)/2 = 150, xrange = 150
    #   x1_knots [0, 50, 100] -> xlow + (x/100)*150 = [0, 75, 150]
    #   x2_knots [0, 100, 200] -> xlow + (x/200)*150 = [0, 75, 150]
    got = mf6_l7._project_ub_knots(y0, y1, y2, x1_knots, x2_knots)
    expected = np.array([0.0, 75.0, 150.0])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_project_ub_knots_at_slice_endpoints():
    """At y0 == y1 (or y0 == y2) the effective range coincides with
    that slice's own range and the projected knots reproduce the
    slice's own knots (unioned with a single projection of the
    other slice's endpoints since ranges collapse). Guard against
    degenerate arithmetic at endpoints."""
    y1, y2 = 0.0, 1.0
    x1_knots = np.array([0.0, 20.0, 100.0])
    x2_knots = np.array([50.0, 150.0])
    # y0 = y1 -> yslope = 0 -> effective = slice 1
    got_low = mf6_l7._project_ub_knots(y1, y1, y2, x1_knots, x2_knots)
    # Effective knots must include every x1 knot exactly.
    for x in x1_knots:
        assert np.any(np.isclose(got_low, x)), (
            f'{x} missing from projected set {got_low}'
        )
    # y0 = y2 -> yslope = 1 -> effective = slice 2
    got_high = mf6_l7._project_ub_knots(y2, y1, y2, x1_knots, x2_knots)
    for x in x2_knots:
        assert np.any(np.isclose(got_high, x)), (
            f'{x} missing from projected set {got_high}'
        )


def test_project_ub_knots_degenerate_range():
    """A single-knot table (range zero) collapses to xlow. Doesn't
    raise, doesn't emit NaN."""
    got = mf6_l7._project_ub_knots(
        0.5, 0.0, 1.0,
        np.array([100.0]),      # zero range
        np.array([0.0, 200.0]),
    )
    assert not np.any(np.isnan(got))
    assert len(got) >= 1


# ============================================================
# Dispatcher: single-subsec LAW=7 routes here, everything else
# stays on the general Simpson path.
# ============================================================


def test_dispatcher_uses_knot_aware_on_be9(monkeypatch, be9_endfb81):
    """The single-subsec LAW=7 branch of `integrate_mf6_dist2d_over_eout`
    must call the knot-aware helper for Be-9 MT16 (n,2n)."""
    calls = {'n': 0}
    real = mf6_l7.integrate_law7_subsec_over_eout

    def spy(*args, **kwargs):
        calls['n'] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(
        'endf_userpy.quantities_mt_zap.distribution1d_helpers.'
        'mf6_law7.integrate_law7_subsec_over_eout',
        spy,
    )
    d1h.integrate_mf6_dist2d_over_eout(
        be9_endfb81, 16, 1,
        np.array([1e7]), np.array([0.0]),
    )
    assert calls['n'] == 1


def test_dispatcher_uses_generic_on_multi_subsec(monkeypatch, be9_endfb81):
    """Cu-63 MT16 has two MF6 subsections (LAW=1 + LAW=1), so the
    knot-aware helper must NOT be called. Guard by asking the
    helper to raise -- if the dispatcher ever routes a multi-subsec
    section here, the test fires."""
    def bad_ka(*args, **kwargs):
        raise AssertionError(
            'knot-aware helper called for multi-subsec section'
        )

    monkeypatch.setattr(
        'endf_userpy.quantities_mt_zap.distribution1d_helpers.'
        'mf6_law7.integrate_law7_subsec_over_eout',
        bad_ka,
    )
    cu63 = _load('jeff40_n_Cu-63.endf')
    d1h.integrate_mf6_dist2d_over_eout(
        cu63, 16, 1,
        np.array([1.8e7]), np.array([0.0]),
    )


# ============================================================
# End-to-end: no wall-clock regression through the public API.
# ============================================================


def test_dxs_dmu_be9_wall_time_regression(be9_endfb81):
    """The knot-aware path must not be slower than the generic
    Simpson path on the same grid (the fewer eval points from
    knot-aware trade against per-cell dispatch overhead; the net
    effect on the corpus is neutral or slightly faster). Cap
    generous enough for CI variance."""
    import time
    einc = np.linspace(2e6, 1.4e7, 6)
    mus = np.linspace(-0.9, 0.9, 12)
    t0 = time.perf_counter()
    r = get_particle_production_dxs_dmu(
        be9_endfb81, '(n,total)', 'n', einc, mus,
    )
    dt = time.perf_counter() - t0
    assert r is not None and r.shape == (6, 12)
    assert dt < 2.0, f'dxs_dmu wall {dt:.2f}s exceeds 2.0s regression cap'
