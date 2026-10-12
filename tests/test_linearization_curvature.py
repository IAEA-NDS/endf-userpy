"""Tests for the curvature-equidistribution linearisation strategy.

Split in two tiers so the suite stays useful without the optional
pieces:

* **Pure-numpy unit tests** of the monitor and the equidistribution
  machinery, on synthetic functions whose second derivative is known
  in closed form. These need neither jax nor the corpus, and they are
  where the mathematical claim (``h^2/8 |f''|`` inverted into an
  equidistributed knot density) is actually pinned.
* **Integration tests** on the Nb-93 corpus file, which need jax in
  64-bit mode and skip cleanly otherwise.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_userpy import linearization_curvature as lc


# ----------------------------------------------------------------------
# Tier 1: synthetic, pure numpy
# ----------------------------------------------------------------------


def _lorentzian(x, x0=0.0, hw=1.0, amp=1.0):
    """``amp / (1 + u^2)`` with ``u = (x - x0)/hw``, plus its exact
    second derivative ``amp/hw^2 * (6u^2 - 2)/(1 + u^2)^3``."""
    u = (x - x0) / hw
    f = amp / (1.0 + u ** 2)
    fpp = amp / hw ** 2 * (6.0 * u ** 2 - 2.0) / (1.0 + u ** 2) ** 3
    return f, fpp


def test_monitor_is_zero_where_curvature_vanishes():
    """A straight line is represented exactly by its endpoints, so the
    monitor must not ask for interior knots."""
    x = np.linspace(0.0, 10.0, 101)
    f = 3.0 * x + 1.0
    fpp = np.zeros_like(x)
    m = lc._monitor(f, fpp, tol_abs=1e-6, tol_rel=0.0)
    assert np.all(m == 0.0)
    knots, phi, clipped = lc._equidistribute(x, m, max_points=1000)
    assert phi == 0.0
    assert not clipped
    np.testing.assert_allclose(knots, [0.0, 10.0])


def test_monitor_scales_as_sqrt_curvature_over_tol():
    """``M = sqrt(|f''| / (8 tol))`` -- pin the constant, not just the
    proportionality, since the ``1/8`` is what ties the monitor to the
    linear-interpolation error identity."""
    f = np.array([1.0, 1.0])
    fpp = np.array([8.0, 32.0])
    m = lc._monitor(f, fpp, tol_abs=1.0, tol_rel=0.0)
    np.testing.assert_allclose(m, [1.0, 2.0])
    # Doubling the tolerance divides the density by sqrt(2).
    m2 = lc._monitor(f, fpp, tol_abs=2.0, tol_rel=0.0)
    np.testing.assert_allclose(m2, np.array([1.0, 2.0]) / np.sqrt(2.0))


def test_equidistribution_meets_tolerance_on_lorentzian():
    """The headline mathematical claim, on a function whose ``f''`` is
    exact: equidistributing the monitor yields a mesh whose linear
    interpolant is within tolerance everywhere.

    Checked against a dense independent grid, not at the knots.
    """
    tol = 1e-3
    lo, hi = -50.0, 50.0
    pilot = np.unique(np.concatenate([
        np.linspace(lo, hi, 2001),
        np.linspace(-5.0, 5.0, 4001),      # resolve the peak
    ]))
    f_p, fpp_p = _lorentzian(pilot)
    m = lc._monitor(f_p, fpp_p, tol_abs=tol, tol_rel=0.0)
    mesh, phi, clipped = lc._equidistribute(pilot, m, max_points=10_000_000)
    assert not clipped
    assert phi > 1.0

    sigma, _ = _lorentzian(mesh)
    dense = np.linspace(lo, hi, 400_001)
    truth, _ = _lorentzian(dense)
    err = np.max(np.abs(truth - np.interp(dense, mesh, sigma)))
    # The bound samples f'' at knots rather than bounding it over each
    # panel, so allow a small constant; the point is that the error
    # lands at the requested scale, not one or two decades above it.
    assert err < 3.0 * tol, f'max error {err:.3e} vs tol {tol:.3e}'


def test_equidistribution_point_count_scales_as_inverse_sqrt_tol():
    """``N ~ tol^(-1/2)`` is the signature of the equidistribution
    principle. A driver that silently fell back to uniform refinement
    would scale differently."""
    lo, hi = -50.0, 50.0
    pilot = np.unique(np.concatenate([
        np.linspace(lo, hi, 2001), np.linspace(-5.0, 5.0, 4001),
    ]))
    f_p, fpp_p = _lorentzian(pilot)
    sizes = {}
    for tol in (1e-2, 1e-3, 1e-4):
        m = lc._monitor(f_p, fpp_p, tol_abs=tol, tol_rel=0.0)
        mesh, _, _ = lc._equidistribute(pilot, m, max_points=10_000_000)
        sizes[tol] = mesh.size
    for coarse, fine in ((1e-2, 1e-3), (1e-3, 1e-4)):
        ratio = sizes[fine] / sizes[coarse]
        assert 2.8 < ratio < 3.5, (
            f'tol {coarse}->{fine} gave point ratio {ratio:.2f}, '
            f'expected ~3.16 (= sqrt(10))'
        )


def test_equidistribute_respects_point_budget():
    x = np.linspace(0.0, 10.0, 1001)
    f, fpp = _lorentzian(x, x0=5.0, hw=0.01, amp=100.0)
    m = lc._monitor(f, fpp, tol_abs=1e-9, tol_rel=0.0)
    mesh, phi, clipped = lc._equidistribute(x, m, max_points=500)
    assert clipped
    assert mesh.size <= 500
    assert phi > 500


def test_equidistribute_handles_flat_monitor_stretches():
    """Zero-curvature stretches make ``Phi`` flat; without the
    monotonicity repair ``np.interp`` would collapse every target in
    that stretch onto one abscissa and the mesh would contain
    duplicate knots."""
    x = np.linspace(0.0, 30.0, 3001)
    f, fpp = _lorentzian(x, x0=15.0, hw=0.05, amp=10.0)
    # Hard-zero the curvature on the outer thirds.
    fpp = np.where((x > 10.0) & (x < 20.0), fpp, 0.0)
    m = lc._monitor(f, fpp, tol_abs=1e-4, tol_rel=0.0)
    mesh, _, _ = lc._equidistribute(x, m, max_points=1_000_000)
    assert mesh.size == np.unique(mesh).size, 'duplicate knots emitted'
    assert np.all(np.diff(mesh) > 0)


# ----------------------------------------------------------------------
# Tier 1b: argument validation (no jax, no corpus)
# ----------------------------------------------------------------------


@pytest.mark.parametrize('kwargs, match', [
    (dict(e_min=0.0, e_max=10.0), 'e_min must be positive'),
    (dict(e_min=-1.0, e_max=10.0), 'e_min must be positive'),
    (dict(e_min=10.0, e_max=10.0), 'strictly less than'),
    (dict(e_min=10.0, e_max=1.0), 'strictly less than'),
    (dict(e_min=1.0, e_max=10.0, max_points=1), 'max_points must be at least 2'),
    (dict(e_min=1.0, e_max=10.0, tol_abs=0.0, tol_rel=0.0), 'must be positive'),
])
def test_invalid_arguments_rejected_before_any_work(kwargs, match):
    """Validation must fire before the evaluator is built, so a bad
    call is cheap and does not need jax or a usable dict."""
    with pytest.raises(ValueError, match=match):
        lc.linearize_curvature({}, '(n,g)', **kwargs)


def test_zero_tolerance_is_an_error_not_a_warning():
    """With both tolerances zero the monitor is unbounded and no finite
    mesh satisfies it. The bisection driver warns and returns its seed;
    here it is a hard error, because returning a mesh that cannot meet
    the stated contract is worse than refusing."""
    with pytest.raises(ValueError, match='tol_abs / tol_rel'):
        lc.linearize_curvature({}, '(n,g)', 1.0, 10.0, tol_abs=0.0, tol_rel=0.0)


# ----------------------------------------------------------------------
# Tier 2: integration on the corpus (needs jax in x64 mode)
# ----------------------------------------------------------------------


def _jax_x64_available():
    """Constructing the JAX backend is what flips ``jax_enable_x64``
    on (``array_ns``), so instantiate it before reading the flag."""
    from endf_userpy.primitives import array_ns
    if 'jax' not in array_ns.available_backends():
        return False
    array_ns.get_backend('jax')
    import jax
    return bool(jax.config.read('jax_enable_x64'))


requires_jax_x64 = pytest.mark.skipif(
    not _jax_x64_available(),
    reason='needs jax with 64-bit mode available',
)


@pytest.fixture(scope='module')
def nb93_dict():
    from endf_parserpy import EndfParserCpp
    from _corpus import resolve_nb93
    path = resolve_nb93()
    if path is None:
        pytest.skip('Nb-93 corpus not present (tests/data_law1_adhoc/fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


# NOTE: deliberately no autouse cache purge. The evaluator key is
# (id(endf_dict), mt, chunk_size); the driver knobs (criterion,
# ref_rule, monitor_safety, split, pilot_*) do not enter it, so
# sharing compiled kernels across tests is safe and avoids a ~13 s
# recompile per corpus test. Only test_evaluator_cache_* clears it.


@requires_jax_x64
def test_requires_x64(monkeypatch):
    """float32 ``f''`` of a resonance whose width is ~1e-4 of its
    centre energy is roundoff, so the module must refuse rather than
    silently build a mesh from noise."""
    import jax
    monkeypatch.setattr(jax.config, 'read', lambda name: False)
    with pytest.raises(RuntimeError, match='64-bit'):
        lc._require_x64()


@requires_jax_x64
@pytest.mark.parametrize('reaction', ['(n,g)', '(n,total)'])
def test_evaluator_matches_user_facing_api(nb93_dict, reaction):
    """The jit evaluator re-composes MF3 + MF2 itself so it can be
    traced, which is exactly the kind of duplication that drifts. Pin
    it against ``get_reaction_xs`` to roundoff."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        report = lc.validate_against_quantities(
            nb93_dict, reaction, 1.0, 1000.0, n_sample=2000,
        )
    assert report['max_rel_diff'] < 1e-10, report


@requires_jax_x64
@pytest.mark.parametrize('reaction', ['(n,g)', '(n,total)'])
def test_mesh_meets_requested_tolerance(nb93_dict, reaction):
    """The contract: linear interpolation on the returned mesh is
    within ``tol_rel`` of the truth *between* knots, not just at them.

    Measured against ``get_reaction_xs`` on an independent dense grid.
    This is the test that the chord-bisection driver fails on
    ``(n,total)`` (it lands at 6.1e-2 for the same request), which is
    the whole reason this module exists.
    """
    from endf_userpy.quantities import get_reaction_xs
    from endf_userpy.run_options import RunOptions

    tol = 1e-3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        mesh, sigma = lc.linearize_curvature(
            nb93_dict, reaction, 1.0, 1000.0,
            tol_rel=tol, max_points=2_000_000,
        )
        verify = np.geomspace(1.0, 1000.0, 200_000)
        truth = np.asarray(get_reaction_xs(
            nb93_dict, reaction, verify,
            options=RunOptions(
                include_resonance=True, above_range='zero',
                resonance_range='nan',
            ),
        ), dtype=float)

    assert mesh[0] == 1.0 and mesh[-1] == 1000.0
    assert np.all(np.diff(mesh) > 0)
    rel = np.abs(truth - np.interp(verify, mesh, sigma)) / np.maximum(
        np.abs(truth), 1e-30,
    )
    achieved = float(np.nanmax(rel))
    assert achieved < 1.5 * tol, (
        f'{reaction}: achieved {achieved:.3e} on a {mesh.size}-point mesh, '
        f'requested {tol:.3e}'
    )


@requires_jax_x64
def test_tolerance_scaling_follows_equidistribution_theory(nb93_dict):
    """``N ~ tol^(-1/2)`` on a real file, not just the synthetic
    Lorentzian. Pins that the driver is genuinely equidistributing and
    has not degenerated into its chord-refinement fallback."""
    sizes = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for tol in (1e-3, 1e-4):
            mesh, _ = lc.linearize_curvature(
                nb93_dict, '(n,total)', 1.0, 1000.0,
                tol_rel=tol, max_points=2_000_000,
            )
            sizes[tol] = mesh.size
    ratio = sizes[1e-4] / sizes[1e-3]
    assert 2.6 < ratio < 3.8, (
        f'point ratio {ratio:.2f} for a 10x tolerance tightening; '
        f'expected ~3.16 (sizes={sizes})'
    )


@requires_jax_x64
def test_mf3_abscissae_are_mandatory_knots(nb93_dict):
    """A lin-lin MF3 panel has ``f'' = 0`` inside and a slope kink at
    each abscissa. AD reports zero curvature there, so the monitor
    would place no knot and interpolating across the kink would break
    the bound. Every in-range MF3 abscissa must survive into the mesh.
    """
    import endf_userpy.primitives.reactions as reac
    mt = reac.translate_reaction_string_to_mt('(n,total)')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ev = lc._get_evaluator(nb93_dict, mt, 4096)
        e_min, e_max = 1.0, 20_000.0
        mesh, _ = lc.linearize_curvature(
            nb93_dict, '(n,total)', e_min, e_max,
            tol_rel=1e-2, max_points=2_000_000,
        )
    knots = ev.mf3_knots
    expected = knots[(knots >= e_min) & (knots <= e_max)]
    if expected.size == 0:
        pytest.skip('no MF3/MT=1 abscissae inside the test window')
    missing = expected[~np.isin(expected, mesh)]
    assert missing.size == 0, f'{missing.size} MF3 abscissae dropped: {missing[:5]}'


@requires_jax_x64
def test_range_edges_are_doubled_points(nb93_dict):
    """MF2 range edges are genuine discontinuities, and must be
    represented the way ENDF-6 does: a repeated abscissa carrying both
    one-sided limits.

    ``resonance_composition`` masks each range with
    ``(E >= EL) & (E < EH)``, so the resonance contribution switches
    off at ``EH`` and the composed cross section steps. An earlier
    revision bracketed the edge with ``np.nextafter`` instead, which
    left a one-ULP segment straddling the jump; that segment can never
    satisfy a chord test (every interior point is still on one side,
    so the error stays at half the jump for any ``h``) and cannot even
    be subdivided, because its midpoint is not representable strictly
    inside it. The refinement looped, emitting one duplicate mesh point
    per pass. Doubling the abscissa removes the pathology and
    reproduces the jump exactly under ``np.interp``.
    """
    import endf_userpy.primitives.reactions as reac
    mt = reac.translate_reaction_string_to_mt('(n,total)')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ev = lc._get_evaluator(nb93_dict, mt, 4096)
        edges = [e for e in ev.range_edges if 1.0 < e < 20_000.0]
        if not edges:
            pytest.skip('no MF2 range edge strictly inside the window')
        mesh, sigma = lc.linearize_curvature(
            nb93_dict, '(n,total)', 1.0, 20_000.0,
            tol_rel=1e-2, max_points=2_000_000,
        )
    for edge in edges:
        hits = np.nonzero(mesh == edge)[0]
        assert hits.size == 2, (
            f'edge {edge!r} should appear exactly twice (left and right '
            f'limit); found {hits.size}'
        )
        lo, hi = sigma[hits[0]], sigma[hits[1]]
        assert lo != hi, (
            f'edge {edge!r} is doubled but both ordinates are {lo!r}; '
            f'a doubled point with equal values carries no information'
        )
        # np.interp must reproduce the jump: just below -> left limit,
        # at/above -> right limit.
        below = float(np.interp(np.nextafter(edge, -np.inf), mesh, sigma))
        assert abs(below - lo) < abs(below - hi), (
            'interpolation just below the edge should approach the '
            'left-hand limit'
        )


@requires_jax_x64
def test_evaluator_cache_reuses_compiled_kernels(nb93_dict):
    """Compiling the twice-differentiated kernel costs ~13 s, so the
    cache is load-bearing, not an optimisation detail. Two calls for
    the same (file, MT) must hand back the same object."""
    import endf_userpy.primitives.reactions as reac
    mt = reac.translate_reaction_string_to_mt('(n,g)')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        first = lc._get_evaluator(nb93_dict, mt, 4096)
        second = lc._get_evaluator(nb93_dict, mt, 4096)
    assert first is second
    lc.clear_evaluator_cache()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        third = lc._get_evaluator(nb93_dict, mt, 4096)
    assert third is not first


@requires_jax_x64
def test_nl_max_capped_to_file_max_l(nb93_dict):
    """``nl_max`` is capped at the file's own maximum L so the
    penetration-factor Newton recurrence is not unrolled past what the
    physics needs. Values at or below ``_HIGH_L_START`` skip the
    recurrence entirely, so this must never exceed it for a low-L
    file, and must never fall below the max L present."""
    import endf_userpy.primitives.reactions as reac
    mt = reac.translate_reaction_string_to_mt('(n,total)')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ev = lc._get_evaluator(nb93_dict, mt, 4096)
    mlbw = [r for r in ev._ranges if 'nl_max' in r[5]]
    if not mlbw:
        pytest.skip('no MLBW range in this file')
    for _, data, _, _, _, kwargs in mlbw:
        max_l = int(max(np.max(np.asarray(data.res_l)),
                        np.max(np.asarray(data.ch_l))))
        assert kwargs['nl_max'] >= max_l + 1
        assert kwargs['nl_max'] >= 6


@requires_jax_x64
def test_sigma_consistent_with_mesh_at_knots(nb93_dict):
    """The returned ``sigma`` must be the cross section *at* the
    returned mesh points -- a merge bug that misaligned the two arrays
    would still produce a plausible-looking mesh."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        mesh, sigma = lc.linearize_curvature(
            nb93_dict, '(n,g)', 1.0, 500.0, tol_rel=1e-2,
            max_points=2_000_000,
        )
        import endf_userpy.primitives.reactions as reac
        ev = lc._get_evaluator(
            nb93_dict, reac.translate_reaction_string_to_mt('(n,g)'), 4096,
        )
        direct = ev.value(mesh)
    np.testing.assert_allclose(sigma, direct, rtol=1e-12, atol=0.0)


# ----------------------------------------------------------------------
# Tier 1c: estimator and tolerance machinery (synthetic, no jax/corpus)
# ----------------------------------------------------------------------


def test_hermite_residual_reduces_to_the_chord_estimate_at_midpoint():
    """``chord``, ``hermite`` and ``hermite_fh`` are the SAME estimator
    to leading order, which is why they measure as interchangeable.

    For the cubic Hermite interpolant ``H`` through ``(f, f')`` at both
    knots, ``H - L`` at ``t = 1/2`` equals ``h(f'_a - f'_b)/8``, which
    is ``-h^2/8 f''`` -- exactly the classical midpoint chord error.
    Pin that identity on a quadratic, where ``f''`` is exact.
    """
    a, b = 0.0, 2.0
    h = b - a
    f = lambda x: 3.0 * x ** 2          # f'' = 6 everywhere
    fp = lambda x: 6.0 * x
    est, x_at = lc._hermite_residual_seg(
        np.array([a]), np.array([b]),
        np.array([f(a)]), np.array([f(b)]),
        np.array([fp(a)]), np.array([fp(b)]),
    )
    # classical midpoint chord error for a quadratic: h^2/8 * |f''|
    expected = h ** 2 / 8.0 * 6.0
    np.testing.assert_allclose(est[0], expected, rtol=1e-12)
    # and for constant curvature the worst point IS the midpoint
    np.testing.assert_allclose(x_at[0], 0.5 * (a + b), rtol=1e-12)


def test_monitor_safety_scales_density_as_sqrt():
    """``monitor_safety`` divides the tolerance used to build the
    monitor, so the monitor -- and hence the requested knot count --
    grows as ``sqrt(safety)``."""
    f = np.array([1.0, 1.0])
    fpp = np.array([8.0, 8.0])
    m1 = lc._monitor(f, fpp, tol_abs=1.0, tol_rel=0.0, safety=1.0)
    m2 = lc._monitor(f, fpp, tol_abs=1.0, tol_rel=0.0, safety=2.0)
    m4 = lc._monitor(f, fpp, tol_abs=1.0, tol_rel=0.0, safety=4.0)
    np.testing.assert_allclose(m2 / m1, np.sqrt(2.0), rtol=1e-12)
    np.testing.assert_allclose(m4 / m1, 2.0, rtol=1e-12)


@pytest.mark.parametrize('bad', [
    dict(criterion='nope'), dict(split='nope'), dict(ref_rule='nope'),
    dict(pilot_strategy='nope'),
])
def test_invalid_knob_values_rejected(bad):
    with pytest.raises(ValueError):
        lc.linearize_curvature({}, '(n,g)', 1.0, 10.0, **bad)


# ----------------------------------------------------------------------
# Tier 2b: discontinuity handling (needs jax + corpus)
# ----------------------------------------------------------------------


@pytest.fixture(scope='module')
def u235_dict():
    from endf_parserpy import EndfParserCpp
    from _corpus import resolve_u235
    path = resolve_u235()
    if path is None:
        pytest.skip('U-235 corpus not present (tests/data_law1_adhoc/fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@requires_jax_x64
def test_discontinuity_energies_collects_both_sources(u235_dict):
    """Jumps come from two places, and both must be found: MF3 doubled
    abscissae (the ENDF-6 discontinuity convention, by far the most
    common -- every resonance evaluation in the corpus has some) and
    MF2 range edges."""
    import endf_userpy.primitives.reactions as reac
    from endf_userpy.primitives import tab1 as tab1_mod
    mt = reac.translate_reaction_string_to_mt('(n,n_0)')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ev = lc._get_evaluator(u235_dict, mt, 1024)
        disc = lc._discontinuity_energies(ev, 1.0, 2250.0)

    tab = tab1_mod.from_endf_dict(u235_dict[3][mt])
    x = np.asarray(tab.x, dtype=float)
    expected_mf3 = x[:-1][np.diff(x) == 0.0]
    expected_mf3 = expected_mf3[(expected_mf3 >= 1.0) & (expected_mf3 <= 2250.0)]
    for e in expected_mf3:
        assert np.any(disc == e), f'MF3 doubled abscissa {e!r} not detected'
    for edge in ev.range_edges:
        if 1.0 <= edge <= 2250.0:
            assert np.any(disc == edge), f'MF2 range edge {edge!r} not detected'


@requires_jax_x64
def test_doubled_points_reproduce_the_jump_and_converge(u235_dict):
    """The U-235 regression: ``EH = 2250`` carries a real 0.32 barn
    step (the URR average does not meet the RRR endpoint), and MF3/MT2
    additionally has a doubled abscissa there.

    Before doubled points this produced nine zero-width segments, a
    permanent ``residual_error`` and duplicate abscissae in the
    returned mesh. All three must now be gone, while the jump itself is
    still represented.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        r = lc.linearize_curvature(
            u235_dict, '(n,n_0)', 1.0, 2250.0, tol_rel=1e-3,
            chunk_size=1024, max_points=4_000_000, refine_passes=12,
            return_diagnostics=True,
        )
    assert r.status == 'converged', r.status
    assert r.n_unrefinable == 0, (
        f'{r.n_unrefinable} segment(s) could not be subdivided; the '
        f'doubled-point representation should have removed these'
    )
    # mesh is sorted, and every repeated abscissa carries a real jump
    assert np.all(np.diff(r.mesh) >= 0)
    dup = np.nonzero(np.diff(r.mesh) == 0.0)[0]
    assert dup.size >= 1, 'expected at least the EH=2250 discontinuity'
    jumps = np.abs(np.diff(r.sigma))[dup]
    assert np.all(jumps > 0), (
        'a duplicated abscissa whose two ordinates agree is noise and '
        'should have been removed by the value-aware dedupe'
    )


@requires_jax_x64
def test_mesh_has_no_information_free_duplicates(u235_dict):
    """Duplicates that encode a jump are kept; duplicates whose
    ordinates agree carry nothing and must be dropped, since
    ``np.interp`` on a repeated x with equal y is pure noise."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        mesh, sigma = lc.linearize_curvature(
            u235_dict, '(n,g)', 1.0, 2250.0, tol_rel=1e-2,
            chunk_size=1024, max_points=4_000_000,
        )
    same_x = np.diff(mesh) == 0.0
    same_y = np.diff(sigma) == 0.0
    assert not np.any(same_x & same_y), (
        'mesh contains duplicate abscissae with identical ordinates'
    )


# ----------------------------------------------------------------------
# Tier 2c: criterion / denominator / pilot equivalences
# ----------------------------------------------------------------------


@requires_jax_x64
def test_chord_and_hermite_criteria_agree(nb93_dict):
    """``chord``, ``hermite`` and ``hermite_fh`` are the same estimator
    to leading order (see the midpoint identity above), so they must
    produce near-identical meshes. Measured on Nb-93 they agree to
    better than 1 percent in point count.

    This pins a finding that cost real effort: an earlier comparison
    appeared to show ``hermite`` was markedly worse, but that was an
    artifact of scaling the relative tolerance by a criterion-dependent
    reference value. With one denominator the three coincide.
    """
    sizes = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for crit in ('chord', 'hermite', 'hermite_fh'):
            mesh, _ = lc.linearize_curvature(
                nb93_dict, '(n,g)', 1.0, 200.0, tol_rel=1e-3,
                criterion=crit, max_points=2_000_000,
            )
            sizes[crit] = mesh.size
    lo, hi = min(sizes.values()), max(sizes.values())
    assert (hi - lo) / lo < 0.02, (
        f'criteria should be interchangeable, got {sizes}'
    )


@requires_jax_x64
def test_ref_rule_min_is_stricter_than_max(nb93_dict):
    """The relative tolerance is scored against sigma *inside* a
    segment, so scaling by the larger endpoint is optimistic. ``min``
    is the conservative choice and must never yield a coarser mesh."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        mesh_max, _ = lc.linearize_curvature(
            nb93_dict, '(n,total)', 1.0, 200.0, tol_rel=1e-3,
            ref_rule='max', max_points=2_000_000)
        mesh_min, _ = lc.linearize_curvature(
            nb93_dict, '(n,total)', 1.0, 200.0, tol_rel=1e-3,
            ref_rule='min', max_points=2_000_000)
    assert mesh_min.size >= mesh_max.size, (
        f'min denominator gave a coarser mesh ({mesh_min.size}) than max '
        f'({mesh_max.size}), which cannot be right'
    )


@requires_jax_x64
@pytest.mark.parametrize('pilot', ['uniform_gamma', 'territory_log'])
def test_both_pilot_strategies_meet_tolerance(nb93_dict, pilot):
    """The pilot only has to *resolve* the monitor, so either placement
    rule must land inside tolerance. ``territory_log`` is the default
    because it reaches the same monitor integral with far fewer points,
    not because the other one is wrong."""
    from endf_userpy.quantities import get_reaction_xs
    from endf_userpy.run_options import RunOptions
    tol = 1e-3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        mesh, sigma = lc.linearize_curvature(
            nb93_dict, '(n,g)', 1.0, 500.0, tol_rel=tol,
            pilot_strategy=pilot, max_points=2_000_000)
        verify = np.geomspace(1.0, 500.0, 200_000)
        truth = np.asarray(get_reaction_xs(
            nb93_dict, '(n,g)', verify,
            options=RunOptions(include_resonance=True, above_range='zero',
                               resonance_range='nan')), dtype=float)
    rel = np.abs(truth - np.interp(verify, mesh, sigma)) / np.maximum(
        np.abs(truth), 1e-30)
    assert float(np.nanmax(rel)) < 1.5 * tol, (
        f'{pilot}: achieved {float(np.nanmax(rel)):.3e} on {mesh.size} points'
    )
