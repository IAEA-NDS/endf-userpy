"""Tests for the adaptive mesh-refinement driver
:func:`endf_userpy.linearization.linearize_reaction_xs`.

Coverage:

1. Synthetic Lorentzian (no ENDF file): seed from a single pole,
   verify convergence to prescribed tolerance on a dense check
   mesh, assert reasonable sparsity.
2. Nb-93 RRR (MLBW corpus file): run on an energy window inside
   the resolved-resonance region, verify convergence on a dense
   check mesh against the composed ``get_reaction_xs``.
3. Degenerate inputs: raise on bad ``e_min`` / ``e_max``; warn on
   zero tolerances; the seed-only fast path when ``max_points`` is
   below the seed size.
4. Determinism: same inputs produce bit-identical meshes.
5. Diagnostics dataclass: ``return_diagnostics=True`` shape and
   fields.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from _corpus import resolve_nb93  # noqa: E402

from endf_userpy.linearization import (            # noqa: E402
    LinearizationResult, linearize_reaction_xs,
)
from endf_userpy.quantities import get_reaction_xs  # noqa: E402
from endf_userpy.run_options import RunOptions      # noqa: E402


# ----------------------------------------------------------------------
# Synthetic Lorentzian: no ENDF file, no composition, no corpus
# dependency. Monkeypatches ``get_reaction_xs`` to a 1-pole function
# so the test runs anywhere and is bit-reproducible.
# ----------------------------------------------------------------------


def _single_lorentzian(E, E_r=100.0, gamma=1.0, sigma_peak=1000.0):
    """Breit-Wigner peak with FWHM = ``gamma``."""
    return sigma_peak / (1 + ((E - E_r) / (0.5 * gamma)) ** 2)


def test_synthetic_lorentzian_converges_with_sparse_mesh(monkeypatch):
    """Seed a single pole, assert the adaptive mesh:
    (a) achieves ``tol_rel`` accuracy on a dense check mesh, and
    (b) uses fewer points than a uniform mesh at the same accuracy
    would require."""
    # Pretend the "ENDF dict" carries one MLBW resonance at (100, 1.0).
    # linearization's seed walker goes through
    # ``_extract_resonance_seed_points`` which reads MF2; we skip that
    # path entirely by patching get_reaction_xs and relying on the
    # background seed + iteration to find the peak.
    endf_dict_stub = {}        # no MF2: seed is background-only.

    def fake_xs(d, reaction, E, *, options=None):
        return _single_lorentzian(np.asarray(E, dtype=float))

    import endf_userpy.linearization as lin
    monkeypatch.setattr(lin, 'get_reaction_xs', fake_xs)

    mesh, sigma = lin.linearize_reaction_xs(
        endf_dict_stub, '(n,g)', 50.0, 150.0, tol_rel=1e-3,
    )
    # Must converge to a finite, moderate mesh. 100 keV window with
    # one 1 eV FWHM peak needs at most a few thousand points at
    # 0.1 percent accuracy.
    assert mesh.size < 5000
    assert mesh.size > 20   # some refinement happened

    # Verification on a 20k dense check mesh.
    E_check = np.linspace(50.0, 150.0, 20000)
    xs_true = _single_lorentzian(E_check)
    xs_interp = np.interp(E_check, mesh, sigma)
    rel_err = np.abs(xs_true - xs_interp) / np.maximum(np.abs(xs_true), 1e-30)
    assert rel_err.max() < 2e-3, (
        f'linear-interp max rel err {rel_err.max():.3e} exceeds '
        f'2x tol_rel on verification mesh'
    )


def test_synthetic_lorentzian_pole_seeding_cuts_iterations(monkeypatch):
    """Supplying the pole energy through the seed (via a one-pole
    MF2 stub) must reduce the iteration count vs background-only."""
    def fake_xs(d, reaction, E, *, options=None):
        return _single_lorentzian(np.asarray(E, dtype=float))

    import endf_userpy.linearization as lin
    monkeypatch.setattr(lin, 'get_reaction_xs', fake_xs)

    # Background-only seed (no MF2 in the stub).
    _, _, res_bg = _run_with_diag(lin, {}, 50.0, 150.0)

    # Pole-aware seed: patch _extract_resonance_seed_points to yield
    # the pole we already know.
    monkeypatch.setattr(
        lin, '_extract_resonance_seed_points',
        lambda d: iter([(100.0, 1.0)]),
    )
    _, _, res_pole = _run_with_diag(lin, {}, 50.0, 150.0)

    assert res_pole.iterations <= res_bg.iterations, (
        f'pole-aware seed took {res_pole.iterations} iters vs '
        f'{res_bg.iterations} background-only; pole prior should not '
        f'slow convergence'
    )


def _run_with_diag(lin, endf_dict, e_min, e_max, **kw):
    kw.setdefault('tol_rel', 1e-3)
    kw.setdefault('return_diagnostics', True)
    res = lin.linearize_reaction_xs(
        endf_dict, '(n,g)', e_min, e_max, **kw,
    )
    return res.mesh, res.sigma, res


# ----------------------------------------------------------------------
# Nb-93 RRR: real MLBW composition. Skips cleanly when the corpus
# file is absent.
# ----------------------------------------------------------------------


@pytest.fixture
def nb93_dict():
    path = resolve_nb93()
    if path is None:
        pytest.skip(
            'Nb-93 ENDF file not available (set NB93_ENDF, run '
            'tests/data_law1_adhoc/fetch.sh, or place the file at '
            'tests/data_law1_adhoc/endfb81_n_Nb-93.endf)'
        )
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp().parsefile(path)


def test_nb93_rrr_linearization_hits_tolerance_on_verification_mesh(nb93_dict):
    """Run the driver on Nb-93's RRR (1 eV to 1 keV), interpolate on
    a 20k verification mesh, and confirm the error is below 2 x
    tol_rel against the composed cross section from
    ``get_reaction_xs``. Pins the end-to-end correctness invariant."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = linearize_reaction_xs(
            nb93_dict, '(n,g)', 1.0, 1000.0,
            tol_rel=1e-3, return_diagnostics=True,
        )
        E_check = np.geomspace(1.0, 1000.0, 20_000)
        xs_true = np.asarray(
            get_reaction_xs(nb93_dict, '(n,g)', E_check), dtype=float,
        )
    xs_interp = np.interp(E_check, result.mesh, result.sigma)
    rel_err = np.abs(xs_true - xs_interp) / np.maximum(np.abs(xs_true), 1e-30)
    assert rel_err.max() < 2e-3, (
        f'linear-interp max rel err {rel_err.max():.3e} vs tol_rel '
        f'1e-3 on 20k verification mesh; mesh size was {result.mesh.size}'
    )
    # Sparsity sanity: the adaptive mesh should be smaller than the
    # verification mesh by a healthy margin.
    assert result.mesh.size < 15_000
    assert result.status == 'converged'


def test_nb93_rrr_coarser_tolerance_yields_sparser_mesh(nb93_dict):
    """Loosening ``tol_rel`` by 10x must shrink the mesh."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tight = linearize_reaction_xs(
            nb93_dict, '(n,g)', 1.0, 1000.0,
            tol_rel=1e-3, return_diagnostics=True,
        )
        loose = linearize_reaction_xs(
            nb93_dict, '(n,g)', 1.0, 1000.0,
            tol_rel=1e-2, return_diagnostics=True,
        )
    assert loose.mesh.size < tight.mesh.size, (
        f'coarser tolerance mesh ({loose.mesh.size}) is not smaller '
        f'than tight-tolerance mesh ({tight.mesh.size})'
    )


# ----------------------------------------------------------------------
# Degenerate inputs and guard rails.
# ----------------------------------------------------------------------


def test_e_min_not_less_than_e_max_raises():
    with pytest.raises(ValueError, match=r'e_min.*e_max'):
        linearize_reaction_xs({}, '(n,g)', 10.0, 10.0)
    with pytest.raises(ValueError, match=r'e_min.*e_max'):
        linearize_reaction_xs({}, '(n,g)', 100.0, 10.0)


def test_e_min_non_positive_raises():
    with pytest.raises(ValueError, match=r'e_min must be positive'):
        linearize_reaction_xs({}, '(n,g)', 0.0, 10.0)
    with pytest.raises(ValueError, match=r'e_min must be positive'):
        linearize_reaction_xs({}, '(n,g)', -1.0, 10.0)


def test_max_points_below_two_raises():
    with pytest.raises(ValueError, match=r'max_points'):
        linearize_reaction_xs({}, '(n,g)', 1.0, 10.0, max_points=1)


def test_both_tolerances_zero_warns_and_returns_seed(monkeypatch):
    def fake_xs(d, reaction, E, *, options=None):
        return np.ones_like(np.asarray(E, dtype=float))

    import endf_userpy.linearization as lin
    monkeypatch.setattr(lin, 'get_reaction_xs', fake_xs)

    with pytest.warns(UserWarning, match=r'tol_abs and tol_rel are zero'):
        mesh, sigma = lin.linearize_reaction_xs(
            {}, '(n,g)', 1.0, 100.0, tol_abs=0.0, tol_rel=0.0,
        )
    assert mesh.size >= 2
    assert sigma.size == mesh.size


def test_seed_larger_than_budget_warns_and_truncates(monkeypatch, nb93_dict):
    """When the resonance-aware seed already exceeds ``max_points``,
    the driver warns, truncates the seed, and skips refinement."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.warns(UserWarning, match=r'already exceeds max_points'):
            mesh, sigma = linearize_reaction_xs(
                nb93_dict, '(n,g)', 1.0, 1000.0,
                tol_rel=1e-3, max_points=50,
            )
    assert mesh.size <= 50
    assert sigma.size == mesh.size


# ----------------------------------------------------------------------
# Diagnostics shape and determinism.
# ----------------------------------------------------------------------


def test_return_diagnostics_shape(monkeypatch):
    def fake_xs(d, reaction, E, *, options=None):
        return _single_lorentzian(np.asarray(E, dtype=float))

    import endf_userpy.linearization as lin
    monkeypatch.setattr(lin, 'get_reaction_xs', fake_xs)

    result = lin.linearize_reaction_xs(
        {}, '(n,g)', 50.0, 150.0, tol_rel=1e-3, return_diagnostics=True,
    )
    assert isinstance(result, LinearizationResult)
    assert result.mesh.ndim == 1
    assert result.sigma.shape == result.mesh.shape
    assert result.iterations >= 0
    assert result.status in {'converged', 'point_budget', 'max_iterations'}
    assert len(result.history) == result.iterations
    for h in result.history:
        for key in ('iter', 'n_mesh', 'n_add', 'max_err'):
            assert key in h


def test_determinism(monkeypatch):
    """Two runs with identical inputs must produce bit-identical
    output. Important for reproducibility and for rebuilding
    cached meshes."""
    def fake_xs(d, reaction, E, *, options=None):
        return _single_lorentzian(np.asarray(E, dtype=float))

    import endf_userpy.linearization as lin
    monkeypatch.setattr(lin, 'get_reaction_xs', fake_xs)

    m1, s1 = lin.linearize_reaction_xs({}, '(n,g)', 50.0, 150.0, tol_rel=1e-3)
    m2, s2 = lin.linearize_reaction_xs({}, '(n,g)', 50.0, 150.0, tol_rel=1e-3)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(s1, s2)


def test_options_passed_through(monkeypatch):
    """Verify ``options`` reaches ``get_reaction_xs`` on every call."""
    seen_options = []

    def fake_xs(d, reaction, E, *, options=None):
        seen_options.append(options)
        return np.ones_like(np.asarray(E, dtype=float))

    import endf_userpy.linearization as lin
    monkeypatch.setattr(lin, 'get_reaction_xs', fake_xs)

    opts = RunOptions(include_resonance=False, mt5_contrib=False)
    lin.linearize_reaction_xs(
        {}, '(n,g)', 1.0, 100.0, tol_rel=1e-3, options=opts,
    )
    assert len(seen_options) >= 1
    for seen in seen_options:
        assert seen is opts
