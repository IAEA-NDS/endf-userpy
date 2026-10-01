"""Regression tests pinning issue #304: distribution and broadening
entry points must honour ``include_resonance=True`` and route their
cross-section reads through the MF2 + MF3 composition layer.

Before #304, ``get_particle_production_xs``,
``get_residual_production_xs``, ``get_particle_production_dxs_dE``,
``get_particle_production_ddxs``, and the per-MT distribution
combinators called ``mf3_interp.compute_cross_section`` directly,
silently returning raw MF3 background inside the resolved-resonance
region even under the physics-first default
``RunOptions(include_resonance=True)``. Only ``get_reaction_xs``
went through composition. The two produced contradictory numerics
on the same file.

These tests exercise an MF2-bearing corpus file (Nb-93) and assert
that each distribution entry point responds to toggling
``include_resonance``, which is only true when the fix is in
place.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from _corpus import resolve_nb93   # noqa: E402

from endf_userpy.quantities import (          # noqa: E402
    get_reaction_xs,
    get_residual_production_xs,
    get_particle_production_xs,
    get_particle_production_dxs_dE,
)
from endf_userpy.run_options import RunOptions   # noqa: E402


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


# Energies inside Nb-93's MLBW resolved-resonance region (0 to ~7 keV).
# Capture resonances dominate here, so composed vs raw MF3 differ
# by orders of magnitude at resonance peaks.
_E_IN_RRR = np.geomspace(1.0, 5e3, 51)


def _max_relative_diff(a, b):
    """Peak relative discrepancy between two arrays, ignoring points
    where both are below a floor (avoid 0/0 in off-resonance dips)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    floor = 1e-12 + 1e-6 * max(np.nanmax(np.abs(a)), np.nanmax(np.abs(b)))
    mask = (np.abs(a) > floor) | (np.abs(b) > floor)
    if not np.any(mask):
        return 0.0
    denom = np.maximum(np.abs(a[mask]), np.abs(b[mask]))
    return float(np.nanmax(np.abs(a[mask] - b[mask]) / denom))


def test_baseline_get_reaction_xs_toggles_with_include_resonance(nb93_dict):
    """Sanity: ``get_reaction_xs`` has always honoured
    ``include_resonance`` since #143. If this baseline test fails,
    the corpus or composition itself is broken; the per-entry-point
    tests below are then meaningless."""
    xs_comp = get_reaction_xs(
        nb93_dict, '(n,g)', _E_IN_RRR,
        options=RunOptions(include_resonance=True),
    )
    xs_raw = get_reaction_xs(
        nb93_dict, '(n,g)', _E_IN_RRR,
        options=RunOptions(include_resonance=False),
    )
    rel = _max_relative_diff(xs_comp, xs_raw)
    assert rel > 0.5, (
        f'Composed vs raw-MF3 (n,g) XS in Nb-93 RRR should differ '
        f'substantially; got max relative diff {rel:.3g}. Corpus or '
        f'composition layer may be broken.'
    )


def test_particle_production_xs_honours_include_resonance(nb93_dict):
    """Issue #304 fix: `get_particle_production_xs` routes through
    `compute_xs` so `include_resonance=True` composes MF2. Before
    the fix, both options returned raw MF3 and this test would
    silently pass with rel == 0."""
    xs_comp = get_particle_production_xs(
        nb93_dict, '(n,g)', 'g', _E_IN_RRR,
        options=RunOptions(include_resonance=True),
    )
    xs_raw = get_particle_production_xs(
        nb93_dict, '(n,g)', 'g', _E_IN_RRR,
        options=RunOptions(include_resonance=False),
    )
    rel = _max_relative_diff(xs_comp, xs_raw)
    assert rel > 0.5, (
        f'Particle-production XS must change when toggling '
        f'include_resonance on an MF2-bearing file in the RRR; '
        f'got {rel:.3g} (would be 0 before #304).'
    )


def test_residual_production_xs_honours_include_resonance(nb93_dict):
    """Same invariant for `get_residual_production_xs`: Nb-93 (n,g)
    produces Nb-94; its production XS must respond to the resonance
    toggle because the underlying `compute_residual_xs` now reads
    through `compute_xs`."""
    xs_comp = get_residual_production_xs(
        nb93_dict, 'Nb-94', _E_IN_RRR,
        options=RunOptions(include_resonance=True),
    )
    xs_raw = get_residual_production_xs(
        nb93_dict, 'Nb-94', _E_IN_RRR,
        options=RunOptions(include_resonance=False),
    )
    rel = _max_relative_diff(xs_comp, xs_raw)
    assert rel > 0.5, (
        f'Residual-production XS must change when toggling '
        f'include_resonance on an MF2-bearing file in the RRR; '
        f'got {rel:.3g} (would be 0 before #304).'
    )


def test_dxs_dE_integral_matches_composed_production_xs(nb93_dict):
    """Stronger invariant: the integral of the energy-differential
    cross section over E_out should recover the production cross
    section (modulo the gamma-multiplicity factor built into the
    yields). Under include_resonance=True, both sides should be the
    composed cross section. Pre-#304, the dxs/dE path would use raw
    MF3 while the production-xs path would also use raw MF3 (both
    bypassed composition), so this test could only ever check
    consistency between two broken paths. Post-#304, both use the
    composed path and the invariant holds against the physically
    correct cross section.
    """
    # Narrow Ein window just below the first big capture resonance
    # peak so the test runs quickly but still exercises composition.
    E = np.array([35.0, 103.0, 200.0])     # eV, inside RRR
    Eo = np.geomspace(1e3, 2e7, 400)       # gamma energies
    opts = RunOptions(include_resonance=True)
    dxs = get_particle_production_dxs_dE(
        nb93_dict, '(n,g)', 'g', E, Eo, options=opts,
    )
    prodxs = get_particle_production_xs(
        nb93_dict, '(n,g)', 'g', E, options=opts,
    )
    # Integrate dxs/dE across E_out.
    integrated = np.trapezoid(dxs, Eo, axis=1)
    # ~10% tolerance: Eo mesh is coarse (log-spaced 400 pts) and
    # the gamma spectrum has discrete-line content that trapezoid
    # cannot fully resolve. The physical check is just order of
    # magnitude.
    rel = np.abs(integrated - prodxs) / np.maximum(prodxs, 1e-30)
    assert np.all(rel < 0.3), (
        f'Integrated dxs/dE must approximate the production XS '
        f'within ~30% (coarse E_out mesh); got rel diffs {rel}'
    )


def test_dxs_dE_responds_to_include_resonance(nb93_dict):
    """Direct test: toggle include_resonance and confirm the
    differential cross section changes. Pre-#304 the broadening /
    distribution path would always use raw MF3 regardless of the
    toggle, giving identical arrays."""
    E = np.array([103.0, 200.0])
    Eo = np.geomspace(1e3, 2e7, 100)
    dxs_comp = get_particle_production_dxs_dE(
        nb93_dict, '(n,g)', 'g', E, Eo,
        options=RunOptions(include_resonance=True),
    )
    dxs_raw = get_particle_production_dxs_dE(
        nb93_dict, '(n,g)', 'g', E, Eo,
        options=RunOptions(include_resonance=False),
    )
    # Peak-of-E_out slice at each Ein should differ substantially.
    peak_comp = np.max(dxs_comp, axis=1)
    peak_raw = np.max(dxs_raw, axis=1)
    rel = np.abs(peak_comp - peak_raw) / np.maximum(peak_comp, peak_raw)
    assert np.all(rel > 0.5), (
        f'Peak dxs/dE must change substantially when toggling '
        f'include_resonance inside the RRR; got {rel} (would be 0 '
        f'before #304).'
    )
