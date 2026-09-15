"""Gamma dxs/dE reconstruction from MF12 discrete photon lines
(issue #36 landmark D1a).

Extends PR #37 (S0): the differential paths no longer crash on
MF12-only files, but they still returned zeros for MTs whose photon
yields live in MF12. This landmark wires an MF12 discrete-line
folder into the broadening dispatcher so the broadened dxs/dE now
contains a kernel-folded peak at each Eg_i tabulated in MF12,
weighted by y_i(E_in) * sigma(E_in).

The continuum contribution (MF12 Eg=0 placeholder with the MF15
spectrum shape) is not yet weighted correctly; that is D1b on the
same branch. Non-broadened dxs/dE still returns zeros for
MF12-only content (discrete lines cannot be represented as delta
peaks on a finite grid). Both are documented as follow-up.
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_dxs_dE,
    get_particle_production_xs,
)
from endf_userpy.quantities_mt_zap import selectors
from endf_userpy.quantities_mt_zap import ddx_broadening as ddxb
from endf_userpy.primitives.physical_constants import PARTICLE_ZAP


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
def fe56_tendl():
    return _load('tendl21_n_Fe-56.endf')


@pytest.fixture(scope='module')
def cu63_jeff40():
    return _load('jeff40_n_Cu-63.endf')


def _gaussian(sigma):
    """Match the kernel that quantities._normalize_broadening builds
    for a scalar broadening argument."""
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))

    def k(delta):
        return norm * np.exp(-0.5 * (delta / sigma) ** 2)
    return k


# ============================================================
# selectors.has_mf12_discrete_lines
# ============================================================


def test_has_mf12_discrete_lines_admits_partial_channel_mt(fe56_tendl):
    """MT 51 in Fe-56 has a single discrete line at 846.778 keV."""
    assert selectors.has_mf12_discrete_lines(
        fe56_tendl, 51, PARTICLE_ZAP['g']
    ) is True


def test_has_mf12_discrete_lines_rejects_neutron_zap(fe56_tendl):
    """MF12 is gamma-only by ENDF-6 convention; the predicate must
    return False for neutron regardless of MF12 content."""
    assert selectors.has_mf12_discrete_lines(
        fe56_tendl, 51, PARTICLE_ZAP['n']
    ) is False


def test_has_mf12_discrete_lines_rejects_mt_without_mf12(fe56_tendl):
    """MT 2 (elastic) has no MF12."""
    assert selectors.has_mf12_discrete_lines(
        fe56_tendl, 2, PARTICLE_ZAP['g']
    ) is False


# ============================================================
# Folder-level: compute_dxs_dE_mf12_discrete_broadened
# ============================================================


def test_folder_places_peak_at_tabulated_photon_energy(fe56_tendl):
    """Fe-56 MT 51 has one photon line at exactly 846.778 keV. The
    Gaussian-broadened dxs/dE peak must sit within one kernel width
    of that energy."""
    kernel = _gaussian(3e4)
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 1.2e6, 701)  # 1 keV resolution
    r = ddxb.compute_dxs_dE_mf12_discrete_broadened(
        fe56_tendl, 51, PARTICLE_ZAP['g'], einc, eouts, kernel,
    )
    peak_eout = eouts[np.argmax(r[0])]
    assert abs(peak_eout - 846.778e3) < 3e4, (
        f'expected peak within kernel width (30 keV) of 846.778 keV, '
        f'got {peak_eout/1e3:.3f} keV'
    )


def test_folder_integral_matches_yield_times_xs(fe56_tendl):
    """For a Gaussian kernel over a wide enough integration window,
    the integral of the folded discrete-line dxs/dE must equal
    sigma(Ein) * sum_i y_i(Ein) (yield = 1 for Fe-56 MT 51's single
    line). The MF3 cross section at 1.5 MeV for MT 51 is finite;
    check the folder integral matches yield * xs within a few
    percent (kernel tails outside the window explain the tolerance)."""
    from endf_userpy.mfsec_interpretation import mf3_interpretation as mf3
    kernel = _gaussian(3e4)
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 1.2e6, 701)
    r = ddxb.compute_dxs_dE_mf12_discrete_broadened(
        fe56_tendl, 51, PARTICLE_ZAP['g'], einc, eouts, kernel,
    )
    integral = np.trapezoid(r[0], eouts)
    xs_mt51 = mf3.compute_cross_section(fe56_tendl, 51, einc)[0]
    # yield is 1 for MT 51 in Fe-56 (single photon line, LO=2)
    expected = xs_mt51 * 1.0
    assert abs(integral - expected) / expected < 0.02, (
        f'expected integral ~ {expected:.3g} b '
        f'(sigma_MT51 * yield=1), got {integral:.3g}'
    )


def test_folder_returns_zero_shaped_array_without_mf12(fe56_tendl):
    """MT 2 has no MF12; the folder must return zero-shaped output
    rather than crash."""
    kernel = _gaussian(3e4)
    einc = np.array([1e7])
    eouts = np.linspace(0.5e6, 1.2e6, 20)
    r = ddxb.compute_dxs_dE_mf12_discrete_broadened(
        fe56_tendl, 2, PARTICLE_ZAP['g'], einc, eouts, kernel,
    )
    assert r.shape == (1, 20)
    np.testing.assert_array_equal(r, 0.0)


def test_folder_rejects_non_gamma_zap(fe56_tendl):
    """MF12 discrete-line broadening is gamma-only."""
    kernel = _gaussian(3e4)
    einc = np.array([1e6])
    eouts = np.linspace(0.5e6, 1.2e6, 20)
    with pytest.raises(ValueError, match='gamma-only'):
        ddxb.compute_dxs_dE_mf12_discrete_broadened(
            fe56_tendl, 51, PARTICLE_ZAP['n'], einc, eouts, kernel,
        )


# ============================================================
# End-to-end via get_particle_production_dxs_dE
# ============================================================


def test_end_to_end_fe56_gamma_peak_at_847keV(fe56_tendl):
    """Broadened dxs/dE for Fe-56 (n,total) with gamma at 1.5 MeV
    incident energy has a clear peak at 847 keV from MT 51's
    de-excitation photon. At this incident energy, MT 91 continuum
    is small and the total gamma production is essentially the
    single MT 51 line."""
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 1.2e6, 701)
    r = get_particle_production_dxs_dE(
        fe56_tendl, '(n,total)', 'g', einc, eouts, broadening=3e4,
    )
    assert r is not None
    peak_eout = eouts[np.argmax(r[0])]
    assert abs(peak_eout - 847e3) < 3e4, (
        f'expected end-to-end peak near 847 keV, got {peak_eout/1e3:.3f} keV'
    )
    # Integral over the peak window should account for the bulk of
    # the total gamma XS at this energy.
    integral = np.trapezoid(r[0], eouts)
    xs = get_particle_production_xs(fe56_tendl, '(n,total)', 'g', einc)[0]
    # At 1.5 MeV Fe-56 the sub-1.2 MeV window catches ~99% of gamma
    # production (MT 91 continuum threshold is > 5 MeV in this file).
    assert integral / xs > 0.9, (
        f'expected integral/XS > 0.9 for Fe-56 at 1.5 MeV, '
        f'got {integral:.3g} / {xs:.3g} = {integral/xs:.2f}'
    )


def test_end_to_end_cu63_gamma_discrete_line_bulk(cu63_jeff40):
    """Cu-63 has more MT 51..79 photon lines (each level in the
    cascade). Their sum should still account for the bulk of the
    gamma production XS at 1.5 MeV, minus the MT 91 continuum piece
    (which threshold is at ~4-5 MeV in Cu-63)."""
    einc = np.array([1.5e6])
    eouts = np.linspace(0.3e6, 2.3e6, 2001)  # 1 keV resolution
    r = get_particle_production_dxs_dE(
        cu63_jeff40, '(n,total)', 'g', einc, eouts, broadening=3e4,
    )
    integral = np.trapezoid(r[0], eouts)
    xs = get_particle_production_xs(cu63_jeff40, '(n,total)', 'g', einc)[0]
    # >= 90% of the total σ_xg should be in the discrete-line window
    # at 1.5 MeV (MT 91 continuum contributes < 10% here).
    assert integral / xs > 0.9, (
        f'expected integral/XS > 0.9 for Cu-63 at 1.5 MeV, '
        f'got {integral:.3g} / {xs:.3g} = {integral/xs:.2f}'
    )


def test_end_to_end_cu63_unbroadened_dxs_dE_no_crash(cu63_jeff40):
    """Regression from S0: unbroadened path must not regress.
    Discrete lines are still missing (delta funcs can't be gridded)
    so the total is small, but the call completes."""
    einc = np.array([1.5e6])
    eouts = np.linspace(0.3e6, 2.3e6, 20)
    r = get_particle_production_dxs_dE(
        cu63_jeff40, '(n,total)', 'g', einc, eouts,
    )
    assert r is not None
    assert r.shape == (1, 20)
    assert not np.any(np.isnan(r))
    assert np.all(r >= 0)


def test_xs_from_pr35_unchanged_by_d1(cu63_jeff40):
    """D1's dxs/dE addition must not perturb the cross-section path
    from PR #35."""
    xs = get_particle_production_xs(cu63_jeff40, '(n,total)', 'g', np.array([1.5e6]))
    assert xs[0] > 0.5
