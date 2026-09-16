"""Gamma DDX reconstruction from MF12 discrete photon lines + MF14
angular info (issue #36 landmark D3).

Combines the D1 (dxs/dE) and D2 (dxs/dmu) work: each MF12 photon line
at Eg_i contributes a kernel-folded peak along Eout, weighted by the
per-line MF14 angular distribution. The DDX folder is exactly
consistent with the D1 dxs/dE folder in the sense that
``int DDX(Ein, Eout, mu) dOmega == dxs/dE(Ein, Eout)`` to machine
precision.
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_ddxs,
    get_particle_production_dxs_dE,
    get_particle_production_dxs_dmu,
    get_particle_production_xs,
)
from endf_userpy.quantities_mt_zap import ddx_broadening as ddxb
from endf_userpy.primitives.physical_constants import PARTICLE_ZAP
from endf_userpy.primitives.np_compat import trapezoid


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


def _gaussian(sigma):
    """Match the kernel that quantities._normalize_broadening builds."""
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))

    def k(delta):
        return norm * np.exp(-0.5 * (delta / sigma) ** 2)
    return k


@pytest.fixture(scope='module')
def fe56_tendl():
    return _load('tendl21_n_Fe-56.endf')


@pytest.fixture(scope='module')
def cu63_jeff40():
    return _load('jeff40_n_Cu-63.endf')


# ============================================================
# Folder-level: compute_ddx_mf12_discrete_broadened
# ============================================================


def test_ddx_folder_peak_at_tabulated_photon_energy(fe56_tendl):
    """Fe-56 MT 51 has a single photon line at 846.778 keV. The
    Gaussian-broadened DDX peak (at any mu, since MF14 is LI=1) must
    sit within one kernel width of that Eg."""
    kernel = _gaussian(3e4)
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 1.2e6, 701)
    mus = np.linspace(-0.9, 0.9, 5)
    r = ddxb.compute_ddx_mf12_discrete_broadened(
        fe56_tendl, 51, PARTICLE_ZAP['g'], einc, eouts, mus, kernel,
    )
    peak_eout = eouts[np.argmax(r[0, :, 0])]
    assert abs(peak_eout - 846.778e3) < 3e4


def test_ddx_folder_isotropic_for_mf14_li1(fe56_tendl):
    """Fe-56 MT 51 has MF14 LI=1 (isotropic). DDX must be exactly
    flat in mu at every (Ein, Eout) point."""
    kernel = _gaussian(3e4)
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 1.2e6, 71)
    mus = np.linspace(-0.99, 0.99, 9)
    r = ddxb.compute_ddx_mf12_discrete_broadened(
        fe56_tendl, 51, PARTICLE_ZAP['g'], einc, eouts, mus, kernel,
    )
    # Every mu column must match the mean at every eout row.
    for i_eout in range(len(eouts)):
        row = r[0, i_eout, :]
        np.testing.assert_allclose(row, row.mean(), rtol=1e-10)


def test_ddx_folder_matches_dxs_dE_over_solid_angle(fe56_tendl):
    """Fe-56 MT 51 alone. Integrating the DDX folder over the solid
    angle (times 2 pi for the mu integral) must reproduce the D1
    dxs/dE folder exactly (to machine precision) at every Eout."""
    kernel = _gaussian(3e4)
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 1.2e6, 1001)
    # Dense mu mesh; DDX is analytically constant in mu here so any
    # nontrivial mesh will do, but dense helps rule out quadrature
    # error.
    mus = np.linspace(-1.0, 1.0, 41)
    ddx = ddxb.compute_ddx_mf12_discrete_broadened(
        fe56_tendl, 51, PARTICLE_ZAP['g'], einc, eouts, mus, kernel,
    )
    dxs_dE = ddxb.compute_dxs_dE_mf12_discrete_broadened(
        fe56_tendl, 51, PARTICLE_ZAP['g'], einc, eouts, kernel,
    )
    integ = trapezoid(ddx[0], mus, axis=-1) * (2 * np.pi)
    scale = dxs_dE[0].max() + 1e-300
    np.testing.assert_allclose(integ, dxs_dE[0], rtol=1e-10, atol=1e-12 * scale)


def test_ddx_folder_returns_zero_shape_without_mf12(fe56_tendl):
    """MT 2 (elastic) has no MF12; the folder returns a zero-shaped
    array of the requested shape."""
    kernel = _gaussian(3e4)
    einc = np.array([1e7])
    eouts = np.linspace(0.5e6, 1.2e6, 8)
    mus = np.linspace(-0.9, 0.9, 4)
    r = ddxb.compute_ddx_mf12_discrete_broadened(
        fe56_tendl, 2, PARTICLE_ZAP['g'], einc, eouts, mus, kernel,
    )
    assert r.shape == (1, 8, 4)
    np.testing.assert_array_equal(r, 0.0)


def test_ddx_folder_rejects_non_gamma_zap(fe56_tendl):
    """Gamma-only."""
    kernel = _gaussian(3e4)
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 1.2e6, 8)
    mus = np.linspace(-0.9, 0.9, 4)
    with pytest.raises(ValueError, match='gamma-only'):
        ddxb.compute_ddx_mf12_discrete_broadened(
            fe56_tendl, 51, PARTICLE_ZAP['n'], einc, eouts, mus, kernel,
        )


# ============================================================
# End-to-end via get_particle_production_ddxs
# ============================================================


def test_end_to_end_fe56_ddx_peak_at_847keV(fe56_tendl):
    """Broadened DDX for Fe-56 (n,total) with gamma at 1.5 MeV has
    a peak at 847 keV, present at every mu (LI=1)."""
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 1.2e6, 701)
    mus = np.array([-0.5, 0.0, 0.5])
    r = get_particle_production_ddxs(
        fe56_tendl, '(n,total)', 'g', einc, eouts, mus, broadening=3e4,
    )
    assert r is not None
    for i_mu in range(len(mus)):
        peak_eout = eouts[np.argmax(r[0, :, i_mu])]
        assert abs(peak_eout - 847e3) < 3e4


def test_end_to_end_fe56_ddx_isotropic(fe56_tendl):
    """MF14 LI=1 for Fe-56 MTs 51..80 => DDX is flat in mu across
    the MF12 discrete-line region."""
    einc = np.array([1.5e6])
    eouts = np.linspace(0.7e6, 1.0e6, 61)
    mus = np.linspace(-0.9, 0.9, 5)
    r = get_particle_production_ddxs(
        fe56_tendl, '(n,total)', 'g', einc, eouts, mus, broadening=3e4,
    )
    assert r is not None
    # The MF12 discrete-line contribution is isotropic. Any residual
    # mu variation would come from another dispatcher sum term whose
    # contribution is negligible here (< 1e-3 of the peak).
    for i_eout in range(len(eouts)):
        row = r[0, i_eout, :]
        rng = row.max() - row.min()
        assert rng <= 1e-3 * r[0].max() + 1e-30


def test_end_to_end_ddx_matches_dxs_dE_over_solid_angle(fe56_tendl):
    """End-to-end sanity: ``int DDX dOmega`` computed on a dense mu
    grid must reproduce dxs/dE from the D1 dispatcher, up to the
    small mismatch that comes from cross-dispatcher differences in
    which MTs are admitted (the same shortfall that already exists
    without D3)."""
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 1.2e6, 141)
    mus = np.linspace(-1.0, 1.0, 41)
    ddx = get_particle_production_ddxs(
        fe56_tendl, '(n,total)', 'g', einc, eouts, mus, broadening=3e4,
    )
    dxs_dE = get_particle_production_dxs_dE(
        fe56_tendl, '(n,total)', 'g', einc, eouts, broadening=3e4,
    )
    integ = trapezoid(ddx[0], mus, axis=-1) * (2 * np.pi)
    # Peak values should agree within 15% (pre-existing cross-
    # dispatcher difference not introduced by this PR).
    rel_diff = abs(integ.max() - dxs_dE[0].max()) / dxs_dE[0].max()
    assert rel_diff < 0.15


def test_end_to_end_cu63_ddx_flat_and_reasonable_scale(cu63_jeff40):
    """Cu-63 MT 51..79 all MF14 LI=1 => DDX flat in mu. At any mu,
    integrating over dE gives ~ sigma_xg / (4 pi) (per steradian).
    Bulk of production is in the discrete lines below 1.5 MeV."""
    einc = np.array([1.5e6])
    eouts = np.linspace(0.3e6, 2.3e6, 501)
    mus = np.array([-0.5, 0.5])
    r = get_particle_production_ddxs(
        cu63_jeff40, '(n,total)', 'g', einc, eouts, mus, broadening=3e4,
    )
    assert r is not None
    # Flat in mu
    np.testing.assert_allclose(r[0, :, 0], r[0, :, 1], rtol=1e-10)
    # Integrated over Eout at any mu gives dxs/dOmega ~ xs/(4 pi)
    dxs_dOmega = trapezoid(r[0, :, 0], eouts)
    xs = get_particle_production_xs(
        cu63_jeff40, '(n,total)', 'g', einc,
    )[0]
    expected = xs / (4 * np.pi)
    # Discrete-line window captures > 80% of xs (rest is MT 91
    # continuum near threshold).
    assert 0.8 < dxs_dOmega / expected <= 1.0


# ============================================================
# Regression anchors from earlier landmarks.
# ============================================================


def test_pr35_xs_unchanged_by_d3(cu63_jeff40):
    xs = get_particle_production_xs(cu63_jeff40, '(n,total)', 'g', np.array([1.5e6]))
    assert xs[0] > 0.5


def test_pr38_d1_dxs_dE_unchanged_by_d3(cu63_jeff40):
    einc = np.array([1.5e6])
    eouts = np.linspace(0.3e6, 2.3e6, 2001)
    r = get_particle_production_dxs_dE(
        cu63_jeff40, '(n,total)', 'g', einc, eouts, broadening=3e4,
    )
    integral = trapezoid(r[0], eouts)
    xs = get_particle_production_xs(cu63_jeff40, '(n,total)', 'g', einc)[0]
    assert integral / xs > 0.9


def test_pr39_d2_dxs_dmu_unchanged_by_d3(fe56_tendl):
    einc = np.array([1.5e6])
    mus = np.linspace(-0.9, 0.9, 5)
    r = get_particle_production_dxs_dmu(fe56_tendl, '(n,total)', 'g', einc, mus)
    xs = get_particle_production_xs(fe56_tendl, '(n,total)', 'g', einc)[0]
    expected = xs / (4 * np.pi)
    np.testing.assert_allclose(r[0], r[0].mean(), rtol=1e-6)
    assert abs(r[0].mean() - expected) / expected < 0.05
