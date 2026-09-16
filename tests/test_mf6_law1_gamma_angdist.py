"""Tests for MF6/LAW=1 discrete-line gamma angular reconstruction
(issue #55).

Before this fix, the angular content of MF6/LAW=1 gamma subsections
(encoded through the LANG parameter and the b(k) amplitude
coefficients) was dropped from `dxs/dmu` and DDX because
`mf6_help.has_angdist_part` only recognised LAW=2/3/4. Files that
declare (n,n_i) gamma cascades in MF6/LAW=1 -- JENDL-5 in particular
-- got a silent zero angular contribution for those MTs.

The fix reads the per-line amplitudes via
`mf6_interp.compute_law1_discrete_lines` and sums them over the
discrete-line index. The sum has the right normalisation relative
to `compute_daxs`: per-line yield weight is embedded in the b(k)
coefficients themselves, so `sum_k amp_disc` gives directly the
discrete-lines contribution to the total angular distribution.
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_xs,
    get_particle_production_dxs_dmu,
    get_particle_production_dxs_dE,
    get_particle_production_ddxs,
)
from endf_userpy.quantities_mt_zap import distribution1d as d1d
from endf_userpy.quantities_mt_zap import selectors
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


@pytest.fixture(scope='module')
def jendl5_cu63():
    """JENDL-5 Cu-63: MT 51..90 put the (n,n_i) de-excitation
    photons in MF6/LAW=1 as pure-discrete ND=1 subsections (each MT
    a single gamma line). The archetypal case for issue #55 -- the
    entire (n,n_i) gamma angular structure was silently dropped
    before this fix."""
    return _load('jendl5_n_Cu-63.endf')


@pytest.fixture(scope='module')
def jeff40_cu63():
    """JEFF-4.0 Cu-63: MTs 51..79 have MF12+MF14 (D2's path); MT 91,
    102, 103, 104, 105, 107 have MF6/LAW=1 with mixed discrete +
    continuum gamma content. Ratio to xs/(4pi) was 0.976 before the
    fix; should now match 1.0 exactly."""
    return _load('jeff40_n_Cu-63.endf')


@pytest.fixture(scope='module')
def fe56_tendl():
    return _load('tendl21_n_Fe-56.endf')


@pytest.fixture(scope='module')
def u235_tendl():
    """TENDL-2021 U-235: many MF6/LAW=1 gamma subsections. Ratio was
    0.458 before the fix -- the largest jump of any corpus file."""
    return _load('tendl21_n_U-235.endf')


# ============================================================
# Dispatcher: compute_angdist_values on a pure-discrete
# MF6/LAW=1 MT returns the correctly-normalised angdist.
# ============================================================


def test_jendl5_mt51_isotropic_returns_half(jendl5_cu63):
    """JENDL-5 MT 51 has a single gamma line in MF6/LAW=1 with
    LANG=1 (Legendre) and b(0) alone (isotropic). `compute_angdist_values`
    must return f(mu) = 0.5 flat, integrating to 1 over mu. Before
    the fix it returned zero (the LAW=1 subsection was dropped)."""
    einc = np.array([1.5e6])
    mus = np.linspace(-0.99, 0.99, 11)
    r = d1d.compute_angdist_values(
        jendl5_cu63, 51, PARTICLE_ZAP['g'], einc, mus,
    )
    assert r.shape == (len(einc), len(mus))
    np.testing.assert_allclose(r, 0.5, rtol=1e-6)


def test_jendl5_mt51_integrates_to_one(jendl5_cu63):
    """Sanity: f(mu) integrates to 1 over [-1, +1]."""
    einc = np.array([1.5e6])
    mus = np.linspace(-1.0, 1.0, 1001)
    r = d1d.compute_angdist_values(
        jendl5_cu63, 51, PARTICLE_ZAP['g'], einc, mus,
    )
    integ = trapezoid(r[0], mus)
    assert abs(integ - 1.0) < 1e-6


# ============================================================
# Selector: has_mf6_law1_discrete_lines already existed for the
# ddx_broadening path; anchor its behaviour on both fixtures so
# the new compute_angdist_values path is gated consistently.
# ============================================================


def test_selector_law1_disc_admits_on_jendl5_partial(jendl5_cu63):
    """JENDL-5 MT 51 gamma: MF6/LAW=1 with ND=1. The selector must
    admit it."""
    assert selectors.has_mf6_law1_discrete_lines(
        jendl5_cu63, 51, PARTICLE_ZAP['g']
    ) is True


def test_selector_law1_disc_admits_on_jeff40_mt102(jeff40_cu63):
    """JEFF-4.0 MT 102 gamma: MF6/LAW=1 mixed discrete + continuum.
    Admitted."""
    assert selectors.has_mf6_law1_discrete_lines(
        jeff40_cu63, 102, PARTICLE_ZAP['g']
    ) is True


# ============================================================
# End-to-end: get_particle_production_dxs_dmu ratio to xs/(4*pi).
# All four corpus files with MF6/LAW=1 gamma content should now be
# very close to 1.0 (isotropic-per-line MF6/LAW=1 in every case).
# ============================================================


def test_end_to_end_jendl5_cu63_ratio_near_one(jendl5_cu63):
    """JENDL-5 was the largest expected improvement (all (n,n_i)
    gamma angular info was dropped). Ratio must recover to ~1.0."""
    einc = np.array([1.5e6])
    mus = np.linspace(-0.9, 0.9, 9)
    r = get_particle_production_dxs_dmu(jendl5_cu63, '(n,total)', 'g', einc, mus)
    xs = get_particle_production_xs(jendl5_cu63, '(n,total)', 'g', einc)[0]
    expected = xs / (4 * np.pi)
    ratio = r[0].mean() / expected
    assert 0.99 < ratio < 1.005, (
        f'expected ratio ~ 1.0, got {ratio:.4f} '
        f'(mean={r[0].mean():.4g}, xs/4pi={expected:.4g})'
    )


def test_end_to_end_jeff40_cu63_ratio_now_one(jeff40_cu63):
    """JEFF-4.0 Cu-63: ratio was 0.976 pre-fix; now 1.000. The
    MF6/LAW=1 discrete lines of MT 102, 103, etc. that were dropped
    now contribute correctly."""
    einc = np.array([1.5e6])
    mus = np.linspace(-0.9, 0.9, 9)
    r = get_particle_production_dxs_dmu(jeff40_cu63, '(n,total)', 'g', einc, mus)
    xs = get_particle_production_xs(jeff40_cu63, '(n,total)', 'g', einc)[0]
    expected = xs / (4 * np.pi)
    ratio = r[0].mean() / expected
    assert 0.999 < ratio < 1.001, (
        f'expected ratio ~ 1.0 exactly, got {ratio:.6f}'
    )


def test_end_to_end_fe56_tendl_ratio_now_one(fe56_tendl):
    """Fe-56: ratio was 0.990 pre-fix; now 1.000."""
    einc = np.array([1.5e6])
    mus = np.linspace(-0.9, 0.9, 9)
    r = get_particle_production_dxs_dmu(fe56_tendl, '(n,total)', 'g', einc, mus)
    xs = get_particle_production_xs(fe56_tendl, '(n,total)', 'g', einc)[0]
    expected = xs / (4 * np.pi)
    ratio = r[0].mean() / expected
    assert 0.999 < ratio < 1.001, (
        f'expected ratio ~ 1.0 exactly, got {ratio:.6f}'
    )


def test_end_to_end_u235_tendl_ratio_now_near_one(u235_tendl):
    """U-235: the largest jump. Ratio was 0.458 pre-fix (~half the
    file's gamma angular info was dropped from MF6/LAW=1); now ~1.0."""
    einc = np.array([1.5e6])
    mus = np.linspace(-0.9, 0.9, 9)
    r = get_particle_production_dxs_dmu(u235_tendl, '(n,total)', 'g', einc, mus)
    xs = get_particle_production_xs(u235_tendl, '(n,total)', 'g', einc)[0]
    expected = xs / (4 * np.pi)
    ratio = r[0].mean() / expected
    assert 0.99 < ratio < 1.005, (
        f'expected ratio ~ 1.0, got {ratio:.4f}'
    )


# ============================================================
# End-to-end isotropy: all these fixtures have LAW=1 subsecs
# with LANG=1 b(0)-only (isotropic), so dxs/dmu should be flat
# in mu.
# ============================================================


def test_jendl5_dxs_dmu_flat_in_mu(jendl5_cu63):
    einc = np.array([1.5e6])
    mus = np.linspace(-0.9, 0.9, 9)
    r = get_particle_production_dxs_dmu(jendl5_cu63, '(n,total)', 'g', einc, mus)
    np.testing.assert_allclose(r[0], r[0].mean(), rtol=1e-3)


# ============================================================
# Regression: XS, dxs/dE, DDX unchanged by this PR.
# ============================================================


def test_xs_unchanged_by_law1_angdist(jendl5_cu63):
    """The XS path (PR #35) integrates over yields x MF3 xs; the
    angular reconstructor doesn't touch it. Must be unchanged."""
    xs = get_particle_production_xs(
        jendl5_cu63, '(n,total)', 'g', np.array([1.5e6]),
    )[0]
    assert xs > 0.1  # loose sanity


def test_dxs_dE_unchanged_by_law1_angdist(jeff40_cu63):
    """PR #38's D1 dxs/dE integrates to y_disc x sigma via the
    kernel-broadened folder plus the compute_dexs continuum branch.
    Independent of the angular reconstructor being added here."""
    einc = np.array([1.5e6])
    eouts = np.linspace(0.3e6, 2.3e6, 2001)
    r = get_particle_production_dxs_dE(
        jeff40_cu63, '(n,total)', 'g', einc, eouts, broadening=3e4,
    )
    integral = trapezoid(r[0], eouts)
    xs = get_particle_production_xs(jeff40_cu63, '(n,total)', 'g', einc)[0]
    assert integral / xs > 0.9


def test_ddx_unchanged_by_law1_angdist(fe56_tendl):
    """PR #40's D3 DDX. Regression anchor."""
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 1.2e6, 71)
    mus = np.linspace(-0.9, 0.9, 5)
    r = get_particle_production_ddxs(
        fe56_tendl, '(n,total)', 'g', einc, eouts, mus, broadening=3e4,
    )
    assert r is not None
    assert not np.any(np.isnan(r))


# ============================================================
# The MF6/LAW=1 diagnosis history (in data_law1_adhoc/README.md)
# notes that Fe-56 MT 106/111/112 have b(k)=0 -- a TENDL evaluator
# choice. `_compute_mf6_law1_disc_angdist` should return an all-
# zeros angdist for such subsecs so the sum stays well-defined.
# ============================================================


def test_zero_amplitude_subsecs_return_zero_angdist(fe56_tendl):
    """Fe-56 MT 106 is a TENDL Σb=0 subsection: ND>0 but all b(k)=0.
    The angular projection helper should return zeros without
    error."""
    einc = np.array([1.4e7])
    mus = np.linspace(-0.9, 0.9, 5)
    r = d1d._compute_mf6_law1_disc_angdist(
        fe56_tendl, 106, PARTICLE_ZAP['g'], einc, mus, to_lab=True,
    )
    assert r.shape == (len(einc), len(mus))
    np.testing.assert_array_equal(r, 0.0)
