"""Tests for gamma production cross section reconstruction from MF12.

Pinned by issue #29: gamma production XS for reactions whose photon
yields are stored in MF12 (rather than MF6) were silently missing from
the sum returned by `get_particle_production_xs(..., 'g', ...)`. The
symptom in Pablo's four-library comparison for 63Cu+n was that only
JENDL-5 (which puts (n,n_1) gamma yields in MF6) showed a rise above
~700 keV, while ENDF/B-VIII.1, JEFF-4.0 and TENDL (which use MF12 for
the same yields) stayed near zero until MT 91 kicked in above ~5 MeV.

Root cause:
- `selectors.contains_zap` rejected (mt, ZAP=gamma) whenever MF6/mt
  existed without a gamma subsection, ignoring MF12/mt entirely.
- `quantities_mt_zap.quantities.compute_yields` had no MF12/MF13
  branch for gamma; it either crashed inside `mf6_interp.compute_yields`
  or fell back to the reaction-string multiplicity table (which
  encodes 0 gamma for MT 51..90).

Fix:
- `contains_zap` treats MF12/MF13 as authoritative for gamma so
  partial-channel MTs are admitted into the sum.
- `compute_yields` routes gamma yields through
  `discrete_quantities.compute_yields`, which reads MF12 (LO=1 tabulated
  yields, LO=2 transition probabilities) or MF13 (production cross
  sections divided by MF3).
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_particle_production_xs
from endf_userpy.quantities_mt_zap import selectors
from endf_userpy.quantities_mt_zap import quantities as quant_mt_zap
from endf_userpy.primitives.physical_constants import PARTICLE_ZAP


ADHOC_DATA_DIR = Path(__file__).resolve().parent / 'data_law1_adhoc'


def _load_endf(fn):
    parser = EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    )
    return parser.parsefile(fn)


def _skip_if_missing(fn, label):
    if not fn.exists():
        pytest.skip(
            f'{label} corpus file {fn.name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh` to populate the corpus'
        )


@pytest.fixture(scope='module')
def fe56_endf_dict():
    """TENDL-2021 Fe-56 from PR #30's ad-hoc corpus. MF12 carries
    photon yields for inelastic MTs 51..80 while MF6 carries only
    the neutron continuum for MT 91 -- exactly the layout that
    triggered issue #29 for 63Cu. Fetched on demand via
    tests/data_law1_adhoc/fetch.sh, so this fixture skips gracefully
    when the file is not present."""
    fn = ADHOC_DATA_DIR / 'tendl21_n_Fe-56.endf'
    if not fn.exists():
        pytest.skip(
            f'{fn.name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh` to populate the corpus'
        )
    parser = EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    )
    return parser.parsefile(fn)


def test_contains_zap_admits_gamma_for_mf12_only_mt(fe56_endf_dict):
    """MT 51 in this file has MF12 (gamma yields) but no MF6 gamma
    subsection. It used to be rejected by `contains_zap`, silently
    dropping the (n,n_1) gamma line from the production sum."""
    g_zap = PARTICLE_ZAP['g']
    # MT 51 = (n,n_1); has MF12 but not MF6-gamma in this evaluation.
    assert selectors.contains_zap(fe56_endf_dict, 51, g_zap) is True
    # MT 91 = (n,n_c); has MF6-gamma. Should still admit gamma.
    assert selectors.contains_zap(fe56_endf_dict, 91, g_zap) is True
    # MT 102 = (n,gamma); the reaction string alone would admit
    # gamma. Should stay admitted.
    assert selectors.contains_zap(fe56_endf_dict, 102, g_zap) is True


def test_contains_zap_unchanged_for_neutron(fe56_endf_dict):
    """The MF12/MF13 special-case is gamma-only. Neutron admission
    must not change: MT 51 emits a neutron, MT 102 does not."""
    n_zap = PARTICLE_ZAP['n']
    assert selectors.contains_zap(fe56_endf_dict, 51, n_zap) is True
    assert selectors.contains_zap(fe56_endf_dict, 91, n_zap) is True
    assert selectors.contains_zap(fe56_endf_dict, 102, n_zap) is False


def test_compute_yields_gamma_from_mf12_nonzero(fe56_endf_dict):
    """Direct check on the yield routing: MT 51 gamma yield above
    threshold must be > 0 (an inelastic-scattering cascade always
    emits at least one photon). Used to fall through to the
    reaction-string multiplicity for (n,n_1) which is 0.
    """
    g_zap = PARTICLE_ZAP['g']
    # Well above the MT 51 threshold (~845 keV for Fe-56)
    e_above = np.array([1.5e6, 5e6])
    y = quant_mt_zap.compute_yields(fe56_endf_dict, 51, g_zap, e_above)
    assert y.shape == e_above.shape
    assert np.all(y > 0), (
        f'expected nonzero gamma yield above MT 51 threshold; got {y}'
    )


def test_gamma_prodxs_includes_mf12_inelastic_contribution(fe56_endf_dict):
    """End-to-end regression: the total gamma production XS between
    the MT 51 threshold and the MT 91 continuum onset must include a
    substantial (>0.1 b) MT 51..80 contribution. Before the fix these
    MTs were silently omitted and the XS in this window dropped by
    more than an order of magnitude.
    """
    # 2 MeV is above the MT 51 threshold but well below where MT 91's
    # (n,n_c) continuum dominates. All non-negligible photon
    # production here comes from MF12/MT 51..80.
    E = np.array([2.0e6])
    xs = get_particle_production_xs(
        fe56_endf_dict, '(n,total)', 'g', E,
    )
    assert xs.shape == E.shape
    assert xs[0] > 0.1, (
        f'expected substantial MF12 gamma production XS at 2 MeV, got {xs[0]}'
    )


def test_gamma_prodxs_monotone_across_first_thresholds(fe56_endf_dict):
    """Coarse shape check: the gamma production XS rises through the
    inelastic-threshold region as more MT 51..80 channels open up.
    If MF12 were still being dropped this rise would be missing.
    """
    E = np.array([0.9e6, 1.5e6, 2.5e6, 4.0e6])
    xs = get_particle_production_xs(
        fe56_endf_dict, '(n,total)', 'g', E,
    )
    diffs = np.diff(xs)
    assert np.all(diffs > 0), (
        f'expected monotonically rising gamma XS in inelastic-threshold '
        f'region, got XS={xs}'
    )


# ============================================================
# Continuum-placeholder regression: MF12 LO=1 files that split a
# reaction's gammas between a discrete cascade (Eg>0) and a continuum
# spectrum (Eg=0 placeholder + MF15) had their yield collapse to
# ~zero after the first draft of the fix because
# discrete_quantities.compute_yields (built for angular-distribution
# reconstruction) intentionally excludes the Eg=0 placeholder. The
# main-path dispatcher now uses compute_total_gamma_yields instead,
# which includes the placeholder.
# ============================================================


@pytest.fixture(scope='module')
def al27_endf_dict():
    """Al-27 from ENDF/B-VIII.1: capture MT 102 uses MF12 LO=1 with
    ~290 discrete lines below thermal plus an Eg=0 placeholder for
    the high-energy continuum. Fetched on demand."""
    fn = ADHOC_DATA_DIR / 'endfb81_n_Al-27.endf'
    _skip_if_missing(fn, 'Al-27')
    return _load_endf(fn)


def test_al27_mt102_gamma_yield_includes_continuum(al27_endf_dict):
    """Above ~10 keV, Al-27 MT 102 gamma yield is carried almost
    entirely by the Eg=0 continuum placeholder (with MF15 giving the
    spectrum shape). If the yield summation excludes the placeholder
    the total drops to essentially 0, and the (n,gamma) contribution
    to sigma_xg silently disappears. The correct total gamma yield
    is ~2-3 photons per capture across the 100 keV .. 14 MeV window.
    """
    einc = np.array([1e5, 1e6, 1.4e7])
    y = quant_mt_zap.compute_yields(
        al27_endf_dict, 102, PARTICLE_ZAP['g'], einc,
    )
    assert y.shape == einc.shape
    # Al-27 (n,gamma) cascade emits at least ~2 gammas per capture.
    # A yield below 1 here means the continuum placeholder is being
    # dropped and the total is only the (near-zero above 10 keV)
    # discrete-line sum.
    assert np.all(y > 1.5), (
        f'expected Al-27 MT 102 gamma yield > 1.5 across [100keV, 14MeV], '
        f'got {y}'
    )


# ============================================================
# MT 5 residual-yield regression: the new `mf6_help.contains_zap`
# guard on compute_yields' MF6 branch must not break the MT 5
# catch-all residual yield path used by compute_xs_mt5_contrib.
# MF6/MT 5 subsections carry residual ZAPs (large ZA codes like
# 92235.0), not particle ZAPs; the guard needs to admit those.
# ============================================================


@pytest.fixture(scope='module')
def u235_endf_dict():
    """TENDL-2021 U-235: MF6/MT 5 declares subsections for many
    fission-fragment residuals including 92235.0 and 92236.0.
    Fetched on demand."""
    fn = ADHOC_DATA_DIR / 'tendl21_n_U-235.endf'
    _skip_if_missing(fn, 'U-235')
    return _load_endf(fn)


def test_mt5_residual_yield_from_mf6_unchanged(u235_endf_dict):
    """compute_yields(mt=5, zap=<residual_ZA>) must continue to
    route through MF6/MT 5's residual subsections. The new
    mf6_help.contains_zap guard could otherwise force a
    reaction-string fallback that has no residual-yield concept
    for MT 5 and would return 0 or crash. Pinning against a
    residual (92235.0 = U-235) known to be present as a MF6/MT 5
    subsection in this file."""
    residual_za = 92235.0
    einc = np.array([1e6, 1e7])
    y = quant_mt_zap.compute_yields(
        u235_endf_dict, 5, residual_za, einc,
    )
    assert y.shape == einc.shape
    # The yield may be very small (rare residual) but must not
    # collapse to exactly zero, which is what a reaction-string
    # fallback would produce.
    assert np.all(y >= 0), f'negative yields returned: {y}'
    assert np.any(y > 0), (
        f'MT 5 residual yield for ZAP={residual_za} is uniformly 0; '
        f'expected the MF6/MT 5 subsection to be found. Got {y}'
    )


# ============================================================
# Cross-library end-to-end check: reproduces the exact pattern
# from Naohiko's issue #29 report on the JEFF-4.0 63Cu file that
# was added to the ad-hoc corpus for this purpose. Fetched on
# demand.
# ============================================================


@pytest.fixture(scope='module')
def cu63_jeff40_endf_dict():
    """JEFF-4.0 63Cu: exactly the file behind Pablo's four-library
    (n,xg) plot. MF6 declares only the scattered neutron for MT
    51..79; MF12 (LO=2 transition probabilities) carries the
    de-excitation photons. Before PR #35 the MT 51 photon
    contribution -- responsible for the visible rise above 700 keV
    that only JENDL-5 was reproducing -- was silently dropped for
    this file."""
    fn = ADHOC_DATA_DIR / 'jeff40_n_Cu-63.endf'
    _skip_if_missing(fn, 'JEFF-4.0 Cu-63')
    return _load_endf(fn)


def test_cu63_gamma_prodxs_shows_nn1_rise_above_700keV(cu63_jeff40_endf_dict):
    """Direct reproduction of the issue: gamma production XS at
    1.5 MeV must be > 0.5 b, driven by (n,n_1) MT 51 photons in
    MF12. Before PR #35 this value was ~0.05 b (only capture and
    MT 91 continuum contributed, and MT 91's continuum threshold
    is above 4 MeV). Anchored at the specific energy Naohiko
    called out ("above ~700 keV")."""
    einc = np.array([1.5e6])
    xs = get_particle_production_xs(
        cu63_jeff40_endf_dict, '(n,total)', 'g', einc,
    )
    assert xs.shape == einc.shape
    # Pre-fix value was ~0.04 b; post-fix ~0.96 b for JEFF-4.0.
    # The 0.5 b threshold catches the regression with wide margin
    # without over-fitting to the exact number.
    assert xs[0] > 0.5, (
        f'expected 63Cu gamma production XS > 0.5 b at 1.5 MeV '
        f'(dominated by MT 51..79 MF12 photons), got {xs[0]:.3g} b'
    )


def test_cu63_gamma_prodxs_at_100keV_matches_capture_only(cu63_jeff40_endf_dict):
    """Sanity floor: below the MT 51 threshold (~669 keV for 63Cu),
    the gamma production XS is dominated by the (n,gamma) capture
    channel. Value should be positive and small (~0.1 b), not the
    ~0 b that a broken yield dispatcher would produce."""
    einc = np.array([1e5])
    xs = get_particle_production_xs(
        cu63_jeff40_endf_dict, '(n,total)', 'g', einc,
    )
    assert 0.01 < xs[0] < 1.0, (
        f'63Cu gamma XS at 100 keV should sit in [0.01, 1.0] b '
        f'from capture alone, got {xs[0]:.3g} b'
    )
