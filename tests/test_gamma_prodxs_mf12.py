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
