"""Safety-guard tests for gamma differential-quantity entry points
(issue #36 landmark S0).

PR #35 fixed the gamma production **cross section** for files that
store partial-channel photon yields in MF12/MF13 (i.e. every library
except JENDL-5). The differential entry points on the same files
started **crashing** because:

- `mf6_help.has_disc_part` / `has_cont_part` / `has_angdist_part`
  called `get_subsecs` unconditionally and raised `IndexError` when
  the requested ZAP was absent from MF6/MT.
- `distribution1d.compute_angdist_values` fell through to a raise
  when neither MF4 (neutron-only) nor MF6 (missing ZAP) could
  reconstruct.
- `distribution1d.compute_energydist_values` reached the MF4 branch
  for gamma zap and tried to convert the neutron angdist to an
  energy distribution using an ejectile mass that gamma does not
  have in the mass table (`KeyError: 'g'`).
- `distribution2d.compute_dist2d_values` raised `ValueError` on
  MF6-only files without a gamma subsection.

This landmark does NOT implement the MF12/MF14 reconstruction (that
is D1/D2/D3 in issue #36). It only makes the differential paths
degrade gracefully -- returning zero-shaped arrays for MTs whose
gamma content is not yet reconstructable -- so that the cumulative
sum over MTs stays well-defined and callers see finite output rather
than IndexError / KeyError / ValueError.
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from endf_userpy.quantities import (
    get_particle_production_xs,
    get_particle_production_dxs_dE,
    get_particle_production_dxs_dmu,
    get_particle_production_ddxs,
)
from endf_userpy.quantities_mt_zap import distribution1d as d1d
from endf_userpy.quantities_mt_zap import distribution2d as d2d
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
def cu63_jeff40():
    """The file behind Naohiko's issue #29 report."""
    return _load('jeff40_n_Cu-63.endf')


@pytest.fixture(scope='module')
def fe56_tendl():
    """MF6 for MT 51..80 with neutron subsections only; MF4 for the
    same MTs (neutron angdist). Exercises the MF4-neutron guard on
    the energydist dispatcher."""
    return _load('tendl21_n_Fe-56.endf')


# ============================================================
# mf6_help.has_*_part predicates return False for missing ZAP
# rather than raising IndexError. Cascades through selectors and
# ddx_broadening automatically.
# ============================================================


def test_mf6_help_has_disc_part_returns_false_for_missing_zap(cu63_jeff40):
    """MT 51 in JEFF-4.0 Cu-63 has MF6 with neutron only (no gamma
    subsection). has_disc_part used to raise IndexError; it now
    returns False."""
    assert mf6_help.has_disc_part(cu63_jeff40, 51, PARTICLE_ZAP['g']) is False


def test_mf6_help_has_cont_part_returns_false_for_missing_zap(cu63_jeff40):
    assert mf6_help.has_cont_part(cu63_jeff40, 51, PARTICLE_ZAP['g']) is False


def test_mf6_help_has_angdist_part_returns_false_for_missing_zap(cu63_jeff40):
    assert mf6_help.has_angdist_part(cu63_jeff40, 51, PARTICLE_ZAP['g']) is False


def test_mf6_help_has_disc_part_unchanged_for_present_zap(cu63_jeff40):
    """MT 91 declares gamma in MF6 for this file; the predicate must
    return whatever it always has for a real present ZAP (a bool)."""
    r = mf6_help.has_disc_part(cu63_jeff40, 91, PARTICLE_ZAP['g'])
    assert isinstance(r, bool)


# ============================================================
# distribution1d dispatchers return zeros for gamma on MTs whose
# angular / energy distribution is not yet reconstructable, instead
# of crashing.
# ============================================================


def test_compute_angdist_returns_finite_for_gamma_on_mf6_neutron_only(cu63_jeff40):
    """MT 51 in JEFF-4.0 Cu-63 has MF6 gamma-less and no MF4. It used
    to raise IndexError before S0 (PR #37); after S0 alone it
    returned zeros; after D2 (issue #36) it now returns the MF14
    isotropic distribution (0.5) since Cu-63 declares MF14/MT 51 with
    LI=1. Either way the cumulative sum in
    get_particle_production_dxs_dmu stays well-defined."""
    einc = np.array([1.5e6])
    mus = np.linspace(-0.9, 0.9, 5)
    r = d1d.compute_angdist_values(
        cu63_jeff40, 51, PARTICLE_ZAP['g'], einc, mus,
    )
    assert r.shape == (len(einc), len(mus))
    # After D2: MF14 LI=1 => f(mu) = 0.5 (isotropic, integrates to 1).
    np.testing.assert_allclose(r, 0.5)


def test_compute_energydist_no_keyerror_for_gamma_on_mf4_neutron_file(fe56_tendl):
    """Fe-56 TENDL has MF4/MT 51 (neutron angdist). Before the
    MF4-neutron-only guard, compute_energydist_values took the MF4
    branch for zap=gamma and crashed inside the LAB kinematic
    conversion with `KeyError: 'g'` (gamma has no ejectile mass).
    Now the guard skips MF4 and the function falls through to
    zeros."""
    einc = np.array([2.0e6])
    eouts = np.linspace(0.5e6, 2.0e6, 8)
    r = d1d.compute_energydist_values(
        fe56_tendl, 51, PARTICLE_ZAP['g'], einc, eouts,
    )
    assert r.shape == (len(einc), len(eouts))
    np.testing.assert_array_equal(r, 0.0)


def test_compute_energydist_still_zeros_for_mf6_neutron_only(cu63_jeff40):
    """MT 51 in JEFF-4.0 Cu-63 has MF6 with only the scattered neutron.
    compute_energydist_values (already fixed in issue #31 to return
    zeros for pure-ND MF6) must not regress here."""
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 2.0e6, 8)
    r = d1d.compute_energydist_values(
        cu63_jeff40, 51, PARTICLE_ZAP['g'], einc, eouts,
    )
    assert r.shape == (len(einc), len(eouts))
    np.testing.assert_array_equal(r, 0.0)


# ============================================================
# distribution2d dispatcher returns zeros in the same shape rather
# than raising ValueError.
# ============================================================


def test_compute_dist2d_returns_zeros_for_gamma_on_mf6_neutron_only(cu63_jeff40):
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 2.0e6, 6)
    mus = np.linspace(-0.9, 0.9, 4)
    r = d2d.compute_dist2d_values(
        cu63_jeff40, 51, PARTICLE_ZAP['g'], einc, eouts, mus,
    )
    assert r.shape == (len(einc), len(eouts), len(mus))
    np.testing.assert_array_equal(r, 0.0)


# ============================================================
# End-to-end: public entry points do not crash on the primary
# reproducer file (63Cu JEFF-4.0), for both broadened and
# unbroadened variants.
# ============================================================


def test_get_particle_production_dxs_dE_no_crash_on_cu63(cu63_jeff40):
    """The reproducer from issue #36. Used to raise
    `IndexError: subsection with ZAP=0.0 not found in MF6/MT51`
    inside compute_energydist_values."""
    einc = np.array([1.5e6])
    eouts = np.linspace(0.1e6, 2.0e6, 12)
    r = get_particle_production_dxs_dE(cu63_jeff40, '(n,total)', 'g', einc, eouts)
    assert r is not None
    assert r.shape == (len(einc), len(eouts))
    assert not np.any(np.isnan(r))
    assert np.all(r >= 0)


def test_get_particle_production_dxs_dE_broadened_no_crash_on_cu63(cu63_jeff40):
    einc = np.array([1.5e6])
    eouts = np.linspace(0.1e6, 2.0e6, 12)
    r = get_particle_production_dxs_dE(
        cu63_jeff40, '(n,total)', 'g', einc, eouts, broadening=1e5,
    )
    assert r is not None
    assert r.shape == (len(einc), len(eouts))
    assert not np.any(np.isnan(r))


def test_get_particle_production_dxs_dmu_no_crash_on_cu63(cu63_jeff40):
    einc = np.array([1.5e6])
    mus = np.linspace(-0.9, 0.9, 5)
    r = get_particle_production_dxs_dmu(cu63_jeff40, '(n,total)', 'g', einc, mus)
    assert r is not None
    assert r.shape == (len(einc), len(mus))
    assert not np.any(np.isnan(r))


def test_get_particle_production_ddxs_no_crash_on_cu63(cu63_jeff40):
    einc = np.array([1.5e6])
    eouts = np.linspace(0.1e6, 2.0e6, 8)
    mus = np.linspace(-0.9, 0.9, 4)
    r = get_particle_production_ddxs(
        cu63_jeff40, '(n,total)', 'g', einc, eouts, mus,
    )
    # May be None if no MT admits continuous DDX for gamma; must not raise.
    if r is not None:
        assert r.shape == (len(einc), len(eouts), len(mus))
        assert not np.any(np.isnan(r))


def test_get_particle_production_ddxs_broadened_no_crash_on_cu63(cu63_jeff40):
    einc = np.array([1.5e6])
    eouts = np.linspace(0.1e6, 2.0e6, 8)
    mus = np.linspace(-0.9, 0.9, 4)
    r = get_particle_production_ddxs(
        cu63_jeff40, '(n,total)', 'g', einc, eouts, mus, broadening=1e5,
    )
    if r is not None:
        assert r.shape == (len(einc), len(eouts), len(mus))
        assert not np.any(np.isnan(r))


def test_get_particle_production_dxs_dE_no_crash_on_fe56(fe56_tendl):
    """Second file layout: MF4 for MT 51 (would trip the KeyError:
    'g' path before the MF4-neutron guard)."""
    einc = np.array([2.0e6])
    eouts = np.linspace(0.5e6, 2.0e6, 8)
    r = get_particle_production_dxs_dE(fe56_tendl, '(n,total)', 'g', einc, eouts)
    assert r is not None
    assert r.shape == (len(einc), len(eouts))
    assert not np.any(np.isnan(r))


def test_get_particle_production_xs_unchanged_on_cu63(cu63_jeff40):
    """XS is what PR #35 fixed; S0 must not perturb it. Anchored at
    the same 1.5 MeV point where PR #35 pinned > 0.5 b."""
    xs = get_particle_production_xs(cu63_jeff40, '(n,total)', 'g', np.array([1.5e6]))
    assert xs[0] > 0.5
