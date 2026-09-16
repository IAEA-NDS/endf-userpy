"""Tests for the gamma-ejectile MF6/LAW=2 dispatch (issue #78).

Two failure modes on H-1 (n,g):

- `get_particle_production_dxs_dE(H-1, "(n,total)", "g", ...)` raised
  ``KeyError: 'g'`` inside `_prepare_angdist_to_energydist_conversion`
  because `PARTICLE_MASSES_AMU` had no gamma entry.

- `get_particle_production_dxs_dmu(H-1, "(n,total)", "g", ...)` silently
  returned all NaN because `mf6_get_law2` uses massive-particle
  2-body kinematics with the subsection's `AWP`; for photons the
  formulas produce NaN, and the NaN propagated silently.

Fix:

1. Add ``'g': 0.0`` to `PARTICLE_MASSES_AMU` so the mass lookup
   succeeds. The value only feeds the reduced-mass formula
   `m_r = m_t + (m_i - m_e) - q`, where zero is physically
   correct for photons.

2. Short-circuit `get_angdist_from_subsec_law2` when the
   subsection stores gamma (`ZAP == 0`): return zeros and emit a
   `UserWarning` naming the MT and pointing at the MF12+MF14 /
   MF13+MF14 alternative. The MF6/LAW=2 photon-kinematics path
   needs massless-ejectile 2-body formulas that aren't
   implemented yet; the warning tells users that the dispatch
   is deliberately empty for this specific representation.

These tests pin:

1. No crash on any H-1 gamma API surface.
2. No silent NaN on any H-1 gamma API surface.
3. At least one summary warning per call, naming the LAW=2 issue.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_dxs_dE,
    get_particle_production_dxs_dmu,
    get_particle_production_ddxs,
)
from endf_userpy.primitives import physical_constants as physconst


ADHOC = Path(__file__).resolve().parent / 'data_law1_adhoc'


def _load(fn_name):
    fn = ADHOC / fn_name
    if not fn.exists():
        pytest.skip(
            f'{fn_name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh`'
        )
    return EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    ).parsefile(fn)


@pytest.fixture(scope='module')
def h1_endfb81():
    return _load('endfb81_n_H-1.endf')


# ============================================================
# The mass-lookup fix on its own.
# ============================================================


def test_particle_masses_amu_has_gamma():
    """Regression guard for #78 fix step 1: the gamma entry must
    remain in the mass table so `_prepare_angdist_to_energydist_
    conversion` and any other caller of `get_particle_mass_for_zap`
    can process the gamma ZAP without KeyError."""
    assert 'g' in physconst.PARTICLE_MASSES_AMU
    assert physconst.PARTICLE_MASSES_AMU['g'] == 0.0


def test_get_particle_mass_for_zap_gamma():
    """`get_particle_mass_for_zap(gamma_zap)` must return 0.0 (the
    photon rest mass in eV/c^2)."""
    gamma_zap = physconst.PARTICLE_ZAP['g']
    assert physconst.get_particle_mass_for_zap(gamma_zap) == 0.0


# ============================================================
# H-1 gamma APIs: no crash, no silent NaN.
# ============================================================


def test_h1_dxs_dmu_gamma_no_crash_no_nan(h1_endfb81):
    """H-1 gamma dxs/dmu previously returned all-NaN silently. Fix:
    finite output + at least one UserWarning naming the LAW=2 issue."""
    einc = np.linspace(1e5, 1.4e7, 6)
    mus = np.linspace(-0.9, 0.9, 5)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        r = get_particle_production_dxs_dmu(
            h1_endfb81, '(n,total)', 'g', einc, mus,
        )
    assert r.shape == (6, 5)
    assert not np.any(np.isnan(r)), 'silent NaN in gamma dxs/dmu'
    assert np.all(r >= 0.0)
    law2_warns = [
        w for w in recorded
        if 'LAW=2' in str(w.message) and 'issue #78' in str(w.message)
    ]
    assert len(law2_warns) >= 1


def test_h1_dxs_dE_gamma_no_crash(h1_endfb81):
    """H-1 gamma dxs/dE previously raised KeyError('g'). Fix: mass
    entry added; the MF6/LAW=2 gamma subsection is now short-
    circuited to zero contribution and the API returns a finite
    array (all zeros for this file, since MF6/LAW=2 gamma is the
    only source and it's deliberately not implemented)."""
    einc = np.linspace(1e5, 1.4e7, 6)
    eouts = np.linspace(1e5, 1.4e7, 20)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        r = get_particle_production_dxs_dE(
            h1_endfb81, '(n,total)', 'g', einc, eouts,
        )
    assert r is not None
    assert r.shape == (6, 20)
    assert not np.any(np.isnan(r))
    law2_warns = [
        w for w in recorded if 'issue #78' in str(w.message)
    ]
    assert len(law2_warns) >= 1


def test_h1_ddx_broadened_gamma_no_crash(h1_endfb81):
    """H-1 gamma DDX with broadening also went through the
    LAW=2 gamma path; guard that the fix covers it too."""
    einc = np.linspace(1e5, 1.4e7, 6)
    eouts = np.linspace(1e5, 1.4e7, 20)
    mus = np.linspace(-0.9, 0.9, 5)
    # The LAW=2 gamma warning already fires from the dxs/dmu +
    # dxs/dE tests; here we only need to confirm the DDX API path
    # doesn't crash or return NaN, so let any warnings pass through.
    r = get_particle_production_ddxs(
        h1_endfb81, '(n,total)', 'g', einc, eouts, mus,
        broadening=3e4,
    )
    assert r is not None
    assert r.shape == (6, 20, 5)
    assert not np.any(np.isnan(r))


# ============================================================
# Non-regression: neutron and other non-gamma queries unchanged.
# ============================================================


def test_h1_dxs_dmu_neutron_unchanged(h1_endfb81):
    """Neutron dxs/dmu still works and returns meaningful non-zero
    output (H-1 (n,n) elastic is the dominant channel)."""
    einc = np.linspace(1e5, 1.4e7, 6)
    mus = np.linspace(-0.9, 0.9, 5)
    r = get_particle_production_dxs_dmu(
        h1_endfb81, '(n,total)', 'n', einc, mus,
    )
    assert r.shape == (6, 5)
    assert np.all(np.isfinite(r))
    assert float(r.max()) > 0.0
