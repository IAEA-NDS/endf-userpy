"""MF6 LAW=5 tabulated nuclear-plus-interference forms (LTP=12/14/15,
issue #333).

The first #264 increment implemented LTP=1 (nuclear amplitude
Legendre expansion). This issue adds the tabulated p_NI(mu) forms
(manual eq. 6.19-6.20): LTP=12 and LTP=15 interpolate p_NI
linear-in-mu, LTP=14 interpolates ln(p_NI) linear-in-mu. For LTP
in {12, 14, 15} the file stores sigma_NI in MF3/MT=2 (not 1.0 as
LTP=1 does), so the handler composes

    sigma_e(mu, E) = sigma_c(mu, E) + sigma_NI(E) * p_NI(mu, E)

and returns the ratio sigma_e / sigma_NI so the DDX pipeline's
MF3 * angdist composition recovers sigma_e. The ratio can be
negative if sigma_NI is negative at destructive-interference
energies; the top-level DDX pipeline composite stays physical.

Pins:
- The pure-Coulomb limit: zeroing out the p_NI tabulation on
  a synthetic dict makes the handler return sigma_c alone.
- Finite / sensible output on the p + C-12 LTP=12 corpus file.
- Top-level DDX APIs (dxs/dmu, dxs/dE, ddxs) return finite
  non-negative on LTP=12 (the file's sigma_NI composition
  preserves physical sign of dsigma/dmu even when the stored
  sigma_NI(E) crosses zero).
- Outside the tabulated [mu_min, mu_max] range the composite
  reduces to Rutherford only (manual eq. 6.20).
- Unsupported LTP mixing / LTP=2 still raises.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf6_law5
from endf_userpy.quantities import (
    get_particle_production_ddxs,
    get_particle_production_dxs_dE,
    get_particle_production_dxs_dmu,
)

from _corpus import resolve_p_c12_law5


@pytest.fixture(scope='module')
def p_c12_endf_dict():
    path = resolve_p_c12_law5()
    if path is None:
        pytest.skip('p + C-12 LAW=5 corpus file not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


# ---- Pure-Coulomb limit on a synthetic LTP=12 subsection ------------


def test_ltp12_zero_pdf_reduces_to_pure_rutherford(p_c12_endf_dict):
    """When every stored p_NI value is zeroed, sigma_e collapses to
    sigma_c. The pipeline convention angdist = sigma_e / sigma_NI
    gives angdist = sigma_c / sigma_NI, and the top-level composite
    MF3(E) * angdist = sigma_c(mu, E) within numerical tolerance.

    Pins the composition is correct for the LTP=12 branch: nuclear
    content turned off, Rutherford pole must dominate on its own."""
    import copy
    d = copy.deepcopy(p_c12_endf_dict)
    sub = d[6][2]['subsection'][1]
    # Zero all p_NI entries (odd-indexed, 1-based). mu knots stay.
    for ei in sub['A'].keys():
        for k in list(sub['A'][ei].keys()):
            if k % 2 == 0:  # even 1-based = p_NI value in each pair
                sub['A'][ei][k] = 0.0

    e_in = np.array([5.0e6])
    mu = np.array([0.3])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        angdist = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            d, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu,
            to_lab=False,
        ))

    # Textbook Rutherford dsigma/dmu at the same (E, mu).
    sec = d[6][2]
    awp = float(sec['subsection'][1]['AWP'])
    awr = float(sec['AWR'])
    z_proj = int(float(sec['subsection'][1]['ZAP'])) // 1000
    z_targ = int(float(d[1][451]['ZA'])) // 1000
    m1_amu = awp * mf6_law5._NEUTRON_MASS_AMU
    a_ratio = awr / awp
    eta = float(mf6_law5._sommerfeld_eta(z_proj, z_targ, m1_amu, 5.0e6, np))
    k = float(mf6_law5._cm_wavenumber_per_sqrt_barn(
        a_ratio, m1_amu, 5.0e6, np,
    ))
    sigma_c_sr = eta ** 2 / (k ** 2 * (1 - 0.3) ** 2)
    sigma_c_mu = sigma_c_sr * (2.0 * np.pi)

    # Pipeline composition: MF3(E) * angdist = sigma_c (because
    # p_NI = 0 everywhere). Fetch MF3 directly and compose.
    xst = d[3][2]['xstable']
    sigma_ni = float(np.interp(5.0e6, xst['E'], xst['xs']))
    composite = sigma_ni * angdist[0, 0]
    rel = abs(composite - sigma_c_mu) / sigma_c_mu
    assert rel < 1e-10, (
        f'MF3 * angdist = {composite:.6e}  Rutherford = {sigma_c_mu:.6e}'
        f'  rel diff = {rel:.3e}'
    )


# ---- Corpus smoke on p + C-12 ----------------------------------------


def test_p_c12_angdist_runs(p_c12_endf_dict):
    """p + C-12 is LTP=12 (74 Ein knots, NL=36). Handler returns a
    finite (possibly sign-mixed) ratio sigma_e / sigma_NI."""
    e_in = np.array([5.0e6, 10.0e6, 20.0e6])
    mu = np.linspace(-0.5, 0.9, 11)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_c12_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu,
            to_lab=False,
        ))
    assert out.shape == (3, 11)
    assert np.all(np.isfinite(out))


def test_p_c12_top_level_dxs_dmu_positive(p_c12_endf_dict):
    """The top-level dxs/dmu pipeline composes MF3 * angdist and
    must give a physically positive dsigma/dmu even when the
    file's sigma_NI(E) is sign-mixed. Pins that the pipeline
    recovers the correct physical cross section."""
    # Pick Ein well above the first MF3 zero crossing (p+C-12 MF3
    # starts positive at ~1 MeV, crosses zero around 1.8 MeV, stays
    # negative afterward). Avoid Ein right at the crossings so we
    # are not sampling the sigma_NI_safe clamp region.
    e_in = np.array([5.0e6, 10.0e6, 20.0e6])
    mu = np.linspace(-0.5, 0.9, 11)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.asarray(get_particle_production_dxs_dmu(
            p_c12_endf_dict, '(p,p_0)', 'p', e_in, mu,
        ))
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.any(out > 0.0)


def test_p_c12_top_level_ddxs_broadened(p_c12_endf_dict):
    """Top-level DDX with broadening on an LTP=12 file. Pipeline
    composes MF3 * angdist * yield into the kinematic-delta folder
    (same path as LAW=1/2/3/4 after #335), returns a non-negative
    DDX."""
    e_in = np.array([5.0e6, 10.0e6])
    e_out = np.linspace(1.0e5, 1.0e7, 20)
    mu = np.linspace(-0.5, 0.9, 11)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.asarray(get_particle_production_ddxs(
            p_c12_endf_dict, '(p,p_0)', 'p', e_in, e_out, mu,
            broadening=1.0e6,
        ))
    assert out.shape == (2, 20, 11)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.any(out > 0.0)


def test_p_c12_top_level_dxs_dE_broadened(p_c12_endf_dict):
    e_in = np.array([5.0e6, 10.0e6])
    e_out = np.linspace(1.0e5, 1.0e7, 20)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.asarray(get_particle_production_dxs_dE(
            p_c12_endf_dict, '(p,p_0)', 'p', e_in, e_out,
            broadening=1.0e6,
        ))
    assert out.shape == (2, 20)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


# ---- Outside the tabulated mu range reduces to Rutherford only ------


def test_outside_tabulated_mu_reduces_to_rutherford(p_c12_endf_dict):
    """p + C-12's tabulated mu grid spans [-1, +0.9962]. A query at
    mu = -0.9999 (inside [-1, mu_min=-1]... actually mu_min IS -1
    for p+C-12, so this is edge). Query at mu = 0.999 which is
    OUTSIDE the knot range [-1, 0.9962]. The composite should equal
    the point-wise Rutherford at that mu alone (p_NI mask = 0 by
    manual eq. 6.20)."""
    sec = p_c12_endf_dict[6][2]
    sub = sec['subsection'][1]
    mu_max_file = max(
        sub['A'][1][k] for k in sub['A'][1].keys() if k % 2 == 1
    )
    # Pick a mu strictly above the top mu knot (eq. 6.20 cuts p_NI
    # to zero there).
    assert mu_max_file < 0.999
    mu_outside = np.array([0.999])
    e_in = np.array([5.0e6])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        angdist = float(mf6_law5.get_angdist_from_subsec_law5(
            p_c12_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu_outside,
            to_lab=False,
        )[0, 0])

    # Textbook Rutherford at (Ein=5 MeV, mu=0.999).
    awp = float(sub['AWP'])
    awr = float(sec['AWR'])
    z_proj = int(float(sub['ZAP'])) // 1000
    z_targ = int(float(p_c12_endf_dict[1][451]['ZA'])) // 1000
    m1_amu = awp * mf6_law5._NEUTRON_MASS_AMU
    a_ratio = awr / awp
    eta = float(mf6_law5._sommerfeld_eta(z_proj, z_targ, m1_amu, 5.0e6, np))
    k = float(mf6_law5._cm_wavenumber_per_sqrt_barn(
        a_ratio, m1_amu, 5.0e6, np,
    ))
    sigma_c_mu = eta ** 2 / (k ** 2 * (1 - 0.999) ** 2) * 2.0 * np.pi

    xst = p_c12_endf_dict[3][2]['xstable']
    sigma_ni = float(np.interp(5.0e6, xst['E'], xst['xs']))
    composite = sigma_ni * angdist
    rel = abs(composite - sigma_c_mu) / sigma_c_mu
    assert rel < 1e-10, (
        f'outside-tabulation composite = {composite:.6e}, pure '
        f'Rutherford = {sigma_c_mu:.6e}, rel = {rel:.3e}'
    )


# ---- Mixed LTP across the NE grid still raises ----------------------


def test_mixed_ltp_raises(p_c12_endf_dict):
    import copy
    d = copy.deepcopy(p_c12_endf_dict)
    sub = d[6][2]['subsection'][1]
    # Patch half the LTP flags to a different value.
    keys = list(sub['LTP'].keys())
    for k in keys[: len(keys) // 2]:
        sub['LTP'][k] = 1
    with pytest.raises(NotImplementedError, match='mixes LTP'):
        mf6_law5.get_angdist_from_subsec_law5(
            d, mt=2, subsec_num=1,
            energies_in=np.array([5.0e6]),
            angle_cosines_out=np.array([0.0]),
            to_lab=False,
        )
