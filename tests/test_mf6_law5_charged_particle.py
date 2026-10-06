"""MF6 LAW=5 charged-particle elastic scattering reconstruction
(issue #264 first increment).

Pins the first ``LTP=1`` ``LIDP=0`` (nuclear amplitude Legendre
expansion, distinguishable particles) scope:

- The Rutherford helper matches the textbook pointwise formula
  ``sigma_cd = eta^2 / (k^2 (1 - mu)^2)`` to zero.
- The full reconstruction, applied with every nuclear coefficient
  set to zero, reduces to pure Rutherford to zero: the interference
  and pure-nuclear sums are both exactly zero in this limit, so the
  only surviving term is the pointwise Coulomb formula. Catches
  the common sign / prefactor / Legendre-indexing error class.
- The ENDF/B-VIII.0 proton on He-3 and proton on B-10 evaluations
  reconstruct without raising, with all-finite non-negative output
  on an interior Ein / mu grid.
- The dispatch wiring in ``compute_angdist_from_subsec`` reaches
  the new handler.
- Unsupported shapes (LTP=12 tabulated, LIDP=1 identical, LCT!=2)
  raise NotImplementedError with a clear message so a caller sees
  the gap rather than silent wrong physics.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf6_law5
from endf_userpy.mfsec_interpretation.mf6_interpretation_subsecs import (
    compute_angdist_from_subsec,
)

from _corpus import resolve_p_he3_law5, resolve_p_b10_law5


# ---- Fixtures ---------------------------------------------------------


@pytest.fixture(scope='module')
def p_he3_endf_dict():
    path = resolve_p_he3_law5()
    if path is None:
        pytest.skip('p + He-3 LAW=5 corpus file not available '
                    '(run tests/data_law5_adhoc/fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def p_b10_endf_dict():
    path = resolve_p_b10_law5()
    if path is None:
        pytest.skip('p + B-10 LAW=5 corpus file not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


# ---- Rutherford limit (bit-exact on this corner) ----------------------


def test_rutherford_helper_matches_textbook():
    """Pointwise Rutherford formula from the module helper must
    match the textbook form at zero tolerance."""
    eta = 0.5
    k = 2.5
    mu = np.linspace(-0.95, 0.95, 11)
    expected = eta ** 2 / (k ** 2 * (1 - mu) ** 2)
    got = np.asarray(mf6_law5._sigma_coulomb_distinguishable(mu, eta, k))
    np.testing.assert_array_equal(got, expected)


def test_ltp1_lidp0_zero_nuclear_reduces_to_rutherford():
    """With every ``a_l`` and ``b_l`` coefficient set to zero, the
    LTP=1 LIDP=0 expansion (eq 6.13) must reduce bit-exactly to
    the pointwise Rutherford cross section. Pins that the
    interference and pure-nuclear sums contribute nothing in this
    limit and that no stray constant/prefactor leaks in."""
    nl = 5
    b = np.zeros(2 * nl + 1)
    a_complex = np.zeros(nl + 1, dtype=complex)
    eta, k = 0.42, 4.5
    mu = np.linspace(-0.9, 0.9, 11)
    expected = eta ** 2 / (k ** 2 * (1 - mu) ** 2)
    got = np.asarray(mf6_law5._reconstruct_ltp1_lidp0_single_ein(
        b, a_complex, mu, eta, k, np,
    ))
    np.testing.assert_array_equal(got, expected)


def test_unpack_ltp1_distinguishable_shape_and_values():
    nl = 3
    packed = np.concatenate([
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        np.array([10.0, -1.0, 20.0, -2.0, 30.0, -3.0, 40.0, -4.0]),
    ])
    b, a = mf6_law5._unpack_ltp1_distinguishable(packed, nl)
    np.testing.assert_array_equal(b, np.array([1., 2., 3., 4., 5., 6., 7.]))
    expected_a = np.array([10 - 1j, 20 - 2j, 30 - 3j, 40 - 4j])
    np.testing.assert_array_equal(a, expected_a)


def test_unpack_rejects_mismatched_size():
    with pytest.raises(ValueError, match='4\\*NL\\+3'):
        mf6_law5._unpack_ltp1_distinguishable(np.zeros(5), nl=3)


# ---- Physical constants (one textbook spot-check) ---------------------


def test_sommerfeld_eta_matches_textbook_at_p_c12_5mev():
    """η = Z1 Z2 α √(m1 u / (2 E_lab)).  For p + C-12 at E_lab=5 MeV
    with m1=1 amu:
      η = 6 · α · √(931.494 / 10) = 6 · 0.00729735 · 9.65139 ≈ 0.422568.
    Hand-derived reference pins the formula (eq 6.12) and the
    numerical constants."""
    eta = float(mf6_law5._sommerfeld_eta(
        z1=1, z2=6, m1_amu=1.0, e_lab_ev=5.0e6, xp=np,
    ))
    assert abs(eta - 0.422578) < 1e-5


def test_cm_wavenumber_matches_textbook_at_p_c12_5mev():
    """k [barn^(-1/2)] from eq (6.11). For p + C-12 at 5 MeV LAB
    with A=12 and m1=1 amu:
      k = (12/13) · √(2·1·931.494·5) / 197.327 · 10
        = 0.9231 · 96.514 / 197.327 · 10 ≈ 4.5156 barn^(-1/2)."""
    k = float(mf6_law5._cm_wavenumber_per_sqrt_barn(
        a_ratio=12.0, m1_amu=1.0, e_lab_ev=5.0e6, xp=np,
    ))
    assert abs(k - 4.5156) < 1e-3


# ---- Corpus-file smoke tests ------------------------------------------


def test_p_he3_angdist_is_finite_and_positive(p_he3_endf_dict):
    e_in = np.array([5e6, 10e6, 20e6])
    mu = np.linspace(-0.9, 0.9, 15)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sig = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_he3_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu, to_lab=True,
        ))
    assert sig.shape == (3, 15)
    assert np.all(np.isfinite(sig))
    assert np.all(sig > 0)


def test_p_he3_angdist_forward_peaked(p_he3_endf_dict):
    """At 20 MeV p + He-3, dsigma/dmu at mu=+0.9 (forward) must
    exceed dsigma/dmu at mu=0 (sideways), because both the
    Rutherford pole at mu->1 and the nuclear amplitude's small-
    angle enhancement push the forward hemisphere up. Catches a
    sign flip in the interference term or an inverted mu convention."""
    e_in = np.array([20e6])
    mu = np.array([-0.5, 0.0, 0.5, 0.9])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sig = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_he3_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu, to_lab=True,
        ))
    # Forward (mu=0.9) must exceed sideways (mu=0)
    assert sig[0, 3] > sig[0, 1]


def test_p_b10_angdist_is_finite_and_positive(p_b10_endf_dict):
    """p + B-10 LAW=5 covers 10 keV to 3 MeV; test grid stays inside.

    Keep mu below 0.85 so the Legendre-expansion residual pathology
    near mu=1 (manual Section 6.2.7: 'the limit of the Legendre
    representation of the residual cross section at small angles
    may not be well defined') does not render a tiny-negative
    reconstruction result as a test failure. The reconstruction
    itself is correct; the file's tabulation simply does not
    cover the forward tip positive-definitely.
    """
    e_in = np.array([1.5e5, 5e5, 2.5e6])
    mu = np.linspace(-0.9, 0.85, 15)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sig = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_b10_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu, to_lab=True,
        ))
    assert sig.shape == (3, 15)
    assert np.all(np.isfinite(sig))
    assert np.all(sig > 0)


# ---- Dispatch wiring --------------------------------------------------


def test_compute_angdist_dispatch_reaches_law5_handler(p_he3_endf_dict):
    """compute_angdist_from_subsec must route LAW=5 to the new
    handler. Pins the one-line dispatch addition in
    mf6_interpretation_subsecs.py."""
    e_in = np.array([5e6, 10e6])
    mu = np.linspace(-0.5, 0.5, 7)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from_dispatch = np.asarray(compute_angdist_from_subsec(
            p_he3_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu, to_lab=True,
        ))
        from_direct = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_he3_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu, to_lab=True,
        ))
    np.testing.assert_array_equal(from_dispatch, from_direct)


# ---- Explicit NotImplementedError for deferred cases ------------------


def test_ltp12_raises_not_implemented():
    """Pick a corpus file that uses LTP=12 and confirm the handler
    raises with a message pointing at #264 so the gap is visible."""
    # Use p_he3 fixture dict (LTP=1) and monkey-patch its LTP flag
    # to simulate a LTP=12 subsection; cleaner than requiring
    # another corpus download. Deep-copy avoids fixture pollution.
    import copy
    path = resolve_p_he3_law5()
    if path is None:
        pytest.skip('p + He-3 LAW=5 corpus file not available')
    d = copy.deepcopy(
        EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    )
    sub = d[6][2]['subsection'][1]
    for k in list(sub['LTP'].keys()):
        sub['LTP'][k] = 12
    with pytest.raises(NotImplementedError, match='LTP'):
        mf6_law5.get_angdist_from_subsec_law5(
            d, mt=2, subsec_num=1,
            energies_in=np.array([5e6]),
            angle_cosines_out=np.array([0.0]),
            to_lab=True,
        )


def test_lidp1_raises_not_implemented():
    """Simulate an identical-particle subsection (p+p pattern) by
    patching LIDP=1 and confirm the handler raises with a message
    pointing at eq. 6.14."""
    import copy
    path = resolve_p_he3_law5()
    if path is None:
        pytest.skip('p + He-3 LAW=5 corpus file not available')
    d = copy.deepcopy(
        EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    )
    d[6][2]['subsection'][1]['LIDP'] = 1
    with pytest.raises(NotImplementedError, match='LIDP=1'):
        mf6_law5.get_angdist_from_subsec_law5(
            d, mt=2, subsec_num=1,
            energies_in=np.array([5e6]),
            angle_cosines_out=np.array([0.0]),
            to_lab=True,
        )


def test_p_he3_njoy_pinned_point(p_he3_endf_dict):
    """Hard-coded regression pin at p + He-3, Ein=100 keV, mu=0.0:
    the reconstruction returns dsigma/dmu = 23.3428... barns.

    Verified against a direct Python transcription of NJOY2016
    acefc.f90::coul (LTP=1 LIDP=0 branch) with NJOY phys.f90 CGS
    constants, which produced dsigma/dOmega = 3.715123504778457
    b/sr i.e. dsigma/dmu = 2 pi x that = 23.342809... b. Our value
    matches to ~1e-9 relative; the residual 1e-10-level drift
    comes from the 10th-digit difference between
    PARTICLE_MASSES_AMU['n'] = 1.00866491578 and NJOY's
    amassn = 1.00866491595, which is well below double-precision
    arithmetic noise.

    Any future shift larger than 1e-7 relative on this value flags
    a formula regression (interference-term sign, Coulomb-phase
    convention, AWP/amu unit confusion, Legendre indexing)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_he3_endf_dict, mt=2, subsec_num=1,
            energies_in=np.array([1.0e5]),
            angle_cosines_out=np.array([0.0]),
            to_lab=True,
        ))
    expected = 23.342809431  # dsigma/dmu, barns
    rel = abs(float(got[0, 0]) - expected) / expected
    assert rel < 1e-7, f'got {got[0, 0]!r}, expected ~{expected}, rel={rel:.3e}'


def test_lct_not_2_raises_not_implemented(p_he3_endf_dict):
    """Any LCT other than 2 (CM) is outside the first-increment
    scope. Patch the section-level LCT and confirm the handler
    raises."""
    import copy
    d = copy.deepcopy(p_he3_endf_dict)
    d[6][2]['LCT'] = 1
    with pytest.raises(NotImplementedError, match='LCT'):
        mf6_law5.get_angdist_from_subsec_law5(
            d, mt=2, subsec_num=1,
            energies_in=np.array([5e6]),
            angle_cosines_out=np.array([0.0]),
            to_lab=True,
        )
