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
    """Hard-coded regression pin at p + He-3, Ein=100 keV,
    mu_CM=0.0: the CM-frame reconstruction returns
    dsigma/dmu = 23.3428... barns.

    NJOY2016 acefc.f90::coul (LTP=1 LIDP=0 branch) with NJOY
    phys.f90 CGS constants produced dsigma/dOmega = 3.715123504778457
    b/sr; multiplied by 2 pi for the per-mu convention gives
    23.342809... b. Pinned against the NJOY output in the stored
    (CM) frame; the ``to_lab=True`` branch adds the two-body
    kinematic Jacobian and is covered by a separate test.

    Any shift larger than 1e-7 relative on this value flags a
    regression in the CM-frame formula (interference-term sign,
    Coulomb-phase convention, AWP/amu unit confusion, Legendre
    indexing)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_he3_endf_dict, mt=2, subsec_num=1,
            energies_in=np.array([1.0e5]),
            angle_cosines_out=np.array([0.0]),
            to_lab=False,
        ))
    expected = 23.342809431  # dsigma/dmu, barns, CM frame
    rel = abs(float(got[0, 0]) - expected) / expected
    assert rel < 1e-7, f'got {got[0, 0]!r}, expected ~{expected}, rel={rel:.3e}'


def test_lab_differs_from_cm_on_light_target(p_he3_endf_dict):
    """On a light target (He-3, awr~3), the CM-to-LAB Jacobian
    shifts the angular shape by several percent at interior mu.
    Pins the Jacobian is actually applied for ``to_lab=True``
    (not a no-op) by requiring the LAB and CM outputs to differ
    materially at mu=0.5, Ein=1 MeV: a user who passed to_lab=True
    before #334 landed got the CM-frame result labelled as LAB
    (a bug the first-increment UserWarning documented)."""
    e_in = np.array([1.0e6])
    mu = np.array([0.5])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cm = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_he3_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu, to_lab=False,
        ))
        lab = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_he3_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu, to_lab=True,
        ))
    rel = abs(float(lab[0, 0]) - float(cm[0, 0])) / float(cm[0, 0])
    assert rel > 0.05, (
        f'to_lab=True must differ from to_lab=False on p+He-3 by '
        f'more than ~5pct at mu=0.5, Ein=1 MeV; got rel={rel:.3e}'
    )


def test_lab_frame_matches_textbook_two_body_jacobian(p_he3_endf_dict):
    """Validate the CM-to-LAB conversion against a textbook two-body
    elastic Jacobian, independent of the handler's own primitives.

    Non-relativistic two-body elastic kinematics for a projectile of
    mass m1 scattering off a stationary target of mass m2 (A = m2/m1):

        mu_LAB = (A mu_CM + 1) / sqrt(A^2 + 2 A mu_CM + 1)

    Differentiating with xw = A^2 + 2 A mu_CM + 1 gives

        d mu_LAB / d mu_CM = A^2 (A + mu_CM) / xw^(3/2)

    and so, by azimuthal symmetry and invariance of dsigma,

        dsigma/dmu_LAB = dsigma/dmu_CM * (d mu_CM / d mu_LAB)
                      = dsigma/dmu_CM * xw^(3/2) / (A^2 |A + mu_CM|)

    which exactly matches the ``convert_angdist_to_labsys`` form.
    Pin the handler against this textbook identity so a regression
    on either side (the handler, or the shared primitive) surfaces.

    Cross-checked at a smooth interior mu on p + He-3 (A ~ 3), far
    from mu = 1 to avoid the Coulomb singularity.
    """
    sec = p_he3_endf_dict[6][2]
    sub = sec['subsection'][1]
    awp = float(sub['AWP'])
    awr = float(sec['AWR'])
    A = awr / awp  # target / projectile mass ratio
    e_in = np.array([5.0e6])  # 5 MeV, interior point not on a knot
    mu_cm_target = np.array([0.3])
    denom = np.sqrt(A ** 2 + 2 * A * mu_cm_target + 1.0)
    mu_lab_target = (A * mu_cm_target + 1.0) / denom
    xw = 1.0 + 2.0 * A * mu_cm_target + A ** 2
    jacobian = xw * np.sqrt(xw) / (A ** 2 * np.abs(A + mu_cm_target))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got_cm = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_he3_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu_cm_target,
            to_lab=False,
        ))
        got_lab = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_he3_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu_lab_target,
            to_lab=True,
        ))

    expected_lab = got_cm[0, 0] * jacobian[0]
    rel = abs(got_lab[0, 0] - expected_lab) / expected_lab
    assert rel < 1e-10, (
        f'LAB from handler = {got_lab[0, 0]:.6e}  '
        f'textbook CM * J = {expected_lab:.6e}  rel={rel:.3e}'
    )


def test_lab_equals_cm_in_heavy_target_limit(p_he3_endf_dict):
    """In the limit awr >> awp the CM and LAB frames coincide; a
    user querying a heavy-target evaluation should get the same
    numeric from to_lab=True and to_lab=False. We don't have a
    neutron-adjacent corpus file heavier than Bi-209, so forge a
    synthetic one by patching awr to a very large value and
    re-running on p+He-3's own coefficients. The CM / LAB
    difference should fall below 1e-3 relative at interior mu."""
    import copy
    d = copy.deepcopy(p_he3_endf_dict)
    d[6][2]['AWR'] = 1.0e6  # effectively infinite target mass
    e_in = np.array([1.0e6])
    mu = np.array([-0.5, 0.0, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cm = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            d, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu, to_lab=False,
        ))
        lab = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            d, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu, to_lab=True,
        ))
    np.testing.assert_allclose(lab, cm, rtol=1e-3)


def _jax_available():
    from endf_userpy.primitives import array_ns
    return 'jax' in array_ns.available_backends()


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jit_matches_numpy(p_he3_endf_dict):
    """``@jax.jit`` over the full handler must trace cleanly (no
    tracer-to-numpy conversion) and return the same value as the
    numpy path. Catches a regression on the vectorise-and-keep-
    tracers path."""
    import jax
    import jax.numpy as jnp
    from endf_userpy.primitives import array_ns
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')
    e_test = np.array([5.0e5, 1.0e6, 1.5e6])
    mu_test = np.array([-0.5, 0.0, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out_np = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_he3_endf_dict, mt=2, subsec_num=1,
            energies_in=e_test, angle_cosines_out=mu_test, to_lab=True,
            xp=xp_np,
        ))

        @jax.jit
        def fn(e):
            return mf6_law5.get_angdist_from_subsec_law5(
                p_he3_endf_dict, mt=2, subsec_num=1,
                energies_in=e, angle_cosines_out=jnp.asarray(mu_test),
                to_lab=True, xp=xp_jax,
            )
        out_jit = np.asarray(fn(jnp.asarray(e_test)))
    np.testing.assert_allclose(out_jit, out_np, rtol=1e-10, atol=0)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_grad_wrt_ein_matches_finite_difference(p_he3_endf_dict):
    """``jax.grad`` wrt the incident energy must flow through the
    section-level TAB2 interpolation of the coefficient matrix, the
    eta / k dependence on E, and the Coulomb phase factor. Pinned
    against central FD at a smooth interior point."""
    import jax
    import jax.numpy as jnp
    from endf_userpy.primitives import array_ns
    xp_jax = array_ns.get_backend('jax')
    # Interior Ein point (not a stored knot: the TAB2 panel-boundary
    # kink gives one-sided AD vs symmetric FD at a knot). He-3's
    # LAW=5 stored grid has knots at 1.0 / 1.5 / 2.0 MeV; 1.4 MeV
    # sits inside a lin-lin panel.
    e_center = 1.4e6
    mu_center = 0.3

    def loss(e_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = mf6_law5.get_angdist_from_subsec_law5(
                p_he3_endf_dict, mt=2, subsec_num=1,
                energies_in=jnp.array([e_scalar]),
                angle_cosines_out=jnp.array([mu_center]),
                to_lab=True, xp=xp_jax,
            )
        return jnp.sum(out)

    g_ad = float(jax.grad(loss)(e_center))
    step = e_center * 1e-5
    g_fd = (float(loss(e_center + step)) - float(loss(e_center - step))) / (2 * step)
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
    assert rel < 1e-4, f'AD={g_ad:.6e}  FD={g_fd:.6e}  rel={rel:.3e}'


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_grad_wrt_mu_matches_finite_difference(p_he3_endf_dict):
    """``jax.grad`` wrt an output cosine must flow through the
    Legendre recurrence, the 1/(1-mu) prefactors in both the
    Rutherford term and the interference, and the Coulomb phase
    factor's log((1-mu)/2)."""
    import jax
    import jax.numpy as jnp
    from endf_userpy.primitives import array_ns
    xp_jax = array_ns.get_backend('jax')
    e_center = 1.0e6
    mu_center = 0.3

    def loss(mu_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = mf6_law5.get_angdist_from_subsec_law5(
                p_he3_endf_dict, mt=2, subsec_num=1,
                energies_in=jnp.array([e_center]),
                angle_cosines_out=jnp.array([mu_scalar]),
                to_lab=True, xp=xp_jax,
            )
        return jnp.sum(out)

    g_ad = float(jax.grad(loss)(mu_center))
    step = 1e-5
    g_fd = (float(loss(mu_center + step)) - float(loss(mu_center - step))) / (2 * step)
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
    assert rel < 1e-5, f'AD={g_ad:.6e}  FD={g_fd:.6e}  rel={rel:.3e}'


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_grad_wrt_file_leaf_coefficient_matches_finite_difference(
    p_he3_endf_dict,
):
    """``jax.grad`` wrt an injected tracer on ``subsec['A']`` (the
    file-stored b_l / a_l coefficients) must propagate end-to-end.
    This is the file-leaf autodiff pattern used elsewhere in the
    codebase (e.g. MF6 LAW=1 b_panels tracers). Pins the xp.stack
    build of coef_matrix: an earlier version that packed the
    coefficients into a numpy array silently collapsed any tracer
    into a concrete float and broke this gradient path."""
    import copy
    import jax
    import jax.numpy as jnp
    from endf_userpy.primitives import array_ns
    xp_jax = array_ns.get_backend('jax')
    orig = float(p_he3_endf_dict[6][2]['subsection'][1]['A'][1][1])
    e_val = sorted(p_he3_endf_dict[6][2]['subsection'][1]['E'].values())[0]

    def loss(coef_val):
        d2 = copy.deepcopy(p_he3_endf_dict)
        d2[6][2]['subsection'][1]['A'][1][1] = coef_val
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = mf6_law5.get_angdist_from_subsec_law5(
                d2, mt=2, subsec_num=1,
                energies_in=jnp.array([e_val]),
                angle_cosines_out=jnp.array([0.0]),
                to_lab=True, xp=xp_jax,
            )
        return jnp.sum(out)

    g_ad = float(jax.grad(loss)(orig))
    # For b_0 at mu=0, the analytic gradient is exactly pi
    # (coefficient enters as 0.5 * b_0 * P_0(0) and we multiply by
    # 2*pi for dsigma/dmu). FD for cross-check anyway.
    step = max(abs(orig), 1e-6) * 1e-4
    g_fd = (float(loss(orig + step)) - float(loss(orig - step))) / (2 * step)
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
    assert rel < 1e-4, f'AD={g_ad:.6e}  FD={g_fd:.6e}  rel={rel:.3e}'


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
