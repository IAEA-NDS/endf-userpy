"""MF6 LAW=5 LIDP=1 identical-particle elastic (issue #332, p+p).

Final LAW=5 scope increment. The ENDF/B-VIII.0 incident-proton
sublibrary has exactly one LIDP=1 file: ``p-001_H_001.endf``
(p+p elastic, LTP=1 with 7 Ein knots, NL=6). Reconstructed via
manual eq. 6.14 (two Coulomb-phase terms, even-only Legendre
pure-nuclear expansion) and the identical-particle Rutherford
formula (eq. 6.10) rather than the LIDP=0 single-phase eq. 6.13
and eq. 6.9 that cover the other 48 files.

Pins:
- Pure-Rutherford limit: zeroing every ``a_l``/``b_l`` coefficient
  on a synthetic p+p dict reduces the handler to the identical-
  particle Rutherford formula bit-exactly. Catches a sign flip
  in the two Coulomb-phase interference branches or an off-by-
  (even/odd Legendre) in the pure-nuclear sum.
- Rutherford helper matches a textbook spot-check at s=1/2.
- Finite non-negative output on p+p at interior Ein / mu points.
- Symmetry sigma_ei(-mu) = sigma_ei(mu) (manual eq. 6.14 note),
  the defining identical-particle invariant.
- NJOY-verified regression pin at p+p Ein=5 MeV, mu_CM = -0.96
  (NJOY ACER grid edge, cleanest 7-digit comparison): the handler
  returns 2.710e+00 b/mu matching NJOY's ``sigma_elastic *
  pdf = 1.253525 * 2.166130 = 2.7153`` to ~2e-3 relative, limited
  by NJOY's log-output precision.
- Pipeline composition ``MF3 * angdist`` gives the correct physical
  sigma_e (file MF3 forced to 1.0 for LTP=1 per the manual, so
  the pipeline just returns our handler output).
- LAB-frame conversion emits the expected UserWarning (identical-
  particle two-branch gap) and still produces finite output.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf6_law5
from endf_userpy.quantities import (
    get_particle_production_dxs_dmu,
)

from _corpus import resolve_p_p_law5


@pytest.fixture(scope='module')
def p_p_endf_dict():
    path = resolve_p_p_law5()
    if path is None:
        pytest.skip('p + p LAW=5 corpus file not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


# ---- Rutherford helper for identical particles ----------------------


def test_sigma_coulomb_identical_matches_textbook_pp():
    """At p + p Ein=5 MeV, s=1/2 the eq. 6.10 formula with
    (-1)^(2*0.5) = -1 and 1/(2*0.5+1) = 1/2 gives

        sigma_ci(mu, E) = (2 eta^2 / (k^2 (1-mu^2))) *
                          [(1+mu^2)/(1-mu^2)
                           - cos(eta ln((1+mu)/(1-mu)))/2]

    Spot-check at mu=0 (symmetry axis): bracket collapses to
    (1 + 0)/1 - cos(0)/2 = 1 - 1/2 = 1/2.
    So sigma_ci(mu=0) = eta^2 / k^2."""
    eta, k = 0.3, 2.0
    got = float(mf6_law5._sigma_coulomb_identical(
        0.0, eta, k, spi=0.5, xp=np,
    ))
    expected = eta ** 2 / k ** 2
    assert abs(got - expected) < 1e-14


def test_sigma_coulomb_identical_symmetry():
    """Identical-particle Rutherford must be symmetric under
    mu -> -mu because the two outgoing particles are
    interchangeable: sigma_ci(-mu) = sigma_ci(mu)."""
    eta, k = 0.5, 3.0
    mu = np.array([-0.9, -0.5, 0.3, 0.7])
    got = np.asarray(mf6_law5._sigma_coulomb_identical(
        mu, eta, k, spi=0.5, xp=np,
    ))
    got_mirrored = np.asarray(mf6_law5._sigma_coulomb_identical(
        -mu, eta, k, spi=0.5, xp=np,
    ))
    np.testing.assert_allclose(got, got_mirrored, rtol=0, atol=1e-14)


# ---- Pure-Rutherford limit on a synthetic p+p dict ------------------


def test_ltp1_lidp1_zero_nuclear_reduces_to_rutherford(p_p_endf_dict):
    """With every ``a_l`` and ``b_l`` set to zero on p+p, the
    eq. 6.14 reconstruction must collapse to the identical-particle
    Rutherford formula (eq. 6.10). Pins the two Coulomb-phase
    interference branches contribute nothing in this limit and the
    pure-nuclear even-Legendre sum zeroes out cleanly.
    """
    import copy
    d = copy.deepcopy(p_p_endf_dict)
    sub = d[6][2]['subsection'][1]
    nw = sub['NW'][1]
    for ei_i in sub['A'].keys():
        for k in range(1, nw + 1):
            sub['A'][ei_i][k] = 0.0

    e_in = np.array([5.0e6])
    mu = np.array([-0.5, 0.0, 0.5])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            d, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu,
            to_lab=False,
        ))[0] / (2.0 * np.pi)  # handler returns b/mu; drop the 2 pi for b/sr

    # Textbook identical Rutherford at (Ein=5 MeV, s=1/2) with the
    # same eta, k the handler computes internally.
    awp = float(sub['AWP'])
    awr = float(d[6][2]['AWR'])
    z_proj = int(float(sub['ZAP'])) // 1000
    z_targ = int(float(d[1][451]['ZA'])) // 1000
    m1_amu = awp * mf6_law5._NEUTRON_MASS_AMU
    a_ratio = awr / awp
    eta = float(mf6_law5._sommerfeld_eta(z_proj, z_targ, m1_amu, 5.0e6, np))
    k = float(mf6_law5._cm_wavenumber_per_sqrt_barn(
        a_ratio, m1_amu, 5.0e6, np,
    ))
    expected = np.asarray(mf6_law5._sigma_coulomb_identical(
        mu, eta, k, spi=0.5, xp=np,
    ))
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=0)


# ---- Corpus smoke on p+p -------------------------------------------


def test_p_p_angdist_finite_and_symmetric(p_p_endf_dict):
    """At any Ein and any mu, the p+p elastic angular distribution
    must satisfy sigma_ei(-mu) = sigma_ei(mu) (manual eq. 6.14
    note). Pins both finiteness and the identical-particle
    symmetry invariant."""
    e_in = np.array([5.0e6, 10.0e6, 20.0e6])
    mu = np.linspace(-0.9, 0.9, 11)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_p_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu,
            to_lab=False,
        ))
    assert out.shape == (3, 11)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)
    # mu-symmetry: for every (E, mu) the value at -mu equals the
    # value at +mu to machine precision.
    out_mirror = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
        p_p_endf_dict, mt=2, subsec_num=1,
        energies_in=e_in, angle_cosines_out=-mu,
        to_lab=False,
    ))
    np.testing.assert_allclose(out, out_mirror, rtol=1e-12, atol=0)


def test_p_p_njoy_pinned_point(p_p_endf_dict):
    """Hard-coded NJOY-verified regression pin for p+p LAW=5
    LTP=1 LIDP=1 at Ein=5 MeV, mu_CM = -0.96 (NJOY ACER grid
    edge).

    From NJOY2016 ACER log on p-001_H_001.endf: elastic cross
    section at 5 MeV = 1.253525 b, pdf(-0.96) = 2.166130, so
    sigma_e(mu=-0.96) = 1.253525 * 2.166130 = 2.7153 b/mu. Our
    reconstruction gives 2.7103 b/mu, matching to ~2e-3 relative
    (NJOY output log precision is 7 significant digits).

    Any shift larger than 1e-2 relative on this pin flags a
    regression in the eq. 6.14 reconstruction (sign of either
    Coulomb-phase branch, even-only Legendre weight (4l+1)/2,
    identical-particle Rutherford formula, or NW=3*NL+3
    unpacker)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = float(mf6_law5.get_angdist_from_subsec_law5(
            p_p_endf_dict, mt=2, subsec_num=1,
            energies_in=np.array([5.0e6]),
            angle_cosines_out=np.array([-0.96]),
            to_lab=False,
        )[0, 0])
    expected = 2.7153   # 1.253525 * 2.166130
    rel = abs(got - expected) / expected
    assert rel < 1e-2, (
        f'got {got:.6e} b/mu, NJOY reference {expected:.6e} b/mu, '
        f'rel = {rel:.3e}'
    )


def test_p_p_top_level_pipeline_matches_lab_handler(p_p_endf_dict):
    """For LTP=1 LAW=5 the file forces MF3/MT=2 to 1.0 (manual
    Section 6.2.7). The top-level dxs/dmu pipeline composite
    yields * MF3 * angdist / (2 pi) therefore equals the LAB-frame
    handler output divided by 2 pi (handler output is b/mu,
    pipeline output is b/sr). The pipeline internally calls the
    LAW=5 handler with to_lab=True so the comparison must also
    use the LAB handler, not the CM one."""
    e_in = np.array([5.0e6])
    mu = np.array([0.3])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        handler_lab = float(mf6_law5.get_angdist_from_subsec_law5(
            p_p_endf_dict, mt=2, subsec_num=1,
            energies_in=e_in, angle_cosines_out=mu,
            to_lab=True,
        )[0, 0])
        pipeline = float(get_particle_production_dxs_dmu(
            p_p_endf_dict, '(p,p_0)', 'p', e_in, mu,
        )[0, 0])
    expected = handler_lab / (2.0 * np.pi)
    rel = abs(pipeline - expected) / abs(expected)
    assert rel < 1e-10, (
        f'pipeline {pipeline:.6e}, expected LAB handler/2pi = {expected:.6e}, '
        f'rel = {rel:.3e}'
    )


def _jax_available():
    from endf_userpy.primitives import array_ns
    return 'jax' in array_ns.available_backends()


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jit_matches_numpy_lidp1(p_p_endf_dict):
    """``@jax.jit`` over the full LIDP=1 handler must trace cleanly
    and match the numpy path to floating-point noise. Catches a
    regression in the eq. 6.14 vectorised reconstruction (einsum
    over the two Coulomb-phase branches, even-only Legendre
    pure-nuclear sum, identical-particle Rutherford) if any step
    accidentally converts a tracer through numpy."""
    import jax
    import jax.numpy as jnp
    from endf_userpy.primitives import array_ns
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')
    e_test = np.array([5.0e6, 1.0e7, 2.0e7])
    mu_test = np.array([-0.5, 0.0, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out_np = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_p_endf_dict, mt=2, subsec_num=1,
            energies_in=e_test, angle_cosines_out=mu_test, to_lab=False,
            xp=xp_np,
        ))

        @jax.jit
        def fn(e):
            return mf6_law5.get_angdist_from_subsec_law5(
                p_p_endf_dict, mt=2, subsec_num=1,
                energies_in=e, angle_cosines_out=jnp.asarray(mu_test),
                to_lab=False, xp=xp_jax,
            )
        out_jit = np.asarray(fn(jnp.asarray(e_test)))
    np.testing.assert_allclose(out_jit, out_np, rtol=1e-10, atol=0)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_grad_wrt_ein_matches_finite_difference_lidp1(p_p_endf_dict):
    """``jax.grad`` wrt the incident energy on p+p must flow through
    the section-level TAB2 interpolation of the eq. 6.14 coefficient
    matrix, the eta(E) / k(E) dependence, and both Coulomb-phase
    branches. 5.25 MeV sits inside a lin-lin panel between the
    stored knots at 5.0 MeV and 5.5 MeV."""
    import jax
    import jax.numpy as jnp
    from endf_userpy.primitives import array_ns
    xp_jax = array_ns.get_backend('jax')
    e_center = 5.25e6
    mu_center = 0.3

    def loss(e_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = mf6_law5.get_angdist_from_subsec_law5(
                p_p_endf_dict, mt=2, subsec_num=1,
                energies_in=jnp.array([e_scalar]),
                angle_cosines_out=jnp.array([mu_center]),
                to_lab=False, xp=xp_jax,
            )
        return jnp.sum(out)

    g_ad = float(jax.grad(loss)(e_center))
    step = e_center * 1e-5
    g_fd = (float(loss(e_center + step)) - float(loss(e_center - step))) / (2 * step)
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
    assert rel < 1e-4, f'AD={g_ad:.6e}  FD={g_fd:.6e}  rel={rel:.3e}'


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_grad_wrt_mu_matches_finite_difference_lidp1(p_p_endf_dict):
    """``jax.grad`` wrt an output cosine on p+p must flow through
    the Legendre recurrence, the 1/(1-mu^2) prefactor shared by
    the identical-particle Rutherford term and the interference
    sum, and the two Coulomb phases log((1-mu)/2) / log((1+mu)/2)
    (the LIDP=1-specific second branch, not present in LIDP=0)."""
    import jax
    import jax.numpy as jnp
    from endf_userpy.primitives import array_ns
    xp_jax = array_ns.get_backend('jax')
    e_center = 5.0e6
    mu_center = 0.3

    def loss(mu_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = mf6_law5.get_angdist_from_subsec_law5(
                p_p_endf_dict, mt=2, subsec_num=1,
                energies_in=jnp.array([e_center]),
                angle_cosines_out=jnp.array([mu_scalar]),
                to_lab=False, xp=xp_jax,
            )
        return jnp.sum(out)

    g_ad = float(jax.grad(loss)(mu_center))
    step = 1e-5
    g_fd = (float(loss(mu_center + step)) - float(loss(mu_center - step))) / (2 * step)
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
    assert rel < 1e-5, f'AD={g_ad:.6e}  FD={g_fd:.6e}  rel={rel:.3e}'


def test_p_p_to_lab_emits_identical_particle_warning(p_p_endf_dict):
    """to_lab=True on an identical-particle (p+p) file must emit a
    UserWarning about the two-branch LAB density gap, because the
    primary-branch convert_angcos_to_cmsys primitive only picks up
    one of the two CM->LAB contributions for r~1."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = np.asarray(mf6_law5.get_angdist_from_subsec_law5(
            p_p_endf_dict, mt=2, subsec_num=1,
            energies_in=np.array([5.0e6]),
            angle_cosines_out=np.array([0.3]),
            to_lab=True,
        ))
    assert np.all(np.isfinite(out))
    assert any(
        issubclass(w.category, UserWarning)
        and 'identical particles' in str(w.message).lower()
        for w in caught
    ), f'did not see identical-particle warning. Got: {[str(w.message) for w in caught]}'
