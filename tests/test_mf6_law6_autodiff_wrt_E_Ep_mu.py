"""Phase-1 autodiff-coverage pins: ``jax.grad`` wrt query
incident energy E, outgoing energy E' and cosine mu through the
MF6 LAW=6 (n-body phase space) reconstruction.

LAW=6 is fully analytic per the ENDF-6 manual sec. 6.2.7 /
Kalbach LA-13166: closed-form kinematic bookkeeping with no
per-Ein interpolation table or panel lookup. Every arithmetic op
in ``_law6_kernel`` flows through the ``xp`` adapter so tracers
on any of the three query axes propagate end-to-end.

Uses H-2 (deuterium) MT=16 (n,2n) subsec 1 from
``tests/data/n-001_H_002.endf`` (committed, small). npsx=3
(two outgoing neutrons + recoiling H), so the reconstruction
hits the C_3 branch of the C_n prefactor. Threshold ~3 MeV, so
query Es from 4 MeV upward stay above threshold; E' and mu are
picked inside the kinematically allowed region.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_subsecs as mf6subsec,
)
from endf_userpy.primitives import array_ns


DATA_DIR = Path(__file__).parent / 'data'


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def h2_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-001_H_002.endf'),
    )


def _fd5(f, x, h):
    return (-float(f(x + 2 * h)) + 8 * float(f(x + h))
            - 8 * float(f(x - h)) + float(f(x - 2 * h))) / (12 * h)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law6_numpy_jax_parity(h2_endf_dict):
    """xp=jax and xp=numpy agree to machine precision on a query
    grid spanning several incident energies, outgoing energies,
    and cosines inside the LAW=6 kinematic support."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([5.0e6, 8.0e6, 1.2e7, 1.8e7], dtype=np.float64)
    eout = np.linspace(1e5, 5e6, 15, dtype=np.float64)
    mu = np.linspace(-0.9, 0.9, 11, dtype=np.float64)

    f_np = np.asarray(mf6subsec.get_dist2d_from_subsec_law6(
        h2_endf_dict, 16, 1, ein, eout, mu, to_lab=True, xp=xp_np,
    ))
    f_jx = np.asarray(mf6subsec.get_dist2d_from_subsec_law6(
        h2_endf_dict, 16, 1,
        jnp.asarray(ein), jnp.asarray(eout), jnp.asarray(mu),
        to_lab=True, xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('E_val', [5.0e6, 8.0e6, 1.2e7, 1.8e7])
def test_law6_grad_wrt_E_matches_fd(h2_endf_dict, E_val):
    """jax.grad(sum(f(E, E', mu)))(E) matches central FD. E enters
    through E_i^max, E_s, and (via the C_n prefactor) the overall
    normalisation."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    eout = jnp.linspace(1e5, 5e6, 15)
    mu = jnp.linspace(-0.9, 0.9, 11)

    def loss(E_scalar):
        return jnp.sum(mf6subsec.get_dist2d_from_subsec_law6(
            h2_endf_dict, 16, 1,
            jnp.array([E_scalar]), eout, mu,
            to_lab=True, xp=xp_jx,
        ))

    grad = float(jax.grad(loss)(jnp.array(E_val)))
    fd = _fd5(loss, jnp.array(E_val), E_val * 1e-4)
    assert np.isfinite(grad), f'grad not finite at E={E_val}'
    if abs(fd) < 1e-30:
        return
    np.testing.assert_allclose(
        grad, fd, rtol=5e-3, atol=1e-30,
        err_msg=f'LAW=6 grad wrt E={E_val}: '
                f'ad={grad:.4e} fd={fd:.4e}',
    )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('Ep_val', [3e5, 1e6, 2e6, 3e6])
def test_law6_grad_wrt_Ep_matches_fd(h2_endf_dict, Ep_val):
    """jax.grad(f(E, E', mu))(E') matches central FD at E's inside
    the kinematic support at fixed E and mu grid."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([1.2e7])
    mu = jnp.linspace(-0.9, 0.9, 11)

    def loss(Ep_scalar):
        return jnp.sum(mf6subsec.get_dist2d_from_subsec_law6(
            h2_endf_dict, 16, 1,
            ein, jnp.array([Ep_scalar]), mu,
            to_lab=True, xp=xp_jx,
        ))

    grad = float(jax.grad(loss)(jnp.array(Ep_val)))
    fd = _fd5(loss, jnp.array(Ep_val), Ep_val * 1e-4)
    assert np.isfinite(grad), f'grad not finite at Ep={Ep_val}'
    if abs(fd) < 1e-30:
        return
    np.testing.assert_allclose(
        grad, fd, rtol=5e-3, atol=1e-30,
        err_msg=f'LAW=6 grad wrt Ep={Ep_val}: '
                f'ad={grad:.4e} fd={fd:.4e}',
    )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('mu_val', [-0.6, -0.2, 0.2, 0.6])
def test_law6_grad_wrt_mu_matches_fd(h2_endf_dict, mu_val):
    """jax.grad(f(E, E', mu))(mu) matches central FD at mus in the
    interior of the allowed range. mu enters E'_c and hence the
    (E_i^max - E'_c)^power factor."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([1.2e7])
    eout = jnp.linspace(3e5, 3e6, 10)

    def loss(mu_scalar):
        return jnp.sum(mf6subsec.get_dist2d_from_subsec_law6(
            h2_endf_dict, 16, 1,
            ein, eout, jnp.array([mu_scalar]),
            to_lab=True, xp=xp_jx,
        ))

    grad = float(jax.grad(loss)(jnp.array(mu_val)))
    fd = _fd5(loss, jnp.array(mu_val), 1e-4)
    assert np.isfinite(grad), f'grad not finite at mu={mu_val}'
    if abs(fd) < 1e-30:
        return
    np.testing.assert_allclose(
        grad, fd, rtol=5e-3, atol=1e-30,
        err_msg=f'LAW=6 grad wrt mu={mu_val}: '
                f'ad={grad:.4e} fd={fd:.4e}',
    )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law6_grad_wrt_E_vector_matches_fd_elementwise(h2_endf_dict):
    """Vector grad wrt (n_ein,) across multiple incident energies
    matches per-E FD element-wise. Pins that the E tracer routes
    to each E's contribution independently."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    Es_np = np.array([5.0e6, 8.0e6, 1.2e7, 1.8e7])
    Es = jnp.asarray(Es_np)
    # Match the scalar-grad test's eout range so we stay in the
    # same regime; a narrower window can land the grad wrt E near
    # a kinematic zero-crossing where FD noise dominates.
    eout = jnp.linspace(1e5, 5e6, 15)
    mu = jnp.linspace(-0.9, 0.9, 11)

    def loss(E):
        return jnp.sum(mf6subsec.get_dist2d_from_subsec_law6(
            h2_endf_dict, 16, 1, E, eout, mu,
            to_lab=True, xp=xp_jx,
        ))

    grad_vec = np.asarray(jax.grad(loss)(Es))
    assert grad_vec.shape == (len(Es_np),)

    for i, E_val in enumerate(Es_np):
        def loss_i(E, i=i):
            return loss(Es.at[i].set(E[0]))
        fd = _fd5(loss_i, jnp.array([E_val]), E_val * 1e-4)
        assert np.isfinite(grad_vec[i])
        if abs(fd) < 1e-30:
            continue
        np.testing.assert_allclose(
            grad_vec[i], fd, rtol=5e-3, atol=1e-30,
            err_msg=f'i={i} E={E_val}: '
                    f'ad={grad_vec[i]:.4e} fd={fd:.4e}',
        )
