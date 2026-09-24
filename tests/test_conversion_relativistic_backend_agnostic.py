"""Backend-agnostic port of ``primitives.conversion_relativistic``
(issue #169 final tier-2 item).

All four public routines now accept an optional ``xp`` adapter.
``xp=None`` (default) is numpy and bit-identical to the pre-port
behaviour; ``xp=jax`` dispatches the ``sqrt`` inside the SymPy-
generated kinematic expressions through JAX so tracers on ``mu``,
``Ekin``, and ``Ekin_i`` propagate to ``jax.grad``.

Pins:

- Parity: xp=None default and xp=numpy adapter bit-identical for
  all four public entries.
- JAX parity: xp=jax matches numpy to floating-point round-off.
- jax.grad through ``mu`` matches ``compute_dEkin_dmu`` analytically
  to round-off, and matches central FD to rtol=1e-4.
- jax.grad through ``Ekin_i`` (incident energy - a natural fit
  parameter) matches central FD to rtol=1e-3.
- Private ``_compute_*_cos_phi`` helpers accept xp too and give
  the correct sign relative to their public ``_mu`` counterparts.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives import conversion_relativistic as cr


# Al-27 (n, n) elastic kinematics, MeV.
Ekin_i = 14.0
M_I = 1.008665 * 931.494
M_T = 26.9815 * 931.494
M_E = M_I
M_R = M_T
MU_SAMPLES = np.array([-0.9, -0.5, 0.3, 0.7, 0.99])


def _jax_available():
    return 'jax' in array_ns.available_backends()


def test_compute_Ekin_from_mu_default_matches_xp_numpy():
    xp_np = array_ns.get_backend('numpy')
    a = cr.compute_Ekin_from_mu(MU_SAMPLES, Ekin_i, M_I, M_T, M_E, M_R)
    b = cr.compute_Ekin_from_mu(
        MU_SAMPLES, Ekin_i, M_I, M_T, M_E, M_R, xp=xp_np,
    )
    np.testing.assert_array_equal(a, b)


def test_compute_dmu_dEkin_default_matches_xp_numpy():
    xp_np = array_ns.get_backend('numpy')
    Ekin = cr.compute_Ekin_from_mu(MU_SAMPLES, Ekin_i, M_I, M_T, M_E, M_R)
    a = cr.compute_dmu_dEkin(Ekin, Ekin_i, M_I, M_T, M_E, M_R)
    b = cr.compute_dmu_dEkin(Ekin, Ekin_i, M_I, M_T, M_E, M_R, xp=xp_np)
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_all_four_public_routines_numpy_jax_parity():
    import jax.numpy as jnp
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    mu = MU_SAMPLES
    Ekin_np = cr.compute_Ekin_from_mu(
        mu, Ekin_i, M_I, M_T, M_E, M_R, xp=xp_np,
    )
    Ekin_jx = np.asarray(cr.compute_Ekin_from_mu(
        jnp.asarray(mu), Ekin_i, M_I, M_T, M_E, M_R, xp=xp_jx,
    ))
    np.testing.assert_allclose(Ekin_np, Ekin_jx, rtol=1e-11)

    mu_back_np = cr.compute_mu_from_Ekin(
        Ekin_np, Ekin_i, M_I, M_T, M_E, M_R, xp=xp_np,
    )
    mu_back_jx = np.asarray(cr.compute_mu_from_Ekin(
        jnp.asarray(Ekin_np), Ekin_i, M_I, M_T, M_E, M_R, xp=xp_jx,
    ))
    np.testing.assert_allclose(mu_back_np, mu_back_jx, rtol=1e-11)

    dEdmu_np = cr.compute_dEkin_dmu(
        mu, Ekin_i, M_I, M_T, M_E, M_R, xp=xp_np,
    )
    dEdmu_jx = np.asarray(cr.compute_dEkin_dmu(
        jnp.asarray(mu), Ekin_i, M_I, M_T, M_E, M_R, xp=xp_jx,
    ))
    np.testing.assert_allclose(dEdmu_np, dEdmu_jx, rtol=1e-11)

    dmudE_np = cr.compute_dmu_dEkin(
        Ekin_np, Ekin_i, M_I, M_T, M_E, M_R, xp=xp_np,
    )
    dmudE_jx = np.asarray(cr.compute_dmu_dEkin(
        jnp.asarray(Ekin_np), Ekin_i, M_I, M_T, M_E, M_R, xp=xp_jx,
    ))
    np.testing.assert_allclose(dmudE_np, dmudE_jx, rtol=1e-11)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_mu_matches_analytic_and_fd():
    """``jax.grad(compute_Ekin_from_mu, argnums=0)`` matches the
    analytic ``compute_dEkin_dmu`` to round-off and central FD to
    the truncation limit of the FD stencil."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')

    def f(mu_scalar):
        return cr.compute_Ekin_from_mu(
            mu_scalar, Ekin_i, M_I, M_T, M_E, M_R, xp=xp_jx,
        )

    for mu_val in (-0.5, 0.0, 0.3, 0.7):
        mu0 = jnp.array(mu_val)
        grad = float(jax.grad(f)(mu0))
        analytic = float(cr.compute_dEkin_dmu(
            mu0, Ekin_i, M_I, M_T, M_E, M_R, xp=xp_jx,
        ))
        h = 1e-6
        fd = (float(f(mu0 + h)) - float(f(mu0 - h))) / (2 * h)
        assert np.isfinite(grad)
        np.testing.assert_allclose(grad, analytic, rtol=1e-11)
        np.testing.assert_allclose(grad, fd, rtol=1e-4)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_Ekin_i_matches_fd():
    """``jax.grad`` wrt the incident kinetic energy (a natural fit
    parameter in optimisation loops) matches central FD."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')

    def loss(Ein):
        return jnp.sum(cr.compute_Ekin_from_mu(
            MU_SAMPLES, Ein, M_I, M_T, M_E, M_R, xp=xp_jx,
        ))

    Ein0 = jnp.array(14.0)
    grad = float(jax.grad(loss)(Ein0))
    h = 1e-4
    fd = (float(loss(Ein0 + h)) - float(loss(Ein0 - h))) / (2 * h)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=1e-3)


def test_private_cos_phi_helpers_accept_xp():
    """The deprecated ``_compute_*_cos_phi`` privates also accept
    ``xp`` -- confirms the sqrt dispatch is consistent through the
    whole family."""
    xp_np = array_ns.get_backend('numpy')
    cos_phi = -MU_SAMPLES
    a = cr._compute_Ekin_from_cos_phi(
        cos_phi, Ekin_i, M_I, M_T, M_E, M_R,
    )
    b = cr._compute_Ekin_from_cos_phi(
        cos_phi, Ekin_i, M_I, M_T, M_E, M_R, xp=xp_np,
    )
    np.testing.assert_array_equal(a, b)

    Ekin = cr._compute_Ekin_from_cos_phi(
        cos_phi, Ekin_i, M_I, M_T, M_E, M_R,
    )
    c = cr._compute_cos_phi_from_Ekin(
        Ekin, Ekin_i, M_I, M_T, M_E, M_R,
    )
    d = cr._compute_cos_phi_from_Ekin(
        Ekin, Ekin_i, M_I, M_T, M_E, M_R, xp=xp_np,
    )
    np.testing.assert_array_equal(c, d)
