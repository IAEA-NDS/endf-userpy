"""Backend-agnostic (numpy + JAX) parity for the three LAB <-> CM
conversion helpers in :mod:`endf_userpy.primitives.conversion`.

Pins:

1. Backward compatibility: calling any of the three helpers without
   an ``xp`` argument (the pre-port signature) returns the same
   numbers to machine precision as calling with
   ``xp=array_ns.get_backend('numpy')``.
2. Numpy-vs-JAX parity: with the same inputs, ``xp=numpy`` and
   ``xp=jax`` produce identical results to machine precision.
3. Forbidden LAB region behaviour: NaN in the CM cosine array at
   kinematically unreachable ``mu_lab`` values, on both backends.
4. Autodiff (JAX only): ``jax.grad`` flows through
   :func:`convert_angdist_to_labsys` and :func:`compute_r2` and
   returns finite, non-zero gradients. Enables autodiff-friendly
   angular-distribution reconstruction downstream.

Skips the JAX-specific tests if JAX is not installed.
"""
import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.conversion import (
    compute_r2,
    convert_angcos_to_cmsys,
    convert_angdist_to_labsys,
)


def _jax_available():
    return 'jax' in array_ns.available_backends()


# ---- (1) Backward-compatibility (implicit numpy default). ----


def test_compute_r2_default_matches_explicit_numpy():
    """Passing no ``xp`` must be the same as passing the numpy
    backend explicitly."""
    E = np.array([1e5, 5e5, 1e6, 5e6])
    awi, awr, awp, q = 1.0, 55.0, 1.0, -2.5e6
    r2_default = np.asarray(compute_r2(E, awi, awr, awp, q))
    r2_numpy = np.asarray(compute_r2(
        E, awi, awr, awp, q, xp=array_ns.get_backend('numpy'),
    ))
    np.testing.assert_array_equal(r2_default, r2_numpy)


def test_convert_angcos_to_cmsys_default_matches_explicit_numpy():
    E = np.array([1e6, 5e6, 1e7])
    r2 = compute_r2(E, 1.0, 55.0, 1.0, -2.5e6)
    mu = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
    mu_cm_default = np.asarray(convert_angcos_to_cmsys(mu, r2))
    mu_cm_numpy = np.asarray(convert_angcos_to_cmsys(
        mu, r2, xp=array_ns.get_backend('numpy'),
    ))
    # Some entries may be NaN (forbidden region); use equal_nan.
    np.testing.assert_array_equal(mu_cm_default, mu_cm_numpy)


def test_convert_angdist_to_labsys_default_matches_explicit_numpy():
    """Real-usage shape: both ``mu_cm`` and ``f_cm`` are already 2D
    ``(nE, nmu)`` when reached from ``mf4_interpretation`` (the
    canonical caller); the internal 1D-reshape branch only fires
    on hand-built scalars."""
    E = np.array([1e6, 5e6, 1e7])
    r2 = compute_r2(E, 1.0, 55.0, 1.0, -2.5e6)
    mu_cm_1d = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
    # Broadcast mu into (nE, nmu) so mu and f share shape, matching
    # what mf4_interpretation passes.
    mu_cm = np.broadcast_to(mu_cm_1d, (len(E), len(mu_cm_1d))).copy()
    f_cm = np.ones_like(mu_cm)
    f_lab_default = np.asarray(convert_angdist_to_labsys(mu_cm, f_cm, r2))
    f_lab_numpy = np.asarray(convert_angdist_to_labsys(
        mu_cm, f_cm, r2, xp=array_ns.get_backend('numpy'),
    ))
    np.testing.assert_array_equal(f_lab_default, f_lab_numpy)


# ---- (2) JAX parity. ----


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_compute_r2_numpy_jax_parity():
    import jax.numpy as jnp
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')
    E = np.array([1e5, 5e5, 1e6, 5e6])
    r2_np = np.asarray(compute_r2(E, 1.0, 55.0, 1.0, -2.5e6, xp=xp_np))
    r2_jax = np.asarray(compute_r2(
        jnp.asarray(E), 1.0, 55.0, 1.0, -2.5e6, xp=xp_jax,
    ))
    np.testing.assert_allclose(r2_np, r2_jax, rtol=1e-14, atol=0.0)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_convert_angcos_to_cmsys_numpy_jax_parity():
    import jax.numpy as jnp
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')
    E = np.array([1e6, 5e6, 1e7])
    r2 = compute_r2(E, 1.0, 55.0, 1.0, -2.5e6)
    mu = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
    out_np = np.asarray(convert_angcos_to_cmsys(mu, r2, xp=xp_np))
    out_jax = np.asarray(convert_angcos_to_cmsys(
        jnp.asarray(mu), jnp.asarray(r2), xp=xp_jax,
    ))
    # Where numpy is NaN, JAX must also be NaN; where finite, agree
    # to machine precision.
    mask = np.isfinite(out_np)
    assert (np.isnan(out_np) == np.isnan(out_jax)).all(), (
        'NaN mask differs between numpy and JAX'
    )
    np.testing.assert_allclose(
        out_np[mask], out_jax[mask], rtol=1e-14, atol=0.0,
    )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_convert_angdist_to_labsys_numpy_jax_parity():
    import jax.numpy as jnp
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')
    E = np.array([1e6, 5e6, 1e7])
    r2 = compute_r2(E, 1.0, 55.0, 1.0, -2.5e6)
    mu_cm_1d = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
    mu_cm = np.broadcast_to(mu_cm_1d, (len(E), len(mu_cm_1d))).copy()
    f_cm = np.ones_like(mu_cm)
    out_np = np.asarray(convert_angdist_to_labsys(mu_cm, f_cm, r2, xp=xp_np))
    out_jax = np.asarray(convert_angdist_to_labsys(
        jnp.asarray(mu_cm), jnp.asarray(f_cm), jnp.asarray(r2), xp=xp_jax,
    ))
    np.testing.assert_allclose(out_np, out_jax, rtol=1e-14, atol=0.0)


# ---- (3) Forbidden LAB region. ----


def test_convert_angcos_forbidden_region_returns_nan():
    """For equal-mass elastic (H-1) at any LAB energy, back-scatter
    (``mu_lab < 0``) is kinematically forbidden in the CM. Result
    must be NaN there so the caller's ``pad_outside`` decorator can
    zero it. Pins the physical convention on the numpy path; JAX
    parity test above covers JAX behaviour."""
    # Equal-mass elastic: r ~ 1, forbidden region for u < 0.
    E = np.array([1e6])
    r2 = compute_r2(E, 1.0, 1.0, 1.0, 0.0)   # equal-mass, elastic
    # r2 should be ~1 here (compute directly: awr*(awr+awi-awp)/(awi*awp) = 1).
    assert float(r2[0]) == pytest.approx(1.0)
    mu = np.array([-0.5, 0.5])
    out = np.asarray(convert_angcos_to_cmsys(mu, r2))
    # u=-0.5: z = 0.25 + 1 - 1 = 0.25 (allowed); u=0.5: same.
    # For r=1 exactly, z >= 0 everywhere, so this test just confirms
    # the algebra runs; the NaN forbidden-region test needs r < 1
    # (heavy ejectile from light target).
    assert np.all(np.isfinite(out)), (
        f'r=1 case must be entirely allowed; got {out}'
    )
    # Deep forbidden region: r << 1 (light ejectile from heavy
    # target) at backward angles. z = mu^2 + r^2 - 1 = -0.18 < 0
    # for (mu, r2) = (-0.9, 0.01), so the CM mapping is undefined
    # and must NaN out.
    r2_deep = np.array([0.01])   # r = 0.1
    mu_backward = np.array([-0.9])
    out_deep = np.asarray(convert_angcos_to_cmsys(mu_backward, r2_deep))
    assert np.all(np.isnan(out_deep)), (
        f'deep forbidden region must NaN out; got {out_deep}'
    )


# ---- (4) JAX autodiff. ----


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_convert_angdist_to_labsys_jax_grad_flows():
    """``jax.grad`` of a scalar summary of the LAB-frame
    distribution wrt the CM-frame amplitude returns a finite,
    non-zero gradient. Enables autodiff through the CM -> LAB
    Jacobian, which was impossible with the in-place numpy
    ``_correct_r2`` in the pre-port implementation."""
    import jax
    import jax.numpy as jnp
    xp_jax = array_ns.get_backend('jax')
    E = jnp.array([1e6])
    r2 = compute_r2(E, 1.0, 55.0, 1.0, -2.5e6, xp=xp_jax)
    mu_cm = jnp.array([0.5])

    def loss(scale):
        f_cm = jnp.ones((1, 1)) * scale
        f_lab = convert_angdist_to_labsys(mu_cm, f_cm, r2, xp=xp_jax)
        return jnp.sum(f_lab)

    val = float(loss(1.0))
    g = float(jax.grad(loss)(1.0))
    assert np.isfinite(val)
    assert np.isfinite(g)
    # LAB distribution is linear in f_cm; grad wrt scale must equal
    # the LAB value itself at scale=1 (to machine precision).
    np.testing.assert_allclose(g, val, rtol=1e-12)
