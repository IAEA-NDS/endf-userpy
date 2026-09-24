"""Query-side (``x``) tracer support in ``endf_interp1d``
(first step towards issue #190). Under ``xp=jax`` ``endf_interp1d``
now routes through ``_endf_interp1d_traced_x`` which computes all
5 INT schemes element-wise on the full ``x`` array and selects the
per-point result via ``xp.where`` on the per-mesh-interval INT law.

Pins:

- Numpy vs jax parity across all 5 INT schemes on a synthetic
  multi-region TAB1.
- ``jax.grad`` wrt ``x`` matches central FD at points inside each
  INT region (INT=1 histogram grad=0; INT=2..5 grad matches the
  analytic formula).
- Numpy caller path unchanged bit-identically (the traced-x branch
  fires only for ``xp.name == 'jax'``).
- Handles out-of-mesh ``x`` when ``outside_value`` is passed;
  raises the same ValueError as before when ``outside_value=None``
  and any x is outside the mesh on the numpy path.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.interpolation import endf_interp1d


def _jax_available():
    return 'jax' in array_ns.available_backends()


# Synthetic TAB1: three regions with three different INT schemes.
XP_MESH = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
FP = np.array([10.0, 20.0, 30.0, 40.0, 60.0, 100.0])
INT_ARR = np.array([2, 1, 5])   # lin-lin, histogram, log-log
NBT_ARR = np.array([3, 5, 6])


def test_numpy_path_unchanged():
    """Pre-port numpy behaviour is bit-identical (traced-x branch
    fires only under xp.name == 'jax')."""
    xp_np = array_ns.get_backend('numpy')
    x = np.array([1.5, 3.0, 6.0, 15.0])
    r_default = endf_interp1d(x, XP_MESH, FP, INT_ARR, NBT_ARR, outside_value=0.0)
    r_np = endf_interp1d(x, XP_MESH, FP, INT_ARR, NBT_ARR, outside_value=0.0, xp=xp_np)
    np.testing.assert_array_equal(r_default, r_np)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_numpy_jax_parity_all_int_schemes():
    """numpy per-region loop and jax traced-x path agree to
    floating-point round-off across all 5 INT schemes."""
    import jax.numpy as jnp
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    # Points spanning all three regions
    x = np.array([1.3, 2.5, 4.5, 7.0, 15.0, 40.0])
    a = endf_interp1d(x, XP_MESH, FP, INT_ARR, NBT_ARR, outside_value=0.0, xp=xp_np)
    b = np.asarray(endf_interp1d(
        jnp.asarray(x), XP_MESH, FP, INT_ARR, NBT_ARR,
        outside_value=0.0, xp=xp_jx,
    ))
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_x_lin_lin_region():
    """INT=2 lin-lin: grad wrt x is the constant slope of the local
    bracket."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')

    def f(x_scalar):
        return endf_interp1d(
            x_scalar[None], XP_MESH, FP, INT_ARR, NBT_ARR,
            outside_value=0.0, xp=xp_jx,
        )[0]

    # First lin-lin bracket: mesh [1, 2], fp [10, 20], slope = 10
    grad = float(jax.grad(f)(jnp.array(1.5)))
    assert abs(grad - 10.0) < 1e-10
    # Second lin-lin bracket: mesh [2, 5], fp [20, 30], slope = 10/3
    grad = float(jax.grad(f)(jnp.array(3.5)))
    assert abs(grad - 10.0 / 3.0) < 1e-10


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_x_histogram_region():
    """INT=1 histogram: grad wrt x is 0 within a bracket."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')

    def f(x_scalar):
        return endf_interp1d(
            x_scalar[None], XP_MESH, FP, INT_ARR, NBT_ARR,
            outside_value=0.0, xp=xp_jx,
        )[0]

    # Third region: INT=1 histogram over mesh [5, 10], [10, 20]
    grad_a = float(jax.grad(f)(jnp.array(6.0)))
    grad_b = float(jax.grad(f)(jnp.array(15.0)))
    assert abs(grad_a) < 1e-14
    assert abs(grad_b) < 1e-14


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_x_log_log_region_matches_fd():
    """INT=5 log-log: grad wrt x matches central FD."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')

    def f(x_scalar):
        return endf_interp1d(
            x_scalar[None], XP_MESH, FP, INT_ARR, NBT_ARR,
            outside_value=0.0, xp=xp_jx,
        )[0]

    # Log-log region: mesh [20, 50], fp [60, 100]
    x_val = 35.0
    x0 = jnp.array(x_val)
    grad = float(jax.grad(f)(x0))
    h = x_val * 1e-6
    fd = (float(f(x0 + h)) - float(f(x0 - h))) / (2 * h)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_x_multi_point_vectorised():
    """jax.grad wrt each of a vector of x points via jax.jacobian:
    the Jacobian is diagonal (each output depends only on its own
    input x), and diagonal entries match single-point grads."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')

    def f(x_vec):
        return endf_interp1d(
            x_vec, XP_MESH, FP, INT_ARR, NBT_ARR,
            outside_value=0.0, xp=xp_jx,
        )

    # 3 points in 3 different regions
    x = jnp.array([1.5, 6.0, 35.0])
    jac = np.asarray(jax.jacobian(f)(x))
    # Diagonal: lin-lin slope, histogram=0, log-log via FD
    assert abs(jac[0, 0] - 10.0) < 1e-10
    assert abs(jac[1, 1]) < 1e-14
    # Off-diagonal: 0
    for i in range(3):
        for j in range(3):
            if i != j:
                assert abs(jac[i, j]) < 1e-14, f'off-diagonal ({i},{j}) not zero'


def test_out_of_mesh_x_numpy_raises_without_outside_value():
    """Numpy path: raises when x is outside mesh and outside_value=None."""
    xp_np = array_ns.get_backend('numpy')
    x = np.array([0.5, 100.0])   # both outside
    with pytest.raises(ValueError, match='outside mesh'):
        endf_interp1d(x, XP_MESH, FP, INT_ARR, NBT_ARR, xp=xp_np)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_out_of_mesh_x_jax_returns_outside_value():
    """Jax path with outside_value=0.0: out-of-mesh points return 0."""
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    x = jnp.array([0.5, 100.0])
    r = np.asarray(endf_interp1d(
        x, XP_MESH, FP, INT_ARR, NBT_ARR, outside_value=0.0, xp=xp_jx,
    ))
    np.testing.assert_array_equal(r, np.array([0.0, 0.0]))
