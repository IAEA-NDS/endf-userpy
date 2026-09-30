"""Composability of ``jax.jit`` with ``jax.grad`` for MF6 LAW=1.

Extends ``test_mf6_law1_top_level_mesh_autodiff`` (which pinned
eager ``jax.grad``) to the two composed patterns a user is likely
to hit in a fitting loop:

* ``jax.grad(jax.jit(loss))`` — jit the forward pass, take gradient
  around it. Common when the forward evaluation is expensive
  enough that jit's compile cost amortises over many gradient
  calls.
* ``jax.jit(jax.grad(loss))`` — precompile the gradient function
  itself. Also common when the gradient is called many times with
  the same input shapes.

Both patterns require that the MF6 LAW=1 preproc and kernel do
not turn per-panel integer count arrays (``nep_arr`` / ``nd_arr``
/ ``na_arr``) into JAX tracers when the whole pipeline runs
inside a jit trace, because those tracers would trip the kernel's
``np.asarray(...) + int(...)`` static reads.
"""
from __future__ import annotations

import copy
import os
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.quantities import get_particle_production_ddxs
from endf_userpy.run_options import RunOptions


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = 'tests/data_law1_adhoc/endfb81_n_Al-27.endf'
    if not os.path.exists(path):
        pytest.skip('Al-27 corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def _loss_and_fd(al27_endf_dict):
    """Shared loss closure + FD baseline for the three grad tests."""
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    sub = al27_endf_dict[6][91]['subsection'][1]
    idx = 8
    original = float(sub['E'][idx])
    ein_val = 0.5 * (original + float(sub['E'][idx + 1]))
    ep_vals = list(sub['Ep'][idx].values())
    eout = np.linspace(max(1e3, ep_vals[0] * 1.1), ep_vals[-1] * 0.9, 3)
    mu = np.array([-0.5, 0.0, 0.5])

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[6][91]['subsection'][1]['E'][idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_particle_production_ddxs(d_t, '(n,n_c)', 'n', np.array([ein_val]), eout, mu, options=RunOptions(backend=xp_jax))
        return jnp.sum(r)

    theta = jnp.array(original)
    eps = original * 1e-4
    fd = (float(loss(jnp.array(original + eps)))
          - float(loss(jnp.array(original - eps)))) / (2.0 * eps)
    return loss, theta, fd


def test_eager_grad_matches_fd(_loss_and_fd):
    """Baseline: plain ``jax.grad`` (no jit) matches FD."""
    import jax

    loss, theta, fd = _loss_and_fd
    grad = float(jax.grad(loss)(theta))
    assert np.isfinite(grad)
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)


def test_grad_of_jit_loss_matches_fd(_loss_and_fd):
    """``jax.grad(jax.jit(loss))``: jit the forward pass, take
    gradient around the jitted function. Requires the preproc /
    kernel to not turn per-panel integer counts into tracers under
    the jit trace.
    """
    import jax

    loss, theta, fd = _loss_and_fd
    grad = float(jax.grad(jax.jit(loss))(theta))
    assert np.isfinite(grad)
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)


def test_jit_of_grad_loss_matches_fd(_loss_and_fd):
    """``jax.jit(jax.grad(loss))``: precompile the gradient function
    itself. Same tracer-integer constraint as the previous test.
    """
    import jax

    loss, theta, fd = _loss_and_fd
    grad_fn = jax.jit(jax.grad(loss))
    # Warmup (compile).
    _ = grad_fn(theta)
    grad = float(grad_fn(theta))
    assert np.isfinite(grad)
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)
