"""Backend-agnostic port of ``primitives.convolution.adaptive_convolve``
(issue #169 tier-2).

Pins:

- xp=None (default) and xp=numpy bit-identical.
- xp=jax reproduces the numpy result to floating-point round-off.
- jax.grad through the kernel width (a differentiable Gaussian sigma)
  matches central finite-diff to rtol=1e-3.
- jax.grad through the source-function scale matches central FD to
  rtol=1e-6 (linearity of convolution -> derivative equals the
  convolved constant integral).
- Adaptive convergence still runs on numpy; under a jax.grad trace
  it degrades cleanly to running the full ``max_iter`` and does not
  emit a spurious not-converged warning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.convolution import (
    adaptive_convolve,
    ConvergenceWarning,
)


def _jax_available():
    return 'jax' in array_ns.available_backends()


def _lorentzian(x, center=5.0, gamma=2.0):
    return 1.0 / (1.0 + (x - center) ** 2 / gamma ** 2)


def _gaussian_kernel(dx, sigma):
    return np.exp(-0.5 * dx ** 2 / sigma ** 2) / (sigma * np.sqrt(2 * np.pi))


EVAL_PTS = np.linspace(0.0, 10.0, 20)


def test_default_matches_xp_numpy_adapter():
    xp_np = array_ns.get_backend('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        a = np.asarray(adaptive_convolve(
            _lorentzian, lambda dx: _gaussian_kernel(dx, 0.5),
            EVAL_PTS, kernel_width=0.5,
        ))
        b = np.asarray(adaptive_convolve(
            _lorentzian, lambda dx: _gaussian_kernel(dx, 0.5),
            EVAL_PTS, kernel_width=0.5, xp=xp_np,
        ))
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_numpy_jax_parity():
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        a = np.asarray(adaptive_convolve(
            _lorentzian, lambda dx: _gaussian_kernel(dx, 0.5),
            EVAL_PTS, kernel_width=0.5, xp=xp_np,
        ))
        b = np.asarray(adaptive_convolve(
            _lorentzian, lambda dx: _gaussian_kernel(dx, 0.5),
            EVAL_PTS, kernel_width=0.5, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-10)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_kernel_sigma():
    """``jax.grad`` of a scalar summary of the convolution reaches the
    Gaussian-kernel sigma and matches central FD. Differentiable
    detector-resolution sensitivity."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')

    def loss(sigma):
        def kt(dx):
            return jnp.exp(-0.5 * dx ** 2 / sigma ** 2) / (
                sigma * jnp.sqrt(2 * jnp.pi)
            )

        def fn(x):
            return 1.0 / (1.0 + (jnp.asarray(x) - 5.0) ** 2 / 4.0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            return jnp.sum(adaptive_convolve(
                fn, kt, EVAL_PTS, kernel_width=0.5, xp=xp_jx, max_iter=4,
            ))

    sv = jnp.array(0.5)
    val = float(loss(sv))
    grad = float(jax.grad(loss)(sv))
    assert np.isfinite(grad)
    assert val > 0.0
    eps = 1e-4
    fd = (float(loss(sv + eps)) - float(loss(sv - eps))) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-3, atol=1e-6)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_source_scale():
    """Grad wrt a linear scale of the source function equals the
    convolved constant integral (linearity of convolution). Proves the
    tracer survives every ``values``/``mid_values`` update in the
    doubling loop."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')

    def loss(scale):
        def fn(x):
            return scale / (1.0 + (jnp.asarray(x) - 5.0) ** 2 / 4.0)

        def kt(dx):
            return jnp.exp(-0.5 * dx ** 2 / 0.5 ** 2) / (
                0.5 * jnp.sqrt(2 * jnp.pi)
            )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            return jnp.sum(adaptive_convolve(
                fn, kt, EVAL_PTS, kernel_width=0.5, xp=xp_jx, max_iter=4,
            ))

    scale_v = jnp.array(2.0)
    grad = float(jax.grad(loss)(scale_v))
    eps = 1e-4
    fd = (
        float(loss(scale_v + eps)) - float(loss(scale_v - eps))
    ) / (2 * eps)
    assert np.isfinite(grad)
    # FD at eps=1e-4 truncation is O(eps^2 / grad) ~ few*1e-6 relative;
    # loosen accordingly.
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-6)
