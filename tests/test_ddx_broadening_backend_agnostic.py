"""Backend-agnostic port of the convolve-based ddx-broadening
dispatchers (issue #169 tier-2 follow-up to PR #181):

- ``ddx_broadening.compute_dxs_dE_broadened``
- ``ddx_broadening.compute_ddx_continuous_broadened``
- ``ddx_broadening.compute_ddx_continuous_broadened_summed``

Composition-layer callers in ``endf_userpy.quantities`` forward ``xp``
to these three, so ``get_particle_production_dxs_dE(..., broadening=,
xp=jax)`` and ``get_particle_production_ddxs(..., broadening=, xp=jax)``
now trace through the FFT convolution to file-side leaves and to the
``kernel`` closure's parameters.

Pins:

- xp=None default and xp=numpy bit-identical on Al-27 (n, 2n).
- xp=jax reproduces the numpy path to FFT round-off (rtol=1e-6).
- ``jax.grad`` of a scalar summary of ``get_particle_production_dxs_dE``
  wrt the Gaussian-kernel sigma matches central FD to rtol=1e-2.
  Proves the tracer flows through the broadening dispatcher, the
  ``adaptive_convolve`` FFT loop, and the boundary xs*yield mult.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.quantities_mt_zap import ddx_broadening as ddxb
from endf_userpy.quantities import (
    get_particle_production_dxs_dE,
    get_particle_production_ddxs,
)

from _corpus import resolve_al27


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip('Al-27 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


SIGMA = 5e4  # eV


def _gaussian(dx, sigma=SIGMA):
    return np.exp(-0.5 * dx ** 2 / sigma ** 2) / (sigma * np.sqrt(2 * np.pi))


def test_dxs_dE_broadened_default_matches_xp_numpy(al27_endf_dict):
    ein = np.array([1.4e7])
    eout = np.linspace(1e5, 5e6, 15)
    xp_np = array_ns.get_backend('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(ddxb.compute_dxs_dE_broadened(
            al27_endf_dict, 16, 1, ein, eout,
            _gaussian, SIGMA,
        ))
        b = np.asarray(ddxb.compute_dxs_dE_broadened(
            al27_endf_dict, 16, 1, ein, eout,
            _gaussian, SIGMA, xp=xp_np,
        ))
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_dxs_dE_broadened_numpy_jax_parity(al27_endf_dict):
    ein = np.array([1.4e7])
    eout = np.linspace(1e5, 5e6, 15)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(ddxb.compute_dxs_dE_broadened(
            al27_endf_dict, 16, 1, ein, eout,
            _gaussian, SIGMA, xp=xp_np,
        ))
        b = np.asarray(ddxb.compute_dxs_dE_broadened(
            al27_endf_dict, 16, 1, ein, eout,
            _gaussian, SIGMA, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-14)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_ddx_continuous_broadened_numpy_jax_parity(al27_endf_dict):
    ein = np.array([1.4e7])
    eout = np.linspace(1e5, 5e6, 12)
    mus = np.linspace(-0.9, 0.9, 4)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(ddxb.compute_ddx_continuous_broadened(
            al27_endf_dict, 16, 1, ein, eout, mus,
            _gaussian, SIGMA, xp=xp_np,
        ))
        b = np.asarray(ddxb.compute_ddx_continuous_broadened(
            al27_endf_dict, 16, 1, ein, eout, mus,
            _gaussian, SIGMA, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-14)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_particle_production_dxs_dE_broadened_parity(al27_endf_dict):
    """Top-level API broadened path: xp=numpy vs xp=jax parity for
    Al-27 (n, 2n) neutron production with a Gaussian broadening
    kernel at sigma=50 keV."""
    ein = np.array([1.4e7])
    eout = np.linspace(1e5, 5e6, 15)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,2n)', 'n', ein, eout,
            broadening=SIGMA, xp=xp_np,
        ))
        b = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,2n)', 'n', ein, eout,
            broadening=SIGMA, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-14)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_particle_production_ddxs_broadened_parity(al27_endf_dict):
    """Top-level API broadened DDX path parity."""
    ein = np.array([1.4e7])
    eout = np.linspace(1e5, 5e6, 12)
    mus = np.linspace(-0.9, 0.9, 4)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(get_particle_production_ddxs(
            al27_endf_dict, '(n,2n)', 'n', ein, eout, mus,
            broadening=SIGMA, xp=xp_np,
        ))
        b = np.asarray(get_particle_production_ddxs(
            al27_endf_dict, '(n,2n)', 'n', ein, eout, mus,
            broadening=SIGMA, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-14)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_broadening_sigma_top_level(al27_endf_dict):
    """Differentiable-broadening flagship demo: ``jax.grad`` of a
    scalar summary of the top-level ``get_particle_production_dxs_dE``
    wrt the Gaussian kernel sigma matches central FD. Proves the
    tracer survives the top-level API dispatch, the composition-
    layer sum, the broadening dispatcher, the ``adaptive_convolve``
    FFT loop, and the yield * xs multiplication."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1.4e7])
    eout = np.linspace(1e5, 5e6, 12)

    def loss(sigma):
        def kernel(dx):
            return jnp.exp(-0.5 * dx ** 2 / sigma ** 2) / (
                sigma * jnp.sqrt(2 * jnp.pi)
            )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return jnp.sum(get_particle_production_dxs_dE(
                al27_endf_dict, '(n,2n)', 'n', ein, eout,
                broadening=(kernel, 5e4),
                xp=xp_jx,
            ))

    sv = jnp.array(5e4)
    val = float(loss(sv))
    grad = float(jax.grad(loss)(sv))
    assert np.isfinite(grad)
    assert val > 0.0
    eps = 5e2
    fd = (float(loss(sv + eps)) - float(loss(sv - eps))) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-2, atol=1e-15)
