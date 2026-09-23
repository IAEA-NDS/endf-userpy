"""Backend-agnostic port of the discrete-line broadening dispatchers
in ``ddx_broadening`` (issue #169 tier-2 final piece).

The pointwise-kernel dispatchers evaluate the kernel closure at
kinematic-delta / discrete-line positions (no adaptive_convolve),
distinct from the FFT-based convolve dispatchers ported in PR #182.

Ported here:

- ``compute_ddx_discrete_broadened`` (2-body discrete-level DDX)
- ``compute_ddx_law1_discrete_broadened`` (MF6/LAW=1 discrete lines DDX)
- ``compute_ddx_mf12_discrete_broadened`` (MF12 photon lines DDX)
- ``compute_ddx_mf13_discrete_broadened`` (MF13 photon lines DDX)
- ``compute_ddx_mf15_continuum_broadened`` (MF15 gamma continuum DDX)
- ``compute_dxs_dE_mf12_discrete_broadened``,
  ``compute_dxs_dE_mf13_discrete_broadened``,
  ``compute_dxs_dE_law1_discrete_broadened``  (1D counterparts)

Composition-layer dispatch: `_get_particle_production_dxs_dE_impl`
and `_get_particle_production_ddxs_impl` in ``quantities.py`` forward
``xp`` to each of the above so ``get_particle_production_dxs_dE(...,
broadening=, xp=jax)`` and ``get_particle_production_ddxs(...)``
trace end-to-end regardless of which specific MT layout the file
carries.

Pins:

- xp=None default and xp=numpy bit-identical for the two main
  gamma paths (Al-27 (n, g), Al-27 (n, inl) discrete inelastic).
- xp=jax reproduces numpy to floating-point round-off on the
  Al-27 (n, g) gamma broadened dxs/dE that visits MF12 discrete +
  MF15 continuum + (via ``distribution1d``) MF14 angular.
- jax.grad of a scalar summary of the top-level API through the
  discrete-line broadening path matches central FD to rtol=1e-2.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
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


SIGMA = 3e4


def _gaussian_np(dx):
    return np.exp(-0.5 * dx ** 2 / SIGMA ** 2) / (SIGMA * np.sqrt(2 * np.pi))


def test_gamma_dxs_dE_broadened_default_matches_xp_numpy(al27_endf_dict):
    """Al-27 (n, g) gamma broadened dxs/dE: default vs explicit xp=numpy."""
    ein = np.array([1e6, 5e6])
    eout = np.linspace(1e5, 8e6, 15)
    xp_np = array_ns.get_backend('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,g)', 'g', ein, eout, broadening=SIGMA,
        ))
        b = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,g)', 'g', ein, eout,
            broadening=SIGMA, xp=xp_np,
        ))
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_gamma_dxs_dE_broadened_numpy_jax_parity(al27_endf_dict):
    """Al-27 (n, g) broadened path: JAX vs numpy parity. Visits
    MF12 discrete lines + MF15 continuum dispatchers."""
    ein = np.array([1e6, 5e6])
    eout = np.linspace(1e5, 8e6, 15)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,g)', 'g', ein, eout,
            broadening=SIGMA, xp=xp_np,
        ))
        b = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,g)', 'g', ein, eout,
            broadening=SIGMA, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-14)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_gamma_ddxs_broadened_numpy_jax_parity(al27_endf_dict):
    """Al-27 (n, g) broadened DDX: JAX vs numpy parity across MF12
    discrete-line, MF15 continuum dispatchers, and MF14 per-line
    angular composition."""
    ein = np.array([1e6, 5e6])
    eout = np.linspace(1e5, 8e6, 10)
    mus = np.linspace(-0.9, 0.9, 4)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(get_particle_production_ddxs(
            al27_endf_dict, '(n,g)', 'g', ein, eout, mus,
            broadening=SIGMA, xp=xp_np,
        ))
        b = np.asarray(get_particle_production_ddxs(
            al27_endf_dict, '(n,g)', 'g', ein, eout, mus,
            broadening=SIGMA, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-14)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_kernel_sigma_through_discrete_line_path(al27_endf_dict):
    """jax.grad of a scalar summary of the top-level
    ``get_particle_production_dxs_dE`` for a broadened (n, g) query
    reaches the Gaussian-kernel closure's sigma and matches central
    FD. Proves the tracer survives every discrete-line broadening
    dispatcher on the gamma path (MF12 lines, MF15 continuum)."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    ein = np.array([5e6])
    eout = np.linspace(1e5, 5e6, 10)

    def loss(sigma):
        def kt(dx):
            return jnp.exp(-0.5 * dx ** 2 / sigma ** 2) / (
                sigma * jnp.sqrt(2 * jnp.pi)
            )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return jnp.sum(get_particle_production_dxs_dE(
                al27_endf_dict, '(n,g)', 'g', ein, eout,
                broadening=(kt, 3e4), xp=xp_jx,
            ))

    sv = jnp.array(3e4)
    val = float(loss(sv))
    grad = float(jax.grad(loss)(sv))
    assert np.isfinite(grad)
    assert val > 0.0
    eps = 3e2
    fd = (float(loss(sv + eps)) - float(loss(sv - eps))) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-2, atol=1e-15)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_kernel_sigma_ddxs_law1_disc_path(al27_endf_dict):
    """Same jax.grad test on the DDX path, which exercises
    ``compute_ddx_law1_discrete_broadened`` (Al-27 (n, 2n) MF6 LAW=1
    ND>0 discrete-line content). Independent grad check from the
    dxs_dE test above."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1.4e7])
    eout = np.linspace(1e5, 5e6, 10)
    mus = np.linspace(-0.9, 0.9, 4)

    def loss(sigma):
        def kt(dx):
            return jnp.exp(-0.5 * dx ** 2 / sigma ** 2) / (
                sigma * jnp.sqrt(2 * jnp.pi)
            )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return jnp.sum(get_particle_production_ddxs(
                al27_endf_dict, '(n,2n)', 'n', ein, eout, mus,
                broadening=(kt, 5e4), xp=xp_jx,
            ))

    sv = jnp.array(5e4)
    val = float(loss(sv))
    grad = float(jax.grad(loss)(sv))
    assert np.isfinite(grad)
    assert val > 0.0
    eps = 5e2
    fd = (float(loss(sv + eps)) - float(loss(sv - eps))) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-2, atol=1e-15)
