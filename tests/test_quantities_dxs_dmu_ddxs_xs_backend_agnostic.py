"""Backend-agnostic port of the remaining user-facing ``get_*``
entry points (issue #169 tier-2):

- ``get_particle_production_xs``
- ``get_particle_production_dxs_dmu``
- ``get_particle_production_ddxs``

Pins:

- ``xp=None`` (default) and ``xp=numpy`` bit-identical.
- ``xp=jax`` reproduces the numpy result to machine precision on
  the raw MF3 / MF6 LAW=1 unbroadened paths.
- ``jax.grad`` of a scalar summary of ``get_particle_production_ddxs``
  reaches a dict-stored MF6 LAW=1 ``b`` coefficient and matches
  central finite-diff. Proves the tracer survives the full DDX
  composition chain from the top-level API through
  ``compute_cumulative_quantity`` -> ``compute_ddxs`` ->
  ``compute_dist2d_values`` -> LAW=1 kernel.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_xs,
    get_particle_production_dxs_dmu,
    get_particle_production_ddxs,
)
from endf_userpy.primitives import array_ns

from _corpus import resolve_al27


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip('Al-27 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


# ---------------------------------------------------------------
# get_particle_production_xs
# ---------------------------------------------------------------

def test_xs_default_matches_xp_numpy(al27_endf_dict):
    ein = np.array([1e6, 5e6, 1.4e7])
    xp_np = array_ns.get_backend('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        default = np.asarray(get_particle_production_xs(
            al27_endf_dict, '(n,2n)', 'n', ein,
        ))
        with_xp = np.asarray(get_particle_production_xs(
            al27_endf_dict, '(n,2n)', 'n', ein, xp=xp_np,
        ))
    np.testing.assert_array_equal(default, with_xp)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_xs_numpy_jax_parity(al27_endf_dict):
    ein = np.array([1e6, 5e6, 1.4e7])
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        f_np = np.asarray(get_particle_production_xs(
            al27_endf_dict, '(n,2n)', 'n', ein, xp=xp_np,
        ))
        f_jx = np.asarray(get_particle_production_xs(
            al27_endf_dict, '(n,2n)', 'n', ein, xp=xp_jx,
        ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-11, atol=1e-30)


# ---------------------------------------------------------------
# get_particle_production_dxs_dmu
# ---------------------------------------------------------------

def test_dxs_dmu_default_matches_xp_numpy(al27_endf_dict):
    ein = np.array([1e6, 5e6, 1.4e7])
    mus = np.linspace(-0.9, 0.9, 7)
    xp_np = array_ns.get_backend('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        default = np.asarray(get_particle_production_dxs_dmu(
            al27_endf_dict, '(n,2n)', 'n', ein, mus,
        ))
        with_xp = np.asarray(get_particle_production_dxs_dmu(
            al27_endf_dict, '(n,2n)', 'n', ein, mus, xp=xp_np,
        ))
    np.testing.assert_array_equal(default, with_xp)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_dxs_dmu_numpy_jax_parity(al27_endf_dict):
    ein = np.array([1e6, 5e6, 1.4e7])
    mus = np.linspace(-0.9, 0.9, 7)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        f_np = np.asarray(get_particle_production_dxs_dmu(
            al27_endf_dict, '(n,2n)', 'n', ein, mus, xp=xp_np,
        ))
        f_jx = np.asarray(get_particle_production_dxs_dmu(
            al27_endf_dict, '(n,2n)', 'n', ein, mus, xp=xp_jx,
        ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-11, atol=1e-30)


# ---------------------------------------------------------------
# get_particle_production_ddxs (unbroadened path)
# ---------------------------------------------------------------

def test_ddxs_default_matches_xp_numpy(al27_endf_dict):
    ein = np.array([1.4e7])
    eout = np.linspace(1e5, 5e6, 8)
    mus = np.linspace(-0.9, 0.9, 5)
    xp_np = array_ns.get_backend('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        default = np.asarray(get_particle_production_ddxs(
            al27_endf_dict, '(n,2n)', 'n', ein, eout, mus,
        ))
        with_xp = np.asarray(get_particle_production_ddxs(
            al27_endf_dict, '(n,2n)', 'n', ein, eout, mus, xp=xp_np,
        ))
    np.testing.assert_array_equal(default, with_xp)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_ddxs_numpy_jax_parity(al27_endf_dict):
    ein = np.array([1.4e7])
    eout = np.linspace(1e5, 5e6, 8)
    mus = np.linspace(-0.9, 0.9, 5)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        f_np = np.asarray(get_particle_production_ddxs(
            al27_endf_dict, '(n,2n)', 'n', ein, eout, mus, xp=xp_np,
        ))
        f_jx = np.asarray(get_particle_production_ddxs(
            al27_endf_dict, '(n,2n)', 'n', ein, eout, mus, xp=xp_jx,
        ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_ddxs_jax_grad_to_law1_b_coeff_end_to_end(al27_endf_dict):
    """Tier-2 flagship demo: ``jax.grad`` of a scalar summary of
    ``get_particle_production_ddxs`` reaches a dict-stored MF6 LAW=1
    L=0 ``b`` coefficient and matches central finite-diff. Proves
    the tracer survives the whole DDX composition chain from the
    top-level API through the compute_ddxs / compute_dist2d_values /
    LAW=1 kernel stack."""
    import jax
    import jax.numpy as jnp

    # Al-27 MF6 MT=16 subsec 1 has ei_mesh[1]=1.4e7, ei_mesh[2]=1.5e7.
    # E_query=1.45e7 lands strictly inside panel 2 (1-indexed b key).
    E_query = 1.45e7
    panel_key = 2

    subsec = al27_endf_dict[6][16]['subsection'][1]
    ep_rows = list(subsec['b'][panel_key].keys())
    ep_row = ep_rows[5]
    coef = 0  # L=0 isotropic component: largest sensitivity
    original = float(subsec['b'][panel_key][ep_row][coef])

    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([E_query])
    eout = jnp.linspace(1e5, 5e6, 8)
    mus = jnp.linspace(-0.9, 0.9, 5)

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[6][16]['subsection'][1]['b'][panel_key][ep_row][coef] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return jnp.sum(get_particle_production_ddxs(
                d_t, '(n,2n)', 'n', ein, eout, mus, xp=xp_jx,
            ))

    val = float(loss(jnp.array(original)))
    grad = float(jax.grad(loss)(jnp.array(original)))
    assert np.isfinite(grad)
    assert val > 0.0
    assert abs(grad) > 0.0
    eps = 1e-3 * abs(original) if original != 0.0 else 1e-6
    fd = (
        float(loss(jnp.array(original + eps)))
        - float(loss(jnp.array(original - eps)))
    ) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-20)
