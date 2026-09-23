"""Composition-layer distribution2d backend-agnostic port
(issue #169). Pins:

- ``distribution2d.compute_dist2d_values`` accepts ``xp=`` and
  passes it through the MF6-dispatcher chain (via
  ``mf6_interpretation.compute_dist2d_values`` -> ``mf6_interpretation_subsecs.compute_dist2d_from_subsec``)
  down to the LAW=1 / 2 / 6 / 7 kernels.
- Default numpy path unchanged; explicit ``xp=numpy`` bit-identical.
- ``xp=jax`` reproduces the numpy result to machine precision.
- ``jax.grad`` through the composition-layer entry reaches a
  dict-stored LAW=1 ``b`` coefficient and matches finite-diff.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.quantities_mt_zap import distribution2d as d2d
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


def test_default_matches_xp_numpy_adapter(al27_endf_dict):
    """distribution2d default (xp=None) is bit-identical to
    explicit xp=numpy on an Al-27 MF6 LAW=1 continuum grid."""
    mt, zap = 91, 1  # (n, n') continuum, neutron ejectile
    ein = np.array([1.05e7])
    eout = np.linspace(1e5, 4e6, 10)
    mu = np.linspace(-0.5, 0.5, 5)
    default = np.asarray(d2d.compute_dist2d_values(
        al27_endf_dict, mt, zap, ein, eout, mu,
    ))
    xp_np = array_ns.get_backend('numpy')
    with_xp = np.asarray(d2d.compute_dist2d_values(
        al27_endf_dict, mt, zap, ein, eout, mu, xp=xp_np,
    ))
    np.testing.assert_array_equal(default, with_xp)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_numpy_jax_parity_on_al27_mf6_law1(al27_endf_dict):
    """JAX adapter reproduces the numpy result to machine precision
    on the MF6 LAW=1 continuum path (the dominant runtime path)."""
    mt, zap = 91, 1
    ein = np.array([1.05e7])
    eout = np.linspace(1e5, 4e6, 10)
    mu = np.linspace(-0.5, 0.5, 5)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    f_np = np.asarray(d2d.compute_dist2d_values(
        al27_endf_dict, mt, zap, ein, eout, mu, xp=xp_np,
    ))
    f_jx = np.asarray(d2d.compute_dist2d_values(
        al27_endf_dict, mt, zap, ein, eout, mu, xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_through_composition_layer_to_law1_b_coeff(al27_endf_dict):
    """Primary issue-#169 demonstration for the composition layer:
    ``jax.grad`` of a scalar summary of
    ``distribution2d.compute_dist2d_values`` reaches a dict-stored
    MF6 LAW=1 angular-parameter leaf and matches central
    finite-diff. Proves the tracer survives the whole dispatcher
    chain (distribution2d -> mf6_interpretation -> mf6_interpretation_subsecs
    -> get_dist2d_from_subsec_law1 -> reconstruction kernel).
    """
    import jax
    import jax.numpy as jnp
    mt, zap = 91, 1
    ein = jnp.array([1.05e7])
    eout = jnp.linspace(1e5, 4e6, 10)
    mu = jnp.linspace(-0.5, 0.5, 5)
    xp_jx = array_ns.get_backend('jax')
    subsec = al27_endf_dict[6][mt]['subsection'][1]
    panel_key, ep_row, coef = 8, 5, 1
    original = float(subsec['b'][panel_key][ep_row][coef])

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[6][mt]['subsection'][1]['b'][panel_key][ep_row][coef] = theta
        return jnp.sum(d2d.compute_dist2d_values(
            d_t, mt, zap, ein, eout, mu, xp=xp_jx,
        ))

    val = float(loss(jnp.array(original)))
    grad = float(jax.grad(loss)(jnp.array(original)))
    assert np.isfinite(grad)
    assert val > 0.0
    eps = 1e-3 * abs(original) if original != 0.0 else 1e-6
    fd = (float(loss(jnp.array(original + eps))) - float(loss(jnp.array(original - eps)))) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-20)
