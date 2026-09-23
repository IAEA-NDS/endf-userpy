"""Composition-layer distribution1d backend-agnostic port
(issue #169). Pins:

- ``distribution1d.compute_angdist_values`` and
  ``compute_energydist_values`` accept ``xp=`` and route it through
  the MF4 / MF5 / MF6 dispatch. The MF12 / MF14 / MF15 fallback
  branches materialise numpy at the boundary (their own xp port is
  tier-2 in the umbrella issue).
- Default numpy path unchanged; explicit ``xp=numpy`` bit-identical.
- ``xp=jax`` reproduces numpy to machine precision for the dominant
  MF6 LAW=1 (n, n') continuum path.
- ``jax.grad`` through ``compute_energydist_values`` reaches a
  dict-stored LAW=1 ``b`` coefficient and matches central
  finite-diff (proves the tracer flows through
  distribution1d -> integrate_mf6_dist2d_over_mu ->
  get_energydist_from_subsec_law1 -> integrator kernel).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.quantities_mt_zap import distribution1d as d1d
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


def test_default_matches_xp_numpy_adapter_energydist(al27_endf_dict):
    """xp=None default bit-identical to xp=numpy for energy dist
    on Al-27 MT=91 (n, n') LAW=1 continuum."""
    mt, zap = 91, 1
    ein = np.array([1.05e7])
    eout = np.linspace(1e5, 4e6, 10)
    default = np.asarray(d1d.compute_energydist_values(
        al27_endf_dict, mt, zap, ein, eout,
    ))
    xp_np = array_ns.get_backend('numpy')
    with_xp = np.asarray(d1d.compute_energydist_values(
        al27_endf_dict, mt, zap, ein, eout, xp=xp_np,
    ))
    np.testing.assert_array_equal(default, with_xp)


def test_default_matches_xp_numpy_adapter_angdist(al27_endf_dict):
    """Same for angular distribution."""
    mt, zap = 91, 1
    ein = np.array([1.05e7])
    mu = np.linspace(-0.5, 0.5, 5)
    default = np.asarray(d1d.compute_angdist_values(
        al27_endf_dict, mt, zap, ein, mu,
    ))
    xp_np = array_ns.get_backend('numpy')
    with_xp = np.asarray(d1d.compute_angdist_values(
        al27_endf_dict, mt, zap, ein, mu, xp=xp_np,
    ))
    np.testing.assert_array_equal(default, with_xp)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_numpy_jax_parity_energydist_al27_mf6_law1(al27_endf_dict):
    """JAX adapter reproduces the numpy result to machine precision
    on the LAW=1 continuum -> mu-integrated dominant path."""
    mt, zap = 91, 1
    ein = np.array([1.05e7])
    eout = np.linspace(1e5, 4e6, 10)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    f_np = np.asarray(d1d.compute_energydist_values(
        al27_endf_dict, mt, zap, ein, eout, xp=xp_np,
    ))
    f_jx = np.asarray(d1d.compute_energydist_values(
        al27_endf_dict, mt, zap, ein, eout, xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_through_composition_layer_to_law1_b_coeff(al27_endf_dict):
    """Primary issue-#169 demonstration for distribution1d:
    ``jax.grad`` of ``compute_energydist_values`` reaches a
    dict-stored MF6 LAW=1 ``b`` coefficient via the whole
    composition chain (distribution1d -> integrate_mf6_dist2d_over_mu
    -> get_energydist_from_subsec_law1 -> integrator kernel) and
    matches central finite-diff."""
    import jax
    import jax.numpy as jnp
    mt, zap = 91, 1
    ein = jnp.array([1.05e7])
    eout = jnp.linspace(1e5, 4e6, 10)
    xp_jx = array_ns.get_backend('jax')
    subsec = al27_endf_dict[6][mt]['subsection'][1]
    panel_key, ep_row, coef = 8, 5, 1
    original = float(subsec['b'][panel_key][ep_row][coef])

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[6][mt]['subsection'][1]['b'][panel_key][ep_row][coef] = theta
        return jnp.sum(d1d.compute_energydist_values(
            d_t, mt, zap, ein, eout, xp=xp_jx,
        ))

    val = float(loss(jnp.array(original)))
    grad = float(jax.grad(loss)(jnp.array(original)))
    assert np.isfinite(grad)
    assert val > 0.0
    eps = 1e-3 * abs(original) if original != 0.0 else 1e-6
    fd = (float(loss(jnp.array(original + eps))) - float(loss(jnp.array(original - eps)))) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-20)
