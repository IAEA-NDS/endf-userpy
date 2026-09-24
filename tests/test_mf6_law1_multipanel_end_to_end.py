"""End-to-end multi-panel jax.grad wrt incident energy E through
the top-level ``get_particle_production_dxs_dE`` (issue #190 --
depends on ``endf_interp1d`` query-side tracer support from PR #192
+ ``mf3.compute_cross_section`` xp threading + the multipanel-
traced kernel dispatch in ``mf6_interpretation_integrals.get_
energydist_from_subsec_law1``).

Pins:

- ``get_particle_production_dxs_dE(..., xp=jax)`` runs to completion
  with a jax tracer E on Al-27 (n, 2n) MT16.
- ``jax.grad`` wrt E matches central FD at three mid-panel E values.
- Numpy path unchanged (default and explicit xp=numpy identical).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.quantities import get_particle_production_dxs_dE

from _corpus import resolve_al27


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip('Al-27 corpus not present')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def test_numpy_path_unchanged(al27_endf_dict):
    xp_np = array_ns.get_backend('numpy')
    ein = np.array([1.42e7, 1.55e7, 1.7e7])
    eout = np.linspace(1e5, 5e6, 8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,2n)', 'n', ein, eout,
        ))
        b = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,2n)', 'n', ein, eout, xp=xp_np,
        ))
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_E_multipanel_through_top_level(al27_endf_dict):
    """jax.grad through the top-level API matches central FD at
    three mid-panel Es. Confirms the query-side tracer chain --
    endf_interp1d + mf3.compute_cross_section + mf6_law1 multi-
    panel-traced kernel + compute_dexs -- all propagate tracers
    on the query axis."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    eout = jnp.linspace(1e5, 5e6, 8)

    def loss(E):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return jnp.sum(get_particle_production_dxs_dE(
                al27_endf_dict, '(n,2n)', 'n', E[None], eout, xp=xp_jx,
            ))

    for E_val in (1.42e7, 1.55e7, 1.7e7):
        E0 = jnp.array(E_val)
        val = float(loss(E0))
        grad = float(jax.grad(loss)(E0))
        fd_h = E_val * 1e-5
        fd = (float(loss(E0 + fd_h)) - float(loss(E0 - fd_h))) / (2 * fd_h)
        assert np.isfinite(grad)
        assert val > 0.0
        if abs(fd) < 1e-30:
            continue
        np.testing.assert_allclose(grad, fd, rtol=5e-2, atol=1e-30)
