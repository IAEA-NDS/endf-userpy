"""Top-level user-facing autodiff (issue #169, tier 1 final):
``jax.grad(get_particle_production_dxs_dE, ...)`` reaches file-side
leaves end-to-end.

Pins:

- ``xp=None`` (default) and explicit ``xp=numpy`` bit-identical.
- ``xp=jax`` reproduces the numpy result to machine precision on
  the MF6 LAW=1 continuum path (dominant runtime path).
- ``jax.grad`` of a scalar summary reaches an MF6 LAW=1 ``b``
  coefficient and matches central finite-diff. Proves the tracer
  survives the whole stack:
  ``get_particle_production_dxs_dE`` -> ``_get_particle_production_dxs_dE_impl``
  -> ``compute_cumulative_quantity`` -> ``compute_dexs`` ->
  ``compute_energydist_values`` -> ``integrate_mf6_dist2d_over_mu``
  -> ``get_energydist_from_subsec_law1`` -> LAW=1 kernel.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_particle_production_dxs_dE
from endf_userpy.primitives import array_ns
from endf_userpy.primitives.helpers import find_interval

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
    """xp=None default bit-identical to explicit xp=numpy on
    ``get_particle_production_dxs_dE`` for Al-27 (n, 2n)."""
    ein = np.array([1.4e7])
    eout = np.linspace(1e5, 5e6, 15)
    xp_np = array_ns.get_backend('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        default = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,2n)', 'n', ein, eout,
        ))
        with_xp = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,2n)', 'n', ein, eout, xp=xp_np,
        ))
    np.testing.assert_array_equal(default, with_xp)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_numpy_jax_parity_al27_n2n(al27_endf_dict):
    """JAX adapter reproduces the numpy result to machine precision
    for the MF6 LAW=1 continuum (n, 2n) dominant path."""
    ein = np.array([1.4e7])
    eout = np.linspace(1e5, 5e6, 15)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        f_np = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,2n)', 'n', ein, eout, xp=xp_np,
        ))
        f_jx = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,2n)', 'n', ein, eout, xp=xp_jx,
        ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_end_to_end_to_law1_b_coeff(al27_endf_dict):
    """The tier-1 demo the whole issue-#169 sequencing was aiming at:
    ``jax.grad`` of a scalar summary of the top-level user-facing
    ``get_particle_production_dxs_dE`` reaches a dict-stored MF6
    LAW=1 ``b`` coefficient and matches central finite-diff. Proves
    the tracer survives the full stack from top-level API through
    every dispatcher, integrator, and reconstruction kernel."""
    import jax
    import jax.numpy as jnp

    # Al-27 MF6 MT=16 (n, 2n) subsec 1 ei_mesh:
    # [1.354634e+07, 1.4e7, 1.5e7, 1.6e7, 1.8e7, 2e7, 1.5e8]
    # -> panel 1 (0-based) brackets E=1.4e7 exactly at its lower edge,
    # so pick a query slightly inside the panel.
    E_query = 1.45e7
    # Bracketing panel index computed from the file's ei_mesh:
    from endf_userpy.mfsec_interpretation import mf6_law1_preproc as mp
    data = mp.mf6_law1_data_from_endf_dict(al27_endf_dict, 16, 1)
    p = int(find_interval(np.asarray(data.ei_mesh), np.array([E_query]))[0])
    panel_key = p + 1  # 1-indexed dict key

    subsec = al27_endf_dict[6][16]['subsection'][1]
    ep_row = list(subsec['b'][panel_key].keys())[5]
    coef = 1
    original = float(subsec['b'][panel_key][ep_row][coef])

    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([E_query])
    eout = jnp.linspace(1e5, 5e6, 15)

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[6][16]['subsection'][1]['b'][panel_key][ep_row][coef] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return jnp.sum(get_particle_production_dxs_dE(
                d_t, '(n,2n)', 'n', ein, eout, xp=xp_jx,
            ))

    val = float(loss(jnp.array(original)))
    grad = float(jax.grad(loss)(jnp.array(original)))
    assert np.isfinite(grad)
    assert val > 0.0
    assert abs(grad) > 0.0
    eps = 1e-3 * abs(original) if original != 0.0 else 1e-6
    fd = (float(loss(jnp.array(original + eps))) - float(loss(jnp.array(original - eps)))) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-20)
