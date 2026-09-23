"""``get_reaction_xs`` backend-agnostic port (issue #169 tier-2).

Pins:

- ``xp=None`` (default) and ``xp=numpy`` bit-identical.
- With ``include_resonance=True`` on an MLBW file, ``xp=jax``
  matches numpy to machine precision.
- ``jax.grad(get_reaction_xs)`` reaches file-side MF2 dict-stored
  resonance parameters (MLBW ``ER``) and matches central finite-
  diff. Composes the top-level API with the MF2 preproc dict-first
  path landed in PRs #170 / #171 / #172.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_reaction_xs
from endf_userpy.primitives import array_ns

from _corpus import resolve_nb93


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def nb93_endf_dict():
    path = resolve_nb93()
    if path is None:
        pytest.skip('Nb-93 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def test_default_matches_xp_numpy_adapter(nb93_endf_dict):
    """xp=None default and xp=numpy adapter bit-identical for
    ``get_reaction_xs`` on the raw-MF3 path (no resonance
    composition)."""
    ein = np.array([1e-3, 1.0, 100.0, 5000.0, 20000.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        default = np.asarray(get_reaction_xs(nb93_endf_dict, '(n,total)', ein))
        xp_np = array_ns.get_backend('numpy')
        with_xp = np.asarray(get_reaction_xs(
            nb93_endf_dict, '(n,total)', ein, xp=xp_np,
        ))
    np.testing.assert_array_equal(default, with_xp)


def test_default_matches_xp_numpy_with_resonance(nb93_endf_dict):
    """Same for the include_resonance=True composition path."""
    ein = np.array([1e-3, 1.0, 100.0])
    default = np.asarray(get_reaction_xs(
        nb93_endf_dict, '(n,total)', ein, include_resonance=True,
    ))
    xp_np = array_ns.get_backend('numpy')
    with_xp = np.asarray(get_reaction_xs(
        nb93_endf_dict, '(n,total)', ein, include_resonance=True, xp=xp_np,
    ))
    np.testing.assert_array_equal(default, with_xp)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_numpy_jax_parity_include_resonance_nb93(nb93_endf_dict):
    """JAX adapter matches numpy to machine precision when
    ``include_resonance=True`` composes MF3 + MF2 MLBW."""
    ein = np.array([1e-3, 1.0, 35.0, 100.0])
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    f_np = np.asarray(get_reaction_xs(
        nb93_endf_dict, '(n,total)', ein, include_resonance=True, xp=xp_np,
    ))
    f_jx = np.asarray(get_reaction_xs(
        nb93_endf_dict, '(n,total)', ein, include_resonance=True, xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_wrt_mlbw_ER_end_to_end_nb93(nb93_endf_dict):
    """Primary tier-2 demonstration: ``jax.grad`` of a scalar
    summary of ``get_reaction_xs`` reaches a dict-stored MLBW
    resonance-energy leaf and matches central finite-diff. Proves
    the tracer survives the full stack from the top-level API,
    through the resonance-composition layer (PR #170's dict-first
    MLBW preproc), the R-matrix reconstruction, and the MF3
    additive background."""
    import jax
    import jax.numpy as jnp

    ein = jnp.array([1e-3, 1.0, 35.0, 100.0])
    xp_jx = array_ns.get_backend('jax')

    l_group = (
        nb93_endf_dict[2][151]['isotope'][1]['range'][1].get('l_group')
        or nb93_endf_dict[2][151]['isotope'][1]['range'][1]['spingroup']
    )
    er_row = list(l_group[1]['ER'].keys())[3]
    original = float(l_group[1]['ER'][er_row])

    def loss(theta):
        d_t = copy.deepcopy(nb93_endf_dict)
        lg = (
            d_t[2][151]['isotope'][1]['range'][1].get('l_group')
            or d_t[2][151]['isotope'][1]['range'][1]['spingroup']
        )
        lg[1]['ER'][er_row] = theta
        return jnp.sum(get_reaction_xs(
            d_t, '(n,total)', ein, include_resonance=True, xp=xp_jx,
        ))

    val = float(loss(jnp.array(original)))
    grad = float(jax.grad(loss)(jnp.array(original)))
    assert np.isfinite(grad)
    assert val > 0.0
    eps = 1e-3 * abs(original) if original != 0.0 else 1e-6
    fd = (float(loss(jnp.array(original + eps))) - float(loss(jnp.array(original - eps)))) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-20)
