"""End-to-end demonstration for issue #154: ``jax.grad`` from a
file-stored Legendre coefficient array through the full MF6 LAW=2
reconstruction chain.

The dict-facing entry point ``get_angdist_from_subsec_law2`` marshals
the ENDF dict via :mod:`mf6_law2_preproc` into a
:class:`MF6Law2Data` dataclass and then hands off to
``_law2_reconstruct_from_data``. Users who want gradients wrt the
tabulated Legendre coefficients build the dataclass, replace its
``coeffs`` field with a JAX tracer via
:func:`dataclasses.replace`, and call the kernel directly with
``xp=array_ns.get_backend('jax')``.

Tests:

1. Dataclass field types are what the reconstruct layer expects.
2. Reconstruct-from-dataclass matches the dict-facing entry point
   bit-for-bit on numpy.
3. ``jax.grad`` of a scalar summary wrt a tracer-replaced ``coeffs``
   returns a finite, non-zero gradient matching finite differences.
4. Bonus (issue #154 shape 3): also exercises the dict-first
   path -- write a JAX tracer directly into ``endf_dict[6][mt]
   ['subsection'][sn]['A'][panel][coef_idx]`` and confirm the
   dict-facing entry point (with a JAX backend) produces a gradient
   through it via the tracer-preserving ``dict2array``.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_subsecs as mf6subsec,
    mf6_law2_preproc as mf6_l2p,
)
from endf_userpy.primitives import array_ns


def _jax_available():
    return 'jax' in array_ns.available_backends()


AL27_PATH = 'tests/data_law1_adhoc/endfb81_n_Al-27.endf'


def _resolve_al27():
    import os
    return AL27_PATH if os.path.exists(AL27_PATH) else None


def _load_al27():
    path = _resolve_al27()
    if path is None:
        pytest.skip(
            'Al-27 ENDF file not available (run '
            'tests/data_law1_adhoc/fetch.sh)'
        )
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def test_preproc_dataclass_fields_shape_and_types():
    d = _load_al27()
    data = mf6_l2p.mf6_law2_data_from_endf_dict(d, 51, 1)
    assert data.lang == 0                # Al-27 MT=51 is Legendre
    assert data.coeffs is not None
    assert data.records is None
    assert data.coeffs.ndim == 2
    assert data.coeffs.shape[0] == data.ei_mesh.shape[0]
    assert data.lct in (1, 2, 3)
    # a_0 == 1 * (0 + 0.5) = 0.5 after ENDF (L + 0.5) normalisation
    np.testing.assert_allclose(data.coeffs[:, 0], 0.5, rtol=1e-14)


def test_kernel_from_dataclass_matches_dict_entry_point_on_numpy():
    d = _load_al27()
    xp_np = array_ns.get_backend('numpy')
    e_in = np.array([5.0e6, 1.0e7])
    mu = np.linspace(-0.9, 0.9, 5)

    data = mf6_l2p.mf6_law2_data_from_endf_dict(d, 51, 1)
    kernel_out = np.asarray(mf6subsec._law2_reconstruct_from_data(
        data, e_in, mu, True, xp_np,
    ))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        dict_out = np.asarray(mf6subsec.get_angdist_from_subsec_law2(
            d, 51, 1, e_in, mu, True, xp=xp_np,
        ))
    np.testing.assert_array_equal(kernel_out, dict_out)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_flows_from_dataclass_coeffs_end_to_end():
    """Primary issue #154 demonstration: jax.grad of a scalar
    summary of the LAW=2 reconstruction wrt a tracer-replaced
    coefficient array."""
    import jax
    import jax.numpy as jnp
    d = _load_al27()
    xp_jax = array_ns.get_backend('jax')
    data = mf6_l2p.mf6_law2_data_from_endf_dict(d, 51, 1)
    coeffs0 = jnp.asarray(data.coeffs)
    e_in = np.array([5.0e6, 1.0e7])
    mu = np.linspace(-0.5, 0.5, 5)

    def loss(coeffs):
        d_traced = dataclasses.replace(data, coeffs=coeffs)
        out = mf6subsec._law2_reconstruct_from_data(
            d_traced, e_in, mu, True, xp_jax,
        )
        return jnp.sum(out)

    val = float(loss(coeffs0))
    grad = np.asarray(jax.grad(loss)(coeffs0))
    assert val > 0.0
    assert grad.shape == coeffs0.shape
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0.0), (
        f'gradient is identically zero -- tracer did not propagate: {grad}'
    )
    # Finite-diff sanity on the (1, 1) coefficient (first non-a_0
    # entry of panel 1).
    eps = 1e-4
    cp = coeffs0.at[1, 1].add(eps)
    cm = coeffs0.at[1, 1].add(-eps)
    fd = (float(loss(cp)) - float(loss(cm))) / (2.0 * eps)
    np.testing.assert_allclose(grad[1, 1], fd, rtol=5e-4, atol=1e-8)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_flows_from_dict_stored_tracer():
    """Issue #154 bonus (dict-first shape): user writes a JAX
    tracer directly into ``endf_dict[6][mt]['subsection'][sn]
    ['A'][panel][coef_idx]`` and calls the dict-facing entry point
    with a JAX backend. The tracer-preserving ``dict2array`` in
    the preproc keeps the tracer alive, and jax.grad reaches back
    through it. Less ergonomic than the dataclass shape (mutating
    a nested dict is unpleasant) but validated for completeness."""
    import copy
    import jax
    import jax.numpy as jnp
    d = _load_al27()
    xp_jax = array_ns.get_backend('jax')
    e_in = np.array([5.0e6])
    # Choose mu != 0 so P_1(mu) != 0 -- the a_1 coefficient we mutate
    # below enters via (L + 0.5) * a_1 * P_1(mu). At mu=0 the analytic
    # gradient is exactly zero and the pin would pass trivially.
    mu = np.array([0.5])

    # Mutate a_1 at the panel that BRACKETS the query E (panel 8 in
    # Al-27 MT=51's mesh: E=4.81 MeV, panel 9 = 5.44 MeV, query =
    # 5 MeV). A coefficient at any non-bracketing panel would not
    # contribute to the reconstruction at 5 MeV and the pin would
    # legitimately see grad=0.
    target_panel = 8
    target_coef_idx = 1  # a_1
    original = float(d[6][51]['subsection'][1]['A'][target_panel][target_coef_idx])

    def loss(theta):
        d_t = copy.deepcopy(d)
        d_t[6][51]['subsection'][1]['A'][target_panel][target_coef_idx] = theta
        out = mf6subsec.get_angdist_from_subsec_law2(
            d_t, 51, 1, e_in, mu, True, xp=xp_jax,
        )
        return jnp.sum(out)

    grad = float(jax.grad(loss)(jnp.array(original)))
    # The a_1 coefficient at the low panel enters the reconstruction
    # linearly (Legendre eval is linear in each a_L), and the sum
    # over a single query point produces a finite non-zero
    # derivative wrt that coefficient.
    assert np.isfinite(grad)
    assert grad != 0.0
    # Finite-diff sanity
    eps = 1e-4
    fp = float(loss(jnp.array(original + eps)))
    fm = float(loss(jnp.array(original - eps)))
    fd = (fp - fm) / (2.0 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-3, atol=1e-8)
