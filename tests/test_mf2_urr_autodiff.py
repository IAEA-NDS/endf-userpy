"""MF2 LRU=2 LRF=2 (URR, Case C) autodiff wrt width parameters.

Both fluctuation-integral quadratures (``gauss_legendre_32`` and
``ross_10``) are covered. Tests use ENDF/B-VIII.1 Nb-93, the
smallest URR file in the adhoc corpus with a mixed set of INT
codes (per J-group).

Autodiff coverage pinned here:

- ``jax.grad`` wrt ``GG`` (capture width) matches central FD.
- ``jax.grad`` wrt ``GN0`` (neutron width, reduced) matches
  central FD on an INT=5 (log-log) group. The pre-fix kernel
  broke here because the log-log-vs-lin-lin fallback ran
  ``bool(xp.all(y_row > 0))`` on a tracer.
- ``jax.grad(jax.jit(loss))`` and ``jax.jit(jax.grad(loss))``
  both compose.
- Numpy vs jax reconstruction is bit-parity on both quadratures.
"""
from __future__ import annotations

import copy
import os
import warnings

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf2_interpretation_urr as urr
from endf_userpy.mfsec_interpretation import (
    mf2_interpretation_urr_preproc as urr_pre,
)


NB93_PATH = 'tests/data_law1_adhoc/endfb81_n_Nb-93.endf'


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


@pytest.fixture(scope='module')
def nb93_dict():
    if not os.path.exists(NB93_PATH):
        pytest.skip('Nb-93 corpus not available')
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(NB93_PATH)


def _perturb_leaf(d, path, key, value):
    """Descend d[path[0]][path[1]]... to a leaf dict, set d[...][key] = value.
    Returns the mutated deep copy."""
    d_t = copy.deepcopy(d)
    cur = d_t
    for p in path:
        cur = cur[p]
    cur[key] = value
    return d_t


NB93_GN0_PATH = (2, 151, 'isotope', 1, 'range', 2, 'l_group', 1, 'subsec', 1, 'GN0')
NB93_GG_PATH = (2, 151, 'isotope', 1, 'range', 2, 'l_group', 1, 'subsec', 1, 'GG')


def test_urr_numpy_jax_parity_gauss_legendre(nb93_dict):
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    data_np = urr_pre.urr_data_from_endf_dict(nb93_dict)
    data_jx = urr_pre.urr_data_from_endf_dict(nb93_dict, xp=xp_jx)
    ein = np.array([9000.0, 35000.0, 200000.0])
    r_np = urr.reconstruct(data_np, ein, xp_np)
    r_jx = urr.reconstruct(data_jx, ein, xp_jx)
    for k in ('sct', 'cap', 'fis', 'tot', 'pot'):
        np.testing.assert_allclose(
            np.asarray(r_np[k]), np.asarray(r_jx[k]),
            rtol=1e-12, atol=1e-14,
        )


def test_urr_numpy_jax_parity_ross_10(nb93_dict):
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    data_np = urr_pre.urr_data_from_endf_dict(nb93_dict)
    data_jx = urr_pre.urr_data_from_endf_dict(nb93_dict, xp=xp_jx)
    ein = np.array([9000.0, 35000.0, 200000.0])
    r_np = urr.reconstruct(data_np, ein, xp_np, quadrature='ross_10')
    r_jx = urr.reconstruct(data_jx, ein, xp_jx, quadrature='ross_10')
    for k in ('sct', 'cap', 'fis', 'tot', 'pot'):
        np.testing.assert_allclose(
            np.asarray(r_np[k]), np.asarray(r_jx[k]),
            rtol=1e-12, atol=1e-14,
        )


def test_urr_grad_wrt_GG_matches_fd(nb93_dict):
    """``jax.grad`` wrt capture width matches central FD."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([9000.0, 35000.0, 200000.0])
    gg_orig = 0.173

    def loss(x):
        d_t = _perturb_leaf(nb93_dict, NB93_GG_PATH[:-1], 'GG', {**dict(nb93_dict[2][151]['isotope'][1]['range'][2]['l_group'][1]['subsec'][1]['GG']), 3: x})
        # Simpler: just mutate the specific key.
        d_t = copy.deepcopy(nb93_dict)
        d_t[2][151]['isotope'][1]['range'][2]['l_group'][1]['subsec'][1]['GG'][3] = x
        data = urr_pre.urr_data_from_endf_dict(d_t, xp=xp_jx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = urr.reconstruct(data, ein, xp_jx)
        return jnp.sum(r['cap'])

    val = float(loss(jnp.array(gg_orig)))
    grad = float(jax.grad(loss)(jnp.array(gg_orig)))
    eps = gg_orig * 1e-3
    fd = (
        float(loss(jnp.array(gg_orig + eps)))
        - float(loss(jnp.array(gg_orig - eps)))
    ) / (2 * eps)
    assert val > 0.0
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-20)


def test_urr_grad_wrt_GN0_int5_group_matches_fd(nb93_dict):
    """``jax.grad`` wrt neutron width on an INT=5 (log-log) group.

    Regression: the pre-fix ``_interp_per_group`` broke here with
    ``TracerBoolConversionError`` because the log-log-vs-lin-lin
    fallback ran ``bool(xp.all(y_row > 0))`` on the tracer'd
    width row.
    """
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([9000.0, 35000.0, 200000.0])
    gn0_orig = 0.013869

    def loss(x):
        d_t = copy.deepcopy(nb93_dict)
        d_t[2][151]['isotope'][1]['range'][2]['l_group'][1]['subsec'][1]['GN0'][3] = x
        data = urr_pre.urr_data_from_endf_dict(d_t, xp=xp_jx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = urr.reconstruct(data, ein, xp_jx)
        return jnp.sum(r['sct'])

    val = float(loss(jnp.array(gn0_orig)))
    grad = float(jax.grad(loss)(jnp.array(gn0_orig)))
    eps = gn0_orig * 1e-3
    fd = (
        float(loss(jnp.array(gn0_orig + eps)))
        - float(loss(jnp.array(gn0_orig - eps)))
    ) / (2 * eps)
    assert val > 0.0
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-20)


def test_urr_jit_composability_GN0(nb93_dict):
    """``jax.grad(jax.jit(loss))`` and ``jax.jit(jax.grad(loss))``
    both match eager ``jax.grad`` on the URR path."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([9000.0, 35000.0, 200000.0])
    gn0_orig = 0.013869

    def loss(x):
        d_t = copy.deepcopy(nb93_dict)
        d_t[2][151]['isotope'][1]['range'][2]['l_group'][1]['subsec'][1]['GN0'][3] = x
        data = urr_pre.urr_data_from_endf_dict(d_t, xp=xp_jx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = urr.reconstruct(data, ein, xp_jx)
        return jnp.sum(r['sct'])

    x = jnp.array(gn0_orig)
    g_eager = float(jax.grad(loss)(x))
    g_grad_of_jit = float(jax.grad(jax.jit(loss))(x))
    jit_grad = jax.jit(jax.grad(loss))
    _ = jit_grad(x)  # warmup
    g_jit_of_grad = float(jit_grad(x))
    for label, val in (
        ('grad(jit(loss))', g_grad_of_jit),
        ('jit(grad(loss))', g_jit_of_grad),
    ):
        assert np.isfinite(val)
        np.testing.assert_allclose(
            val, g_eager, rtol=1e-6, atol=1e-20,
            err_msg=f'{label} vs eager grad',
        )


def test_urr_grad_wrt_GG_matches_fd_ross_10(nb93_dict):
    """``jax.grad`` wrt GG also reaches through the alternative
    Ross-10 quadrature path. Regression for the in-place r[:, g] =
    write in ``_reconstruct_ross_moments`` that was jax-unsafe
    before this fix.
    """
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([9000.0, 35000.0, 200000.0])
    gg_orig = 0.173

    def loss(x):
        d_t = copy.deepcopy(nb93_dict)
        d_t[2][151]['isotope'][1]['range'][2]['l_group'][1]['subsec'][1]['GG'][3] = x
        data = urr_pre.urr_data_from_endf_dict(d_t, xp=xp_jx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = urr.reconstruct(data, ein, xp_jx, quadrature='ross_10')
        return jnp.sum(r['cap'])

    val = float(loss(jnp.array(gg_orig)))
    grad = float(jax.grad(loss)(jnp.array(gg_orig)))
    eps = gg_orig * 1e-3
    fd = (
        float(loss(jnp.array(gg_orig + eps)))
        - float(loss(jnp.array(gg_orig - eps)))
    ) / (2 * eps)
    assert val > 0.0
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-20)
