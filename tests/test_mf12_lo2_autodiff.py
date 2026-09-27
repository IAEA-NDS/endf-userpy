"""MF12 LO=2 transition-probability cascade autodiff.

Companion to ``test_mf12_trans2yield.py`` (Fortran-parity pins the
port). This file covers the ``xp`` adapter wired through the
kernel + helpers: numpy is bit-identical to the pre-rewrite output
via the Fortran-parity suite, and ``jax.grad`` reaches ``TP``
(direct transition probabilities) and ``GP`` (photon-vs-conversion
branching) on real corpus files.

The photon-energy axis is a discrete lookup (``argmin`` under
tracers, ``find_indices_with_tol`` under concrete inputs), so
grads wrt ``ES_NS`` / ``ES`` / ``QM`` / ``QI`` are structurally
zero at fixed query energies. That behaviour is pinned here as a
separate test so a future rewrite that plumbs those leaves through
does not silently break it.
"""
from __future__ import annotations

import copy
import os
import warnings

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf12_interpretation as mf12


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


PB208_PATH = 'tests/data_law1_adhoc/endfb81_n_Pb-208.endf'
K39_PATH = 'tests/data_law1_adhoc/endfb81_n_K-39.endf'
NI58_PATH = 'tests/data_law1_adhoc/endfb81_n_Ni-58.endf'


@pytest.fixture(scope='module')
def pb208_dict():
    if not os.path.exists(PB208_PATH):
        pytest.skip('Pb-208 corpus not available')
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(PB208_PATH)


@pytest.fixture(scope='module')
def k39_dict():
    if not os.path.exists(K39_PATH):
        pytest.skip('K-39 corpus not available')
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(K39_PATH)


@pytest.fixture(scope='module')
def ni58_dict():
    if not os.path.exists(NI58_PATH):
        pytest.skip('Ni-58 corpus not available')
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(NI58_PATH)


def _list_key(container, idx):
    """Return the key to write into a dict-like or list container
    at the ``idx``-th position (endf_parserpy uses both)."""
    if hasattr(container, 'keys'):
        return list(container.keys())[idx]
    return idx


def test_lo2_numpy_jax_parity_pb208_mt53(pb208_dict):
    """Numpy and jax adapters agree on the transition-probability
    cascade for a mid-cascade Pb-208 MT."""
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    res_np = mf12.compute_photon_yields_from_transition_probabilities(
        pb208_dict, 53, xp=xp_np,
    )
    res_jx = mf12.compute_photon_yields_from_transition_probabilities(
        pb208_dict, 53, xp=xp_jx,
    )
    # The jax path emits a fixed-shape (j0*j0,) grid; the numpy
    # path filters zero-yield entries. Match by comparing the
    # nonzero-yield rows only, sorted by descending photon_energy.
    pe_np = np.asarray(res_np['photon_energy'])
    py_np = np.asarray(res_np['photon_yield'])
    pe_jx_full = np.asarray(res_jx['photon_energy'])
    py_jx_full = np.asarray(res_jx['photon_yield'])
    keep_jx = py_jx_full > 0.0
    pe_jx = pe_jx_full[keep_jx]
    py_jx = py_jx_full[keep_jx]
    order_np = np.argsort(-pe_np)
    order_jx = np.argsort(-pe_jx)
    np.testing.assert_allclose(pe_np[order_np], pe_jx[order_jx], rtol=1e-12)
    np.testing.assert_allclose(py_np[order_np], py_jx[order_jx], rtol=1e-12)


def test_lo2_grad_wrt_TP_matches_fd_pb208(pb208_dict):
    """``jax.grad`` wrt an MF12 LO=2 direct transition probability
    (TP) matches central FD on Pb-208 MT=53."""
    import jax
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    target_mt = 53
    res_np = mf12.compute_photon_yields_from_transition_probabilities(
        pb208_dict, target_mt, xp=xp_np,
    )
    pe_query = np.asarray(res_np['photon_energy']).copy()
    ein = np.array([1e6, 5e6, 1e7])
    tp_dict = pb208_dict[12][target_mt]['TP']
    tp_key = _list_key(tp_dict, 1)  # second TP entry (in the cascade)
    tp0 = float(tp_dict[tp_key])

    def loss(x):
        d_t = copy.deepcopy(pb208_dict)
        d_t[12][target_mt]['TP'][tp_key] = x
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y = mf12.compute_photon_yields(
                d_t, target_mt, ein, pe_query, xp=xp_jx,
            )
        return jnp.sum(y)

    val = float(loss(jnp.array(tp0)))
    grad = float(jax.grad(loss)(jnp.array(tp0)))
    eps = 1e-3
    fd = (
        float(loss(jnp.array(tp0 + eps)))
        - float(loss(jnp.array(tp0 - eps)))
    ) / (2 * eps)
    assert val > 0.0
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-30)


def test_lo2_grad_wrt_GP_matches_fd_k39(k39_dict):
    """``jax.grad`` wrt an MF12 LO=2 photon-vs-conversion factor
    (GP, only present when LG=2) matches central FD on K-39 MT=51."""
    import jax
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    # find an LG=2 MT
    target_mt = None
    for mt, sec in k39_dict[12].items():
        if sec.get('LO') == 2 and sec.get('LG') == 2:
            target_mt = mt
            break
    assert target_mt is not None, 'expected an LG=2 MT in K-39'

    res_np = mf12.compute_photon_yields_from_transition_probabilities(
        k39_dict, target_mt, xp=xp_np,
    )
    pe_query = np.asarray(res_np['photon_energy']).copy()
    ein = np.array([1e6, 5e6, 1e7])
    gp_dict = k39_dict[12][target_mt]['GP']
    gp_key = _list_key(gp_dict, 0)
    gp0 = float(gp_dict[gp_key])

    def loss(x):
        d_t = copy.deepcopy(k39_dict)
        d_t[12][target_mt]['GP'][gp_key] = x
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y = mf12.compute_photon_yields(
                d_t, target_mt, ein, pe_query, xp=xp_jx,
            )
        return jnp.sum(y)

    val = float(loss(jnp.array(gp0)))
    grad = float(jax.grad(loss)(jnp.array(gp0)))
    eps = 1e-3
    fd = (
        float(loss(jnp.array(gp0 + eps)))
        - float(loss(jnp.array(gp0 - eps)))
    ) / (2 * eps)
    assert val > 0.0
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-30)


def test_lo2_jit_composability_tp(pb208_dict):
    """Both ``jax.grad(jax.jit(loss))`` and ``jax.jit(jax.grad(loss))``
    match eager ``jax.grad`` for LO=2 wrt TP on Pb-208 MT=53."""
    import jax
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    target_mt = 53
    res_np = mf12.compute_photon_yields_from_transition_probabilities(
        pb208_dict, target_mt, xp=xp_np,
    )
    pe_query = np.asarray(res_np['photon_energy']).copy()
    ein = np.array([1e6, 5e6, 1e7])
    tp_dict = pb208_dict[12][target_mt]['TP']
    tp_key = _list_key(tp_dict, 1)
    tp0 = float(tp_dict[tp_key])

    def loss(x):
        d_t = copy.deepcopy(pb208_dict)
        d_t[12][target_mt]['TP'][tp_key] = x
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y = mf12.compute_photon_yields(
                d_t, target_mt, ein, pe_query, xp=xp_jx,
            )
        return jnp.sum(y)

    x = jnp.array(tp0)
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


def test_lo2_argmin_lookup_gives_zero_grad_wrt_level_energies(pb208_dict):
    """The photon-energy match is a discrete lookup, so
    ``jax.grad`` wrt ``QM`` / ``ES_NS`` / ``ES`` at fixed query
    photon energies is structurally zero. This test pins that
    contract so a future rewrite that plumbs these leaves through
    with continuous relaxation does not silently break it.
    """
    import jax
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    target_mt = 53
    res_np = mf12.compute_photon_yields_from_transition_probabilities(
        pb208_dict, target_mt, xp=xp_np,
    )
    pe_query = np.asarray(res_np['photon_energy']).copy()
    ein = np.array([1e6, 5e6, 1e7])

    def loss_esns(x):
        d_t = copy.deepcopy(pb208_dict)
        d_t[12][target_mt]['ES_NS'] = x
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y = mf12.compute_photon_yields(
                d_t, target_mt, ein, pe_query, xp=xp_jx,
            )
        return jnp.sum(y)

    esns0 = float(pb208_dict[12][target_mt]['ES_NS'])
    g_esns = float(jax.grad(loss_esns)(jnp.array(esns0)))
    assert g_esns == 0.0

    def loss_qm(x):
        d_t = copy.deepcopy(pb208_dict)
        d_t[3][51]['QM'] = x
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y = mf12.compute_photon_yields(
                d_t, target_mt, ein, pe_query, xp=xp_jx,
            )
        return jnp.sum(y)

    qm0 = float(pb208_dict[3][51]['QM'])
    g_qm = float(jax.grad(loss_qm)(jnp.array(qm0)))
    assert g_qm == 0.0


def test_lo2_ni58_simple_case_jit(ni58_dict):
    """Simplest LO=2 case (Ni-58 MT=51 single-line, yield 1) still
    survives ``jax.jit`` on the wrapper, without any tracer file
    leaves. Regression for the concrete + jit code path."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([1e6, 5e6, 1e7])
    photon_energies = np.array([1454000.0])

    @jax.jit
    def f(ein):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return mf12.compute_photon_yields(
                ni58_dict, 51, ein, photon_energies, xp=xp_jx,
            )

    y = np.asarray(f(ein).block_until_ready())
    np.testing.assert_allclose(y, np.ones((3, 1)), rtol=1e-12)
