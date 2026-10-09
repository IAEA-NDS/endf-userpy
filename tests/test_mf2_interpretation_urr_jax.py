"""Tests for the JAX fluctuation-integral kernel
(:mod:`mf2_interpretation_urr_jax`) behind
``mf2_interpretation_urr.reconstruct`` on the ``'jax'`` backend with
the default Gauss-Legendre-32 quadrature.

Unit tests pin the per-channel mode selection and the factor
algebra against ``mf2_interpretation_urr._channel_factor``;
reconstruction tests run ENDF/B-VIII.1 Nb-93 (mixed INT=2 / INT=5
groups) in its native form and with the DOF / width tables replaced
so every mode ('exp', 'int', 'int0', 'general', 'zero') is
exercised on real tables.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf2_interpretation_urr as urr
from endf_userpy.mfsec_interpretation import (
    mf2_interpretation_urr_preproc as urr_pre,
)

jax = pytest.importorskip('jax')
import jax.numpy as jnp  # noqa: E402

from endf_userpy.mfsec_interpretation import mf2_interpretation_urr_jax as urr_jax  # noqa: E402


NB93_PATH = 'tests/data_law1_adhoc/endfb81_n_Nb-93.endf'
NE = 700          # not a multiple of the 256-energy chunk
KEYS = ('sct', 'cap', 'fis', 'rxx', 'pot', 'tot')


# ------------------------------------------------------------------
# Unit tests: mode selection and factor algebra.
# ------------------------------------------------------------------

def test_channel_mode_selection():
    assert urr_jax.channel_mode(np.array([0.0, 0.0])) == 'exp'
    assert urr_jax.channel_mode(np.array([1.0, 2.0, 4.0])) == 'int'
    assert urr_jax.channel_mode(np.array([0.0, 2.0, 3.0])) == 'int0'
    assert urr_jax.channel_mode(np.array([1.0, 1.0019])) == 'general'
    assert urr_jax.channel_mode(np.array([2.0]),
                                np.zeros((1, 5))) == 'zero'
    assert urr_jax.channel_mode(np.array([2.0]),
                                np.array([[0.0, 1e-3]])) == 'int'
    # A traced DOF row or width table never specialises on values.
    seen = []
    jax.make_jaxpr(lambda nu: seen.append(urr_jax.channel_mode(nu)))(
        jnp.array([1.0]))
    jax.make_jaxpr(lambda tab: seen.append(
        urr_jax.channel_mode(np.array([0.0]), tab)))(jnp.zeros((1, 3)))
    assert seen == ['general', 'exp']


@pytest.mark.parametrize('nu_row', [
    [0.0, 0.0, 0.0],
    [1.0, 2.0, 4.0],
    [0.0, 3.0, 1.0],
    [1.5, 0.0, 2.3],
])
def test_channel_factors_match_numpy_definition(nu_row):
    """All three orders from the shared-base algebra equal the
    independent ``_channel_factor`` evaluation."""
    nu = np.array([nu_row])                                  # (1, nJ)
    alpha = np.array([[0.0, 1e-3, 0.7], [2.5, 0.0, 30.0]])  # (2, nJ)
    t = np.asarray(urr._T_NODES)
    mode = urr_jax.channel_mode(nu)
    got = urr_jax._channel_factors(
        jnp.asarray(alpha)[..., None], jnp.asarray(nu)[..., None],
        jnp.asarray(t), mode,
    )
    xp = array_ns.get_backend('numpy')
    for order in (0, 1, 2):
        if mode == 'exp' and order == 2:
            continue   # neutron-only order; neutron DOF is never 0
        ref = urr._channel_factor(alpha, np.broadcast_to(nu, alpha.shape),
                                  t, order, xp)
        np.testing.assert_allclose(np.asarray(got[order]), ref,
                                   rtol=1e-14, atol=0.0,
                                   err_msg=f'mode={mode} order={order}')


# ------------------------------------------------------------------
# Reconstruction on Nb-93, native and with forced modes.
# ------------------------------------------------------------------

@pytest.fixture(scope='module')
def nb93():
    if not os.path.exists(NB93_PATH):
        pytest.skip('Nb-93 corpus not available')
    from endf_parserpy import EndfParserCpp
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(NB93_PATH)
    rng = d[2][151]['isotope'][1]['range']
    ri = [k for k, r in rng.items() if r['LRU'] == 2][0]
    e = np.geomspace(rng[ri]['EL'], rng[ri]['EH'], NE)
    return urr_pre.urr_data_from_endf_dict(d, range_idx=ri), e


def _variants(data):
    """Native data plus replacements forcing the remaining modes on
    real width tables: fission 'int0' / competitive 'general' with
    non-zero widths, and the reverse."""
    nj = len(np.asarray(data.group_l))
    gg = np.asarray(data.table_gg)
    mix0 = np.where(np.arange(nj) % 2 == 0, 0.0, 2.0)
    frac = 1.0 + 0.37 * (np.arange(nj) % 3)
    return {
        'native': data,
        'f-int0_x-general': dataclasses.replace(
            data, group_amuf=mix0, table_gf=0.5 * gg,
            group_amux=frac, table_gx=0.2 * gg,
        ),
        'f-general_x-int': dataclasses.replace(
            data, group_amuf=frac, table_gf=0.3 * gg,
            group_amux=np.full(nj, 4.0), table_gx=0.1 * gg,
        ),
    }


def _modes(data):
    return (urr_jax.channel_mode(data.group_amun),) + tuple(
        urr_jax.channel_mode(getattr(data, 'group_' + c),
                             getattr(data, 'table_' + t))
        for c, t in (('amug', 'gg'), ('amuf', 'gf'), ('amux', 'gx')))


def test_urr_jax_matches_numpy_all_modes(nb93):
    data, e = nb93
    seen = set()
    for name, dv in _variants(data).items():
        seen.update(_modes(dv))
        ref = urr.reconstruct(dv, e, array_ns.get_backend('numpy'))
        got = urr.reconstruct(dv, e, array_ns.get_backend('jax'))
        for key in KEYS:
            np.testing.assert_allclose(
                np.asarray(got[key]), ref[key],
                rtol=1e-13, atol=1e-14 * max(np.max(np.abs(ref[key])), 1.0),
                err_msg=f'{name}: jax vs numpy disagree on {key}',
            )
    assert seen >= {'exp', 'int', 'int0', 'general', 'zero'}, seen


def _iter_avals(jaxpr):
    for eqn in jaxpr.eqns:
        for v in eqn.outvars:
            yield v.aval
        for p in eqn.params.values():
            subs = p if isinstance(p, (list, tuple)) else (p,)
            for sub in subs:
                inner = getattr(sub, 'jaxpr', sub)
                if hasattr(inner, 'eqns'):
                    yield from _iter_avals(inner)


def test_urr_jax_has_no_node_resolved_full_mesh_intermediate(nb93):
    """No intermediate as large as ``(NE, nJ, Nq)``: the node axis is
    only ever materialised per energy chunk."""
    data, _ = nb93
    xp = array_ns.get_backend('jax')
    ne = 4096
    nj = len(np.asarray(data.group_l))
    nq = len(urr._T_NODES)

    def f(e):
        return urr.reconstruct(data, e, xp)['tot']

    jaxpr = jax.make_jaxpr(f)(jnp.linspace(1e3, 5e4, ne))
    largest = max(int(np.prod(a.shape)) for a in _iter_avals(jaxpr.jaxpr)
                  if hasattr(a, 'shape'))
    assert largest < ne * nj * nq // 4, largest


def test_urr_jax_grad_matches_forward_mode(nb93):
    """Reverse-mode ``jax.grad`` through the checkpointed chunk map wrt
    the neutron, capture and fission width tables matches forward-mode
    ``jax.jvp`` along a random direction (fission table traced, so its
    channel runs the general path even where the native table is 0)."""
    data, e = nb93
    dv = _variants(data)['f-int0_x-general']
    xp = array_ns.get_backend('jax')
    keys = ('table_gn0', 'table_gg', 'table_gf')

    def loss(*tabs):
        out = urr.reconstruct(dataclasses.replace(dv, **dict(zip(keys, tabs))),
                              e, xp)
        return jnp.sum(out['tot'] + out['fis'] + out['cap'])

    args = tuple(jnp.asarray(getattr(dv, k)) for k in keys)
    grads = jax.grad(loss, argnums=(0, 1, 2))(*args)
    rng = np.random.default_rng(3)
    for a, key in enumerate(keys):
        v = jnp.asarray(rng.normal(size=args[a].shape)) * args[a]
        tangents = tuple(v if b == a else jnp.zeros_like(args[b])
                         for b in range(3))
        _, jvp = jax.jvp(loss, args, tangents)
        rev = float(jnp.sum(grads[a] * v))
        assert np.isfinite(rev) and abs(float(jvp)) > 0.0, key
        assert abs(rev - float(jvp)) <= 1e-10 * abs(float(jvp)), (key, rev, jvp)
