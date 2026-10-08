"""Tests for the JAX blocked channel-sum accumulator
(:mod:`mf2_interpretation_mlbw_jax`) behind
``mf2_interpretation_mlbw.reconstruct`` on the ``'jax'`` backend.

Synthetic data with four channels of mixed L (so the per-L factor
tables have several rows and the competitive ``lx = |L - 2|``
remapping for a spin-0 target differs from L), a competitive width
with a threshold, and a resonance count that is not a multiple of
the block size.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.tab1 import TAB1
from endf_userpy.mfsec_interpretation import mf2_interpretation_mlbw as mlbw

jax = pytest.importorskip('jax')
import jax.numpy as jnp  # noqa: E402


NRES = 203
NE = 500
# One (block, NE) float64 block of 7 resonances: 29 blocks, last one padded.
MAX_BYTES = 8 * NE * 7
KEYS = ('sct', 'cap', 'fis', 'pot', 'rxx', 'tot')


def _constant_tab1(value: float) -> TAB1:
    return TAB1(
        x=np.array([1e-5, 1e10], dtype=np.float64),
        y=np.array([value, value], dtype=np.float64),
        nbt=np.array([1], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )


def _four_channel_data(seed=0):
    rng = np.random.default_rng(seed)
    ch_l = np.array([0, 1, 2, 1], dtype=np.int32)
    res_channel = rng.integers(0, 4, NRES).astype(np.int32)
    return mlbw.MLBWData(
        abn=1.0, spi=0.0, ki=2.196771e-3, qx=-50.0,
        r_a=_constant_tab1(0.62), r_ap=_constant_tab1(0.65),
        ch_l=ch_l,
        ch_g=np.array([1.0, 1.0, 1.0, 1.0]),
        res_channel=res_channel,
        res_l=ch_l[res_channel],
        res_er=np.sort(rng.uniform(-20.0, 1000.0, NRES)),
        res_gn=rng.uniform(1e-3, 5e-2, NRES),
        res_gg=rng.uniform(0.02, 0.05, NRES),
        res_gf=rng.uniform(0.0, 0.02, NRES),
        res_gx=rng.uniform(0.0, 0.03, NRES),
    )


def test_mlbw_jax_blocked_matches_numpy_multi_channel():
    """Blocked-scan JAX reconstruction agrees with the dense numpy
    path to round-off on every partial cross section."""
    data = _four_channel_data()
    e = np.linspace(1e-3, 1100.0, NE)
    ref = mlbw.reconstruct(data, e, array_ns.get_backend('numpy'))
    got = mlbw.reconstruct(data, e, array_ns.get_backend('jax'),
                           _max_intermediate_bytes=MAX_BYTES)
    for key in KEYS:
        np.testing.assert_allclose(
            np.asarray(got[key]), ref[key],
            rtol=1e-12, atol=1e-14 * np.max(np.abs(ref[key])),
            err_msg=f'blocked jax vs numpy disagree on {key}',
        )
    # Not vacuous: capture, fission and competitive all resonate.
    for key in ('cap', 'fis', 'rxx'):
        assert np.max(ref[key]) > 1.0, key


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


def test_mlbw_jax_has_no_dense_energy_by_resonance_intermediate():
    """The traced JAX reconstruction never materialises an array as
    large as ``(NE, NRES)``: per-resonance terms are evaluated and
    summed block by block."""
    data = _four_channel_data()
    xp = array_ns.get_backend('jax')

    def f(e):
        return mlbw.reconstruct(data, e, xp,
                                _max_intermediate_bytes=MAX_BYTES)['tot']

    jaxpr = jax.make_jaxpr(f)(jnp.linspace(1e-3, 1100.0, NE))
    largest = max(int(np.prod(a.shape)) for a in _iter_avals(jaxpr.jaxpr)
                  if hasattr(a, 'shape'))
    assert largest < NE * NRES // 4, largest


def test_mlbw_jax_blocked_grad_matches_forward_mode_and_fd():
    """Reverse-mode ``jax.grad`` through the checkpointed blocked scan
    wrt every per-resonance width and energy matches forward-mode
    ``jax.jvp`` along a random direction, and a central finite
    difference on one neutron width in the padded last block."""
    import dataclasses
    base = _four_channel_data()
    xp = array_ns.get_backend('jax')
    e = jnp.linspace(1e-3, 1100.0, NE)
    keys = ('res_er', 'res_gn', 'res_gg', 'res_gf', 'res_gx')

    def loss(*leaves):
        data = dataclasses.replace(base, **dict(zip(keys, leaves)))
        out = mlbw.reconstruct(data, e, xp, _max_intermediate_bytes=MAX_BYTES)
        return jnp.sum(out['tot'] + out['rxx'])

    args = tuple(jnp.asarray(getattr(base, k)) for k in keys)
    grads = jax.grad(loss, argnums=tuple(range(len(keys))))(*args)
    rng = np.random.default_rng(1)
    for a, key in enumerate(keys):
        v = jnp.asarray(rng.normal(size=NRES))
        tangents = tuple(v if b == a else jnp.zeros(NRES)
                         for b in range(len(keys)))
        _, jvp = jax.jvp(loss, args, tangents)
        rev = float(jnp.dot(grads[a], v))
        assert abs(float(jvp)) > 0.0, key
        assert abs(rev - float(jvp)) <= 1e-10 * abs(float(jvp)), (key, rev, jvp)

    i = NRES - 1                                   # in the padded last block
    h = 1e-2 * float(args[1][i])
    fd = (float(loss(args[0], args[1].at[i].add(h), *args[2:]))
          - float(loss(args[0], args[1].at[i].add(-h), *args[2:]))) / (2 * h)
    assert abs(float(grads[1][i]) - fd) <= 1e-3 * abs(fd), (grads[1][i], fd)
