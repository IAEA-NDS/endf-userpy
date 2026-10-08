"""Tests for the JAX blocked R-matrix accumulator
(:mod:`mf2_interpretation_reichmoore_jax`) behind
``mf2_interpretation_reichmoore.reconstruct`` on the ``'jax'``
backend.

Synthetic multi-group data with a resonance count that is not a
multiple of the block size, so the zero-weight padding, the
one-hot group scatter and the per-group channel counts are all
exercised.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.tab1 import TAB1
from endf_userpy.mfsec_interpretation import mf2_interpretation_reichmoore as rm

jax = pytest.importorskip('jax')
import jax.numpy as jnp  # noqa: E402

from endf_userpy.mfsec_interpretation import mf2_interpretation_reichmoore_jax as rm_jax  # noqa: E402


NRES = 203
NE = 500
# One (block, NE) float64 block of 7 resonances: 29 blocks, last one padded.
MAX_BYTES = 8 * NE * 7


def _constant_tab1(value: float) -> TAB1:
    return TAB1(
        x=np.array([1e-5, 1e10], dtype=np.float64),
        y=np.array([value, value], dtype=np.float64),
        nbt=np.array([1], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )


def _three_group_data(seed=0):
    """Three J·π groups: (L=0, no fission), (L=1, one fission
    channel), (L=2, two fission channels), each with its own channel
    radius."""
    rng = np.random.default_rng(seed)
    res_group = rng.integers(0, 3, NRES).astype(np.int32)
    nfis = np.array([0, 1, 2], dtype=np.int32)
    gf1 = rng.normal(0.0, 0.1, NRES) * (nfis[res_group] >= 1)
    gf2 = rng.normal(0.0, 0.1, NRES) * (nfis[res_group] >= 2)
    return rm.RMData(
        abn=1.0, spi=0.5, ki=2.196771e-3,
        r_a=_constant_tab1(0.96), r_ap=_constant_tab1(0.96),
        group_l=np.array([0, 1, 2], dtype=np.int32),
        group_g=np.array([0.25, 0.75, 0.5], dtype=np.float64),
        group_nfis=nfis,
        res_group=res_group,
        res_er=np.sort(rng.uniform(-20.0, 1000.0, NRES)),
        res_gn=rng.uniform(1e-4, 1e-2, NRES) * rng.choice([-1, 1], NRES),
        res_gg=rng.uniform(0.02, 0.05, NRES),
        res_gf1=gf1,
        res_gf2=gf2,
        group_r_a=np.array([0.9, 0.95, 1.0]),
        group_r_ap=np.array([0.92, 0.96, 1.01]),
    )


def test_block_size_respects_byte_cap_and_upper_bound():
    assert rm_jax.block_size(NE, MAX_BYTES) == 7
    assert rm_jax.block_size(1_000_000, 512 * 2**20) == 67
    assert rm_jax.block_size(10, 512 * 2**20) == rm_jax.MAX_BLOCK
    assert rm_jax.block_size(10**9, 1) == 1


def test_rm_jax_blocked_matches_numpy_multi_group():
    """Blocked-scan JAX reconstruction agrees with the dense numpy
    path to round-off on every channel, across groups with 0/1/2
    fission channels and different L."""
    data = _three_group_data()
    e = np.linspace(1e-3, 1100.0, NE)
    ref = rm.reconstruct(data, e, array_ns.get_backend('numpy'))
    got = rm.reconstruct(data, e, array_ns.get_backend('jax'),
                         _max_intermediate_bytes=MAX_BYTES)
    # Absolute floor scaled to each channel's peak: ``cap = 1 - Σ|U|²``
    # cancels at capture minima (down to ~1e-5 of the peak here), where
    # numpy and numba already differ by ~1e-10 relative.
    for key in ('sct', 'cap', 'fis', 'pot', 'tot'):
        np.testing.assert_allclose(
            np.asarray(got[key]), ref[key],
            rtol=1e-12, atol=1e-14 * np.max(np.abs(ref[key])),
            err_msg=f'blocked jax vs numpy disagree on {key}',
        )
    # Not vacuous: all three channels carry resonant structure.
    assert np.max(ref['fis']) > 1.0 and np.max(ref['cap']) > 1.0


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


def test_rm_jax_has_no_dense_energy_by_resonance_intermediate():
    """The traced JAX reconstruction never materialises an array as
    large as ``(NE, NRES)``: the R-matrix is accumulated block by
    block. The dense build this replaced held a ``(NE, NRES)``
    complex inverse denominator per group (51 GB at 1 M energies on
    U-235)."""
    data = _three_group_data()
    xp = array_ns.get_backend('jax')

    def f(e):
        return rm.reconstruct(data, e, xp,
                              _max_intermediate_bytes=MAX_BYTES)['tot']

    jaxpr = jax.make_jaxpr(f)(jnp.linspace(1e-3, 1100.0, NE))
    largest = max(int(np.prod(a.shape)) for a in _iter_avals(jaxpr.jaxpr)
                  if hasattr(a, 'shape'))
    assert largest < NE * NRES // 4, largest


def test_rm_jax_blocked_grad_matches_forward_mode_and_fd():
    """Reverse-mode ``jax.grad`` through the checkpointed blocked scan
    wrt every per-resonance leaf matches forward-mode ``jax.jvp``
    along a random direction (all entries, including the padded last
    block) and a central finite difference on one capture width.

    Fission widths are left out: ``_signed_sqrt`` has an infinite
    derivative at the zero widths of resonances in groups without
    that channel (NaN gradient, same on the dense path)."""
    import dataclasses
    base = _three_group_data()
    xp = array_ns.get_backend('jax')
    e = jnp.linspace(1e-3, 1100.0, NE)
    keys = ('res_er', 'res_gn', 'res_gg')

    def loss(*leaves):
        data = dataclasses.replace(base, **dict(zip(keys, leaves)))
        out = rm.reconstruct(data, e, xp, _max_intermediate_bytes=MAX_BYTES)
        return jnp.sum(out['cap'] + out['fis'] + out['sct'])

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
    h = 1e-2 * float(args[2][i])
    fd = (float(loss(*args[:2], args[2].at[i].add(h)))
          - float(loss(*args[:2], args[2].at[i].add(-h)))) / (2 * h)
    assert abs(float(grads[2][i]) - fd) <= 1e-3 * abs(fd), (grads[2][i], fd)
