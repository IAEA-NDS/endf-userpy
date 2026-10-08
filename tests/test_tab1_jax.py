"""JAX TAB1 fast path (:mod:`endf_userpy.primitives.tab1_jax`): with a
concrete query mesh and concrete abscissae, ``tab1.interp`` on jax does
the panel lookup on numpy and the arithmetic in one jitted kernel.
Results must equal the generic traceable path; traced queries must
still take the generic path; gradients wrt traced ordinates must flow.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns, tab1

jax = pytest.importorskip('jax')
import jax.numpy as jnp  # noqa: E402

from endf_userpy.primitives import tab1_jax  # noqa: E402


def _table(laws, n=14, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(1e-2, 1e3, n))
    x[n // 2] = x[n // 2 - 1]
    y = rng.uniform(0.1, 5.0, n)
    y[3] = y[2]
    nbt = np.linspace(0, n - 1, len(laws) + 1).astype(np.int32)[1:]
    return tab1.TAB1(x=x, y=y, nbt=nbt, intp=np.asarray(laws, np.int32))


XP = array_ns.get_backend('jax')
Q = np.concatenate([np.geomspace(1e-3, 2e3, 700), _table((2,)).x])


@pytest.mark.parametrize('laws', [(2,), (2, 5), (1, 2, 3, 4, 5)])
@pytest.mark.parametrize('side', ['right', 'left'])
@pytest.mark.parametrize('outside', [0.0, float('nan')])
def test_host_path_matches_generic_path(laws, side, outside, monkeypatch):
    t = _table(laws)
    got = np.asarray(tab1.interp(t, Q, XP, outside_value=outside, side=side))
    monkeypatch.setattr(tab1, '_host_arrays', lambda x, t: None)
    ref = np.asarray(tab1.interp(t, Q, XP, outside_value=outside, side=side))
    np.testing.assert_array_equal(got, ref)


def test_host_path_used_and_compiled_once_per_mesh(monkeypatch):
    """Concrete inputs route through the jitted kernel; repeated calls
    with a NaN fill reuse the compiled kernel (a static NaN argument
    would miss the cache on every call since NaN != NaN)."""
    calls = []
    orig = tab1_jax.interp_from_lookup
    monkeypatch.setattr(tab1_jax, 'interp_from_lookup',
                        lambda *a, **k: calls.append(1) or orig(*a, **k))
    t = _table((2,))
    q = np.linspace(0.5, 900.0, 333)
    tab1.interp(t, q, XP, outside_value=float('nan'))
    n0 = orig._cache_size()
    for seed in (1, 2, 3):
        tab1.interp(_table((2,), seed=seed), q, XP, outside_value=float('nan'))
    assert len(calls) == 4
    assert orig._cache_size() == n0


def test_traced_query_takes_generic_path(monkeypatch):
    t = _table((2, 5))
    calls = []
    monkeypatch.setattr(tab1_jax, 'interp_from_lookup',
                        lambda *a, **k: calls.append(1))
    out = jax.jit(lambda q: tab1.interp(t, q, XP))(jnp.asarray(Q))
    assert calls == []
    monkeypatch.undo()
    np.testing.assert_allclose(np.asarray(out),
                               np.asarray(tab1.interp(t, Q, XP)),
                               rtol=1e-15, atol=0.0)


@pytest.mark.parametrize('laws', [(2,), (2, 5)])
def test_grad_wrt_traced_ordinates_matches_generic(laws, monkeypatch):
    t = _table(laws)

    def loss(y):
        tt = tab1.TAB1(x=t.x, y=y, nbt=t.nbt, intp=t.intp)
        return jnp.sum(tab1.interp(tt, Q, XP) ** 2)

    g = jax.grad(loss)(jnp.asarray(t.y))
    monkeypatch.setattr(tab1, '_host_arrays', lambda x, t: None)
    g_ref = jax.grad(loss)(jnp.asarray(t.y))
    assert np.all(np.isfinite(np.asarray(g)))
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref),
                               rtol=1e-13, atol=0.0)
