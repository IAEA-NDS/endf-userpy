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


# ------------------------------------------------------------------
# interpolation._endf_interp1d_traced_x (MF6 yields etc.)
# ------------------------------------------------------------------

from endf_userpy.primitives import interpolation  # noqa: E402


def _traced(fn, q):
    """Force the generic traced-x path by tracing the query."""
    return np.asarray(jax.jit(fn)(jnp.asarray(q)))


@pytest.mark.parametrize('laws', [(2,), (2, 5), (1, 2, 3, 4, 5)])
@pytest.mark.parametrize('outside', [0.0, None])
@pytest.mark.parametrize('batched', [False, True])
def test_traced_x_host_path_matches_traced_path(laws, outside, batched,
                                                monkeypatch):
    t = _table(laws)
    fp = np.stack([t.y, 2.0 * t.y + 0.1]) if batched else t.y
    q = Q if outside is not None else np.clip(Q, t.x[0], t.x[-1])
    calls = []
    orig = tab1_jax.traced_x_from_lookup
    monkeypatch.setattr(tab1_jax, 'traced_x_from_lookup',
                        lambda *a, **k: calls.append(1) or orig(*a, **k))

    def f(qq):
        return interpolation._endf_interp1d_traced_x(
            qq, t.x, fp, t.intp, t.nbt + 1, outside, XP)

    got = np.asarray(f(q))
    assert calls == [1]
    ref = _traced(f, q)
    assert len(calls) == 1          # traced query: generic path
    np.testing.assert_allclose(got, ref, rtol=1e-15, atol=0.0)
    assert got.shape == ((2, q.size) if batched else (q.size,))


def test_traced_x_host_path_grad_wrt_ordinates():
    t = _table((2, 5))

    def loss(fp, use_host):
        q = jnp.asarray(Q) if not use_host else Q
        return jnp.sum(interpolation._endf_interp1d_traced_x(
            q, t.x, fp, t.intp, t.nbt + 1, 0.0, XP) ** 2)

    g = jax.grad(loss)(jnp.asarray(t.y), True)
    g_ref = jax.jit(jax.grad(loss), static_argnums=1)(jnp.asarray(t.y), False)
    assert np.all(np.isfinite(np.asarray(g)))
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref),
                               rtol=1e-13, atol=0.0)
