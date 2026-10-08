"""Interpolation-law dispatch in the backend-agnostic TAB1 paths
(``primitives.tab1.interp`` / ``_apply_law_vectorised`` and
``primitives.interpolation._endf_interp1d_traced_x``): only the laws
a table can select are evaluated, with results identical to the full
five-law dispatch.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns, interpolation, tab1


def _table(laws, n=12, seed=0):
    """Positive-y table with a doubled x (step), a flat panel and one
    region per entry of ``laws``."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(1e-2, 1e3, n))
    x[n // 2] = x[n // 2 - 1]
    y = rng.uniform(0.1, 5.0, n)
    y[3] = y[2]
    nr = len(laws)
    nbt = np.linspace(0, n - 1, nr + 1).astype(np.int32)[1:]
    return tab1.TAB1(x=x, y=y, nbt=nbt, intp=np.asarray(laws, np.int32))


LAW_SETS = [(1,), (2,), (3,), (4,), (5,), (2, 5), (1, 2, 3, 4, 5), (12, 2)]


def test_static_laws():
    assert tab1._static_laws(np.array([2, 2])) == (2,)
    assert tab1._static_laws(np.array([12, 5])) == (2, 5)   # INT % 10
    assert tab1._static_laws(np.array([0])) == tab1._ALL_LAWS


def _backends():
    return [b for b in ('numpy', 'jax') if b in array_ns.available_backends()]


@pytest.mark.parametrize('backend', _backends())
@pytest.mark.parametrize('laws', LAW_SETS)
@pytest.mark.parametrize('side', ['right', 'left'])
def test_tab1_reduced_dispatch_matches_full(backend, laws, side, monkeypatch):
    xp = array_ns.get_backend(backend)
    t = _table(laws)
    q = np.concatenate([np.geomspace(1e-3, 2e3, 500), t.x])
    got = np.asarray(tab1.interp(t, q, xp, side=side))
    monkeypatch.setattr(tab1, '_static_laws', lambda intp: tab1._ALL_LAWS)
    ref = np.asarray(tab1.interp(t, q, xp, side=side))
    np.testing.assert_array_equal(got, ref)


def _traced_x_full_dispatch(x, xp_mesh, fp, int_arr, nbt_arr, outside_value, xp):
    """Oracle: the pre-change body of ``_endf_interp1d_traced_x``
    (all five laws evaluated, nested select), verbatim."""
    _small = 1.0e-38
    x = xp.asarray(x)
    fp = xp.asarray(fp)
    xp_mesh_xp = xp.asarray(xp_mesh)
    n_mesh = int(xp_mesh_xp.shape[0])
    ipm = xp.asarray(interpolation.convert_interp_repr(
        np.asarray(int_arr), np.asarray(nbt_arr)))
    idx = xp.clip(xp.searchsorted(xp_mesh_xp, x, side='right') - 1, 0, n_mesh - 2)
    x1 = xp.take(xp_mesh_xp, idx)
    x2 = xp.take(xp_mesh_xp, idx + 1)
    y1 = xp.take(fp, idx, axis=-1)
    y2 = xp.take(fp, idx + 1, axis=-1)
    dx = x2 - x1
    dx_safe = xp.where(dx == 0.0, 1.0, dx)
    x1_pos = xp.where(x1 > 0.0, x1, _small)
    x2_pos = xp.where(x2 > 0.0, x2, _small)
    y1_pos = xp.where(y1 > 0.0, y1, _small)
    y2_pos = xp.where(y2 > 0.0, y2, _small)
    x_pos = xp.where(x > 0.0, x, _small)
    r1 = y1
    r2 = y1 + (x - x1) * (y2 - y1) / dx_safe
    log_x_ratio = xp.log(x_pos / x1_pos)
    log_x2_ratio = xp.log(x2_pos / x1_pos)
    log_x2_ratio_safe = xp.where(log_x2_ratio == 0.0, 1.0, log_x2_ratio)
    r3 = y1 + log_x_ratio * (y2 - y1) / log_x2_ratio_safe
    log_y_ratio = xp.log(y2_pos / y1_pos)
    r4 = y1_pos * xp.exp((x - x1) * log_y_ratio / dx_safe)
    r5 = y1_pos * xp.exp(log_x_ratio * log_y_ratio / log_x2_ratio_safe)
    t = xp.take(ipm, idx + 1)
    result = xp.where(t == 1, r1, xp.where(t == 2, r2, xp.where(
        t == 3, r3, xp.where(t == 4, r4, r5))))
    is_inside = (x >= xp_mesh_xp[0]) & (x <= xp_mesh_xp[-1])
    return xp.where(is_inside, result, outside_value)


@pytest.mark.parametrize('backend', _backends())
@pytest.mark.parametrize('laws', LAW_SETS)
def test_traced_x_reduced_dispatch_matches_full(backend, laws):
    xp = array_ns.get_backend(backend)
    t = _table(laws)
    q = np.concatenate([np.geomspace(1e-3, 2e3, 500), t.x])
    got = interpolation._endf_interp1d_traced_x(
        q, t.x, t.y, t.intp % 10, t.nbt + 1, 0.0, xp)
    ref = _traced_x_full_dispatch(
        q, t.x, t.y, t.intp % 10, t.nbt + 1, 0.0, xp)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.skipif('jax' not in array_ns.available_backends(),
                    reason='jax not installed')
def test_lin_lin_tables_trace_no_transcendentals():
    """A lin-lin-only table (the common MF3 / MF6-yield case) traces no
    log / exp in either interpolator; the full dispatch traced two of
    each per point."""
    import jax
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')
    t = _table((2,))
    q = jnp.geomspace(1e-3, 2e3, 64)
    for fn in (lambda x: tab1.interp(t, x, xp),
               lambda x: interpolation._endf_interp1d_traced_x(
                   x, t.x, t.y, t.intp, t.nbt + 1, 0.0, xp)):
        prims = {e.primitive.name for e in jax.make_jaxpr(fn)(q).jaxpr.eqns}
        assert not prims & {'log', 'exp'}, prims


@pytest.mark.skipif('jax' not in array_ns.available_backends(),
                    reason='jax not installed')
@pytest.mark.parametrize('laws', [(2,), (2, 5)])
def test_reduced_dispatch_grad_matches_full(laws, monkeypatch):
    """``jax.grad`` wrt the tabulated y values and the query x is the
    same with the reduced and the full dispatch."""
    import jax
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')
    t = _table(laws)
    q = jnp.geomspace(2e-2, 9e2, 200)

    def loss(y, x):
        tt = tab1.TAB1(x=t.x, y=y, nbt=t.nbt, intp=t.intp)
        return jnp.sum(tab1.interp(tt, x, xp) ** 2)

    g = jax.grad(loss, argnums=(0, 1))(jnp.asarray(t.y), q)
    monkeypatch.setattr(tab1, '_static_laws', lambda intp: tab1._ALL_LAWS)
    g_ref = jax.grad(loss, argnums=(0, 1))(jnp.asarray(t.y), q)
    for a, b in zip(g, g_ref):
        assert np.all(np.isfinite(np.asarray(a)))
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=1e-13, atol=0.0)
