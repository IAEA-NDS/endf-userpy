"""JAX R-matrix path of the LRF=7 KRM=3 reconstruction: on the
``'jax'`` backend ``mf2_interpretation_rml.reconstruct`` builds every
J-group's R-matrix with one blocked resonance scan
(:func:`mf2_interpretation_reichmoore_jax.accumulate_r_matrix`)
instead of a dense ``(ne, nres)`` inverse denominator per group.

Runs on the ENDF/B-VIII.1 Pu-239 corpus file (2 J-groups with
NCH in {3, 4}, 2085 resonances); skips if the corpus is absent.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf2_interpretation_rml as rml
from endf_userpy.mfsec_interpretation.mf2_interpretation_rml_preproc import (
    rml_data_from_endf_dict,
)
from endf_userpy.primitives import array_ns

from _corpus import resolve_pu239_rml

jax = pytest.importorskip('jax')
import jax.numpy as jnp  # noqa: E402


NE = 400
KEYS = ('sct', 'cap', 'fis', 'pot', 'tot')


@pytest.fixture(scope='module')
def pu239():
    path = resolve_pu239_rml()
    if path is None:
        pytest.skip('LRF=7 corpus file not present (see fetch.sh)')
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    return (rml_data_from_endf_dict(d, xp=array_ns.get_backend('numpy')),
            rml_data_from_endf_dict(d, xp=array_ns.get_backend('jax')))


def test_rml_jax_scan_matches_numpy_pu239(pu239):
    """Blocked-scan JAX reconstruction agrees with the dense numpy
    path to round-off across the resolved range."""
    data_np, data_jx = pu239
    e = np.geomspace(1e-3, 4000.0, NE)
    ref = rml.reconstruct(data_np, e, array_ns.get_backend('numpy'))
    got = rml.reconstruct(data_jx, e, array_ns.get_backend('jax'))
    # Absolute floor scaled to each channel's peak: capture is
    # ``1 - Σ|U|²`` and cancels at its minima.
    for key in KEYS:
        np.testing.assert_allclose(
            np.asarray(got[key]), ref[key],
            rtol=1e-11, atol=1e-13 * np.max(np.abs(ref[key])),
            err_msg=f'jax scan vs numpy disagree on {key}',
        )
    assert np.max(ref['fis']) > 100.0


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


def test_rml_jax_has_no_dense_energy_by_resonance_intermediate(pu239):
    """The traced JAX reconstruction never materialises an array as
    large as ``(ne, nres)``."""
    _, data_jx = pu239
    xp = array_ns.get_backend('jax')
    nres = int(data_jx.res_er.shape[0])

    def f(e):
        return rml.reconstruct(data_jx, e, xp)['tot']

    jaxpr = jax.make_jaxpr(f)(jnp.geomspace(1e-3, 4000.0, NE))
    largest = max(int(np.prod(a.shape)) for a in _iter_avals(jaxpr.jaxpr)
                  if hasattr(a, 'shape'))
    assert largest < NE * nres // 4, largest


def test_rml_jax_grad_matches_forward_mode_and_fd(pu239):
    """Reverse-mode ``jax.grad`` through the checkpointed scan wrt the
    resonance energies matches forward-mode ``jax.jvp`` along a random
    direction; wrt a non-zero channel width in each J-group it matches
    a central finite difference. (Forward mode through the full
    ``res_gam`` is NaN when the file has a genuine zero width of a
    group's own particle channel: the amplitude's derivative is
    infinite there; see the zero-width convention test below.)"""
    _, data_jx = pu239
    xp = array_ns.get_backend('jax')
    e = jnp.geomspace(1e-3, 4000.0, NE)

    def loss_er(er):
        out = rml.reconstruct(dataclasses.replace(data_jx, res_er=er), e, xp)
        return jnp.sum(out['tot'] + out['fis'])

    er = jnp.asarray(data_jx.res_er)
    v = jnp.asarray(np.random.default_rng(1).normal(size=er.shape))
    rev = float(jnp.dot(jax.grad(loss_er)(er), v))
    _, fwd = jax.jvp(loss_er, (er,), (v,))
    assert abs(float(fwd)) > 0.0
    assert abs(rev - float(fwd)) <= 1e-9 * abs(float(fwd)), (rev, fwd)

    gam0 = jnp.asarray(data_jx.res_gam)

    def loss_gam(x, r, c):
        d2 = dataclasses.replace(data_jx, res_gam=gam0.at[r, c].set(x))
        out = rml.reconstruct(d2, e, xp)
        return jnp.sum(out['tot'] + out['fis'])

    groups = np.asarray(data_jx.res_group)
    for g in np.unique(groups):
        r = int(np.flatnonzero(groups == g)[-1])
        c = int(np.flatnonzero(np.asarray(gam0[r]))[-1])
        x0 = float(gam0[r, c])
        ad = float(jax.grad(loss_gam)(jnp.asarray(x0), r, c))
        h = 1e-3 * abs(x0)
        fd = (float(loss_gam(jnp.asarray(x0 + h), r, c))
              - float(loss_gam(jnp.asarray(x0 - h), r, c))) / (2 * h)
        assert abs(fd) > 0.0
        assert abs(ad - fd) <= 1e-4 * abs(fd), (g, r, c, ad, fd)


def test_rml_grad_wrt_res_gam_zero_only_where_width_cannot_matter(pu239):
    """Zero-width gradient convention: every zero ``res_gam`` entry that
    cannot affect the output (padded channel slot, or a channel of
    another J-group) has gradient exactly 0 instead of NaN; only
    genuine zero widths of a group's own particle channels keep the
    amplitude's infinite derivative."""
    _, data_jx = pu239
    xp = array_ns.get_backend('jax')
    e = jnp.geomspace(1e-3, 4000.0, NE)
    gam0 = np.asarray(data_jx.res_gam)

    def loss(gam):
        out = rml.reconstruct(dataclasses.replace(data_jx, res_gam=gam), e, xp)
        return jnp.sum(out['tot'] + out['fis'])

    g = np.asarray(jax.grad(loss)(jnp.asarray(gam0)))
    groups = np.asarray(data_jx.res_group)
    real_zero = np.zeros(gam0.shape, dtype=bool)
    for gi in np.unique(groups):
        particle = rml._classify_group_channels(data_jx, int(gi))[1]
        rows = groups == gi
        for c in particle:
            real_zero[rows, c] = gam0[rows, c] == 0.0
    unused_zero = (gam0 == 0.0) & ~real_zero
    assert unused_zero.any()
    np.testing.assert_array_equal(g[unused_zero], 0.0)
    assert np.all(np.isfinite(g[~real_zero]))
