"""jax.jit / jax.grad follow-ups of issue #220 on the committed
``tests/data`` files (run without the fetch-on-demand corpus):

- dxs/dmu with traced angle cosines (the E' integrators segment on
  mu; traced mu was refused with NotImplementedError);
- MF6 LAW=7 dxs/dE under jit over the energies with a log-in-y outer
  Ein law (INT=4 / 5), which is not linear in the slice amplitudes.
  No corpus file carries it, so the Be-9 (n,2n) LAW=7 section is
  edited in memory.
"""
import copy
import os
import warnings

import numpy as np
import pytest

from endf_userpy.quantities import (
    get_particle_production_dxs_dE,
    get_particle_production_dxs_dmu,
)
from endf_userpy.quantities_mt_zap import quantities as qz
from endf_userpy.run_options import RunOptions

jax = pytest.importorskip('jax')
jnp = jax.numpy

_DATA = os.path.join(os.path.dirname(__file__), 'data')


def _parse(name):
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp(ignore_missing_tpid=True, ignore_zero_mismatch=True,
                         accept_spaces=True).parsefile(os.path.join(_DATA, name))


@pytest.fixture(scope='module')
def be9():
    d = _parse('n-004_Be_009.endf')
    assert d[6][16]['subsection'][1]['LAW'] == 7
    return d


def _assert_matches(out, ref):
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
    np.testing.assert_array_equal(out == 0.0, ref == 0.0)
    fin = np.isfinite(ref)
    assert np.count_nonzero(ref[fin]) > 10
    np.testing.assert_allclose(out[fin], ref[fin], rtol=1e-9, atol=0.0)


@pytest.mark.parametrize('name', ['n-004_Be_009.endf', 'n-001_H_002.endf'])
@pytest.mark.parametrize('traced', ['mu', 'both'])
def test_dxs_dmu_under_jit_over_angle_cosines(name, traced):
    """Be-9: MF6 LAW=7 (knot-aware E' integrator); H-2: LAW=6
    (adaptive Simpson over E')."""
    d = _parse(name)
    ein = np.geomspace(1.3e5, 1.9e7, 10)
    mu = np.linspace(-0.95, 0.97, 7)
    opts = RunOptions(backend='jax')

    def call(e, m, o):
        return get_particle_production_dxs_dmu(d, '(n,total)', 'n', e, m,
                                               options=o)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = np.asarray(call(ein, mu, RunOptions(backend='numpy')))
        if traced == 'mu':
            out = jax.jit(lambda m: call(ein, m, opts))(jnp.asarray(mu))
        else:
            out = jax.jit(lambda e, m: call(e, m, opts))(
                jnp.asarray(ein), jnp.asarray(mu))
    _assert_matches(np.asarray(out), ref)


def test_dxs_dmu_grad_wrt_mu_law7(be9):
    """jax.grad wrt the outgoing cosine through the LAW=7 E' integral
    agrees with a central difference."""
    ein = np.array([1.2e7])
    opts = RunOptions(backend='jax')

    def loss(m):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return get_particle_production_dxs_dmu(
                be9, '(n,2n)', 'n', ein, m.reshape(1), options=opts).sum()
    for m0 in (-0.43, 0.27):            # between the 0.1-spaced mu knots
        g = float(jax.grad(loss)(jnp.asarray(m0)))
        h = 1e-4
        fd = float((loss(jnp.asarray(m0 + h)) - loss(jnp.asarray(m0 - h))) / (2 * h))
        assert np.isfinite(g) and g != 0.0
        np.testing.assert_allclose(g, fd, rtol=1e-4)


@pytest.mark.parametrize('code', [3, 4, 5])
def test_law7_dxs_de_under_jit_with_outer_int(be9, code):
    d = copy.deepcopy(be9)
    sub = d[6][16]['subsection'][1]
    sub['E_interpol']['INT'] = [code] * len(sub['E_interpol']['INT'])
    ein = np.geomspace(2e6, 1.9e7, 10)
    eout = np.geomspace(1e4, 1e7, 12)
    opts = RunOptions(backend='jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = np.asarray(qz.compute_dexs(d, 16, 1, ein, eout,
                                         options=RunOptions(backend='numpy')))
        out = np.asarray(jax.jit(lambda e: qz.compute_dexs(
            d, 16, 1, e, eout, options=opts))(jnp.asarray(ein)))
    _assert_matches(out, ref)


def test_law7_outer_int_changes_the_spectrum(be9):
    """Guard against a vacuous INT test: the log-y law (INT=4) gives a
    different spectrum than lin-lin, also at the top-level API."""
    d4 = copy.deepcopy(be9)
    sub = d4[6][16]['subsection'][1]
    sub['E_interpol']['INT'] = [4] * len(sub['E_interpol']['INT'])
    ein = np.geomspace(2e6, 1.9e7, 6)
    eout = np.geomspace(1e5, 5e6, 6)
    opts = RunOptions(backend='numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lin = np.asarray(get_particle_production_dxs_dE(
            be9, '(n,2n)', 'n', ein, eout, options=opts))
        log = np.asarray(get_particle_production_dxs_dE(
            d4, '(n,2n)', 'n', ein, eout, options=opts))
    assert np.nanmax(np.abs(log - lin) / np.maximum(np.abs(lin), 1e-300)) > 1e-3
