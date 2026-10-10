"""Particle production when no reaction in the file produces the particle.

ENDF/B-VIII.1 Pu-239 has no MT carrying proton production. The four
``get_particle_production_*`` entry points used to return ``None`` for
such a query (silently); they now return zeros of their usual output
shape and emit one summary UserWarning, consistent with
``get_reaction_xs`` for an untabulated reaction.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy import quantities as Q
from endf_userpy.primitives import array_ns
from endf_userpy.run_options import RunOptions

from _corpus import resolve_pu239_rml

E = np.geomspace(1e-3, 2e7, 7)
EOUT = np.geomspace(1e3, 1e7, 4)
MU = np.linspace(-1.0, 1.0, 3)
CASES = [
    ('get_particle_production_xs', (E,), (7,)),
    ('get_particle_production_dxs_dE', (E, EOUT), (7, 4)),
    ('get_particle_production_dxs_dmu', (E, MU), (7, 3)),
    ('get_particle_production_ddxs', (E, EOUT, MU), (7, 4, 3)),
]


@pytest.fixture(scope='module')
def pu239():
    path = resolve_pu239_rml()       # ENDF/B-VIII.1 Pu-239
    if path is None:
        pytest.skip('Pu-239 corpus file not present (see fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.mark.parametrize('backend', ['numpy', 'numba', 'jax'])
@pytest.mark.parametrize('fn,args,shape', CASES)
def test_nothing_produced_gives_zeros_and_one_warning(pu239, backend, fn,
                                                      args, shape):
    if backend not in array_ns.available_backends():
        pytest.skip(f'{backend} not installed')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = getattr(Q, fn)(pu239, '(n,total)', 'p', *args,
                             options=RunOptions(backend=backend))
    assert out is not None
    out = np.asarray(out)
    assert out.shape == shape
    assert np.all(out == 0.0)
    msgs = [str(w.message) for w in caught if 'no reaction in the file produces' in str(w.message)]
    assert len(msgs) == 1 and "'p'" in msgs[0]


def test_produced_particle_has_no_such_warning(pu239):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = Q.get_particle_production_xs(pu239, '(n,total)', 'n', E,
                                           options=RunOptions(backend='numpy'))
    assert np.nanmax(out) > 0.0
    assert not any('no reaction in the file produces' in str(w.message)
                   for w in caught)


def test_nothing_produced_under_jit_over_energies(pu239):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    opts = RunOptions(backend='jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = jax.jit(lambda e: Q.get_particle_production_xs(
            pu239, '(n,total)', 'p', e, options=opts))(jnp.asarray(E))
    assert np.asarray(out).shape == (7,) and np.all(np.asarray(out) == 0.0)
