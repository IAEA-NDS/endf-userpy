"""MF6 LAW=1 energy spectrum on the traced (jax.jit) path when the
panels list discrete lines before the continuum.

ENDF-6 LAW=1 panels store their ND discrete energies first -- in
JEFF-4.0 Cu-63 (n,g) 183 photon lines in descending order -- and the
continuum (ascending) after them. The traced spectrum kernel
(`mf6_law1_multipanel_traced`) bracketed E' with `searchsorted` over
the whole unsorted row, so under jax.jit over the incident energies
part of the continuum came out zero: `get_particle_production_dxs_dE`
for photons was silently wrong (up to 100 % per point, -7 % in total).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_particle_production_dxs_dE
from endf_userpy.run_options import RunOptions

from _corpus import resolve_cu63_rml

jax = pytest.importorskip('jax')
import jax.numpy as jnp  # noqa: E402


@pytest.fixture(scope='module')
def cu63():
    path = resolve_cu63_rml()          # JEFF-4.0 Cu-63
    if path is None:
        pytest.skip('Cu-63 corpus file not present (see fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True, ignore_zero_mismatch=True,
                         accept_spaces=True).parsefile(path)


def test_cu63_photon_panels_store_discrete_lines_unsorted(cu63):
    sub = cu63[6][102]['subsection'][1]
    assert sub['LAW'] == 1 and max(sub['ND'].values()) > 1


def test_photon_dxs_dE_under_jit_matches_numpy(cu63):
    ein = np.geomspace(1e5, 1.9e7, 12)
    eout = np.geomspace(1e4, 1e7, 12)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = np.asarray(get_particle_production_dxs_dE(
            cu63, '(n,total)', 'g', ein, eout,
            options=RunOptions(backend='numpy')))
        opts = RunOptions(backend='jax')
        got = np.asarray(jax.jit(lambda e: get_particle_production_dxs_dE(
            cu63, '(n,total)', 'g', e, eout, options=opts))(jnp.asarray(ein)))
    assert np.count_nonzero(ref) > 100
    np.testing.assert_array_equal(got == 0.0, ref == 0.0)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=0.0)
