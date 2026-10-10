"""MF6 LAW=7 energy spectra: the numba backend refines like numpy.

`mf6_law7_integrals` integrates the LAW=7 (mu, E') tabulation over mu
with a composite Simpson rule and adaptively refines unconverged
segments. The refinement needs concrete values, so it is skipped on
the jax backend only (documented: a few permille). The condition read
``xp.name == 'numpy'``, which also skipped it on the numba backend
(numpy arrays underneath): JEFF-4.0 H-2 (n,2n) neutron spectra on the
numba backend differed from numpy by up to 6.5e-4.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.quantities_mt_zap import quantities as qz
from endf_userpy.run_options import RunOptions

from _corpus import resolve_h2


def test_law7_spectrum_numba_backend_equals_numpy():
    if 'numba' not in array_ns.available_backends():
        pytest.skip('numba not installed')
    path = resolve_h2()
    if path is None:
        pytest.skip('H-2 corpus file not present (see fetch.sh)')
    d = EndfParserCpp(ignore_missing_tpid=True, ignore_zero_mismatch=True,
                      accept_spaces=True).parsefile(path)
    assert d[6][16]['subsection'][1]['LAW'] == 7
    ein = np.geomspace(5e6, 1.9e7, 6)
    eout = np.geomspace(1e4, 1e7, 12)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = np.asarray(qz.compute_dexs(d, 16, 1, ein, eout,
                                         options=RunOptions(backend='numpy')))
        got = np.asarray(qz.compute_dexs(d, 16, 1, ein, eout,
                                         options=RunOptions(backend='numba')))
    assert np.count_nonzero(ref) > 20
    np.testing.assert_array_equal(got, ref)
