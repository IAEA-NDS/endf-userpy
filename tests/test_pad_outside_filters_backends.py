"""The ``pad_outside_*`` query filters of MF4 / MF6 zero-fill
incident energies outside a section's tabulated range. They skipped
the mask for every backend not named 'numpy' (meant for JAX tracers),
which also caught the numba backend: out-of-range energies then
reached ``endf_interp1d`` and raised (U-235 MF6 LAW=2 dxs/dmu)."""
import numpy as np
import pytest

from endf_userpy.mfsec_interpretation import mf4_interpretation_helpers as mf4h
from endf_userpy.mfsec_interpretation import mf6_interpretation_helpers as mf6h
from endf_userpy.primitives import array_ns


def _mf6_dict():
    return {6: {51: {'subsection': {1: {'E': {1: 1e6, 2: 2e6, 3: 5e6}}}}}}


def _mf4_dict():
    return {4: {51: {'E': {1: 1e6, 2: 2e6, 3: 5e6}}}}


ENERGIES = np.array([5e5, 1e6, 3e6, 5e6, 6e6])
INSIDE = np.array([False, True, True, True, False])


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_mf6_filter_masks_out_of_range_energies(backend):
    if backend == 'numba':
        pytest.importorskip('numba')
    xp = array_ns.get_backend(backend)
    mask = mf6h._filter_energies_in(ENERGIES, _mf6_dict(), 51, 1, xp=xp)
    np.testing.assert_array_equal(mask, INSIDE)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_mf4_filter_masks_out_of_range_energies(backend):
    if backend == 'numba':
        pytest.importorskip('numba')
    xp = array_ns.get_backend(backend)
    mask = mf4h._filter_energies_in(ENERGIES, _mf4_dict(), 51, xp=xp)
    np.testing.assert_array_equal(mask, INSIDE)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_mf6_eout_filter_masks_negative_energies(backend):
    if backend == 'numba':
        pytest.importorskip('numba')
    xp = array_ns.get_backend(backend)
    mask = mf6h._filter_energies_out(np.array([-1.0, 0.0, 1.0]), xp=xp)
    np.testing.assert_array_equal(mask, [False, False, True])


def test_numba_dxs_dmu_matches_numpy_below_mf6_range():
    pytest.importorskip('numba')
    """End to end: U-235 (ENDF/B-VIII.1) MF6 LAW=2 sections start above
    part of the incident-energy grid; numba raised there."""
    import os
    import sys
    import warnings
    sys.path.insert(0, os.path.dirname(__file__))
    from _corpus import resolve_u235_endfb81
    from endf_userpy.quantities import get_particle_production_dxs_dmu
    from endf_userpy.run_options import RunOptions
    path = resolve_u235_endfb81()
    if path is None:
        pytest.skip('U-235 corpus file not present (see fetch.sh)')
    from endf_parserpy import EndfParserCpp
    d = EndfParserCpp(ignore_missing_tpid=True, ignore_zero_mismatch=True,
                      accept_spaces=True).parsefile(path)
    ein = np.geomspace(1e5, 1.9e7, 12)
    mu = np.linspace(-1.0, 1.0, 5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref, out = (np.asarray(get_particle_production_dxs_dmu(
            d, '(n,total)', 'n', ein, mu, options=RunOptions(backend=b)))
            for b in ('numpy', 'numba'))
    assert np.all(ref > 0.0)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=0.0)
