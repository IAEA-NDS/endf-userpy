"""Tests that no `Conversion of an array with ndim > 0 to a scalar
is deprecated` `DeprecationWarning` fires from any user-facing API
on the corpus (issue #83).

The pre-fix Fortran callsite in
`mf6_interpretation_integrals.get_energydist_from_subsec_law1` /
`get_energydist_from_subsec_law1_dynamic_mesh` passed
`cur_eu = np.array([eu[i]], order='F')` where `feep_points_law1con`
/ `feep_full_law1con` declare the first argument as a scalar
double. f2py's `double_from_pyobj` chain calls `.item()` on any
ndim>0 input, which emits the warning on NumPy >= 1.25 and becomes
a hard `TypeError` in a future NumPy release. The audit hit the
warning 2790 times across 15 corpus files -- silent-but-real
forward-compatibility debt.

Fix: `cur_eu = float(eu[i])` at both callsites. These tests promote
the DeprecationWarning to an error and exercise every public
API path that reaches the affected Fortran integrator on a broad
corpus slice.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_xs,
    get_particle_production_dxs_dE,
    get_particle_production_dxs_dmu,
    get_particle_production_ddxs,
)


ADHOC = Path(__file__).resolve().parent / 'data_law1_adhoc'

DEPRECATION_MSG = 'Conversion of an array with ndim.*deprecated'


def _load(fn_name):
    fn = ADHOC / fn_name
    if not fn.exists():
        pytest.skip(
            f'{fn_name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh`'
        )
    return EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    ).parsefile(fn)


@pytest.fixture(scope='module')
def fe56():
    return _load('tendl21_n_Fe-56.endf')


@pytest.fixture(scope='module')
def cu63():
    return _load('jeff40_n_Cu-63.endf')


@pytest.fixture(scope='module')
def al27():
    return _load('endfb81_n_Al-27.endf')


@pytest.fixture(scope='module')
def be9():
    return _load('endfb81_n_Be-9.endf')


# ============================================================
# Reproducers: paths that used to emit hundreds of warnings each.
# ============================================================


def _no_deprec_call(call, *args, **kwargs):
    """Run `call(*args, **kwargs)` and error out if any
    `Conversion of an array with ndim > 0 to a scalar` deprecation
    fires."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        r = call(*args, **kwargs)
    hits = [
        w for w in recorded
        if issubclass(w.category, DeprecationWarning)
        and 'ndim > 0 to a scalar' in str(w.message)
    ]
    assert not hits, (
        f'{call.__name__} emitted {len(hits)} numpy-scalar '
        f'DeprecationWarning(s); first: {str(hits[0].message)[:200]}'
    )
    return r


@pytest.mark.parametrize('fixture_name', ['fe56', 'cu63', 'al27', 'be9'])
def test_dxs_dE_no_scalar_deprecation_unbroadened(request, fixture_name):
    """Unbroadened dxs/dE via LAW=1 dispatcher: the biggest source
    of the pre-fix warning volume. Uses `(n,2n)` to route through
    LAW=1 without triggering the elastic MF4 conversion path."""
    endf = request.getfixturevalue(fixture_name)
    einc = np.linspace(1e5, 1.4e7, 6)
    eouts = np.linspace(1e5, 1.4e7, 20)
    _no_deprec_call(
        get_particle_production_dxs_dE,
        endf, '(n,2n)', 'n', einc, eouts,
    )


@pytest.mark.parametrize('fixture_name', ['fe56', 'cu63', 'al27', 'be9'])
def test_dxs_dE_no_scalar_deprecation_broadened(request, fixture_name):
    endf = request.getfixturevalue(fixture_name)
    einc = np.linspace(1e5, 1.4e7, 6)
    eouts = np.linspace(1e5, 1.4e7, 20)
    _no_deprec_call(
        get_particle_production_dxs_dE,
        endf, '(n,2n)', 'n', einc, eouts, broadening=3e4,
    )


@pytest.mark.parametrize('fixture_name', ['fe56', 'cu63', 'al27', 'be9'])
def test_ddx_broadened_no_scalar_deprecation(request, fixture_name):
    endf = request.getfixturevalue(fixture_name)
    einc = np.linspace(1e5, 1.4e7, 6)
    eouts = np.linspace(1e5, 1.4e7, 20)
    mus = np.linspace(-0.9, 0.9, 5)
    _no_deprec_call(
        get_particle_production_ddxs,
        endf, '(n,2n)', 'n', einc, eouts, mus, broadening=3e4,
    )


@pytest.mark.parametrize('fixture_name', ['fe56', 'cu63', 'al27', 'be9'])
def test_particle_production_xs_no_scalar_deprecation(
    request, fixture_name,
):
    """XS-only path doesn't reach the LAW=1 integrator, but this is
    a broader sanity guard."""
    endf = request.getfixturevalue(fixture_name)
    einc = np.linspace(1e5, 1.4e7, 6)
    _no_deprec_call(
        get_particle_production_xs, endf, '(n,total)', 'n', einc,
    )
    _no_deprec_call(
        get_particle_production_dxs_dmu, endf, '(n,total)', 'n',
        einc, np.linspace(-0.9, 0.9, 5),
    )
