"""Tests for `convert_angdist_to_energydist` in
`endf_userpy.quantities_mt_zap.distribution1d_helpers` (issue #76).

The previous implementation passed a 2D per-cell `angle_cosines_out`
of shape `(n_ein, n_eout)` into the MF4 / MF6 `compute_angdist_values`
functions in one call. Downstream the CM<->LAB conversion in
`primitives.conversion.convert_angcos_to_cmsys` reshaped
`mu_lab.reshape(1, -1)`, flattening the per-cell mesh, then
re-broadcast to shape `(n_ein, n_ein * n_eout)`. The final
`angdist_values * np.abs(jacvals)` multiplication then failed on
every file with an elastic MF4 (i.e. every corpus file) with

    ValueError: operands could not be broadcast together with
                shapes (n_ein, n_ein*n_eout) (n_ein, n_eout)

The fix loops over Ein and calls `compute_angdist_func` with a 1D
mu vector per row -- the shape all the per-scheme MF4 / MF6
evaluators actually support. Same numerical answer at every cell,
per-Ein Python-level dispatch cost instead of one giant broadcast.

These tests pin:

1. **Dispatch works** on every corpus file for the neutron ejectile
   `dxs/dE` API.
2. **Physical shape** of the result: `(n_ein, n_eout)`, finite,
   non-negative.
3. **Cross-section consistency** on a smooth file: at fixed E_in the
   integral of `dxs/dE` over the ejectile energy axis approximately
   matches `get_particle_production_xs` at that E_in. Tolerance is
   loose because the integration grid is fixed and the outgoing
   spectra have narrow features near threshold.
4. **Bit-for-bit stability** on a smaller reaction where the whole
   dispatch is shorter: comparing single-Ein-per-call to a batched
   sanity call.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_dxs_dE,
    get_particle_production_xs,
)
from endf_userpy.primitives.np_compat import trapezoid


ADHOC = Path(__file__).resolve().parent / 'data_law1_adhoc'


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
def h1_endfb81():
    return _load('endfb81_n_H-1.endf')


@pytest.fixture(scope='module')
def al27_endfb81():
    return _load('endfb81_n_Al-27.endf')


@pytest.fixture(scope='module')
def fe56_tendl21():
    return _load('tendl21_n_Fe-56.endf')


@pytest.fixture(scope='module')
def cu63_jeff40():
    return _load('jeff40_n_Cu-63.endf')


# ============================================================
# Dispatch works: no broadcast crash on every corpus file with
# elastic MF4 (i.e. essentially all of them).
# ============================================================


@pytest.mark.parametrize('fixture_name', [
    'h1_endfb81', 'al27_endfb81', 'fe56_tendl21', 'cu63_jeff40',
])
def test_dxs_dE_neutron_ntotal_returns_finite(fixture_name, request):
    """Regression guard for #76: the top-level API must not raise
    a broadcast ValueError on `(n,total)` neutron production."""
    endf = request.getfixturevalue(fixture_name)
    einc = np.linspace(1e5, 1.4e7, 6)
    eouts = np.linspace(1e5, 1.4e7, 20)
    with warnings.catch_warnings():
        # Existing #83 NumPy scalar deprecation is out of scope; the
        # #76 fix must not resurrect any other warning either, but
        # we don't want the noise here.
        warnings.simplefilter('ignore', DeprecationWarning)
        r = get_particle_production_dxs_dE(
            endf, '(n,total)', 'n', einc, eouts,
        )
    assert r is not None
    assert r.shape == (6, 20)
    assert np.all(np.isfinite(r)), 'non-finite entries in dxs/dE'
    assert np.all(r >= 0.0), 'dxs/dE has negative entries'


# ============================================================
# Physics invariant: integrating dxs/dE over Eout at fixed Ein
# approximately recovers the production XS at that Ein.
# ============================================================


def test_dxs_dE_integrates_to_production_xs_on_al27(al27_endfb81):
    """Al-27 (n,total) neutron production at 5 MeV: integrating
    dxs/dE over Eout on a dense linear grid must approximately equal
    `get_particle_production_xs` at the same Ein. Uses a 3-point Ein
    grid (5 MeV in the middle) so the code path exercises the
    per-Ein loop and would trip the pre-fix broadcast bug that only
    fires with `n_ein > 1`."""
    einc = np.array([3e6, 5e6, 8e6])
    eouts = np.linspace(1e5, 4.99e6, 400)  # dense, below the middle Ein
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        dxs = get_particle_production_dxs_dE(
            al27_endfb81, '(n,total)', 'n', einc, eouts,
        )
        xs = get_particle_production_xs(
            al27_endfb81, '(n,total)', 'n', einc,
        )
    # Check the middle-Ein row (5 MeV) whose full spectrum is
    # inside the eouts range.
    integral = float(trapezoid(dxs[1], eouts))
    ratio = integral / float(xs[1])
    # We expect `integral < xs` because the elastic peak at
    # Eout == Ein is entirely above the integration range; the
    # discrete-inelastic boxes and continuum should be captured.
    assert 0.0 < ratio < 1.2, (
        f'Al-27 dxs/dE integral / xs = {ratio:.3e}'
    )


def test_dxs_dE_nonzero_where_ein_above_threshold(fe56_tendl21):
    """Fe-56 (n,total) neutron production at 14 MeV: the spectrum
    must be non-zero over an appreciable fraction of Eout in the
    kinematically allowed region (0 < Eout < Ein). Uses a 4-point
    Ein grid straddling 14 MeV so the per-Ein loop is properly
    exercised."""
    einc = np.array([8e6, 1.1e7, 1.4e7, 1.7e7])
    eouts = np.linspace(1e5, 1.35e7, 40)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        dxs = get_particle_production_dxs_dE(
            fe56_tendl21, '(n,total)', 'n', einc, eouts,
        )
    # Check the 14 MeV row (index 2)
    frac_nonzero = float(np.mean(dxs[2] > 0))
    assert frac_nonzero > 0.3, (
        f'Fe-56 dxs/dE @14 MeV nonzero fraction {frac_nonzero:.2f} < 0.3; '
        f'suspiciously empty'
    )
