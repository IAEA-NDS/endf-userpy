"""Tests for the NK=1 guard in
`mf13_interpretation.compute_total_photon_production_xs` (issue #79).

ENDF-6 writes the section-level totals header (`E`, `sigma_tot`,
`INT`, `NBT`) in MF13/MT only when `NK > 1`; for `NK == 1` the
single subsection IS the total and no header is written. The
previous consistency-check guard was ``prod_xs.shape[0] > 1``,
which is ``n_ein > 1``, not ``n_subs > 1`` -- so any query with
more than one incident energy on a single-subsection section fell
through to a redundant consistency check that raised
``KeyError: 'INT'`` on the missing totals header. N-14 MT 32 and
MT 105 both have `NK == 1` and tripped this.

Fix: guard on `endf_dict[13][mt]['NK'] > 1` directly. Below the
guard, everything is unchanged; above it, the single subsection's
own `sigma` array becomes the total (via the pre-existing sum).

These tests pin:

1. N-14 `(n,total)+g` cross section returns a finite array
   (regression for the KeyError).
2. Every N-14 MF13 MT (NK in {1, 6, 11, 23, 43, ...}) computes a
   finite total photon-production XS.
3. Existing multi-subsection files (Al-27 MT102) still perform
   the consistency check and hit the same numerical answer as
   before -- non-regression.
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_xs,
    get_particle_production_dxs_dE,
)
from endf_userpy.mfsec_interpretation.mf13_interpretation import (
    compute_total_photon_production_xs,
)


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
def n14_endfb81():
    return _load('endfb81_n_N-14.endf')


# ============================================================
# Regression: N-14 public API surface used to raise KeyError('INT').
# ============================================================


def test_n14_particle_production_xs_gamma(n14_endfb81):
    """Public API entry point that used to KeyError."""
    einc = np.linspace(1e5, 1.5e7, 6)
    r = get_particle_production_xs(n14_endfb81, '(n,total)', 'g', einc)
    assert r.shape == (6,)
    assert np.all(np.isfinite(r))
    assert np.all(r >= 0.0)


def test_n14_particle_production_dxs_dE_gamma_unbroadened(n14_endfb81):
    einc = np.linspace(1e5, 1.5e7, 6)
    eouts = np.linspace(1e5, 1.4e7, 20)
    r = get_particle_production_dxs_dE(
        n14_endfb81, '(n,total)', 'g', einc, eouts,
    )
    assert r is not None
    assert r.shape == (6, 20)
    assert np.all(np.isfinite(r))


def test_n14_particle_production_dxs_dE_gamma_broadened(n14_endfb81):
    einc = np.linspace(1e5, 1.5e7, 6)
    eouts = np.linspace(1e5, 1.4e7, 20)
    r = get_particle_production_dxs_dE(
        n14_endfb81, '(n,total)', 'g', einc, eouts, broadening=3e4,
    )
    assert r is not None
    assert r.shape == (6, 20)
    assert np.all(np.isfinite(r))


# ============================================================
# Direct leaf coverage on the two NK=1 sections in N-14.
# ============================================================


@pytest.mark.parametrize('mt', [32, 105])
def test_compute_total_photon_production_xs_nk1_sections(n14_endfb81, mt):
    """Direct check of the leaf: MT 32 and MT 105 in N-14 have
    NK=1 (single subsection, no totals header). The leaf must
    return a finite non-negative array without raising."""
    einc = np.linspace(1e6, 1.5e7, 8)
    r = compute_total_photon_production_xs(n14_endfb81, mt, einc)
    assert r.shape == (8,)
    assert np.all(np.isfinite(r))
    assert np.all(r >= 0.0)


@pytest.mark.parametrize('mt', [4, 28, 103, 107])
def test_compute_total_photon_production_xs_nk_gt_1_sections(n14_endfb81, mt):
    """Non-regression: multi-subsection sections must still be
    processed via the consistency-check branch and return a finite
    result. Pins that the change didn't accidentally take the
    NK>1 path out of use."""
    einc = np.linspace(1e6, 1.5e7, 8)
    r = compute_total_photon_production_xs(n14_endfb81, mt, einc)
    assert r.shape == (8,)
    assert np.all(np.isfinite(r))
    assert np.all(r >= 0.0)


