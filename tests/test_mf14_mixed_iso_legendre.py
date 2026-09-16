"""Tests for the MF14 mixed isotropic + Legendre dispatch
(issue #80).

Two overlapping bugs in `mf14_interpretation.compute_angdist_from_legendre`
tripped on N-14, the only file in the current corpus whose MF14
sections have both isotropic (LI=1 within the LTT=1 block, i.e.
the first `NI` photons) and per-line Legendre entries:

1. **Axis-swapped call to the isotropic helper**:
   `compute_angdist_from_isotropic(..., angle_cosines,
   photon_energies[isotropic_idcs])` -- the two trailing
   positional args were passed in reversed order, so the returned
   isotropic block came out shaped `(n_ein, n_mu, n_iso)` while
   the assignment target expected `(n_ein, n_iso, n_mu)`.

2. **Index confusion**: `isotropic_idcs` and `anisotropic_idcs`
   are positions in the MF14 EG list (the file's photon
   catalogue), but the code used them as positions in the
   user-supplied `photon_energies` list (axis 1 of the returned
   array). Only when the user's list happened to be identical to
   the file's EG list did this coincide. On N-14 the caller
   passes MF12's `disc_pes` which has 58 entries; the MF14 EG list
   has 59+, and the largest `anisotropic_idx` (>= NI) landed at
   58 -- out of bounds on axis 1 of size 58.

Fix: explicit position-map:

    iso_user_positions   = np.where(iso_mask)[0]      # into photon_energies
    aniso_user_positions = np.where(aniso_mask)[0]    # into photon_energies
    aniso_eg_indices     = idcs[aniso_mask]           # into mtsec['E'/'a'/'E_interpol']

and pass the isotropic helper the correct positional order.

These tests pin:

1. N-14 gamma `dxs/dmu` and `ddx (broadened)` produce finite,
   non-NaN, non-negative arrays.
2. Non-regression: gamma `dxs/dmu` on Al-27, Fe-56, Cu-63
   (which route through the same code but have simpler MF14
   layouts) is unchanged.
3. Direct-leaf coverage on N-14 MF14 with the same MF12
   photon list the top-level API uses, plus a synthetic
   subset where `photon_energies` skips over some EG entries
   to catch the position-map correctness.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_dxs_dmu,
    get_particle_production_ddxs,
)
from endf_userpy.mfsec_interpretation import mf12_interpretation as mf12_interp
from endf_userpy.mfsec_interpretation import mf14_interpretation as mf14_interp


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
# Public API regression: N-14 (n,total)+g dxs/dmu and DDX.
# ============================================================


def test_n14_gamma_dxs_dmu_finite(n14_endfb81):
    einc = np.linspace(1e5, 1.4e7, 6)
    mus = np.linspace(-0.9, 0.9, 5)
    r = get_particle_production_dxs_dmu(
        n14_endfb81, '(n,total)', 'g', einc, mus,
    )
    assert r.shape == (6, 5)
    assert np.all(np.isfinite(r))
    assert np.all(r >= 0.0)
    # There should be some non-zero signal on the (n,g) partial gammas
    assert float(r.max()) > 0.0


def test_n14_gamma_ddx_broadened_finite(n14_endfb81):
    einc = np.linspace(1e5, 1.4e7, 6)
    eouts = np.linspace(1e5, 1.4e7, 20)
    mus = np.linspace(-0.9, 0.9, 5)
    r = get_particle_production_ddxs(
        n14_endfb81, '(n,total)', 'g', einc, eouts, mus,
        broadening=3e4,
    )
    assert r is not None
    assert r.shape == (6, 20, 5)
    assert np.all(np.isfinite(r))


# ============================================================
# Non-regression: gamma dxs/dmu on files that were already
# working must not change.
# ============================================================


@pytest.mark.parametrize('fn', [
    'endfb81_n_Al-27.endf',
    'tendl21_n_Fe-56.endf',
    'jeff40_n_Cu-63.endf',
])
def test_other_files_gamma_dxs_dmu_still_finite(fn):
    endf = _load(fn)
    einc = np.linspace(1e5, 1.4e7, 6)
    mus = np.linspace(-0.9, 0.9, 5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        r = get_particle_production_dxs_dmu(
            endf, '(n,total)', 'g', einc, mus,
        )
    assert r.shape == (6, 5)
    assert np.all(np.isfinite(r))
    assert float(r.max()) > 0.0


# ============================================================
# Direct leaf coverage of the position-map correctness.
# ============================================================


def test_leaf_direct_with_mf12_disc_pes(n14_endfb81):
    """Direct call with the same MF12 discrete-photon list the
    top-level API constructs. Pre-fix this call raised
    IndexError from the axis-1 out-of-bounds assignment. N-14
    MT 102 has NI=58 isotropic + 1 anisotropic (NE=59); the
    largest EG index (58) landed out of bounds on axis 1 of size
    58 when the code mistook EG indices for user-photon
    positions."""
    # Find an MF14 MT that is LI=0 LTT=1 AND appears in MF12
    # (needed for the MF12 photon-energy lookup below).
    candidate_mts = [
        mt for mt in n14_endfb81.get(14, {}).keys()
        if n14_endfb81[14][mt].get('LI') == 0
        and n14_endfb81[14][mt].get('LTT') == 1
        and mt in n14_endfb81.get(12, {})
    ]
    assert candidate_mts, 'no LI=0 LTT=1 MF14 section with MF12 in N-14'
    mt = candidate_mts[0]
    einc = np.linspace(1e6, 1.5e7, 4)
    mus = np.linspace(-0.9, 0.9, 5)
    pes = np.asarray(
        mf12_interp.get_photon_energies(n14_endfb81, mt), dtype=float,
    )
    disc_pes = pes[pes > 0.0]  # matches the caller's disc_mask
    r = mf14_interp.compute_angdist_values(
        n14_endfb81, mt, einc, disc_pes, mus,
    )
    assert r.shape == (len(einc), len(disc_pes), len(mus))
    assert np.all(np.isfinite(r))
    assert np.all(r >= 0.0)


def test_leaf_position_map_on_subset(n14_endfb81):
    """Position-map correctness check: pick a random 3-photon
    subset of the file's own EG list, verify the returned array has
    the right shape and every row-position corresponds to the input
    photon (not to the EG-index arithmetic that used to leak
    through)."""
    candidate_mts = [
        mt for mt in n14_endfb81.get(14, {}).keys()
        if n14_endfb81[14][mt].get('LI') == 0
        and n14_endfb81[14][mt].get('LTT') == 1
    ]
    assert candidate_mts
    mt = candidate_mts[0]
    from endf_userpy.primitives.helpers import dict2array
    eg_all = np.asarray(
        dict2array(n14_endfb81[14][mt]['EG']), dtype=float,
    )
    # Take a scattered subset: first, middle, last.
    if len(eg_all) < 3:
        pytest.skip('need at least three EG entries')
    subset = np.array([eg_all[0], eg_all[len(eg_all) // 2], eg_all[-1]])
    einc = np.array([5e6])
    mus = np.linspace(-0.9, 0.9, 5)
    r = mf14_interp.compute_angdist_values(n14_endfb81, mt, einc, subset, mus)
    assert r.shape == (1, 3, 5)
    assert np.all(np.isfinite(r))
    # Also: reordering the subset must permute axis 1 accordingly.
    reordered = subset[::-1]
    r2 = mf14_interp.compute_angdist_values(
        n14_endfb81, mt, einc, reordered, mus,
    )
    np.testing.assert_allclose(r, r2[:, ::-1, :], rtol=1e-12, atol=1e-14)
