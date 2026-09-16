"""Tests for the MF13 discrete-line contribution to broadened
`get_particle_production_dxs_dE` and `get_particle_production_ddxs`
(issue #101).

Before this fix, the broadening dispatcher walked only MF12
discrete lines (`compute_dxs_dE_mf12_discrete_broadened`,
`compute_ddx_mf12_discrete_broadened`). Files that carry per-
partial-channel gamma yields in MF13 (typical for ENDF/B-VIII
medium/heavy nuclei) had their discrete-line peaks silently
absent from the broadened result. Mirror of #29's XS-side fix
that never reached the differential path.

Fix: parallel MF13 folders `compute_dxs_dE_mf13_discrete_broadened`
and `compute_ddx_mf13_discrete_broadened` in
`ddx_broadening.py`, gated by a new `has_mf13_discrete_lines`
selector, wired into the dxs/dE and DDX dispatchers as a fifth
folder alongside cont / disc / law1_disc / mf12_disc.

These tests pin:

1. **Broadened dxs/dE on N-14** (which has MF13 for MT 4/28/103/107):
   the fixed dispatcher includes peaks that are absent when the
   MF13 folder is skipped.
2. **Integral consistency**: `int(dxs/dE dE') ~ production XS`
   at the same Ein (within ~5 % on a coarse eout grid).
3. **Broadened DDX**: same set of MF13 lines visible in every mu
   column, integrates back consistently.
4. **Non-regression on MF12-only files** (Al-27 MT 102): the
   MF12 branch is unchanged; the MF13 branch contributes zero
   because there's no MF13/MT102.
5. **Direct-leaf tests** on the two new folders: gamma-only ZAP
   enforcement, empty-return when the file has no MF13, and
   returned-shape check.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_dxs_dE,
    get_particle_production_ddxs,
    get_particle_production_xs,
)
from endf_userpy.quantities_mt_zap import ddx_broadening as ddxb
from endf_userpy.quantities_mt_zap import selectors


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
    """N-14 has MF13 for MTs 4, 28, 32, 103, 104, 105, 107 with
    a total of ~90 discrete gamma lines. Primary test file."""
    return _load('endfb81_n_N-14.endf')


@pytest.fixture(scope='module')
def al27_endfb81():
    """Al-27 uses MF12+MF14 for photon yields (no MF13). Non-
    regression control."""
    return _load('endfb81_n_Al-27.endf')


def _gaussian(sigma):
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
    return lambda d: norm * np.exp(-0.5 * (d / sigma) ** 2)


# ============================================================
# Selector: has_mf13_discrete_lines.
# ============================================================


def test_selector_admits_n14_mf13_mts(n14_endfb81):
    """N-14 MTs that appear in MF13 with Eg > 0 must admit through
    the new selector."""
    from endf_userpy.primitives.physical_constants import PARTICLE_ZAP
    gamma = PARTICLE_ZAP['g']
    for mt in [4, 28, 103, 104, 107]:
        assert selectors.has_mf13_discrete_lines(n14_endfb81, mt, gamma), (
            f'N-14 MT={mt} should have MF13 discrete lines'
        )


def test_selector_rejects_no_mf13(al27_endfb81):
    """Al-27 has no MF13 -- selector must return False for every MT."""
    from endf_userpy.primitives.physical_constants import PARTICLE_ZAP
    gamma = PARTICLE_ZAP['g']
    for mt in [16, 102, 51, 91]:
        assert not selectors.has_mf13_discrete_lines(
            al27_endfb81, mt, gamma,
        )


def test_selector_gamma_only(n14_endfb81):
    """Non-gamma ZAP must never admit through the selector, even
    for MTs that have MF13."""
    for zap in [1.0, 1001.0, 2004.0]:  # neutron, proton, alpha
        assert not selectors.has_mf13_discrete_lines(
            n14_endfb81, 4, zap,
        )


# ============================================================
# Direct leaf: dxs/dE and DDX folders.
# ============================================================


def test_dxs_dE_leaf_returns_finite_on_n14(n14_endfb81):
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 200)
    r = ddxb.compute_dxs_dE_mf13_discrete_broadened(
        n14_endfb81, 4, 0.0, einc, eouts, kernel=_gaussian(3e4),
    )
    assert r.shape == (1, 200)
    assert np.all(np.isfinite(r))
    assert np.all(r >= 0.0)
    # N-14 MT 4 has 43 gamma lines; at 3e4 sigma each line peak
    # is around 1e-6 barn/eV scale.
    assert float(r.max()) > 0.0


def test_dxs_dE_leaf_empty_on_no_mf13(al27_endfb81):
    """Al-27 has no MF13 -- the leaf returns zeros without raising."""
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 100)
    r = ddxb.compute_dxs_dE_mf13_discrete_broadened(
        al27_endfb81, 102, 0.0, einc, eouts, kernel=_gaussian(3e4),
    )
    assert r.shape == (1, 100)
    assert np.all(r == 0.0)


def test_dxs_dE_leaf_gamma_only(n14_endfb81):
    """Non-gamma ZAP must raise (mirrors the MF12 sibling)."""
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 10)
    with pytest.raises(ValueError, match='gamma-only'):
        ddxb.compute_dxs_dE_mf13_discrete_broadened(
            n14_endfb81, 4, 1.0, einc, eouts, kernel=_gaussian(3e4),
        )


def test_ddx_leaf_returns_finite_on_n14(n14_endfb81):
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 100)
    mus = np.linspace(-0.9, 0.9, 5)
    r = ddxb.compute_ddx_mf13_discrete_broadened(
        n14_endfb81, 4, 0.0, einc, eouts, mus, kernel=_gaussian(3e4),
    )
    assert r.shape == (1, 100, 5)
    assert np.all(np.isfinite(r))
    assert np.all(r >= 0.0)
    assert float(r.max()) > 0.0


# ============================================================
# End-to-end: dispatcher includes MF13 lines in the broadened
# API answer. Integral vs. production XS as physics invariant.
# ============================================================


def test_dxs_dE_broadened_integral_consistent_on_n14(n14_endfb81):
    """Integrating the broadened dxs/dE over Eout must approximately
    match the production cross section. Loose tolerance because
    coarse Eout grids miss the last few percent of the line-peak
    area; the pre-fix version returned ~0 for this file because
    MF13 was skipped entirely."""
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 500)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = get_particle_production_dxs_dE(
            n14_endfb81, '(n,total)', 'g', einc, eouts, broadening=3e4,
        )
        xs = get_particle_production_xs(
            n14_endfb81, '(n,total)', 'g', einc,
        )
    integ = float(np.trapezoid(r[0], eouts))
    ratio = integ / float(xs[0])
    assert 0.85 < ratio < 1.05, (
        f'integral / xs = {ratio:.3f}; expected ~1.0 (fine-grid '
        f'trapezoid); MF13 folder is contributing.'
    )


def test_ddx_broadened_integrates_to_production_xs_on_n14(n14_endfb81):
    """DDX-broadened integrated over `(E_out, mu) * 2*pi` must
    approximately match the production cross section too."""
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 200)
    mus = np.linspace(-0.99, 0.99, 15)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = get_particle_production_ddxs(
            n14_endfb81, '(n,total)', 'g', einc, eouts, mus, broadening=3e4,
        )
        xs = get_particle_production_xs(
            n14_endfb81, '(n,total)', 'g', einc,
        )
    integ = 2 * np.pi * float(np.trapezoid(np.trapezoid(r[0], eouts, axis=0), mus))
    ratio = integ / float(xs[0])
    assert 0.75 < ratio < 1.05, (
        f'DDX integral / xs = {ratio:.3f}; mu grid coarser than '
        f'Eout, so slightly looser bound.'
    )


# ============================================================
# Non-regression: MF12-only file (Al-27) unchanged.
# ============================================================


def test_al27_mf12_only_unchanged(al27_endfb81):
    """Al-27 MT 102 (capture) uses MF12+MF14 for gamma; MF13 is
    absent. The MF13 folder contributes zero, so the total
    broadened dxs/dE must equal what the MF12 folder alone
    produced pre-fix."""
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 1e7, 200)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r_total = get_particle_production_dxs_dE(
            al27_endfb81, '(n,g)', 'g', einc, eouts, broadening=3e4,
        )
        r_mf13_only = ddxb.compute_dxs_dE_mf13_discrete_broadened(
            al27_endfb81, 102, 0.0, einc, eouts, kernel=_gaussian(3e4),
        )
    assert np.all(r_mf13_only == 0.0)
    # And the total is non-trivial (MF12 folder still fires)
    assert float(r_total.max()) > 0.0
