"""Tests for the fission-gamma admission fix (issue #126).

Before this fix, `selectors.contains_zap` had a hard-coded early
return for fission MTs (18, 19, 20, 21, 38) that admitted ONLY
neutrons -- even on files that carry fission-gamma content in
MF12/MF13/MF15 (Pu-239, U-238, U-235 in the ad-hoc corpus). Every
fission-gamma query returned zero (or `None` after the None-
handling short-circuit in `compute_cumulative_quantity`), silently.

Fix: loosen the fission special case in `contains_zap` to admit
gammas when the file declares them in MF12 or MF13. In
`compute_yields`, split the `mt == 18` branch to route neutrons
via MF1/MT456 nubar and gammas via `compute_total_gamma_yields`
(same MF12/MF13 path as inelastic partial-channel gammas). Other
ZAPs on fission MTs still raise a clean ValueError with a message
naming both supported ZAPs -- charged fission fragments are not
representable through this route.

Corpus reproducer: U-238 JENDL-5, MT18 carries fission XS in MF3
plus a single Eg=0 continuum-placeholder in MF12/MF14/MF15. The
reaction string `(n,fission)` maps to MT18. Pre-fix, the gamma
production query returns None; post-fix, ~7.3 barn at 14 MeV
(fission XS ~1.15 b times ~6.3 photons/fission).
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_xs,
    get_particle_production_dxs_dE,
    get_particle_production_ddxs,
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


# ============================================================
# Primary reproducer: U-238 (n,fission) gamma XS
# ============================================================


@pytest.mark.parametrize('fn_name', [
    'jendl5_n_U-238.endf',
    'endfb81_n_Pu-239.endf',
    'tendl21_n_U-235.endf',
])
def test_fission_gamma_xs_nonzero(fn_name):
    """Every fission file in the ad-hoc corpus that carries
    MF15 for MT18 must now produce nonzero fission-gamma XS."""
    endf = _load(fn_name)
    einc = np.array([1.4e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,fission)', 'g', einc)
    assert xs is not None, (
        f'{fn_name}: fission gamma XS is None -- pre-fix behaviour '
        f'(contains_zap short-circuited fission MTs to neutrons only)'
    )
    assert xs[0] > 0, (
        f'{fn_name}: fission gamma XS is zero ({xs[0]:.4g}); expected '
        f'a nontrivial value (few barn from ~6 photons/fission)'
    )
    # Sanity: fission gamma production is bounded above by
    # multiplicity * fission XS. For U-238 at 14 MeV ~1.2 b * 7-8
    # photons ~ 8-10 barn. Loose upper bound at 30 barn.
    assert xs[0] < 30, (
        f'{fn_name}: fission gamma XS {xs[0]:.4g} is implausibly '
        f'large; something over-counts'
    )


# ============================================================
# Definitional consistency: dxs/dE and DDX integrate to the
# same value; DDX integral over mu equals dxs/dE.
# ============================================================


def test_u238_fission_gamma_ddx_matches_dxs_dE():
    """U-238 at 14 MeV: integrating the broadened DDX over
    dOmega = 2 pi dmu at each E_out must equal the broadened
    1D dxs/dE at the same (Ein, E_out). Pins the MF15 continuum
    normalisation on the fission channel."""
    endf = _load('jendl5_n_U-238.endf')
    einc = np.array([1.4e7])
    eouts = np.linspace(1e4, 1.4e7, 400)
    mus = np.linspace(-1.0, 1.0, 21)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dxs_dE = get_particle_production_dxs_dE(
            endf, '(n,fission)', 'g', einc, eouts, broadening=200e3,
        )
        ddx = get_particle_production_ddxs(
            endf, '(n,fission)', 'g', einc, eouts, mus, broadening=200e3,
        )
    ddx_int_mu = 2 * np.pi * trapezoid(ddx, mus, axis=-1)
    ddx_int = float(trapezoid(ddx_int_mu[0], eouts))
    dxs_dE_int = float(trapezoid(dxs_dE[0], eouts))
    diff = abs(ddx_int - dxs_dE_int) / dxs_dE_int
    assert diff < 0.01, (
        f'DDX / dxs_dE integral inconsistency: DDX={ddx_int:.6g}, '
        f'dxs_dE={dxs_dE_int:.6g}, delta={diff:.2%}'
    )


# ============================================================
# Non-regression: neutron admission on fission MTs unchanged.
# ============================================================


def test_fission_neutron_xs_unchanged():
    """Pre-existing MT18 fission-neutron production must still
    work. The fix loosened contains_zap to admit gammas, but
    neutrons stayed always-admitted."""
    endf = _load('jendl5_n_U-238.endf')
    einc = np.array([1.4e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs_n = get_particle_production_xs(
            endf, '(n,fission)', 'n', einc,
        )
    assert xs_n is not None
    # U-238 at 14 MeV: fission XS ~1.2 b times ~4 prompt neutrons
    # per fission ~ 5 barn. Very loose bounds.
    assert 1 < xs_n[0] < 20, (
        f'fission neutron XS {xs_n[0]:.4g} outside expected range '
        f'(loose 1-20 barn window for U-238 at 14 MeV)'
    )


# ============================================================
# Unsupported ZAPs on fission MTs get a clean, informative error.
# ============================================================


def test_fission_unsupported_zap_raises():
    """A ZAP that's neither neutron nor gamma-with-MF12/MF13 on a
    fission MT should raise ValueError with both supported ZAPs
    named, not silently produce zero."""
    endf = _load('jendl5_n_U-238.endf')
    einc = np.array([1.4e7])
    # ZAP=1001 = proton. Not supported for fission.
    from endf_userpy.quantities_mt_zap.quantities import compute_yields
    with pytest.raises(ValueError, match='fission'):
        compute_yields(endf, 18, 1001.0, einc)
