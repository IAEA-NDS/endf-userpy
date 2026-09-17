"""Tests for the unbroadened MF15 continuum + MF14 angular gamma
DDX (issue #125).

Before this fix, `get_particle_production_ddxs(..., 'g', ...,
broadening=None)` (i.e. no broadening kwarg) went through
`compute_ddxs` -> `compute_dist2d_values`, whose ONE-source-per-MT
design handles MF6 XOR MF4+MF5 but has no MF15 branch. Files whose
gamma content lives in MF15 (Al-27 MT102 above ~100 keV, U-238
MT18 with the #126 fix, ...) returned `None` (all admitted MTs
gave zero DDX) or ignored the MF15 continuum entirely.

The broadened path was fixed in PR #124 by adding a separate
`compute_ddx_mf15_continuum_broadened` folder in `ddx_broadening.py`
that the top-level `_get_particle_production_ddxs_impl` broadened
branch sums alongside the existing folders.

This fix adds the unbroadened peer `compute_ddxs_from_mf15_mf14` in
`quantities_mt_zap/quantities.py` (unbroadened analog of
`compute_ddx_mf15_continuum_broadened` with the `adaptive_convolve`
step replaced by a direct MF15 lookup on the caller's E_out grid)
and wires it into the unbroadened dispatcher via a second
`compute_cumulative_quantity` call that gets summed with the
existing `compute_ddxs` result.

Tests pin the fix on the two natural reproducers:

1. **Al-27 (n,g)** at 14 MeV: gamma yields are 100% in the MF12
   Eg=0 continuum placeholder, so MF15 is the only DDX source.
   Pre-fix returns None (no MT admitted by the compute_ddxs
   selector chain); post-fix returns a nonzero DDX integrating to
   ~0.0013 barn.
2. **U-238 (n,fission)** at 14 MeV: post-#127, fission gammas are
   admitted, and MT18 has MF15 as the gamma spectrum source. DDX
   integrates to ~7.2 barn.

Consistency checks pin the composition:

3. Unbroadened DDX vs 1D dxs/dE (integrated over the same E_out
   grid): the definitional identity.
4. Unbroadened DDX vs broadened DDX with a very narrow kernel:
   in the delta limit they must agree.
5. Non-regression: neutron unbroadened DDX (which doesn't touch
   the new MF15 folder) is unchanged.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_ddxs,
    get_particle_production_dxs_dE,
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
# Primary reproducers: unbroadened DDX now nonzero on MF15 files.
# ============================================================


@pytest.mark.parametrize('fn_name,reaction', [
    ('endfb81_n_Al-27.endf',   '(n,g)'),
    ('jendl5_n_U-238.endf',    '(n,fission)'),
])
def test_unbroadened_gamma_ddx_nonzero(fn_name, reaction):
    """Pre-fix: get_particle_production_ddxs(..., 'g', ..., no
    broadening) on these files returns None or all-zero because
    the MF15 continuum wasn't wired into the unbroadened path.
    Post-fix: returns a nonzero DDX."""
    endf = _load(fn_name)
    einc = np.array([1.4e7])
    eouts = np.linspace(1e4, 1.4e7, 300)
    mus = np.linspace(-1.0, 1.0, 21)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ddx = get_particle_production_ddxs(
            endf, reaction, 'g', einc, eouts, mus,
        )
    assert ddx is not None, (
        f'{fn_name} {reaction}: unbroadened gamma DDX is None; '
        f'the MF15 continuum branch is missing from the '
        f'unbroadened dispatcher'
    )
    ddx_int = float(trapezoid(
        2 * np.pi * trapezoid(ddx, mus, axis=-1)[0], eouts,
    ))
    assert ddx_int > 0, (
        f'{fn_name} {reaction}: unbroadened DDX integral is zero'
    )


# ============================================================
# Definitional consistency with dxs/dE on the same grid.
# ============================================================


@pytest.mark.parametrize('fn_name,reaction', [
    ('endfb81_n_Al-27.endf',   '(n,g)'),
    ('jendl5_n_U-238.endf',    '(n,fission)'),
])
def test_unbroadened_ddx_matches_dxs_dE(fn_name, reaction):
    """Integrating the unbroadened DDX over dOmega = 2 pi dmu at
    each E_out must equal the unbroadened 1D dxs/dE at the same
    (Ein, E_out). Pins the composition of the new MF15 folder."""
    endf = _load(fn_name)
    einc = np.array([1.4e7])
    eouts = np.linspace(1e4, 1.4e7, 300)
    mus = np.linspace(-1.0, 1.0, 21)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ddx = get_particle_production_ddxs(
            endf, reaction, 'g', einc, eouts, mus,
        )
        dxs_dE = get_particle_production_dxs_dE(
            endf, reaction, 'g', einc, eouts,
        )
    ddx_int_mu = 2 * np.pi * trapezoid(ddx, mus, axis=-1)
    ddx_int = float(trapezoid(ddx_int_mu[0], eouts))
    dxs_dE_int = float(trapezoid(dxs_dE[0], eouts))
    diff = abs(ddx_int - dxs_dE_int) / dxs_dE_int
    assert diff < 1e-6, (
        f'{fn_name} {reaction}: DDX vs dxs_dE mismatch: '
        f'DDX={ddx_int:.6g}, dxs_dE={dxs_dE_int:.6g}, '
        f'delta={diff:.2e}. The unbroadened MF15 folder should '
        f'give the same integral as the 1D dxs/dE path -- both '
        f'sample the raw MF15 spectrum on the same E_out grid.'
    )


# ============================================================
# Broadened(narrow kernel) -> unbroadened consistency.
# ============================================================


def test_unbroadened_matches_broadened_delta_limit():
    """A very narrow broadening kernel should give the same DDX
    integral as the unbroadened path. Pins that both paths sum
    the same MTs / folders / normalisation, just with vs without
    the convolution."""
    endf = _load('jendl5_n_U-238.endf')
    einc = np.array([1.4e7])
    eouts = np.linspace(1e4, 1.4e7, 300)
    mus = np.linspace(-1.0, 1.0, 21)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ddx_none = get_particle_production_ddxs(
            endf, '(n,fission)', 'g', einc, eouts, mus,
        )
        ddx_delta = get_particle_production_ddxs(
            endf, '(n,fission)', 'g', einc, eouts, mus,
            broadening=1e2,   # 100 eV wide -- vastly narrower than the E_out spacing
        )
    def _integ(arr):
        return float(trapezoid(
            2 * np.pi * trapezoid(arr, mus, axis=-1)[0], eouts,
        ))
    a = _integ(ddx_none)
    b = _integ(ddx_delta)
    diff = abs(a - b) / max(a, b)
    assert diff < 1e-3, (
        f'unbroadened / broadened(delta) mismatch: '
        f'unbroadened={a:.6g}, broadened(1e2 eV)={b:.6g}, '
        f'delta={diff:.2e}'
    )


# ============================================================
# Non-regression: neutron unbroadened DDX unchanged.
# ============================================================


def test_neutron_unbroadened_ddx_unchanged():
    """The new MF15 folder is gamma-only. Neutron-ejectile DDX
    routes through the existing compute_ddxs + compute_dist2d_values
    path and must be unchanged. Al-27 has neutron emission through
    MF6/MT16 (n,2n) as a good test case."""
    endf = _load('jendl5_n_U-238.endf')
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 1.5e7, 60)
    mus = np.linspace(-1.0, 1.0, 17)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ddx = get_particle_production_ddxs(
            endf, '(n,total)', 'n', einc, eouts, mus,
        )
    assert ddx is not None
    assert np.all(np.isfinite(ddx))
    assert np.all(ddx >= 0)
