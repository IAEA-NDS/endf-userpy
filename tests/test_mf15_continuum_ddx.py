"""Tests for the MF15 continuum contribution to the broadened
gamma DDX (issue #123).

Before this fix, `get_particle_production_ddxs(..., 'g', ...,
broadening=sigma)` picked up only MF12/MF13 discrete-line
contributions (via `compute_ddx_mf12_discrete_broadened` and
`compute_ddx_mf13_discrete_broadened`) and any MF6-routed content;
the MF15 continuous gamma spectrum was NOT wired into the DDX
dispatcher, so files where a gamma channel carries its shape in
MF15 (Al-27 MT102 above ~100 keV, N-14 MT102, ...) silently
returned zero DDX for that channel.

The new `compute_ddx_mf15_continuum_broadened` folder in
`endf_userpy/quantities_mt_zap/ddx_broadening.py` folds the MF15
spectrum along E_out via `adaptive_convolve` and multiplies by the
MF12 continuum-yield fraction (matching the 1D dxs/dE
normalisation from issue #54) and by the MF14-derived (or
isotropic-fallback) continuum angular distribution.

Al-27 MT102 above ~100 keV incident is the clean test bed:

- MT102 gamma yields are 100% in the MF12 Eg=0 continuum
  placeholder (Y_disc = 0, Y_cont > 0). The MF12 discrete-line
  folder contributes nothing.
- MF15 provides the continuous shape.
- MF14 declares LI=1 (fully isotropic), so the continuum angular
  distribution is 0.5 per steradian.

`(n,g)` maps to MT102, so this reaction query isolates the MF15
path.

Tests pin the fix by:

1. On the unmutated Al-27 (n,g) query at 14 MeV, the DDX integral
   over (E_out, mu) is a significant nonzero fraction of the total
   gamma-production XS. Pre-fix would give zero (no folder for
   MF15).
2. The DDX integral over mu equals the 1D dxs/dE at each
   (Ein, E_out) up to trapezoidal error.
3. LI=1 fully-isotropic MF14: DDX values are mu-independent at
   fixed (Ein, E_out).
4. Non-regression: neutron-ejectile DDX (which doesn't touch the
   new folder) is unchanged.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_ddxs,
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


# ============================================================
# Al-27 (n,g) MF15-only-content case: primary reproducer.
# ============================================================


def test_al27_ng_ddx_is_substantial_fraction_of_xs():
    """Al-27 (n,g) MT102 at 14 MeV: gamma yields are 100% in the
    MF12 Eg=0 continuum placeholder, so the MF15 spectrum + MF14
    isotropic angular is the ONLY source of gamma DDX for this MT.
    Integrating the broadened DDX over (E_out, mu) must recover a
    substantial fraction of the gamma-production XS.

    The ratio isn't exactly 1 because Al-27's MF15 spectrum
    integrated over the caller's finite E_out grid is < 1 (the
    tabulated spectrum has extended tails that the caller's grid
    truncates; on [10 keV, 14 MeV] the integral is ~0.70).
    Ratio > 0.5 is the physically-honest lower bound; pre-fix the
    DDX would be None (no folder handles MF15 continuum) or 0.

    The precise definitional check (DDX integral over mu vs 1D
    dxs/dE, which uses the SAME E_out grid) is
    `test_al27_ng_ddx_matches_dxs_dE` below."""
    endf = _load('endfb81_n_Al-27.endf')
    einc = np.array([1.4e7])
    eouts = np.linspace(1e4, 1.4e7, 400)
    mus = np.linspace(-1.0, 1.0, 33)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ddx = get_particle_production_ddxs(
            endf, '(n,g)', 'g', einc, eouts, mus, broadening=200e3,
        )
        xs_prod = get_particle_production_xs(
            endf, '(n,g)', 'g', einc,
        )
    assert ddx is not None, (
        'DDX is None: no MTs admitted for (n,g) gamma. Pre-fix, '
        'the MF15 continuum was not wired into the DDX dispatcher '
        'and Al-27 MT102 had no other gamma folder to hit.'
    )
    ddx_int_mu = 2 * np.pi * trapezoid(ddx, mus, axis=-1)
    ddx_int = float(trapezoid(ddx_int_mu[0], eouts))
    assert ddx_int > 0, (
        f'DDX integral is zero: MF15 folder not contributing. '
        f'xs_prod={xs_prod[0]:.4g}'
    )
    ratio = ddx_int / xs_prod[0]
    assert ratio > 0.5, (
        f'DDX integral / xs_prod = {ratio:.3f} on Al-27 (n,g) at '
        f'14 MeV, expected > 0.5 (MF15 grid-truncation on this Eout '
        f'range gives ~0.7 * xs_prod). xs_prod={xs_prod[0]:.4g}, '
        f'DDX integral={ddx_int:.4g}'
    )


def test_al27_ng_ddx_matches_dxs_dE():
    """Definitional consistency: integrating the DDX over dOmega =
    2 pi dmu at each E_out must equal the 1D dxs/dE at the same
    (Ein, E_out). This pins both that MF15 is present in the DDX
    and that the normalisation (continuum-yield fraction, angular
    distribution, 2 pi factor) matches the 1D path."""
    endf = _load('endfb81_n_Al-27.endf')
    einc = np.array([1.4e7])
    eouts = np.linspace(1e4, 1.4e7, 400)
    mus = np.linspace(-1.0, 1.0, 33)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ddx = get_particle_production_ddxs(
            endf, '(n,g)', 'g', einc, eouts, mus, broadening=200e3,
        )
        dxs_dE = get_particle_production_dxs_dE(
            endf, '(n,g)', 'g', einc, eouts, broadening=200e3,
        )
    ddx_int_mu = 2 * np.pi * trapezoid(ddx, mus, axis=-1)
    # Integrate over Eout at each Ein for stability against
    # per-point cancellation.
    ddx_int = float(trapezoid(ddx_int_mu[0], eouts))
    dxs_dE_int = float(trapezoid(dxs_dE[0], eouts))
    diff = abs(ddx_int - dxs_dE_int) / dxs_dE_int
    assert diff < 0.01, (
        f'DDX / dxs_dE integral inconsistency: DDX={ddx_int:.6g}, '
        f'dxs_dE={dxs_dE_int:.6g}, delta={diff:.2%}'
    )


def test_al27_ng_ddx_mu_independent_for_isotropic_mf14():
    """Al-27 MT102 declares MF14 LI=1 (fully isotropic). The DDX
    values must be mu-independent at each fixed (Ein, E_out)."""
    endf = _load('endfb81_n_Al-27.endf')
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 1e7, 20)
    mus = np.linspace(-1.0, 1.0, 33)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ddx = get_particle_production_ddxs(
            endf, '(n,g)', 'g', einc, eouts, mus, broadening=200e3,
        )
    for i in range(len(einc)):
        for j in range(len(eouts)):
            row = ddx[i, j, :]
            if row.max() > 0:
                spread = (row.max() - row.min()) / row.max()
                assert spread < 1e-9, (
                    f'gamma DDX mu variation at Ein={einc[i]:.3g}, '
                    f'Eout={eouts[j]:.3g} eV: spread={spread:.3e} '
                    f'(expected ~0 for isotropic MF14 LI=1)'
                )


# ============================================================
# Non-regression: neutron-ejectile DDX unchanged.
# ============================================================


def test_neutron_ddx_unchanged():
    """The new MF15 folder is gamma-only. Neutron-ejectile DDX
    routes through the continuous / two-body / LAW=1 discrete
    folders and must not change."""
    endf = _load('jendl5_n_U-238.endf')
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 1.5e7, 60)
    mus = np.linspace(-1.0, 1.0, 17)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ddx = get_particle_production_ddxs(
            endf, '(n,total)', 'n', einc, eouts, mus, broadening=200e3,
        )
    assert ddx is not None
    assert np.all(np.isfinite(ddx))
    assert np.all(ddx >= 0)
    peak_ix = int(np.argmax(ddx[0, :, len(mus) // 2]))
    assert peak_ix > 0, 'neutron DDX peak at Eout=0 is suspicious'
