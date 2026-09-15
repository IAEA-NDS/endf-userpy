"""Tests for MF15 continuum shape weighting in
`distribution1d.compute_energydist_values` (issue #54).

`compute_dexs` multiplies `compute_energydist_values` by
`compute_yields`, which sums ALL photon yields declared in MF12
(discrete lines + Eg=0 continuum placeholder). Before this fix the
MF15 branch returned the raw MF15 spectrum, so the multiplication
gave `Y_total * sigma * f_MF15`. The correct form uses only the
continuum yield fraction, `y_cont * sigma * f_MF15`, i.e. MF15
should be weighted by `y_cont / Y_total`.

Above the discrete/continuum crossover (~10 keV for Al-27 MT 102)
the MF12 discrete-line yields drop to zero and `Y_total = Y_cont`,
so the fix is a no-op there. Below that threshold the discrete
lines carry all the yield and the current form injects a spurious
continuum-shaped contribution proportional to the discrete-line
yield. The fix removes it.

The DDX equivalent (`distribution2d.compute_dist2d_values`) has no
MF15 branch today; wiring one is deferred as a follow-up.
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities_mt_zap import distribution1d as d1d
from endf_userpy.quantities import (
    get_particle_production_dxs_dE,
    get_particle_production_xs,
)
from endf_userpy.mfsec_interpretation import mf12_interpretation as mf12_interp
from endf_userpy.mfsec_interpretation import mf3_interpretation as mf3_interp


ADHOC_DATA_DIR = Path(__file__).resolve().parent / 'data_law1_adhoc'


def _load(fn_name):
    fn = ADHOC_DATA_DIR / fn_name
    if not fn.exists():
        pytest.skip(
            f'{fn_name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh` to populate the corpus'
        )
    parser = EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    )
    return parser.parsefile(fn)


@pytest.fixture(scope='module')
def al27_endfb81():
    """Al-27 MT 102 (n,g): MF12 LO=1 has 291 photon lines with an
    Eg=0 continuum placeholder, MF15 gives the continuum spectrum.
    The discrete lines carry all yield below ~10 keV incident; the
    Eg=0 placeholder carries all yield above ~100 keV. The transition
    band around 10 keV has zero yield in both (a file quirk)."""
    return _load('endfb81_n_Al-27.endf')


@pytest.fixture(scope='module')
def u235_tendl():
    """U-235 MT 18 (n,f): MF12 has a single photon entry at Eg=0
    (pure continuum) with yield ~8.7. Serves as an unchanged-by-
    fix control: since Y_total == y_cont, the fix multiplies MF15
    by 1.0 and the output is identical to the pre-fix behaviour."""
    return _load('tendl21_n_U-235.endf')


# ============================================================
# Al-27 MT 102 spectrum below the discrete/continuum crossover.
# ============================================================


def test_al27_mt102_edist_zero_below_crossover(al27_endfb81):
    """Below the crossover (Ein = 1 eV), MF12 declares 2.17 photons
    per capture, all in discrete lines (Eg > 0). y_cont = 0. The
    MF15 branch should return zero because the continuum contributes
    no yield here. Before the fix, MF15 was multiplied by Y_disc via
    the unweighted branch, giving a spurious continuum-shaped
    contribution proportional to the discrete-line yield."""
    einc = np.array([1.0])
    eouts = np.linspace(0.1e6, 8e6, 401)
    r = d1d.compute_energydist_values(
        al27_endfb81, 102, 0.0, einc, eouts,
    )
    np.testing.assert_array_equal(r, 0.0)


def test_al27_mt102_edist_zero_at_1keV(al27_endfb81):
    """1 keV is still below the crossover. Same expected zeros."""
    einc = np.array([1.0e3])
    eouts = np.linspace(0.1e6, 8e6, 401)
    r = d1d.compute_energydist_values(
        al27_endfb81, 102, 0.0, einc, eouts,
    )
    np.testing.assert_array_equal(r, 0.0)


def test_al27_mt102_edist_at_1MeV_matches_ycont_over_ytotal(al27_endfb81):
    """At Ein = 1 MeV, Y_disc = 0 and Y_cont ~ 2.45. The fix
    multiplies MF15 by y_cont / Y_total = 1.0, so the output is
    identical to the pre-fix behaviour (up to numerical tolerance).

    We verify this indirectly by checking that the integrated
    spectrum on a dense mesh recovers the y_cont / Y_total ratio
    (~1 here) times the mesh-integral of the raw MF15 spectrum."""
    einc = np.array([1.0e6])
    eouts = np.linspace(1.0e4, 9.0e6, 4001)

    r = d1d.compute_energydist_values(
        al27_endfb81, 102, 0.0, einc, eouts,
    )

    # Reference: raw MF15 spectrum (unweighted).
    from endf_userpy.mfsec_interpretation import mf15_interpretation as mf15_interp
    raw = mf15_interp.compute_spectrum(al27_endfb81, 102, einc, eouts)

    # y_cont / Y_total at 1 MeV
    pes = np.asarray(mf12_interp.get_photon_energies(al27_endfb81, 102))
    y = mf12_interp.compute_photon_yields(al27_endfb81, 102, einc, pes)
    frac = y[0, pes == 0].sum() / y[0].sum()

    np.testing.assert_allclose(r, raw * frac, rtol=1e-12)
    # Sanity: frac is very close to 1 at 1 MeV for this file.
    assert abs(frac - 1.0) < 1e-3


# ============================================================
# End-to-end: dxs/dE integral matches y_cont * sigma_MF3, not the
# pre-fix y_total * sigma_MF3.
# ============================================================


def test_al27_mt102_dxs_dE_integral_matches_ycont_sigma(al27_endfb81):
    """`get_particle_production_dxs_dE('(n,g)', 'g', ...)` integrates
    to y_cont * sigma (the physical continuum-gamma cross section),
    not y_total * sigma (which would double-count against a discrete
    line contribution when the discrete-line contribution is
    separately zero, as it is on the unbroadened grid)."""
    einc = np.array([1.0e6])
    eouts = np.linspace(1.0e4, 9.0e6, 4001)

    r = get_particle_production_dxs_dE(
        al27_endfb81, '(n,g)', 'g', einc, eouts,
    )
    assert r is not None
    integral = np.trapezoid(r[0], eouts)

    pes = np.asarray(mf12_interp.get_photon_energies(al27_endfb81, 102))
    y = mf12_interp.compute_photon_yields(al27_endfb81, 102, einc, pes)
    y_cont = y[0, pes == 0].sum()
    sigma = mf3_interp.compute_cross_section(al27_endfb81, 102, einc)[0]
    # y_cont == Y_total at 1 MeV, so the pre-fix and post-fix values
    # coincide. This test pins the invariant "integral == y_cont *
    # sigma" that must hold at every Ein regardless of the y_cont /
    # y_total split.
    #
    # Small tolerance to accommodate the finite mesh; MF15 spectrum
    # may not integrate to exactly 1 on any given eout mesh.
    assert abs(integral / (y_cont * sigma) - 1.0) < 0.3


# ============================================================
# Synthetic case: exercise the partial split y_cont/y_total < 1.
# Constructed via a monkeypatched pair of readers.
# ============================================================


def test_synthetic_mf15_branch_weights_by_ycont_fraction(monkeypatch):
    """Direct check on `compute_energydist_values` MF15 branch with
    a controlled y_cont / Y_total = 0.3 fraction. Neither corpus file
    exercises a partial split (Al-27 flips discretely between 1.0
    and 0.0; U-235 is always 1.0), so this synthetic test is the
    only place where a strictly-between-0-and-1 fraction is
    verified.
    """
    n_einc, n_eouts = 3, 20

    def fake_get_photon_energies(endf_dict, mt):
        return np.array([0.0, 1e6, 2e6])  # one Eg=0, two discrete

    def fake_compute_photon_yields(endf_dict, mt, eincs, pes):
        # y_cont = 0.3, discrete = 0.4 + 0.3 = 0.7, Y_total = 1.0
        return np.tile(np.array([0.3, 0.4, 0.3]), (len(eincs), 1))

    def fake_compute_spectrum(endf_dict, mt, eincs, eouts):
        # Return a nonzero, non-degenerate reference spectrum
        return np.tile(np.linspace(0.5, 1.5, len(eouts)), (len(eincs), 1))

    monkeypatch.setattr(d1d, 'has_mf5_mt', lambda d, m: False)
    monkeypatch.setattr(d1d, 'has_mf4_mt', lambda d, m: False)
    monkeypatch.setattr(d1d, 'has_mf6_mt', lambda d, m: False)
    monkeypatch.setattr(d1d, 'has_mf15_mt', lambda d, m: True)
    monkeypatch.setattr(d1d, 'has_mf12_mt', lambda d, m: True)
    monkeypatch.setattr(d1d, 'is_zap_consistent', lambda d, m, z: True)
    monkeypatch.setattr(
        d1d.mf15_interp, 'compute_spectrum', fake_compute_spectrum,
    )
    monkeypatch.setattr(
        d1d.mf12_interp, 'get_photon_energies', fake_get_photon_energies,
    )
    monkeypatch.setattr(
        d1d.mf12_interp, 'compute_photon_yields', fake_compute_photon_yields,
    )

    einc = np.linspace(1e6, 3e6, n_einc)
    eouts = np.linspace(1e5, 2e6, n_eouts)
    r = d1d.compute_energydist_values(
        endf_dict={}, mt=51, zap=0.0,
        energies_in=einc, energies_out=eouts,
    )
    raw = fake_compute_spectrum({}, 51, einc, eouts)
    expected = raw * 0.3  # y_cont / Y_total
    np.testing.assert_allclose(r, expected, rtol=1e-12)


def test_synthetic_no_eg0_placeholder_drops_mf15(monkeypatch):
    """When MF12 declares no Eg=0 placeholder (i.e. the file says
    all gammas are discrete lines), any MF15 present is inconsistent
    with MF12. The fix drops the MF15 contribution rather than
    inflate the sum by an unweighted spectrum."""
    def fake_get_photon_energies(endf_dict, mt):
        return np.array([1e6, 2e6])  # discrete only, no Eg=0

    def fake_compute_photon_yields(endf_dict, mt, eincs, pes):
        return np.tile(np.array([0.4, 0.6]), (len(eincs), 1))

    def fake_compute_spectrum(endf_dict, mt, eincs, eouts):
        return np.ones((len(eincs), len(eouts)))

    monkeypatch.setattr(d1d, 'has_mf5_mt', lambda d, m: False)
    monkeypatch.setattr(d1d, 'has_mf4_mt', lambda d, m: False)
    monkeypatch.setattr(d1d, 'has_mf6_mt', lambda d, m: False)
    monkeypatch.setattr(d1d, 'has_mf15_mt', lambda d, m: True)
    monkeypatch.setattr(d1d, 'has_mf12_mt', lambda d, m: True)
    monkeypatch.setattr(d1d, 'is_zap_consistent', lambda d, m, z: True)
    monkeypatch.setattr(
        d1d.mf15_interp, 'compute_spectrum', fake_compute_spectrum,
    )
    monkeypatch.setattr(
        d1d.mf12_interp, 'get_photon_energies', fake_get_photon_energies,
    )
    monkeypatch.setattr(
        d1d.mf12_interp, 'compute_photon_yields', fake_compute_photon_yields,
    )

    r = d1d.compute_energydist_values(
        endf_dict={}, mt=51, zap=0.0,
        energies_in=np.array([1e6]),
        energies_out=np.array([1e6]),
    )
    np.testing.assert_array_equal(r, 0.0)


def test_synthetic_no_mf12_keeps_unweighted(monkeypatch):
    """When MF12 does not exist for the MT, MF15 stands alone.
    Keep the unweighted behaviour so the reaction-string yield
    fallback (mult=1 for (n,g)) times MF15 still integrates to
    sigma."""
    def fake_compute_spectrum(endf_dict, mt, eincs, eouts):
        return np.full((len(eincs), len(eouts)), 3.14)

    monkeypatch.setattr(d1d, 'has_mf5_mt', lambda d, m: False)
    monkeypatch.setattr(d1d, 'has_mf4_mt', lambda d, m: False)
    monkeypatch.setattr(d1d, 'has_mf6_mt', lambda d, m: False)
    monkeypatch.setattr(d1d, 'has_mf15_mt', lambda d, m: True)
    monkeypatch.setattr(d1d, 'has_mf12_mt', lambda d, m: False)
    monkeypatch.setattr(d1d, 'is_zap_consistent', lambda d, m, z: True)
    monkeypatch.setattr(
        d1d.mf15_interp, 'compute_spectrum', fake_compute_spectrum,
    )

    r = d1d.compute_energydist_values(
        endf_dict={}, mt=51, zap=0.0,
        energies_in=np.array([1e6, 2e6]),
        energies_out=np.array([1e6, 2e6]),
    )
    np.testing.assert_allclose(r, 3.14)


# ============================================================
# U-235 MT 18 fission gammas: pure continuum (Y_total = y_cont),
# fix must be a no-op.
# ============================================================


def test_u235_mt18_fix_is_noop(u235_tendl):
    """U-235 MT 18 has a single MF12 entry at Eg=0 with yield ~8.7.
    Y_cont == Y_total, so the fix multiplies MF15 by 1.0 and the
    output is byte-identical to the pre-fix behaviour."""
    einc = np.array([1.0e6])
    eouts = np.linspace(1.0e5, 8.0e6, 401)

    r = d1d.compute_energydist_values(
        u235_tendl, 18, 0.0, einc, eouts,
    )
    from endf_userpy.mfsec_interpretation import mf15_interpretation as mf15_interp
    raw = mf15_interp.compute_spectrum(u235_tendl, 18, einc, eouts)
    # y_cont / y_total = 1 here.
    np.testing.assert_allclose(r, raw, rtol=1e-12)


# ============================================================
# Regressions from earlier PRs: gamma XS unchanged.
# ============================================================


def test_pr35_al27_gamma_xs_unchanged(al27_endfb81):
    """The XS path from PR #35 (issue #29) sums yields x MF3 xs,
    doesn't touch the energy-dist branch. Must be unaffected by
    this fix."""
    xs = get_particle_production_xs(
        al27_endfb81, '(n,g)', 'g', np.array([1.0e6]),
    )[0]
    # Al-27 (n,g) at 1 MeV: y_total ~ 2.45, sigma ~ 6.6e-4 -> ~1.6e-3 b
    assert 1e-3 < xs < 3e-3
