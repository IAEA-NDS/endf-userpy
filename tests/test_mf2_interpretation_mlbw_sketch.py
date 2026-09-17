"""Sketch-level tests for backend-agnostic MLBW reconstruction.

Verifies that the numpy MLBW reconstruction
(:mod:`endf_userpy.mfsec_interpretation.mf2_interpretation_mlbw`)
handles the physics correctly and that the same code, run through
the JAX backend of :mod:`endf_userpy.primitives.array_ns`, gives
bit-comparable results. Not an ENDF-preprocessing test -- the
data structures are hand-built to isolate the physics.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.tab1 import TAB1
from endf_userpy.mfsec_interpretation import mf2_interpretation_mlbw as mlbw


def _constant_tab1(value: float, e_lo: float = 1e-5, e_hi: float = 2e7) -> TAB1:
    """Single-panel TAB1 that returns `value` on [e_lo, e_hi]."""
    return TAB1(
        x=np.array([e_lo, e_hi], dtype=np.float64),
        y=np.array([value, value], dtype=np.float64),
        nbt=np.array([1], dtype=np.int32),   # last panel endpoint = NP-1 = 1
        intp=np.array([2], dtype=np.int32),  # lin-lin
    )


def _single_resonance_data(er: float, gn: float, gg: float,
                           two_j: int = 1, spi: float = 0.5,
                           ki: float = 1e-5, r_a: float = 0.6,
                           r_ap: float = 0.6) -> mlbw.MLBWData:
    """Toy MLBW input: one s-wave (L=0) channel, one resonance."""
    g_stat = (two_j + 1) / (2 * (2 * spi + 1))
    return mlbw.MLBWData(
        abn=1.0,
        spi=spi,
        ki=ki,
        qx=0.0,
        r_a=_constant_tab1(r_a),
        r_ap=_constant_tab1(r_ap),
        ch_l=np.array([0], dtype=np.int32),
        ch_g=np.array([g_stat], dtype=np.float64),
        res_channel=np.array([0], dtype=np.int32),
        res_l=np.array([0], dtype=np.int32),
        res_er=np.array([er], dtype=np.float64),
        res_gn=np.array([gn], dtype=np.float64),
        res_gg=np.array([gg], dtype=np.float64),
        res_gf=np.array([0.0], dtype=np.float64),
        res_gx=np.array([0.0], dtype=np.float64),
    )


# ============================================================
# Backend adapter surface.
# ============================================================


def test_numpy_backend_is_always_available():
    assert 'numpy' in array_ns.available_backends()


def test_numpy_backend_forwards_ufuncs():
    xp = array_ns.get_backend('numpy')
    assert xp.name == 'numpy'
    x = xp.asarray([1.0, 4.0, 9.0])
    np.testing.assert_array_equal(xp.sqrt(x), np.array([1.0, 2.0, 3.0]))


def test_numpy_segment_sum():
    xp = array_ns.get_backend('numpy')
    data = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    seg = xp.asarray([0, 1, 1, 2, 0])
    out = xp.segment_sum(data, seg, 3)
    np.testing.assert_array_equal(out, np.array([6.0, 5.0, 4.0]))


def test_numpy_scan_all_none_returns_final_carry():
    xp = array_ns.get_backend('numpy')

    def step(carry, x):
        return carry + x, None

    carry, ys = xp.scan(step, 0.0, xp.asarray([1.0, 2.0, 3.0, 4.0]))
    assert carry == 10.0
    assert ys is None


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match='unknown backend'):
        array_ns.get_backend('theano')


# ============================================================
# TAB1 interpolation.
# ============================================================


def test_tab1_interp_lin_lin_basic():
    from endf_userpy.primitives import tab1 as tab1_mod
    xp = array_ns.get_backend('numpy')
    t = TAB1(
        x=np.array([1.0, 2.0, 4.0], dtype=np.float64),
        y=np.array([10.0, 20.0, 40.0], dtype=np.float64),
        nbt=np.array([2], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )
    xs = xp.asarray([1.0, 1.5, 2.0, 3.0, 4.0])
    ys = tab1_mod.interp(t, xs, xp)
    np.testing.assert_allclose(ys, [10.0, 15.0, 20.0, 30.0, 40.0])


def test_tab1_interp_extrapolation_returns_zero():
    from endf_userpy.primitives import tab1 as tab1_mod
    xp = array_ns.get_backend('numpy')
    t = _constant_tab1(1.0, e_lo=1.0, e_hi=10.0)
    # 0.5 is below the table, 100 is above; both should hit the
    # ilaw=0 sentinel and return 0.
    xs = xp.asarray([0.5, 1.0, 5.0, 10.0, 100.0])
    ys = tab1_mod.interp(t, xs, xp)
    np.testing.assert_allclose(ys, [0.0, 1.0, 1.0, 1.0, 0.0])


# ============================================================
# Penetration / shift / phase factors.
# ============================================================


def test_pnt_shf_L0_closed_form():
    """L=0: P(rho) = rho, S(rho) = 0."""
    from endf_userpy.mfsec_interpretation import mf2_interpretation_factors as factors
    xp = array_ns.get_backend('numpy')
    rho = xp.asarray([0.1, 1.0, 3.14])
    L = xp.asarray([0, 0, 0])
    p, s = factors.pnt_shf(rho, L, xp)
    np.testing.assert_allclose(p, rho)
    np.testing.assert_allclose(s, [0.0, 0.0, 0.0])


def test_pnt_shf_L1_closed_form():
    """L=1: P = rho^3/(1+rho^2), S = -1/(1+rho^2)."""
    from endf_userpy.mfsec_interpretation import mf2_interpretation_factors as factors
    xp = array_ns.get_backend('numpy')
    rho = xp.asarray([0.5, 1.0, 2.0])
    L = xp.asarray([1, 1, 1])
    p, s = factors.pnt_shf(rho, L, xp)
    r2 = np.asarray([0.25, 1.0, 4.0])
    np.testing.assert_allclose(p, np.asarray([0.5, 1.0, 2.0]) * r2 / (1 + r2))
    np.testing.assert_allclose(s, -1.0 / (1 + r2))


def test_phase_L0_closed_form():
    """L=0: phi(rho) = rho."""
    from endf_userpy.mfsec_interpretation import mf2_interpretation_factors as factors
    xp = array_ns.get_backend('numpy')
    rho = xp.asarray([0.1, 1.0, 3.14])
    L = xp.asarray([0, 0, 0])
    ph = factors.phase(rho, L, xp)
    np.testing.assert_allclose(ph, rho)


def test_phase_L1_closed_form():
    """L=1: phi(rho) = rho - atan(rho)."""
    from endf_userpy.mfsec_interpretation import mf2_interpretation_factors as factors
    xp = array_ns.get_backend('numpy')
    rho = xp.asarray([0.5, 1.0, 2.0])
    L = xp.asarray([1, 1, 1])
    ph = factors.phase(rho, L, xp)
    np.testing.assert_allclose(ph, rho - np.arctan(rho))


# ============================================================
# MLBW physics: single-resonance sanity.
# ============================================================


def test_mlbw_no_resonances_gives_only_potential():
    """A range with an empty resonance table returns only the
    hard-sphere potential scattering; capture, fission,
    competitive all zero."""
    data = mlbw.MLBWData(
        abn=1.0, spi=0.5, ki=1e-5, qx=0.0,
        r_a=_constant_tab1(0.6),
        r_ap=_constant_tab1(0.6),
        ch_l=np.array([0], dtype=np.int32),
        ch_g=np.array([1.0], dtype=np.float64),
        res_channel=np.array([], dtype=np.int32),
        res_l=np.array([], dtype=np.int32),
        res_er=np.array([], dtype=np.float64),
        res_gn=np.array([], dtype=np.float64),
        res_gg=np.array([], dtype=np.float64),
        res_gf=np.array([], dtype=np.float64),
        res_gx=np.array([], dtype=np.float64),
    )
    xp = array_ns.get_backend('numpy')
    einc = np.array([1.0, 100.0, 1e4], dtype=np.float64)
    xs = mlbw.reconstruct(data, einc, xp)
    np.testing.assert_allclose(xs['cap'], 0.0, atol=1e-30)
    np.testing.assert_allclose(xs['fis'], 0.0, atol=1e-30)
    np.testing.assert_allclose(xs['rxx'], 0.0, atol=1e-30)
    assert np.all(xs['pot'] > 0)
    # Sct = pot at zero-resonance limit (A_ch = B_ch = 0). The
    # identity (1-cos)^2 + sin^2 = 2(1-cos) is exact in real
    # arithmetic but not in float64 (the two forms take different
    # numerical paths), so allow ~few * eps relative.
    np.testing.assert_allclose(xs['sct'], xs['pot'], rtol=1e-6)


def test_mlbw_capture_peaks_at_er():
    """A single s-wave resonance: capture XS must peak at Er."""
    data = _single_resonance_data(er=100.0, gn=1.0, gg=0.1)
    xp = array_ns.get_backend('numpy')
    # Fine grid around the resonance.
    einc = np.linspace(95.0, 105.0, 201)
    xs = mlbw.reconstruct(data, einc, xp)
    peak_ix = int(np.argmax(xs['cap']))
    # Grid spacing is 0.05 eV; peak should be within a couple of
    # bins of Er=100.
    assert abs(einc[peak_ix] - 100.0) < 0.2


def test_mlbw_capture_lorentzian_width():
    """For a single narrow s-wave resonance with Gamma_n << Gamma_g,
    the capture cross section is nearly Lorentzian with FWHM = Gamma.
    Check the FWHM matches Gamma to ~10% (limited by shifted E_r
    and small deviations from the pure Lorentzian at Gamma_n
    comparable to sensitivity)."""
    er = 100.0
    gn = 0.01
    gg = 1.0
    gamma_total = gn + gg  # dominant for a capture-dominated resonance
    data = _single_resonance_data(er=er, gn=gn, gg=gg)
    xp = array_ns.get_backend('numpy')
    einc = np.linspace(er - 5.0, er + 5.0, 5001)
    xs = mlbw.reconstruct(data, einc, xp)

    peak = xs['cap'].max()
    half_max_mask = xs['cap'] > peak / 2
    # Distance between first and last True index gives the FWHM in
    # units of the grid spacing.
    hi = int(np.argmax(half_max_mask[::-1]))
    lo = int(np.argmax(half_max_mask))
    fwhm = einc[len(einc) - 1 - hi] - einc[lo]
    assert abs(fwhm - gamma_total) < 0.1 * gamma_total, (
        f'FWHM {fwhm} not close to Gamma {gamma_total}'
    )


def test_mlbw_total_equals_partials():
    """`tot` field is `sct + cap + fis + rxx` by construction."""
    data = _single_resonance_data(er=50.0, gn=0.5, gg=0.3)
    xp = array_ns.get_backend('numpy')
    einc = np.linspace(30.0, 70.0, 101)
    xs = mlbw.reconstruct(data, einc, xp)
    np.testing.assert_allclose(
        xs['tot'],
        xs['sct'] + xs['cap'] + xs['fis'] + xs['rxx'],
        rtol=1e-12,
    )


# ============================================================
# Backend equivalence: numpy vs JAX must agree bit-for-bit.
# ============================================================


def _jax_available() -> bool:
    return 'jax' in array_ns.available_backends()


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_numpy_jax_agree_single_resonance():
    """The same MLBW case run through both backends must give
    numerically identical outputs. If this test ever fires it's
    a bug in the backend adapter (or in one of the low-level
    dispatch paths, e.g. `xp.select` semantics)."""
    data = _single_resonance_data(er=100.0, gn=0.5, gg=0.3)
    einc = np.linspace(95.0, 105.0, 51)
    xs_np = mlbw.reconstruct(data, einc, array_ns.get_backend('numpy'))
    xs_jax = mlbw.reconstruct(data, einc, array_ns.get_backend('jax'))
    for key in ('sct', 'cap', 'fis', 'pot', 'rxx', 'tot'):
        np.testing.assert_allclose(
            np.asarray(xs_np[key]),
            np.asarray(xs_jax[key]),
            rtol=1e-12, atol=1e-30,
            err_msg=f'numpy vs jax disagree on {key}',
        )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_numpy_jax_agree_no_resonances():
    """Zero-resonance / potential-only case: often uncovers
    edge-case backend differences in empty-array handling
    (segment_sum with num_segments > 0 but zero data points).
    """
    data = mlbw.MLBWData(
        abn=1.0, spi=0.5, ki=1e-5, qx=0.0,
        r_a=_constant_tab1(0.6),
        r_ap=_constant_tab1(0.6),
        ch_l=np.array([0], dtype=np.int32),
        ch_g=np.array([1.0], dtype=np.float64),
        res_channel=np.array([], dtype=np.int32),
        res_l=np.array([], dtype=np.int32),
        res_er=np.array([], dtype=np.float64),
        res_gn=np.array([], dtype=np.float64),
        res_gg=np.array([], dtype=np.float64),
        res_gf=np.array([], dtype=np.float64),
        res_gx=np.array([], dtype=np.float64),
    )
    einc = np.array([1.0, 100.0, 1e4], dtype=np.float64)
    xs_np = mlbw.reconstruct(data, einc, array_ns.get_backend('numpy'))
    xs_jax = mlbw.reconstruct(data, einc, array_ns.get_backend('jax'))
    for key in ('sct', 'cap', 'fis', 'pot', 'rxx', 'tot'):
        np.testing.assert_allclose(
            np.asarray(xs_np[key]),
            np.asarray(xs_jax[key]),
            rtol=1e-12, atol=1e-30,
        )
