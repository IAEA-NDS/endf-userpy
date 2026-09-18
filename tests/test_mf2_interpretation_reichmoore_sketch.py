"""Sketch tests for Reich-Moore reconstruction.

Same style as ``test_mf2_interpretation_mlbw_sketch.py``: synthetic
one-group / one-resonance cases where the answer is either analytic
or is well-known to match a simpler formalism (MLBW here).
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.tab1 import TAB1
from endf_userpy.mfsec_interpretation import mf2_interpretation_reichmoore as rm
from endf_userpy.mfsec_interpretation import mf2_interpretation_mlbw as mlbw


# ============================================================
# Helpers.
# ============================================================


def _constant_tab1(value: float) -> TAB1:
    """TAB1 with a single lin-lin panel that is effectively constant."""
    return TAB1(
        x=np.array([1e-5, 1e10], dtype=np.float64),
        y=np.array([value, value], dtype=np.float64),
        nbt=np.array([1], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )


def _single_res_elastic_capture(er=100.0, gn=0.5, gg=0.3, L=0, spi=0.5):
    """One J^π group, one resonance, no fission. Statistical weight
    g_J = 1 for simplicity (spi = 0.5, take J = 0.5 → 2J+1 = 2,
    2(2I+1) = 4 → g_J = 0.5). We override g_J to 1.0 to make the
    normalisation match a trivial MLBW test case."""
    return rm.RMData(
        abn=1.0,
        spi=spi,
        ki=1e-4,
        r_a=_constant_tab1(0.6),
        r_ap=_constant_tab1(0.6),
        group_l=np.array([L], dtype=np.int32),
        group_g=np.array([1.0], dtype=np.float64),
        group_nfis=np.array([0], dtype=np.int32),
        res_group=np.array([0], dtype=np.int32),
        res_er=np.array([er], dtype=np.float64),
        res_gn=np.array([gn], dtype=np.float64),
        res_gg=np.array([gg], dtype=np.float64),
        res_gf1=np.array([0.0], dtype=np.float64),
        res_gf2=np.array([0.0], dtype=np.float64),
    )


def _single_res_with_fission(
    er=100.0, gn=0.5, gg=0.3, gf1=0.2, L=0, spi=0.5,
):
    """One J^π group, one resonance, 1 fission channel."""
    return rm.RMData(
        abn=1.0,
        spi=spi,
        ki=1e-4,
        r_a=_constant_tab1(0.6),
        r_ap=_constant_tab1(0.6),
        group_l=np.array([L], dtype=np.int32),
        group_g=np.array([1.0], dtype=np.float64),
        group_nfis=np.array([1], dtype=np.int32),
        res_group=np.array([0], dtype=np.int32),
        res_er=np.array([er], dtype=np.float64),
        res_gn=np.array([gn], dtype=np.float64),
        res_gg=np.array([gg], dtype=np.float64),
        res_gf1=np.array([gf1], dtype=np.float64),
        res_gf2=np.array([0.0], dtype=np.float64),
    )


# ============================================================
# Physics tests.
# ============================================================


def test_rm_hard_sphere_phase_uses_scattering_radius():
    """The hard-sphere phase in Ω_c uses the SCATTERING radius R'
    (r_ap), not the channel radius a (r_a). When r_a and r_ap differ
    (NAPS=2 evaluations), sct in the no-resonance limit must be
    driven by r_ap alone: `sct == 4π/k² g_J sin²(φ_L(rho_ap))`.

    Regression: earlier versions of this sketch used r_a for the
    hard-sphere phase, which gives the wrong potential-scattering
    limit when r_a != r_ap. Since MLBW already gets this right and
    all previous RM tests happened to use r_a = r_ap = 0.6, the
    bug was invisible."""
    r_a_val, r_ap_val = 0.5, 0.9      # distinct
    data = rm.RMData(
        abn=1.0, spi=0.5, ki=1e-4,
        r_a=_constant_tab1(r_a_val),
        r_ap=_constant_tab1(r_ap_val),
        group_l=np.array([0], dtype=np.int32),
        group_g=np.array([1.0], dtype=np.float64),
        group_nfis=np.array([0], dtype=np.int32),
        res_group=np.array([], dtype=np.int32),
        res_er=np.array([], dtype=np.float64),
        res_gn=np.array([], dtype=np.float64),
        res_gg=np.array([], dtype=np.float64),
        res_gf1=np.array([], dtype=np.float64),
        res_gf2=np.array([], dtype=np.float64),
    )
    xp = array_ns.get_backend('numpy')
    einc = np.array([1.0, 100.0, 1e4], dtype=np.float64)
    xs = rm.reconstruct(data, einc, xp)
    # sct should equal pot, and both should be driven by r_ap.
    np.testing.assert_allclose(xs['sct'], xs['pot'], rtol=1e-10, atol=1e-30)
    # And the value must match the r_ap-based expectation, not the
    # r_a one (the pre-fix bug gave the latter).
    ki = data.ki
    k2 = (ki * ki) * einc
    rho_ap = ki * np.sqrt(einc) * r_ap_val
    expected = 4.0 * np.pi / k2 * np.sin(rho_ap) ** 2
    np.testing.assert_allclose(xs['sct'], expected, rtol=1e-10, atol=1e-30)


def test_rm_no_resonances_off_peak_is_potential():
    """With no resonances, elastic reduces to potential (4π/k² g_J
    sin²φ), and capture / fission are zero."""
    data = rm.RMData(
        abn=1.0, spi=0.5, ki=1e-4,
        r_a=_constant_tab1(0.6), r_ap=_constant_tab1(0.6),
        group_l=np.array([0], dtype=np.int32),
        group_g=np.array([1.0], dtype=np.float64),
        group_nfis=np.array([0], dtype=np.int32),
        res_group=np.array([], dtype=np.int32),
        res_er=np.array([], dtype=np.float64),
        res_gn=np.array([], dtype=np.float64),
        res_gg=np.array([], dtype=np.float64),
        res_gf1=np.array([], dtype=np.float64),
        res_gf2=np.array([], dtype=np.float64),
    )
    xp = array_ns.get_backend('numpy')
    einc = np.array([1.0, 100.0, 1e4], dtype=np.float64)
    xs = rm.reconstruct(data, einc, xp)
    # Cap and fis should be zero. Complex-arithmetic roundoff in the
    # unitarity check (|U|^2 ~ 1 ± 1e-8) can leak into `cap` as a tiny
    # residue; hence atol rather than array_equal.
    np.testing.assert_allclose(
        np.asarray(xs['cap']), 0.0, atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(xs['fis']), 0.0, atol=1e-30,
    )
    # sct should equal pot (elastic-only, no absorption).
    np.testing.assert_allclose(
        np.asarray(xs['sct']), np.asarray(xs['pot']),
        rtol=1e-10, atol=1e-30,
    )


def test_rm_capture_peaks_at_er():
    """Single s-wave resonance, capture cross section has its peak
    at E = E_r."""
    data = _single_res_elastic_capture(er=100.0, gn=0.5, gg=0.3)
    xp = array_ns.get_backend('numpy')
    einc = np.linspace(95.0, 105.0, 201)
    xs = rm.reconstruct(data, einc, xp)
    cap = np.asarray(xs['cap'])
    i_peak = int(np.argmax(cap))
    e_peak = einc[i_peak]
    assert abs(e_peak - 100.0) < 0.1, (
        f'capture peak at {e_peak} eV, expected ~100 eV'
    )
    # Capture must be positive at the peak (sanity).
    assert cap[i_peak] > 0.0


def test_rm_capture_positive_definite():
    """Reich-Moore capture cross section is nonnegative everywhere:
    ``σ_cap = (π/k²) g_J [1 - Σ_c |U_{0c}|²]`` is manifestly
    nonnegative from unitarity. Numerical roundoff can flip the sign
    slightly; here we check |negative| is at machine level of the peak."""
    data = _single_res_elastic_capture(er=100.0, gn=0.5, gg=0.3)
    xp = array_ns.get_backend('numpy')
    einc = np.linspace(50.0, 150.0, 501)
    xs = rm.reconstruct(data, einc, xp)
    cap = np.asarray(xs['cap'])
    peak = float(np.max(cap))
    assert np.all(cap > -1e-12 * peak), 'capture went negative beyond roundoff'


def test_rm_matches_mlbw_in_elastic_capture_limit():
    """For a single non-fission resonance, R-M and MLBW give
    nearly identical results. R-M folds Γ_γ into the complex
    denominator; MLBW keeps it real. The two agree to a few %
    near the peak for moderate widths (in the narrow-resonance
    limit exactly)."""
    xp = array_ns.get_backend('numpy')
    er, gn, gg = 100.0, 0.5, 0.3
    rm_data = _single_res_elastic_capture(er=er, gn=gn, gg=gg)
    mlbw_data = mlbw.MLBWData(
        abn=1.0, spi=0.5, ki=1e-4, qx=0.0,
        r_a=_constant_tab1(0.6), r_ap=_constant_tab1(0.6),
        ch_l=np.array([0], dtype=np.int32),
        ch_g=np.array([1.0], dtype=np.float64),
        res_channel=np.array([0], dtype=np.int32),
        res_l=np.array([0], dtype=np.int32),
        res_er=np.array([er], dtype=np.float64),
        res_gn=np.array([gn], dtype=np.float64),
        res_gg=np.array([gg], dtype=np.float64),
        res_gf=np.array([0.0], dtype=np.float64),
        res_gx=np.array([0.0], dtype=np.float64),
    )
    einc = np.linspace(95.0, 105.0, 51)
    xs_rm = rm.reconstruct(rm_data, einc, xp)
    xs_mlbw = mlbw.reconstruct(mlbw_data, einc, xp)

    cap_rm = np.asarray(xs_rm['cap'])
    cap_mlbw = np.asarray(xs_mlbw['cap'])
    peak = float(np.max(cap_mlbw))
    # 5% is a generous bound; the two formulations differ by O(Γ²/E²)
    # away from the peak. Not-so-narrow (gn+gg=0.8 eV at Er=100 eV) so
    # small percent differences are expected.
    np.testing.assert_allclose(cap_rm, cap_mlbw, rtol=0.05, atol=0.01 * peak)


def test_rm_with_fission_produces_fission_cross_section():
    """One resonance with a fission channel: fission peaks at Er,
    fission > 0."""
    data = _single_res_with_fission(er=100.0, gn=0.5, gg=0.3, gf1=0.2)
    xp = array_ns.get_backend('numpy')
    einc = np.linspace(95.0, 105.0, 201)
    xs = rm.reconstruct(data, einc, xp)
    fis = np.asarray(xs['fis'])
    i_peak = int(np.argmax(fis))
    assert abs(einc[i_peak] - 100.0) < 0.1
    assert fis[i_peak] > 0.0
    # Total should equal sct + cap + fis identically.
    np.testing.assert_allclose(
        np.asarray(xs['tot']),
        np.asarray(xs['sct']) + np.asarray(xs['cap']) + np.asarray(xs['fis']),
        rtol=1e-14, atol=1e-30,
    )


def test_rm_unitarity_bound():
    """R-M cross sections respect the unitary bound:
    ``σ_scat + σ_cap + σ_fis <= (4π / k²) Σ_J g_J`` per J·π group
    (up to |1-U_00|² <= 4 which saturates at π/k² per group).
    Here we do a coarser check: total ≤ 5 × the unitary bound to
    catch gross sign / factor errors."""
    data = _single_res_with_fission(er=100.0, gn=0.5, gg=0.3, gf1=0.2)
    xp = array_ns.get_backend('numpy')
    einc = np.linspace(50.0, 150.0, 501)
    xs = rm.reconstruct(data, einc, xp)
    tot = np.asarray(xs['tot'])
    ki = data.ki
    e_safe = np.maximum(einc, 0.0)
    k2 = (ki * ki) * e_safe
    # 4π/k² per group with g_J = 1
    bound = 4.0 * np.pi / np.where(k2 > 0, k2, 1e30)
    assert np.all(tot <= 5.0 * bound + 1e-6)


# ============================================================
# Backend equivalence.
# ============================================================


def _jax_available() -> bool:
    return 'jax' in array_ns.available_backends()


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_numpy_jax_agree_single_resonance():
    data = _single_res_elastic_capture(er=100.0, gn=0.5, gg=0.3)
    einc = np.linspace(95.0, 105.0, 51)
    xs_np = rm.reconstruct(data, einc, array_ns.get_backend('numpy'))
    xs_jax = rm.reconstruct(data, einc, array_ns.get_backend('jax'))
    for key in ('sct', 'cap', 'fis', 'pot', 'tot'):
        np.testing.assert_allclose(
            np.asarray(xs_np[key]),
            np.asarray(xs_jax[key]),
            rtol=1e-10, atol=1e-30,
            err_msg=f'numpy vs jax disagree on {key}',
        )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_numpy_jax_agree_with_fission():
    data = _single_res_with_fission(er=100.0, gn=0.5, gg=0.3, gf1=0.2)
    einc = np.linspace(95.0, 105.0, 51)
    xs_np = rm.reconstruct(data, einc, array_ns.get_backend('numpy'))
    xs_jax = rm.reconstruct(data, einc, array_ns.get_backend('jax'))
    for key in ('sct', 'cap', 'fis', 'pot', 'tot'):
        np.testing.assert_allclose(
            np.asarray(xs_np[key]),
            np.asarray(xs_jax[key]),
            rtol=1e-10, atol=1e-30,
        )
