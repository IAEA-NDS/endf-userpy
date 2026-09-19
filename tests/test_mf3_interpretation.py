"""Sketch tests for the backend-agnostic MF3 reconstruction and
for the MF2 + MF3 composition on all three backends including a
JAX sensitivity smoke test.

Same naming convention as the MLBW / RM sketch tests. When a
future PR promotes these into the production API, move them
into ``test_mf3_interpretation.py`` alongside the existing
policy-heavy tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.tab1 import TAB1
from endf_userpy.mfsec_interpretation import mf2_interpretation_mlbw as mlbw
from endf_userpy.mfsec_interpretation import mf3_interpretation as mf3


# ============================================================
# Synthetic ENDF-dict fixtures.
# ============================================================


def _synthetic_endf_dict(mt=102, e_lo=1.0, e_hi=1000.0, xs_at_lo=10.0,
                          xs_at_hi=100.0, law=2):
    """Minimal ENDF-6 dict rendering: MF3/MT/xstable with 2 points
    and one interp region."""
    return {
        3: {
            mt: {
                'xstable': {
                    'NBT': [2],       # 1-based -> becomes [1] after from_endf_dict
                    'INT': [law],
                    'E': [e_lo, e_hi],
                    'xs': [xs_at_lo, xs_at_hi],
                }
            }
        }
    }


def _synthetic_mlbw_data(er=100.0, gn=0.5, gg=0.3):
    """Single s-wave resonance, matches the MLBW sketch tests."""
    r = TAB1(
        x=np.array([1e-5, 1e10], dtype=np.float64),
        y=np.array([0.6, 0.6], dtype=np.float64),
        nbt=np.array([1], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )
    return mlbw.MLBWData(
        abn=1.0, spi=0.5, ki=1e-4, qx=0.0,
        r_a=r, r_ap=r,
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


# ============================================================
# MF3 backend-agnostic: correctness against the existing
# numpy-only compute_cross_section (below the RRR so no policy
# warnings interfere).
# ============================================================


def test_mf3_agnostic_matches_existing_compute_cross_section():
    d = _synthetic_endf_dict(mt=1)
    einc = np.array([1.0, 5.0, 100.0, 1000.0])
    ref = mf3.compute_cross_section(d, 1, einc, resonance_range='ignore')
    xp = array_ns.get_backend('numpy')
    got = np.asarray(mf3.compute_cross_section_agnostic(d, 1, einc, xp))
    # `resonance_range='ignore'` is not one of the standard policies,
    # so `compute_cross_section` would raise; run without a value:
    ref = mf3.compute_cross_section(d, 1, einc, resonance_range='warn')
    np.testing.assert_allclose(got, ref, rtol=1e-14, atol=1e-30)


def test_mf3_agnostic_extrapolation_returns_zero_default():
    d = _synthetic_endf_dict(mt=1, e_lo=1.0, e_hi=1000.0)
    xp = array_ns.get_backend('numpy')
    einc = np.array([0.5, 500.0, 2000.0])
    xs = np.asarray(mf3.compute_cross_section_agnostic(d, 1, einc, xp))
    assert xs[0] == 0.0     # below mesh
    assert xs[2] == 0.0     # above mesh
    # middle point: linear interpolation between (1, 10) and (1000, 100)
    expected = 10.0 + (500.0 - 1.0) * (100.0 - 10.0) / (1000.0 - 1.0)
    assert abs(xs[1] - expected) < 1e-12


def test_mf3_agnostic_outside_value_nan():
    d = _synthetic_endf_dict(mt=1)
    xp = array_ns.get_backend('numpy')
    einc = np.array([0.5, 2000.0])
    xs = np.asarray(mf3.compute_cross_section_agnostic(
        d, 1, einc, xp, outside_value=float('nan'),
    ))
    assert np.all(np.isnan(xs))


# ============================================================
# Backend equivalence for MF3.
# ============================================================


def _jax_available():
    return 'jax' in array_ns.available_backends()


def _numba_available():
    return 'numba' in array_ns.available_backends()


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf3_agnostic_numpy_jax_agree():
    d = _synthetic_endf_dict(mt=1)
    einc = np.linspace(1.0, 1000.0, 51)
    xs_np = np.asarray(mf3.compute_cross_section_agnostic(
        d, 1, einc, array_ns.get_backend('numpy'),
    ))
    xs_jax = np.asarray(mf3.compute_cross_section_agnostic(
        d, 1, einc, array_ns.get_backend('jax'),
    ))
    np.testing.assert_allclose(xs_np, xs_jax, rtol=1e-14, atol=1e-30)


@pytest.mark.skipif(not _numba_available(), reason='numba not installed')
def test_mf3_agnostic_numpy_numba_agree():
    d = _synthetic_endf_dict(mt=1)
    einc = np.linspace(1.0, 1000.0, 51)
    xs_np = np.asarray(mf3.compute_cross_section_agnostic(
        d, 1, einc, array_ns.get_backend('numpy'),
    ))
    # The numba backend inherits from NumpyBackend for adapter surface,
    # so this just goes through numpy under the hood -- the MF3 kernel
    # doesn't have a numba-specialized dispatch (unlike MF2 formalisms
    # which have their own @njit kernels). Still validates that
    # the code path selects and runs cleanly.
    xs_nb = np.asarray(mf3.compute_cross_section_agnostic(
        d, 1, einc, array_ns.get_backend('numba'),
    ))
    np.testing.assert_allclose(xs_np, xs_nb, rtol=1e-14, atol=1e-30)


# ============================================================
# End-to-end: MF2 (MLBW) + MF3 (background) total XS pipeline
# on all three backends, plus a JAX sensitivity smoke test.
#
# This is the "step 4" deliverable: reconstruct total XS from
# a resonance formalism combined with a tabulated background
# under a single call, then differentiate through the whole
# pipeline via jax.grad.
# ============================================================


def _total_xs(endf_dict, mlbw_data, mt_bg, energies_in, xp):
    """Combined MF3 background + MF2 MLBW resonance reconstruction.

    Total XS = (MF3 MT=mt_bg background) + (MLBW capture + scattering).

    This is what a production pipeline does: MF2 gives the
    resonance contribution, MF3 provides the smooth-varying
    background beyond / beneath the resonance region and any
    fine correction inside. Both go through the array-namespace
    adapter, so the composition runs unchanged on any backend
    and stays differentiable end-to-end on JAX.
    """
    bg = mf3.compute_cross_section_agnostic(
        endf_dict, mt_bg, energies_in, xp,
    )
    mlbw_xs = mlbw.reconstruct(mlbw_data, energies_in, xp)
    # sct + cap == elastic + capture, which is what MT=1 (total)
    # would be for a non-fissile target with rxx=0 (our test case).
    return bg + mlbw_xs['sct'] + mlbw_xs['cap']


def test_end_to_end_mf2_mf3_composition_numpy():
    """The composition runs end-to-end and returns sensible numbers:
    background at the tabulated points, resonance peak visible."""
    d = _synthetic_endf_dict(
        mt=1, e_lo=1.0, e_hi=200.0, xs_at_lo=5.0, xs_at_hi=5.0,
    )
    data = _synthetic_mlbw_data(er=100.0, gn=0.5, gg=0.3)
    xp = array_ns.get_backend('numpy')
    einc = np.linspace(80.0, 120.0, 201)
    tot = np.asarray(_total_xs(d, data, 1, einc, xp))
    # Well away from the resonance, tot ~ background + potential.
    # Near the peak (E=100), tot >> background.
    i_peak = int(np.argmax(tot))
    assert abs(einc[i_peak] - 100.0) < 0.5
    assert tot[i_peak] > 100.0             # peak dominates background
    assert tot[0] < tot[i_peak]            # away from peak is smaller


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_end_to_end_composition_backend_equivalence():
    """Total XS from the composition agrees numpy vs JAX."""
    d = _synthetic_endf_dict(mt=1)
    data = _synthetic_mlbw_data(er=100.0, gn=0.5, gg=0.3)
    einc = np.linspace(50.0, 200.0, 101)
    tot_np = np.asarray(_total_xs(d, data, 1, einc, array_ns.get_backend('numpy')))
    tot_jax = np.asarray(_total_xs(d, data, 1, einc, array_ns.get_backend('jax')))
    np.testing.assert_allclose(tot_np, tot_jax, rtol=1e-10, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_end_to_end_jax_autodiff_dtotal_dEr_matches_fd():
    """Payoff test: `jax.grad` of the total XS wrt the first
    resonance's E_r goes end-to-end through MF3 interpolation
    plus MLBW reconstruction and matches finite differences.

    A fitting workflow (adjusting E_r to match measured total XS)
    can therefore be built on top of this composition with no
    extra glue: JAX's autodiff walks the whole pipeline."""
    import jax
    import jax.numpy as jnp

    xp = array_ns.get_backend('jax')

    def total_at(er, e_query):
        d = _synthetic_endf_dict(mt=1)
        # Rebuild MLBWData with er as a JAX tracer so autodiff
        # follows it into `mlbw.reconstruct`.
        r = TAB1(
            x=np.array([1e-5, 1e10], dtype=np.float64),
            y=np.array([0.6, 0.6], dtype=np.float64),
            nbt=np.array([1], dtype=np.int32),
            intp=np.array([2], dtype=np.int32),
        )
        data = mlbw.MLBWData(
            abn=1.0, spi=0.5, ki=1e-4, qx=0.0,
            r_a=r, r_ap=r,
            ch_l=np.array([0], dtype=np.int32),
            ch_g=np.array([1.0], dtype=np.float64),
            res_channel=np.array([0], dtype=np.int32),
            res_l=np.array([0], dtype=np.int32),
            res_er=jnp.stack([er]),
            res_gn=jnp.array([0.5], dtype=jnp.float64),
            res_gg=jnp.array([0.3], dtype=jnp.float64),
            res_gf=jnp.array([0.0], dtype=jnp.float64),
            res_gx=jnp.array([0.0], dtype=jnp.float64),
        )
        return _total_xs(d, data, 1, jnp.array([e_query]), xp)[0]

    er0 = 100.0
    for e_query in (99.5, 100.5):     # a bit off the peak: nonzero slope
        grad_fn = jax.grad(total_at, argnums=0)
        dtot_dEr_ad = float(grad_fn(er0, e_query))

        h = 1e-4
        dtot_dEr_fd = (
            float(total_at(er0 + h, e_query))
            - float(total_at(er0 - h, e_query))
        ) / (2 * h)

        assert abs(dtot_dEr_ad - dtot_dEr_fd) < 1e-4 * max(
            abs(dtot_dEr_fd), 1.0,
        ), (
            f'end-to-end autodiff mismatch at E={e_query} eV: '
            f'ad={dtot_dEr_ad:.6e} fd={dtot_dEr_fd:.6e}'
        )
