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


def test_tab1_interp_extrapolation_uses_outside_value():
    """`outside_value` sends out-of-mesh queries to that value
    instead of 0. `nan` is the flag policy used by the higher-level
    quantities API."""
    from endf_userpy.primitives import tab1 as tab1_mod
    xp = array_ns.get_backend('numpy')
    t = _constant_tab1(1.0, e_lo=1.0, e_hi=10.0)
    xs = xp.asarray([0.5, 5.0, 100.0])
    ys = tab1_mod.interp(t, xs, xp, outside_value=float('nan'))
    assert np.isnan(ys[0]) and np.isnan(ys[2])
    assert ys[1] == 1.0


def test_tab1_from_endf_dict_flat():
    """`from_endf_dict` accepts the flat layout used by many TAB1
    records in the endf_parserpy rendering."""
    from endf_userpy.primitives import tab1 as tab1_mod
    xp = array_ns.get_backend('numpy')
    d = {
        'NBT': [3],                              # 1-based -> becomes [2]
        'INT': [2],
        'E': [1.0, 2.0, 4.0],
        'xs': [10.0, 20.0, 40.0],
    }
    t = tab1_mod.from_endf_dict(d, x_key='E', y_key='xs')
    assert t.nbt.tolist() == [2]                 # subtracted 1
    ys = tab1_mod.interp(t, xp.asarray([1.0, 1.5, 4.0]), xp)
    np.testing.assert_allclose(ys, [10.0, 15.0, 40.0])


def test_tab1_from_endf_dict_wrapped_xstable():
    """`from_endf_dict` drills into an `xstable` sub-dict, matching
    what MF3 records look like straight from endf_parserpy."""
    from endf_userpy.primitives import tab1 as tab1_mod
    xp = array_ns.get_backend('numpy')
    d = {
        'xstable': {
            'NBT': [3],
            'INT': [2],
            'E': [1.0, 2.0, 4.0],
            'xs': [10.0, 20.0, 40.0],
        }
    }
    t = tab1_mod.from_endf_dict(d, x_key='E', y_key='xs')
    ys = tab1_mod.interp(t, xp.asarray([2.0]), xp)
    np.testing.assert_allclose(ys, [20.0])


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


# ============================================================
# Backend equivalence: numpy vs numba must agree numerically.
#
# Numba uses `fastmath=True`, which allows the compiler to
# reassociate float ops for FMA fusion. That legitimately relaxes
# bit-equivalence to something looser than the numpy-vs-JAX check
# but is still expected to hold at rtol=1e-10.
# ============================================================


def _numba_available() -> bool:
    return 'numba' in array_ns.available_backends()


@pytest.mark.skipif(not _numba_available(), reason='numba not installed')
def test_numpy_numba_agree_single_resonance():
    data = _single_resonance_data(er=100.0, gn=0.5, gg=0.3)
    einc = np.linspace(95.0, 105.0, 51)
    xs_np = mlbw.reconstruct(data, einc, array_ns.get_backend('numpy'))
    xs_nb = mlbw.reconstruct(data, einc, array_ns.get_backend('numba'))
    for key in ('sct', 'cap', 'fis', 'pot', 'rxx', 'tot'):
        np.testing.assert_allclose(
            np.asarray(xs_np[key]),
            np.asarray(xs_nb[key]),
            rtol=1e-10, atol=1e-30,
            err_msg=f'numpy vs numba disagree on {key}',
        )


# ============================================================
# Numpy memory-safety chunking (Item 1 of the production-readiness
# roadmap): reconstruct auto-chunks NE when the (NE, nres)
# intermediate exceeds NUMPY_MAX_INTERMEDIATE_BYTES.
# ============================================================


def test_numpy_chunk_size_helper_bounds():
    """Simple sanity on the chunk-size helper."""
    from endf_userpy.mfsec_interpretation.mf2_interpretation_mlbw import (
        _numpy_chunk_size,
    )
    # nres=100 -> 1600 bytes per row. max_bytes=16000 -> chunk=10.
    assert _numpy_chunk_size(1000, 100, 16000) == 10
    # If the whole grid fits, chunk is NE.
    assert _numpy_chunk_size(100, 100, 10 ** 9) == 100
    # Degenerate nres=0 still returns a positive chunk.
    assert _numpy_chunk_size(1000, 0, 1000) >= 1


def test_numpy_chunking_output_bit_identical_to_unchunked():
    """Force chunking by passing a tiny max-intermediate-bytes cap.
    The result must be bit-identical to the unchunked path (both
    are pure numpy arithmetic on the same energies, only the loop
    granularity differs)."""
    xp = array_ns.get_backend('numpy')
    data = _single_resonance_data(er=100.0, gn=0.5, gg=0.3)
    einc = np.linspace(95.0, 105.0, 501)
    xs_full = mlbw.reconstruct(data, einc, xp)
    # Tiny cap -> chunk every ~1 point (bytes_per_row = 1 * 16 = 16;
    # max_bytes=64 -> chunk=4).
    xs_chunked = mlbw.reconstruct(
        data, einc, xp, _max_intermediate_bytes=64,
    )
    for key in ('sct', 'cap', 'fis', 'pot', 'rxx', 'tot'):
        np.testing.assert_array_equal(
            np.asarray(xs_full[key]),
            np.asarray(xs_chunked[key]),
            err_msg=f'chunked vs unchunked disagree on {key}',
        )


def test_numpy_chunking_activates_when_grid_would_be_too_big():
    """Chunking must actually trigger when the intermediate exceeds
    the cap — otherwise the safety net is toothless. Detect by
    monkey-patching the sub-call to record how many times it was
    invoked."""
    from unittest import mock
    from endf_userpy.mfsec_interpretation import mf2_interpretation_mlbw as m
    xp = array_ns.get_backend('numpy')
    data = _single_resonance_data(er=100.0, gn=0.5, gg=0.3)
    einc = np.linspace(95.0, 105.0, 40)

    original = m.reconstruct
    call_count = {'n': 0}
    def counting(*a, **kw):
        call_count['n'] += 1
        return original(*a, **kw)

    # Fresh call with tight cap -> must recurse into sub-chunks.
    with mock.patch.object(m, 'reconstruct', counting):
        # Wire the recursion through the mock, then invoke the real.
        original(data, einc, xp, _max_intermediate_bytes=64)
    # 1 outer call + Nchunks sub-calls, each with _skip_chunk=True.
    # Chunk size = 64/16 = 4; NE=40 -> 10 sub-chunks.
    assert call_count['n'] >= 5, (
        f'expected chunking to trigger multiple sub-calls; got {call_count["n"]}'
    )


@pytest.mark.skipif(not _numba_available(), reason='numba not installed')
@pytest.mark.parametrize('L', [0, 1, 2, 3, 4, 5, 6, 7])
def test_numpy_numba_agree_all_L_single_resonance(L):
    """Numba MLBW must reproduce the numpy path for arbitrary L,
    including L>=6 (which uses the Newton recurrence rather than
    closed forms). Regression guard on the L>5 numba extension.

    Uses a single s-wave-shaped resonance and rewrites both ``res_l``
    and ``ch_l`` to the target L; ki / r_a are tuned so rho stays in
    a numerically comfortable range at the resonance energy for every
    L (larger L needs larger rho to lift the tail penetration factor
    above ``_EPS``, otherwise the width normalization gn / P_L(|E_r|)
    underflows and both backends collapse to zero in the same way,
    passing the assertion but proving nothing)."""
    # ki=0.1, r_a=5 -> rho ~ 5 at Er=100, well above the P_L
    # underflow boundary for L up to 8.
    data = _single_resonance_data(er=100.0, gn=0.5, gg=0.3,
                                  ki=1e-1, r_a=5.0, r_ap=5.0)
    data.res_l[0] = L
    data.ch_l[0] = L
    einc = np.linspace(95.0, 105.0, 51)
    xs_np = mlbw.reconstruct(data, einc, array_ns.get_backend('numpy'))
    xs_nb = mlbw.reconstruct(data, einc, array_ns.get_backend('numba'))
    for key in ('sct', 'cap', 'fis', 'pot', 'rxx', 'tot'):
        np.testing.assert_allclose(
            np.asarray(xs_np[key]),
            np.asarray(xs_nb[key]),
            rtol=1e-9, atol=1e-30,
            err_msg=f'L={L}: numpy vs numba disagree on {key}',
        )


# ============================================================
# Sensitivity: JAX autodiff through the MLBW reconstruction.
#
# The whole point of routing the physics through the array-ns
# adapter is that `jax.grad` on the JAX backend gets sensitivity
# for free. These tests validate `dsigma_cap / dE_r` against a
# finite-difference reference and demonstrate the pattern.
# ============================================================


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_autodiff_dsigma_cap_dEr_matches_finite_difference():
    """d sigma_cap / d E_r at a single query energy, computed by
    :func:`jax.grad`, agrees with a central finite difference.

    This is the sensitivity-workflow smoke test: if this ever
    fires, the differentiability of the sketch through the
    JAX backend is broken."""
    import jax
    import jax.numpy as jnp

    xp = array_ns.get_backend('jax')

    def cap_at(er_value, e_query):
        # Rebuild the dataclass with the perturbed er inside the
        # traced function so JAX sees er_value as a tracer.
        data = mlbw.MLBWData(
            abn=1.0, spi=0.5, ki=1e-4, qx=0.0,
            r_a=_constant_tab1(0.6),
            r_ap=_constant_tab1(0.6),
            ch_l=np.array([0], dtype=np.int32),
            ch_g=np.array([1.0], dtype=np.float64),
            res_channel=np.array([0], dtype=np.int32),
            res_l=np.array([0], dtype=np.int32),
            res_er=jnp.array([er_value], dtype=jnp.float64),
            res_gn=jnp.array([0.5], dtype=jnp.float64),
            res_gg=jnp.array([0.3], dtype=jnp.float64),
            res_gf=jnp.array([0.0], dtype=jnp.float64),
            res_gx=jnp.array([0.0], dtype=jnp.float64),
        )
        xs = mlbw.reconstruct(data, jnp.array([e_query]), xp)
        return xs['cap'][0]

    # Evaluate the derivative at three points around the resonance:
    # left slope, peak, right slope. All three should agree with FD.
    er_0 = 100.0
    for e_query in (99.5, 100.0, 100.5):
        # Autodiff: gradient with respect to the first argument (E_r).
        grad_fn = jax.grad(cap_at, argnums=0)
        dcap_dEr_ad = float(grad_fn(er_0, e_query))

        # Central finite difference reference.
        h = 1e-4
        dcap_dEr_fd = (
            float(cap_at(er_0 + h, e_query))
            - float(cap_at(er_0 - h, e_query))
        ) / (2 * h)

        assert abs(dcap_dEr_ad - dcap_dEr_fd) < 1e-4 * max(
            abs(dcap_dEr_fd), 1.0,
        ), (
            f'autodiff vs FD disagree at E={e_query} eV: '
            f'ad={dcap_dEr_ad:.6e} fd={dcap_dEr_fd:.6e}'
        )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_jacobian_across_widths_matches_finite_difference():
    """Full Jacobian d sigma_cap / d (E_r, Gamma_n, Gamma_g) at one
    incident energy via :func:`jax.jacrev`, validated against
    central finite differences component-by-component. This is
    what a fitting workflow actually consumes; if this passes,
    autodiff is trustworthy for MLBW."""
    import jax
    import jax.numpy as jnp

    xp = array_ns.get_backend('jax')

    def cap_scalar(theta):
        er, gn, gg = theta[0], theta[1], theta[2]
        data = mlbw.MLBWData(
            abn=1.0, spi=0.5, ki=1e-4, qx=0.0,
            r_a=_constant_tab1(0.6), r_ap=_constant_tab1(0.6),
            ch_l=np.array([0], dtype=np.int32),
            ch_g=np.array([1.0], dtype=np.float64),
            res_channel=np.array([0], dtype=np.int32),
            res_l=np.array([0], dtype=np.int32),
            res_er=jnp.stack([er]),
            res_gn=jnp.stack([gn]),
            res_gg=jnp.stack([gg]),
            res_gf=jnp.array([0.0], dtype=jnp.float64),
            res_gx=jnp.array([0.0], dtype=jnp.float64),
        )
        return mlbw.reconstruct(data, jnp.array([100.0]), xp)['cap'][0]

    theta0 = jnp.array([100.0, 0.5, 0.3], dtype=jnp.float64)
    J_ad = jax.jacrev(cap_scalar)(theta0)
    assert J_ad.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(J_ad)))

    # Component-wise central finite differences with parameter-relative
    # steps (Er ~ 100, widths ~ 1e-1, so a common absolute step would
    # be badly scaled).
    steps = jnp.array([1e-3, 1e-6, 1e-6])
    for i in range(3):
        h = steps[i]
        e_i = jnp.zeros(3).at[i].set(h)
        f_plus = float(cap_scalar(theta0 + e_i))
        f_minus = float(cap_scalar(theta0 - e_i))
        fd = (f_plus - f_minus) / (2 * float(h))
        ad = float(J_ad[i])
        rel = abs(ad - fd) / max(abs(fd), 1e-6)
        assert rel < 1e-3, (
            f'component {i}: ad={ad:.6e} fd={fd:.6e} rel={rel:.2e}'
        )
