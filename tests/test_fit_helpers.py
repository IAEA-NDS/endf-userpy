"""Sketch tests for :func:`endf_userpy.primitives.fit_helpers.chunked_chi2`.

Covers:

- Numerical correctness across chunk sizes and against a plain
  numpy summation.
- Backend equivalence numpy vs JAX vs numba (all three should give
  the same scalar loss).
- End-to-end autodiff: ``jax.grad(chunked_chi2)(theta)`` matches
  central finite differences for a small MLBW-shaped predict_fn.

The bounded-memory behaviour under ``jax.grad`` cannot be asserted
inside a pytest without a subprocess and platform-specific memory
measurement; that finding is documented in the module docstring and
in the scratchpad benchmark scripts.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.fit_helpers import chunked_chi2
from endf_userpy.primitives.tab1 import TAB1
from endf_userpy.mfsec_interpretation import mf2_interpretation_mlbw as mlbw


# ============================================================
# A tiny synthetic problem used across the tests.
# ============================================================


def _predict_line(theta, e_slice):
    """Toy predict_fn: theta = (slope, intercept), sigma = slope * E + intercept."""
    return theta[0] * e_slice + theta[1]


def _synthetic_line_measurement(theta_true=(0.5, 2.0), N=1000, rng_seed=0):
    energies = np.linspace(0.1, 100.0, N)
    y_meas = theta_true[0] * energies + theta_true[1]
    rng = np.random.default_rng(rng_seed)
    y_meas = y_meas + rng.normal(0.0, 0.05, N)
    y_err = np.full(N, 0.05)
    return energies, y_meas, y_err


# ============================================================
# Correctness: chunked matches plain numpy.
# ============================================================


def test_chunked_matches_direct_numpy_sum():
    """Loss computed via chunked_chi2 must equal the direct
    (sigma - y_meas)/err summation to floating precision."""
    xp = array_ns.get_backend('numpy')
    theta = np.array([0.5, 2.0])
    energies, y_meas, y_err = _synthetic_line_measurement()
    sigma_direct = _predict_line(theta, energies)
    loss_direct = float(np.sum(((sigma_direct - y_meas) / y_err) ** 2))
    for chunk in (10, 100, 333, 1000, 1001):
        loss_chunked = float(chunked_chi2(
            _predict_line, theta, energies, y_meas, y_err, xp, chunk=chunk,
        ))
        assert abs(loss_chunked - loss_direct) < 1e-8 * max(loss_direct, 1.0), (
            f'chunk={chunk}: loss={loss_chunked} vs direct={loss_direct}'
        )


def test_chunked_handles_indivisible_length():
    """A chunk size that does not divide N should still give the
    correct total via the tail-chunk path."""
    xp = array_ns.get_backend('numpy')
    theta = np.array([0.5, 2.0])
    # Length 1000, chunk 333 -> 3 full + 1 tail of 1.
    energies, y_meas, y_err = _synthetic_line_measurement(N=1000)
    loss = float(chunked_chi2(
        _predict_line, theta, energies, y_meas, y_err, xp, chunk=333,
    ))
    sigma_direct = _predict_line(theta, energies)
    ref = float(np.sum(((sigma_direct - y_meas) / y_err) ** 2))
    assert abs(loss - ref) < 1e-8 * max(ref, 1.0)


def test_chunked_rejects_nonpositive_chunk():
    xp = array_ns.get_backend('numpy')
    theta = np.array([0.5, 2.0])
    energies, y_meas, y_err = _synthetic_line_measurement()
    with pytest.raises(ValueError, match='chunk'):
        chunked_chi2(
            _predict_line, theta, energies, y_meas, y_err, xp, chunk=0,
        )


# ============================================================
# Backend equivalence.
# ============================================================


def _jax_available():
    return 'jax' in array_ns.available_backends()


def _numba_available():
    return 'numba' in array_ns.available_backends()


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_numpy_jax_agree_on_toy_loss():
    # Instantiating the JAX backend first flips jax_enable_x64 on, so
    # `jnp.asarray(np.float64_array)` keeps float64 precision.
    xp_jax = array_ns.get_backend('jax')
    import jax.numpy as jnp
    theta_np = np.array([0.5, 2.0])
    energies, y_meas, y_err = _synthetic_line_measurement(N=500)
    loss_np = float(chunked_chi2(
        _predict_line, theta_np, energies, y_meas, y_err,
        array_ns.get_backend('numpy'), chunk=100,
    ))
    loss_jax = float(chunked_chi2(
        _predict_line,
        jnp.asarray(theta_np, dtype=jnp.float64),
        jnp.asarray(energies, dtype=jnp.float64),
        jnp.asarray(y_meas, dtype=jnp.float64),
        jnp.asarray(y_err, dtype=jnp.float64),
        xp_jax, chunk=100,
    ))
    assert abs(loss_np - loss_jax) < 1e-10 * max(loss_np, 1.0)


@pytest.mark.skipif(not _numba_available(), reason='numba not installed')
def test_numpy_numba_agree_on_toy_loss():
    theta = np.array([0.5, 2.0])
    energies, y_meas, y_err = _synthetic_line_measurement(N=500)
    loss_np = float(chunked_chi2(
        _predict_line, theta, energies, y_meas, y_err,
        array_ns.get_backend('numpy'), chunk=100,
    ))
    loss_nb = float(chunked_chi2(
        _predict_line, theta, energies, y_meas, y_err,
        array_ns.get_backend('numba'), chunk=100,
    ))
    assert abs(loss_np - loss_nb) < 1e-14 * max(loss_np, 1.0)


# ============================================================
# End-to-end autodiff: jax.grad(chunked_chi2) == finite differences.
# ============================================================


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_matches_finite_difference_on_toy():
    """gradient via jax.grad through chunked_chi2 agrees with a
    central FD reference."""
    import jax
    import jax.numpy as jnp

    xp = array_ns.get_backend('jax')
    theta0 = jnp.array([0.5, 2.0], dtype=jnp.float64)
    energies, y_meas, y_err = _synthetic_line_measurement(N=1000, rng_seed=1)
    energies = jnp.asarray(energies); y_meas = jnp.asarray(y_meas)
    y_err = jnp.asarray(y_err)

    def loss(th):
        return chunked_chi2(
            _predict_line, th, energies, y_meas, y_err, xp, chunk=333,
        )

    grad_ad = jax.grad(loss)(theta0)
    grad_fd = np.zeros(2)
    h = 1e-5
    for i in range(2):
        ei = jnp.zeros(2).at[i].set(h)
        grad_fd[i] = (float(loss(theta0 + ei))
                       - float(loss(theta0 - ei))) / (2 * h)
    for i in range(2):
        rel = abs(float(grad_ad[i]) - grad_fd[i]) / max(abs(grad_fd[i]), 1.0)
        assert rel < 1e-4, f'component {i}: ad={grad_ad[i]} fd={grad_fd[i]}'


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_end_to_end_through_mlbw_reconstruction():
    """Real payoff test: jax.grad through chunked_chi2 wrapped around
    an MLBW reconstruction, validated against finite differences on
    dchi2/dEr. This is the shape a real resonance fit would take."""
    import jax
    import jax.numpy as jnp

    xp = array_ns.get_backend('jax')
    er_true = 100.0

    r_tab = TAB1(
        x=np.array([1e-5, 1e10], dtype=np.float64),
        y=np.array([0.6, 0.6], dtype=np.float64),
        nbt=np.array([1], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )

    def build_data(theta):
        return mlbw.MLBWData(
            abn=1.0, spi=0.5, ki=1e-4, qx=0.0,
            r_a=r_tab, r_ap=r_tab,
            ch_l=np.array([0], dtype=np.int32),
            ch_g=np.array([1.0], dtype=np.float64),
            res_channel=np.array([0], dtype=np.int32),
            res_l=np.array([0], dtype=np.int32),
            res_er=jnp.stack([theta[0]]),
            res_gn=jnp.array([0.5], dtype=jnp.float64),
            res_gg=jnp.array([0.3], dtype=jnp.float64),
            res_gf=jnp.array([0.0], dtype=jnp.float64),
            res_gx=jnp.array([0.0], dtype=jnp.float64),
        )

    def predict_mlbw_cap(theta, e_slice):
        data = build_data(theta)
        return mlbw.reconstruct(data, e_slice, xp)['cap']

    # Synthetic measurement at the true Er.
    energies = np.linspace(95.0, 105.0, 200)
    y_meas = np.asarray(predict_mlbw_cap(jnp.array([er_true]),
                                         jnp.asarray(energies)))
    y_err = np.maximum(y_meas * 0.01, 1e-6)

    theta0 = jnp.array([er_true + 0.1], dtype=jnp.float64)
    energies_j = jnp.asarray(energies)
    y_meas_j = jnp.asarray(y_meas)
    y_err_j = jnp.asarray(y_err)

    def loss(th):
        return chunked_chi2(
            predict_mlbw_cap, th, energies_j, y_meas_j, y_err_j,
            xp, chunk=50,
        )

    grad_ad = float(jax.grad(loss)(theta0)[0])
    h = 1e-4
    ep = jnp.array([h])
    grad_fd = (float(loss(theta0 + ep))
                - float(loss(theta0 - ep))) / (2 * h)
    rel = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1.0)
    assert rel < 1e-3, f'ad={grad_ad} fd={grad_fd}'
