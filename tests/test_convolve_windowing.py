"""Regression tests for adaptive_convolve windowing (#306 mesh
follow-up)."""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives.convolution import adaptive_convolve


def _gauss(sigma):
    inv = 1.0 / sigma
    norm = inv / np.sqrt(2 * np.pi)

    def kernel(delta):
        return norm * np.exp(-0.5 * (delta * inv) ** 2)
    return kernel


def _lorentz_peaks_1d():
    # Narrow features that stress adaptive_convolve's mesh refinement.
    centers = np.array([1.0, 3.5, 7.0, 12.0])
    widths = np.array([0.05, 0.08, 0.12, 0.03])

    def f(E):
        E = np.asarray(E)
        out = np.zeros_like(E)
        for c, w in zip(centers, widths):
            out = out + (w / np.pi) / ((E - c) ** 2 + w ** 2)
        return out
    return f


def _lorentz_peaks_2d():
    """Returns ``f`` producing shape ``(n_lead, n_mesh)`` by scaling
    the 1D integrand by a leading-axis weight."""
    base = _lorentz_peaks_1d()
    weights = np.array([1.0, 0.7, 1.3, 0.5, 2.0, 0.1, 1.5])  # 7 rows

    def f(E):
        inner = base(E)
        return weights[:, None] * inner[None, :]
    return f


# ----------------------------------------------------------------------
# Windowed partition tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize('wkw', [5.0, 15.0, 50.0, 1.0e9])
def test_windowed_matches_single_window_1d(wkw):
    """Point-centric windowed partition must reproduce the single-
    window result for a 1D ``f``, for a range of window spans
    including the single-window-fallback case (very large span)."""
    f = _lorentz_peaks_1d()
    sigma = 0.5
    kernel = _gauss(sigma)
    eval_pts = np.linspace(0.0, 15.0, 40)
    ref = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=1.0e9,
    )
    got = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=wkw,
    )
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=1e-6)


@pytest.mark.parametrize('wkw', [5.0, 15.0, 50.0])
def test_windowed_matches_single_window_2d(wkw):
    """Point-centric windowed partition must reproduce the single-
    window result for a 2D ``f``."""
    f = _lorentz_peaks_2d()
    sigma = 0.5
    kernel = _gauss(sigma)
    eval_pts = np.linspace(0.0, 15.0, 40)
    ref = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=1.0e9,
    )
    got = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=wkw,
    )
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=1e-6)


def test_scattered_eval_points_merge_to_few_segments():
    """Scattered eval points far apart (>> margin) must each live in
    their own merged segment; result must still match the single-
    window baseline."""
    f = _lorentz_peaks_1d()
    sigma = 0.5
    kernel = _gauss(sigma)
    # Four well-separated points: distance >> 5*sigma = 2.5
    eval_pts = np.array([1.2, 6.5, 11.5, 14.0])
    ref = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=1.0e9,
    )
    got = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=5.0,  # 2.5 energy units per window
    )
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=1e-6)


def test_clustered_eval_points_merge_to_single_segment():
    """Densely clustered eval points (spacing < 2*margin) must
    coalesce into one segment."""
    f = _lorentz_peaks_1d()
    sigma = 0.5
    kernel = _gauss(sigma)
    # Dense cluster of 10 points within 1 energy unit
    eval_pts = np.linspace(5.0, 6.0, 10)
    ref = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=1.0e9,
    )
    got = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=5.0,
    )
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=1e-6)


def test_single_eval_point_1d():
    """One eval point: trivial single-segment path."""
    f = _lorentz_peaks_1d()
    sigma = 0.5
    kernel = _gauss(sigma)
    eval_pts = np.array([3.5])
    ref = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=1.0e9,
    )
    got = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=5.0,
    )
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=1e-6)


def test_unsorted_eval_points_raise_without_mesh_bounds():
    """Concrete, unsorted eval_points without mesh_bounds must raise
    helpfully; the point-centric merge is only correct for sorted
    inputs."""
    f = _lorentz_peaks_1d()
    sigma = 0.5
    kernel = _gauss(sigma)
    eval_pts = np.array([3.0, 1.0, 5.0, 2.0])
    with pytest.raises(ValueError, match='sorted'):
        adaptive_convolve(
            f, kernel, eval_pts, kernel_width=sigma,
        )


def test_fixed_window_path_with_mesh_bounds():
    """Explicit mesh_bounds triggers the fixed-grid partition path
    (same code path tracer uses). Must reproduce the single-window
    baseline."""
    f = _lorentz_peaks_2d()
    sigma = 0.5
    kernel = _gauss(sigma)
    eval_pts = np.linspace(0.0, 15.0, 40)
    margin = 5.0 * sigma
    ref = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=1.0e9,
    )
    got = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        mesh_bounds=(float(eval_pts.min()) - margin,
                     float(eval_pts.max()) + margin),
        window_kernel_widths=15.0,
    )
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=1e-6)


# ----------------------------------------------------------------------
# xp.scan-based windowing (jit-friendly path)
# ----------------------------------------------------------------------
#
# The fixed-grid path routes through ``xp.scan`` so the window body
# traces once under jax jit. These tests verify the scan path is a
# numerical no-op on the numpy backend (where xp.scan is a Python
# for-loop) and parity-matches the single-window baseline on jax
# eager + jax.jit.


def test_scan_path_matches_single_window_padded():
    """n_eval not divisible by per_window forces trailing padding;
    the stripped output must still match the single-window baseline
    (1D f)."""
    f = _lorentz_peaks_1d()
    sigma = 0.5
    kernel = _gauss(sigma)
    eval_pts = np.linspace(0.0, 15.0, 37)  # 37 is prime; forces padding
    margin = 5.0 * sigma
    ref = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=1.0e9,
    )
    got = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        mesh_bounds=(float(eval_pts.min()) - margin,
                     float(eval_pts.max()) + margin),
        window_kernel_widths=10.0,
    )
    assert got.shape == ref.shape == (37,)
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=1e-6)


def test_scan_path_matches_single_window_2d_padded():
    """2D ``f`` + trailing padding + scan path parity."""
    f = _lorentz_peaks_2d()
    sigma = 0.5
    kernel = _gauss(sigma)
    eval_pts = np.linspace(0.0, 15.0, 37)
    margin = 5.0 * sigma
    ref = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=1.0e9,
    )
    got = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma,
        mesh_bounds=(float(eval_pts.min()) - margin,
                     float(eval_pts.max()) + margin),
        window_kernel_widths=10.0,
    )
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=1e-6)


def test_scan_path_jax_eager_parity():
    """Under eager jax, the scan path uses lax.scan; result must
    match numpy (single-window)."""
    try:
        from endf_userpy.primitives import array_ns
        xp_jax = array_ns.get_backend('jax')
    except (ImportError, ValueError):
        pytest.skip('jax not installed')

    sigma = 0.5
    kernel = _gauss(sigma)
    # Use a scalar-returning f that works with jax arrays
    centers = np.array([1.0, 3.5, 7.0, 12.0])
    widths = np.array([0.05, 0.08, 0.12, 0.03])
    centers_j = xp_jax.asarray(centers)
    widths_j = xp_jax.asarray(widths)

    def f_jax(E):
        out = xp_jax.zeros_like(E)
        for c, w in zip(centers_j, widths_j):
            out = out + (w / xp_jax.pi) / ((E - c) ** 2 + w ** 2)
        return out

    def f_np(E):
        E = np.asarray(E)
        out = np.zeros_like(E)
        for c, w in zip(centers, widths):
            out = out + (w / np.pi) / ((E - c) ** 2 + w ** 2)
        return out

    eval_pts = np.linspace(0.0, 15.0, 40)
    margin = 5.0 * sigma

    ref = adaptive_convolve(
        f_np, kernel, eval_pts, kernel_width=sigma,
        window_kernel_widths=1.0e9,
    )
    got = adaptive_convolve(
        f_jax, kernel, eval_pts, kernel_width=sigma,
        mesh_bounds=(float(eval_pts.min()) - margin,
                     float(eval_pts.max()) + margin),
        window_kernel_widths=15.0,
        xp=xp_jax,
    )
    got_np = np.asarray(got)
    assert got_np.shape == ref.shape
    np.testing.assert_allclose(got_np, ref, rtol=1e-2, atol=1e-5)


def test_scan_path_jax_jit_compiles():
    """Under @jax.jit with tracer eval_points and explicit windowing
    kwargs, the scan path must trace+compile in bounded time and
    produce the same output as numpy single-window."""
    try:
        import jax
        from endf_userpy.primitives import array_ns
        xp_jax = array_ns.get_backend('jax')
    except (ImportError, ValueError):
        pytest.skip('jax not installed')

    sigma = 0.5
    kernel = _gauss(sigma)
    centers_j = xp_jax.asarray([1.0, 3.5, 7.0, 12.0])
    widths_j = xp_jax.asarray([0.05, 0.08, 0.12, 0.03])

    def f_jax(E):
        out = xp_jax.zeros_like(E)
        for c, w in zip(centers_j, widths_j):
            out = out + (w / xp_jax.pi) / ((E - c) ** 2 + w ** 2)
        return out

    margin = 5.0 * sigma
    eval_pts_np = np.linspace(0.0, 15.0, 40)
    mesh_bounds = (float(eval_pts_np.min()) - margin,
                   float(eval_pts_np.max()) + margin)

    def _call(eval_pts_t):
        return adaptive_convolve(
            f_jax, kernel, eval_pts_t, kernel_width=sigma,
            mesh_bounds=mesh_bounds,
            window_kernel_widths=15.0,
            xp=xp_jax,
        )

    import time
    jitted = jax.jit(_call)
    eval_pts_j = xp_jax.asarray(eval_pts_np)
    t0 = time.perf_counter()
    out = jitted(eval_pts_j)
    jax.block_until_ready(out)
    t_compile = time.perf_counter() - t0
    # Compile under scan should be seconds, not many minutes. Give
    # a generous ceiling of 60s so the test is stable across
    # machines but still catches a trace-time explosion.
    assert t_compile < 60.0, (
        f'jit compile under scan took {t_compile:.1f}s; something '
        f'is unrolling W windows at trace time'
    )

    # Steady-state call
    t0 = time.perf_counter()
    out2 = jitted(eval_pts_j)
    jax.block_until_ready(out2)
    t_steady = time.perf_counter() - t0
    # Steady should be much less than compile
    assert t_steady < t_compile

    # Correctness: compare to numpy single-window baseline
    def f_np(E):
        E = np.asarray(E)
        out = np.zeros_like(E)
        for c, w in zip([1.0, 3.5, 7.0, 12.0], [0.05, 0.08, 0.12, 0.03]):
            out = out + (w / np.pi) / ((E - c) ** 2 + w ** 2)
        return out
    ref = adaptive_convolve(
        f_np, kernel, eval_pts_np, kernel_width=sigma,
        window_kernel_widths=1.0e9,
    )
    got = np.asarray(out)
    np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-5)
