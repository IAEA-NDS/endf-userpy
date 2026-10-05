"""Regression tests for adaptive_convolve chunking (#306 mesh
follow-up): verify that chunk_size is a numerical no-op and that
leading-axis chunking produces the same output as the un-chunked
path."""
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
    the 1D integrand by a leading-axis weight. Lets us probe
    leading-axis chunking."""
    base = _lorentz_peaks_1d()
    weights = np.array([1.0, 0.7, 1.3, 0.5, 2.0, 0.1, 1.5])  # 7 rows

    def f(E):
        inner = base(E)
        return weights[:, None] * inner[None, :]
    return f


def _lorentz_peaks_3d():
    """Returns ``f`` producing shape ``(A, B, n_mesh)`` so the
    chunking path over flattened leading axes runs."""
    base = _lorentz_peaks_1d()
    A, B = 3, 5
    rng = np.random.default_rng(42)
    weights = rng.uniform(0.5, 2.0, size=(A, B))

    def f(E):
        inner = base(E)
        return weights[..., None] * inner[None, None, :]
    return f


@pytest.mark.parametrize('chunk_size', [None, 32, 128, 1024])
def test_f_eval_chunking_is_numerical_noop_1d(chunk_size):
    """Chunking the mesh evaluation along the mesh axis must not
    change results beyond numeric roundoff."""
    f = _lorentz_peaks_1d()
    sigma = 0.5
    kernel = _gauss(sigma)
    eval_pts = np.linspace(0.0, 15.0, 40)
    ref = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma, chunk_size=None,
    )
    got = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma, chunk_size=chunk_size,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('chunk_size', [None, 2, 4, 32])
def test_leading_axis_chunking_is_numerical_noop_2d(chunk_size):
    """Chunking the flattened leading axes before fftconvolve must
    not change results beyond numeric roundoff."""
    f = _lorentz_peaks_2d()
    sigma = 0.5
    kernel = _gauss(sigma)
    eval_pts = np.linspace(0.0, 15.0, 40)
    ref = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma, chunk_size=None,
    )
    got = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma, chunk_size=chunk_size,
    )
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('chunk_size', [None, 2, 7, 128])
def test_leading_axis_chunking_is_numerical_noop_3d(chunk_size):
    """Flattened-leading chunking must preserve the output shape
    and value for a 3D ``f`` (two leading axes)."""
    f = _lorentz_peaks_3d()
    sigma = 0.5
    kernel = _gauss(sigma)
    eval_pts = np.linspace(0.0, 15.0, 40)
    ref = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma, chunk_size=None,
    )
    got = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma, chunk_size=chunk_size,
    )
    assert got.shape == ref.shape == (3, 5, 40)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_chunk_size_none_matches_single_block():
    """chunk_size=None (disabled) must give the same answer as a
    very large chunk_size that fits everything in one block."""
    f = _lorentz_peaks_2d()
    sigma = 0.5
    kernel = _gauss(sigma)
    eval_pts = np.linspace(0.0, 15.0, 40)
    disabled = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma, chunk_size=None,
    )
    one_block = adaptive_convolve(
        f, kernel, eval_pts, kernel_width=sigma, chunk_size=10 ** 9,
    )
    np.testing.assert_array_equal(disabled, one_block)


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
