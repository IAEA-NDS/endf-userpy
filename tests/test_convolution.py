"""Tests for the generic adaptive_convolve primitive.

Validation cases use closed-form convolutions so the numerical result
can be checked against an exact analytic expression independently of
any ENDF data.
"""
import numpy as np
import pytest

from endf_userpy.primitives.convolution import (
    adaptive_convolve,
    ConvergenceWarning,
)


def gaussian(x, sigma, mu=0.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def test_gaussian_x_gaussian_is_gaussian():
    """N(0, s_f) * N(0, s_k) = N(0, sqrt(s_f^2 + s_k^2)). Exact identity."""
    sigma_f, sigma_k = 0.7, 0.4
    sigma_y = np.sqrt(sigma_f ** 2 + sigma_k ** 2)
    eval_points = np.linspace(-3.0, 3.0, 31)
    expected = gaussian(eval_points, sigma_y)

    result = adaptive_convolve(
        lambda E: gaussian(E, sigma_f),
        lambda dE: gaussian(dE, sigma_k),
        eval_points,
        kernel_width=sigma_k,
        rtol=1e-6,
        max_iter=10,
    )
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_constant_function_preserved():
    """A unit-norm kernel must leave a constant function unchanged
    in the interior (boundaries are excluded by the mesh extension)."""
    eval_points = np.linspace(-2.0, 2.0, 21)
    sigma_k = 0.3
    c = 7.5
    result = adaptive_convolve(
        lambda E: np.full_like(E, c),
        lambda dE: gaussian(dE, sigma_k),
        eval_points,
        kernel_width=sigma_k,
        rtol=1e-6,
    )
    np.testing.assert_allclose(result, c, atol=1e-5)


def test_tighter_rtol_gives_higher_accuracy():
    """Convergence tolerance actually controls accuracy: requesting
    rtol=1e-6 must beat rtol=1e-2 on the Gaussian benchmark."""
    sigma_f, sigma_k = 0.5, 0.3
    sigma_y = np.sqrt(sigma_f ** 2 + sigma_k ** 2)
    eval_points = np.linspace(-2.0, 2.0, 17)
    expected = gaussian(eval_points, sigma_y)

    def run(rtol):
        return adaptive_convolve(
            lambda E: gaussian(E, sigma_f),
            lambda dE: gaussian(dE, sigma_k),
            eval_points,
            kernel_width=sigma_k,
            rtol=rtol,
            max_iter=10,
            richardson=False,
        )

    err_loose = np.max(np.abs(run(1e-2) - expected))
    err_tight = np.max(np.abs(run(1e-6) - expected))
    assert err_tight < err_loose
    assert err_tight < 1e-4


def test_richardson_improves_over_raw():
    """Richardson extrapolation must produce a smaller residual than
    the last-iteration value on a smooth analytic benchmark."""
    sigma_f, sigma_k = 0.5, 0.3
    sigma_y = np.sqrt(sigma_f ** 2 + sigma_k ** 2)
    eval_points = np.linspace(-1.5, 1.5, 11)
    expected = gaussian(eval_points, sigma_y)

    common = dict(
        f=lambda E: gaussian(E, sigma_f),
        kernel=lambda dE: gaussian(dE, sigma_k),
        eval_points=eval_points,
        kernel_width=sigma_k,
        rtol=1e-3,
        max_iter=10,
    )
    raw = adaptive_convolve(richardson=False, **common)
    rich = adaptive_convolve(richardson=True, **common)
    assert np.max(np.abs(rich - expected)) <= np.max(np.abs(raw - expected))


def test_batched_input_matches_per_slice():
    """f returning a multi-dim array convolves each slice and matches
    per-slice calls."""
    sigmas = np.array([0.4, 0.6, 0.8])
    sigma_k = 0.3
    eval_points = np.linspace(-2.0, 2.0, 21)

    def f_batched(E):
        return np.stack([gaussian(E, s) for s in sigmas], axis=0)

    batched = adaptive_convolve(
        f_batched,
        lambda dE: gaussian(dE, sigma_k),
        eval_points,
        kernel_width=sigma_k,
        rtol=1e-5,
    )

    per_slice = np.stack(
        [
            adaptive_convolve(
                lambda E, s=s: gaussian(E, s),
                lambda dE: gaussian(dE, sigma_k),
                eval_points,
                kernel_width=sigma_k,
                rtol=1e-5,
            )
            for s in sigmas
        ],
        axis=0,
    )
    # Tolerance set to a few rtols' worth: the batched call's global
    # convergence check may finish on a slightly different mesh than
    # any individual per-slice call, so we don't expect bit equality.
    np.testing.assert_allclose(batched, per_slice, rtol=1e-4, atol=1e-6)


def test_f_called_only_on_new_midpoints():
    """Doubling reuses prior evaluations: across iterations, f is
    called on the initial mesh once, then only on inserted midpoints."""
    sigma_f, sigma_k = 0.5, 0.3
    eval_points = np.linspace(-1.5, 1.5, 11)
    seen = []

    def tracking_f(E):
        seen.append(np.asarray(E).copy())
        return gaussian(E, sigma_f)

    adaptive_convolve(
        tracking_f,
        lambda dE: gaussian(dE, sigma_k),
        eval_points,
        kernel_width=sigma_k,
        rtol=1e-4,
        max_iter=10,
    )

    # The union of all calls should have no duplicates (every E is
    # evaluated exactly once).
    all_E = np.concatenate(seen)
    assert len(all_E) == len(np.unique(all_E)), (
        f"f was evaluated at {len(all_E)} points but only "
        f"{len(np.unique(all_E))} are distinct -- value reuse broken."
    )


def test_convergence_warning_when_max_iter_too_low():
    """If max_iter is too low to satisfy rtol, a ConvergenceWarning
    is emitted and a best-effort result is still returned."""
    eval_points = np.linspace(-1.0, 1.0, 11)
    with pytest.warns(ConvergenceWarning):
        result = adaptive_convolve(
            lambda E: gaussian(E, 0.5),
            lambda dE: gaussian(dE, 0.3),
            eval_points,
            kernel_width=0.3,
            rtol=1e-12,  # impossible to satisfy in two iterations
            max_iter=2,
            min_iter=2,
        )
    assert result.shape == eval_points.shape


def test_rejects_non_1d_eval_points():
    with pytest.raises(ValueError, match="1D"):
        adaptive_convolve(
            lambda E: gaussian(E, 0.5),
            lambda dE: gaussian(dE, 0.3),
            np.zeros((3, 4)),
            kernel_width=0.3,
        )


def test_rejects_non_positive_kernel_width():
    with pytest.raises(ValueError, match="positive"):
        adaptive_convolve(
            lambda E: gaussian(E, 0.5),
            lambda dE: gaussian(dE, 0.3),
            np.linspace(-1, 1, 11),
            kernel_width=0.0,
        )


def test_off_centre_eval_points():
    """Output points far from zero work the same way: the mesh tracks
    eval_points, not the origin."""
    sigma_f, sigma_k = 0.5, 0.3
    sigma_y = np.sqrt(sigma_f ** 2 + sigma_k ** 2)
    mu = 1000.0
    eval_points = np.linspace(mu - 2.0, mu + 2.0, 21)
    expected = gaussian(eval_points, sigma_y, mu=mu)

    result = adaptive_convolve(
        lambda E: gaussian(E, sigma_f, mu=mu),
        lambda dE: gaussian(dE, sigma_k),
        eval_points,
        kernel_width=sigma_k,
        rtol=1e-5,
    )
    np.testing.assert_allclose(result, expected, atol=1e-5)
