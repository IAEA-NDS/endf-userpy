"""Unit tests for :func:`endf_userpy.primitives.interpolation.integrate_tab1_panels`.

Panel-exact closed-form integrals verified against analytic
integrals for each ENDF INT code.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.interpolation import integrate_tab1_panels


def _run(x0, x1, y0, y1, x_end, code, backend='numpy'):
    xp = array_ns.get_backend(backend)
    return float(integrate_tab1_panels(
        np.array([x0], dtype=float),
        np.array([x1], dtype=float),
        np.array([y0], dtype=float),
        np.array([y1], dtype=float),
        np.array([x_end], dtype=float),
        np.array([code], dtype=int),
        xp=xp,
    )[0])


def test_int1_histogram_full_panel():
    """INT=1: y = y0 constant on [x0, x1). Integral = y0*(x1-x0)."""
    assert _run(1.0, 3.0, 5.0, 999.0, 3.0, code=1) == pytest.approx(10.0)


def test_int1_histogram_truncated():
    """INT=1 truncated at x_end < x1. Value stays y0."""
    assert _run(1.0, 3.0, 5.0, 999.0, 2.5, code=1) == pytest.approx(7.5)


def test_int2_lin_lin_full_panel():
    """INT=2: y linear from (0, 0) to (2, 4). Integral = 0.5*2*4 = 4."""
    assert _run(0.0, 2.0, 0.0, 4.0, 2.0, code=2) == pytest.approx(4.0)


def test_int2_lin_lin_truncated():
    """INT=2 truncated: y at x=1 is 2 -> trapezoid 0.5*(0+2)*1 = 1."""
    assert _run(0.0, 2.0, 0.0, 4.0, 1.0, code=2) == pytest.approx(1.0)


def test_int3_lin_log_analytic():
    """INT=3: y(x) = 1 + log(x) on [1, e]. Analytic integral = e."""
    got = _run(1.0, np.e, 1.0, 2.0, np.e, code=3)
    assert got == pytest.approx(np.e, rel=1e-12)


def test_int3_lin_log_truncated_analytic():
    """INT=3 truncated to x_end < x1: y(x) = 1 + log(x)/log(e) on
    [1, e]. Analytic ∫_1^{x_end} = x_end*log(x_end) - x_end + 1 +
    (x_end - 1) = x_end*log(x_end). Check at x_end = sqrt(e)."""
    x_end = np.sqrt(np.e)
    got = _run(1.0, np.e, 1.0, 2.0, x_end, code=3)
    # y(x) = 1 + log(x); ∫_1^{xe} (1 + log(x)) dx = xe + xe*log(xe) - xe = xe*log(xe)
    # For xe = sqrt(e): xe * log(xe) = sqrt(e) * 0.5
    expected = x_end * 0.5
    assert got == pytest.approx(expected, rel=1e-12)


def test_int4_log_lin_analytic():
    """INT=4: y(x) = exp(x-1) on [1, 2]. y0=1, y1=e. Analytic = e-1."""
    got = _run(1.0, 2.0, 1.0, np.e, 2.0, code=4)
    assert got == pytest.approx(np.e - 1.0, rel=1e-12)


def test_int4_log_lin_alpha_zero_limit():
    """INT=4 with y0 == y1: alpha = 0 limit is y0*(x_end-x0)."""
    got = _run(1.0, 3.0, 2.0, 2.0, 3.0, code=4)
    assert got == pytest.approx(4.0, rel=1e-12)


def test_int5_log_log_analytic():
    """INT=5: y(x) = x^2 on [1, 2]. Analytic ∫ = 7/3."""
    got = _run(1.0, 2.0, 1.0, 4.0, 2.0, code=5)
    assert got == pytest.approx(7.0 / 3.0, rel=1e-12)


def test_int5_log_log_alpha_minus_one_limit():
    """INT=5 with alpha = -1: y(x) = y0*x0/x. Analytic ∫ = y0*x0*log(x_end/x0)."""
    # y0=2, x0=1, y1=1, x1=2 -> alpha = log(1/2)/log(2/1) = -1.
    got = _run(1.0, 2.0, 2.0, 1.0, 2.0, code=5)
    expected = 2.0 * 1.0 * np.log(2.0 / 1.0)
    assert got == pytest.approx(expected, rel=1e-12)


def test_int5_log_log_truncated():
    """INT=5: y(x) = x^2, truncated to x_end=1.5. ∫_1^{1.5} x^2 = 1.5^3/3 - 1/3."""
    got = _run(1.0, 2.0, 1.0, 4.0, 1.5, code=5)
    expected = (1.5 ** 3 - 1.0) / 3.0
    assert got == pytest.approx(expected, rel=1e-12)


def test_int3_x0_zero_falls_back_to_lin_lin():
    """INT=3 with x0 == 0 (log-undefined) falls back to lin-lin trapezoid."""
    got = _run(0.0, 2.0, 0.0, 4.0, 2.0, code=3)
    assert got == pytest.approx(4.0, rel=1e-12)  # matches lin-lin


def test_int4_y_zero_falls_back_to_lin_lin():
    """INT=4 with y0 == 0 (log-undefined) falls back to lin-lin."""
    got = _run(1.0, 2.0, 0.0, 2.0, 2.0, code=4)
    assert got == pytest.approx(1.0, rel=1e-12)  # lin-lin: 0.5*(0+2)*1


def test_zero_width_panel_returns_zero():
    """Zero-width panel (x0 == x1) returns 0 for any INT."""
    for code in (1, 2, 3, 4, 5):
        assert _run(1.0, 1.0, 5.0, 5.0, 1.0, code=code) == 0.0


@pytest.mark.skipif(
    'jax' not in array_ns.available_backends(), reason='jax not installed',
)
def test_jax_grad_wrt_y0_matches_analytic():
    """jax.grad wrt y0 through the panel-exact integrator matches
    the analytic derivative: for INT=2, d(area)/d(y0) = 0.5 * dx."""
    import jax
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')

    def loss(y0):
        return integrate_tab1_panels(
            jnp.array([0.0]), jnp.array([2.0]),
            jnp.array([y0]), jnp.array([4.0]),
            jnp.array([2.0]),
            np.array([2], dtype=int),
            xp=xp,
        ).sum()

    grad = float(jax.grad(loss)(jnp.array(1.0)))
    # area = 0.5 * (y0 + y1) * (x1 - x0) = 0.5 * (y0 + 4) * 2 = y0 + 4
    # d/d y0 = 1.0
    assert grad == pytest.approx(1.0, rel=1e-12)
