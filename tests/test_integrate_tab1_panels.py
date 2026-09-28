"""Unit tests for :func:`endf_userpy.primitives.interpolation.integrate_tab1_panels`.

Panel-exact closed-form integrals verified against analytic
integrals for each ENDF INT code.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.interpolation import integrate_tab1_panels


def _run(x0, x1, y0, y1, x_end, code, backend='numpy', x_start=None):
    xp = array_ns.get_backend(backend)
    if x_start is None:
        x_start = x0
    return float(integrate_tab1_panels(
        np.array([x0], dtype=float),
        np.array([x1], dtype=float),
        np.array([y0], dtype=float),
        np.array([y1], dtype=float),
        np.array([x_start], dtype=float),
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


def test_int2_lin_lin_interior_slice():
    """Both endpoints strictly interior: linear from (0, 0) to
    (2, 4). Integral of y=2x from x=0.5 to x=1.5 is [x^2] = 2.
    """
    assert _run(0.0, 2.0, 0.0, 4.0, x_start=0.5, x_end=1.5, code=2) == pytest.approx(
        2.0, rel=1e-12,
    )


def test_int1_histogram_interior_slice():
    """INT=1 histogram sliced in the interior: constant y0."""
    got = _run(1.0, 5.0, 3.0, 999.0, x_start=2.0, x_end=4.0, code=1)
    assert got == pytest.approx(6.0, rel=1e-12)


def test_int3_lin_log_interior_slice():
    """INT=3: y=1+log(x). Integrate from sqrt(e) to e:
    ``[x*log(x)]_{sqrt(e)}^{e} = e - sqrt(e)/2``."""
    got = _run(1.0, np.e, 1.0, 2.0, x_start=np.sqrt(np.e), x_end=np.e, code=3)
    expected = np.e - 0.5 * np.sqrt(np.e)
    assert got == pytest.approx(expected, rel=1e-12)


def test_int4_log_lin_interior_slice():
    """INT=4: y=exp(x-1). Integrate from x=1.25 to x=1.75:
    e^{0.75} - e^{0.25}."""
    got = _run(1.0, 2.0, 1.0, np.e, x_start=1.25, x_end=1.75, code=4)
    expected = np.exp(0.75) - np.exp(0.25)
    assert got == pytest.approx(expected, rel=1e-12)


def test_int5_log_log_interior_slice():
    """INT=5: y=x^2. Integrate from x=1.25 to x=1.75:
    (1.75^3 - 1.25^3) / 3."""
    got = _run(1.0, 2.0, 1.0, 4.0, x_start=1.25, x_end=1.75, code=5)
    expected = (1.75**3 - 1.25**3) / 3.0
    assert got == pytest.approx(expected, rel=1e-12)


def test_full_panel_equals_sum_of_two_halves_all_ints():
    """Panel-additivity: integrating [x0, x1] equals the sum of
    integrating [x0, x_mid] and [x_mid, x1] for every INT.
    """
    x0, x1 = 1.0, 4.0
    y0, y1 = 2.0, 8.0
    x_mid = 2.5
    for code in (1, 2, 3, 4, 5):
        full = _run(x0, x1, y0, y1, x1, code, x_start=x0)
        left = _run(x0, x1, y0, y1, x_mid, code, x_start=x0)
        right = _run(x0, x1, y0, y1, x1, code, x_start=x_mid)
        assert full == pytest.approx(left + right, rel=1e-12), (
            f'INT={code}: full={full}, split={left+right}'
        )


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
            jnp.array([0.0]), jnp.array([2.0]),
            np.array([2], dtype=int),
            xp=xp,
        ).sum()

    grad = float(jax.grad(loss)(jnp.array(1.0)))
    # area = 0.5 * (y0 + y1) * (x1 - x0) = 0.5 * (y0 + 4) * 2 = y0 + 4
    # d/d y0 = 1.0
    assert grad == pytest.approx(1.0, rel=1e-12)


# ----------------------------------------------------------------------
# Autodiff wrt x-arguments (x0, x1, x_start, x_end).
#
# Fundamental-theorem invariant: for a truly panel-exact integrator,
# d(area)/d(x_end) equals the integrand y(x_end), and
# d(area)/d(x_start) equals -y(x_start). This holds for every INT code.
# The x0/x1 gradients are cross-checked against central finite
# differences.
# ----------------------------------------------------------------------


def _y_at(x, x0, x1, y0, y1, code):
    """Reference integrand y(x) for each ENDF INT code."""
    if code == 1:
        return y0
    if code == 2:
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    if code == 3:
        return y0 + (y1 - y0) * np.log(x / x0) / np.log(x1 / x0)
    if code == 4:
        return y0 * np.exp((x - x0) / (x1 - x0) * np.log(y1 / y0))
    if code == 5:
        return y0 * (x / x0) ** (np.log(y1 / y0) / np.log(x1 / x0))
    raise ValueError(code)


# Panels chosen so log-INT paths are well-defined (x, y strictly
# positive) and truncated interior slices exercise all four x-args.
_AUTODIFF_PANELS = [
    (1, 1.0, 4.0, 3.0, 3.0, 1.5, 3.2),
    (2, 1.0, 4.0, 2.0, 8.0, 1.5, 3.2),
    (3, 1.0, 4.0, 2.0, 8.0, 1.5, 3.2),
    (4, 1.0, 4.0, 2.0, 8.0, 1.5, 3.2),
    (5, 1.0, 4.0, 2.0, 8.0, 1.5, 3.2),
]


@pytest.mark.skipif(
    'jax' not in array_ns.available_backends(), reason='jax not installed',
)
@pytest.mark.parametrize('code,x0,x1,y0,y1,xs,xe', _AUTODIFF_PANELS)
def test_jax_grad_wrt_x_end_matches_integrand_all_ints(
    code, x0, x1, y0, y1, xs, xe,
):
    """d(area)/d(x_end) equals the integrand y(x_end) exactly (FTC)."""
    import jax
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')

    def loss(x_end):
        return integrate_tab1_panels(
            jnp.array([x0]), jnp.array([x1]),
            jnp.array([y0]), jnp.array([y1]),
            jnp.array([xs]), jnp.array([x_end]),
            np.array([code], dtype=int),
            xp=xp,
        )[0]

    grad = float(jax.grad(loss)(jnp.array(xe)))
    expected = _y_at(xe, x0, x1, y0, y1, code)
    assert grad == pytest.approx(expected, rel=1e-6), (
        f'INT={code}: grad={grad} vs y({xe})={expected}'
    )


@pytest.mark.skipif(
    'jax' not in array_ns.available_backends(), reason='jax not installed',
)
@pytest.mark.parametrize('code,x0,x1,y0,y1,xs,xe', _AUTODIFF_PANELS)
def test_jax_grad_wrt_x_start_matches_integrand_all_ints(
    code, x0, x1, y0, y1, xs, xe,
):
    """d(area)/d(x_start) equals -y(x_start) exactly (FTC)."""
    import jax
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')

    def loss(x_start):
        return integrate_tab1_panels(
            jnp.array([x0]), jnp.array([x1]),
            jnp.array([y0]), jnp.array([y1]),
            jnp.array([x_start]), jnp.array([xe]),
            np.array([code], dtype=int),
            xp=xp,
        )[0]

    grad = float(jax.grad(loss)(jnp.array(xs)))
    expected = -_y_at(xs, x0, x1, y0, y1, code)
    assert grad == pytest.approx(expected, rel=1e-6), (
        f'INT={code}: grad={grad} vs -y({xs})={expected}'
    )


@pytest.mark.skipif(
    'jax' not in array_ns.available_backends(), reason='jax not installed',
)
@pytest.mark.parametrize('code,x0,x1,y0,y1,xs,xe', _AUTODIFF_PANELS)
def test_jax_grad_wrt_panel_endpoints_matches_fd(
    code, x0, x1, y0, y1, xs, xe,
):
    """jax.grad wrt x0 and x1 matches central finite differences."""
    import jax
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')

    def loss_x0(v):
        return integrate_tab1_panels(
            jnp.array([v]), jnp.array([x1]),
            jnp.array([y0]), jnp.array([y1]),
            jnp.array([xs]), jnp.array([xe]),
            np.array([code], dtype=int),
            xp=xp,
        )[0]

    def loss_x1(v):
        return integrate_tab1_panels(
            jnp.array([x0]), jnp.array([v]),
            jnp.array([y0]), jnp.array([y1]),
            jnp.array([xs]), jnp.array([xe]),
            np.array([code], dtype=int),
            xp=xp,
        )[0]

    eps = 1e-5
    g0 = float(jax.grad(loss_x0)(jnp.array(x0)))
    fd0 = (float(loss_x0(x0 + eps)) - float(loss_x0(x0 - eps))) / (2 * eps)
    g1 = float(jax.grad(loss_x1)(jnp.array(x1)))
    fd1 = (float(loss_x1(x1 + eps)) - float(loss_x1(x1 - eps))) / (2 * eps)
    # For INT=1 x0/x1 don't enter the area formula; grad and fd are both
    # exactly zero.
    tol = 1e-5 if code == 1 else 1e-4
    assert g0 == pytest.approx(fd0, rel=tol, abs=1e-10), (
        f'INT={code} d/dx0: grad={g0}, fd={fd0}'
    )
    assert g1 == pytest.approx(fd1, rel=tol, abs=1e-10), (
        f'INT={code} d/dx1: grad={g1}, fd={fd1}'
    )


@pytest.mark.skipif(
    'jax' not in array_ns.available_backends(), reason='jax not installed',
)
def test_jax_grad_wrt_x_args_composes_with_jit():
    """jax.grad(jax.jit(f)) and jax.jit(jax.grad(f)) both work and agree.

    A mixed batch of INT codes goes through a single call so the
    xp.where dispatch across formulas is exercised under trace.
    """
    import jax
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')

    def f(x_end):
        return integrate_tab1_panels(
            jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            jnp.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            jnp.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            jnp.array([1.2, 1.2, 1.2, 1.2, 1.2]),
            x_end,
            np.array([1, 2, 3, 4, 5], dtype=int),
            xp=xp,
        ).sum()

    x = jnp.array([1.5, 1.5, 1.5, 1.5, 1.5])
    g_eager = jax.grad(f)(x)
    g_jit_of_grad = jax.jit(jax.grad(f))(x)
    g_grad_of_jit = jax.grad(jax.jit(f))(x)
    # Fundamental-theorem: d/d x_end[i] equals y_i(x_end).
    expected = np.array([
        _y_at(1.5, 1.0, 2.0, 1.0, 2.0, code) for code in (1, 2, 3, 4, 5)
    ])
    np.testing.assert_allclose(np.asarray(g_eager), expected, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(g_jit_of_grad), np.asarray(g_eager), rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(g_grad_of_jit), np.asarray(g_eager), rtol=1e-12,
    )
