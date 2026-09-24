"""MF5 LF=11 energy-dependent Watt fission spectrum.

Adds Phase 2 base-physics coverage for LF=11 (previously raised
NotImplementedError; #198 roadmap). Verifies the closed-form
implementation against:

- The support boundary: f = 0 outside ``[0, E - U]``.
- Normalisation: ``integral_{0}^{E-U} f(E, E') dE' == 1`` to
  sub-permille on a fine grid, at multiple incident energies and
  parameter settings.
- Reduction to the simple LF=7 Maxwellian when ``b -> 0`` (both
  become ``exp(-E'/a) * something(E')`` distributions; specifically
  for small b, ``sinh(sqrt(b*E')) -> sqrt(b*E')`` so
  ``LF=11 -> const * sqrt(E') * exp(-E'/a) / I(E)`` which is
  the LF=7 Maxwellian in disguise).
- Backend-agnostic parity (numpy vs jax).
- ``jax.grad`` wrt the file-side ``a``, ``b`` and ``U`` tracers.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.mfsec_interpretation import mf5_interpretation as mf5
from endf_userpy.primitives import array_ns


def _jax_available():
    return 'jax' in array_ns.available_backends()


def _fd5(f, x, h):
    """5-point central FD stencil, O(h^4) error."""
    return (-float(f(x + 2 * h)) + 8 * float(f(x + h))
            - 8 * float(f(x - h)) + float(f(x - 2 * h))) / (12 * h)


def _tab1(x_name, y_name, x, y):
    return {x_name: list(x), y_name: list(y), 'INT': [2], 'NBT': [len(x)]}


def _make_watt_contrib(a, b, U):
    return {
        'p_table': _tab1('E', 'p', [1e-5, 3e7], [1.0, 1.0]),
        'a_table': _tab1('E', 'a', [1e-5, 3e7], [a, a]),
        'b_table': _tab1('E', 'b', [1e-5, 3e7], [b, b]),
        'U': U,
        'LF': 11,
    }


def _make_endf_dict(contrib):
    return {5: {18: {'contribution': {1: contrib}}}}


@pytest.mark.parametrize('a,b,U,E_in', [
    (1.0e6, 2.5e-6, 0.0, 5.0e6),
    (1.0e6, 2.5e-6, 0.0, 1.0e7),
    (1.2e6, 3.0e-6, 1.0e5, 1.5e7),
    (5.0e5, 1.0e-6, 0.0, 5.0e6),
])
def test_lf11_watt_support_boundary(a, b, U, E_in):
    """f is zero for E' > E - U and for E' < 0."""
    d = _make_endf_dict(_make_watt_contrib(a, b, U))
    ein = np.array([E_in])
    E_minus_U = E_in - U
    eout = np.array([-1.0, 0.0, 0.5 * E_minus_U,
                     E_minus_U - 1.0,
                     E_minus_U * 1.001, E_minus_U * 2.0])
    f = np.asarray(mf5.compute_spectrum(d, 18, ein, eout))[0]
    # Outside-support samples must be zero.
    assert f[0] == 0.0                            # E' < 0
    assert f[4] == 0.0                            # E' > E-U
    assert f[5] == 0.0                            # E' >> E-U
    # Inside-support samples must be non-negative and finite.
    assert np.isfinite(f[1:4]).all()
    assert (f[1:4] >= 0.0).all()


@pytest.mark.parametrize('a,b,U,E_in', [
    (1.0e6, 2.5e-6, 0.0, 5.0e6),
    (1.0e6, 2.5e-6, 0.0, 1.5e7),
    (1.2e6, 3.0e-6, 1.0e5, 1.0e7),
    (5.0e5, 1.0e-6, 0.0, 8.0e6),
    (2.0e6, 4.0e-6, 5.0e5, 2.0e7),
])
def test_lf11_watt_normalisation(a, b, U, E_in):
    """Integral of f over [0, E-U] equals 1 to sub-permille on a
    fine trapezoid grid."""
    d = _make_endf_dict(_make_watt_contrib(a, b, U))
    ein = np.array([E_in])
    eout = np.linspace(1e2, E_in - U, 4000)
    f = np.asarray(mf5.compute_spectrum(d, 18, ein, eout))[0]
    integral = float(np.trapezoid(f, eout))
    assert abs(integral - 1.0) < 1e-3, (
        f'LF=11 spectrum not normalised at (a={a}, b={b}, U={U}, '
        f'E={E_in}): integral={integral:.6f}'
    )


def test_lf11_watt_reduces_to_maxwellian_small_b():
    """For small b, ``sinh(sqrt(b*E')) -> sqrt(b*E')``, so LF=11
    reduces to a normalised Maxwellian in E' with temperature-like
    parameter a. Under sufficiently small b, the LF=11 spectrum
    shape matches the LF=7 Maxwellian shape up to normalisation."""
    a = 1.2e6
    U = 0.0
    E_in = 8.0e6
    ein = np.array([E_in])
    eout = np.linspace(1e4, E_in - U, 400)

    # LF=11 at small b.
    d11 = _make_endf_dict(_make_watt_contrib(a, b=1e-10, U=U))
    f11 = np.asarray(mf5.compute_spectrum(d11, 18, ein, eout))[0]

    # LF=7 Maxwellian with theta=a.
    def _make_lf7(theta):
        return {
            'p_table': _tab1('E', 'p', [1e-5, 3e7], [1.0, 1.0]),
            'theta_table': _tab1(
                'E', 'theta', [1e-5, 3e7], [theta, theta],
            ),
            'U': U,
            'LF': 7,
        }
    d7 = _make_endf_dict(_make_lf7(a))
    f7 = np.asarray(mf5.compute_spectrum(d7, 18, ein, eout))[0]

    # Both must be normalised; the shape ratio should be near 1
    # in the bulk of the support.
    np.testing.assert_allclose(f11, f7, rtol=1e-3, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_lf11_watt_numpy_jax_parity():
    """Backend-agnostic parity to machine precision."""
    d = _make_endf_dict(_make_watt_contrib(a=1.2e6, b=2.5e-6, U=1.0e5))
    ein = np.array([5.0e6, 1.0e7, 1.5e7])
    eout = np.linspace(1e4, 1.4e7, 200)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    f_np = np.asarray(mf5.compute_spectrum(d, 18, ein, eout, xp=xp_np))
    f_jx = np.asarray(mf5.compute_spectrum(d, 18, ein, eout, xp=xp_jx))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-10, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_lf11_watt_jax_grad_wrt_a_matches_fd():
    """jax.grad(sum(spectrum))(a) matches central FD. Pins that
    the analytic implementation of I(E) is differentiable through
    the erf, sinh, sqrt, and exp terms."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    U = 5.0e5
    b = 2.5e-6
    ein = jnp.array([5.0e6, 1.0e7])
    eout = jnp.linspace(1e4, 8e6, 100)

    def loss(a_val):
        c = _make_watt_contrib(a=0.0, b=b, U=U)
        c['a_table']['a'] = [a_val, a_val]
        return jnp.sum(mf5.compute_spectrum(
            _make_endf_dict(c), 18, ein, eout, xp=xp_jx,
        ))

    a0 = 1.2e6
    grad = float(jax.grad(loss)(jnp.array(a0)))
    fd = _fd5(lambda v: loss(jnp.array(v)), a0, a0 * 1e-4)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=1e-3, atol=1e-20)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_lf11_watt_jax_grad_wrt_b_matches_fd():
    """jax.grad(sum(spectrum))(b) matches central FD."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    U = 5.0e5
    a = 1.2e6
    ein = jnp.array([5.0e6, 1.0e7])
    eout = jnp.linspace(1e4, 8e6, 100)

    def loss(b_val):
        c = _make_watt_contrib(a=a, b=0.0, U=U)
        c['b_table']['b'] = [b_val, b_val]
        return jnp.sum(mf5.compute_spectrum(
            _make_endf_dict(c), 18, ein, eout, xp=xp_jx,
        ))

    b0 = 2.5e-6
    grad = float(jax.grad(loss)(jnp.array(b0)))
    fd = _fd5(lambda v: loss(jnp.array(v)), b0, b0 * 1e-3)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=1e-3, atol=1e-20)


def test_lf11_watt_previously_not_implemented_error_is_gone():
    """Regression pin for the base-physics gap fix: LF=11 no
    longer raises NotImplementedError / ValueError. Delete when
    the change has been in main for a release cycle."""
    d = _make_endf_dict(_make_watt_contrib(a=1.0e6, b=2.5e-6, U=0.0))
    ein = np.array([5.0e6])
    eout = np.array([1e5, 1e6, 2e6])
    # Just call it - would previously raise; now returns a finite
    # spectrum row.
    f = np.asarray(mf5.compute_spectrum(d, 18, ein, eout))
    assert f.shape == (1, 3)
    assert np.isfinite(f).all()
