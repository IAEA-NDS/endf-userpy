"""Tests for compute_ddx_continuous_broadened.

These tests monkeypatch the dist2d / yields / xs primitives so the
broadening wrapper can be checked against closed-form references
without depending on any real ENDF data.
"""
import numpy as np
import pytest

from endf_userpy.quantities_mt_zap import ddx_broadening as ddxb


def _gaussian(x, sigma, mu=0.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


@pytest.fixture
def patched_environment(monkeypatch):
    """Replace dist2d, yields, and cross-section primitives with
    user-controlled stand-ins. The fixture returns a setter object
    so each test installs the stand-in it needs."""
    state = {}

    def fake_dist2d(endf_dict, mt, zap, einc, eouts, mus, to_lab=True):
        return state['dist2d'](einc, eouts, mus)

    def fake_yields(endf_dict, mt, zap, einc, include_discrete=True, level=None):
        return state['yields'](einc)

    def fake_xs(endf_dict, mt, einc):
        return state['xs'](einc)

    monkeypatch.setattr(ddxb, 'compute_dist2d_values', fake_dist2d)
    monkeypatch.setattr(ddxb, 'compute_yields', fake_yields)
    monkeypatch.setattr(
        ddxb.mf3_interp, 'compute_cross_section', fake_xs
    )

    def setter(dist2d=None, yields=None, xs=None):
        if dist2d is not None:
            state['dist2d'] = dist2d
        if yields is not None:
            state['yields'] = yields
        if xs is not None:
            state['xs'] = xs

    return setter


def test_constant_spectrum_preserved(patched_environment):
    """Constant dist2d * unit-norm kernel = same constant, in the
    interior of the eout window. With xs=yield=1 the result is
    exactly 1/(2 pi) everywhere by the compute_ddxs normalization."""
    n_einc, n_mu = 2, 3
    patched_environment(
        dist2d=lambda einc, eouts, mus: np.ones((len(einc), len(eouts), len(mus))),
        yields=lambda einc: np.ones(len(einc)),
        xs=lambda einc: np.ones(len(einc)),
    )

    einc = np.array([1.0e6, 5.0e6])
    eouts = np.linspace(-2.0e6, 2.0e6, 21)
    mus = np.linspace(-1.0, 1.0, n_mu)
    sigma = 3.0e5

    result = ddxb.compute_ddx_continuous_broadened(
        endf_dict=None, mt=0, zap=1,
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        kernel=lambda d: _gaussian(d, sigma),
        kernel_width=sigma,
        rtol=1e-5,
    )

    assert result.shape == (n_einc, len(eouts), n_mu)
    expected = 1.0 / (2 * np.pi)
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_gaussian_spectrum_broadens_analytically(patched_environment):
    """Gaussian dist2d (in E_out) folded with Gaussian kernel produces
    a wider Gaussian with sigma_y = sqrt(sigma_f^2 + sigma_k^2). The
    xs/yield/2pi prefactor follows the existing compute_ddxs convention."""
    sigma_f = 4.0e5
    sigma_k = 2.0e5
    e_centre = 1.4e7

    def dist2d(einc, eouts, mus):
        spec = _gaussian(eouts, sigma_f, mu=e_centre)
        return np.broadcast_to(
            spec[None, :, None],
            (len(einc), len(eouts), len(mus)),
        ).copy()

    patched_environment(
        dist2d=dist2d,
        yields=lambda einc: np.full(len(einc), 2.5),
        xs=lambda einc: np.full(len(einc), 0.7),
    )

    einc = np.array([1.4e7])
    eouts = np.linspace(e_centre - 2.0e6, e_centre + 2.0e6, 41)
    mus = np.array([-0.5, 0.0, 0.5])

    result = ddxb.compute_ddx_continuous_broadened(
        endf_dict=None, mt=0, zap=1,
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        kernel=lambda d: _gaussian(d, sigma_k),
        kernel_width=sigma_k,
        rtol=1e-6,
        max_iter=10,
    )

    sigma_y = np.sqrt(sigma_f ** 2 + sigma_k ** 2)
    expected_spec = _gaussian(eouts, sigma_y, mu=e_centre)
    expected = expected_spec[None, :, None] * (2.5 * 0.7 / (2 * np.pi))

    # Broadcast expected across mu axis explicitly so assert_allclose
    # checks every slice.
    expected = np.broadcast_to(expected, result.shape)
    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-12)


def test_integral_preserved(patched_environment):
    """Convolution by a unit-norm kernel preserves the integral of f
    over E_out. So the total production cross section per (E_in, mu)
    bin -- integral of DDX over E_out -- should match before/after
    broadening."""
    sigma_f = 4.0e5
    sigma_k = 2.0e5
    e_centre = 1.4e7

    def dist2d(einc, eouts, mus):
        spec = _gaussian(eouts, sigma_f, mu=e_centre)
        return np.broadcast_to(
            spec[None, :, None],
            (len(einc), len(eouts), len(mus)),
        ).copy()

    patched_environment(
        dist2d=dist2d,
        yields=lambda einc: np.full(len(einc), 1.5),
        xs=lambda einc: np.full(len(einc), 0.3),
    )

    einc = np.array([1.4e7])
    eouts = np.linspace(e_centre - 3.0e6, e_centre + 3.0e6, 401)
    mus = np.array([0.0])

    broadened = ddxb.compute_ddx_continuous_broadened(
        endf_dict=None, mt=0, zap=1,
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        kernel=lambda d: _gaussian(d, sigma_k),
        kernel_width=sigma_k,
        rtol=1e-5,
        max_iter=10,
    )

    # f integrates to 1 (unit-norm Gaussian over an infinite domain;
    # the eouts grid spans +/- 7.5 sigma so the truncation loss is
    # negligible). Hence the production-side integral is xs * yield
    # / (2 pi) per (E_in, mu) bin = 0.3 * 1.5 / (2 pi).
    integral = np.trapezoid(broadened[0, :, 0], eouts)
    expected_integral = 1.5 * 0.3 / (2 * np.pi)
    assert abs(integral - expected_integral) / expected_integral < 1e-3
