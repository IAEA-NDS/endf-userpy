"""Unbroadened DDX / dxs/dE contribution from MF12 and MF13
discrete gamma lines (issue #266).

Before this coverage, ``get_particle_production_ddxs`` and
``get_particle_production_dxs_dE`` for gamma emission summed the
MF15 continuum and the reconstructed continuous DDX but silently
dropped the discrete photon lines when no ``broadening=`` kernel
was passed. This module pins the new behaviour:

* Each MF12 (or MF13) discrete line at ``Eg_i`` is placed on the
  caller's ``E_out`` grid at the nearest bin with height
  ``1 / bin_width`` so a midpoint-rule quadrature over ``E_out``
  recovers the yield-weighted amplitude.
* Integration over ``E_out`` reproduces the per-line photon
  production ``sigma(Ein) * y_i(Ein)`` (up to the ``1 / (2 pi)``
  factor on DDX).
* Under ``xp=jax`` reverse-mode ``jax.grad`` reaches the yields
  and cross section through the accumulator.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.quantities import (
    get_particle_production_ddxs,
    get_particle_production_dxs_dE,
)
from endf_userpy.quantities_mt_zap import ddx_broadening as ddxb


def _local_widths(eouts):
    """Same convention as ddx_broadening._grid_local_widths."""
    n = len(eouts)
    w = np.empty(n, dtype=float)
    w[0] = eouts[1] - eouts[0]
    w[-1] = eouts[-1] - eouts[-2]
    w[1:-1] = 0.5 * (eouts[2:] - eouts[:-2])
    return w


@pytest.fixture(scope='module')
def h2_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        'tests/data/n-001_H_002.endf',
    )


def test_h2_unbroadened_dxs_dE_places_line_at_nearest_bin(h2_endf_dict):
    """H-2 MT=102 has one MF12 discrete photon line at Eg=6.251 MeV.
    The unbroadened dxs/dE spike must land on the E_out bin closest
    to that Eg."""
    ein = np.array([1e6])
    eouts = np.linspace(0.0, 1e7, 51)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = get_particle_production_dxs_dE(
            h2_endf_dict, '(n,g)', 'g', ein, eouts,
        )
    assert r is not None
    nonzero = np.nonzero(r[0])[0]
    assert nonzero.size == 1, f'expected one spike, got {nonzero.size}'
    eg_actual = 6.251002e6
    expected_bin = int(np.argmin(np.abs(eouts - eg_actual)))
    assert nonzero[0] == expected_bin


def test_h2_unbroadened_dxs_dE_recovers_yield_weighted_prodxs(h2_endf_dict):
    """Integrating dxs/dE over E_out (midpoint quadrature) must
    recover sigma(Ein) * y_i(Ein) for the single discrete line.
    """
    from endf_userpy.mfsec_interpretation import (
        mf3_interpretation as mf3_interp,
        mf12_interpretation as mf12_interp,
    )
    ein = np.array([1e6])
    eouts = np.linspace(0.0, 1e7, 51)
    widths = _local_widths(eouts)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = get_particle_production_dxs_dE(
            h2_endf_dict, '(n,g)', 'g', ein, eouts,
        )
    integ = float(np.sum(r[0] * widths))

    xs = float(mf3_interp.compute_cross_section(h2_endf_dict, 102, ein)[0])
    egs = mf12_interp.get_photon_energies(h2_endf_dict, 102)
    yields = mf12_interp.compute_photon_yields(
        h2_endf_dict, 102, ein, np.asarray(egs, dtype=float),
    )
    disc_mask = np.asarray(egs) > 0.0
    y_disc = float(yields[0, disc_mask].sum())
    expected = xs * y_disc
    np.testing.assert_allclose(integ, expected, rtol=1e-10)


def test_h2_unbroadened_ddx_solid_angle_integral_matches_dxs_dE(h2_endf_dict):
    """The DDX integrated over the full solid angle (2 pi times the
    mu integral over [-1, +1]) must reproduce the 1D dxs/dE. Since
    H-2 MT=102 has no MF14, the per-line angular distribution is
    isotropic and the mu integral is exactly 2 (over mu in [-1, +1]).
    """
    ein = np.array([1e6])
    eouts = np.linspace(0.0, 1e7, 51)
    mus = np.linspace(-1.0, 1.0, 21)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r_ddx = get_particle_production_ddxs(
            h2_endf_dict, '(n,g)', 'g', ein, eouts, mus,
        )
        r_dxs = get_particle_production_dxs_dE(
            h2_endf_dict, '(n,g)', 'g', ein, eouts,
        )
    # trapezoidal integral over mu, times 2 pi for the solid-angle
    ddx_int_mu = np.trapezoid(r_ddx[0], mus, axis=-1) * 2 * np.pi
    np.testing.assert_allclose(ddx_int_mu, r_dxs[0], rtol=1e-6)


def test_h2_ddx_shape_and_placement(h2_endf_dict):
    """The DDX must be zero everywhere except at the nearest E_out
    bin to the single discrete Eg, and flat in mu (isotropic
    fallback with no MF14).
    """
    ein = np.array([1e6])
    eouts = np.linspace(0.0, 1e7, 51)
    mus = np.array([-0.5, 0.0, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r_ddx = get_particle_production_ddxs(
            h2_endf_dict, '(n,g)', 'g', ein, eouts, mus,
        )
    assert r_ddx.shape == (1, 51, 3)
    nonzero_bins = np.nonzero(r_ddx[0].sum(axis=-1))[0]
    assert nonzero_bins.size == 1
    row = r_ddx[0, nonzero_bins[0], :]
    np.testing.assert_allclose(row, row[0], rtol=1e-12)


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_flows_through_accumulator():
    """``jax.grad`` reaches the per-line weight (yield times cross
    section) and the per-line angular distribution through the
    unbroadened accumulator. Test at the primitive layer with
    synthetic inputs so the file structure is not on the critical
    path.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    n_einc, n_eouts, n_mus = 1, 21, 3
    eouts = np.linspace(0.0, 10.0, n_eouts)
    Eg_disc = np.array([2.5, 7.5])

    per_line_angdist_np = np.full((n_einc, len(Eg_disc), n_mus), 0.5)

    def loss(weight):
        w_ei = weight.reshape(n_einc, len(Eg_disc))
        r = ddxb._accumulate_discrete_lines_ddx(
            eouts, Eg_disc, w_ei, jnp.asarray(per_line_angdist_np),
            n_einc, n_eouts, n_mus, xp_jax,
        )
        return jnp.sum(r)

    theta0 = jnp.array([1.0, 2.0])
    grad = jax.grad(loss)(theta0)
    eps = 1e-4
    fd = np.empty(2)
    for i in range(2):
        plus = theta0.at[i].add(eps)
        minus = theta0.at[i].add(-eps)
        fd[i] = (float(loss(plus)) - float(loss(minus))) / (2 * eps)
    np.testing.assert_allclose(np.asarray(grad), fd, rtol=1e-5)
