"""Backend-agnostic port of the MF12 / MF13 / MF14 / MF15 photon-path
modules (issue #169 tier-2, gamma composition).

Adds an optional ``xp`` adapter to:

- ``mf12_interpretation.compute_photon_yields``
- ``mf13_interpretation.compute_total_photon_production_xs``
  (and its ``compute_photon_production_xs`` sibling)
- ``mf14_interpretation.compute_angdist_values``
- ``mf15_interpretation.compute_spectrum``

Also threads xp through the composition layer helpers that consume
them:

- ``quantities_mt_zap.quantities.compute_ddxs_from_mf15_mf14``
- ``quantities_mt_zap.distribution1d._compute_mf14_gamma_angdist``
  and the MF15 gamma branch of ``compute_energydist_values``

Pins:

- xp=None default and xp=numpy bit-identical for every ported entry.
- xp=jax reproduces numpy to machine precision on the gamma paths
  (Al-27 (n, g) MT102, JENDL-5 C-12 MT51/MT102).
- jax.grad reaches a dict-stored MF15 subsection-probability leaf
  through the top-level ``get_particle_production_dxs_dE`` gamma
  path and matches central finite-diff to rtol=1e-4.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import (
    mf12_interpretation as mf12,
    mf13_interpretation as mf13,
    mf14_interpretation as mf14,
    mf15_interpretation as mf15,
)
from endf_userpy.quantities import (
    get_particle_production_dxs_dE,
    get_particle_production_dxs_dmu,
    get_particle_production_ddxs,
    get_particle_production_xs,
)

from _corpus import resolve_al27, resolve_c12, resolve_nb93


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip('Al-27 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def c12_endf_dict():
    path = resolve_c12()
    if path is None:
        pytest.skip('C-12 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def nb93_endf_dict():
    path = resolve_nb93()
    if path is None:
        pytest.skip('Nb-93 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


# ---------------------------------------------------------------
# Leaf-module parity: xp=None (default) vs xp=numpy vs xp=jax
# ---------------------------------------------------------------

def test_mf12_compute_photon_yields_default_matches_xp_numpy(al27_endf_dict):
    """MF12 LO=1 tabulated yields: default and xp=numpy identical."""
    ein = np.array([1e5, 5e5, 1e6, 5e6, 1e7])
    egs = mf12.get_photon_energies(al27_endf_dict, 102)
    xp_np = array_ns.get_backend('numpy')
    a = np.asarray(mf12.compute_photon_yields(al27_endf_dict, 102, ein, egs))
    b = np.asarray(
        mf12.compute_photon_yields(al27_endf_dict, 102, ein, egs, xp=xp_np)
    )
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf12_compute_photon_yields_numpy_jax_parity(al27_endf_dict):
    ein = np.array([1e5, 5e5, 1e6, 5e6, 1e7])
    egs = mf12.get_photon_energies(al27_endf_dict, 102)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    a = np.asarray(
        mf12.compute_photon_yields(al27_endf_dict, 102, ein, egs, xp=xp_np)
    )
    b = np.asarray(
        mf12.compute_photon_yields(al27_endf_dict, 102, ein, egs, xp=xp_jx)
    )
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-30)


def test_mf13_total_prod_default_matches_xp_numpy(nb93_endf_dict):
    """MF13 total photon production: Nb-93 MT3 (JEFF-4.0 style, NK=1)."""
    if 13 not in nb93_endf_dict:
        pytest.skip('Nb-93 corpus has no MF13')
    mt = next(iter(nb93_endf_dict[13].keys()))
    ein = np.array([1e6, 5e6, 1e7])
    xp_np = array_ns.get_backend('numpy')
    a = np.asarray(
        mf13.compute_total_photon_production_xs(nb93_endf_dict, mt, ein)
    )
    b = np.asarray(
        mf13.compute_total_photon_production_xs(nb93_endf_dict, mt, ein, xp=xp_np)
    )
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf13_total_prod_numpy_jax_parity(nb93_endf_dict):
    if 13 not in nb93_endf_dict:
        pytest.skip('Nb-93 corpus has no MF13')
    mt = next(iter(nb93_endf_dict[13].keys()))
    ein = np.array([1e6, 5e6, 1e7])
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    a = np.asarray(
        mf13.compute_total_photon_production_xs(nb93_endf_dict, mt, ein, xp=xp_np)
    )
    b = np.asarray(
        mf13.compute_total_photon_production_xs(nb93_endf_dict, mt, ein, xp=xp_jx)
    )
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-30)


def test_mf14_angdist_default_matches_xp_numpy_isotropic(al27_endf_dict):
    """MF14 LI=1 isotropic (Al-27 MT102)."""
    ein = np.array([1e5, 5e5, 1e6])
    egs = mf14.get_photon_energies(al27_endf_dict, 102)
    if egs is None:
        egs = np.array([0.0])
    mus = np.linspace(-0.9, 0.9, 5)
    xp_np = array_ns.get_backend('numpy')
    a = np.asarray(mf14.compute_angdist_values(al27_endf_dict, 102, ein, egs, mus))
    b = np.asarray(
        mf14.compute_angdist_values(al27_endf_dict, 102, ein, egs, mus, xp=xp_np)
    )
    np.testing.assert_array_equal(a, b)


def test_mf14_angdist_default_matches_xp_numpy_legendre(c12_endf_dict):
    """MF14 LI=0 LTT=1 Legendre (JENDL-5 C-12 MT51)."""
    if 14 not in c12_endf_dict or 51 not in c12_endf_dict[14]:
        pytest.skip('C-12 corpus has no MF14/51')
    mtsec = c12_endf_dict[14][51]
    if mtsec['LI'] != 0 or mtsec.get('LTT') != 1:
        pytest.skip('MF14/51 is not Legendre in this file')
    ein = np.array([5e6, 1e7, 1.4e7])
    egs = mf14.get_photon_energies(c12_endf_dict, 51)
    mus = np.linspace(-0.9, 0.9, 5)
    xp_np = array_ns.get_backend('numpy')
    a = np.asarray(mf14.compute_angdist_values(c12_endf_dict, 51, ein, egs, mus))
    b = np.asarray(
        mf14.compute_angdist_values(c12_endf_dict, 51, ein, egs, mus, xp=xp_np)
    )
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf14_angdist_numpy_jax_parity_legendre(c12_endf_dict):
    if 14 not in c12_endf_dict or 51 not in c12_endf_dict[14]:
        pytest.skip('C-12 corpus has no MF14/51')
    mtsec = c12_endf_dict[14][51]
    if mtsec['LI'] != 0 or mtsec.get('LTT') != 1:
        pytest.skip('MF14/51 is not Legendre in this file')
    ein = np.array([5e6, 1e7, 1.4e7])
    egs = mf14.get_photon_energies(c12_endf_dict, 51)
    mus = np.linspace(-0.9, 0.9, 5)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    a = np.asarray(
        mf14.compute_angdist_values(c12_endf_dict, 51, ein, egs, mus, xp=xp_np)
    )
    b = np.asarray(
        mf14.compute_angdist_values(c12_endf_dict, 51, ein, egs, mus, xp=xp_jx)
    )
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-30)


def test_mf15_compute_spectrum_default_matches_xp_numpy(al27_endf_dict):
    """MF15 continuous gamma spectrum (Al-27 MT102)."""
    ein = np.array([1e5, 5e5, 1e6, 5e6, 1e7])
    eout = np.linspace(0.0, 8e6, 15)
    xp_np = array_ns.get_backend('numpy')
    a = np.asarray(mf15.compute_spectrum(al27_endf_dict, 102, ein, eout))
    b = np.asarray(
        mf15.compute_spectrum(al27_endf_dict, 102, ein, eout, xp=xp_np)
    )
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf15_compute_spectrum_numpy_jax_parity(al27_endf_dict):
    ein = np.array([1e5, 5e5, 1e6, 5e6, 1e7])
    eout = np.linspace(0.0, 8e6, 15)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    a = np.asarray(mf15.compute_spectrum(al27_endf_dict, 102, ein, eout, xp=xp_np))
    b = np.asarray(mf15.compute_spectrum(al27_endf_dict, 102, ein, eout, xp=xp_jx))
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-30)


# ---------------------------------------------------------------
# End-to-end gamma composition parity
# ---------------------------------------------------------------

@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_end_to_end_gamma_dxs_dE_parity(al27_endf_dict):
    """Al-27 (n, g) gamma dxs/dE composes MF3 xs, MF12 yields, MF15
    spectrum. xp=jax matches xp=numpy to machine precision."""
    ein = np.array([1e5, 5e5, 1e6, 5e6])
    eout = np.linspace(1e5, 8e6, 20)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        a = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,g)', 'g', ein, eout, xp=xp_np,
        ))
        b = np.asarray(get_particle_production_dxs_dE(
            al27_endf_dict, '(n,g)', 'g', ein, eout, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_end_to_end_gamma_dxs_dmu_parity(al27_endf_dict):
    """Al-27 (n, g) gamma dxs/dmu composes MF14 angular through the
    composition helper."""
    ein = np.array([1e5, 5e5, 1e6, 5e6])
    mus = np.linspace(-0.9, 0.9, 5)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        a = np.asarray(get_particle_production_dxs_dmu(
            al27_endf_dict, '(n,g)', 'g', ein, mus, xp=xp_np,
        ))
        b = np.asarray(get_particle_production_dxs_dmu(
            al27_endf_dict, '(n,g)', 'g', ein, mus, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_end_to_end_gamma_ddxs_parity(al27_endf_dict):
    """Al-27 (n, g) gamma DDX composes ``compute_ddxs_from_mf15_mf14``
    (MF12 yields + MF14 angular + MF15 spectrum)."""
    ein = np.array([1e5, 5e5])
    eout = np.linspace(1e5, 8e6, 10)
    mus = np.linspace(-0.9, 0.9, 4)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        a = np.asarray(get_particle_production_ddxs(
            al27_endf_dict, '(n,g)', 'g', ein, eout, mus, xp=xp_np,
        ))
        b = np.asarray(get_particle_production_ddxs(
            al27_endf_dict, '(n,g)', 'g', ein, eout, mus, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_end_to_end_gamma_xs_parity(al27_endf_dict):
    """Al-27 (n, g) gamma production XS composes MF3 xs * MF12 yields
    with MF13-only fast path passing through xp too."""
    ein = np.array([1e5, 5e5, 1e6, 5e6])
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        a = np.asarray(get_particle_production_xs(
            al27_endf_dict, '(n,g)', 'g', ein, xp=xp_np,
        ))
        b = np.asarray(get_particle_production_xs(
            al27_endf_dict, '(n,g)', 'g', ein, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-30)


# ---------------------------------------------------------------
# jax.grad end-to-end: MF15 subsection probability leaf
# ---------------------------------------------------------------

@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_end_to_end_to_mf15_subsec_prob(al27_endf_dict):
    """jax.grad of a scalar summary of
    ``get_particle_production_dxs_dE`` reaches a dict-stored MF15
    subsection-probability leaf and matches central finite-diff.
    Proves the tracer survives the full stack from top-level API,
    through the composition layer, into
    ``mf15_interpretation.compute_spectrum`` -> ``interp_tab1``.
    """
    import jax
    import jax.numpy as jnp

    p_list = al27_endf_dict[15][102]['subsection'][1]['rtfm_tab1']['p']
    idx = 0
    orig = float(p_list[idx])

    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([1e5, 5e5, 1e6])
    eout = jnp.linspace(1e5, 5e6, 15)

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        new_p = list(d_t[15][102]['subsection'][1]['rtfm_tab1']['p'])
        new_p[idx] = theta
        d_t[15][102]['subsection'][1]['rtfm_tab1']['p'] = new_p
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return jnp.sum(get_particle_production_dxs_dE(
                d_t, '(n,g)', 'g', ein, eout, xp=xp_jx,
            ))

    val = float(loss(jnp.array(orig)))
    grad = float(jax.grad(loss)(jnp.array(orig)))
    assert np.isfinite(grad)
    assert val > 0.0
    assert abs(grad) > 0.0
    eps = 1e-3 if orig == 0 else 1e-3 * abs(orig)
    fd = (
        float(loss(jnp.array(orig + eps)))
        - float(loss(jnp.array(orig - eps)))
    ) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-15)
