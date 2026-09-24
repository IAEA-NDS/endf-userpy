"""Phase-1 autodiff-coverage pins for MF14 LTT=1 (photon
Legendre angular distribution).

Current state (Phase 1 sweep):

- numpy vs jax parity on concrete inputs: works.
  ``compute_angdist_values`` routes anisotropic gamma lines to
  ``compute_angdist_from_legendre`` which is xp-agnostic on the
  file-side coefficient axis.
- ``jax.grad`` wrt query-side E or mu: does NOT work today. Same
  root cause as MF6 LAW=2 (issue #201): the shared primitive
  ``evaluate_interp_legendre_polynomials`` calls
  ``x = np.asarray(x); mu = np.asarray(mu)`` on entry, which
  materialises jax tracers.

Fixing #201 also fixes MF14 LTT=1 for both E and mu. Both live
in the same Phase 3 tabulated-MF autodiff scope.

Uses C-12 MT=51 (first inelastic level) from
``tests/data_law1_adhoc/jendl5_n_C-12.endf``: LI=0 LTT=1 with
anisotropic gamma Legendre data.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf14_interpretation as mf14
from endf_userpy.primitives import array_ns

from _corpus import resolve_c12


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def c12_endf_dict():
    path = resolve_c12()
    if path is None:
        pytest.skip('C-12 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def _photon_energies_c12_mt51(endf_dict):
    """Return the discrete gamma photon energies MF14 MT=51
    declares. Kept small (a few lines) so the test grid stays
    modest."""
    egs = mf14.get_photon_energies(endf_dict, 51)
    return np.asarray(egs, dtype=np.float64)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf14_ltt1_numpy_jax_parity(c12_endf_dict):
    """xp=jax and xp=numpy return the same values on a query grid
    spanning several incident Es and mus in [-0.9, 0.9]. Pins the
    working concrete-input path on C-12 MT=51 (LTT=1 Legendre)."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    egs = _photon_energies_c12_mt51(c12_endf_dict)
    # Pick a subset of photon lines (first ~3) for a modest grid.
    egs_sub = egs[:3]
    ein = np.array([5.0e6, 8.0e6, 1.2e7, 1.6e7], dtype=np.float64)
    mu = np.linspace(-0.9, 0.9, 15, dtype=np.float64)

    f_np = np.asarray(mf14.compute_angdist_values(
        c12_endf_dict, 51, ein, egs_sub, mu, xp=xp_np,
    ))
    f_jx = np.asarray(mf14.compute_angdist_values(
        c12_endf_dict, 51,
        jnp.asarray(ein), egs_sub, jnp.asarray(mu),
        xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-10, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf14_ltt1_grad_wrt_E_currently_raises(c12_endf_dict):
    """Regression pin for the query-side E-tracer gap. Same
    root cause as issue #201: shared primitive
    ``evaluate_interp_legendre_polynomials`` materialises the
    tracer x via ``np.asarray`` on entry."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    egs = _photon_energies_c12_mt51(c12_endf_dict)[:3]
    mu = jnp.linspace(-0.9, 0.9, 15)

    def loss(E_scalar):
        return jnp.sum(mf14.compute_angdist_values(
            c12_endf_dict, 51,
            jnp.array([E_scalar]), egs, mu,
            xp=xp_jx,
        ))

    try:
        import jax.errors
        expected_exc = jax.errors.TracerArrayConversionError
    except AttributeError:
        expected_exc = Exception
    with pytest.raises(expected_exc):
        jax.grad(loss)(jnp.array(1.0e7))


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf14_ltt1_grad_wrt_mu_currently_raises(c12_endf_dict):
    """Same regression pin on the mu axis."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    egs = _photon_energies_c12_mt51(c12_endf_dict)[:3]
    ein = jnp.array([5.0e6, 1.0e7, 1.5e7])

    def loss(mu_scalar):
        return jnp.sum(mf14.compute_angdist_values(
            c12_endf_dict, 51,
            ein, egs, jnp.array([mu_scalar]),
            xp=xp_jx,
        ))

    try:
        import jax.errors
        expected_exc = jax.errors.TracerArrayConversionError
    except AttributeError:
        expected_exc = Exception
    with pytest.raises(expected_exc):
        jax.grad(loss)(jnp.array(0.3))
