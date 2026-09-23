"""MF4 backend-agnostic port: numpy default / xp adapter parity
plus dict-first JAX autodiff on Legendre coefficients.

Companion to the pre-existing Fortran-parity tests in
``test_mf4_interpretation.py`` (which pin the numerics against the
Fortran reference). This file exercises the ``xp=`` adapter added
in the issue #169 backend-agnostic port and pins:

- numpy default vs explicit ``xp=numpy`` adapter: bit-identical.
- numpy vs JAX adapter on a real MF4 LTT=1 (Legendre) file: match
  to machine precision.
- ``jax.grad`` through the reconstruction reaches back to a
  Legendre coefficient stored at ``endf_dict[4][mt]['a'][en_idx]
  [L_idx]`` and matches central finite-diff.

Runs without the Fortran extension (issue-#168 pattern).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf4_interpretation as mf4py
from endf_userpy.primitives import array_ns


DATA_DIR = Path(__file__).resolve().parent / 'data'


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def be9_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        DATA_DIR / 'n-004_Be_009.endf',
    )


def test_default_matches_xp_numpy_adapter(be9_endf_dict):
    """xp=None (default) is bit-identical to xp=numpy on Be-9 MT=2
    Legendre reconstruction."""
    mt = 2
    energies = np.linspace(1e4, 1e6, 6)
    mu = np.linspace(-0.5, 0.5, 5)
    default = np.asarray(mf4py.compute_angdist_values(
        be9_endf_dict, mt, energies, mu, True,
    ))
    xp_np = array_ns.get_backend('numpy')
    with_xp = np.asarray(mf4py.compute_angdist_values(
        be9_endf_dict, mt, energies, mu, True, xp=xp_np,
    ))
    np.testing.assert_array_equal(default, with_xp)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_numpy_jax_parity_on_be9_mt2(be9_endf_dict):
    """The JAX adapter reproduces the numpy result to machine
    precision on Be-9 MT=2 (LTT=1 Legendre). No perturbation --
    just adapter equivalence."""
    mt = 2
    energies = np.linspace(1e4, 1e6, 6)
    mu = np.linspace(-0.5, 0.5, 5)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    f_np = np.asarray(mf4py.compute_angdist_values(
        be9_endf_dict, mt, energies, mu, True, xp=xp_np,
    ))
    f_jx = np.asarray(mf4py.compute_angdist_values(
        be9_endf_dict, mt, energies, mu, True, xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-11, atol=1e-14)


def _find_first_nonzero_legendre_row(a_dict):
    """Return the smallest energy-key at which any Legendre
    coefficient is nonzero. Rows near the elastic threshold on
    isotope evaluations are all zero (isotropic in CM) so pick a
    row that actually contributes to grad."""
    for k, coeffs in a_dict.items():
        if any(abs(float(c)) > 1e-6 for c in coeffs.values()):
            return k
    raise AssertionError('all Legendre rows are zero')


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_from_dict_stored_legendre_coeff_be9_mt2(be9_endf_dict):
    """Primary issue-#169 demonstration for MF4: write a JAX tracer
    into ``endf_dict[4][2]['a'][en_idx][L_idx]``, call
    ``compute_angdist_values`` with ``xp=jax``, and confirm
    ``jax.grad`` matches central finite-difference.

    Query energy is chosen just above the tracer's own ``E`` slot
    so the incident-energy interpolation actively touches the
    tracer's row (making the analytic derivative nonzero and
    finite-diff-resolvable)."""
    import jax
    import jax.numpy as jnp
    mt = 2
    d = be9_endf_dict
    a_dict = d[4][mt]['a']
    en_key = _find_first_nonzero_legendre_row(a_dict)
    en_val = float(d[4][mt]['E'][en_key])
    L_key = list(a_dict[en_key].keys())[0]
    original = float(a_dict[en_key][L_key])
    xp_jx = array_ns.get_backend('jax')
    energies = jnp.array([en_val * 1.1])
    mu = jnp.array([0.3])

    def loss(theta):
        d_t = copy.deepcopy(d)
        d_t[4][mt]['a'][en_key][L_key] = theta
        return jnp.sum(mf4py.compute_angdist_values(
            d_t, mt, energies, mu, True, xp=xp_jx,
        ))

    val = float(loss(jnp.array(original)))
    grad = float(jax.grad(loss)(jnp.array(original)))
    assert np.isfinite(grad)
    assert val > 0.0
    assert abs(grad) > 0.0
    eps = 1e-3 * abs(original) if original != 0.0 else 1e-6
    lp = float(loss(jnp.array(original + eps)))
    lm = float(loss(jnp.array(original - eps)))
    fd = (lp - lm) / (2.0 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-20)
