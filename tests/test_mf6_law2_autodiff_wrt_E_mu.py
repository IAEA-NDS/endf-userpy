"""Phase-1 autodiff-coverage pins for MF6 LAW=2 (discrete two-body).

Current state (as of Phase 1 sweep):

- numpy vs jax parity on concrete inputs: works. The LAW=2
  reconstruction (issue #158) is backend-agnostic when both
  ``energies_in`` and ``angle_cosines_out`` are concrete arrays.
- ``jax.grad`` wrt file-side Legendre coefficients: works
  (pinned in ``test_mf6_law2_autodiff_from_coeffs.py``, issue #154).
- ``jax.grad`` wrt query-side E or mu: does NOT work today.
  ``_law2_reconstruct_from_data`` calls
  ``evaluate_interp_legendre_polynomials(np.asarray(energies_in,
  dtype=float), np.asarray(mu_eff), ...)``, and the primitive
  itself does ``x = np.asarray(x)`` on entry. Both materialise
  jax tracers with ``TracerArrayConversionError``.

The fix is the same "query-side x-tracer path" pattern PR #192
added for ``endf_interp1d``: replace the panel-lookup on
concrete Ein with a nested ``xp.where`` per-panel evaluation.
Tracked as issue #201.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_subsecs as mf6subsec,
)
from endf_userpy.primitives import array_ns

from _corpus import resolve_al27


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip('Al-27 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law2_numpy_jax_parity(al27_endf_dict):
    """xp=jax and xp=numpy return the same values on a query grid
    spanning several Ein panels and mus in [-0.9, 0.9] on Al-27
    MT=51 sub=1 (LANG=0 Legendre, LCT=2 CM). Pins the working
    concrete-input path."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1.5e6, 3.0e6, 8.0e6, 1.5e7, 5e7], dtype=np.float64)
    mu = np.linspace(-0.9, 0.9, 21, dtype=np.float64)

    f_np = np.asarray(mf6subsec.get_angdist_from_subsec_law2(
        al27_endf_dict, 51, 1, ein, mu, to_lab=True, xp=xp_np,
    ))
    f_jx = np.asarray(mf6subsec.get_angdist_from_subsec_law2(
        al27_endf_dict, 51, 1, jnp.asarray(ein), jnp.asarray(mu),
        to_lab=True, xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-10, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law2_grad_wrt_E_currently_raises(al27_endf_dict):
    """Regression pin for the query-side E-tracer gap (Phase 1
    finding). ``_law2_reconstruct_from_data`` materialises the
    tracer via ``np.asarray(energies_in, dtype=float)`` before
    handing off to ``evaluate_interp_legendre_polynomials``, which
    itself calls ``x = np.asarray(x)`` on entry. Delete this test
    once the primitive gets its x-tracer path."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    mu = jnp.linspace(-0.9, 0.9, 21)

    def loss(E_scalar):
        return jnp.sum(mf6subsec.get_angdist_from_subsec_law2(
            al27_endf_dict, 51, 1, jnp.array([E_scalar]), mu,
            to_lab=True, xp=xp_jx,
        ))

    try:
        import jax.errors
        expected_exc = jax.errors.TracerArrayConversionError
    except AttributeError:
        expected_exc = Exception
    with pytest.raises(expected_exc):
        jax.grad(loss)(jnp.array(3.0e6))


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law2_grad_wrt_mu_currently_raises(al27_endf_dict):
    """Same regression pin on the mu axis: convert_angcos_to_cmsys
    output flows into np.asarray inside the LAW=2 kernel, which
    also fails on tracers. Delete when the primitive supports
    tracer x and mu."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([2.0e6, 5.0e6, 1.5e7])

    def loss(mu_scalar):
        return jnp.sum(mf6subsec.get_angdist_from_subsec_law2(
            al27_endf_dict, 51, 1, ein, jnp.array([mu_scalar]),
            to_lab=True, xp=xp_jx,
        ))

    try:
        import jax.errors
        expected_exc = jax.errors.TracerArrayConversionError
    except AttributeError:
        expected_exc = Exception
    with pytest.raises(expected_exc):
        jax.grad(loss)(jnp.array(0.3))
