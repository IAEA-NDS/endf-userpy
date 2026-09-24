"""Phase-1 autodiff-coverage pins for MF6 LAW=2 (discrete two-body).

Current state (as of the #201 Legendre-path fix):

- numpy vs jax parity on concrete inputs: works.
- ``jax.grad`` wrt file-side Legendre coefficients: works
  (pinned in ``test_mf6_law2_autodiff_from_coeffs.py``, issue #154).
- ``jax.grad`` wrt query-side E or mu on LANG=0 (Legendre): works
  after the #201 Legendre-path fix (this PR).
- ``jax.grad`` wrt query-side E or mu on LANG=12/14 (tabulated):
  still blocked on the ``interp_tab2`` traced-x fast path
  (#201 PR-B). Al-27 MT=51 sub=1 is LANG=0 so the tests below
  exercise the working path.
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


def _fd5(f, x, h):
    return (-float(f(x + 2 * h)) + 8 * float(f(x + h))
            - 8 * float(f(x - h)) + float(f(x - 2 * h))) / (12 * h)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('E_val', [1.5e6, 3.0e6, 8.0e6, 1.5e7, 5.0e7])
def test_law2_grad_wrt_E_matches_fd(al27_endf_dict, E_val):
    """``jax.grad(sum(f(E, mu)))(E)`` matches central FD on the
    LANG=0 Legendre path (unblocked by the #201 Legendre-path
    fix)."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    mu = jnp.linspace(-0.9, 0.9, 21)

    def loss(E_scalar):
        return jnp.sum(mf6subsec.get_angdist_from_subsec_law2(
            al27_endf_dict, 51, 1, jnp.array([E_scalar]), mu,
            to_lab=True, xp=xp_jx,
        ))

    grad = float(jax.grad(loss)(jnp.array(E_val)))
    fd = _fd5(lambda v: loss(jnp.array(v)), E_val, E_val * 1e-4)
    assert np.isfinite(grad)
    # atol=1e-6 covers near-stationary Es where the true grad is
    # small and 5-point FD noise (~1e-7 on this problem) dominates.
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('mu_val', [-0.7, -0.3, 0.0, 0.4, 0.8])
def test_law2_grad_wrt_mu_matches_fd(al27_endf_dict, mu_val):
    """``jax.grad(sum(f(E, mu)))(mu)`` matches central FD (unblocked
    by the #201 Legendre-path fix)."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([2.0e6, 5.0e6, 1.5e7])

    def loss(mu_scalar):
        return jnp.sum(mf6subsec.get_angdist_from_subsec_law2(
            al27_endf_dict, 51, 1, ein, jnp.array([mu_scalar]),
            to_lab=True, xp=xp_jx,
        ))

    grad = float(jax.grad(loss)(jnp.array(mu_val)))
    fd = _fd5(lambda v: loss(jnp.array(v)), mu_val, 1e-4)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)
