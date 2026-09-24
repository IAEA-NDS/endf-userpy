"""Grad wrt query Ein through the MF5 LF=1 tabulated and MF15 LF=1
tabulated photon-spectrum reconstructions.

Both are direct-``interp_tab2`` reconstructions: the traced-x fast
path added for #201 PR-B already propagates ``jax.grad`` wrt Ein
through them, so this file is a Phase-1-style verification pin.

MF15 additionally had a stray ``np.asarray(energies_in)`` in
``_compute_prob`` that materialised the tracer; this file goes
with the fix that swaps it for ``xp.asarray``.

Corpus: Nb-93 (endfb81) has both MF5/MT=16 LF=1 (n,2n prompt
neutron spectrum) and MF15/MT=3 (nonelastic photon spectrum).
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf5_interpretation as mf5,
    mf15_interpretation as mf15,
)
from endf_userpy.primitives import array_ns

from _corpus import resolve_nb93


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def nb93_endf_dict():
    path = resolve_nb93()
    if path is None:
        pytest.skip('Nb-93 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def _fd5(f, x, h):
    return (-float(f(x + 2 * h)) + 8 * float(f(x + h))
            - 8 * float(f(x - h)) + float(f(x - 2 * h))) / (12 * h)


# -------------------------------------------------------------------
# MF5 LF=1 (tabulated fission neutron spectrum)
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf5_lf1_numpy_jax_parity(nb93_endf_dict):
    """xp=jax vs xp=numpy agree on the LF=1 tabulated spectrum."""
    import jax.numpy as jnp
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1.0e7, 1.5e7, 2.0e7, 5.0e7])
    eout = np.linspace(1e4, 5e6, 20)
    f_np = np.asarray(mf5.compute_spectrum(
        nb93_endf_dict, 16, ein, eout, xp=xp_np,
    ))
    f_jx = np.asarray(mf5.compute_spectrum(
        nb93_endf_dict, 16, jnp.asarray(ein), jnp.asarray(eout),
        xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-9, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('E_val', [1.5e7, 3.0e7, 6.0e7, 1.0e8])
def test_mf5_lf1_grad_wrt_E_matches_fd(nb93_endf_dict, E_val):
    """``jax.grad(sum(f))(E)`` matches central 5-point FD at Es
    across the LF=1 Ein panels (Nb-93 MT=16, threshold ~8.93 MeV)."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    eout = jnp.linspace(1e4, 5e6, 20)

    def loss(E_scalar):
        return jnp.sum(mf5.compute_spectrum(
            nb93_endf_dict, 16, jnp.array([E_scalar]), eout, xp=xp_jx,
        ))

    grad = float(jax.grad(loss)(jnp.array(E_val)))
    fd = _fd5(lambda v: loss(jnp.array(v)), E_val, E_val * 1e-4)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


# -------------------------------------------------------------------
# MF15 LF=1 (tabulated photon spectrum)
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf15_lf1_numpy_jax_parity(nb93_endf_dict):
    """xp=jax vs xp=numpy agree on the MF15 LF=1 photon spectrum."""
    import jax.numpy as jnp
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1.0e5, 1.0e6, 1.0e7, 1.5e7])
    eout = np.linspace(1e4, 5e6, 20)
    f_np = np.asarray(mf15.compute_spectrum(
        nb93_endf_dict, 3, ein, eout, xp=xp_np,
    ))
    f_jx = np.asarray(mf15.compute_spectrum(
        nb93_endf_dict, 3, jnp.asarray(ein), jnp.asarray(eout),
        xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-9, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('E_val', [1e5, 1e6, 1e7, 1.5e7])
def test_mf15_lf1_grad_wrt_E_matches_fd(nb93_endf_dict, E_val):
    """``jax.grad(sum(f))(E)`` matches central 5-point FD at Es
    across the MF15 LF=1 Ein panels. Fixed with the
    ``xp.asarray`` swap in ``_compute_prob`` accompanying this
    PR."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    eout = jnp.linspace(1e4, 5e6, 20)

    def loss(E_scalar):
        return jnp.sum(mf15.compute_spectrum(
            nb93_endf_dict, 3, jnp.array([E_scalar]), eout, xp=xp_jx,
        ))

    grad = float(jax.grad(loss)(jnp.array(E_val)))
    fd = _fd5(lambda v: loss(jnp.array(v)), E_val, E_val * 1e-4)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)
