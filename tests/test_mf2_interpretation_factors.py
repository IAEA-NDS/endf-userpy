"""Penetration / shift / hard-sphere phase factors
(:mod:`mf2_interpretation_factors`): the scalar-L fast path must
return exactly what the vectorised masked-selection path returns,
and must not evaluate the other L branches.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf2_interpretation_factors as fac


RHO = np.concatenate([[0.0, 1e-300, 1e-12], np.geomspace(1e-6, 1e8, 200)])


def _backends():
    return [b for b in ('numpy', 'jax') if b in array_ns.available_backends()]


@pytest.mark.parametrize('backend', _backends())
@pytest.mark.parametrize('nl_max', [6, 8, 10])
@pytest.mark.parametrize('L', range(10))
def test_scalar_L_matches_array_L_bitwise(backend, nl_max, L):
    """Every scalar form of L (int, numpy int, 0-d array) gives
    bit-identical P, S, phi to the per-element array-L path,
    including L >= nl_max (clipped to the last Newton step) and
    rho = 0 / huge rho."""
    xp = array_ns.get_backend(backend)
    rho = xp.asarray(RHO)
    l_arr = np.full(RHO.shape, L)
    p_ref, s_ref = fac.pnt_shf(rho, l_arr, xp, nl_max)
    ph_ref = fac.phase(rho, l_arr, xp, nl_max)
    for l_scalar in (L, np.int32(L), np.array(L)):
        p, s = fac.pnt_shf(rho, l_scalar, xp, nl_max)
        ph = fac.phase(rho, l_scalar, xp, nl_max)
        for got, ref in ((p, p_ref), (s, s_ref), (ph, ph_ref)):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_static_l_classification():
    assert fac._static_l(3) == 3
    assert fac._static_l(np.int64(2)) == 2
    assert fac._static_l(np.array(4)) == 4
    assert fac._static_l(np.array([1, 2])) is None
    assert fac._static_l(-1) is None
    assert fac._static_l(True) is None


@pytest.mark.skipif('jax' not in array_ns.available_backends(),
                    reason='jax not installed')
def test_scalar_L_traces_only_its_own_branch():
    """For a concrete scalar L the traced program contains only that
    L's closed form: no stack of the six low-L branches, no masked
    selection, no Newton steps for L < 6. (The vectorised path traces
    all six branches plus nl_max - 6 Newton steps for every call; on
    U-235 those per-group calls were ~3 s of a 10 s eager profile.)"""
    import jax
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')
    rho = jnp.asarray(RHO)
    for L in (0, 1, 3, 5):
        for fn in (lambda r: fac.pnt_shf(r, L, xp),
                   lambda r: fac.phase(r, L, xp)):
            prims = [e.primitive.name for e in jax.make_jaxpr(fn)(rho).jaxpr.eqns]
            assert 'concatenate' not in prims, (L, prims)
            assert 'select_n' not in prims, (L, prims)
            assert len(prims) <= 30, (L, len(prims))
    # Traced L keeps working through the vectorised path (jit vs eager
    # XLA fusion: ulp-level differences only).
    out = jax.jit(lambda r, l: fac.pnt_shf(r, l, xp)[0])(rho, jnp.asarray(2))
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(fac.pnt_shf(rho, np.full(RHO.shape, 2), xp)[0]),
        rtol=1e-15, atol=0.0)
