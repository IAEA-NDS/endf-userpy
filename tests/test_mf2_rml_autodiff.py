"""MF2 LRU=1 LRF=7 (R-Matrix Limited, KRM=3) autodiff wrt resonance
parameters on a real corpus file.

Companion to :mod:`test_mf2_rml_reconstruct` (which covers grad
wrt ``res_er`` on a synthetic dataclass). Pinned coverage:

- Numpy / jax bit-parity on JEFF-4.0 Cu-63 (six spin groups, 1098
  resonances).
- ``jax.grad`` wrt the resonance parameters ``res_er`` and
  ``res_gam``.
- ``jax.grad`` wrt the fittable geometry leaves ``abn`` (isotopic
  abundance), ``ki`` (target-mass-derived wavenumber prefactor),
  and ``ch_ape`` (per-channel penetration radius).
- ``jax.grad(jax.jit(loss))`` and ``jax.jit(jax.grad(loss))`` both
  compose across these leaves.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf2_interpretation_rml as rml
from endf_userpy.mfsec_interpretation import (
    mf2_interpretation_rml_preproc as rml_pre,
)


CU63_PATH = 'tests/data_law1_adhoc/jeff40_n_Cu-63.endf'


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


@pytest.fixture(scope='module')
def cu63_rml_data():
    if not os.path.exists(CU63_PATH):
        pytest.skip('Cu-63 corpus not available')
    from endf_parserpy import EndfParserCpp
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(CU63_PATH)
    xp_jx = array_ns.get_backend('jax')
    return (
        rml_pre.rml_data_from_endf_dict(d),
        rml_pre.rml_data_from_endf_dict(d, xp=xp_jx),
    )


def test_rml_numpy_jax_parity_cu63(cu63_rml_data):
    data_np, data_jx = cu63_rml_data
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([100.0, 500.0, 1e4, 5e4])
    r_np = rml.reconstruct(data_np, ein, xp_np)
    r_jx = rml.reconstruct(data_jx, ein, xp_jx)
    for k in ('tot', 'sct', 'cap', 'fis'):
        if k in r_np:
            np.testing.assert_allclose(
                np.asarray(r_np[k]), np.asarray(r_jx[k]),
                rtol=1e-12, atol=1e-14,
            )


def test_rml_grad_wrt_res_er_shift_cu63(cu63_rml_data):
    """``jax.grad`` wrt a shift added to res_er[0] matches central FD."""
    import jax
    import jax.numpy as jnp
    _, data_jx = cu63_rml_data
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([100.0, 500.0])

    def loss(shift):
        d2 = replace(data_jx, res_er=data_jx.res_er.at[0].add(shift))
        return jnp.sum(rml.reconstruct(d2, ein, xp_jx)['tot'])

    val = float(loss(jnp.array(0.0)))
    grad = float(jax.grad(loss)(jnp.array(0.0)))
    eps = 1.0
    fd = (
        float(loss(jnp.array(eps)))
        - float(loss(jnp.array(-eps)))
    ) / (2 * eps)
    assert val > 0.0
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-20)


def test_rml_grad_wrt_res_gam_cu63(cu63_rml_data):
    """``jax.grad`` wrt a channel-width leaf matches central FD."""
    import jax
    import jax.numpy as jnp
    _, data_jx = cu63_rml_data
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([100.0, 500.0])
    gam_orig = float(data_jx.res_gam[0, 1])  # first resonance, second channel

    def loss(x):
        new_gam = data_jx.res_gam.at[0, 1].set(x)
        d2 = replace(data_jx, res_gam=new_gam)
        return jnp.sum(rml.reconstruct(d2, ein, xp_jx)['tot'])

    val = float(loss(jnp.array(gam_orig)))
    grad = float(jax.grad(loss)(jnp.array(gam_orig)))
    eps = abs(gam_orig) * 1e-3
    fd = (
        float(loss(jnp.array(gam_orig + eps)))
        - float(loss(jnp.array(gam_orig - eps)))
    ) / (2 * eps)
    assert val > 0.0
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-20)


def test_rml_grad_wrt_abn_cu63(cu63_rml_data):
    """``jax.grad`` wrt isotopic abundance (per-isotope prefactor)
    matches central FD."""
    import jax
    import jax.numpy as jnp
    _, data_jx = cu63_rml_data
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([100.0, 500.0])
    abn0 = float(data_jx.abn)

    def loss(x):
        d2 = replace(data_jx, abn=x)
        return jnp.sum(rml.reconstruct(d2, ein, xp_jx)['tot'])

    grad = float(jax.grad(loss)(jnp.array(abn0)))
    eps = abs(abn0) * 1e-3
    fd = (
        float(loss(jnp.array(abn0 + eps)))
        - float(loss(jnp.array(abn0 - eps)))
    ) / (2 * eps)
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-20)


def test_rml_grad_wrt_ki_cu63(cu63_rml_data):
    """``jax.grad`` wrt the wavenumber prefactor ``ki`` (derived
    smoothly from the target mass ``pp_mb``) matches central FD."""
    import jax
    import jax.numpy as jnp
    _, data_jx = cu63_rml_data
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([100.0, 500.0])
    ki0 = float(data_jx.ki)

    def loss(x):
        d2 = replace(data_jx, ki=x)
        return jnp.sum(rml.reconstruct(d2, ein, xp_jx)['tot'])

    grad = float(jax.grad(loss)(jnp.array(ki0)))
    eps = abs(ki0) * 1e-3
    fd = (
        float(loss(jnp.array(ki0 + eps)))
        - float(loss(jnp.array(ki0 - eps)))
    ) / (2 * eps)
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-20)


def test_rml_grad_wrt_ch_ape_cu63(cu63_rml_data):
    """``jax.grad`` wrt a per-channel penetration radius matches
    central FD. Channel radii are routinely fit against thermal
    cross sections and coherent scattering lengths."""
    import jax
    import jax.numpy as jnp
    _, data_jx = cu63_rml_data
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([100.0, 500.0])
    ape0 = float(data_jx.ch_ape[0, 1])

    def loss(x):
        new_ape = data_jx.ch_ape.at[0, 1].set(x)
        d2 = replace(data_jx, ch_ape=new_ape)
        return jnp.sum(rml.reconstruct(d2, ein, xp_jx)['tot'])

    grad = float(jax.grad(loss)(jnp.array(ape0)))
    eps = abs(ape0) * 1e-3
    fd = (
        float(loss(jnp.array(ape0 + eps)))
        - float(loss(jnp.array(ape0 - eps)))
    ) / (2 * eps)
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-20)


def test_rml_jit_composability_ch_ape(cu63_rml_data):
    """jit compose wrt a channel-radius leaf. Regression: the
    pre-fix preproc built ``ch_ape`` as a numpy array with indexed
    assignments, so a tracer channel-radius fed through
    ``replace(data, ch_ape=<traced>)`` never reached the
    reconstruction. Fix builds ``ch_ape`` xp-native (padded
    row-list stacked with xp.stack)."""
    import jax
    import jax.numpy as jnp
    _, data_jx = cu63_rml_data
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([100.0, 500.0])
    ape0 = float(data_jx.ch_ape[0, 1])

    def loss(x):
        new_ape = data_jx.ch_ape.at[0, 1].set(x)
        d2 = replace(data_jx, ch_ape=new_ape)
        return jnp.sum(rml.reconstruct(d2, ein, xp_jx)['tot'])

    x = jnp.array(ape0)
    g_eager = float(jax.grad(loss)(x))
    g_grad_of_jit = float(jax.grad(jax.jit(loss))(x))
    jit_grad = jax.jit(jax.grad(loss))
    _ = jit_grad(x)
    g_jit_of_grad = float(jit_grad(x))
    for label, val in (
        ('grad(jit(loss))', g_grad_of_jit),
        ('jit(grad(loss))', g_jit_of_grad),
    ):
        np.testing.assert_allclose(
            val, g_eager, rtol=1e-6, atol=1e-20,
            err_msg=f'{label} vs eager grad',
        )


def test_rml_jit_composability_res_er(cu63_rml_data):
    """``jax.grad(jax.jit(loss))`` and ``jax.jit(jax.grad(loss))``
    both match eager grad.

    Regression: the pre-fix preproc routed ``pp_ma`` / ``pp_mb`` etc.
    through xp, so under jit ``float(data.pp_ma[ppi - 1])`` in
    ``_channel_kind`` hit ConcretizationTypeError. Fix keeps
    structural metadata on numpy.
    """
    import jax
    import jax.numpy as jnp
    _, data_jx = cu63_rml_data
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([100.0, 500.0])

    def loss(shift):
        d2 = replace(data_jx, res_er=data_jx.res_er.at[0].add(shift))
        return jnp.sum(rml.reconstruct(d2, ein, xp_jx)['tot'])

    x = jnp.array(0.0)
    g_eager = float(jax.grad(loss)(x))
    g_grad_of_jit = float(jax.grad(jax.jit(loss))(x))
    jit_grad = jax.jit(jax.grad(loss))
    _ = jit_grad(x)  # warmup
    g_jit_of_grad = float(jit_grad(x))
    for label, val in (
        ('grad(jit(loss))', g_grad_of_jit),
        ('jit(grad(loss))', g_jit_of_grad),
    ):
        assert np.isfinite(val)
        np.testing.assert_allclose(
            val, g_eager, rtol=1e-6, atol=1e-20,
            err_msg=f'{label} vs eager grad',
        )
