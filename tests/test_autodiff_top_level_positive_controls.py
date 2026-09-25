"""End-to-end autodiff positive controls at the top-level API.

The existing autodiff tests exercise the reconstruction-layer entry
points (``mf6subsec.get_dist2d_from_subsec_law1``,
``mf6_law1_kernel.reconstruct``, ``mf6subsec.get_angdist_from_subsec_law2``,
etc.) with tracers on file-side coefficients. This file walks the
whole ``endf_userpy.quantities.*`` API instead, so we know the
top-level entry points that users actually call preserve tracer
identity from a dict-stored leaf through to the final result.

Coverage:

1. MF6 LAW=1 continuum ``b`` leaf visible to
   ``get_particle_production_ddxs`` (unbroadened DDX summed across
   MTs, Al-27 MT=91 continuum contributes; requires adhoc corpus,
   skipped on CI).
2. MF6 LAW=2 discrete two-body Legendre ``A`` leaf visible to
   ``get_particle_production_dxs_dmu`` (Al-27 MT=51 first discrete
   inelastic level; requires adhoc corpus, skipped on CI).
3. MF4 LTT=1 Legendre ``a`` leaf visible to
   ``get_particle_production_dxs_dmu`` (Be-9 elastic MT=2,
   committed corpus, always runs).
4. ``jax.jit`` with a :class:`StaticEndfDict` as ``static_argnums``
   compiles ``get_reaction_xs`` once and reuses the compiled kernel
   across calls with different query grids (H-1 committed corpus,
   always runs).

The first three tests each do a central finite-difference
sanity-check against the injected leaf so a silent tracer-drop
somewhere in the top-level dispatch is caught.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.static_dict import wrap_endf_dict
from endf_userpy.quantities import (
    get_particle_production_ddxs,
    get_particle_production_dxs_dmu,
    get_reaction_xs,
)

from _corpus import resolve_al27


DATA_DIR = Path(__file__).parent / 'data'


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip('Al-27 corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def be9_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-004_Be_009.endf'),
    )


@pytest.fixture(scope='module')
def h1_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-001_H_001.endf'),
    )


def test_top_level_ddxs_grad_wrt_mf6_law1_b(al27_endf_dict):
    """Inject a JAX tracer at Al-27 MT=91 LAW=1 ``b[panel][ep][coef]``
    and confirm ``jax.grad`` through ``get_particle_production_ddxs``
    returns a finite non-zero gradient that matches central FD.

    MT=91 is the (n,n') continuum with LANG=2 Kalbach-Mann; the
    coefficient perturbed is ``f0`` at panel=1, ep_row=2, coef=0.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    # Perturbation site chosen empirically so the FD sensitivity is
    # meaningfully nonzero: panel 7 (Ein ~9.5-1e7 MeV bracket), ep_row 5,
    # coef 0 (the Kalbach f0 multiplicity leaf). Query Eout brackets the
    # perturbed Ep row so the coefficient participates in the local
    # interpolation.
    panel, ep_row, coef = 7, 5, 0
    subsec = al27_endf_dict[6][91]['subsection'][1]
    original = float(subsec['b'][panel][ep_row][coef])
    assert original != 0.0, 'sentinel: chose a zero-valued coefficient'
    ein = np.array([0.5 * (subsec['E'][panel] + subsec['E'][panel + 1])])
    ep_at_row = subsec['Ep'][panel][ep_row]
    eout = np.array([ep_at_row * 0.99, ep_at_row * 1.01])
    mu = np.array([-0.5, 0.0, 0.5])

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[6][91]['subsection'][1]['b'][panel][ep_row][coef] = theta
        out = get_particle_production_ddxs(
            d_t, '(n,n_c)', 'n', ein, eout, mu, xp=xp_jax,
        )
        return jnp.sum(out)

    val = float(loss(jnp.array(original)))
    grad = float(jax.grad(loss)(jnp.array(original)))
    assert np.isfinite(val)
    assert np.isfinite(grad)
    assert grad != 0.0, (
        'gradient is zero: tracer did not propagate through the '
        'top-level get_particle_production_ddxs path'
    )
    eps = original * 1e-4 if original != 0 else 1e-4
    fd = (float(loss(jnp.array(original + eps)))
          - float(loss(jnp.array(original - eps)))) / (2.0 * eps)
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


def test_top_level_dxs_dmu_grad_wrt_mf6_law2_A(al27_endf_dict):
    """Inject a JAX tracer at Al-27 MT=51 LAW=2 Legendre coefficient
    ``A[row][coef]`` and confirm ``jax.grad`` through
    ``get_particle_production_dxs_dmu`` returns a finite non-zero
    gradient that matches central FD.

    MT=51 is the first discrete inelastic level; LAW=2 LANG=0
    stores Legendre coefficients directly. dxs_dmu integrates out
    Eout so the discrete kinematic delta lands cleanly.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    row, coef = 14, 1
    original = float(
        al27_endf_dict[6][51]['subsection'][1]['A'][row][coef]
    )
    assert original != 0.0, 'sentinel: chose a zero-valued coefficient'
    ein = np.array([7.8e6, 8.2e6, 8.5e6])
    mu = np.array([-0.5, 0.0, 0.5])

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[6][51]['subsection'][1]['A'][row][coef] = theta
        out = get_particle_production_dxs_dmu(
            d_t, '(n,n_1)', 'n', ein, mu, xp=xp_jax,
        )
        return jnp.sum(out)

    val = float(loss(jnp.array(original)))
    grad = float(jax.grad(loss)(jnp.array(original)))
    assert np.isfinite(val)
    assert np.isfinite(grad)
    assert grad != 0.0, (
        'gradient is zero: tracer did not propagate through the '
        'top-level get_particle_production_dxs_dmu path'
    )
    eps = abs(original) * 1e-4 if original != 0 else 1e-4
    fd = (float(loss(jnp.array(original + eps)))
          - float(loss(jnp.array(original - eps)))) / (2.0 * eps)
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


def test_top_level_dxs_dmu_grad_wrt_mf4_ltt1_a(be9_endf_dict):
    """Inject a JAX tracer at Be-9 MF4/MT=2 LTT=1 Legendre
    coefficient ``a[row][coef]`` and confirm ``jax.grad`` through
    ``get_particle_production_dxs_dmu`` returns a finite non-zero
    gradient that matches central FD.

    Elastic scattering uses MF4 for the angular distribution. LTT=1
    (Legendre) stores coefficients at ``d[4][2]['a'][row][coef]``.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    row, coef = 80, 1
    original = float(be9_endf_dict[4][2]['a'][row][coef])
    assert original != 0.0, 'sentinel: chose a zero-valued coefficient'
    ein = np.array([7.6e6, 7.9e6])  # brackets row 80 (E=7.5e6) and row 81 (E=8.03e6)
    mu = np.array([-0.5, 0.0, 0.5])

    def loss(theta):
        d_t = copy.deepcopy(be9_endf_dict)
        d_t[4][2]['a'][row][coef] = theta
        out = get_particle_production_dxs_dmu(
            d_t, '(n,n_0)', 'n', ein, mu, xp=xp_jax,
        )
        return jnp.sum(out)

    val = float(loss(jnp.array(original)))
    grad = float(jax.grad(loss)(jnp.array(original)))
    assert np.isfinite(val)
    assert np.isfinite(grad)
    assert grad != 0.0, (
        'gradient is zero: tracer did not propagate through the '
        'top-level get_particle_production_dxs_dmu MF4 LTT=1 path'
    )
    eps = abs(original) * 1e-4 if original != 0 else 1e-4
    fd = (float(loss(jnp.array(original + eps)))
          - float(loss(jnp.array(original - eps)))) / (2.0 * eps)
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


def test_jax_jit_accepts_static_endf_dict_wrapper(h1_endf_dict):
    """``StaticEndfDict`` from :func:`wrap_endf_dict` is a valid
    ``static_argnums`` argument for ``jax.jit``: identity-hashable,
    so the compiled kernel is memoised by wrapper identity.

    Positive control that the second wrapper primitive use case
    (jit static-arg handle) works end-to-end through the top-level
    ``get_reaction_xs`` API.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    w = wrap_endf_dict(h1_endf_dict)

    def pipeline(wrapped, ein):
        return get_reaction_xs(wrapped, '(n,g)', ein, xp=xp_jax)

    pipeline_jit = jax.jit(pipeline, static_argnums=(0,))
    ein1 = jnp.array([1e4, 1e5, 1e6])
    ein2 = jnp.array([2e4, 2e5, 2e6])

    r1 = np.asarray(pipeline_jit(w, ein1))
    r2 = np.asarray(pipeline_jit(w, ein2))
    assert np.all(np.isfinite(r1))
    assert np.all(np.isfinite(r2))

    # Numeric parity with the raw (non-jit) path.
    r1_ref = np.asarray(get_reaction_xs(
        h1_endf_dict, '(n,g)', np.asarray(ein1), xp=xp_jax,
    ))
    np.testing.assert_allclose(r1, r1_ref, rtol=1e-5, atol=0)

    # A different wrapper around the same underlying dict has
    # distinct identity, so jit sees a fresh static and recompiles;
    # the numeric result must still match.
    w_alt = wrap_endf_dict(h1_endf_dict)
    assert w is not w_alt
    r_alt = np.asarray(pipeline_jit(w_alt, ein1))
    np.testing.assert_allclose(r_alt, r1, rtol=1e-5, atol=0)
