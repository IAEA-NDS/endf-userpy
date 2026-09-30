"""``jax.grad`` wrt MF6 LAW=7 ``ei_mesh`` knot through both the
reconstruction layer and the top-level API, with ``jax.jit``
composability probed for each.

LAW=7 stores an outgoing-energy pdf ``f(E_in, mu, E')`` as a
tabulation over ``(E_in_i, mu_j)`` slices, each with its own E'
mesh. The outer Ein axis is the natural mesh-knot autodiff
target. The per-slice mu meshes and per-cell Ep/f tab1 records
are handled by the numpy inner path in
``_get_dist2d_from_subsec_law7_traced_x`` and are not covered
here (separate scope, since the traced kernel keeps them numpy
to bound the compiled graph size).

Uses JEFF-4.0 H-2 (n,2n) MT=16 — the canonical single-subsection
LAW=7 in the adhoc corpus.
"""
from __future__ import annotations

import copy
import os
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf6_interpretation_subsecs as mf6subsec
from endf_userpy.quantities import get_particle_production_ddxs
from endf_userpy.run_options import RunOptions


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


@pytest.fixture(scope='module')
def h2_endf_dict():
    path = 'tests/data_law1_adhoc/jeff40_n_H-2.endf'
    if not os.path.exists(path):
        pytest.skip('H-2 corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def test_law7_reconstruction_layer_ei_mesh_matches_fd(h2_endf_dict):
    """Perturb ``sub['E'][idx]`` on H-2 MT=16 LAW=7 and confirm
    ``jax.grad`` through ``get_dist2d_from_subsec_law7`` matches
    central FD. This exercises the fix in the traced-x fast path.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    sub = h2_endf_dict[6][16]['subsection'][1]
    E = sub['E']
    idx = 2
    original = float(E[idx])
    ein_val = 0.5 * (original + float(E[idx + 1]))
    eout = np.array([1e5, 5e5, 1e6])
    mu = np.array([-0.5, 0.0, 0.5])

    def loss(theta):
        d_t = copy.deepcopy(h2_endf_dict)
        d_t[6][16]['subsection'][1]['E'][idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = mf6subsec.get_dist2d_from_subsec_law7(
                d_t, 16, 1, np.array([ein_val]), eout, mu, True,
                xp=xp_jax,
            )
        return jnp.sum(r)

    theta = jnp.array(original)
    val = float(loss(theta))
    grad = float(jax.grad(loss)(theta))
    eps = original * 1e-4
    fd = (float(loss(jnp.array(original + eps)))
          - float(loss(jnp.array(original - eps)))) / (2.0 * eps)

    assert np.isfinite(val)
    assert np.isfinite(grad)
    assert abs(fd) > 0.0, (
        'FD is exactly zero: the perturbed mesh knot does not affect '
        'the reconstructed DDX at this query grid'
    )
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)


def test_law7_top_level_ei_mesh_matches_fd(h2_endf_dict):
    """Same perturbation, but reached through
    ``get_particle_production_ddxs('(n,2n)', 'n', ...)``.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    E = h2_endf_dict[6][16]['subsection'][1]['E']
    idx = 2
    original = float(E[idx])
    ein_val = 0.5 * (original + float(E[idx + 1]))
    eout = np.array([1e5, 5e5, 1e6])
    mu = np.array([-0.5, 0.0, 0.5])

    def loss(theta):
        d_t = copy.deepcopy(h2_endf_dict)
        d_t[6][16]['subsection'][1]['E'][idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_particle_production_ddxs(d_t, '(n,2n)', 'n', np.array([ein_val]), eout, mu, options=RunOptions(backend=xp_jax))
        return jnp.sum(r)

    theta = jnp.array(original)
    val = float(loss(theta))
    grad = float(jax.grad(loss)(theta))
    eps = original * 1e-4
    fd = (float(loss(jnp.array(original + eps)))
          - float(loss(jnp.array(original - eps)))) / (2.0 * eps)

    assert np.isfinite(val)
    assert np.isfinite(grad)
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)


def test_law7_top_level_ei_mesh_jit_composability(h2_endf_dict):
    """``jax.grad(jax.jit(loss))`` and ``jax.jit(jax.grad(loss))``
    both work end-to-end for LAW=7 mesh-knot autodiff through the
    top-level API. Matches the composability guarantee added for
    MF6 LAW=1 in PR #254.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    E = h2_endf_dict[6][16]['subsection'][1]['E']
    idx = 2
    original = float(E[idx])
    ein_val = 0.5 * (original + float(E[idx + 1]))
    eout = np.array([1e5, 5e5, 1e6])
    mu = np.array([-0.5, 0.0, 0.5])

    def loss(theta):
        d_t = copy.deepcopy(h2_endf_dict)
        d_t[6][16]['subsection'][1]['E'][idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_particle_production_ddxs(d_t, '(n,2n)', 'n', np.array([ein_val]), eout, mu, options=RunOptions(backend=xp_jax))
        return jnp.sum(r)

    theta = jnp.array(original)
    eps = original * 1e-4
    fd = (float(loss(jnp.array(original + eps)))
          - float(loss(jnp.array(original - eps)))) / (2.0 * eps)
    assert abs(fd) > 0.0

    g_grad_of_jit = float(jax.grad(jax.jit(loss))(theta))
    jit_grad = jax.jit(jax.grad(loss))
    _ = jit_grad(theta)  # warmup compile
    g_jit_of_grad = float(jit_grad(theta))

    for label, val in (
        ('grad(jit(loss))', g_grad_of_jit),
        ('jit(grad(loss))', g_jit_of_grad),
    ):
        assert np.isfinite(val), f'{label} returned non-finite {val}'
        np.testing.assert_allclose(val, fd, rtol=5e-3, atol=1e-30,
                                    err_msg=f'{label} vs FD')
