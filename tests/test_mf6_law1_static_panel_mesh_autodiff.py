"""``jax.grad`` wrt MF6 LAW=1 continuum ``ei_mesh`` knot via the
static-panel entry point of ``mf6_law1_kernel.reconstruct``.

The MF6 LAW=1 top-level path (``get_particle_production_ddxs``)
does its own panel enumeration on a concrete numpy mesh via
``np.asarray(data.ei_mesh) + find_interval + np.unique(idcs)``,
which fundamentally cannot accept a tracer mesh. Refactoring the
section-wide default path to lift the tracer through the panel
loop is a larger arc (auxiliary concrete-mesh handoff or
``pure_callback`` for panel-finding).

The ``panel_idx=`` static-panel entry, which
``mf6_law1_kernel.reconstruct`` documents as the autodiff
entry point for query-side ``energies_in``, was itself broken for
mesh autodiff until the ``_f6law1con_panel_pair_bc`` fix that lands
alongside this test: it used ``np.asarray(data.ei_mesh)`` and
``float(...)`` for the panel-endpoint scalars ``e1`` / ``e2`` and
per-panel Ep endpoints. Replacing those with xp-native indexing
lets mesh-knot and Ep-endpoint autodiff flow through the two-panel
unit-base transform and outer E interpolation, at the price of the
user knowing which panel their query lies in.

This test pins the static-panel path only. Full top-level API
support for MF6 LAW=1 mesh autodiff remains a follow-up.
"""
from __future__ import annotations

import dataclasses
import os
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf6_law1_preproc, mf6_law1_kernel


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = 'tests/data_law1_adhoc/endfb81_n_Al-27.endf'
    if not os.path.exists(path):
        pytest.skip('Al-27 corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def test_grad_wrt_law1_ei_mesh_static_panel_matches_fd(al27_endf_dict):
    """Perturb ``data.ei_mesh[panel]`` on Al-27 MT=91 (LANG=2
    Kalbach-Mann) and confirm ``jax.grad`` through
    ``mf6_law1_kernel.reconstruct(..., panel_idx=panel)`` matches
    central FD.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    data = mf6_law1_preproc.mf6_law1_data_from_endf_dict(
        al27_endf_dict, 91, 1, xp=xp_jax,
    )
    panel = 7
    original = float(data.ei_mesh[panel])
    ein = jnp.array([
        0.5 * (original + float(data.ei_mesh[panel + 1]))
    ])
    eout = jnp.linspace(1e3, 5e5, 3)
    mu = jnp.array([-0.5, 0.0, 0.5])

    def loss(theta):
        new_ei = data.ei_mesh.at[panel].set(theta)
        data_t = dataclasses.replace(data, ei_mesh=new_ei)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = mf6_law1_kernel.reconstruct(
                data_t, ein, eout, mu, True, xp=xp_jax,
                panel_idx=panel,
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
        "the two-panel amplitude at this query grid; test is uninformative"
    )
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)
