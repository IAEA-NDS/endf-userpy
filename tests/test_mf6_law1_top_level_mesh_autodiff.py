"""``jax.grad`` wrt MF6 LAW=1 file-side ``E`` (incident-energy
mesh knot) through the top-level ``get_particle_production_ddxs``
API — the section-wide default path.

Companion to ``test_mf6_law1_static_panel_mesh_autodiff`` (which
covered the ``panel_idx=`` reconstruction-layer entry). This test
exercises the code path a user calling the top-level dispatch
hits, without needing to know which panel their query lies in.

The default-path kernel used to force-cast ``data.ei_mesh`` to
numpy via ``np.asarray`` and enumerate panels via
``np.unique(find_interval(...))``, which raised on a tracer
mesh. The refactor lands a full-eval ``xp.where`` fallback path
that triggers under a tracer mesh; the concrete-mesh code path
(the fast sparse-scatter loop) is preserved for numpy and
concrete-JAX callers.
"""
from __future__ import annotations

import copy
import os
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.quantities import get_particle_production_ddxs


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


def test_grad_wrt_mf6_law1_mesh_knot_via_top_level_matches_fd(al27_endf_dict):
    """Perturb ``endf_dict[6][91]['subsection'][1]['E'][idx]`` and
    confirm ``jax.grad`` through ``get_particle_production_ddxs``
    matches central FD.

    Uses Al-27 MT=91 (continuum inelastic, LANG=2 Kalbach). idx=8
    is a mid-mesh knot; query ein sits inside the (E[8], E[9])
    bracket and query eout covers the low-Ep panel that carries
    the strongest Kalbach f0 sensitivity to a small mesh shift.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    sub = al27_endf_dict[6][91]['subsection'][1]
    E_dict = sub['E']
    idx = 8
    original = float(E_dict[idx])
    ein_val = 0.5 * (original + float(E_dict[idx + 1]))
    ep_vals = list(sub['Ep'][idx].values())
    eout = np.linspace(max(1e3, ep_vals[0] * 1.1), ep_vals[-1] * 0.9, 3)
    mu = np.array([-0.5, 0.0, 0.5])

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[6][91]['subsection'][1]['E'][idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_particle_production_ddxs(
                d_t, '(n,n_c)', 'n',
                np.array([ein_val]), eout, mu, xp=xp_jax,
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
