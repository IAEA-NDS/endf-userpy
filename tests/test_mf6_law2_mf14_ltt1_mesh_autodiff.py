"""``jax.grad`` wrt file-side incident-energy mesh knots for
MF6 LAW=2 (discrete two-body) and MF14 LTT=1 (photon angular).

Continues the mesh-knot autodiff widening pattern established for
MF3 (PR #241), MF5 LF=1 (PR #243), MF4 LTT=2 (PR #244), and MF15
LF=1 (PR #245). Both fixes here are one-liner ``xp=xp``
plumbing in the preproc plus the ``_filter_energies_in`` helper
short-circuit shared with the MF4 arc.

Coverage:

1. MF6 LAW=2 mesh knot via top-level ``get_particle_production_dxs_dmu``
   on Al-27 MT=51 (first discrete inelastic level).
2. MF14 LTT=1 per-line E-mesh knot via ``mf14.compute_angdist_values``
   on N-14 MT=4 (Legendre photon angular).
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf14_interpretation as mf14
from endf_userpy.quantities import get_particle_production_dxs_dmu

from _corpus import resolve_al27


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
def n14_endf_dict():
    import os
    path = 'tests/data_law1_adhoc/endfb81_n_N-14.endf'
    if not os.path.exists(path):
        pytest.skip('N-14 corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def test_grad_wrt_mf6_law2_mesh_knot_matches_fd(al27_endf_dict):
    """Perturb ``subsec['E'][idx]`` on Al-27 MT=51 LAW=2 and
    confirm ``jax.grad`` through ``get_particle_production_dxs_dmu``
    matches central FD.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    subsec = al27_endf_dict[6][51]['subsection'][1]
    E = subsec['E']
    idx = 4
    original = float(E[idx])
    ein_val = 0.5 * (original + float(E[idx + 1]))
    mu = np.array([0.0, 0.5])

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[6][51]['subsection'][1]['E'][idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_particle_production_dxs_dmu(
                d_t, '(n,n_1)', 'n', np.array([ein_val]), mu, xp=xp_jax,
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
        'FD is exactly zero: mesh knot does not participate in the '
        'reconstruction at the chosen query grid'
    )
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)


def test_grad_wrt_mf14_ltt1_per_line_mesh_knot_matches_fd(n14_endf_dict):
    """Perturb the per-photon-line Ein mesh knot ``mtsec['E'][eg_idx][k]``
    on N-14 MT=4 LTT=1 and confirm ``jax.grad`` through
    ``mf14.compute_angdist_values`` matches central FD.

    Uses photon-line index 42 (EG=4.915 MeV): the first line whose
    Legendre coefficients array has a nonzero entry (row 3, coef 2 =
    0.164). Query ein sits inside the ``(E[42][2], E[42][3])`` bracket
    so the perturbed knot ``E[42][3]`` participates in the coefficient
    interpolation.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    eg_idx = 42
    mesh_idx = 3
    original = float(n14_endf_dict[14][4]['E'][eg_idx][mesh_idx])
    ein_val = 0.5 * (
        float(n14_endf_dict[14][4]['E'][eg_idx][mesh_idx - 1]) + original
    )
    photon_e = np.array([float(n14_endf_dict[14][4]['EG'][eg_idx])])
    mu = np.array([-0.5, 0.0, 0.5])

    def loss(theta):
        d_t = copy.deepcopy(n14_endf_dict)
        d_t[14][4]['E'][eg_idx][mesh_idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = mf14.compute_angdist_values(
                d_t, 4, np.array([ein_val]), photon_e, mu, xp=xp_jax,
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
        'FD is exactly zero: mesh knot does not participate in the '
        'reconstruction at the chosen query grid'
    )
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)
