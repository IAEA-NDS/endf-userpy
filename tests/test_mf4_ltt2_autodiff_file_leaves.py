"""``jax.grad`` wrt MF4 LTT=2 file-side leaves through the
top-level ``get_particle_production_dxs_dmu`` API.

The existing ``test_mf4_autodiff_wrt_E_mu`` pins autodiff wrt the
QUERY-side E / mu for LTT=1 and LTT=2. This file adds the
complementary FILE-side pins for LTT=2 (tabulated angular
distribution):

1. ``mf4sec['E'][idx]`` -- incident-energy mesh knot of the TAB2
   record. Requires the ``pad_outside_angdist_values`` fast-path
   selector to tolerate a JAX-tracer mesh (fixed alongside this
   test).
2. ``mf4sec['angtable'][row]['f'][k]`` -- tabulated angular
   distribution value at a specific (Ein-row, mu-index).

Both are checked against central FD with a sentinel that FD is
meaningfully nonzero (avoids the zero-equals-zero failure mode
noted in PR #242).

Uses Al-27 (n,elastic) MT=2 LCT=2 (CM frame) from the adhoc
corpus: LTT=2 with a 60-point Ein mesh and 101-point mu grid per
row. Query mu is transformed to CM before the tabulated interp,
so the perturbation site must be picked at the CM mu bracket,
not the naive LAB mu index.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
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


def test_grad_wrt_mf4_ltt2_mesh_knot_matches_fd(al27_endf_dict):
    """Perturb the E-mesh knot at row 30 (Ein=13 MeV) of the
    MT=2 LTT=2 record and confirm ``jax.grad`` through
    ``get_particle_production_dxs_dmu`` matches central FD.

    Query ein sits inside the panel (E[30], E[31]) so the perturbed
    knot participates in the outer TAB2 interpolation.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    E = al27_endf_dict[4][2]['E']
    idx = 30
    original = float(E[idx])
    ein_val = 0.5 * (original + float(E[idx + 1]))
    mu_query = np.array([-0.495])

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[4][2]['E'][idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_particle_production_dxs_dmu(
                d_t, '(n,n_0)', 'n', np.array([ein_val]), mu_query,
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
        'FD is exactly zero: mesh knot does not participate in the '
        'reconstruction at the chosen query grid'
    )
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)


def test_grad_wrt_mf4_ltt2_f_leaf_matches_fd(al27_endf_dict):
    """Perturb one tabulated ``f`` value at row 30, mu-index 24
    (mu=-0.52) and confirm ``jax.grad`` through
    ``get_particle_production_dxs_dmu`` matches central FD.

    Al-27 elastic is stored in the CM frame (LCT=2); LAB mu=-0.495
    at Ein=13.5 MeV maps to CM mu ~= -0.523, which sits in the
    ``mu[23]=-0.54`` / ``mu[24]=-0.52`` bracket. Perturbing
    ``f[24]`` therefore contributes to the inner-interp result at
    this query. Picking a different mu index whose bracket does not
    contain the CM-transformed query mu would produce a spurious
    zero-equals-zero pass.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    row_key = 30
    f_idx = 24
    original = float(
        al27_endf_dict[4][2]['angtable'][row_key]['f'][f_idx]
    )
    ein_val = 1.35e7
    mu_query = np.array([-0.495])

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[4][2]['angtable'][row_key]['f'][f_idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_particle_production_dxs_dmu(
                d_t, '(n,n_0)', 'n', np.array([ein_val]), mu_query,
                xp=xp_jax,
            )
        return jnp.sum(r)

    theta = jnp.array(original)
    val = float(loss(theta))
    grad = float(jax.grad(loss)(theta))
    eps = abs(original) * 1e-3
    fd = (float(loss(jnp.array(original + eps)))
          - float(loss(jnp.array(original - eps)))) / (2.0 * eps)

    assert np.isfinite(val)
    assert np.isfinite(grad)
    assert abs(fd) > 0.0, (
        'FD is exactly zero: f leaf does not participate in the '
        'reconstruction at the chosen CM-transformed query mu'
    )
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)
