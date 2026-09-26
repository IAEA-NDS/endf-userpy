"""``jax.grad`` wrt MF5 LF=1 file-side leaves through the top-level
``get_particle_production_dxs_dE`` API.

The reconstruction-layer LF=1 autodiff wrt QUERY-side E is already
pinned by ``test_mf5_mf15_lf1_grad_wrt_E``. This file adds
complementary pins for FILE-side leaves that a user would perturb
in a sensitivity study on a tabulated MF5 spectrum:

1. ``contrib['E'][idx]`` -- incident-energy mesh knot of the LF=1
   TAB2 record. Requires the interp_tab2 traced-x path to carry
   the mesh xp-native (fixed alongside this test).
2. ``contrib['spectrum'][row]['g'][k]`` -- tabulated outgoing
   spectrum value at a specific (Ein-row, Eout-index).

Both are checked against central FD with a sentinel that the FD
magnitude is meaningfully nonzero (avoids the zero-equals-zero
failure mode: a leaf whose interpolation bracket does not overlap
the query grid would give FD=0 and grad=0 trivially).

Uses Nb-93 (n,x) MT=91 continuum inelastic from the adhoc corpus
(single LF=1 contribution, ZAP=1 neutron).
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.quantities import get_particle_production_dxs_dE

from _corpus import resolve_nb93


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


@pytest.fixture(scope='module')
def nb93_endf_dict():
    path = resolve_nb93()
    if path is None:
        pytest.skip('Nb-93 corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def test_grad_wrt_mf5_lf1_mesh_knot_matches_fd(nb93_endf_dict):
    """Perturb the E-mesh knot at contribution row 4 and confirm
    ``jax.grad`` through ``get_particle_production_dxs_dE`` matches
    central FD.

    Query ein sits inside the panel (E[4], E[5]) so the perturbed
    knot participates in the outer TAB2 interpolation.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    contrib = nb93_endf_dict[5][91]['contribution'][1]
    idx = 4
    original = float(contrib['E'][idx])
    ein_val = 0.5 * (original + float(contrib['E'][idx + 1]))
    eout = np.array([2.0e4, 3.0e4, 5.0e4])

    def loss(theta):
        d_t = copy.deepcopy(nb93_endf_dict)
        d_t[5][91]['contribution'][1]['E'][idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_particle_production_dxs_dE(
                d_t, '(n,n_c)', 'n', np.array([ein_val]), eout,
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
        'reconstruction at the chosen query grid; test is uninformative'
    )
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)


def test_grad_wrt_mf5_lf1_g_leaf_matches_fd(nb93_endf_dict):
    """Perturb one tabulated spectrum value g[k] on Ein-row 4 and
    confirm ``jax.grad`` through ``get_particle_production_dxs_dE``
    matches central FD.

    Query eout brackets the perturbed spectrum's Eout knot so the
    leaf participates in the inner ``interp_tab1`` bracket at that
    Ein-row's spectrum panel.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    contrib = nb93_endf_dict[5][91]['contribution'][1]
    ein_row = 4
    g_idx = 2
    spec = contrib['spectrum'][ein_row]
    original = float(spec['g'][g_idx])
    ep_at_leaf = float(spec['Eout'][g_idx])
    # Query eout points that bracket the perturbed g leaf's Eout
    # knot, so the coefficient enters the panel interpolation.
    eout = np.array([ep_at_leaf * 0.7, ep_at_leaf, ep_at_leaf * 1.3])
    # Query ein sits inside the panel between Ein-rows 4 and 5, so
    # the outer TAB2 interp uses this spectrum row's tab1.
    ein_val = 0.5 * (float(contrib['E'][ein_row])
                     + float(contrib['E'][ein_row + 1]))

    def loss(theta):
        d_t = copy.deepcopy(nb93_endf_dict)
        d_t[5][91]['contribution'][1]['spectrum'][ein_row]['g'][g_idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_particle_production_dxs_dE(
                d_t, '(n,n_c)', 'n', np.array([ein_val]), eout,
                xp=xp_jax,
            )
        return jnp.sum(r)

    theta = jnp.array(original)
    val = float(loss(theta))
    grad = float(jax.grad(loss)(theta))
    eps = abs(original) * 1e-2 if original != 0.0 else 1e-10
    fd = (float(loss(jnp.array(original + eps)))
          - float(loss(jnp.array(original - eps)))) / (2.0 * eps)

    assert np.isfinite(val)
    assert np.isfinite(grad)
    assert abs(fd) > 0.0, (
        'FD is exactly zero: g leaf does not participate in the '
        'reconstruction at the chosen query grid; test is uninformative'
    )
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)
