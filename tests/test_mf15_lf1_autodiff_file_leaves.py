"""``jax.grad`` wrt MF15 LF=1 file-side leaves through the
``mf15_interpretation.compute_spectrum`` reconstruction entry
point.

MF15 LF=1 is the photon energy spectrum analog of MF5 LF=1. The
existing ``test_mf5_mf15_lf1_grad_wrt_E`` covers the QUERY-side E
autodiff; this file adds the complementary FILE-side pin for the
incident-energy mesh knot (``contrib['E'][idx]``), which requires
that the mesh flows through ``interp_tab2``'s traced-x variant
without a numpy force-cast.

The tabulated ``g`` leaf autodiff via
``contrib['rtfm1_tab'][row]['g'][k]`` already works (routes through
``interp_tab1``'s xp-aware path). It is pinned here for
completeness.

Uses the Nb-93 (n,x) MT=3 photon-production section from the adhoc
corpus.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf15_interpretation as mf15

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


def test_grad_wrt_mf15_lf1_mesh_knot_matches_fd(nb93_endf_dict):
    """Perturb the E-mesh knot at contribution row 8 of MF15/MT=3
    and confirm ``jax.grad`` through ``mf15.compute_spectrum``
    matches central FD.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    mt = 3
    contrib = nb93_endf_dict[15][mt]['subsection'][1]
    idx = 8
    original = float(contrib['E'][idx])
    ein_val = 0.5 * (original + float(contrib['E'][idx + 1]))
    eout = np.linspace(1e5, 1e6, 4)

    def loss(theta):
        d_t = copy.deepcopy(nb93_endf_dict)
        d_t[15][mt]['subsection'][1]['E'][idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = mf15.compute_spectrum(
                d_t, mt, np.array([ein_val]), eout, xp=xp_jax,
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


def test_grad_wrt_mf15_lf1_g_leaf_matches_fd(nb93_endf_dict):
    """Perturb one tabulated ``g`` value at contribution Ein-row 8,
    gamma-index 2 and confirm ``jax.grad`` matches central FD.

    Query eout brackets the perturbed Egamma knot so the leaf
    participates in the inner ``interp_tab1`` bracket at that
    Ein-row's spectrum panel.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    mt = 3
    contrib = nb93_endf_dict[15][mt]['subsection'][1]
    ein_row = 8
    g_idx = 2
    spec = contrib['rtfm1_tab'][ein_row]
    original = float(spec['g'][g_idx])
    eg_at_leaf = float(spec['Egamma'][g_idx])
    eout = np.array([eg_at_leaf * 0.7, eg_at_leaf, eg_at_leaf * 1.3])
    ein_val = 0.5 * (float(contrib['E'][ein_row])
                     + float(contrib['E'][ein_row + 1]))

    def loss(theta):
        d_t = copy.deepcopy(nb93_endf_dict)
        d_t[15][mt]['subsection'][1]['rtfm1_tab'][ein_row]['g'][g_idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = mf15.compute_spectrum(
                d_t, mt, np.array([ein_val]), eout, xp=xp_jax,
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
        'reconstruction at the chosen query grid'
    )
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)
