"""``jax.grad`` wrt MF2/MT151 LRU=1 LRF=2 (MLBW) scattering-radius
AP and per-L mass-ratio AWRI leaves through the top-level
``get_reaction_xs(options=RunOptions(include_resonance=True))`` API.

Extends the resonance-parameter autodiff coverage (already pinned
for ER / Γn / Γγ / Γf in ``test_mf2_resonance_autodiff_wrt_E`` and
``test_get_reaction_xs_backend_agnostic``) to the scattering
radius and channel mass-ratio, which are common evaluation
fitting parameters (SAMMY, EMPIRE, ...).

Requires the Nb-93 adhoc corpus (LRU=1 LRF=2 MLBW).
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.quantities import get_reaction_xs
from endf_userpy.run_options import RunOptions

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


def test_grad_wrt_mlbw_AP_matches_fd(nb93_endf_dict):
    """Perturb the range-level scattering radius AP and confirm
    ``jax.grad`` through ``get_reaction_xs(options=RunOptions(include_resonance=True))``
    matches central FD.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    d_range = nb93_endf_dict[2][151]['isotope'][1]['range'][1]
    original = float(d_range['AP'])
    ein = np.array([1e-3, 1.0, 35.0, 100.0])

    def loss(theta):
        d_t = copy.deepcopy(nb93_endf_dict)
        d_t[2][151]['isotope'][1]['range'][1]['AP'] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_reaction_xs(d_t, '(n,total)', ein, options=RunOptions(include_resonance=True, backend=xp_jax))
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
        'FD is exactly zero: AP does not contribute to the sum-over-'
        'ein cross section at this query grid; test is uninformative'
    )
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)


def test_grad_wrt_mlbw_AWRI_matches_fd(nb93_endf_dict):
    """Perturb the per-L AWRI (channel mass ratio) and confirm
    ``jax.grad`` through ``get_reaction_xs(options=RunOptions(include_resonance=True))``
    matches central FD.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    lg = (
        nb93_endf_dict[2][151]['isotope'][1]['range'][1].get('l_group')
        or nb93_endf_dict[2][151]['isotope'][1]['range'][1]['spingroup']
    )
    original = float(lg[1]['AWRI'])
    ein = np.array([1.0, 35.0, 100.0])

    def loss(theta):
        d_t = copy.deepcopy(nb93_endf_dict)
        lg2 = (
            d_t[2][151]['isotope'][1]['range'][1].get('l_group')
            or d_t[2][151]['isotope'][1]['range'][1]['spingroup']
        )
        lg2[1]['AWRI'] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_reaction_xs(d_t, '(n,total)', ein, options=RunOptions(include_resonance=True, backend=xp_jax))
        return jnp.sum(r)

    theta = jnp.array(original)
    val = float(loss(theta))
    grad = float(jax.grad(loss)(theta))
    eps = original * 1e-6
    fd = (float(loss(jnp.array(original + eps)))
          - float(loss(jnp.array(original - eps)))) / (2.0 * eps)

    assert np.isfinite(val)
    assert np.isfinite(grad)
    assert abs(fd) > 0.0, (
        'FD is exactly zero: AWRI does not affect the reconstructed '
        'XS at this query grid'
    )
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)
