"""``jax.grad`` wrt MF2/MT151 LRU=1 LRF=3 (Reich-Moore) file-side
scalar leaves through the top-level
``get_reaction_xs(include_resonance=True)`` API.

Complements the MLBW AP/AWRI autodiff pinned in PR #249. R-M's
per-L scattering radius ``APL`` and channel mass-ratio ``AWRI``
are the physically-fittable degrees of freedom exposed here;
the range-level ``AP`` is a fallback for files that do not
populate ``APL`` per L. All three now propagate JAX tracers
through the R-M preproc and the resonance reconstruction.

Uses Pb-208 (LRU=1 LRF=3, per-L APL populated) from the adhoc
corpus.
"""
from __future__ import annotations

import copy
import os
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.quantities import get_reaction_xs


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


@pytest.fixture(scope='module')
def pb208_endf_dict():
    path = 'tests/data_law1_adhoc/endfb81_n_Pb-208.endf'
    if not os.path.exists(path):
        pytest.skip('Pb-208 corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def _get_lg(endf_dict):
    r1 = endf_dict[2][151]['isotope'][1]['range'][1]
    return r1.get('l_group') or r1['spingroup']


def test_grad_wrt_rm_APL_matches_fd(pb208_endf_dict):
    """Perturb the L=0 per-L scattering radius APL and confirm
    ``jax.grad`` through ``get_reaction_xs(include_resonance=True)``
    matches central FD.

    Pb-208 populates APL for every L (0.967 and 0.975 in the two
    L-groups of the RRR range), so the reconstruction picks up APL
    via the ``_r_ap_for_L`` dispatch. Perturbing APL therefore
    changes both the effective scattering radius and the channel
    radius (via ``_channel_radius`` on NAPS=1).
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    lg = _get_lg(pb208_endf_dict)
    original = float(lg[1]['APL'])
    ein = np.array([100.0, 1000.0, 10000.0])

    def loss(theta):
        d_t = copy.deepcopy(pb208_endf_dict)
        _get_lg(d_t)[1]['APL'] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_reaction_xs(
                d_t, '(n,total)', ein,
                include_resonance=True, xp=xp_jax,
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
        'FD is exactly zero: APL does not contribute to the sum-over-'
        'ein cross section at this query grid; test is uninformative'
    )
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)


def test_grad_wrt_rm_AWRI_matches_fd(pb208_endf_dict):
    """Perturb the L=0 per-L AWRI (channel mass ratio) and confirm
    ``jax.grad`` through ``get_reaction_xs(include_resonance=True)``
    matches central FD.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    lg = _get_lg(pb208_endf_dict)
    original = float(lg[1]['AWRI'])
    ein = np.array([100.0, 1000.0, 10000.0])

    def loss(theta):
        d_t = copy.deepcopy(pb208_endf_dict)
        _get_lg(d_t)[1]['AWRI'] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = get_reaction_xs(
                d_t, '(n,total)', ein,
                include_resonance=True, xp=xp_jax,
            )
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
