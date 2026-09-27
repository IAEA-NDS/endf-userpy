"""``jax.grad`` wrt MF13 incident-energy mesh knot through the
photon-production total-XS reconstruction.

Small one-liner ``xp=xp`` fix in
``mf13_interpretation._compute_total_production_xs``. Mirrors the
mesh-knot pattern established for MF3 (PR #241), MF5 LF=1
(PR #243), MF4 LTT=2 (PR #244), MF15 LF=1 (PR #245), and
MF6 LAW=2 / MF14 LTT=1 (PR #248).

Uses B-11 (endfb81, MT=4) which has a 180-point MF13 E mesh.
"""
from __future__ import annotations

import copy
import os
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf13_interpretation as mf13


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


@pytest.fixture(scope='module')
def b11_endf_dict():
    path = 'tests/data_law1_adhoc/endfb81_n_B-11.endf'
    if not os.path.exists(path):
        pytest.skip('B-11 corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def test_grad_wrt_mf13_mesh_knot_matches_fd(b11_endf_dict):
    """Perturb the incident-energy mesh knot at row 100 of MF13/MT=4
    and confirm ``jax.grad`` through
    ``mf13._compute_total_production_xs`` matches central FD.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    mt = 4
    E = b11_endf_dict[13][mt]['E']
    idx = 100
    original = float(E[idx])
    ein_val = 0.5 * (original + float(E[idx + 1]))

    def loss(theta):
        d_t = copy.deepcopy(b11_endf_dict)
        d_t[13][mt]['E'][idx] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = mf13._compute_total_production_xs(
                d_t, mt, np.array([ein_val]), xp=xp_jax,
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
