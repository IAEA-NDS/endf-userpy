"""Deeper ``interp_tab1`` tracer support inside ``interp_tab2``
(issue #169 tier-2).

Before this fix, ``interp_tab2`` ran the inner per-panel
``interp_tab1`` evaluations on numpy (no ``xp=xp`` forwarding), so
JAX tracers stored in the per-panel ``tab1_records[i][fp_name]``
list were silently dropped at the numpy boundary. Only tracers in
the outer x-mesh interpolation survived.

This test pins that inner tracer flow now propagates end-to-end,
using a minimal synthetic TAB2 case where the analytic derivative
wrt a single f-value leaf can be verified by inspection.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.interpolation import interp_tab2


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_interp_tab2_inner_tracer_flow_through_f_leaf():
    """A JAX tracer stored in ``tab1_records[0]['f'][1]`` survives
    the inner interp_tab1 call and ``jax.grad`` reaches it.

    Setup: TAB2 x-mesh [0, 10]; two TAB1 records at those x-nodes,
    each with y-mesh [0, 1, 2]. Query x=5.0, y=1.5 blends both
    records at 50/50 (x-side) and interpolates linearly between
    y=1 and y=2 in each record (50/50 y-side). Perturbing
    tab_a['f'][1] (the tab_a value at y=1) contributes
    x-weight (0.5) * y-weight (0.5) = 0.25 to the query output.
    """
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')

    tab1_a = {
        'y': [0.0, 1.0, 2.0], 'f': [1.0, 2.0, 3.0],
        'INT': [2], 'NBT': [3],
    }
    tab1_b = {
        'y': [0.0, 1.0, 2.0], 'f': [2.0, 4.0, 5.0],
        'INT': [2], 'NBT': [3],
    }

    def loss(theta):
        ta = {**tab1_a, 'f': [tab1_a['f'][0], theta, tab1_a['f'][2]]}
        res = interp_tab2(
            np.array([5.0]), np.array([1.5]),
            xp_mesh=np.array([0.0, 10.0]),
            int_arr=np.array([2]), nbt_arr=np.array([2]),
            tab1_records=[ta, tab1_b],
            yp_name='y', fp_name='f',
            xp=xp_jx,
        )
        return jnp.sum(res)

    sv = jnp.array(2.0)
    val = float(loss(sv))
    grad = float(jax.grad(loss)(sv))
    eps = 1e-4
    fd = (float(loss(sv + eps)) - float(loss(sv - eps))) / (2 * eps)

    assert np.isfinite(grad)
    # Analytic derivative wrt tab_a['f'][1] at (x=5, y=1.5): 0.25
    np.testing.assert_allclose(grad, 0.25, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(grad, fd, rtol=1e-6)
    # Analytic value at (x=5, y=1.5):
    #   tab_a at y=1.5 -> 0.5*2 + 0.5*3 = 2.5
    #   tab_b at y=1.5 -> 0.5*4 + 0.5*5 = 4.5
    #   blend x=5      -> 0.5*2.5 + 0.5*4.5 = 3.5
    assert abs(val - 3.5) < 1e-10
