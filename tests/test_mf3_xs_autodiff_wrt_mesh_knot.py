"""``jax.grad`` wrt an MF3 XS mesh knot through the top-level
``get_reaction_xs`` API.

The existing ``test_mf3_xs_autodiff_wrt_E`` pins autodiff wrt the
QUERY-side incident energy (``x`` in ``interp_tab1``). This file
pins the complementary capability: autodiff wrt a FILE-SIDE mesh
knot (``xstab['E'][idx]``), which requires that the mesh flows
through the JAX-native interpolation path without a numpy
force-cast.

Panel-finding (the categorical "which interval" index) is
inherently non-differentiable, but the interpolated value between
knots is a smooth linear-in-knots function almost everywhere.
``jax.grad`` handles the split: the searchsort output is treated
as stop-gradient (its dtype is integer), while the mesh values at
the selected indices carry the gradient.

The query energy is chosen strictly interior to the interpolation
bracket enclosing the perturbed knot so the FD is well-defined
away from the knot boundary.
"""
from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.quantities import get_reaction_xs


DATA_DIR = Path(__file__).parent / 'data'


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


@pytest.fixture(scope='module')
def h1_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-001_H_001.endf'),
    )


def test_mf3_grad_wrt_mesh_knot_matches_fd(h1_endf_dict):
    """Perturb one MF3 E-mesh knot on H-1 MT=102 (n,g) and confirm
    ``jax.grad`` through ``get_reaction_xs`` matches central FD.

    Query point is picked at the midpoint of the bracket enclosing
    the perturbed knot on the upper side (E[idx], E[idx+1]), where
    the interpolated XS is a smooth function of E[idx].
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    E_list = h1_endf_dict[3][102]['xstable']['E']
    idx = len(E_list) // 2
    original = float(E_list[idx])
    e_next = float(E_list[idx + 1])
    ein_val = 0.5 * (original + e_next)

    def loss(theta):
        d_t = copy.deepcopy(h1_endf_dict)
        d_t[3][102]['xstable']['E'][idx] = theta
        ein = jnp.asarray(np.array([ein_val]))
        xs = get_reaction_xs(d_t, '(n,g)', ein, xp=xp_jax)
        return jnp.sum(xs)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        theta = jnp.array(original)
        val = float(loss(theta))
        grad = float(jax.grad(loss)(theta))
        eps = original * 1e-4
        lp = float(loss(jnp.array(original + eps)))
        lm = float(loss(jnp.array(original - eps)))
        fd = (lp - lm) / (2.0 * eps)

    assert np.isfinite(val)
    assert np.isfinite(grad)
    assert grad != 0.0, (
        'gradient is zero: mesh tracer did not propagate through '
        'get_reaction_xs into the interpolation kernel'
    )
    assert fd != 0.0, 'FD is zero: chose a knot with no local sensitivity'
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)


def test_mf3_grad_wrt_multiple_mesh_knots_via_sum(h1_endf_dict):
    """Sum-of-values wrt a leaf that is written into two mesh knots
    tests that ``jax.grad`` correctly accumulates contributions from
    both perturbation sites, ruling out an implementation that
    silently ties gradients only to the last-written leaf.
    """
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    E_list = h1_endf_dict[3][102]['xstable']['E']
    idx_a, idx_b = len(E_list) // 3, 2 * len(E_list) // 3
    e_a, e_b = float(E_list[idx_a]), float(E_list[idx_b])
    ein_a = 0.5 * (e_a + float(E_list[idx_a + 1]))
    ein_b = 0.5 * (e_b + float(E_list[idx_b + 1]))

    def loss(theta):
        d_t = copy.deepcopy(h1_endf_dict)
        d_t[3][102]['xstable']['E'][idx_a] = theta + (e_a - 1.0)
        d_t[3][102]['xstable']['E'][idx_b] = theta + (e_b - 1.0)
        ein = jnp.asarray(np.array([ein_a, ein_b]))
        return jnp.sum(get_reaction_xs(d_t, '(n,g)', ein, xp=xp_jax))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        theta = jnp.array(1.0)
        grad = float(jax.grad(loss)(theta))
        eps = 1e-4
        lp = float(loss(jnp.array(1.0 + eps)))
        lm = float(loss(jnp.array(1.0 - eps)))
        fd = (lp - lm) / (2.0 * eps)

    assert np.isfinite(grad)
    assert grad != 0.0
    assert fd != 0.0
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-30)
