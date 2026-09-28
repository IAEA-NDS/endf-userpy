"""Autodiff wrt LAW=7 table amplitudes through the E' integrator.

``integrate_law7_subsec_over_eout`` is the panel-exact
angle-differential entry point for MF6 LAW=7 subsections. This
module pins the property that under ``xp=jax`` reverse-mode
gradients flow cleanly through the ``f`` amplitudes of the four
bracketing tables, and that the numerical result is bit-identical
to the pre-existing pure-numpy path.

Uses ``tests/data/n-004_Be_009.endf`` MT=16 (Be-9 (n,2n),
single-subsection LAW=7 in the small committed corpus).
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf6_law7_integrals as mf6_law7
from endf_userpy.quantities_mt_zap.distribution1d_helpers import (
    integrate_mf6_dist2d_over_eout,
)


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


@pytest.fixture(scope='module')
def be9_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        'tests/data/n-004_Be_009.endf',
    )


# Cell chosen so the perturbation is actually inside the bracket for
# the probe query (Ein=2.7 MeV, mu near 0.05): Ein slice 3 (2.5 MeV)
# is the lower Ein bracket; mu bin 12 (mu=0.1) is one of the two mu
# tables that get picked at u=0.05; index 10 is a mid-range non-zero
# Ep entry.
_CELL = (3, 12, 10)
_EIN = 2.7e6
_MUS = np.array([0.05])


def _run_with_perturbed_f(endf_dict, cell, value, xp):
    """Deep-copy the dict, overwrite one f-entry, run the integrator."""
    Ein_i, mu_i, idx = cell
    d_t = copy.deepcopy(endf_dict)
    d_t[6][16]['subsection'][1]['table'][Ein_i][mu_i]['f'][idx] = (
        float(value)
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return mf6_law7.integrate_law7_subsec_over_eout(
            d_t, 16, 1, np.array([_EIN]), _MUS, True, xp=xp,
        )


def test_numpy_and_jax_produce_bit_identical_result(be9_endf_dict):
    """The xp threading must not disturb the pre-refactor numerics."""
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')
    orig = float(
        be9_endf_dict[6][16]['subsection'][1]['table'][_CELL[0]][_CELL[1]][
            'f'
        ][_CELL[2]]
    )
    res_np = _run_with_perturbed_f(be9_endf_dict, _CELL, orig, xp_np)
    res_jax = _run_with_perturbed_f(be9_endf_dict, _CELL, orig, xp_jax)
    np.testing.assert_allclose(
        np.asarray(res_np), np.asarray(res_jax), rtol=1e-12, atol=1e-30,
    )


def _grad_loss(endf_dict, cell):
    """Build a scalar loss that injects a jnp-array f-column so
    ``jax.grad`` can trace through ``theta`` at position ``idx``.
    """
    import jax.numpy as jnp
    xp_jax = array_ns.get_backend('jax')
    Ein_i, mu_i, idx = cell

    def loss(theta):
        d_t = copy.deepcopy(endf_dict)
        orig_f = d_t[6][16]['subsection'][1]['table'][Ein_i][mu_i]['f']
        new_f = jnp.asarray([float(v) for v in orig_f])
        new_f = new_f.at[idx].set(theta)
        d_t[6][16]['subsection'][1]['table'][Ein_i][mu_i]['f'] = new_f
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = mf6_law7.integrate_law7_subsec_over_eout(
                d_t, 16, 1, np.array([_EIN]), _MUS, True, xp=xp_jax,
            )
        return jnp.sum(r)

    return loss


def test_jax_grad_wrt_f_amplitude_matches_fd(be9_endf_dict):
    """``jax.grad`` wrt one f-entry inside the integration bracket
    matches central finite differences.
    """
    import jax
    import jax.numpy as jnp

    orig = float(
        be9_endf_dict[6][16]['subsection'][1]['table'][_CELL[0]][_CELL[1]][
            'f'
        ][_CELL[2]]
    )
    loss = _grad_loss(be9_endf_dict, _CELL)

    theta0 = jnp.array(orig)
    grad = float(jax.grad(loss)(theta0))
    eps = max(abs(orig) * 1e-3, 1e-10)
    fd = (
        float(loss(jnp.array(orig + eps)))
        - float(loss(jnp.array(orig - eps)))
    ) / (2.0 * eps)

    assert np.isfinite(grad)
    assert abs(fd) > 0.0, (
        'FD is exactly zero: the perturbed f-entry does not affect '
        'the integrated DDX at the probe query. Pick a different cell.'
    )
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-30)


def test_jax_grad_jit_composability(be9_endf_dict):
    """``jax.grad(jax.jit(loss))`` and ``jax.jit(jax.grad(loss))``
    both trace through the E' integrator and agree with the eager
    gradient.
    """
    import jax
    import jax.numpy as jnp

    orig = float(
        be9_endf_dict[6][16]['subsection'][1]['table'][_CELL[0]][_CELL[1]][
            'f'
        ][_CELL[2]]
    )
    loss = _grad_loss(be9_endf_dict, _CELL)
    theta0 = jnp.array(orig)

    g_eager = float(jax.grad(loss)(theta0))
    g_grad_of_jit = float(jax.grad(jax.jit(loss))(theta0))
    jit_grad = jax.jit(jax.grad(loss))
    _ = jit_grad(theta0)  # warmup compile
    g_jit_of_grad = float(jit_grad(theta0))

    for label, val in (
        ('grad(jit(loss))', g_grad_of_jit),
        ('jit(grad(loss))', g_jit_of_grad),
    ):
        assert np.isfinite(val), f'{label} returned non-finite {val}'
        np.testing.assert_allclose(
            val, g_eager, rtol=1e-6, atol=1e-30,
            err_msg=f'{label} disagrees with eager grad',
        )


def test_top_level_helper_returns_xp_array_on_h2():
    """``integrate_mf6_dist2d_over_eout`` on the single-subsection
    LAW=7 fast path (JEFF-4.0 H-2 (n,2n)) now returns an xp-native
    array directly (no numpy-materialise round-trip) and preserves
    the numeric result.
    """
    import os
    path = 'tests/data_law1_adhoc/jeff40_n_H-2.endf'
    if not os.path.exists(path):
        pytest.skip('H-2 adhoc corpus not present')
    import jax.numpy as jnp
    endf_dict = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    xp_jax = array_ns.get_backend('jax')
    xp_np = array_ns.get_backend('numpy')

    ein = np.array([5e6])
    mus = np.array([0.0, 0.5])
    r_np = integrate_mf6_dist2d_over_eout(
        endf_dict, 16, 1, ein, mus, True, xp=xp_np,
    )
    r_jax = integrate_mf6_dist2d_over_eout(
        endf_dict, 16, 1, ein, mus, True, xp=xp_jax,
    )
    assert isinstance(r_jax, jnp.ndarray)
    np.testing.assert_allclose(
        np.asarray(r_np), np.asarray(r_jax), rtol=1e-12, atol=1e-30,
    )
