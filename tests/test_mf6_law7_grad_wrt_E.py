"""MF6 LAW=7 (tabulated E'/mu double-differential) grad wrt query
Ein through the traced-x fast path (Phase 3 last item of roadmap
#198).

Before this PR the ``get_dist2d_from_subsec_law7`` reconstruction
looped over Ein query points via ``float(e_in[i])`` and called
``find_interval`` on concrete Ein, materialising any JAX tracer.
The traced-x variant loops over panels (concrete), evaluates each
panel's ``(mu, Ep)`` inner amplitude via the numpy-only unit-base
``interp_tab2`` (unaffected by the outer tracer), and broadcasts
the outer 2-point Ein interp with ``xp.where`` panel masking.

Uses Be-9 MT=16 (n,2n) subsec 1 (LAW=7 in the committed corpus).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_subsecs as mf6subsec,
)
from endf_userpy.primitives import array_ns


DATA_DIR = Path(__file__).parent / 'data'


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def be9_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-004_Be_009.endf'),
    )


def _fd5(f, x, h):
    return (-float(f(x + 2 * h)) + 8 * float(f(x + h))
            - 8 * float(f(x - h)) + float(f(x - 2 * h))) / (12 * h)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law7_numpy_jax_parity(be9_endf_dict):
    """xp=jax and xp=numpy give the same DDX on the LAW=7 corpus
    (Be-9 MT=16 sub 1). The traced-x path evaluates the same
    physics via a different graph."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([3.0e6, 5.0e6, 8.0e6, 1.2e7, 1.8e7])
    eout = np.linspace(1e5, 5e6, 15)
    mu = np.linspace(-0.9, 0.9, 11)

    f_np = np.asarray(mf6subsec.get_dist2d_from_subsec_law7(
        be9_endf_dict, 16, 1, ein, eout, mu, to_lab=True, xp=xp_np,
    ))
    f_jx = np.asarray(mf6subsec.get_dist2d_from_subsec_law7(
        be9_endf_dict, 16, 1,
        jnp.asarray(ein), jnp.asarray(eout), jnp.asarray(mu),
        to_lab=True, xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-9, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('E_val', [3.0e6, 5.0e6, 8.0e6, 1.2e7, 1.8e7])
def test_law7_grad_wrt_E_matches_fd(be9_endf_dict, E_val):
    """jax.grad(sum(f))(E) matches central FD across the LAW=7
    Ein panels."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    eout = jnp.linspace(1e5, 4e6, 12)
    mu = jnp.linspace(-0.9, 0.9, 11)

    def loss(E_scalar):
        return jnp.sum(mf6subsec.get_dist2d_from_subsec_law7(
            be9_endf_dict, 16, 1,
            jnp.array([E_scalar]), eout, mu,
            to_lab=True, xp=xp_jx,
        ))

    grad = float(jax.grad(loss)(jnp.array(E_val)))
    fd = _fd5(lambda v: loss(jnp.array(v)), E_val, E_val * 1e-4)
    assert np.isfinite(grad), f'grad not finite at E={E_val}'
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law7_grad_wrt_E_vector_matches_fd_elementwise(be9_endf_dict):
    """Vector grad wrt (n_ein,) across multiple LAW=7 Ein panels
    matches per-E FD element-wise."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    Es_np = np.array([3.0e6, 5.0e6, 8.0e6, 1.2e7, 1.8e7])
    Es = jnp.asarray(Es_np)
    eout = jnp.linspace(1e5, 4e6, 12)
    mu = jnp.linspace(-0.9, 0.9, 11)

    def loss(E):
        return jnp.sum(mf6subsec.get_dist2d_from_subsec_law7(
            be9_endf_dict, 16, 1, E, eout, mu,
            to_lab=True, xp=xp_jx,
        ))

    grad_vec = np.asarray(jax.grad(loss)(Es))
    assert grad_vec.shape == (len(Es_np),)

    for i, E_val in enumerate(Es_np):
        def loss_i(E, i=i):
            return loss(Es.at[i].set(E[0]))
        fd = _fd5(loss_i, jnp.array([E_val]), E_val * 1e-4)
        assert np.isfinite(grad_vec[i])
        np.testing.assert_allclose(
            grad_vec[i], fd, rtol=5e-3, atol=1e-6,
            err_msg=f'i={i} E={E_val}',
        )
