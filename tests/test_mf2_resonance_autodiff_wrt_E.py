"""Phase-1 autodiff-coverage pins: ``jax.grad`` wrt incident energy
E through the top-level API with ``include_resonance=True`` on
both MF2 formalisms currently supported by the resonance
reconstruction (LRF=2 MLBW and LRF=3 Reich-Moore).

The reconstruction layers (``mf2_interpretation_mlbw`` and
``mf2_interpretation_reichmoore``) are array-namespace agnostic;
composition via ``quantities_mt_zap.resonance_composition`` uses
the ``xp`` adapter selected by ``resonance_backend='jax'`` +
``xp=jax``. Grad wrt E should flow end-to-end through:

    ``get_reaction_xs`` -> compute_xs -> resonance composition ->
    resonance-formalism kernel (MLBW / R-M) at tracer E,
    plus MF3 additive background at tracer E.

Tests:

- numpy vs jax parity at representative Es inside the RRR.
- ``jax.grad(sum(xs))(E)`` matches central FD at mid-panel Es in
  the RRR for MLBW (Nb-93) and R-M (Al-27).
- Vector grad wrt (n_E,) with Es across several resonances
  matches per-E FD element-wise, pinning that the E tracer is
  correctly routed to each resonance's contribution rather than
  smearing between them.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_reaction_xs
from endf_userpy.primitives import array_ns

from _corpus import resolve_nb93, resolve_al27


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def nb93_endf_dict():
    path = resolve_nb93()
    if path is None:
        pytest.skip('Nb-93 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip('Al-27 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def _get_xs_jax(d, reaction, E, resonance_backend='jax'):
    """Wrapper: get_reaction_xs under xp=jax with resonance
    composition on. ``resonance_backend='jax'`` keeps the whole
    reconstruction on jax so the E tracer threads through."""
    xp_jx = array_ns.get_backend('jax')
    return get_reaction_xs(
        d, reaction, E,
        include_resonance=True,
        resonance_backend=resonance_backend,
        xp=xp_jx,
    )


def _fd5(loss, E_val, h):
    """5-point central FD stencil, O(h^4) error. Necessary inside
    the RRR where the total XS has narrow-Lorentzian curvature and
    3-point central FD gets ~1% error at reasonable h."""
    import jax.numpy as jnp
    f_p2 = float(loss(jnp.array([E_val + 2 * h])))
    f_p1 = float(loss(jnp.array([E_val + h])))
    f_m1 = float(loss(jnp.array([E_val - h])))
    f_m2 = float(loss(jnp.array([E_val - 2 * h])))
    return (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * h)


# -------------------------------------------------------------------
# LRF=2 MLBW on Nb-93 (RRR: [1e-5, 7000] eV)
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mlbw_numpy_jax_parity_inside_rrr(nb93_endf_dict):
    """Composition matches between xp=numpy and xp=jax inside the
    LRF=2 MLBW RRR. Sanity-checks that the resonance_backend='jax'
    path gives the same numbers as the default numpy reconstruction
    at concrete Es (no grad yet)."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1e-3, 1.0, 35.0, 100.0, 500.0, 2000.0])

    xs_np = np.asarray(get_reaction_xs(
        nb93_endf_dict, '(n,total)', ein,
        include_resonance=True, xp=xp_np,
    ))
    xs_jx = np.asarray(get_reaction_xs(
        nb93_endf_dict, '(n,total)', jnp.asarray(ein),
        include_resonance=True, resonance_backend='jax', xp=xp_jx,
    ))
    np.testing.assert_allclose(xs_np, xs_jx, rtol=1e-9, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('E_val,fd_h_rel', [
    (35.0, 1e-4),
    (100.0, 1e-4),
    (500.0, 1e-4),
    (2000.0, 1e-4),
])
def test_mlbw_grad_wrt_E_matches_fd(nb93_endf_dict, E_val, fd_h_rel):
    """jax.grad(sum(xs))(E) matches central FD at Es inside the
    Nb-93 MLBW RRR. Loose tolerance because the total XS varies
    steeply between resonance peaks."""
    import jax
    import jax.numpy as jnp

    def loss(E):
        return _get_xs_jax(nb93_endf_dict, '(n,total)', E).sum()

    E0 = jnp.array([E_val])
    grad = float(jax.grad(lambda e: loss(e))(E0)[0])

    fd = _fd5(loss, E_val, E_val * fd_h_rel)
    assert np.isfinite(grad), f'grad not finite at E={E_val}'
    if abs(fd) < 1e-30:
        return
    np.testing.assert_allclose(
        grad, fd, rtol=5e-3, atol=1e-30,
        err_msg=f'MLBW @ E={E_val}: ad={grad:.4e} fd={fd:.4e}',
    )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mlbw_grad_wrt_E_vector_matches_fd_elementwise(nb93_endf_dict):
    """Vector grad wrt (n_E,) with Es across several MLBW
    resonances matches per-E FD element-wise. Pins that the E
    tracer routes to each resonance's contribution independently."""
    import jax
    import jax.numpy as jnp

    Es_np = np.array([10.0, 100.0, 500.0, 2000.0])
    Es = jnp.asarray(Es_np)

    def loss(E):
        return _get_xs_jax(nb93_endf_dict, '(n,total)', E).sum()

    grad_vec = np.asarray(jax.grad(loss)(Es))
    assert grad_vec.shape == (len(Es_np),)

    for i, E_val in enumerate(Es_np):
        h = E_val * 1e-4
        def loss_i(E, i=i):
            return loss(Es.at[i].set(E[0]))
        fd = _fd5(loss_i, E_val, h)
        assert np.isfinite(grad_vec[i])
        if abs(fd) < 1e-30:
            continue
        np.testing.assert_allclose(
            grad_vec[i], fd, rtol=5e-3, atol=1e-30,
            err_msg=f'i={i} E={E_val}: ad={grad_vec[i]:.4e} fd={fd:.4e}',
        )


# -------------------------------------------------------------------
# LRF=3 Reich-Moore on Al-27 (RRR: [1e-5, 8.45e5] eV)
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_reichmoore_numpy_jax_parity_inside_rrr(al27_endf_dict):
    """Composition matches between xp=numpy and xp=jax inside the
    LRF=3 R-M RRR on Al-27."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1e3, 5e3, 5e4, 1e5, 5e5])

    with warnings.catch_warnings():
        # Al-27's MF3 in the RRR is subtractive background but
        # include_resonance=True composes it. No warning expected,
        # but silence any leftover for robustness.
        warnings.simplefilter('ignore', UserWarning)
        xs_np = np.asarray(get_reaction_xs(
            al27_endf_dict, '(n,total)', ein,
            include_resonance=True, xp=xp_np,
        ))
        xs_jx = np.asarray(get_reaction_xs(
            al27_endf_dict, '(n,total)', jnp.asarray(ein),
            include_resonance=True, resonance_backend='jax', xp=xp_jx,
        ))
    np.testing.assert_allclose(xs_np, xs_jx, rtol=1e-9, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('E_val,fd_h_rel', [
    (5e3, 1e-4),
    (5e4, 1e-4),
    (1e5, 1e-4),
    (5e5, 1e-4),
])
def test_reichmoore_grad_wrt_E_matches_fd(al27_endf_dict, E_val, fd_h_rel):
    """jax.grad(sum(xs))(E) matches central FD at Es inside the
    Al-27 R-M RRR."""
    import jax
    import jax.numpy as jnp

    def loss(E):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return _get_xs_jax(al27_endf_dict, '(n,total)', E).sum()

    E0 = jnp.array([E_val])
    grad = float(jax.grad(lambda e: loss(e))(E0)[0])

    fd = _fd5(loss, E_val, E_val * fd_h_rel)
    assert np.isfinite(grad), f'grad not finite at E={E_val}'
    if abs(fd) < 1e-30:
        return
    np.testing.assert_allclose(
        grad, fd, rtol=5e-3, atol=1e-30,
        err_msg=f'R-M @ E={E_val}: ad={grad:.4e} fd={fd:.4e}',
    )
