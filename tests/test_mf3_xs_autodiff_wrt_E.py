"""Phase-1 autodiff-coverage pins: ``jax.grad`` wrt incident energy
E through the top-level MF3 background cross-section API
(``get_reaction_xs``).

The interp primitives are x-tracer aware (PR #192) and the MF3
composition is xp-agnostic (PR #193), so grad wrt E should already
flow end-to-end for any reaction whose XS is a straight MF3 lookup
(no resonance composition, no LAW=1 continuum reconstruction). This
file pins that claim.

Tests:

- ``get_reaction_xs`` under xp=jax matches the numpy result.
- ``jax.grad(sum(get_reaction_xs)(E))`` matches central FD element-
  wise at Es across smooth intervals of the tabulated grid.
- Grad handles a query grid that spans multiple interp panels (the
  numerical fix from #192 is that interp with x-tracer selects the
  correct law per panel via nested xp.where).

Uses ``tests/data/n-004_Be_009.endf`` (committed, small). MT2
(elastic) has a smooth 222-point grid; MT102 (radiative capture)
has a 1256-point 1/v-like grid at low energy plus higher-order
structure above. Both are pure MF3 lookups with no resonance
region on this file.

Known gap (Phase 1 finding, tracked as issue #196): the MT5
fallback path in ``get_reaction_xs`` (``quantities.py`` around
line 405) is documented as "numpy-internal for now (tier-2
remainder)" and calls ``np.asarray(cur_xs, dtype=float)`` on the
compute_xs result. Under xp=jax this crashes with
``TracerArrayConversionError`` whenever
``reac.is_unique_path_to_residual(proj, mt)`` returns True.
Reactions where the fallback fires (e.g. ``(n,g)`` MT=102) must
be called with ``mt5_contrib=False`` to skip the fallback and
get an xp-native return. Elastic ``(n,n)`` is not classified as
a unique-path-to-residual so the fallback does not fire and
autodiff works out of the box. See
``test_grad_wrt_E_ngamma_mt5_default_raises`` below for the
regression pin.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.quantities import get_reaction_xs


DATA_DIR = Path(__file__).parent / 'data'


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def be9_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-004_Be_009.endf'),
    )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_reaction_xs_numpy_jax_parity_elastic(be9_endf_dict):
    """xp=jax path produces the same values as xp=numpy at
    representative Es across the tabulated grid. Elastic (n,n) is
    not classified as unique-path-to-residual, so the MT5 fallback
    path does not fire."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    energies = np.array(
        [1e-3, 1e0, 1e3, 1e5, 1e6, 5e6, 1.5e7], dtype=np.float64,
    )
    xs_np = np.asarray(get_reaction_xs(
        be9_endf_dict, '(n,n)', energies, xp=xp_np,
    ))
    xs_jx = np.asarray(get_reaction_xs(
        be9_endf_dict, '(n,n)', jnp.asarray(energies), xp=xp_jx,
    ))
    np.testing.assert_allclose(xs_np, xs_jx, rtol=1e-10, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_reaction_xs_numpy_jax_parity_ngamma_mt5_off(be9_endf_dict):
    """Same parity check for (n,g) with ``mt5_contrib=False`` to
    dodge the tier-2-remainder MT5 fallback path."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    energies = np.array(
        [1e-3, 1e0, 1e3, 1e5, 1e6, 5e6, 1.5e7], dtype=np.float64,
    )
    xs_np = np.asarray(get_reaction_xs(
        be9_endf_dict, '(n,g)', energies, mt5_contrib=False, xp=xp_np,
    ))
    xs_jx = np.asarray(get_reaction_xs(
        be9_endf_dict, '(n,g)', jnp.asarray(energies),
        mt5_contrib=False, xp=xp_jx,
    ))
    np.testing.assert_allclose(xs_np, xs_jx, rtol=1e-10, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize(
    'reaction,mt5_contrib,E_val,fd_h_rel',
    [
        ('(n,n)', True, 1e6, 1e-4),
        ('(n,n)', True, 5e6, 1e-4),
        ('(n,n)', True, 1.2e7, 1e-4),
        ('(n,g)', False, 1e-2, 1e-3),
        ('(n,g)', False, 1e2, 1e-3),
        ('(n,g)', False, 1e6, 1e-4),
    ],
)
def test_grad_wrt_E_matches_fd(
    be9_endf_dict, reaction, mt5_contrib, E_val, fd_h_rel,
):
    """jax.grad(sum(xs))(E) matches central FD at smooth-interior Es.

    ``mt5_contrib=False`` is required on reactions the code
    classifies as unique-path-to-residual (e.g. (n,g) MT=102),
    since the MT5 fallback path is not xp-agnostic yet. Elastic
    (n,n) is unaffected."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')

    def loss(E):
        return get_reaction_xs(
            be9_endf_dict, reaction, E,
            mt5_contrib=mt5_contrib, xp=xp_jx,
        ).sum()

    E0 = jnp.array([E_val])
    grad = float(jax.grad(lambda e: loss(e))(E0)[0])

    h = E_val * fd_h_rel
    fd = (float(loss(jnp.array([E_val + h])))
          - float(loss(jnp.array([E_val - h])))) / (2 * h)
    assert np.isfinite(grad), f'grad not finite at E={E_val}'
    if abs(fd) < 1e-30:
        return
    np.testing.assert_allclose(
        grad, fd, rtol=5e-3, atol=1e-30,
        err_msg=f'{reaction} @ E={E_val}: ad={grad:.4e} fd={fd:.4e}',
    )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_grad_wrt_E_ngamma_mt5_default_raises(be9_endf_dict):
    """Regression pin for the known MT5-fallback gap: (n,g) with
    the default ``mt5_contrib=True`` raises
    ``TracerArrayConversionError`` under xp=jax because the
    fallback path materialises via ``np.asarray``. Delete this
    test once the fallback is ported."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')

    def loss(E):
        return get_reaction_xs(
            be9_endf_dict, '(n,g)', E, xp=xp_jx,
        ).sum()

    try:
        import jax.errors
        expected_exc = jax.errors.TracerArrayConversionError
    except AttributeError:
        expected_exc = Exception
    with pytest.raises(expected_exc):
        jax.grad(lambda e: loss(e))(jnp.array([1e2]))


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_grad_wrt_E_vector_matches_fd_elementwise(be9_endf_dict):
    """Vector grad wrt (n_E,) — Es spread over several interp panels
    — matches per-E FD element-wise. Pins that the x-tracer path
    per PR #192 selects the correct interp law per point without
    smearing across panel boundaries."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    Es_np = np.array([1e-2, 1e0, 1e3, 1e5, 1e6, 5e6, 1.2e7])
    Es = jnp.asarray(Es_np)

    def loss(E):
        # (n,n) is not classified as unique-path-to-residual, so
        # the MT5 fallback path does not fire; keep the default
        # mt5_contrib=True to also exercise the fallback-guard.
        return get_reaction_xs(
            be9_endf_dict, '(n,n)', E, xp=xp_jx,
        ).sum()

    grad_vec = np.asarray(jax.grad(loss)(Es))
    assert grad_vec.shape == (len(Es_np),)

    for i, E_val in enumerate(Es_np):
        h = E_val * 1e-4
        Es_p = Es.at[i].set(E_val + h)
        Es_m = Es.at[i].set(E_val - h)
        fd = (float(loss(Es_p)) - float(loss(Es_m))) / (2 * h)
        assert np.isfinite(grad_vec[i]), f'grad not finite at i={i}'
        if abs(fd) < 1e-30:
            continue
        np.testing.assert_allclose(
            grad_vec[i], fd, rtol=5e-3, atol=1e-30,
            err_msg=f'i={i} E={E_val}: ad={grad_vec[i]:.4e} fd={fd:.4e}',
        )
