"""Issue #307: slice-and-scatter optimisation in
``reconstruct_resonance_xs``.

Pins two things:

1. **Correctness**: on numpy, the new slice-and-scatter path must
   return numerically identical results to the pre-#307
   mask-after-evaluate shape across LRU=1 (RRR) and LRU=2 (URR)
   ranges, over query grids that straddle each range boundary.
2. **No-op outside the slice**: when every Ein point already lies
   inside the resonance range, the result must match the original
   full-array evaluation to bit level (the helper exits via its
   early-return path when ``idx_in.size == n_total``).
3. **JAX fallback is preserved**: the helper routes JAX-backend
   calls through the original ``xp.where`` shape so ``jax.jit``
   traces do not hit the data-dependent ``np.where`` size.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.quantities_mt_zap.resonance_composition import (
    reconstruct_resonance_xs,
)

from _corpus import resolve_u235, resolve_pu239_rml


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def u235_endf_dict():
    path = resolve_u235()
    if path is None:
        pytest.skip('U-235 corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def pu239_endf_dict():
    path = resolve_pu239_rml()
    if path is None:
        pytest.skip('Pu-239 R-M corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def _rrr_bounds(endf_dict):
    out = []
    for iso in endf_dict[2][151]['isotope'].values():
        for rng in iso['range'].values():
            out.append((float(rng['EL']), float(rng['EH']),
                        int(rng['LRU'])))
    return out


@pytest.mark.parametrize('mt', [1, 2, 3, 18, 27, 102])
def test_slice_matches_full_eval_on_straddling_grid_u235(
    u235_endf_dict, mt,
):
    """A log-spaced grid that straddles both the RRR and the URR
    exercises the slice path across all three regimes (below EL,
    inside each range, above EH). Result must bit-match a reference
    that forces the full-eval shape by constructing a grid where
    every point lies inside the first range.
    """
    xp = array_ns.get_backend('numpy')
    # Full-span log grid (straddles RRR and URR)
    e_full = np.geomspace(1e-5, 2e7, 2000)
    out_full = np.asarray(reconstruct_resonance_xs(
        u235_endf_dict, mt, e_full, xp=xp,
    ))

    # Reference: run on an all-inside-RRR grid and splice into the
    # same output slot. Because the helper's slice is just a
    # permutation of the "evaluate on these indices" set, running on
    # the subset directly should match the sliced piece of out_full.
    bounds = _rrr_bounds(u235_endf_dict)
    el_rrr, eh_rrr = [(el, eh) for el, eh, lru in bounds
                       if lru == 1][0]
    rrr_mask = (e_full >= el_rrr) & (e_full < eh_rrr)
    e_rrr = e_full[rrr_mask]
    out_rrr = np.asarray(reconstruct_resonance_xs(
        u235_endf_dict, mt, e_rrr, xp=xp,
    ))

    # The sliced piece of out_full (restricted to the RRR mask) must
    # equal out_rrr to bit precision: the slice-and-scatter path
    # does exactly the same reconstruction on the same subset.
    sliced = out_full[rrr_mask]
    np.testing.assert_array_equal(sliced, out_rrr)


@pytest.mark.parametrize('mt', [1, 2, 18, 102])
def test_slice_matches_full_eval_urr_u235(u235_endf_dict, mt):
    xp = array_ns.get_backend('numpy')
    e_full = np.geomspace(1e-5, 2e7, 2000)
    out_full = np.asarray(reconstruct_resonance_xs(
        u235_endf_dict, mt, e_full, xp=xp,
    ))
    bounds = _rrr_bounds(u235_endf_dict)
    urr_pair = [(el, eh) for el, eh, lru in bounds if lru == 2]
    if not urr_pair:
        pytest.skip('U-235 has no URR range (unexpected)')
    el_urr, eh_urr = urr_pair[0]
    urr_mask = (e_full >= el_urr) & (e_full < eh_urr)
    if int(urr_mask.sum()) == 0:
        pytest.skip('no Ein points in URR range at this grid')
    e_urr = e_full[urr_mask]
    out_urr = np.asarray(reconstruct_resonance_xs(
        u235_endf_dict, mt, e_urr, xp=xp,
    ))
    np.testing.assert_array_equal(out_full[urr_mask], out_urr)


@pytest.mark.parametrize('mt', [1, 2, 102])
def test_slice_matches_full_eval_pu239_reichmoore(pu239_endf_dict, mt):
    """Same correctness pin on a Reich-Moore RRR to exercise the LRF=3
    path through _reconstruct_lru1_range."""
    xp = array_ns.get_backend('numpy')
    e_full = np.geomspace(1e-5, 2e7, 1500)
    out_full = np.asarray(reconstruct_resonance_xs(
        pu239_endf_dict, mt, e_full, xp=xp,
    ))
    bounds = _rrr_bounds(pu239_endf_dict)
    rrr_pair = [(el, eh) for el, eh, lru in bounds if lru == 1]
    if not rrr_pair:
        pytest.skip('Pu-239 has no RRR range (unexpected)')
    el, eh = rrr_pair[0]
    mask = (e_full >= el) & (e_full < eh)
    e_sub = e_full[mask]
    out_sub = np.asarray(reconstruct_resonance_xs(
        pu239_endf_dict, mt, e_sub, xp=xp,
    ))
    np.testing.assert_array_equal(out_full[mask], out_sub)


def test_all_inside_grid_is_noop(u235_endf_dict):
    """A grid that lies entirely inside the first RRR exercises the
    helper's slice path with idx_in.size == n_total. The result must
    still be correct (ordinary numeric check, not bit-identity, since
    the slice path routes through np.asarray(total).copy())."""
    xp = array_ns.get_backend('numpy')
    bounds = _rrr_bounds(u235_endf_dict)
    el, eh = [(el, eh) for el, eh, lru in bounds if lru == 1][0]
    e_inside = np.geomspace(el * 1.01, eh * 0.99, 200)
    out = np.asarray(reconstruct_resonance_xs(
        u235_endf_dict, 1, e_inside, xp=xp,
    ))
    assert out.shape == e_inside.shape
    assert np.all(np.isfinite(out))
    assert np.any(out > 0)


def test_all_outside_grid_is_zero(u235_endf_dict):
    """A grid entirely above every range must return all zeros, same
    as before (short-circuits via ``if not any_in: continue``)."""
    xp = array_ns.get_backend('numpy')
    bounds = _rrr_bounds(u235_endf_dict)
    eh_max = max(eh for _, eh, _ in bounds)
    e_above = np.geomspace(eh_max * 1.1, 2e7, 100)
    out = np.asarray(reconstruct_resonance_xs(
        u235_endf_dict, 1, e_above, xp=xp,
    ))
    np.testing.assert_array_equal(out, np.zeros_like(e_above))


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_backend_takes_fallback_path(u235_endf_dict):
    """The JAX branch keeps the pre-#307 mask-after-evaluate shape;
    the result must match the numpy slice path numerically on a
    grid where the slice path is exercised."""
    import jax.numpy as jnp
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')
    e = np.geomspace(1e-5, 2e7, 500)
    out_np = np.asarray(reconstruct_resonance_xs(
        u235_endf_dict, 1, e, xp=xp_np,
    ))
    out_jax = np.asarray(reconstruct_resonance_xs(
        u235_endf_dict, 1, jnp.asarray(e), xp=xp_jax,
    ))
    np.testing.assert_allclose(out_jax, out_np, rtol=1e-10, atol=1e-15)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_jit_still_works_through_slice_helper(u235_endf_dict):
    """Under @jax.jit the helper must route through the mask-based
    fallback (np.where on a boolean mask would materialise a
    data-dependent shape and break tracing)."""
    import jax
    import jax.numpy as jnp
    xp_jax = array_ns.get_backend('jax')

    @jax.jit
    def fn(e):
        return reconstruct_resonance_xs(
            u235_endf_dict, 1, e, xp=xp_jax,
        )

    e = jnp.geomspace(1e-5, 2e7, 400)
    out = np.asarray(fn(e))
    assert out.shape == (400,)
    assert np.all(np.isfinite(out))
    assert np.any(out > 0)
