"""Regression tests pinning jit-safety of the MF2 + MF3 composition
path under ``RunOptions(backend='jax', include_resonance=True)``.

Issue #308 opened a concern that composition routed through
``primitives.tab1.interp`` and would trigger a
``TracerArrayConversionError`` on tracer input. The Reich-Moore
(LRF=3) composition path turned out to be jit-safe end-to-end
(verified by the jit tests in
:mod:`test_mf6_law1_gamma_isotropic_collapse` after the #308
cleanup PR reverted the ``include_resonance=False`` opt-out).
Issue #314 then separately fixed the MLBW (LRF=2) preprocessor's
unconditional-dummy-channel append so MLBW is jit-safe too.

These tests pin both composition surfaces (RM and MLBW) under
``@jax.jit`` and under eager JAX, so a regression in either
preprocessor's jit-safety surfaces here before showing up inside
a user workload.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.dirname(__file__))
from _corpus import resolve_n14, resolve_nb93, resolve_u235   # noqa: E402

from endf_userpy.primitives import array_ns   # noqa: E402
from endf_userpy.quantities import (   # noqa: E402
    get_particle_production_xs,
    get_reaction_xs,
)
from endf_userpy.run_options import RunOptions   # noqa: E402


def _jax_available() -> bool:
    try:
        import jax   # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture
def u235_dict():
    """TENDL-2021 U-235, Reich-Moore (LRF=3) in the RRR."""
    path = resolve_u235()
    if path is None:
        pytest.skip(
            'U-235 ENDF file not available (set U235_ENDF, run '
            'tests/data_law1_adhoc/fetch.sh, or place the file at '
            'tests/data_law1_adhoc/tendl21_n_U-235.endf)'
        )
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp().parsefile(path)


@pytest.fixture
def nb93_dict():
    """ENDF/B-VIII.1 Nb-93, MLBW (LRF=2) in the RRR."""
    path = resolve_nb93()
    if path is None:
        pytest.skip(
            'Nb-93 ENDF file not available (set NB93_ENDF, run '
            'tests/data_law1_adhoc/fetch.sh, or place the file at '
            'tests/data_law1_adhoc/endfb81_n_Nb-93.endf)'
        )
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp().parsefile(path)


@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_get_reaction_xs_under_jit_with_rm_composition(u235_dict):
    """Issue #308 regression: ``get_reaction_xs`` under ``@jax.jit``
    with ``backend='jax'`` and ``include_resonance=True`` (default)
    must trace cleanly through the Reich-Moore composition and
    ``primitives.tab1.interp``. Numerics are cross-checked against
    the numpy path to catch silent divergence."""
    import jax
    import jax.numpy as jnp

    ein_np = np.geomspace(1.0, 500.0, 16)   # inside U-235 RRR
    xp_jax = array_ns.get_backend('jax')
    xp_np = array_ns.get_backend('numpy')
    opts_jax = RunOptions(backend=xp_jax)   # include_resonance=True default
    opts_np = RunOptions(backend=xp_np)

    @jax.jit
    def jit_go(ein_arg):
        return get_reaction_xs(
            u235_dict, '(n,g)', ein_arg, options=opts_jax,
        )

    out_jit = np.asarray(jit_go(jnp.asarray(ein_np)))
    out_np = np.asarray(
        get_reaction_xs(u235_dict, '(n,g)', ein_np, options=opts_np)
    )
    rel = np.abs(out_jit - out_np) / np.maximum(np.abs(out_np), 1e-30)
    assert np.all(rel < 1e-6), (
        f'JIT composition under backend=jax disagrees with the numpy '
        f'path by max rel diff {rel.max():.3e}; expected machine-'
        f'noise agreement since both paths go through the same '
        f'resonance_composition code'
    )


@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_get_reaction_xs_eager_jax_with_rm_composition(u235_dict):
    """Eager (non-jit) JAX call through Reich-Moore composition.
    Catches regressions where composition only works under jit's
    constant-folding but breaks on live tracers."""
    import jax.numpy as jnp

    ein = jnp.array([1.0, 10.0, 100.0, 500.0])
    xp_jax = array_ns.get_backend('jax')
    out = get_reaction_xs(
        u235_dict, '(n,g)', ein,
        options=RunOptions(backend=xp_jax),
    )
    out_np = np.asarray(out)
    assert out_np.shape == (4,)
    assert np.all(np.isfinite(out_np))
    assert np.all(out_np > 0.0)


@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_get_reaction_xs_under_jit_with_mlbw_composition(nb93_dict):
    """Issue #314 regression: MLBW composition under ``@jax.jit``
    with ``backend='jax'`` and ``include_resonance=True`` must not
    raise ``TracerBoolConversionError`` from the dummy-channel
    branch in the MLBW preprocessor, and must match the numpy path
    numerically. Fix is in
    :mod:`mfsec_interpretation.mf2_interpretation_mlbw_preproc`:
    the dummy potential-only channel is now appended
    unconditionally with its weight clamped to a non-negative
    scalar, so the preproc output shape no longer depends on a
    (possibly-traced) ``gj_dif``."""
    import jax
    import jax.numpy as jnp

    ein_np = np.geomspace(1.0, 500.0, 16)
    xp_jax = array_ns.get_backend('jax')
    xp_np = array_ns.get_backend('numpy')
    opts_jax = RunOptions(backend=xp_jax)
    opts_np = RunOptions(backend=xp_np)

    @jax.jit
    def jit_go(ein_arg):
        return get_reaction_xs(
            nb93_dict, '(n,g)', ein_arg, options=opts_jax,
        )

    out_jit = np.asarray(jit_go(jnp.asarray(ein_np)))
    out_np = np.asarray(
        get_reaction_xs(nb93_dict, '(n,g)', ein_np, options=opts_np)
    )
    rel = np.abs(out_jit - out_np) / np.maximum(np.abs(out_np), 1e-30)
    assert np.all(rel < 1e-6), (
        f'JIT MLBW composition disagrees with numpy by max rel diff '
        f'{rel.max():.3e}; dummy-channel append may be contributing '
        f'non-zero where it should contribute zero'
    )


@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_get_reaction_xs_eager_jax_with_mlbw_composition(nb93_dict):
    """Eager JAX through MLBW composition. Catches regressions in
    the dummy-channel append (issue #314) that would only surface
    on live tracers."""
    import jax.numpy as jnp

    ein = jnp.array([1.0, 10.0, 100.0, 1000.0])
    xp_jax = array_ns.get_backend('jax')
    out = get_reaction_xs(
        nb93_dict, '(n,g)', ein,
        options=RunOptions(backend=xp_jax),
    )
    out_np = np.asarray(out)
    assert out_np.shape == (4,)
    assert np.all(np.isfinite(out_np))
    assert np.all(out_np > 0.0)


def _load_corpus(path):
    if path is None:
        pytest.skip('corpus file not present (see tests/data_law1_adhoc/fetch.sh)')
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp().parsefile(path)


@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
@pytest.mark.parametrize('resolve,particle', [
    # MF1 nubar (MT452 / MT456) multiplies the fission cross section:
    # the readers used to np.asarray the (traced) query energies.
    (resolve_u235, 'n'),
    # MF13 NK > 1: the total-vs-partials consistency check compared
    # traced values, and the MF3 denominator of the photon yield was
    # always evaluated on numpy.
    (resolve_n14, 'g'),
])
def test_get_particle_production_xs_under_jit_over_energies(resolve, particle):
    """``get_particle_production_xs`` under ``jax.jit`` with the query
    energies as the traced argument agrees with the numpy backend."""
    import jax
    import jax.numpy as jnp

    d = _load_corpus(resolve())
    ein = np.geomspace(1e-3, 2e7, 64)
    opts_jax = RunOptions(backend='jax')

    @jax.jit
    def jit_go(ein_arg):
        return get_particle_production_xs(
            d, '(n,total)', particle, ein_arg, options=opts_jax,
        )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.asarray(jit_go(jnp.asarray(ein)))
        ref = np.asarray(get_particle_production_xs(
            d, '(n,total)', particle, ein,
            options=RunOptions(backend='numpy'),
        ))
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
    fin = np.isfinite(ref)
    assert np.any(ref[fin] > 0.0)
    np.testing.assert_allclose(out[fin], ref[fin], rtol=1e-9, atol=0.0)


def _with_rrr_ap(d, value):
    """Copy of ``d`` with the first resolved range's AP replaced (only
    the dicts on the path are copied)."""
    d2 = dict(d)
    d2[2] = dict(d[2])
    d2[2][151] = dict(d[2][151])
    iso = dict(d2[2][151]['isotope'])
    d2[2][151]['isotope'] = iso
    iso[1] = dict(iso[1])
    rngs = dict(iso[1]['range'])
    iso[1]['range'] = rngs
    rngs[1] = dict(rngs[1])
    rngs[1]['AP'] = value
    return d2


@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_jit_with_fixed_mesh_reconstructs_only_in_range_points(
        u235_dict, monkeypatch):
    """``jax.jit`` over a resonance parameter with the energy mesh
    closed over: the composition still finds the in-range points on the
    host (``_stage_energies`` / ``_QueryState.host_energies``), so
    Reich-Moore runs on those points only -- not on the full mesh as
    for traced energies -- and the result matches numpy."""
    import jax
    import jax.numpy as jnp
    from endf_userpy.mfsec_interpretation import (
        mf2_interpretation_reichmoore as rm,
    )

    rng = u235_dict[2][151]['isotope'][1]['range'][1]
    el, eh = float(rng['EL']), float(rng['EH'])
    ein = np.geomspace(1e-3, 2e7, 8192)
    n_in = int(np.count_nonzero((ein >= el) & (ein < eh)))
    assert 0 < n_in < ein.size - 2048      # slicing applies

    sizes = []
    orig = rm.reconstruct

    def spy(data, energies, xp, *args, **kwargs):
        sizes.append(int(np.shape(energies)[0]))
        return orig(data, energies, xp, *args, **kwargs)

    monkeypatch.setattr(rm, 'reconstruct', spy)
    opts = RunOptions(backend='jax')
    ap = float(rng['AP'])

    @jax.jit
    def go(ap_arg):
        return get_reaction_xs(
            _with_rrr_ap(u235_dict, ap_arg), '(n,total)', ein, options=opts,
        )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.asarray(go(jnp.asarray(ap)))
        ref = np.asarray(get_reaction_xs(
            u235_dict, '(n,total)', ein, options=RunOptions(backend='numpy'),
        ))
    assert sizes and all(s == n_in for s in sizes)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
    fin = np.isfinite(ref)
    np.testing.assert_allclose(out[fin], ref[fin], rtol=1e-9, atol=0.0)


@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_jit_with_fixed_mesh_stages_energies_behind_a_barrier(u235_dict):
    """The closed-over mesh enters the traced program once, through
    ``optimization_barrier``, so XLA does not constant-fold everything
    computed from it at compile time."""
    import jax
    import jax.numpy as jnp

    ein = np.geomspace(1e-3, 2e7, 64)
    opts = RunOptions(backend='jax')
    rng = u235_dict[2][151]['isotope'][1]['range'][1]

    def go(ap_arg):
        return get_reaction_xs(
            _with_rrr_ap(u235_dict, ap_arg), '(n,total)', ein, options=opts,
        )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        jaxpr = str(jax.make_jaxpr(go)(jnp.asarray(float(rng['AP']))))
    assert 'optimization_barrier' in jaxpr
