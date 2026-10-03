"""Regression tests pinning jit-safety of the MF2 + MF3 composition
path under ``RunOptions(backend='jax', include_resonance=True)``.

Issue #308 opened a concern that composition routed through
``primitives.tab1.interp`` and would trigger a
``TracerArrayConversionError`` on tracer input, forcing the jit
test suite in :mod:`test_mf6_law1_gamma_isotropic_collapse` to
pass ``include_resonance=False`` as a workaround. Follow-up
investigation found:

- The Reich-Moore (LRF=3) composition path IS jit-safe end-to-end;
  the ``test_mf6_law1_gamma_isotropic_collapse`` jit tests pass
  with the physics-first ``include_resonance=True`` default on
  U-233 and Al-27 (opt-out reverted in this PR).
- The MLBW (LRF=2) preprocessor contains an unconditional Python
  branch (``if gj_dif > 1e-30`` in ``_build_channel_table``) that
  triggers a ``TracerBoolConversionError`` under ``@jax.jit``.
  Tracked separately so #308 can close on the resolved RM
  surface.

These tests pin the RM composition surface. The eager-JAX test
below additionally exercises MLBW (Nb-93) in the eager-trace
regime, which does work.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.dirname(__file__))
from _corpus import resolve_nb93, resolve_u235   # noqa: E402

from endf_userpy.primitives import array_ns   # noqa: E402
from endf_userpy.quantities import get_reaction_xs   # noqa: E402
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
def test_get_reaction_xs_eager_jax_with_mlbw_composition(nb93_dict):
    """Eager JAX through MLBW composition works even though the
    jit path fails (see the module docstring for the MLBW preproc
    tracer-branch issue). Pins that the autodiff / eager jax path
    is not broken."""
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
