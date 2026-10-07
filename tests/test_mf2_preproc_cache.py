"""StaticEndfDict-based memoization of MF2 preproc results (#350).

Each of the four MF2 formalism preproc entry points
(``mlbw_data_from_endf_dict``, ``rm_data_from_endf_dict``,
``rml_data_from_endf_dict``, ``urr_data_from_endf_dict``) now
caches on the wrapper's ``_preproc_cache`` when the caller
passes a :class:`StaticEndfDict`. Pins:

- Cache key fires with the expected namespace and (iso, rng) tag.
- Cached result is bit-identical to a fresh build.
- Raw-dict path skips the cache (no cache attribute to anchor
  lifetime on).
- Repeated calls return the SAME object (identity-based reuse,
  not just equal values).
- Correctness end-to-end: wrapping the dict and running
  ``get_reaction_xs`` returns the same cross section as the
  unwrapped path.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives.static_dict import StaticEndfDict, wrap_endf_dict


def _try_load(path):
    try:
        return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    except FileNotFoundError:
        return None


HAS_RM = _try_load(
    'tests/data_law1_adhoc/jendl5_n_Cu-63.endf'
) is not None
HAS_MLBW = _try_load(
    'tests/data_law1_adhoc/endfb81_n_Nb-93.endf'
) is not None


@pytest.mark.skipif(not HAS_RM, reason='Cu-63 corpus not fetched')
def test_rm_preproc_cache_hits_on_wrapped_dict():
    """Reich-Moore preproc cache: wrap once, call twice, second
    call returns the SAME dataclass object as the first (identity-
    based reuse). Also pins the cache key format."""
    from endf_userpy.mfsec_interpretation.mf2_interpretation_reichmoore_preproc import (
        rm_data_from_endf_dict,
    )
    d = _try_load('tests/data_law1_adhoc/jendl5_n_Cu-63.endf')
    dw = wrap_endf_dict(d)
    data_a = rm_data_from_endf_dict(dw)
    data_b = rm_data_from_endf_dict(dw)
    assert data_a is data_b, 'cache should return the same object'
    assert ('mf2_rm_preproc', 1, 1) in dw._preproc_cache


@pytest.mark.skipif(not HAS_MLBW, reason='Nb-93 corpus not fetched')
def test_mlbw_preproc_cache_hits_on_wrapped_dict():
    """MLBW preproc cache: identity-reuse pin on Nb-93 ENDF/B-VIII.1
    (LRU=1 LRF=2)."""
    from endf_userpy.mfsec_interpretation.mf2_interpretation_mlbw_preproc import (
        mlbw_data_from_endf_dict,
    )
    d = _try_load('tests/data_law1_adhoc/endfb81_n_Nb-93.endf')
    dw = wrap_endf_dict(d)
    data_a = mlbw_data_from_endf_dict(dw)
    data_b = mlbw_data_from_endf_dict(dw)
    assert data_a is data_b
    assert ('mf2_mlbw_preproc', 1, 1) in dw._preproc_cache


@pytest.mark.skipif(not HAS_RM, reason='Cu-63 corpus not fetched')
def test_rm_preproc_raw_dict_skips_cache():
    """A raw dict has no ``_preproc_cache`` attribute. Preproc
    must still work (no AttributeError) and must NOT return the
    same object twice (because there is no cache)."""
    from endf_userpy.mfsec_interpretation.mf2_interpretation_reichmoore_preproc import (
        rm_data_from_endf_dict,
    )
    d = _try_load('tests/data_law1_adhoc/jendl5_n_Cu-63.endf')
    data_a = rm_data_from_endf_dict(d)
    data_b = rm_data_from_endf_dict(d)
    assert data_a is not data_b
    assert not hasattr(d, '_preproc_cache')


@pytest.mark.skipif(not HAS_RM, reason='Cu-63 corpus not fetched')
def test_rm_preproc_cache_value_identical_to_fresh_build():
    """A cached preproc has the same per-field numerical content
    as a fresh build on the unwrapped dict."""
    from endf_userpy.mfsec_interpretation.mf2_interpretation_reichmoore_preproc import (
        rm_data_from_endf_dict,
    )
    d = _try_load('tests/data_law1_adhoc/jendl5_n_Cu-63.endf')
    dw = wrap_endf_dict(d)
    fresh = rm_data_from_endf_dict(d)
    cached = rm_data_from_endf_dict(dw)
    # Compare every numeric field. Dataclass fields can be numpy
    # arrays, scalars, or TAB1 subdataclasses; handle each.
    import dataclasses
    def _compare(name, a, b):
        if a is None:
            assert b is None, name
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=name)
        elif dataclasses.is_dataclass(a):
            for ff in dataclasses.fields(a):
                _compare(
                    f'{name}.{ff.name}',
                    getattr(a, ff.name), getattr(b, ff.name),
                )
        else:
            assert a == b, f'{name}: {a} != {b}'
    for f in dataclasses.fields(fresh):
        _compare(f.name, getattr(fresh, f.name), getattr(cached, f.name))


@pytest.mark.skipif(not HAS_RM, reason='Cu-63 corpus not fetched')
def test_get_reaction_xs_bitexact_wrapped_vs_unwrapped():
    """End-to-end: wrapping the dict must not change the numeric
    output of ``get_reaction_xs``; only the per-call cost."""
    import warnings
    from endf_userpy.quantities import get_reaction_xs
    from endf_userpy.run_options import RunOptions
    d = _try_load('tests/data_law1_adhoc/jendl5_n_Cu-63.endf')
    dw = wrap_endf_dict(d)
    e_in = np.array([1.0, 1e3, 1e5, 1e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs_raw = get_reaction_xs(
            d, '(n,total)', e_in,
            options=RunOptions(backend='numba'),
        )
        xs_wrap = get_reaction_xs(
            dw, '(n,total)', e_in,
            options=RunOptions(backend='numba'),
        )
    np.testing.assert_allclose(xs_wrap, xs_raw, rtol=1e-13, atol=0)


def test_wrap_endf_dict_idempotent():
    """``wrap_endf_dict`` on a wrapper returns the same wrapper.
    Pins the StaticEndfDict contract referenced by the preproc
    cache (re-wrap = fresh cache, same wrap = same cache)."""
    d = {1: {451: {'ZA': 92235}}}
    dw1 = wrap_endf_dict(d)
    dw2 = wrap_endf_dict(dw1)
    assert dw1 is dw2
    dw3 = wrap_endf_dict(d)  # fresh wrap: separate cache
    assert dw3 is not dw1
    assert isinstance(dw3, StaticEndfDict)
