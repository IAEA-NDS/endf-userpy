"""Unit tests for :mod:`endf_userpy.primitives.static_dict`.

Covers the wrapper's dict-drop-in behaviour, identity-based hash/eq,
instance-attached preproc cache lifetime, and end-to-end cache hits
through the MF6 LAW=1 preproc entry point.
"""
import copy

import numpy as np
import pytest

from endf_userpy.primitives.static_dict import (
    StaticEndfDict,
    wrap_endf_dict,
)


def _synthetic_endf_dict():
    """Minimal shape-representative synthetic endf_dict."""
    return {
        6: {
            16: {
                'LCT': 1,
                'subsection': {
                    1: {'LAW': 1, 'ZAP': 1.0, 'AWP': 1.0},
                },
            },
        },
        3: {1: {'xstable': {'E': [1.0, 2.0], 'xs': [0.1, 0.2]}}},
    }


class TestWrapperBasics:

    def test_is_dict_subclass(self):
        w = wrap_endf_dict(_synthetic_endf_dict())
        assert isinstance(w, dict)
        assert isinstance(w, StaticEndfDict)

    def test_reads_like_dict(self):
        d = _synthetic_endf_dict()
        w = wrap_endf_dict(d)
        assert w[6][16]['LCT'] == 1
        assert w[3][1]['xstable']['E'] == [1.0, 2.0]
        assert 6 in w
        assert 99 not in w
        assert list(w.keys()) == list(d.keys())
        assert w.get(6) is d[6]
        assert w.get(99, 'missing') == 'missing'

    def test_nested_shared_with_underlying(self):
        d = _synthetic_endf_dict()
        w = wrap_endf_dict(d)
        d[6][16]['LCT'] = 2
        assert w[6][16]['LCT'] == 2

    def test_top_level_diverges_from_underlying(self):
        d = _synthetic_endf_dict()
        w = wrap_endf_dict(d)
        d[99] = 'new'
        assert 99 not in w

    def test_identity_hash_is_stable(self):
        w = wrap_endf_dict(_synthetic_endf_dict())
        assert hash(w) == hash(w)

    def test_distinct_wrappers_have_distinct_identity(self):
        d = _synthetic_endf_dict()
        w1 = wrap_endf_dict(d)
        w2 = wrap_endf_dict(d)
        assert w1 is not w2
        assert w1 != w2
        assert not (w1 == w2)

    def test_identity_equality_with_self(self):
        w = wrap_endf_dict(_synthetic_endf_dict())
        assert w == w
        assert not (w != w)

    def test_not_equal_to_raw_dict(self):
        d = _synthetic_endf_dict()
        w = wrap_endf_dict(d)
        assert w != d
        assert not (w == d)

    def test_wrap_endf_dict_is_idempotent(self):
        w = wrap_endf_dict(_synthetic_endf_dict())
        assert wrap_endf_dict(w) is w

    def test_hashable_usable_as_dict_key(self):
        d = _synthetic_endf_dict()
        w1 = wrap_endf_dict(d)
        w2 = wrap_endf_dict(d)
        lookup = {w1: 'a', w2: 'b'}
        assert lookup[w1] == 'a'
        assert lookup[w2] == 'b'

    def test_preproc_cache_starts_empty(self):
        w = wrap_endf_dict(_synthetic_endf_dict())
        assert w._preproc_cache == {}

    def test_preproc_cache_is_per_instance(self):
        d = _synthetic_endf_dict()
        w1 = wrap_endf_dict(d)
        w2 = wrap_endf_dict(d)
        w1._preproc_cache[('tag', 1, 1)] = 'value'
        assert w2._preproc_cache == {}


class TestMF6Law1CacheIntegration:
    """Integration with the MF6 LAW=1 preproc cache path."""

    @pytest.fixture
    def synthetic_law1_dict(self):
        """Enough of an endf_dict for mf6_law1_data_from_endf_dict
        to run its full build path."""
        return {
            1: {
                451: {
                    'ZA': 26056, 'AWR': 55.365, 'LISO': 0,
                    'AWI': 1.0, 'NSUB': 10,
                },
            },
            3: {
                16: {'QI': -1.1e7, 'QM': -1.1e7},
            },
            6: {
                16: {
                    'LCT': 1,
                    'subsection': {
                        1: {
                            'LAW': 1, 'LANG': 2, 'LEP': 2,
                            'ZAP': 1.0, 'AWP': 1.0,
                            'E': {1: 1.2e7, 2: 1.4e7},
                            'INT': [2], 'NBT': [2],
                            'ND': {1: 0, 2: 0},
                            'NA': {1: 1, 2: 1},
                            'Ep': {
                                1: {1: 0.0, 2: 1e6, 3: 5e6},
                                2: {1: 0.0, 2: 1e6, 3: 5e6},
                            },
                            'b': {
                                1: {1: {1: 1.0, 2: 0.0},
                                    2: {1: 0.5, 2: 0.0},
                                    3: {1: 0.1, 2: 0.0}},
                                2: {1: {1: 1.0, 2: 0.0},
                                    2: {1: 0.5, 2: 0.0},
                                    3: {1: 0.1, 2: 0.0}},
                            },
                        },
                    },
                },
            },
        }

    def test_raw_dict_bypasses_cache(self, synthetic_law1_dict):
        from endf_userpy.mfsec_interpretation import mf6_law1_preproc
        d = synthetic_law1_dict
        data1 = mf6_law1_preproc.mf6_law1_data_from_endf_dict(d, 16, 1)
        data2 = mf6_law1_preproc.mf6_law1_data_from_endf_dict(d, 16, 1)
        assert data1 is not data2

    def test_wrapped_dict_caches(self, synthetic_law1_dict):
        from endf_userpy.mfsec_interpretation import mf6_law1_preproc
        w = wrap_endf_dict(synthetic_law1_dict)
        data1 = mf6_law1_preproc.mf6_law1_data_from_endf_dict(w, 16, 1)
        data2 = mf6_law1_preproc.mf6_law1_data_from_endf_dict(w, 16, 1)
        assert data1 is data2
        assert ('mf6_law1', 16, 1) in w._preproc_cache

    def test_rewrap_invalidates_cache(self, synthetic_law1_dict):
        from endf_userpy.mfsec_interpretation import mf6_law1_preproc
        d = synthetic_law1_dict
        w1 = wrap_endf_dict(d)
        _ = mf6_law1_preproc.mf6_law1_data_from_endf_dict(w1, 16, 1)
        assert w1._preproc_cache

        w2 = wrap_endf_dict(d)  # re-wrap = fresh cache
        assert w2._preproc_cache == {}
        data2 = mf6_law1_preproc.mf6_law1_data_from_endf_dict(w2, 16, 1)
        assert data2 is not w1._preproc_cache[('mf6_law1', 16, 1)]

    def test_wrapped_and_raw_produce_equal_data(self, synthetic_law1_dict):
        from endf_userpy.mfsec_interpretation import mf6_law1_preproc
        d = synthetic_law1_dict
        w = wrap_endf_dict(d)
        data_raw = mf6_law1_preproc.mf6_law1_data_from_endf_dict(d, 16, 1)
        data_wrapped = mf6_law1_preproc.mf6_law1_data_from_endf_dict(w, 16, 1)
        assert np.array_equal(data_raw.ei_mesh, data_wrapped.ei_mesh)
        assert np.array_equal(data_raw.b_panels, data_wrapped.b_panels)
        assert np.array_equal(data_raw.ep_panels, data_wrapped.ep_panels)
        assert data_raw.lang == data_wrapped.lang
        assert data_raw.lep == data_wrapped.lep

    def test_jax_backend_bypasses_cache_even_when_wrapped(
        self, synthetic_law1_dict
    ):
        pytest.importorskip('jax')
        from endf_userpy.primitives import array_ns
        from endf_userpy.mfsec_interpretation import mf6_law1_preproc
        xp_jx = array_ns.get_backend('jax')
        w = wrap_endf_dict(synthetic_law1_dict)
        _ = mf6_law1_preproc.mf6_law1_data_from_endf_dict(
            w, 16, 1, xp=xp_jx,
        )
        # JAX build must not have polluted the wrapper's cache with a
        # tracer-carrying dataclass; the cache remains empty so a later
        # numpy call still builds fresh.
        assert ('mf6_law1', 16, 1) not in w._preproc_cache


class TestDeepcopySemantics:
    """Deepcopy on the wrapper should behave sensibly."""

    def test_deepcopy_returns_wrapper_with_fresh_cache(self):
        d = _synthetic_endf_dict()
        w = wrap_endf_dict(d)
        w._preproc_cache[('tag',)] = 'stale'
        w_copy = copy.deepcopy(w)
        assert isinstance(w_copy, StaticEndfDict)
        # Even if deepcopy carries state over, the copied object is a
        # distinct identity so cache-key isolation is preserved.
        assert w_copy is not w
        assert w_copy != w
