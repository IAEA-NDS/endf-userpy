"""endf_parserpy renamed the per-L group table under an MF2/MT151
range from ``spingroup`` (<= 0.13) to ``l_group`` (>= 0.17). The
preproc modules read the table via ``_get_l_group`` which tries
both spellings; this test locks in the two-name compatibility so
a future parser change (or a downgrade) doesn't silently break
resonance reconstruction on a real file.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.mfsec_interpretation import (
    mf2_interpretation_mlbw_preproc as _mlbw_pre,
)


def _minimal_range_body(group_key):
    """Return an LRU=1 LRF=2 (MLBW) range dict skeleton with the
    per-L group table stored under ``group_key`` and one trivial
    (potential-only) group at L=0. Not enough to reconstruct a
    cross section against; sufficient to exercise the key-lookup
    path in :func:`_get_l_group`.
    """
    l_group = {
        1: {
            'L': 0, 'AWRI': 1.0, 'APL': 0.0, 'NRS': 0,
            'ER': [], 'AJ': [], 'GT': [], 'GN': [], 'GG': [], 'GF': [],
        },
    }
    return {group_key: l_group, 'EL': 1e-5, 'EH': 1000.0}


@pytest.mark.parametrize('group_key', ['l_group', 'spingroup'])
def test_get_l_group_accepts_both_parser_spellings(group_key):
    d_range = _minimal_range_body(group_key)
    grp = _mlbw_pre._get_l_group(d_range)
    assert set(grp.keys()) == {1}
    assert grp[1]['L'] == 0


def test_get_l_group_missing_both_raises_clearly():
    """A range record from a parser that used neither spelling
    would be a genuinely unrecognised layout; we want a clear
    error rather than a stray ``KeyError`` from a downstream
    subscript."""
    d_range = {'EL': 1e-5, 'EH': 1000.0}   # no group table at all
    with pytest.raises(KeyError, match=r"l_group.*spingroup"):
        _mlbw_pre._get_l_group(d_range)


def test_get_l_group_prefers_l_group_when_both_present():
    """If a parser (hypothetically) emitted both keys, prefer the
    new one (``l_group``) so we don't accidentally consume stale
    ``spingroup`` residues."""
    d_range = {
        'EL': 1e-5, 'EH': 1000.0,
        'l_group':  {1: {'L': 0, 'AWRI': 1.0, 'APL': 0.0, 'NRS': 0,
                         'ER': [1.0], 'AJ': [], 'GT': [], 'GN': [],
                         'GG': [], 'GF': []}},
        'spingroup': {1: {'L': 9, 'AWRI': 99.0, 'APL': 0.0, 'NRS': 0,
                          'ER': [], 'AJ': [], 'GT': [], 'GN': [],
                          'GG': [], 'GF': []}},
    }
    grp = _mlbw_pre._get_l_group(d_range)
    assert grp[1]['L'] == 0, 'should have preferred l_group (L=0), not spingroup (L=9)'
    assert np.array_equal(grp[1]['ER'], [1.0])
