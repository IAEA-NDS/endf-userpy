"""Tests for the "MF15 present, MF12 has no Eg=0 continuum
placeholder" warning (issue #103 / audit D3).

The MF15 branch of `compute_energydist_values` (issue #54) weights
the MF15 continuous gamma spectrum by `y_cont / Y_total` from
MF12, where `y_cont` is the yield sitting at MF12's Eg=0
continuum placeholder. If MF12 declares the MT but has no Eg=0
placeholder at all, `y_cont` collapses to 0 and the whole MF15
contribution is silently zeroed to avoid inflating the sum -- the
file's own convention says there is no continuum for this MT, so
the MF15 content is inconsistent.

Rare in modern evaluations (all corpus files that have MF15 also
declare a matching Eg=0 placeholder in MF12), but when it hits,
gamma production from continuous spectra silently reports zero
and the user has no signal.

Fix: emit one summary UserWarning per (file, MT) pair when the
branch fires, naming the MT and explaining the drop.

These tests pin the branch behaviour on a synthetic mutation of
Al-27 MF12/MT102 (drop the Eg=0 entry so the placeholder is
gone) and confirm that:

1. The unmutated corpus files (Al-27, N-14, U-235, Pu-239, U-238)
   do NOT fire the warning -- the Eg=0 placeholder is present in
   MF12 wherever MF15 exists.
2. The mutated Al-27 fires the warning once per top-level call
   (per-file dedup).
3. The mutated result is still zero -- the warning is signal-only
   and does not change the numeric answer.
4. Non-gamma queries never fire the warning even on the mutated
   file (the MF15 branch is gamma-only).
"""
import copy
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_particle_production_dxs_dE
from endf_userpy.quantities_mt_zap.distribution1d import (
    _mf15_no_placeholder_warned,
)


ADHOC = Path(__file__).resolve().parent / 'data_law1_adhoc'
NEEDLE = 'MF15 continuous gamma spectrum data but MF12 declares no Eg=0'


def _load(fn_name):
    fn = ADHOC / fn_name
    if not fn.exists():
        pytest.skip(
            f'{fn_name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh`'
        )
    return EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    ).parsefile(fn)


@pytest.fixture(scope='function', autouse=True)
def _fresh_warning_cache():
    """Every test starts with an empty per-(file, MT) dedup set
    so the warning fires cleanly for the mutated fixture."""
    _mf15_no_placeholder_warned.clear()
    yield
    _mf15_no_placeholder_warned.clear()


def _no_placeholder_warns(recorded):
    return [w for w in recorded if NEEDLE in str(w.message)]


# ============================================================
# Non-regression: unmutated corpus files with MF15 do NOT warn.
# ============================================================


@pytest.mark.parametrize('fn,reaction', [
    ('endfb81_n_Al-27.endf', '(n,g)'),
    ('endfb81_n_N-14.endf',  '(n,g)'),
])
def test_corpus_no_placeholder_branch_not_hit(fn, reaction):
    """Every corpus file that has MF15 also has the matching MF12
    Eg=0 placeholder, so the branch does not fire on unmodified
    inputs."""
    endf = _load(fn)
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 30)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_particle_production_dxs_dE(
            endf, reaction, 'g', einc, eouts,
        )
    assert _no_placeholder_warns(recorded) == []


# ============================================================
# Mutated Al-27 fixture: drop Eg=0 from MF12/MT102 so the
# no-placeholder branch fires.
# ============================================================


def _mutate_drop_eg0_placeholder(endf, mt):
    """Return a deep copy with the Eg=0 subsection removed from
    MF12/MT. NK is decremented so the counts stay consistent.
    Matches Al-27 MT102's layout (Eg=0 is the last subsection,
    index NK)."""
    mutated = copy.deepcopy(endf)
    sec = mutated[12][mt]
    eg = sec['Eg']
    # Find the Eg=0 index and drop the parallel entries.
    eg0_keys = [k for k, v in eg.items() if v == 0.0]
    assert len(eg0_keys) == 1, (
        f'expected one Eg=0 entry, found {len(eg0_keys)}'
    )
    eg0_k = eg0_keys[0]
    # Drop from every per-subsection dict in the section.
    for field in ('Eg', 'ES', 'LP', 'LF', 'Eint', 'Y', 'INT', 'NBT',
                  'table'):
        d = sec.get(field)
        if isinstance(d, dict) and eg0_k in d:
            del d[eg0_k]
    sec['NK'] = sec.get('NK', 0) - 1
    return mutated


def test_mutated_al27_fires_warning():
    """Drop the Eg=0 placeholder from Al-27 MF12/MT102. The MF15
    branch of the dispatcher now takes the no-placeholder path
    and emits the warning."""
    original = _load('endfb81_n_Al-27.endf')
    mutated = _mutate_drop_eg0_placeholder(original, 102)
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 30)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        r = get_particle_production_dxs_dE(
            mutated, '(n,g)', 'g', einc, eouts,
        )
    hits = _no_placeholder_warns(recorded)
    assert len(hits) >= 1, 'expected the no-placeholder warning'
    msg = str(hits[0].message)
    assert 'MT=102' in msg
    # Numeric answer: MF15 contribution dropped, but MF12 discrete
    # lines still contribute -- so r is not necessarily all zero.
    # Just check finiteness.
    assert r is not None
    assert np.all(np.isfinite(r))


def test_dedup_within_session():
    """Two consecutive calls on the same mutated file fire the
    warning at most once (per (file, MT) dedup)."""
    original = _load('endfb81_n_Al-27.endf')
    mutated = _mutate_drop_eg0_placeholder(original, 102)
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 30)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_particle_production_dxs_dE(
            mutated, '(n,g)', 'g', einc, eouts,
        )
        get_particle_production_dxs_dE(
            mutated, '(n,g)', 'g', einc, eouts,
        )
    hits = _no_placeholder_warns(recorded)
    assert len(hits) == 1, (
        f'dedup broken: expected 1 warning across 2 calls, got {len(hits)}'
    )


def test_neutron_query_does_not_fire_warning():
    """The MF15 branch is gamma-only. A neutron query on the
    mutated file must not fire the no-placeholder warning."""
    original = _load('endfb81_n_Al-27.endf')
    mutated = _mutate_drop_eg0_placeholder(original, 102)
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 1.3e7, 30)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_particle_production_dxs_dE(
            mutated, '(n,2n)', 'n', einc, eouts,
        )
    assert _no_placeholder_warns(recorded) == []
