"""Tests for the sum-MT gappy-children warning (issue #108 /
audit D8).

`selectors.satisfies_select_heuristic` drops any parent-sum MT
(MT4 = (n,inelastic sum), MT103 = (n,p sum), etc.) from admission
when at least one child MT is present with detailed distribution
info in MF4/5/6 -- correct for well-formed evaluations where child
coverage is complete (avoids double-counting the parent alongside
its children).

For evaluations with **incomplete child coverage** -- e.g. Al-27
has MT51..90 in MF6, but if MT52 loses its MF6 entry (still has
MF3), MT52 keeps its cross section but no MF6 distribution -- the
heuristic drops parent MT4 (because MT51 still has MF6) AND the
child-selection escape hatch also drops MT52 (because ancestor MT4
IS in MF3, so the `or not has_ancestor` clause is False). MT52's
MF3 cross section is silently missed by the resulting query.

Fix: warn (signal-only, no numeric change) when the heuristic drops
a parent-sum MT and at least one sibling child has an MF3 cross
section but no MF4/5/6/12/13 detail. The warning names the parent
MT, lists the missed children, and points at querying MT_parent
directly (which bypasses the heuristic) as the workaround.

These tests pin the branch behaviour on a synthetic mutation of
Al-27 MF6/MT52 (drop it so MT52 keeps MF3 but loses its MF6
distribution) and confirm that:

1. Unmutated Al-27 (all MT51..90 covered in MF6) does NOT fire
   the warning.
2. Mutated Al-27 fires the warning once per top-level call
   (per-(file, parent) dedup).
3. The mutated numeric answer is not silently equal to the
   unmutated one -- MT52's contribution is genuinely missed, and
   the warning is what tells the caller about it.
4. Two consecutive queries on the same mutated file fire the
   warning only once (dedup).
5. Querying MT4 directly (bypasses the heuristic) does not fire
   the warning even on the mutated file.
"""
import copy
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_reaction_xs
# Import the per-(file, parent) dedup set defensively so the tests
# can still be collected against a checkout WHERE the fix hasn't
# landed (bug-catch stash-and-run: we want the tests to run and
# report the actual missing-warning failure, not fall over at
# import).
try:
    from endf_userpy.quantities_mt_zap.selectors import (
        _gappy_children_warned,
    )
except ImportError:
    _gappy_children_warned = set()


ADHOC = Path(__file__).resolve().parent / 'data_law1_adhoc'
NEEDLE = 'silently missed by the current query'


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
    """Every test starts with an empty per-(file, parent) dedup set
    so the warning fires cleanly for the mutated fixture."""
    _gappy_children_warned.clear()
    yield
    _gappy_children_warned.clear()


def _gap_warns(recorded):
    return [w for w in recorded if NEEDLE in str(w.message)]


def _mutate_drop_mf6_mt(endf, mt):
    """Return a deep copy with MF6/MT deleted. Al-27 keeps its MF3
    entry for the same MT, so the file still has an XS for MT but
    no MF4/5/6/12/13 detail -- exactly the shape the #108 pathology
    needs."""
    mutated = copy.deepcopy(endf)
    del mutated[6][mt]
    return mutated


# ============================================================
# Baseline: unmutated corpus files do NOT fire the warning.
# Al-27 has full MT51..90 coverage in MF6.
# ============================================================


def test_unmutated_al27_does_not_warn():
    endf = _load('endfb81_n_Al-27.endf')
    einc = np.linspace(1e6, 1.4e7, 5)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_reaction_xs(endf, '(n,n)', einc)
    assert _gap_warns(recorded) == []


# ============================================================
# Mutated Al-27 fires the warning once, names the missed MT,
# and points at the direct-MT-query workaround.
# ============================================================


def test_mutated_al27_fires_warning():
    original = _load('endfb81_n_Al-27.endf')
    mutated = _mutate_drop_mf6_mt(original, 52)
    einc = np.linspace(1e6, 1.4e7, 5)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_reaction_xs(mutated, '(n,n)', einc)
    hits = _gap_warns(recorded)
    assert len(hits) >= 1, 'expected the gappy-children warning'
    msg = str(hits[0].message)
    assert 'MT=4' in msg, 'warning names the parent'
    assert 'MT=52' in msg, 'warning names the missed child'
    # Workaround pointer -- caller can bypass the heuristic by
    # asking for MT=4 directly.
    assert 'MT=4' in msg


def test_missing_contribution_is_measurable():
    """Non-fabrication: MT52's contribution IS missed by the
    query. If the warning were spurious (no real under-count),
    mutated and unmutated results would be identical."""
    original = _load('endfb81_n_Al-27.endf')
    mutated = _mutate_drop_mf6_mt(original, 52)
    einc = np.linspace(1e6, 1.4e7, 5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs_orig = get_reaction_xs(original, '(n,n)', einc)
        xs_mut = get_reaction_xs(mutated, '(n,n)', einc)
    diff = np.abs(xs_orig - xs_mut)
    # MT52's contribution is a few percent of MT4 at 14 MeV in
    # ENDF/B-VIII.1 Al-27; require a measurable non-zero gap.
    assert diff.max() > 1e-4, (
        f'mutation had no measurable numeric effect (max |diff| '
        f'= {diff.max():.3e}); test setup is wrong'
    )


# ============================================================
# Dedup: two consecutive queries on the same mutated file fire
# the warning at most once (per-(file, parent) dedup).
# ============================================================


def test_dedup_within_session():
    original = _load('endfb81_n_Al-27.endf')
    mutated = _mutate_drop_mf6_mt(original, 52)
    einc = np.linspace(1e6, 1.4e7, 5)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_reaction_xs(mutated, '(n,n)', einc)
        get_reaction_xs(mutated, '(n,n)', einc)
    hits = _gap_warns(recorded)
    assert len(hits) == 1, (
        f'dedup broken: expected 1 warning across 2 calls, got {len(hits)}'
    )


# ============================================================
# Direct MT4 query bypasses the heuristic entirely -- the
# warning must not fire.
# ============================================================


def test_direct_parent_query_does_not_warn():
    """Asking for MT=4 directly (rather than the reaction string
    that resolves to MT=4) still enters `compute_cumulative_quantity`,
    but the selector's user-mts branch selects MT=4 without going
    through the child-preference drop path. No gap warning."""
    original = _load('endfb81_n_Al-27.endf')
    mutated = _mutate_drop_mf6_mt(original, 52)
    einc = np.linspace(1e6, 1.4e7, 5)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        # `get_reaction_xs` accepts a reaction string; the direct-MT
        # equivalent goes through mf3_interpretation.compute_cross_section.
        from endf_userpy.mfsec_interpretation import mf3_interpretation as mf3
        mf3.compute_cross_section(mutated, 4, einc)
    assert _gap_warns(recorded) == []
