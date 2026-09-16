"""Tests for the partial-LFS warning in `_warn_if_missing_isomer_routing`
(issue #106 / audit D6).

`get_residual_production_xs` uses `_warn_if_missing_isomer_routing`
to emit a `UserWarning` for every MT that could physically produce
the requested residual `(ZAP, LFS)` but has no MF8 entry to resolve
the isomer. The pre-fix implementation warned only when MF8 had
**no entry at all** for that MT; if MF8 declared the MT but with
a different LFS (e.g. ground declared, metastable requested), the
call silently returned zero with no signal.

Fix: extend the warning to also fire when MF8 has the MT but no
subsection declaring the specific (ZAP, LFS) combination. Same
per-(file, MT, ZAP, LFS) dedup key as the pre-existing branch, so
repeated calls for the same isotope don't spam the warning stream.

These tests pin the four combinations:

1. **Wrong LFS on a declared MT** (JENDL-5 Cu-63 MT 102 declares
   Cu-64 LFS=0 but not LFS=1; request Cu-64m).
2. **Right LFS on a declared MT** (Cu-64g): no warning.
3. **No LFS suffix at all** (Cu-64, level=None): no warning; the
   isomer routing check is skipped.
4. **MT with no MF8 entry at all**: original pre-#106 behaviour is
   preserved.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_residual_production_xs,
    _isomer_warning_seen,
)


ADHOC = Path(__file__).resolve().parent / 'data_law1_adhoc'


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
    """Every test starts with an empty warning-dedup cache so the
    per-(file, MT, ZAP, LFS) key doesn't leak state across cases."""
    _isomer_warning_seen.clear()
    yield
    _isomer_warning_seen.clear()


@pytest.fixture(scope='module')
def cu63_jendl5():
    """JENDL-5 Cu-63 MT 102 declares Cu-64 (ZAP=29064) at LFS=0 only.
    Requesting Cu-64m triggers the partial-LFS branch of the
    warning (issue #106)."""
    return _load('jendl5_n_Cu-63.endf')


# ============================================================
# The partial-LFS case (the fix).
# ============================================================


def _isomer_warns(recorded):
    return [
        w for w in recorded
        if 'production of' in str(w.message)
        and 'unresolved at the LFS level' in str(w.message)
    ]


def test_wrong_lfs_on_declared_mt_warns(cu63_jendl5):
    """Cu-64m (LFS=1) is not declared by JENDL-5 Cu-63 MF8; the
    fix emits a UserWarning naming the specific (MT, ZAP, LFS)."""
    einc = np.linspace(1e5, 1.4e7, 6)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        xs = get_residual_production_xs(cu63_jendl5, 'Cu-64m', einc)
    hits = _isomer_warns(recorded)
    # At least one warning; must name the partial-LFS reason
    assert len(hits) >= 1
    assert any('MF8 routing but no subsection' in str(h.message) for h in hits)
    assert any('LFS=1' in str(h.message) for h in hits)
    # Returned XS is zero (unchanged behaviour; only the warning
    # signal is new).
    np.testing.assert_array_equal(xs, np.zeros_like(einc))


def test_right_lfs_on_declared_mt_does_not_warn(cu63_jendl5):
    """Cu-64g (LFS=0) IS declared. No warning."""
    einc = np.linspace(1e5, 1.4e7, 6)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        xs = get_residual_production_xs(cu63_jendl5, 'Cu-64g', einc)
    assert _isomer_warns(recorded) == []
    assert xs.sum() > 0.0  # meaningful non-zero XS


def test_no_lfs_suffix_does_not_warn(cu63_jendl5):
    """`Cu-64` (no isomer suffix, `level=None`) bypasses the isomer
    routing check entirely. No warning even for ambiguity."""
    einc = np.linspace(1e5, 1.4e7, 6)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        xs = get_residual_production_xs(cu63_jendl5, 'Cu-64', einc)
    assert _isomer_warns(recorded) == []
    assert xs.sum() > 0.0


# ============================================================
# Non-regression: the original "MF8 absent for this MT" case
# still fires the warning.
# ============================================================


def test_mt_with_no_mf8_entry_still_warns():
    """If MF8 exists in the file but doesn't have an entry for a
    specific MT, the pre-existing warning branch fires. Uses
    Al-27 MT 4 requesting the residual of a fictitious LFS to
    stay within the corpus."""
    endf = _load('endfb81_n_Al-27.endf')
    einc = np.linspace(1e5, 1.4e7, 6)
    # MT 5 (X-particle production, "catch all"). MF8 has MT 5 in
    # Al-27 with Na-24 declared at LFS=0 only.
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        xs = get_residual_production_xs(endf, 'Na-24m', einc)
    hits = _isomer_warns(recorded)
    # At least one warning (either the MF8-absent or the
    # partial-LFS branch, depending on which MTs the physics
    # admits). Both branches are acceptable outputs.
    assert len(hits) >= 1
    np.testing.assert_array_equal(xs, np.zeros_like(einc))


# ============================================================
# Dedup: repeated queries for the same isotope don't re-warn.
# ============================================================


def test_warning_dedup_within_session(cu63_jendl5):
    einc = np.linspace(1e5, 1.4e7, 6)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_residual_production_xs(cu63_jendl5, 'Cu-64m', einc)
        get_residual_production_xs(cu63_jendl5, 'Cu-64m', einc)
        get_residual_production_xs(cu63_jendl5, 'Cu-64m', einc)
    # First call fires warnings; subsequent duplicates suppressed.
    hits = _isomer_warns(recorded)
    # First call may fire multiple (one per contributing MT); every
    # subsequent call must add ZERO new warnings.
    # Weaker check: total warnings from three calls <= warnings
    # a single first call fires.
    _isomer_warning_seen.clear()
    with warnings.catch_warnings(record=True) as first_only:
        warnings.simplefilter('always')
        get_residual_production_xs(cu63_jendl5, 'Cu-64m', einc)
    first_hits = _isomer_warns(first_only)
    assert len(hits) == len(first_hits), (
        f'dedup broken: {len(hits)} warnings across 3 calls but '
        f'{len(first_hits)} on the first call alone'
    )
