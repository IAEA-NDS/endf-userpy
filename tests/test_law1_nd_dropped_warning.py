"""Tests for the MF6/LAW=1 ND>0 discrete-line drop warning
(issue #102 / audit D2).

`get_particle_production_dxs_dE(broadening=None)` walks only the
continuum part of each admitted MT. MF6/LAW=1 subsections with
ND>0 (discrete-line encoding, JENDL-5 partial-inelastic gamma
cascades and Al-27 (n,n_i)/(n,p_i)/(n,a_i) cascades) have their
discrete deltas correctly excluded from the sum -- a delta on a
finite E' grid integrates to zero almost everywhere -- but the
exclusion was silent. The 1D dxs/dE for such MTs showed only the
continuum background and users had no signal that the discrete
lines were being dropped.

Fix (Option D from the design discussion): keep the existing
"unbroadened means unbroadened" numeric behaviour but emit ONE
summary UserWarning per top-level call naming every dropped MT
and pointing at the `broadening=` argument that routes the
delta through the MF6/LAW=1 discrete-line broadening folder in
`ddx_broadening.compute_dxs_dE_law1_discrete_broadened`.

Mirrors `_warn_discrete_dropped_from_unbroadened_ddx` (issue
#21) for the DDX side.

These tests pin:

1. Files with MF6/LAW=1 ND>0 gamma content (Al-27, JENDL-5
   Cu-63) fire the warning on the unbroadened path with the
   naming details.
2. Passing `broadening=` suppresses the warning (the deltas are
   folded rather than dropped).
3. Files with no MF6/LAW=1 ND>0 for the requested ejectile
   (H-1) do not warn.
4. Non-regression: numeric answer of the unbroadened call is
   unchanged (the warning is signal-only; no numeric change).
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_particle_production_dxs_dE


ADHOC = Path(__file__).resolve().parent / 'data_law1_adhoc'
NEEDLE = 'LAW=1 ND>0'


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


def _law1_nd_warns(recorded):
    return [w for w in recorded if NEEDLE in str(w.message)]


# ============================================================
# Warning fires on files with MF6/LAW=1 ND>0 gamma content.
# ============================================================


@pytest.mark.parametrize('fn', [
    'endfb81_n_Al-27.endf',    # (n,n_i), (n,p_i), (n,a_i) cascades
    'jendl5_n_Cu-63.endf',     # (n,n_i) cascade
])
def test_unbroadened_dxs_dE_warns_on_law1_nd(fn):
    endf = _load(fn)
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 30)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_particle_production_dxs_dE(
            endf, '(n,total)', 'g', einc, eouts,
        )
    hits = _law1_nd_warns(recorded)
    assert len(hits) >= 1, (
        f'{fn}: expected LAW=1 ND>0 dropped warning, none fired'
    )
    msg = str(hits[0].message)
    # Warning must name at least one MT and point at broadening=
    assert 'MT=' in msg
    assert 'broadening' in msg


def test_broadened_dxs_dE_does_not_warn():
    """Passing `broadening=` routes the deltas through the MF6/LAW=1
    discrete-line broadening folder rather than dropping them; no
    LAW=1 ND>0 drop warning fires."""
    endf = _load('endfb81_n_Al-27.endf')
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 30)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_particle_production_dxs_dE(
            endf, '(n,total)', 'g', einc, eouts, broadening=3e4,
        )
    assert _law1_nd_warns(recorded) == []


# ============================================================
# Files without MF6/LAW=1 ND>0 content don't warn.
# ============================================================


def test_no_law1_nd_content_does_not_warn():
    """H-1 has MF6/MT102 with LAW=2 gamma (angular-only, no ND>0
    lines) and LAW=4 for the recoil deuteron. No MF6/LAW=1 ND>0
    content anywhere -- the warning must not fire."""
    endf = _load('endfb81_n_H-1.endf')
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 30)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_particle_production_dxs_dE(
            endf, '(n,total)', 'g', einc, eouts,
        )
    assert _law1_nd_warns(recorded) == []


def test_neutron_query_does_not_warn():
    """The warning is gamma-only (has_mf6_law1_discrete_lines
    returns False for ZAP != gamma). A neutron query on the same
    file must not fire the LAW=1 ND>0 warning even if the file has
    LAW=1 ND>0 gamma content."""
    endf = _load('endfb81_n_Al-27.endf')
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 30)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_particle_production_dxs_dE(
            endf, '(n,2n)', 'n', einc, eouts,
        )
    assert _law1_nd_warns(recorded) == []


# ============================================================
# Non-regression: numeric result unchanged.
# ============================================================


def test_numeric_unchanged_after_warning_addition():
    """The warning is signal-only. The unbroadened numeric answer
    is unchanged; the continuum contribution of each admitted MT
    is what it was pre-fix. Compare an in-warning call (default
    warnings) against one where the warning is suppressed."""
    endf = _load('endfb81_n_Al-27.endf')
    einc = np.array([1.4e7])
    eouts = np.linspace(1e5, 8e6, 40)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r_ignore = get_particle_production_dxs_dE(
            endf, '(n,total)', 'g', einc, eouts,
        )
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        r_always = get_particle_production_dxs_dE(
            endf, '(n,total)', 'g', einc, eouts,
        )
    np.testing.assert_array_equal(r_ignore, r_always)
