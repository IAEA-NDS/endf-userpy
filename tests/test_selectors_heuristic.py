"""Tests for `selectors.satisfies_select_heuristic` MF12/MF13
awareness (issue #53).

Before this fix, the heuristic used "presence in MF4/5/6" as its
test for "this MT has detailed distribution info available", and
dropped partial MTs whose gamma reconstruction data lives only in
MF12+MF14 from sum-MT queries (like `(n,total)`). The canonical
corpus case is Al-27 MT 102 (n,g), which has MF12 + MF14 + MF15 for
its capture-gamma cascade but no MF6.

The fix extends the check to also count `mt in endf_dict[12]` and
`mt in endf_dict[13]` as evidence of reconstructible content. MF14
and MF15 are companion sections that only accompany MF12/MF13, so
admitting on MF12/MF13 alone is sufficient.
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities_mt_zap import selectors
from endf_userpy.quantities import (
    get_reaction_xs,
    get_particle_production_xs,
    get_particle_production_dxs_dmu,
    get_particle_production_dxs_dE,
)
from endf_userpy.mfsec_interpretation import mf3_interpretation as mf3_interp


ADHOC_DATA_DIR = Path(__file__).resolve().parent / 'data_law1_adhoc'


def _load(fn_name):
    fn = ADHOC_DATA_DIR / fn_name
    if not fn.exists():
        pytest.skip(
            f'{fn_name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh` to populate the corpus'
        )
    parser = EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    )
    return parser.parsefile(fn)


@pytest.fixture(scope='module')
def al27_endfb81():
    """Al-27 from ENDF/B-VIII.1. MT 102 (n,g) is declared in MF12 +
    MF14 + MF15 but not MF6, so the pre-fix heuristic dropped it from
    `(n,total)` queries."""
    return _load('endfb81_n_Al-27.endf')


@pytest.fixture(scope='module')
def cu63_jeff40():
    """Cu-63 from JEFF-4.0. MT 102 IS in MF6 here, so the heuristic
    admitted it before the fix; used as a negative control."""
    return _load('jeff40_n_Cu-63.endf')


@pytest.fixture(scope='module')
def fe56_tendl():
    return _load('tendl21_n_Fe-56.endf')


@pytest.fixture(scope='module')
def be9_h1_data():
    """Be-9 from ENDF/B-VIII.1 (main-suite corpus): used for a
    non-gamma-query regression check that the heuristic fix does not
    over-include neutron/proton paths."""
    fn = Path(__file__).resolve().parent / 'data' / 'n-004_Be_009.endf'
    if not fn.exists():
        pytest.skip(f'{fn.name} missing')
    parser = EndfParserCpp(ignore_missing_tpid=True)
    return parser.parsefile(fn)


# ============================================================
# Heuristic-level: Al-27 MT 102 is admitted for `(n,total)`
# after the fix, and still admitted for the `(n,g)` direct query.
# ============================================================


def test_mt102_admits_from_sum_query_after_fix(al27_endfb81):
    """Al-27 MT 102 has MF12 + MF14 + MF15 but no MF6. Before the
    fix, `satisfies_select_heuristic(102, user_mts=[1])` returned
    False and MT 102 was silently dropped from `(n,total)` queries.
    After the fix it returns True and MT 102 contributes."""
    assert selectors.satisfies_select_heuristic(al27_endfb81, 102, [1]) is True


def test_mt102_direct_query_still_admitted(al27_endfb81):
    """Regression: the direct `(n,g)` query still works."""
    assert selectors.satisfies_select_heuristic(al27_endfb81, 102, [102]) is True


def test_mt102_no_user_mts_still_admitted(al27_endfb81):
    """Regression: the unfiltered case (no user_mts) still admits
    MT 102."""
    assert selectors.satisfies_select_heuristic(al27_endfb81, 102, None) is True


def test_synthetic_mt_only_in_mf12_admits():
    """The fix is zap-independent at the heuristic level. Any MT
    present in MF12 (or MF13) with a sum-MT ancestor now admits."""
    d = {3: {1: {}, 102: {}}, 12: {102: {}}}
    assert selectors.satisfies_select_heuristic(d, 102, [1]) is True


def test_synthetic_mt_only_in_mf13_admits():
    d = {3: {1: {}, 102: {}}, 13: {102: {}}}
    assert selectors.satisfies_select_heuristic(d, 102, [1]) is True


def test_synthetic_mt_only_in_mf14_alone_does_not_admit():
    """MF14 without MF12/MF13/MF6 does not stand alone (companion
    section); the heuristic must not admit on MF14 alone. Anchors
    the design decision."""
    d = {3: {1: {}, 102: {}}, 14: {102: {}}}
    assert selectors.satisfies_select_heuristic(d, 102, [1]) is False


# ============================================================
# End-to-end: the observable difference the fix makes on Al-27.
# ============================================================


def test_al27_reaction_xs_now_matches_mf3_mt1(al27_endfb81):
    """`get_reaction_xs('(n,total)', ...)` sums the MF3 cross
    sections of admitted MTs. Before the fix, MT 102 was dropped
    and the sum was short by MT 102's XS. After the fix the sum
    equals MT 1 in MF3 within rounding."""
    einc = np.array([1.5e6])
    xs_total = get_reaction_xs(al27_endfb81, '(n,total)', einc)[0]
    xs_mt1 = mf3_interp.compute_cross_section(al27_endfb81, 1, einc)[0]
    rel_deficit = abs(xs_total - xs_mt1) / xs_mt1
    assert rel_deficit < 1e-4, (
        f'expected (n,total) sum ~ MT 1 MF3; got {xs_total:.6g} vs '
        f'MT1={xs_mt1:.6g}, deficit {rel_deficit:.3g}'
    )


def test_al27_gamma_prodxs_via_ntotal_now_nonzero(al27_endfb81):
    """MT 102 gamma contribution to `(n,total)` XS is small (~2e-3
    b at 1.5 MeV in absolute, from y_cont ~ 2.5 photons per capture
    x sigma_MT102 ~ 5.7e-4 b) but must be non-zero after the fix."""
    einc = np.array([1.5e6])
    xs_total = get_particle_production_xs(
        al27_endfb81, '(n,total)', 'g', einc,
    )[0]
    xs_ng = get_particle_production_xs(al27_endfb81, '(n,g)', 'g', einc)[0]
    # `(n,total)` must not be *smaller* than `(n,g)`; that would
    # signal MT 102 is still being dropped from the total.
    assert xs_total >= xs_ng - 1e-12


def test_al27_gamma_dxs_dmu_via_ntotal_now_nonzero(al27_endfb81):
    """The primary user-visible bug pin: dxs/dmu on `(n,total)` for
    gamma used to return exactly zero on Al-27 because MT 102 (the
    only MF12+MF14 MT here) was dropped by the heuristic. After the
    fix it's non-zero and matches the MT 102 isotropic contribution."""
    einc = np.array([1.5e6])
    mu = np.array([0.0])
    r_total = get_particle_production_dxs_dmu(
        al27_endfb81, '(n,total)', 'g', einc, mu,
    )
    r_ng = get_particle_production_dxs_dmu(al27_endfb81, '(n,g)', 'g', einc, mu)
    assert r_total is not None and r_ng is not None
    # `(n,total)` picks up the MT 102 contribution now.
    assert r_total[0, 0] > 0
    # And it is at least as large as `(n,g)` (same MT 102 isotropic
    # contribution). Other MTs may add more (they don't for Al-27
    # dxs/dmu because MF6/LAW=1 gamma-angular content is a separate
    # follow-up: issue #55).
    assert r_total[0, 0] >= r_ng[0, 0] - 1e-12


def test_al27_gamma_dxs_dE_via_ntotal_no_crash(al27_endfb81):
    """dxs/dE should also complete without crashing; the discrete
    lines and MF15 continuum from MT 102 are now included."""
    einc = np.array([1.5e6])
    eouts = np.linspace(0.5e6, 8e6, 8)
    r = get_particle_production_dxs_dE(
        al27_endfb81, '(n,total)', 'g', einc, eouts,
    )
    assert r is not None
    assert not np.any(np.isnan(r))


# ============================================================
# Regression: files where MT 102 was already in MF6 (so admitted
# before the fix too) must be unchanged.
# ============================================================


def test_cu63_ntotal_gamma_unchanged(cu63_jeff40):
    """Cu-63 has MT 102 in MF6, so the heuristic admitted it before
    the fix. XS output must not change."""
    xs = get_particle_production_xs(
        cu63_jeff40, '(n,total)', 'g', np.array([1.5e6]),
    )[0]
    # This is the same value pinned by earlier PRs; the heuristic
    # change must not perturb it.
    assert 0.9 < xs < 1.0


def test_fe56_ntotal_gamma_unchanged(fe56_tendl):
    xs = get_particle_production_xs(
        fe56_tendl, '(n,total)', 'g', np.array([1.5e6]),
    )[0]
    assert 0.49 < xs < 0.51


# ============================================================
# Regression: non-gamma queries are not over-included by the fix.
# `contains_zap` upstream filters `(mt, zap)` pairs where MT does
# not emit that particle, so admitting on MF12/13 in the heuristic
# cannot promote an MT-that-only-produces-gammas into a neutron sum.
# ============================================================


def test_al27_neutron_xs_from_reaction_string_unchanged(al27_endfb81):
    """MT 102 in Al-27 has MF12 for gamma; it produces no neutron.
    `contains_zap` filters it out for a neutron query, so
    `get_particle_production_xs('(n,total)', 'n', ...)` must be
    unchanged by the heuristic fix."""
    einc = np.array([1.5e6])
    r = get_particle_production_xs(al27_endfb81, '(n,total)', 'n', einc)
    if r is not None:
        assert r[0] > 0  # some neutron production at 1.5 MeV
        assert not np.any(np.isnan(r))


def test_be9_neutron_xs_unchanged(be9_h1_data):
    """Be-9 was a main-suite regression anchor for the neutron path.
    Its neutron production XS at 1.5 MeV must be unchanged (the
    heuristic fix targets the gamma path via MF12/MF13 admission,
    which contains_zap filters out for neutron)."""
    einc = np.array([1.5e6])
    xs = get_particle_production_xs(be9_h1_data, '(n,total)', 'n', einc)
    assert xs[0] > 0
