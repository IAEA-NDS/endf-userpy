"""Regression tests for endf_userpy.quantities_mt_zap.selectors.

Pinned by issue #19: a JENDL-5 + NJOY (HEATR) merged file contains
MTs that are not reaction channels (MT 301..450) plus the X-particle
production sums (MT 201..207). Public APIs that iterate every MT in
MF3 must not raise on those rows.
"""

import pytest

import endf_userpy.quantities_mt_zap.selectors as selectors
import endf_userpy.primitives.physical_constants as physconst


def _neutron_endf_dict():
    """Smallest dict that satisfies prop.get_projectile (NSUB=10)."""
    return {1: {451: {'NSUB': 10}}}


@pytest.mark.parametrize("mt", [301, 302, 318, 444, 445, 446, 447, 450])
def test_contains_zap_skips_heatr_mts(mt):
    endf_dict = _neutron_endf_dict()
    zap_n = physconst.PARTICLE_ZAP['n']
    assert selectors.contains_zap(endf_dict, mt, zap_n) is False


@pytest.mark.parametrize("mt", [251, 252, 253])
def test_contains_zap_skips_dosimetry_like_mts(mt):
    endf_dict = _neutron_endf_dict()
    zap_n = physconst.PARTICLE_ZAP['n']
    assert selectors.contains_zap(endf_dict, mt, zap_n) is False


@pytest.mark.parametrize("mt,particle", [
    (201, 'n'), (202, 'g'), (203, 'p'), (204, 'd'),
    (205, 't'), (206, 'h'), (207, 'a'),
])
def test_contains_zap_x_production_matches_particle(mt, particle):
    """MT 201..207 are admitted for the matching particle and rejected
    for any other. The partials-vs-sum exclusion is satisfies_select_
    heuristic's job, not contains_zap's."""
    endf_dict = _neutron_endf_dict()
    zap = physconst.PARTICLE_ZAP[particle]
    assert selectors.contains_zap(endf_dict, mt, zap) is True
    other = physconst.PARTICLE_ZAP['p' if particle != 'p' else 'n']
    assert selectors.contains_zap(endf_dict, mt, other) is False


def test_contains_zap_real_channel_still_resolves():
    endf_dict = _neutron_endf_dict()
    zap_n = physconst.PARTICLE_ZAP['n']
    zap_g = physconst.PARTICLE_ZAP['g']
    # (n,2n) — emits two neutrons, no gamma.
    assert selectors.contains_zap(endf_dict, 16, zap_n) is True
    assert selectors.contains_zap(endf_dict, 16, zap_g) is False


def test_contains_zap_fission_special_case_preserved():
    endf_dict = _neutron_endf_dict()
    zap_n = physconst.PARTICLE_ZAP['n']
    zap_p = physconst.PARTICLE_ZAP['p']
    assert selectors.contains_zap(endf_dict, 18, zap_n) is True
    assert selectors.contains_zap(endf_dict, 18, zap_p) is False


# ---- X-production fallback resolution in satisfies_select_heuristic --------

ZAP_N = physconst.PARTICLE_ZAP['n']


def test_select_heuristic_excludes_x_production_sum_when_partials_have_xs():
    """When MF3 has neutron-emitting partials, MT 201 is suppressed."""
    endf_dict = {
        1: {451: {'NSUB': 10}},
        3: {16: {}, 17: {}, 91: {}, 201: {}},
    }
    assert not selectors.satisfies_select_heuristic(
        endf_dict, 201, op='xs', zap=ZAP_N
    )


def test_select_heuristic_admits_x_production_sum_without_partials():
    """Truncated file: only MT 201 in MF3, no neutron-emitting partials."""
    endf_dict = {
        1: {451: {'NSUB': 10}},
        3: {201: {}},
    }
    assert selectors.satisfies_select_heuristic(
        endf_dict, 201, op='xs', zap=ZAP_N
    )


def test_select_heuristic_partials_unchanged_when_x_branch_picks_partials():
    """Defer to existing logic: a user-named partial is still admitted."""
    endf_dict = {
        1: {451: {'NSUB': 10}},
        3: {16: {}, 17: {}, 91: {}, 201: {}},
    }
    assert selectors.satisfies_select_heuristic(
        endf_dict, 16, [16], op='xs', zap=ZAP_N
    )


def test_select_heuristic_x_branch_inert_without_op_zap():
    """Backward compat: callers that don't pass op/zap behave as before."""
    endf_dict = {
        1: {451: {'NSUB': 10}},
        3: {16: {}, 201: {}},
    }
    # MT 201 isn't in any SUM_RULES sum nor a partial of one, so the
    # legacy branch admits it (no sum-rule conflict).
    assert selectors.satisfies_select_heuristic(endf_dict, 201) is True


def test_select_heuristic_for_dexs_prefers_partials_with_mf6():
    endf_dict = {
        1: {451: {'NSUB': 10}},
        3: {16: {}, 201: {}},
        6: {16: {}},
    }
    assert not selectors.satisfies_select_heuristic(
        endf_dict, 201, op='dexs', zap=ZAP_N
    )


def test_select_heuristic_for_dexs_falls_back_to_sum_when_only_sum_has_mf6():
    endf_dict = {
        1: {451: {'NSUB': 10}},
        3: {16: {}, 201: {}},
        6: {201: {}},
    }
    assert selectors.satisfies_select_heuristic(
        endf_dict, 201, op='dexs', zap=ZAP_N
    )


def test_select_heuristic_for_dexs_excludes_partials_when_falling_back_to_sum():
    """When the partials path can't answer, partial MTs are also excluded
    so the cumulative sum doesn't mix the two sources."""
    endf_dict = {
        1: {451: {'NSUB': 10}},
        3: {16: {}, 201: {}},
        6: {201: {}},  # only sum has MF6
    }
    # MT 16 is a neutron-emitting partial; under the all-or-nothing
    # policy the partials path is rejected (no MF6 for 16), so MT 16
    # itself is also excluded from the dexs sum.
    assert not selectors.satisfies_select_heuristic(
        endf_dict, 16, op='dexs', zap=ZAP_N
    )


def test_select_heuristic_x_branch_unrelated_mt_falls_through():
    """An MT that is neither MT 201..207 nor a contributing partial is
    not affected by the X-production branch."""
    endf_dict = {
        1: {451: {'NSUB': 10}},
        3: {102: {}, 201: {}},
    }
    zap_g = physconst.PARTICLE_ZAP['g']
    # MT 102 (n,gamma) emits gammas, not neutrons; for zap=n, MT 102
    # is unrelated to the MT 201 decision and falls through to legacy
    # logic, which admits it (no sum rule conflict).
    assert selectors.satisfies_select_heuristic(
        endf_dict, 102, op='xs', zap=ZAP_N
    ) is True
    # And MT 102 is correctly handled by the gamma branch (sum mt 202).
    # Without a partial detail check failing, MT 102 still falls
    # through; the call should not raise.
    selectors.satisfies_select_heuristic(
        endf_dict, 102, op='xs', zap=zap_g
    )
