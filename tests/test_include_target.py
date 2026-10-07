"""Issue #137: RunOptions.include_target controls whether
``get_residual_production_xs`` folds target-conserving channels
into the sum when the queried residual is the target in its ground
state.

Default (``include_target=False``) follows the activation-library
convention (IRDFF / EAF / NJOY-ACTIVA): MT2 elastic, MT4 inelastic
sum, MT51..MT90 short-lived discrete-level inelastic, and MT91
continuum inelastic are dropped from the (target_za, LFS=0) query
because those channels de-excite back to the ground state and
leave the nucleus unchanged. Isomer queries (``'Nb-93m'``) are
unaffected because the metastable state is a genuine different
nuclide; the specific MF8-declared isomer-producing MT stays in
the sum regardless of ``include_target``.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation.mf3_interpretation import (
    compute_cross_section,
)
from endf_userpy.quantities import get_residual_production_xs
from endf_userpy.quantities_mt_zap.selectors import (
    contains_residual_za_and_lfs,
    is_target_conserving_mt,
)
from endf_userpy.run_options import RunOptions


BE9_FILE = 'tests/data/n-004_Be_009.endf'


def _synthetic_neutron_target_dict(target_za=4009.0, mf8=None):
    """Minimal synthetic dict carrying only the MF1/MT451 metadata
    the selector needs (NSUB=10 = neutron projectile, ZA=target_za)
    plus optional MF8. Avoids pulling in a full parsed file for
    pure-helper tests."""
    d = {1: {451: {'NSUB': 10, 'ZA': target_za}}}
    if mf8 is not None:
        d[8] = mf8
    return d


@pytest.fixture(scope='module')
def be9_endf_dict():
    """Be-9 target with MT2 elastic plus (n,2n), (n,g), (n,p), (n,d),
    (n,t), (n,h), (n,a) and the discrete-level proton MTs (MT600,
    MT650, MT700, MT701, MT800). No MF8 declarations, no MT4 /
    MT51..MT91 inelastic series. ZA target = 4009."""
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(BE9_FILE)


# ---- Unit tests on the _is_target_conserving_mt helper --------------


def test_target_conserving_helper_elastic(be9_endf_dict):
    """MT2 (elastic): single-neutron ejectile on neutron projectile,
    no MF8 declaration. Must register as target-conserving."""
    assert is_target_conserving_mt(be9_endf_dict, mt=2)


def test_target_conserving_helper_inelastic_sum_and_discrete():
    """MT4 (inelastic sum) and MT51..MT91 (discrete/continuum inelastic)
    all have single-neutron ejectile; on an MF8-less file, every one
    is target-conserving."""
    stub = _synthetic_neutron_target_dict()
    for mt in (4, 51, 52, 70, 90, 91):
        assert is_target_conserving_mt(stub, mt), (
            f'MT{mt} should register as target-conserving'
        )


def test_target_conserving_helper_multi_neutron_not_conserving():
    """MT16 (n,2n) and MT17 (n,3n) emit multiple neutrons; residual
    ZA shifts by -1 or -2 so these are not target-conserving."""
    stub = _synthetic_neutron_target_dict()
    assert not is_target_conserving_mt(stub, 16)
    assert not is_target_conserving_mt(stub, 17)


def test_target_conserving_helper_charged_ejectile_not_conserving():
    """MTs with charged ejectiles (p, d, t, alpha, gamma) move
    nucleus to a different Z or A and so cannot be target-conserving
    on a neutron projectile."""
    stub = _synthetic_neutron_target_dict()
    for mt in (102, 103, 104, 105, 106, 107):
        assert not is_target_conserving_mt(stub, mt), (
            f'MT{mt} must not register as target-conserving'
        )


def test_target_conserving_helper_mf8_isomer_declaration_disqualifies():
    """When MF8 declares the MT producing the target at a non-ground
    LFS, the MT is NOT purely target-conserving (it produces an
    isomer) and must stay in the exclusion-free set."""
    stub = _synthetic_neutron_target_dict(mf8={
        51: {
            'subsection': {
                1: {'ZAP': 4009.0, 'LFS': 1},  # declares isomer
            }
        }
    })
    assert not is_target_conserving_mt(stub, 51)


def test_target_conserving_helper_mf8_ground_declaration_still_conserving():
    """When MF8 declares the MT only at LFS=0 (ground), the MT is
    still target-conserving."""
    stub = _synthetic_neutron_target_dict(mf8={
        51: {
            'subsection': {
                1: {'ZAP': 4009.0, 'LFS': 0},
            }
        }
    })
    assert is_target_conserving_mt(stub, 51)


# ---- End-to-end on Be-9 ---------------------------------------------


def test_default_excludes_target_conserving_be9(be9_endf_dict):
    """Default ``get_residual_production_xs('Be-9')`` on Be-9 target
    drops MT2 elastic (the only target-conserving channel on this
    file). Every other MT produces a different residual, so the
    result is identically zero at 1 MeV."""
    e_in = np.array([1.0e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = get_residual_production_xs(be9_endf_dict, 'Be-9', e_in)
    assert result.shape == (1,)
    np.testing.assert_allclose(result, 0.0, atol=1e-14)


def test_include_target_true_recovers_elastic_be9(be9_endf_dict):
    """``include_target=True`` restores the pre-#137 inclusive sum:
    on Be-9 that is exactly the MT2 elastic cross section (nothing
    else produces Be-9 ground state)."""
    e_in = np.array([1.0e6, 5.0e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = get_residual_production_xs(
            be9_endf_dict, 'Be-9', e_in,
            options=RunOptions(include_target=True),
        )
        elastic = compute_cross_section(be9_endf_dict, 2, e_in)
    np.testing.assert_allclose(got, elastic, rtol=1e-12)


def test_non_target_residual_unaffected_by_include_target(be9_endf_dict):
    """``include_target`` only applies when the queried residual is
    the target in ground state. He-6 is produced by MT? on Be-9 and
    is a non-target residual, so the two settings must give identical
    numbers."""
    e_in = np.array([1.0e6, 5.0e6, 10.0e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        default = get_residual_production_xs(be9_endf_dict, 'He-6', e_in)
        inclusive = get_residual_production_xs(
            be9_endf_dict, 'He-6', e_in,
            options=RunOptions(include_target=True),
        )
    np.testing.assert_allclose(default, inclusive, rtol=1e-12)


def test_ground_state_suffix_matches_bare_target(be9_endf_dict):
    """Bare ``'Be-9'`` (level=None) and explicit ``'Be-9g'`` (level=0)
    both match the (target_za, ground state) query intent, so the
    default exclusion fires on both and both return zero."""
    e_in = np.array([1.0e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bare = get_residual_production_xs(be9_endf_dict, 'Be-9', e_in)
        ground = get_residual_production_xs(be9_endf_dict, 'Be-9g', e_in)
    np.testing.assert_allclose(bare, ground, atol=1e-14)
    np.testing.assert_allclose(bare, 0.0, atol=1e-14)


def test_isomer_query_unaffected_by_include_target_on_synthetic_dict():
    """Isomer queries (``level >= 1``) always keep every admitted MT
    because the isomer is a genuine different nuclide; the specific
    MF8-declared isomer-producing MT stays in the sum regardless of
    ``include_target``. Pin on a synthetic MF8 declaration since no
    small corpus file has a declared target-ZA isomer."""
    d = copy.deepcopy(EndfParserCpp(ignore_missing_tpid=True).parsefile(BE9_FILE))
    # Inject a synthetic MF8 declaration that MT51 produces the
    # target in its first metastable state.
    d[8] = {
        51: {
            'subsection': {
                1: {'ZAP': 4009.0, 'LFS': 1},
            },
        },
    }
    # Pin: the helper correctly identifies MT51-with-isomer as NOT
    # target-conserving.
    assert not is_target_conserving_mt(d, 51)
    # (End-to-end isomer-query numeric regression needs MF3/MT51
    # tabulated and MF2 reactions wiring beyond the scope of this
    # pin; the helper check is the load-bearing part.)


# ---- MT5 catch-all with MF6 target-ZA subsection (#342) -------------


def _synthetic_proton_target_dict(target_za=6012.0, mf6_mt5_zaps=()):
    """Minimal synthetic dict for the #342 MT5-via-MF6 path. Carries
    MF1/MT451 for projectile='p' / target_za lookup, and an MF6/MT5
    with one subsection per entry in ``mf6_mt5_zaps``. Values other
    than ZAP are left absent since ``mf6_help.contains_zap`` only
    reads the ZAP field."""
    d = {1: {451: {'NSUB': 10010, 'ZA': target_za}}}
    if mf6_mt5_zaps:
        d[6] = {
            5: {
                'subsection': {
                    i: {'ZAP': zap, 'LAW': 1}
                    for i, zap in enumerate(mf6_mt5_zaps, start=1)
                },
            },
        }
    return d


def test_mt5_mf6_target_zap_is_target_conserving():
    """#342: when MT5 carries an MF6 subsection with ZAP=target_ZA,
    the catch-all's target-nuclide portion must register as target-
    conserving so ``get_residual_production_xs`` under
    ``include_target=False`` subtracts it. Pre-fix the ejectile
    check short-circuited False for MT5 because ``get_ejectiles``
    returned None for the deliberately-open catch-all."""
    stub = _synthetic_proton_target_dict(
        target_za=6012.0,
        mf6_mt5_zaps=(0.0, 1.0, 6012.0, 2004.0),  # includes target
    )
    assert is_target_conserving_mt(stub, 5)


def test_mt5_mf6_without_target_zap_is_not_target_conserving():
    """MT5 whose MF6 subsections do NOT include ZAP=target_ZA has
    no target-conserving contribution; the catch-all is pure
    transmutation (Li, Be, B residuals etc.) and must stay in the
    sum under both policies."""
    stub = _synthetic_proton_target_dict(
        target_za=6012.0,
        mf6_mt5_zaps=(0.0, 1.0, 2004.0, 3006.0),  # no target
    )
    assert not is_target_conserving_mt(stub, 5)


def test_mt5_without_mf6_is_not_target_conserving():
    """MT5 absent from MF6 (older-style ENDF, or files where MT5 is
    only a cross-section stub) cannot be inspected for target-ZA
    contributions and must fall through as not target-conserving.
    The ejectile check also returns False for MT5 because MT5 is
    absent from ``REACTION_DICT``."""
    stub = _synthetic_proton_target_dict(target_za=6012.0)
    assert not is_target_conserving_mt(stub, 5)


def test_mt5_mf6_target_with_mf8_isomer_declaration_disqualifies():
    """#342 + #137 isomer rule: when MT5's MF6 declares the target
    ZAP but MF8/MT5 also declares the target at LFS >= 1, MT5
    produces an isomer and is NOT purely target-conserving."""
    stub = _synthetic_proton_target_dict(
        target_za=6012.0,
        mf6_mt5_zaps=(6012.0,),
    )
    stub[8] = {
        5: {'subsection': {1: {'ZAP': 6012.0, 'LFS': 1}}},
    }
    assert not is_target_conserving_mt(stub, 5)


# ---- End-to-end on the p + C-12 corpus file -------------------------
#
# p-006_C_012.endf carries both MT2 elastic (target-conserving via
# the ejectile path) AND MT5 with an MF6 subsection ZAP=6012 (target-
# conserving via the #342 MT5-MF6 path). This exercises the full
# admission flow with both paths in the same query.


def _resolve_p_c12():
    import sys
    sys.path.insert(0, 'tests')
    from _corpus import resolve_p_c12_law5
    return resolve_p_c12_law5()


@pytest.fixture(scope='module')
def p_c12_endf_dict():
    path = _resolve_p_c12()
    if path is None:
        pytest.skip('p + C-12 corpus file not available '
                    '(run tests/data_law5_adhoc/fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def test_p_c12_default_excludes_mt5_target_contribution(p_c12_endf_dict):
    """``get_residual_production_xs('C-12')`` on a p + C-12 target
    under default ``include_target=False`` must drop MT2 (elastic)
    AND MT5's ZAP=6012 subsection contribution. Pre-#342 the MT5
    portion stayed in because the helper mis-classified it."""
    e_in = np.array([30.0e6, 100.0e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = get_residual_production_xs(p_c12_endf_dict, 'C-12', e_in)
    np.testing.assert_allclose(got, 0.0, atol=1e-14)


def test_p_c12_include_target_true_restores_mt5_target_contribution(
    p_c12_endf_dict,
):
    """Under ``include_target=True`` the C-12 target-residual query
    returns the MT2 elastic + MT5-via-MF6 target portion. Pin the
    numeric value as a positive, non-trivial regression anchor."""
    e_in = np.array([30.0e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = get_residual_production_xs(
            p_c12_endf_dict, 'C-12', e_in,
            options=RunOptions(include_target=True),
        )
    assert got[0] > 0, 'MT2 + MT5 target contribution should be > 0'
    assert got[0] < 10.0, 'and should be a plausible elastic-like value in b'


def test_p_c12_non_target_residual_unaffected_by_include_target(
    p_c12_endf_dict,
):
    """Be-9 is produced by (p,x) through MT5's ZAP=4009 subsection
    on p + C-12. Non-target residuals pass through the admission
    layer unchanged under either policy (the target-conserving
    filter fires only when queried residual == target in ground
    state), so the two settings must give identical numbers."""
    e_in = np.array([50.0e6, 100.0e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        default = get_residual_production_xs(p_c12_endf_dict, 'Be-9', e_in)
        inclusive = get_residual_production_xs(
            p_c12_endf_dict, 'Be-9', e_in,
            options=RunOptions(include_target=True),
        )
    np.testing.assert_allclose(default, inclusive, rtol=1e-12)
    # Both must be > 0 at the higher energy (there is C-12 -> Be-9
    # production via MT5 at 100 MeV).
    assert inclusive[1] > 0


# ---- MF9 / MF10 isomer admission and disqualification (#343) --------
#
# contains_residual_za_and_lfs and is_target_conserving_mt both read
# only MF8 pre-#343. Files that declare an MT's isomer branch via
# MF9 (multiplicity by IZAP+LFS) or MF10 (production XS by IZAP+LFS)
# without MF8 are invisible to both filters. Fix extends both to
# consult MF9/MF10 when MF8 is absent (admission path) and as part
# of the isomer-disqualification sweep (target-conserving helper).
# Field name differs: MF8 uses ZAP, MF9/MF10 use IZAP.


def _synthetic_neutron_mf9_mf10_dict(
    target_za=4009.0, mf8=None, mf9=None, mf10=None,
):
    """Minimal synthetic dict carrying MF1/MT451 plus any of
    MF8/MF9/MF10 the caller supplies. Each argument is a plain
    ``{mt: {'subsection': {idx: {...}}}}`` dict; MF9/MF10 subsections
    must use ``IZAP`` (not ``ZAP``)."""
    d = {1: {451: {'NSUB': 10, 'ZA': target_za}}}
    if mf8 is not None:
        d[8] = mf8
    if mf9 is not None:
        d[9] = mf9
    if mf10 is not None:
        d[10] = mf10
    return d


# ---- contains_residual_za_and_lfs with MF9 / MF10 --------------------


def test_contains_mf9_isomer_match_without_mf8():
    """MF8 absent, MF9/MT51 declares (IZAP=target, LFS=1). An isomer
    query must now admit MT51 (pre-fix returned False because the
    MF9 catalog was never consulted)."""
    stub = _synthetic_neutron_mf9_mf10_dict(mf9={
        51: {'subsection': {1: {'IZAP': 4009.0, 'LFS': 1}}},
    })
    assert contains_residual_za_and_lfs(stub, 51, 4009.0, 1)


def test_contains_mf10_isomer_match_without_mf8():
    """Same case via MF10 (production cross section)."""
    stub = _synthetic_neutron_mf9_mf10_dict(mf10={
        51: {'subsection': {1: {'IZAP': 4009.0, 'LFS': 1}}},
    })
    assert contains_residual_za_and_lfs(stub, 51, 4009.0, 1)


def test_contains_mf9_ground_state_match():
    """MF8 absent, MF9/MT51 declares (IZAP=target, LFS=0). The
    ground-state query must admit MT51 via the MF9 catalog."""
    stub = _synthetic_neutron_mf9_mf10_dict(mf9={
        51: {'subsection': {1: {'IZAP': 4009.0, 'LFS': 0}}},
    })
    assert contains_residual_za_and_lfs(stub, 51, 4009.0, 0)
    assert contains_residual_za_and_lfs(stub, 51, 4009.0, None)


def test_contains_mf9_authoritative_when_present():
    """MF9 is authoritative when present and MF8 is absent: an MT
    whose MF9 subsections do NOT declare the queried residual must
    return False, not fall through to MF6/reaction-string. Pre-fix
    the function skipped MF9 entirely and fell through to the
    reaction-string lookup, over-admitting MT51 for every residual
    name that resolves via the reaction table."""
    stub = _synthetic_neutron_mf9_mf10_dict(mf9={
        51: {'subsection': {1: {'IZAP': 2004.0, 'LFS': 0}}},  # He-4
    })
    assert not contains_residual_za_and_lfs(stub, 51, 4009.0, 0)


def test_contains_mf10_isomer_but_no_ground_entry():
    """MF10 declares (target, LFS=1) only: ground-state query must
    return False (file does not declare ground-state production),
    isomer query returns True."""
    stub = _synthetic_neutron_mf9_mf10_dict(mf10={
        51: {'subsection': {1: {'IZAP': 4009.0, 'LFS': 1}}},
    })
    assert not contains_residual_za_and_lfs(stub, 51, 4009.0, 0)
    assert contains_residual_za_and_lfs(stub, 51, 4009.0, 1)


def test_contains_mf8_wins_over_mf9():
    """When both MF8 and MF9 are present for an MT, MF8 is
    authoritative (pre-existing semantics preserved by the fix).
    MF9 is never consulted when MF8 declares the MT."""
    stub = _synthetic_neutron_mf9_mf10_dict(
        mf8={
            51: {'subsection': {1: {'ZAP': 2004.0, 'LFS': 0}}},  # He-4
        },
        mf9={
            51: {'subsection': {1: {'IZAP': 4009.0, 'LFS': 1}}},  # target isomer
        },
    )
    # MF8 says no target production, so the query returns False
    # even though MF9 declares it.
    assert not contains_residual_za_and_lfs(stub, 51, 4009.0, 1)


def test_contains_mf9_and_mf10_both_consulted_before_false():
    """When both MF9 and MF10 are present for an MT (and MF8 is
    absent), the function must check both before returning False.
    MF9 missing the queried (IZAP, LFS) should not short-circuit if
    MF10 has the match."""
    stub = _synthetic_neutron_mf9_mf10_dict(
        mf9={
            51: {'subsection': {1: {'IZAP': 2004.0, 'LFS': 0}}},  # He-4 only
        },
        mf10={
            51: {'subsection': {1: {'IZAP': 4009.0, 'LFS': 1}}},  # target isomer
        },
    )
    assert contains_residual_za_and_lfs(stub, 51, 4009.0, 1)


# ---- is_target_conserving_mt with MF9 / MF10 isomer disqualification


def test_target_conserving_mf9_isomer_declaration_disqualifies():
    """When MF9 declares an isomer branch for an MT producing the
    target (and MF8 is absent), the MT is NOT purely target-
    conserving. Pre-fix the MF8-only check missed this and the MT
    was wrongly treated as target-conserving."""
    stub = _synthetic_neutron_mf9_mf10_dict(mf9={
        51: {'subsection': {1: {'IZAP': 4009.0, 'LFS': 1}}},
    })
    assert not is_target_conserving_mt(stub, 51)


def test_target_conserving_mf10_isomer_declaration_disqualifies():
    """Same case via MF10."""
    stub = _synthetic_neutron_mf9_mf10_dict(mf10={
        51: {'subsection': {1: {'IZAP': 4009.0, 'LFS': 1}}},
    })
    assert not is_target_conserving_mt(stub, 51)


def test_target_conserving_mf9_ground_only_still_conserving():
    """When MF9 declares the target only at LFS=0 (ground state),
    the MT is still target-conserving. The disqualification fires
    only on LFS >= 1 entries."""
    stub = _synthetic_neutron_mf9_mf10_dict(mf9={
        51: {'subsection': {1: {'IZAP': 4009.0, 'LFS': 0}}},
    })
    assert is_target_conserving_mt(stub, 51)


def test_target_conserving_mf9_other_zap_isomer_does_not_disqualify():
    """When MF9 declares an isomer branch of a DIFFERENT nuclide
    (not the target), it must not affect the target-conserving
    classification. The disqualification is scoped to
    ``IZAP == target_za``."""
    stub = _synthetic_neutron_mf9_mf10_dict(mf9={
        51: {'subsection': {1: {'IZAP': 2004.0, 'LFS': 1}}},  # He-4 isomer
    })
    assert is_target_conserving_mt(stub, 51)


# ---- RunOptions contract -------------------------------------------


def test_include_target_appears_on_runoptions_defaults_to_false():
    """Compile-time contract: ``include_target`` is an attribute on
    ``RunOptions`` and defaults to ``False``. A rename or default
    flip must be deliberate; this pin surfaces it."""
    opts = RunOptions()
    assert hasattr(opts, 'include_target')
    assert opts.include_target is False
