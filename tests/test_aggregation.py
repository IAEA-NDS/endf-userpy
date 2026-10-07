"""Issue #135: RunOptions.aggregation = 'top_down' vs 'bottom_up'.

Default ('bottom_up') preserves the pre-#135 behaviour: scalar XS
queries reconstruct the aggregate from the most fine-grained MF3
MTs available (leaf frontier). Top-down uses the most aggregated
MF3-tabulated MT that can address the query; on files that
tabulate both the parent sum-MT and its children, the parent is
returned directly and interpolation-grid-mismatch error between
separately-tabulated children is avoided (this is the error that
produced the ~0.04% Nb-93 TENDL-2025 discrepancy in the issue).

Differential APIs silently ignore ``aggregation`` and always use
bottom-up because the ENDF-6 data model forces it: sum MTs do not
carry per-ejectile distributions. Scalar XS entry points emit a
single summary UserWarning per top-level call when top-down is
requested but the user's MT is absent from MF3 and the admission
frontier has to descend.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation.mf3_interpretation import (
    compute_cross_section,
)
from endf_userpy.quantities import (
    get_particle_production_dxs_dE,
    get_reaction_xs,
    get_residual_production_xs,
)
from endf_userpy.quantities_mt_zap.selectors import (
    is_mf_sum_tree_leaf,
    is_mf_sum_tree_root,
)
from endf_userpy.run_options import RunOptions


BE9_FILE = 'tests/data/n-004_Be_009.endf'


# ---- RunOptions validation -----------------------------------------


def test_aggregation_default_is_bottom_up():
    opts = RunOptions()
    assert opts.aggregation == 'bottom_up'


def test_aggregation_accepts_top_down():
    opts = RunOptions(aggregation='top_down')
    assert opts.aggregation == 'top_down'


def test_aggregation_rejects_unknown_value():
    with pytest.raises(ValueError, match='aggregation must be one of'):
        RunOptions(aggregation='sideways')


# ---- Primitive helpers on synthetic dicts --------------------------


def _mf3_dict(mts):
    """Minimal dict carrying MF3 with the listed MTs (just presence,
    no actual XS content needed by the selector primitives)."""
    return {3: {mt: {} for mt in mts}}


def test_is_mf_sum_tree_root_single_root():
    """MT1 is root when it's in MF3; MT2 and MT3 (its children in
    SUM_RULES) have MT1 as ancestor so they're not roots."""
    d = _mf3_dict({1, 2, 3, 51, 52})
    assert is_mf_sum_tree_root(d, 1)
    assert not is_mf_sum_tree_root(d, 2)
    assert not is_mf_sum_tree_root(d, 3)
    assert not is_mf_sum_tree_root(d, 51)


def test_is_mf_sum_tree_root_missing_parent_promotes_children():
    """When MT1 is absent but MT2 and MT3 are in MF3, both are
    sum-tree-roots. MT51 (child of MT4 which is child of MT3) is
    still not a root because MT3 is in MF3."""
    d = _mf3_dict({2, 3, 51, 52})
    assert is_mf_sum_tree_root(d, 2)
    assert is_mf_sum_tree_root(d, 3)
    assert not is_mf_sum_tree_root(d, 51)


def test_is_mf_sum_tree_leaf_single_leaf():
    """Dual check: MT51 is a sum-tree-leaf (no descendant in MF3);
    MT1 is not (MT2 or MT3 are descendants in MF3)."""
    d = _mf3_dict({1, 2, 3, 51})
    assert is_mf_sum_tree_leaf(d, 51)
    assert is_mf_sum_tree_leaf(d, 2)  # MT2 has no SUM_RULES children
    assert not is_mf_sum_tree_leaf(d, 1)
    assert not is_mf_sum_tree_leaf(d, 3)


# ---- Scalar XS: top-down matches direct MF3 ------------------------


@pytest.fixture(scope='module')
def be9_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(BE9_FILE)


def test_top_down_total_matches_direct_mf3_be9(be9_endf_dict):
    """``get_reaction_xs('(n,total)')`` under top-down returns the
    MF3/MT1 interpolation bit-for-bit, with no reconstruction-from-
    children grid-mismatch error. Pins the core fix for #135."""
    e_in = np.array([2.5e6, 7.5e6, 14.0e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        direct = compute_cross_section(be9_endf_dict, 1, e_in)
        top_down = get_reaction_xs(
            be9_endf_dict, '(n,total)', e_in,
            options=RunOptions(aggregation='top_down'),
        )
    np.testing.assert_allclose(top_down, direct, rtol=0, atol=1e-14)


def test_bottom_up_default_still_reconstructs_be9(be9_endf_dict):
    """Default ``aggregation='bottom_up'`` continues to reconstruct
    from leaves; the result matches direct MF3/MT1 within
    interpolation-grid tolerance but is NOT bit-exact."""
    e_in = np.array([2.5e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        direct = compute_cross_section(be9_endf_dict, 1, e_in)
        bottom_up = get_reaction_xs(be9_endf_dict, '(n,total)', e_in)
    assert np.allclose(bottom_up, direct, rtol=1e-5, atol=0)


def test_top_down_nonelas_respects_user_scope_be9(be9_endf_dict):
    """When the user asks for ``(n,nonelas)`` (MT3) and both MT1
    and MT3 are in MF3, top-down admission must NOT climb above MT3
    to MT1 (which is outside the user's scope). Pins the user-scope
    ceiling walk in the top-down branch of satisfies_select_heuristic.
    Pre-fix the ceiling check was skipped and the admission returned
    zero (MT3 dropped because MT1 was in MF3)."""
    e_in = np.array([5.0e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        direct = compute_cross_section(be9_endf_dict, 3, e_in)
        top_down = get_reaction_xs(
            be9_endf_dict, '(n,nonelas)', e_in,
            options=RunOptions(aggregation='top_down'),
        )
    np.testing.assert_allclose(top_down, direct, rtol=0, atol=1e-14)
    assert float(top_down[0]) > 0


# ---- Fallback warning on scalar XS ---------------------------------


def test_top_down_warning_when_user_mt_not_in_mf3_be9(be9_endf_dict):
    """``get_reaction_xs('(n,n_1)')`` (MT51) on Be-9, which does NOT
    tabulate MT51 in MF3, must emit a single summary UserWarning
    naming MT51 and still return a sensible value (the top-down
    admission frontier descends to MT51's children, which also
    aren't present, so the result is zero -- the warning explains
    why)."""
    e_in = np.array([5.0e6])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _ = get_reaction_xs(
            be9_endf_dict, '(n,n_1)', e_in,
            options=RunOptions(aggregation='top_down'),
        )
    agg_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "aggregation='top_down'" in str(w.message)
    ]
    assert len(agg_warnings) == 1, (
        f'expected exactly one aggregation UserWarning, got '
        f'{len(agg_warnings)}: {[str(w.message) for w in agg_warnings]}'
    )
    assert 'MT=51' in str(agg_warnings[0].message)


def test_top_down_no_warning_when_user_mt_in_mf3_be9(be9_endf_dict):
    """``get_reaction_xs('(n,total)')`` with top-down on Be-9: MT1
    is in MF3, so no fallback happens and no aggregation warning
    is emitted."""
    e_in = np.array([2.5e6])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _ = get_reaction_xs(
            be9_endf_dict, '(n,total)', e_in,
            options=RunOptions(aggregation='top_down'),
        )
    agg_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "aggregation='top_down'" in str(w.message)
    ]
    assert len(agg_warnings) == 0


# ---- Differential APIs silently ignore aggregation ------------------


def test_differential_api_ignores_aggregation_be9(be9_endf_dict):
    """``get_particle_production_dxs_dE`` always uses bottom-up
    regardless of ``options.aggregation``: the ENDF-6 data model
    requires per-ejectile distributions from the leaves. The two
    policies must give identical outputs on a differential query."""
    e_in = np.array([5.0e6])
    e_out = np.linspace(0.0, 5.0e6, 20)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bu = get_particle_production_dxs_dE(
            be9_endf_dict, '(n,total)', 'n', e_in, e_out,
        )
        td = get_particle_production_dxs_dE(
            be9_endf_dict, '(n,total)', 'n', e_in, e_out,
            options=RunOptions(aggregation='top_down'),
        )
    np.testing.assert_allclose(td, bu, rtol=0, atol=1e-14)


# ---- Residual production respects aggregation ----------------------


def test_residual_production_bottom_up_unchanged_be9(be9_endf_dict):
    """Non-target residual on Be-9 (He-6 via (n,alpha)): default
    bottom-up admission is unchanged by #135. Pin the pre-#135
    behavior as a non-regression."""
    e_in = np.array([5.0e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        he6_default = get_residual_production_xs(
            be9_endf_dict, 'He-6', e_in,
        )
        he6_bottom_up = get_residual_production_xs(
            be9_endf_dict, 'He-6', e_in,
            options=RunOptions(aggregation='bottom_up'),
        )
    np.testing.assert_allclose(he6_default, he6_bottom_up, rtol=0, atol=1e-14)
    assert float(he6_default[0]) > 0
