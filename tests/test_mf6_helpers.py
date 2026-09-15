"""Synthetic-dict tests for mfsec_interpretation.mf6_interpretation_helpers.

Pinned by an external code review of the gamma / MF6 path. The
predicates `has_disc_part`, `has_cont_part`, and `has_angdist_part`
are called from selectors.py, distribution1d.py, distribution2d.py,
and ddx_broadening.py to decide which contributions to admit into
DDX / dxs/dE / dxs/dmu sums. A subtle bug in `has_cont_part` early-
returned False on the first LAW=1 subsection when that subsection
was discrete-only, silently dropping continuum contributions from
later subsections for the same (MT, ZAP).

`daniel_notes.txt` explicitly notes that a (MT, ZAP) can have more
than one subsection (same ZAP with different LIP, for instance)
that must be summed, so this is a real corpus shape.
"""
import numpy as np

from endf_userpy.mfsec_interpretation import mf6_interpretation_helpers as mf6_help


def _law1_subsec(zap, nd_per_panel, nep_per_panel):
    """Build a LAW=1 subsection with the shape the predicates expect.

    `nd_per_panel` and `nep_per_panel` are equal-length lists of
    per-panel ND and NEP values. The predicates convert
    subsec['ND'].values() -> np.array and check whether any panel
    has NEP > ND (continuum present) or ND > 0 (discrete lines
    present).
    """
    assert len(nd_per_panel) == len(nep_per_panel)
    return {
        'ZAP': zap,
        'LAW': 1,
        'ND': {i: v for i, v in enumerate(nd_per_panel, start=1)},
        'NEP': {i: v for i, v in enumerate(nep_per_panel, start=1)},
    }


def _mf6_dict(*subsecs_for_mt51):
    """Wrap subsections into the endf_dict shape:
    endf_dict[6][51]['subsection'][idx] = subsec."""
    return {
        6: {
            51: {
                'subsection': {i: s for i, s in enumerate(subsecs_for_mt51, start=1)},
            }
        }
    }


def test_has_cont_part_single_discrete_only_subsec():
    """Baseline: single LAW=1 subsec with NEP == ND (discrete-only)
    reports no continuum. Unchanged by the fix."""
    d = _mf6_dict(_law1_subsec(0.0, [1], [1]))
    assert mf6_help.has_cont_part(d, 51, 0.0) is False


def test_has_cont_part_single_continuum_subsec():
    """Baseline: single LAW=1 subsec with NEP > ND reports continuum
    present. Unchanged by the fix."""
    d = _mf6_dict(_law1_subsec(0.0, [1], [5]))
    assert mf6_help.has_cont_part(d, 51, 0.0) is True


def test_has_cont_part_multi_subsec_first_discrete_only():
    """The bug this test pins. Two LAW=1 subsections for the same
    ZAP: the first is discrete-only (NEP == ND), the second has
    continuum (NEP > ND). Before the fix, `has_cont_part` returned
    False after inspecting only the first subsec; downstream
    `has_continuous_ddx` and the continuum branches of
    distribution1d / distribution2d then silently dropped the second
    subsec's continuum contribution."""
    d = _mf6_dict(
        _law1_subsec(0.0, [1], [1]),   # discrete-only
        _law1_subsec(0.0, [1], [5]),   # continuum
    )
    assert mf6_help.has_cont_part(d, 51, 0.0) is True


def test_has_cont_part_multi_subsec_first_continuum():
    """Order-invariance sanity check: first subsec with continuum,
    second discrete-only. Was already True before the fix; must
    still be True after."""
    d = _mf6_dict(
        _law1_subsec(0.0, [1], [5]),
        _law1_subsec(0.0, [1], [1]),
    )
    assert mf6_help.has_cont_part(d, 51, 0.0) is True


def test_has_cont_part_multi_subsec_all_discrete_only():
    """Order-invariance sanity check: all subsecs discrete-only,
    result must be False."""
    d = _mf6_dict(
        _law1_subsec(0.0, [1], [1]),
        _law1_subsec(0.0, [1], [1]),
    )
    assert mf6_help.has_cont_part(d, 51, 0.0) is False


def test_has_cont_part_missing_zap_returns_false():
    """S0 guard: missing ZAP in MF6 must not raise IndexError; must
    return False. Overlaps with tests in test_gamma_differential_guards
    but re-anchored here on a purely synthetic dict so it does not
    depend on the ad-hoc corpus."""
    d = _mf6_dict(_law1_subsec(1.0, [1], [5]))  # neutron only
    assert mf6_help.has_cont_part(d, 51, 0.0) is False


def test_has_disc_part_multi_subsec_first_continuum_only():
    """Symmetric coverage of has_disc_part: this predicate already
    correctly iterates through all subsecs and returns True as soon
    as any subsec has ND > 0. Anchor its multi-subsec behaviour
    here."""
    d = _mf6_dict(
        _law1_subsec(0.0, [0], [5]),   # continuum only (ND=0)
        _law1_subsec(0.0, [3], [3]),   # discrete-only (ND=3, NEP=3)
    )
    assert mf6_help.has_disc_part(d, 51, 0.0) is True


def test_has_disc_part_all_continuum_only():
    d = _mf6_dict(
        _law1_subsec(0.0, [0], [5]),
        _law1_subsec(0.0, [0], [3]),
    )
    assert mf6_help.has_disc_part(d, 51, 0.0) is False
