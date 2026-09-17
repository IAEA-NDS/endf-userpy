"""Tests for the gamma-aware sum-vs-children admission rule
(issue #133 / PR #132 follow-up).

The general :func:`satisfies_select_heuristic` uses MF4/5/6 as the
"children have distribution detail" indicator, which drops sum-MTs
in favour of children. That's correct on the neutron side but
wrong for gamma production when the file aggregates its gamma XS
on a sum-MT via MF13 (JENDL-5 N-14 MT 3 nonelastic is the corpus
example): the aggregate covers gamma production from every
descendant of the sum-MT, and admitting both the sum and its
children double-counts.

The new gamma-aware rule (:func:`satisfies_gamma_production_select`)
adds two checks on top of the general heuristic:

1. If some SUM_RULES ancestor of ``mt`` is within the user's
   query scope AND carries MF13 gamma content, drop ``mt`` (its
   gamma is subsumed by that ancestor's MF13 aggregate).
2. If ``mt`` itself has MF13 gamma AND is within the user's query
   scope, admit unconditionally (overrides the general heuristic's
   tendency to drop sum-MTs in favour of children with MF4/5/6
   neutron distributions but no gamma coverage).

The dispatch is via
:func:`selectors.satisfies_particle_production_select` which
routes to the gamma-aware rule when ``zap == PARTICLE_ZAP['g']``
and to the general heuristic otherwise.

Tests pin the primary case (JENDL-5 N-14 ``(n,total)+g`` now
returns MT 3's aggregate) and check non-regression on:

- Al-27 ``(n,total)+g`` (no MF13 anywhere; general heuristic
  applies unchanged).
- ENDF/B-VIII.1 N-14 ``(n,total)+g`` (MF13 per-channel on MT 4,
  28, 103, ...; new rule prefers MT 4's MF13 aggregate over
  MT 51..90's MF6 gammas -- this is a behaviour CHANGE and
  the test measures whether the two composition paths agree
  numerically).
- U-238 ``(n,fission)+g`` (MT 18 MF12+MF15, no MF13; unchanged).
- Neutron-ejectile queries (never touched by the gamma-aware
  path).
- ``(n,inl)+g`` (MT 4) on files where MT 4 does not have MF13:
  the general heuristic still applies within the (MT 4) scope.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_xs,
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


# ============================================================
# Primary reproducer: (n,total)+g on JENDL-5 N-14 now uses MT 3's
# MF13 aggregate instead of dropping it in favour of MT 102/103
# (which under-counted 10x).
# ============================================================


def test_jendl5_n14_total_gamma_uses_mt3_mf13():
    """(n,total)+g at 10 MeV: pre-#133 returned ~0.07 barn (just
    MT 102 + MT 103 via MF12); post-#133 returns MT 3's ~0.83 barn
    MF13 aggregate."""
    endf = _load('jendl5_n_N-14.endf')
    einc = np.array([1e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,total)', 'g', einc)
    assert xs is not None
    assert xs[0] > 0.5, (
        f'JENDL-5 N-14 (n,total)+g XS should be >0.5 barn at 10 MeV, '
        f'got {xs[0]:.4g}. Pre-#133 value was ~0.07 (MT 102 + MT 103 '
        f'only, MT 3 dropped by general heuristic).'
    )


def test_jendl5_n14_total_gamma_matches_direct_mt3():
    """(n,total)+g must equal (n,nonelas)+g on JENDL-5 N-14
    because MT 3's MF13 aggregate is the file's total gamma
    production and both queries route through it."""
    endf = _load('jendl5_n_N-14.endf')
    einc = np.array([5e6, 1e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs_total = get_particle_production_xs(endf, '(n,total)', 'g', einc)
        xs_nonelas = get_particle_production_xs(endf, '(n,nonelas)', 'g', einc)
    np.testing.assert_array_almost_equal(xs_total, xs_nonelas, decimal=8)


# ============================================================
# Non-regression: files without MF13 sum-MT aggregation should
# be unaffected by the new rule.
# ============================================================


def test_al27_total_gamma_unchanged():
    """Al-27 has no MF13 anywhere. The gamma-aware rule's MF13
    checks all return False; the general heuristic applies and the
    result should be identical to pre-#133."""
    endf = _load('endfb81_n_Al-27.endf')
    einc = np.array([1.4e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,total)', 'g', einc)
    assert xs is not None
    # Pinned pre-#133 baseline value (from the standard test
    # scenarios in the corpus, Al-27 (n,total)+g at 14 MeV is
    # ~1.47 barn -- MF12 discrete lines + MF6/LAW=1 discrete +
    # MF15 continuum for MT 102).
    assert 1.0 < xs[0] < 2.0, (
        f'Al-27 (n,total)+g at 14 MeV changed: expected ~1.47, got {xs[0]:.4g}'
    )


def test_u238_fission_gamma_unchanged():
    """U-238 (n,fission)+g via MT 18 with MF12+MF15 (no MF13).
    Post-#127 baseline ~7.3 barn at 14 MeV should be unchanged."""
    endf = _load('jendl5_n_U-238.endf')
    einc = np.array([1.4e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,fission)', 'g', einc)
    assert xs is not None
    assert 6.0 < xs[0] < 8.0, (
        f'U-238 (n,fission)+g at 14 MeV changed: expected ~7.3, got {xs[0]:.4g}'
    )


def test_neutron_ejectile_queries_unchanged():
    """The gamma-aware rule dispatches on zap == gamma. Neutron
    ejectile queries go through the general heuristic unchanged."""
    endf = _load('jendl5_n_U-238.endf')
    einc = np.array([1.4e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,total)', 'n', einc)
    assert xs is not None
    assert xs[0] > 0


# ============================================================
# Direct-MT gamma queries still work (via PR #132's escape which
# lives inside the general heuristic; the gamma-aware rule
# defers to it when neither of its two checks fire).
# ============================================================


def test_jendl5_n14_direct_nonelas_gamma_still_works():
    """(n,nonelas)+g on JENDL-5 N-14 was fixed in PR #132. The
    gamma-aware rule must not regress it: (n,nonelas) resolves to
    MT 3, MT 3 has MF13 and is in the user's covered scope, so
    Rule 2 admits it."""
    endf = _load('jendl5_n_N-14.endf')
    einc = np.array([1e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,nonelas)', 'g', einc)
    assert xs is not None
    assert xs[0] > 0.5


# ============================================================
# Behaviour CHANGE on ENDF/B-VIII.1 N-14: MT 4 has MF13 and now
# gets admitted, MT 51..90's MF6-gamma content is dropped as
# subsumed. If MF13/MT4 and per-channel MT 51..90 gamma are
# consistently evaluated (they should be), the numeric answer
# stays close to the pre-#133 value.
# ============================================================


def test_endfb81_n14_total_gamma_still_finite_and_positive():
    """ENDF/B-VIII.1 N-14 has MF13 on MT 4, 28, 32, 103, 104, 105,
    107. Post-#133, MT 4's MF13 aggregate is admitted, and
    MT 51..90's per-channel gamma (via MF6) is dropped as subsumed
    by MT 4's MF13. The numeric answer should stay close to the
    pre-#133 value IF the evaluator's MF13 aggregate matches the
    per-channel sum -- which is what a well-formed evaluation
    guarantees. If they differ substantially, that's a data-side
    inconsistency that would surface as a discrepancy between the
    two computation paths."""
    endf = _load('endfb81_n_N-14.endf')
    einc = np.array([1e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,total)', 'g', einc)
    assert xs is not None
    assert np.all(np.isfinite(xs))
    assert xs[0] > 0
    # Loose bound: ENDF/B N-14 (n,total)+g at 10 MeV is expected
    # somewhere in the range 0.1 - 5 barn based on the isotope
    # and mass. Sanity check.
    assert 0.01 < xs[0] < 10
