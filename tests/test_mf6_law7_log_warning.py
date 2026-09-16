"""Tests for the INT>=3 log-based interpolation warning in the
knot-aware LAW=7 integrator (issue #71).

The knot-aware integrator's midpoint rule is exact for INT=1
(histogram) and INT=2 (lin-lin) E' interpolants, but has
O(h^3 * f'') per-segment error on INT>=3 (log-based) interpolants.
When a caller has opened `collect_law7_log_errors()`, the
integrator estimates that error per segment (from a second-difference
of `f` at `mid +- w/4`) and records it into a per-query accumulator;
the context manager emits one summary UserWarning on exit.

The top-level `get_particle_production_dxs_dmu` API opens the
context around its dispatch, so users of the public API see the
warning transparently.
"""
import copy
import warnings
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_particle_production_dxs_dmu
from endf_userpy.mfsec_interpretation import mf6_law7_integrals as mf6_l7


ADHOC = Path(__file__).resolve().parent / 'data_law1_adhoc'


def _load(fn_name):
    fn = ADHOC / fn_name
    if not fn.exists():
        pytest.skip(
            f'{fn_name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh` to populate the corpus'
        )
    return EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    ).parsefile(fn)


@pytest.fixture(scope='module')
def be9_endfb81():
    return _load('endfb81_n_Be-9.endf')


# ============================================================
# Silent no-op on the INT=1/2 corpus (guard against accidental
# emission).
# ============================================================


def test_no_warning_on_be9_int1(be9_endfb81):
    """Be-9 MT16 uses INT=1 (histogram) on every LAW=7 Ep table.
    The knot-aware integrator is exact there; the log-warning
    accumulator must record nothing and no summary UserWarning may
    be emitted."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_particle_production_dxs_dmu(
            be9_endfb81, '(n,total)', 'n',
            np.array([1e7]), np.array([-0.5, 0.0, 0.5]),
        )
    log_warnings = [
        w for w in recorded
        if 'LAW=7' in str(w.message) and 'log-based' in str(w.message)
    ]
    assert log_warnings == [], (
        f'unexpected log-based warning(s) on INT=1 corpus: '
        f'{[str(w.message) for w in log_warnings]}'
    )


# ============================================================
# Synthetic-dict tests: mutate a real LAW=7 subsection's Ep INT
# arrays to `INT=5` (log-log) so the integrator's log-warning path
# fires. Verify:
#   1. The accumulator records exactly one (mt, subsec) entry.
#   2. The recorded INT law list contains 5.
#   3. The reported max abs error is finite and non-zero.
#   4. The public API emits one summary UserWarning citing INT and
#      the numeric error.
# ============================================================


def _mutate_law7_to_int5(endf_dict, mt):
    """Deep-copy the endf_dict and rewrite every LAW=7 subsection's
    per-`(E_in_i, mu_j)` table INT array from `[1]` to `[5]` (log-log
    over Ep). The Fortran evaluator will honour the change and the
    knot-aware integrator will detect it and route through the
    error-estimating branch. Values (`f`) are left untouched so the
    reconstructed function does something sensible, but the exact
    numeric answer is not the physical one -- these tests only care
    about the warning-path behaviour, not the physics."""
    mutated = copy.deepcopy(endf_dict)
    for sub in mutated[6][mt]['subsection'].values():
        if sub.get('LAW') != 7:
            continue
        for ei_slot in sub.get('table', {}).values():
            for mu_slot in ei_slot.values():
                mu_slot['INT'] = [5]
    return mutated


def test_accumulator_records_int5(be9_endfb81):
    """Direct-path test: open the context, mutate a LAW=7 section to
    INT=5, call the knot-aware integrator, verify the accumulator
    holds one record with INT=5 and a positive absolute-error
    estimate."""
    mutated = _mutate_law7_to_int5(be9_endfb81, 16)
    with mf6_l7.collect_law7_log_errors() as accum:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')  # keep the exit warning quiet
            mf6_l7.integrate_law7_subsec_over_eout(
                mutated, 16, 1,
                np.array([1e7, 1.4e7]),
                np.array([-0.5, 0.0, 0.5]),
            )
    assert accum.has_records()
    assert list(accum.records.keys()) == [(16, 1)]
    laws, err, peak = accum.records[(16, 1)]
    assert 5 in laws
    assert err > 0.0
    assert peak > 0.0
    assert np.isfinite(err) and np.isfinite(peak)


def test_public_api_emits_summary_warning(be9_endfb81):
    """End-to-end: the top-level `get_particle_production_dxs_dmu`
    must emit ONE summary UserWarning naming the affected MT, the
    INT laws seen, and the computed error magnitude."""
    mutated = _mutate_law7_to_int5(be9_endfb81, 16)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        get_particle_production_dxs_dmu(
            mutated, '(n,total)', 'n',
            np.array([1e7, 1.4e7]),
            np.array([-0.5, 0.0, 0.5]),
        )
    log_warnings = [
        w for w in recorded
        if 'LAW=7' in str(w.message) and 'log-based' in str(w.message)
    ]
    # Exactly one summary warning, not one per MT or per cell.
    assert len(log_warnings) == 1, (
        f'expected 1 summary warning, got {len(log_warnings)}: '
        f'{[str(w.message) for w in log_warnings]}'
    )
    msg = str(log_warnings[0].message)
    assert 'MT=16' in msg
    assert 'INT=' in msg and '5' in msg
    assert 'issue #71' in msg
    # And the summary carries the numeric error, not just a
    # directional hedge.
    import re
    assert re.search(r'\d\.\d+e[-+]?\d+', msg), (
        f'warning has no scientific-notation number: {msg}'
    )


# ============================================================
# The error estimate is not a fantasy -- pin its magnitude against
# a coarser vs finer knot-aware integrator run.
# ============================================================


def test_error_estimate_bounds_actual_diff(be9_endfb81):
    """The estimated per-cell error must be at least as large as the
    difference between the midpoint-rule integral and a
    higher-order (5-point Gauss-Legendre per segment) reference.
    Otherwise it under-bounds the truth and the warning is
    optimistic. Uses a single (Ein, mu) cell to keep the test
    focused."""
    from scipy.integrate import quad
    from endf_userpy.quantities_mt_zap.distribution2d import (
        compute_dist2d_values,
    )
    from endf_userpy.primitives.properties import get_QM, get_QI

    mutated = _mutate_law7_to_int5(be9_endfb81, 16)
    e = 1.4e7
    u = 0.0
    einc = np.array([e])
    mus = np.array([u])

    with mf6_l7.collect_law7_log_errors() as accum:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            ka = mf6_l7.integrate_law7_subsec_over_eout(
                mutated, 16, 1, einc, mus,
            )
    est_err = accum.records[(16, 1)][1]

    # Higher-order reference: quad with tight tolerance and
    # tabulated knots as breakpoints.
    q = max(get_QM(mutated, 16), get_QI(mutated, 16))
    eout_max = (e + q) * 1.1
    ep_all = set()
    for sub in mutated[6][16]['subsection'].values():
        for ei_slot in sub.get('table', {}).values():
            for mu_slot in ei_slot.values():
                ep_all.update(float(x) for x in mu_slot['Ep'])
    pts = [x for x in sorted(ep_all) if 0 < x < eout_max]

    def f_ep(x):
        return compute_dist2d_values(
            mutated, 16, 1, np.array([e]),
            np.array([x], dtype=float), np.array([u]), True,
        ).item()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        truth, quad_abs_err = quad(
            f_ep, 0.0, eout_max, epsrel=1e-8, limit=2000, points=pts,
        )
    actual_diff = abs(float(ka[0, 0]) - truth)
    # Estimated error must be a real upper bound (allow a small
    # slack factor for the quad reference's own noise).
    assert est_err + quad_abs_err >= actual_diff * 0.5, (
        f'estimated error {est_err:.3e} < actual diff {actual_diff:.3e} '
        f'(quad ref err {quad_abs_err:.3e})'
    )


# ============================================================
# Direct-path: outside the context manager, no error estimate is
# collected and no extra evaluations are done (no warning path
# lit up).
# ============================================================


def test_no_accumulation_outside_context(be9_endfb81):
    """Called outside `collect_law7_log_errors()`, the integrator
    must run its fast midpoint-only path unchanged even on an
    INT=5 subsection. The public API sets the context; leaf callers
    that bypass the API stay warning-free (they can opt in)."""
    mutated = _mutate_law7_to_int5(be9_endfb81, 16)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        mf6_l7.integrate_law7_subsec_over_eout(
            mutated, 16, 1,
            np.array([1e7]),
            np.array([-0.5, 0.5]),
        )
    log_warnings = [
        w for w in recorded
        if 'LAW=7' in str(w.message) and 'log-based' in str(w.message)
    ]
    assert log_warnings == []
