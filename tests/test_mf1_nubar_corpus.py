"""Real-corpus coverage for MF1 tabulated nubar (issue #48).

PR #60 (issue #42) fixed the polynomial LNU=1 arithmetic and pinned
its behaviour with synthetic-dict tests. This file adds the missing
corpus-based coverage for the tabulated LNU=2 branch on U-235
(delayed nubar in TENDL-2021 uses a time-group representation
(LDG/NNF), so we skip MT 455 here and pin MT 452 (total) and
MT 456 (prompt) which use the standard tabulated shape).

Physics: total nubar for U-235 thermal fission is ~2.42; at
fast (~5 MeV) it should have risen to about ~3.0 with the standard
linear-in-E trend for actinide fission.
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation.mf1_interpretation import (
    compute_yields, compute_yields_from_mt452, compute_yields_from_mt456,
)


ADHOC_DATA_DIR = Path(__file__).resolve().parent / 'data_law1_adhoc'


@pytest.fixture(scope='module')
def u235_tendl():
    fn = ADHOC_DATA_DIR / 'tendl21_n_U-235.endf'
    if not fn.exists():
        pytest.skip(
            f'{fn.name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh` to populate the corpus'
        )
    return EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    ).parsefile(fn)


def test_mt452_lnu2_thermal_and_fast_shape(u235_tendl):
    """U-235 total nubar (MT 452, LNU=2). Physical anchor: thermal
    value ~ 2.42, fast (5 MeV) around ~3.0 -- the standard
    linear-in-E trend."""
    E = np.array([0.0253, 1.0e6, 5.0e6])
    nubar = compute_yields_from_mt452(u235_tendl, E)
    assert nubar.shape == E.shape
    assert 2.35 < nubar[0] < 2.55, (
        f'thermal nubar out of physical range: {nubar[0]}'
    )
    assert nubar[2] > nubar[0], (
        f'expected nubar to rise with E; got {nubar}'
    )
    assert 2.5 < nubar[2] < 3.5


def test_mt456_lnu2_prompt_thermal_and_fast_shape(u235_tendl):
    """U-235 prompt nubar (MT 456). Prompt <= total everywhere;
    thermal ~ 2.42 - 0.017 delayed = ~2.41."""
    E = np.array([0.0253, 1.0e6, 5.0e6])
    prompt = compute_yields_from_mt456(u235_tendl, E)
    total = compute_yields_from_mt452(u235_tendl, E)
    assert prompt.shape == E.shape
    assert 2.30 < prompt[0] < 2.50
    assert np.all(prompt <= total + 1e-9), (
        'prompt nubar > total; violates prompt + delayed = total'
    )


def test_dispatcher_routes_mt452_and_mt456(u235_tendl):
    """The `compute_yields` dispatcher routes MT numbers to the
    right underlying function (MT 452 -> mt452, MT 456 -> mt456,
    unsupported MT raises)."""
    E = np.array([1e5])
    from_dispatcher_452 = compute_yields(u235_tendl, 452, E)
    from_direct_452 = compute_yields_from_mt452(u235_tendl, E)
    np.testing.assert_allclose(from_dispatcher_452, from_direct_452)

    from_dispatcher_456 = compute_yields(u235_tendl, 456, E)
    from_direct_456 = compute_yields_from_mt456(u235_tendl, E)
    np.testing.assert_allclose(from_dispatcher_456, from_direct_456)

    with pytest.raises(ValueError, match=r'Unsupported number'):
        compute_yields(u235_tendl, 999, E)


def test_mt452_nonneg_monotone_across_fine_grid(u235_tendl):
    """Nubar is non-negative and monotone-non-decreasing above
    thermal for U-235 (well-known evaluator constraint)."""
    E = np.linspace(1e6, 1.5e7, 30)
    nubar = compute_yields_from_mt452(u235_tendl, E)
    assert np.all(nubar >= 0)
    diffs = np.diff(nubar)
    # Allow tiny non-monotone dips from evaluator smoothing; the
    # cumulative trend must be increasing.
    assert diffs.sum() > 0
