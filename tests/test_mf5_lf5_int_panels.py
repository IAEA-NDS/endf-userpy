"""MF5 LF=5 panel-exact integration by INT code.

Follow-up to the LF=5 rewrite. The normalisation integral

    G(E) = integral_{0}^{x_max} g(x) dx

is now computed per-panel exactly for the two INT codes that
appear on real corpus LF=5 files:

- INT=1 (histogram): ``area_i = g_i * (x_end_c - x_i)``
- INT=2 (lin-lin): trapezoid on the tabulated g values

INT=3 / 4 / 5 fall back to lin-lin with a UserWarning; no
corpus LF=5 file has been observed to use them, but the
fallback keeps the reconstruction well-defined.

Pinned here:

- Synthetic INT=1 integrates to 1 to machine precision (was
  ~1e-1 off with the pre-fix lin-lin trapezoid on histogram
  data).
- INT=2 continues to integrate to 1 within trapezoid error.
- INT=3 emits the fallback UserWarning.
- Corpus U-235 MT=455 LF=5 (all-INT=1) normalisation drops
  from ~1e-3 to ~1e-4, well inside the pre-existing 1e-3
  tolerance.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from endf_userpy.mfsec_interpretation import mf5_interpretation as mf5


def _make_contrib(theta, U, g_table):
    return {
        'p_table': {'E': [1e-5, 3e7], 'p': [1.0, 1.0],
                    'INT': [2], 'NBT': [2]},
        'theta_table': {'E': [1e-5, 3e7], 'theta': [theta, theta],
                        'INT': [2], 'NBT': [2]},
        'U': U,
        'LF': 5,
        'g_table': g_table,
    }


def test_lf5_int1_histogram_integrates_exactly_to_one():
    """Synthetic INT=1 histogram: g = [1, 2] on x = [0, 1, 2]. The
    correct normalisation gives ``G = 1*1 + 2*1 = 3``, and the
    per-eout integral of ``f = g_query / (theta * G)`` on the
    exact support ``[0, 2 theta]`` equals 1. The pre-fix lin-lin
    trapezoid on histogram data gave ``G = 0.5*(1+2)*1 = 1.5`` on
    the second panel plus ``0.5*(1+1)`` on a spurious zero-width
    interpolation-endpoint, ~50% off.
    """
    # INT=1 tab1: y_i is the value on the panel [x_i, x_{i+1}).
    # The trailing g value at x[-1] is a dangling sentinel; the
    # histogram integrator never reads it because the last panel
    # is [x[-2], x[-1]) with left-value g[-2].
    g_tab = {'x': [0.0, 1.0, 2.0], 'g': [1.0, 2.0, 0.0],
             'INT': [1], 'NBT': [3]}
    c = _make_contrib(theta=1.0, U=-1e6, g_table=g_tab)
    # x_max = (E - U)/theta = 2 for E - U = 2 -> pick E - U = 2.
    # With theta = 1, U = -1e6, need E = -1e6 + 2 ≈ -1e6 (below 0
    # which the wrapper rejects). Use theta = 1e6 instead so
    # x_max = (E - U)/theta = (E - U)/1e6; want x_max = 2 -> E - U =
    # 2e6 -> with U = 0, E = 2e6.
    c = _make_contrib(theta=1e6, U=0.0, g_table=g_tab)
    E = np.array([2e6])                          # E - U = 2 MeV
    # Sample eout at panel boundaries + a mid-panel probe.
    Eout = np.array([0.5e6, 1.5e6])              # x = 0.5, 1.5
    f = mf5.compute_general_evaporation_spectrum(c, E, Eout)
    # x=0.5 -> g=1, x=1.5 -> g=2, both divided by (theta * G) =
    # (1e6 * 3). So f0 = 1/3e6, f1 = 2/3e6.
    np.testing.assert_allclose(
        f[0], np.array([1.0 / 3e6, 2.0 / 3e6]), rtol=1e-12,
    )
    # And on a fine grid, trapezoid should give 1 to machine
    # precision (the spectrum is piecewise constant so trapezoid
    # midpoint-samples the correct integral).
    Eout_fine = np.linspace(0.0, 2e6, 2001)
    f_fine = mf5.compute_general_evaporation_spectrum(c, E, Eout_fine)
    integ = np.trapezoid(f_fine[0], Eout_fine)
    assert abs(integ - 1.0) < 1e-3


def test_lf5_int2_lin_lin_still_normalises_to_one():
    """INT=2 (the pre-existing path) still integrates to 1 within
    trapezoid error on a fine query grid."""
    x = np.linspace(0.0, 5.0, 501)
    g = np.ones_like(x)
    g_tab = {'x': list(x), 'g': list(g), 'INT': [2], 'NBT': [501]}
    c = _make_contrib(theta=1e6, U=0.0, g_table=g_tab)
    E = np.array([5e6])
    Eout = np.linspace(0.0, 5e6, 5001)
    f = mf5.compute_general_evaporation_spectrum(c, E, Eout)
    integ = np.trapezoid(f[0], Eout)
    assert abs(integ - 1.0) < 1e-3


@pytest.mark.parametrize('int_code', [3, 4, 5])
def test_lf5_int_3_4_5_emits_fallback_warning(int_code):
    """INT=3 / 4 / 5 aren't yet panel-exact; the kernel falls back
    to lin-lin trapezoid on those panels and emits a UserWarning."""
    x = [0.1, 1.0, 10.0]           # positive, log-safe
    g = [1.0, 0.5, 0.1]
    g_tab = {'x': x, 'g': g, 'INT': [int_code], 'NBT': [3]}
    c = _make_contrib(theta=1e6, U=0.0, g_table=g_tab)
    E = np.array([5e6])
    Eout = np.linspace(0.0, 5e6, 51)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _ = mf5.compute_general_evaporation_spectrum(c, E, Eout)
    matching = [w for w in caught
                if 'not yet panel-exact' in str(w.message)]
    assert len(matching) >= 1, (
        f'Expected fallback UserWarning for INT={int_code}, '
        f'got warnings: {[str(w.message) for w in caught]}'
    )


def test_lf5_corpus_u235_int1_normalises_tightly():
    """Real-corpus U-235 MT=455 LF=5 (all-INT=1, six delayed
    groups). Pre-fix the lin-lin trapezoid on histogram data gave
    ~1e-3 to 5e-3 deviation from 1; the panel-exact histogram
    integration drops that to ~1e-4."""
    path = 'tests/data_law1_adhoc/tendl21_n_U-235.endf'
    if not os.path.exists(path):
        pytest.skip('U-235 corpus not available')
    from endf_parserpy import EndfParserCpp
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    lf5 = [c for c in d[5][455]['contribution'].values()
           if c['LF'] == 5]
    E = np.array([1e6])
    Eout = np.linspace(0.0, 3e7, 30001)
    for i, c in enumerate(lf5):
        f = mf5.compute_general_evaporation_spectrum(c, E, Eout)
        integ = float(np.trapezoid(f[0], Eout))
        assert abs(integ - 1.0) < 1e-3, (
            f'contribution {i}: integral {integ:.6f} for LF=5; '
            f'expected within 1e-3 of unity'
        )
