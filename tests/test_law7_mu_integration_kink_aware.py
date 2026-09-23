"""Kink-aware LAW=7 mu-integration parity (issue #169 tier-2).

`integrate_law7_subsec_over_mu` in `mf6_law7_integrals` mirrors the
existing knot-aware E' integrator: for each `E_in`, segment `mu` at
the union of the two bracketing Ein slices' tabulated mu knots (the
genuine kinks in the LAW=7 reconstruction) and apply per-segment
composite Simpson with a two-level Richardson-style refinement on
unconverged segments.

Compared to the general-purpose adaptive Simpson path in
`distribution1d_helpers._adaptive_simpson_along_axis` (initial_n=81,
max_iter=3), the kink-aware integrator is measurably more accurate
on the corpus's single-subsec LAW=7 case (JEFF-4.0 H-2 (n,2n) MT16)
because the segment edges land on the primary kinks that a uniform
mesh spans.

Dispatch: `distribution1d_helpers.integrate_mf6_dist2d_over_mu`
routes single-subsec LAW=7 to the new integrator; other LAWs and
multi-subsec cases keep the existing paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf6_law7_integrals as mf6_law7
from endf_userpy.quantities_mt_zap import distribution1d_helpers as d1dh
from endf_userpy.quantities_mt_zap.distribution2d import compute_dist2d_values

from _corpus import resolve_h2


@pytest.fixture(scope='module')
def h2_endf_dict():
    path = resolve_h2()
    if path is None:
        pytest.skip('JEFF-4.0 H-2 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def _high_res_reference(endf_dict, mt, zap, einc, eouts):
    """Adaptive Simpson at very high resolution (n=641 -> 1281 pts,
    max 1 doubling). Empirically converges to within ~1e-4 of a
    2001-point run on the H-2 LAW=7 case, well below the accuracy
    both integrators achieve; adequate as a fair reference."""
    return d1dh._adaptive_simpson_along_axis(
        lambda mu: compute_dist2d_values(
            endf_dict, mt, zap, einc, eouts, mu, True,
        ),
        -1.0, 1.0, rtol=1e-8, initial_n=641, max_iter=1,
    )


def _max_rel(a, b):
    denom = 0.5 * (np.abs(a) + np.abs(b)) + 1e-30
    return float(np.max(np.abs(a - b) / denom))


def test_h2_mt16_single_subsec_law7_more_accurate_than_adaptive_simpson(
    h2_endf_dict,
):
    """Kink-aware LAW=7 mu-integrator beats the general adaptive-
    Simpson fallback on the corpus's canonical single-subsec LAW=7
    case (JEFF-4.0 H-2 (n,2n) MT16)."""
    d = h2_endf_dict
    sub = d[6][16]['subsection'][1]
    assert sub['LAW'] == 7 and len(d[6][16]['subsection']) == 1

    einc = np.array([1.4e7])
    eouts = np.linspace(1e4, 5e6, 25)
    zap = sub['ZAP']

    res_kink = mf6_law7.integrate_law7_subsec_over_mu(
        d, 16, 1, einc, eouts,
    )
    res_simp = d1dh._integrate_mf6_dist2d_over_mu_default(
        d, 16, zap, einc, eouts,
    )
    ref = _high_res_reference(d, 16, zap, einc, eouts)

    err_kink = _max_rel(res_kink, ref)
    err_simp = _max_rel(res_simp, ref)

    # Both should be reasonable; kink-aware should be strictly better.
    assert err_kink < 5e-4, f'kink integrator error {err_kink:.2e} too large'
    assert err_kink < err_simp, (
        f'kink-aware ({err_kink:.2e}) not better than adaptive '
        f'Simpson ({err_simp:.2e}) on H-2 MT16'
    )


def test_h2_mt16_dispatch_routes_to_kink_aware(h2_endf_dict):
    """`integrate_mf6_dist2d_over_mu` routes single-subsec LAW=7 to
    the new integrator (identity check by calling both explicitly
    and confirming they match)."""
    d = h2_endf_dict
    sub = d[6][16]['subsection'][1]
    zap = sub['ZAP']

    einc = np.array([1.4e7])
    eouts = np.linspace(1e4, 5e6, 15)

    via_dispatcher = d1dh.integrate_mf6_dist2d_over_mu(
        d, 16, zap, einc, eouts,
    )
    via_direct = mf6_law7.integrate_law7_subsec_over_mu(
        d, 16, 1, einc, eouts,
    )
    np.testing.assert_array_equal(np.asarray(via_dispatcher), via_direct)


def test_output_shape_and_nonnegativity(h2_endf_dict):
    """LAW=7 pdf is non-negative; the integrated dxs/dE (up to the
    xs*yield factor that lives outside this integrator) must be
    non-negative too. Also pins the (n_einc, n_eout) return shape."""
    d = h2_endf_dict
    einc = np.array([5e6, 1e7, 2e7])
    eouts = np.linspace(1e4, 3e6, 12)
    res = mf6_law7.integrate_law7_subsec_over_mu(d, 16, 1, einc, eouts)
    assert res.shape == (3, 12)
    # Small negative FFT/round-off noise is acceptable at ~1e-30 scale
    # but nothing near the peak. The pdf integral over mu on a valid
    # (Ein, Eout) point should be positive; on out-of-kinematic-range
    # points it should be zero (see below).
    assert np.min(res) >= -1e-14


def test_outside_einc_range_returns_zero(h2_endf_dict):
    """Ein below the section's mesh (2 MeV threshold for H-2 (n,2n))
    or above its upper limit produces a zero row."""
    d = h2_endf_dict
    sub = d[6][16]['subsection'][1]
    ei_mesh = list(sub['E'].values())
    ein_below = np.array([ei_mesh[0] * 0.99])
    ein_above = np.array([ei_mesh[-1] * 1.01])
    eouts = np.linspace(1e4, 3e6, 5)

    res_below = mf6_law7.integrate_law7_subsec_over_mu(
        d, 16, 1, ein_below, eouts,
    )
    res_above = mf6_law7.integrate_law7_subsec_over_mu(
        d, 16, 1, ein_above, eouts,
    )
    np.testing.assert_array_equal(res_below, 0.0)
    np.testing.assert_array_equal(res_above, 0.0)


def test_wrong_law_raises():
    """Passing a non-LAW=7 subsection is a programmer error, not a
    silent fallthrough. Uses a minimal synthetic dict to avoid any
    corpus assumption."""
    synth = {6: {99: {'subsection': {1: {'LAW': 2}}}}}
    with pytest.raises(ValueError, match='not LAW=7'):
        mf6_law7.integrate_law7_subsec_over_mu(
            synth, 99, 1, np.array([1e7]), np.array([1e6]),
        )
