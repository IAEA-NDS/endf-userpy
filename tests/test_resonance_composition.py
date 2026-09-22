"""Tests for the MF2 + MF3 composition layer
(:mod:`endf_userpy.quantities_mt_zap.resonance_composition`) and its
plumbing into the user-facing :func:`endf_userpy.quantities.get_reaction_xs`
via ``include_resonance=True``.

Coverage:

1. Synthetic dict without MF2: composition == raw MF3.
2. MT that has no MF2 contribution (e.g. MT=16): composition == raw MF3.
3. Nb-93 (MLBW): composition MT=2 equals the direct MLBW.reconstruct
   ``sct`` partial + the MF3 elastic background (matches whichever
   the file happens to store).
4. Nb-93: outside the RRR, composition equals raw MF3.
5. U-235 (Reich-Moore): composition sums for MT=1 equals the R-M
   ``tot`` + MF3(mt=1); numpy vs numba backend equivalence.
6. Top-level ``get_reaction_xs(..., include_resonance=True)`` returns
   the composed cross section, and equals the same call composed
   through :func:`compute_reconstructed_cross_section` directly for
   MT=2 elastic on Nb-93.

Skip cleanly when the corpus files are absent so a fresh checkout
still passes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_userpy import quantities as user_api
from endf_userpy.primitives import array_ns
from endf_userpy.quantities_mt_zap import (
    resonance_composition as res_comp,
)
from endf_userpy.mfsec_interpretation import (
    mf2_interpretation_mlbw as mlbw,
    mf2_interpretation_mlbw_preproc as mlbw_pre,
    mf2_interpretation_reichmoore as rm,
    mf2_interpretation_reichmoore_preproc as rm_pre,
)

from _corpus import resolve_nb93, resolve_u235, resolve_nd143


def _numba_available():
    return 'numba' in array_ns.available_backends()


# ============================================================
# 1. Synthetic dict: no MF2 -> composition == raw MF3.
# ============================================================


def _tab1_dict(x, y):
    return {
        'NBT': [len(x)], 'INT': [2],   # lin-lin
        'E': list(map(float, x)),
        'xs': list(map(float, y)),
    }


def _synthetic_mf3_only_dict():
    """A minimal endf_dict with MF3/MT=102 only (no MF2)."""
    e = np.array([1e-5, 1.0, 1e2, 1e5, 2e7])
    y = np.array([100.0, 1.0, 0.1, 0.01, 0.001])
    return {
        3: {102: {'xstable': _tab1_dict(e, y)}},
    }


def test_reconstruct_resonance_xs_zero_without_mf2():
    d = _synthetic_mf3_only_dict()
    e_query = np.array([1e-3, 1.0, 1e4])
    xp = array_ns.get_backend('numpy')
    out = res_comp.reconstruct_resonance_xs(d, 102, e_query, xp)
    assert np.array_equal(np.asarray(out), np.zeros(3))


def test_compute_reconstructed_equals_mf3_without_mf2():
    d = _synthetic_mf3_only_dict()
    e_query = np.array([1e-3, 1.0, 1e4])
    xp = array_ns.get_backend('numpy')
    composed = res_comp.compute_reconstructed_cross_section(
        d, 102, e_query, xp,
    )
    mf3_only = np.interp(e_query, [1e-5, 1.0, 1e2, 1e5, 2e7],
                         [100.0, 1.0, 0.1, 0.01, 0.001])
    np.testing.assert_allclose(np.asarray(composed), mf3_only, rtol=1e-10)


# ------------------------------------------------------------
# 1b. URR handling: LSSF=0 must warn; LSSF=1 must be silent.
# ------------------------------------------------------------


def _dict_with_urr(lssf):
    """Minimal dict with MF3 + one LRU=2 URR range at ``lssf``.
    The URR range is a placeholder shell (no valid width tables);
    the URR preproc will fail on it. Sufficient to exercise the
    "LSSF=0 URR that the reconstruction cannot handle" path and
    the LSSF=1 silent path."""
    d = _synthetic_mf3_only_dict()
    d[2] = {
        151: {
            'isotope': {
                1: {
                    'ABN': 1.0,
                    'range': {
                        1: {
                            'LRU': 2, 'LRF': 2, 'LSSF': int(lssf),
                            'EL': 1e3, 'EH': 1e5,
                            'NAPS': 0, 'NRO': 0,
                            'SPI': 0.0, 'AP': 0.8, 'NLS': 0,
                        }
                    },
                }
            }
        }
    }
    return d


def _dict_with_valid_lssf0_urr(mt=102):
    """Minimal dict with MF3(mt) + one valid LSSF=0 LRU=2 LRF=2
    URR range and just enough MF1 for the incident-particle
    lookup. The URR range has one spin group with constant widths
    at two energy knots, so the URR kernel actually produces a
    positive contribution to be composed with MF3."""
    e = np.array([1e-5, 500.0, 1e3, 5e3, 1e4, 5e4, 1e5, 2e7])
    y = np.array([100.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.001])
    d = {
        1: {451: {'NSUB': 10}},        # neutron incident
        3: {mt: {'xstable': _tab1_dict(e, y)}},
        2: {151: {'isotope': {1: {
            'ABN': 1.0,
            'range': {
                1: {
                    'LRU': 2, 'LRF': 2, 'LSSF': 0,
                    'EL': 1e3, 'EH': 1e5,
                    'NAPS': 0, 'NRO': 0,
                    'SPI': 0.0, 'AP': 0.6,
                    'NLS': 1,
                    'l_group': {1: {
                        'L': 0, 'AWRI': 232.0, 'NJS': 1,
                        'j_group': {1: {
                            'AJ': 0.5,
                            'AMUN': 1.0, 'AMUG': 0.0,
                            'AMUF': 0.0, 'AMUX': 0.0,
                            'NE': 2, 'INT': 2,
                            'ES': {1: 1e3, 2: 1e5},
                            'D':  {1: 20.0, 2: 20.0},
                            'GN0': {1: 5e-4, 2: 5e-4},
                            'GG':  {1: 0.03, 2: 0.03},
                            'GF':  {1: 0.0, 2: 0.0},
                            'GX':  {1: 0.0, 2: 0.0},
                        }},
                    }},
                }
            },
        }}}},
    }
    return d


def test_lssf1_urr_is_silent():
    """LSSF=1 URR: MF3 IS the physical XS in the URR, no
    reconstruction expected. Our LRU=1 filter drops URR entirely,
    which is silently correct for LSSF=1 -- no warning must
    fire."""
    d = _dict_with_urr(lssf=1)
    e_query = np.array([1e-3, 1.0, 1e4])
    xp = array_ns.get_backend('numpy')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = res_comp.reconstruct_resonance_xs(d, 102, e_query, xp)
    assert np.array_equal(np.asarray(out), np.zeros(3))
    lssf_warnings = [w for w in caught if 'LSSF' in str(w.message)]
    assert not lssf_warnings, (
        f'LSSF=1 URR must not warn; got: {[str(w.message) for w in lssf_warnings]}'
    )


def test_lssf0_urr_warns_once_naming_the_range():
    """LSSF=0 URR: MF3 is a background to a URR reconstruction
    we don't implement. Emit exactly one UserWarning per call
    naming the iso/rng."""
    d = _dict_with_urr(lssf=0)
    e_query = np.array([1e-3, 1.0, 1e4])
    xp = array_ns.get_backend('numpy')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res_comp.reconstruct_resonance_xs(d, 102, e_query, xp)
    lssf_warnings = [w for w in caught if 'LSSF' in str(w.message)]
    assert len(lssf_warnings) == 1, (
        f'expected exactly 1 LSSF=0 URR warning, got {len(lssf_warnings)}: '
        f'{[str(w.message) for w in lssf_warnings]}'
    )
    msg = str(lssf_warnings[0].message)
    assert 'LSSF=0' in msg
    assert 'iso=1' in msg and 'rng=1' in msg


def test_no_urr_stays_silent():
    """A file with only LRU=1 (no URR) must not fire any
    LSSF warning path."""
    d = _synthetic_mf3_only_dict()
    e_query = np.array([1e-3, 1.0])
    xp = array_ns.get_backend('numpy')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res_comp.reconstruct_resonance_xs(d, 102, e_query, xp)
    lssf_warnings = [w for w in caught if 'LSSF' in str(w.message)]
    assert not lssf_warnings


def test_valid_lssf0_urr_composes_silently_and_adds_contribution():
    """LSSF=0 URR with a preproc-able, kernel-able range: no
    warning, and the URR reconstruction actually contributes to
    the composed cross section (result > MF3 alone in the URR
    energy window)."""
    d = _dict_with_valid_lssf0_urr(mt=102)
    xp = array_ns.get_backend('numpy')

    # Query points: two inside the URR [1e3, 1e5], two outside.
    e_query = np.array([500.0, 2e3, 3e4, 2e5])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res_only = np.asarray(res_comp.reconstruct_resonance_xs(
            d, 102, e_query, xp,
        ))
    lssf_warnings = [w for w in caught if 'LSSF' in str(w.message)]
    assert not lssf_warnings, (
        f'valid LSSF=0 URR must not fire the "unhandled" warning; '
        f'got: {[str(w.message) for w in lssf_warnings]}'
    )

    # Outside URR: zero contribution.
    assert res_only[0] == 0.0, 'below URR range: contribution must be 0'
    assert res_only[3] == 0.0, 'above URR range: contribution must be 0'
    # Inside URR: positive contribution.
    assert res_only[1] > 0.0, 'inside URR: expected a positive contribution'
    assert res_only[2] > 0.0, 'inside URR: expected a positive contribution'

    # Composition: MF3(mt=102) + URR contribution.
    composed = np.asarray(res_comp.compute_reconstructed_cross_section(
        d, 102, e_query, xp,
    ))
    from endf_userpy.mfsec_interpretation import mf3_interpretation
    mf3_xs = np.asarray(mf3_interpretation.compute_cross_section_agnostic(
        d, 102, e_query, xp,
    ))
    np.testing.assert_allclose(composed, mf3_xs + res_only, rtol=1e-12)


def test_lssf0_urr_with_unsupported_lrf_warns_with_lrf_reason():
    """LSSF=0 URR with LRF=1 (Case A, not yet supported by the
    kernel): warning fires and names the LRF explicitly so the
    caller can tell it's a format issue, not a bug."""
    d = _dict_with_urr(lssf=0)
    d[2][151]['isotope'][1]['range'][1]['LRF'] = 1
    e_query = np.array([5e3])
    xp = array_ns.get_backend('numpy')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res_comp.reconstruct_resonance_xs(d, 102, e_query, xp)
    lssf_warnings = [w for w in caught if 'LSSF' in str(w.message)]
    assert len(lssf_warnings) == 1
    msg = str(lssf_warnings[0].message)
    assert 'LRF=1' in msg, (
        f'warning should name the unsupported LRF; got: {msg}'
    )


# ============================================================
# 2. Nb-93 (MLBW). Real file, so we skip when it is not
#    on disk.
# ============================================================


@pytest.fixture
def nb93_dict():
    path = resolve_nb93()
    if path is None:
        pytest.skip(
            'Nb-93 ENDF file not available (set NB93_ENDF, run '
            'tests/data_law1_adhoc/fetch.sh, or place the file at '
            'tests/data_law1_adhoc/endfb81_n_Nb-93.endf)'
        )
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp().parsefile(path)


def test_nb93_composition_mt2_matches_manual_sum(nb93_dict):
    """For MT=2 elastic in Nb-93 (MLBW), composition should equal
    the direct MLBW ``sct`` partial + MF3(2) tabulated on the query
    grid. If MF3(2) in the RRR is exactly zero (common for
    non-actinides), composition equals the MLBW ``sct`` alone."""
    xp = array_ns.get_backend('numpy')
    data = mlbw_pre.mlbw_data_from_endf_dict(nb93_dict)
    e_query = np.geomspace(1.0, 5e3, 501)   # inside RRR
    recon = mlbw.reconstruct(data, e_query, xp)
    from endf_userpy.mfsec_interpretation import mf3_interpretation
    mf3_xs = mf3_interpretation.compute_cross_section_agnostic(
        nb93_dict, 2, e_query, xp,
    )
    expected = np.asarray(recon['sct']) + np.asarray(mf3_xs)
    composed = res_comp.compute_reconstructed_cross_section(
        nb93_dict, 2, e_query, xp,
    )
    np.testing.assert_allclose(
        np.asarray(composed), expected, rtol=1e-12, atol=0.0,
    )


def test_nb93_composition_outside_rrr_equals_mf3(nb93_dict):
    """Above the RRR upper bound, the MF2 contribution is zero
    (clipped by the ``in_range`` mask) and the composed cross
    section must equal the raw MF3 interpolation."""
    xp = array_ns.get_backend('numpy')
    ranges_lru1 = []
    for iso in nb93_dict[2][151]['isotope'].values():
        for rng in iso['range'].values():
            if int(rng.get('LRU', 0)) == 1:
                ranges_lru1.append((float(rng['EL']), float(rng['EH'])))
    assert ranges_lru1, 'Nb-93 file unexpectedly has no LRU=1 range'
    eh = max(r[1] for r in ranges_lru1)
    e_above = np.geomspace(eh * 2, eh * 20, 51)
    from endf_userpy.mfsec_interpretation import mf3_interpretation
    mf3_xs = mf3_interpretation.compute_cross_section_agnostic(
        nb93_dict, 2, e_above, xp,
    )
    composed = res_comp.compute_reconstructed_cross_section(
        nb93_dict, 2, e_above, xp,
    )
    np.testing.assert_allclose(np.asarray(composed), np.asarray(mf3_xs),
                               rtol=1e-12)


def test_nb93_mt16_untouched_by_include_resonance(nb93_dict):
    """MT=16 (n,2n) is not in ``_MLBW_MT_TO_KEYS``, so
    ``reconstruct_resonance_xs`` returns zero even inside the
    resonance range and composition equals raw MF3."""
    xp = array_ns.get_backend('numpy')
    e_query = np.geomspace(1e6, 1.9e7, 51)
    contribution = res_comp.reconstruct_resonance_xs(
        nb93_dict, 16, e_query, xp,
    )
    assert np.all(np.asarray(contribution) == 0.0)


def test_nb93_get_reaction_xs_include_resonance_matches_composition(
    nb93_dict,
):
    """``get_reaction_xs(..., include_resonance=True)`` must
    produce the same array as calling
    :func:`compute_reconstructed_cross_section` directly for the
    same MT (single-MT reaction: elastic = MT=2)."""
    e_query = np.geomspace(1.0, 5e3, 201)
    xp = array_ns.get_backend('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        via_api = user_api.get_reaction_xs(
            nb93_dict, '(n,n_0)', e_query,
            include_resonance=True,
        )
        via_direct = res_comp.compute_reconstructed_cross_section(
            nb93_dict, 2, e_query, xp,
        )
    np.testing.assert_allclose(
        np.asarray(via_api), np.asarray(via_direct), rtol=1e-12,
    )


# ============================================================
# 3. U-235 (Reich-Moore).
# ============================================================


@pytest.fixture
def u235_dict():
    path = resolve_u235()
    if path is None:
        pytest.skip(
            'U-235 corpus file not available (run '
            'tests/data_law1_adhoc/fetch.sh to populate, or set '
            'U235_ENDF)'
        )
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp().parsefile(path)


def test_u235_composition_mt1_matches_manual_sum(u235_dict):
    xp = array_ns.get_backend('numpy')
    data = rm_pre.rm_data_from_endf_dict(u235_dict)
    e_query = np.array([0.025, 1.0, 100.0])   # inside RRR, cheap
    recon = rm.reconstruct(data, e_query, xp)
    from endf_userpy.mfsec_interpretation import mf3_interpretation
    mf3_xs = mf3_interpretation.compute_cross_section_agnostic(
        u235_dict, 1, e_query, xp,
    )
    expected = np.asarray(recon['tot']) + np.asarray(mf3_xs)
    composed = res_comp.compute_reconstructed_cross_section(
        u235_dict, 1, e_query, xp,
    )
    np.testing.assert_allclose(
        np.asarray(composed), expected, rtol=1e-12,
    )


@pytest.mark.skipif(not _numba_available(),
                    reason='numba not installed')
def test_u235_composition_numpy_vs_numba(u235_dict):
    """Same composition on numpy vs numba backends should match
    to within the ~1e-12 machine tolerance for a small query
    grid. Exercises the ``xp``-threading through the composition
    layer end-to-end."""
    e_query = np.array([0.025, 1.0, 100.0])
    xp_np = array_ns.get_backend('numpy')
    xp_nb = array_ns.get_backend('numba')
    composed_np = np.asarray(
        res_comp.compute_reconstructed_cross_section(
            u235_dict, 102, e_query, xp_np,
        )
    )
    composed_nb = np.asarray(
        res_comp.compute_reconstructed_cross_section(
            u235_dict, 102, e_query, xp_nb,
        )
    )
    np.testing.assert_allclose(composed_np, composed_nb, rtol=1e-10)


# ============================================================
# 6. Range-seam handling (issue #149).
#
# ENDF-6 stores adjacent resonance ranges as [EL, EH] intervals
# with the upper endpoint of one range equal to the lower
# endpoint of the next (RRR EH == URR EL; URR EH == "above URR"
# start). Naively including both endpoints in the range mask
# double-counts every query at exactly the shared boundary.
#
# The convention matches PR #146's tab1 side='right' default:
# at the doubled-x seam, the RIGHT-limit range wins, i.e. the
# higher-energy range owns the value at the shared E.
# ============================================================


def test_seam_urr_eh_only_mf3_fires_at_upper_endpoint():
    """At E == URR EH (the URR upper endpoint), the URR
    reconstruction must NOT contribute -- the value there is
    owned by MF3, which starts at that energy (or extends past
    it via its above-URR tabulation). Pre-fix this fired the
    URR contribution AND MF3, giving a factor-2 wrong value at
    exactly the seam.
    """
    d = _dict_with_valid_lssf0_urr(mt=102)
    xp = array_ns.get_backend('numpy')
    # URR range in this synthetic dict: [1e3, 1e5].
    # Sample: just below EH, at EH, just above EH.
    e_query = np.array([1e5 - 1.0, 1e5, 1e5 + 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        res_only = np.asarray(res_comp.reconstruct_resonance_xs(
            d, 102, e_query, xp,
        ))
        composed = np.asarray(res_comp.compute_reconstructed_cross_section(
            d, 102, e_query, xp,
        ))
    from endf_userpy.mfsec_interpretation import mf3_interpretation
    mf3_xs = np.asarray(mf3_interpretation.compute_cross_section_agnostic(
        d, 102, e_query, xp,
    ))
    # Just below EH: URR fires.
    assert res_only[0] > 0.0
    # At exactly EH: URR must NOT fire (right-limit convention).
    assert res_only[1] == 0.0, (
        f'URR contribution at exact upper endpoint E=EH must be zero '
        f'(right-limit convention; issue #149), got {res_only[1]}'
    )
    # Above EH: URR does not fire.
    assert res_only[2] == 0.0
    # Composed at EH: exactly MF3 (no URR added on top).
    np.testing.assert_allclose(composed[1], mf3_xs[1], rtol=1e-14, atol=0.0)
    # Composed at EH+1 also just MF3.
    np.testing.assert_allclose(composed[2], mf3_xs[2], rtol=1e-14, atol=0.0)


def test_seam_urr_el_fires_inclusively_at_lower_endpoint():
    """At E == URR EL, the URR reconstruction DOES fire (the
    lower endpoint is inclusive -- the URR range owns its own
    lower boundary). Symmetric to the upper-endpoint half-open
    rule: [EL, EH). Guards against overshooting the fix into a
    double-open interval that leaves EL uncovered.
    """
    d = _dict_with_valid_lssf0_urr(mt=102)
    xp = array_ns.get_backend('numpy')
    # URR range: [1e3, 1e5].
    e_query = np.array([1e3 - 1.0, 1e3, 1e3 + 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        res_only = np.asarray(res_comp.reconstruct_resonance_xs(
            d, 102, e_query, xp,
        ))
    # Below EL: no URR.
    assert res_only[0] == 0.0
    # At exactly EL: URR fires.
    assert res_only[1] > 0.0, (
        f'URR contribution at exact lower endpoint E=EL must be positive '
        f'(EL is inclusive; issue #149), got {res_only[1]}'
    )
    # Just above EL: URR fires (continuous with EL).
    assert res_only[2] > 0.0
    np.testing.assert_allclose(res_only[1], res_only[2], rtol=1e-3)


@pytest.fixture
def nd143_dict():
    """ENDF/B-VIII.1 Nd-143: RRR (MLBW) up to 5503 eV + LSSF=0
    URR from 5503 eV to 225 keV + MF3-above-URR. Used to pin
    the range-seam convention on real data with both a RRR/URR
    seam and a URR/MF3 seam in the same file."""
    path = resolve_nd143()
    if path is None:
        pytest.skip(
            'Nd-143 ENDF file not available (set ND143_ENDF, run '
            'tests/data_law1_adhoc/fetch.sh, or place the file at '
            'tests/data_law1_adhoc/endfb81_n_Nd-143.endf)'
        )
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp().parsefile(path, include=[1, 2, 3])


def test_nd143_rrr_urr_seam_takes_urr_value(nd143_dict):
    """RRR/URR seam at E = 5503 eV on ENDF/B-VIII.1 Nd-143.
    Pre-fix, composition returned RRR(EH) + URR(EL) summed
    (~163 barn on MT=2 vs the correct URR-side value ~23 barn,
    a factor of 7 wrong at the seam).
    """
    xp = array_ns.get_backend('numpy')
    e_seam = 5503.0    # = RRR EH = URR EL
    e_query = np.array([e_seam - 1e-3, e_seam, e_seam + 1e-3])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for mt in (2, 102):
            xs = np.asarray(res_comp.compute_reconstructed_cross_section(
                nd143_dict, mt, e_query, xp,
            ))
            # Below the seam: RRR value. Above the seam: URR value.
            # At the seam: URR value (right-limit convention).
            below, at, above = xs
            # The at-seam value must equal the just-above-seam value
            # (URR side), not the just-below value (RRR side).
            assert abs(at - above) < 0.01 * abs(above), (
                f'MT={mt}: at-seam value {at:.5f} should match URR side '
                f'{above:.5f} (right-limit); got {abs(at-above):.5f} away. '
                f'Below-seam RRR value {below:.5f} is the double-count '
                f'value if the bug is back.'
            )
            # Sanity: below-seam (RRR) is a real resonance, distinct
            # from URR value.
            assert abs(below - above) > 0.1 * above, (
                f'MT={mt}: below/above seam should differ (RRR peak vs '
                f'URR average); got below={below} above={above}. Test '
                f'setup may not be exercising the seam.'
            )


def test_nd143_urr_mf3_seam_takes_mf3_value(nd143_dict):
    """URR/MF3 seam at E = 225000 eV on ENDF/B-VIII.1 Nd-143.
    Pre-fix, composition summed URR(EH) + MF3(EH) at the exact
    seam, giving a factor-2 wrong value there.
    """
    xp = array_ns.get_backend('numpy')
    e_seam = 225000.0   # = URR EH; MF3 starts at (or extends past) this E
    e_query = np.array([e_seam - 1e-3, e_seam, e_seam + 1e-3])
    from endf_userpy.mfsec_interpretation import mf3_interpretation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for mt in (2, 102):
            xs = np.asarray(res_comp.compute_reconstructed_cross_section(
                nd143_dict, mt, e_query, xp,
            ))
            mf3_at_seam = float(np.asarray(
                mf3_interpretation.compute_cross_section_agnostic(
                    nd143_dict, mt, np.array([e_seam]), xp,
                )
            )[0])
            below, at, above = xs
            # Above-seam and at-seam should both be exactly MF3 alone.
            np.testing.assert_allclose(
                above, mf3_at_seam, rtol=1e-6,
                err_msg=f'MT={mt}: above URR EH must equal MF3',
            )
            np.testing.assert_allclose(
                at, mf3_at_seam, rtol=1e-6,
                err_msg=(f'MT={mt}: at URR EH must equal MF3 (right-limit '
                         f'convention; issue #149), got {at:.5f} vs MF3 '
                         f'{mf3_at_seam:.5f}. Below-seam value {below:.5f} '
                         f'includes the URR contribution.'),
            )
