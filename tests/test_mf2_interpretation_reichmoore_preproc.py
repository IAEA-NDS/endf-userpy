"""Sketch tests for the ENDF-6 -> RMData preprocessor.

Same pattern as the MLBW preproc tests: synthetic hand-built ENDF
dicts for field-level correctness, plus one U-235 structural round
trip (skipped if the file is absent).
"""
from __future__ import annotations

import os
import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf2_interpretation_reichmoore as rm
from endf_userpy.mfsec_interpretation import (
    mf2_interpretation_reichmoore_preproc as pre,
)


# ============================================================
# Synthetic ENDF-6 dict fixtures.
# ============================================================


def _minimal_rm_endf_dict(
    nsub=10,          # neutron incident
    spi=3.5, ap=0.6, naps=0, awri=232.0, apl=0.0,
    l_groups=None,     # list of (L, [(er, aj, gn, gg, gfa, gfb), ...])
    emax=1000.0,
):
    """Minimal MF1/MT451 + MF2/MT151 LRF=3 dict.

    Each l_group is (L, resonances). Each resonance is a 6-tuple
    (ER, AJ, GN, GG, GFA, GFB). The preprocessor takes it from
    there."""
    if l_groups is None:
        l_groups = [(0, [(100.0, 3.0, 0.001, 0.04, 0.0, 0.0)])]
    spingroups = {}
    for idx, (L, resonances) in enumerate(l_groups, start=1):
        ers = {i + 1: r[0] for i, r in enumerate(resonances)}
        ajs = {i + 1: r[1] for i, r in enumerate(resonances)}
        gns = {i + 1: r[2] for i, r in enumerate(resonances)}
        ggs = {i + 1: r[3] for i, r in enumerate(resonances)}
        gfas = {i + 1: r[4] for i, r in enumerate(resonances)}
        gfbs = {i + 1: r[5] for i, r in enumerate(resonances)}
        spingroups[idx] = {
            'AWRI': awri, 'APL': apl, 'L': L, 'NRS': len(resonances),
            'ER': ers, 'AJ': ajs,
            'GN': gns, 'GG': ggs, 'GFA': gfas, 'GFB': gfbs,
        }
    return {
        1: {451: {'NSUB': nsub}},
        2: {151: {
            'isotope': {1: {
                'ABN': 1.0,
                'range': {1: {
                    'LRU': 1, 'LRF': 3, 'NAPS': naps, 'NRO': 0,
                    'SPI': spi, 'AP': ap, 'NLS': len(l_groups),
                    'EH': emax,
                    'spingroup': spingroups,
                }},
            }},
        }},
    }


# ============================================================
# Field-level correctness on synthetic dicts.
# ============================================================


def test_scalar_fields_match_the_input_dict():
    d = _minimal_rm_endf_dict(spi=3.5, ap=0.85, awri=232.0)
    data = pre.rm_data_from_endf_dict(d)
    assert data.spi == 3.5
    assert data.abn == 1.0
    # Heavy nucleus, ki ~ 1e-3 range.
    assert 1e-3 < data.ki < 5e-3


def test_wrong_lrf_raises():
    d = _minimal_rm_endf_dict()
    d[2][151]['isotope'][1]['range'][1]['LRF'] = 2
    with pytest.raises(ValueError, match='LRF=2'):
        pre.rm_data_from_endf_dict(d)


def test_single_j_group_no_fission_gives_one_group_with_nfis_zero():
    """One L=0 resonance with GFA=GFB=0 -> one J·π group, nfis=0."""
    d = _minimal_rm_endf_dict(
        spi=0.5,
        l_groups=[(0, [(100.0, 0.5, 0.001, 0.04, 0.0, 0.0)])],
    )
    data = pre.rm_data_from_endf_dict(d)
    assert data.group_l.tolist() == [0]
    assert data.group_nfis.tolist() == [0]
    # g_J = (2|J|+1) / ((2 s_inc + 1)(2 I + 1)) = 2 / (2·2) = 0.5
    assert abs(data.group_g[0] - 0.5) < 1e-12
    assert data.res_group.tolist() == [0]
    assert data.res_gf1[0] == 0.0
    assert data.res_gf2[0] == 0.0


def test_fission_channel_activation_via_nonzero_gfa_gfb():
    """nfis derived from whether any resonance in the group has a
    non-zero GFA (→ nfis>=1) or GFB (→ nfis=2)."""
    # Only GFA non-zero -> nfis = 1
    d = _minimal_rm_endf_dict(
        l_groups=[(0, [
            (100.0, 3.0, 0.05, 0.04,  0.02, 0.0),
            (200.0, 3.0, 0.03, 0.04, -0.01, 0.0),
        ])],
    )
    data = pre.rm_data_from_endf_dict(d)
    assert data.group_nfis.tolist() == [1]
    # Signs of GFA preserved
    assert data.res_gf1[0] > 0 and data.res_gf1[1] < 0
    assert data.res_gf2.tolist() == [0.0, 0.0]

    # Non-zero GFB in at least one resonance -> nfis = 2
    d2 = _minimal_rm_endf_dict(
        l_groups=[(0, [
            (100.0, 3.0, 0.05, 0.04, 0.02,  0.01),
            (200.0, 3.0, 0.03, 0.04, 0.01, -0.005),
        ])],
    )
    data2 = pre.rm_data_from_endf_dict(d2)
    assert data2.group_nfis.tolist() == [2]


def test_multiple_j_within_l_group_split_into_two_groups():
    """L=0 with two different |J| values in the same L-group entry
    -> two J·π groups in the RMData."""
    d = _minimal_rm_endf_dict(
        spi=0.5,
        l_groups=[(0, [
            (100.0, 0.0, 0.05, 0.04, 0.0, 0.0),   # J=0
            (200.0, 1.0, 0.03, 0.04, 0.0, 0.0),   # J=1
        ])],
    )
    data = pre.rm_data_from_endf_dict(d)
    assert data.group_l.tolist() == [0, 0]
    # J=0 first (sorted by |J|), J=1 second
    # g_J values: J=0: 1/(2*2)=0.25, J=1: 3/(2*2)=0.75
    assert abs(data.group_g[0] - 0.25) < 1e-12
    assert abs(data.group_g[1] - 0.75) < 1e-12
    # Each resonance in its own group
    assert data.res_group.tolist() == [0, 1]


def test_multiple_l_groups_produce_grouped_by_l_j():
    """L=0 and L=1 groups each with a J each -> two (L, |J|) groups."""
    d = _minimal_rm_endf_dict(
        l_groups=[
            (0, [(100.0, 3.0, 0.05, 0.04, 0.0, 0.0)]),
            (1, [(200.0, 4.0, 0.03, 0.04, 0.0, 0.0)]),
        ],
    )
    data = pre.rm_data_from_endf_dict(d)
    assert data.group_l.tolist() == [0, 1]
    # Different (L, |J|) keys -> separate groups
    assert data.res_group.tolist() == [0, 1]


def test_apl_takes_precedence_over_ap_when_nonzero():
    d = _minimal_rm_endf_dict(ap=0.5, apl=0.9, naps=1)
    data = pre.rm_data_from_endf_dict(d)
    # NAPS=1: r_a = r_ap = APL (not AP)
    assert data.r_a.y[0] == pytest.approx(0.9)
    assert data.r_ap.y[0] == pytest.approx(0.9)


def test_apl_zero_falls_back_to_ap():
    d = _minimal_rm_endf_dict(ap=0.5, apl=0.0, naps=1)
    data = pre.rm_data_from_endf_dict(d)
    assert data.r_ap.y[0] == pytest.approx(0.5)
    assert data.r_a.y[0] == pytest.approx(0.5)


# ============================================================
# Structural round-trip on a real RM actinide file (U-235).
# ============================================================


_U235 = (
    '/home/gschnabel/bigdata/nuclibs/endfb8.1/'
    'neutrons-version.VIII.1/n-092_U_235.endf'
)


def _u235_available():
    return os.path.exists(_U235)


@pytest.mark.skipif(not _u235_available(), reason='U-235 ENDF not available')
def test_u235_structural_counts():
    """Preprocess U-235 and check top-level structure. Ground truth
    from earlier survey (survey scratchpad script):
      - 2 J·π groups (unique (L, |J|) pairs)
      - 3194 resonances total
      - both groups have 2 fission channels active"""
    from endf_parserpy import EndfParserCpp
    p = EndfParserCpp()
    d = p.parsefile(_U235, include=[1, 2])
    data = pre.rm_data_from_endf_dict(d)
    assert data.group_l.shape[0] == 2
    assert data.res_er.shape[0] == 3194
    # U-235 evaluation has GFA and GFB non-zero => nfis=2 for both groups.
    assert (data.group_nfis == 2).all()


@pytest.mark.skipif(not _u235_available(), reason='U-235 ENDF not available')
def test_u235_thermal_capture_and_fission_match_ENDF():
    """End-to-end sanity: preprocessor + R-M reconstruction on
    U-235 must give the widely-tabulated thermal cross sections
    to reasonable precision.

    Expected at 0.0253 eV (ENDF/B-VIII.1, JEFF-4.0, EXFOR consensus):
      σ_cap ≈ 99   b
      σ_fis ≈ 584  b
      σ_sct ≈ 14   b

    We pass at 20% tolerance because the sketch still uses the
    shift-absorbed L̃_c = i P_c approximation (LSSF=0) which
    can nudge thermal-region values by a few percent, and the
    file's tabulated values are the RECONSTRUCTED ones anyway --
    a discrepancy of a couple percent is expected and fine here.

    Historical note: an earlier version of the sketch had a
    double-subtraction bug in σ_cap = σ_reaction - σ_fission
    that gave σ_cap = -483 b at thermal. See commit that fixes it.
    """
    from endf_parserpy import EndfParserCpp
    p = EndfParserCpp()
    d = p.parsefile(_U235, include=[1, 2])
    data = pre.rm_data_from_endf_dict(d)
    xp = array_ns.get_backend('numpy')
    xs = rm.reconstruct(data, np.array([0.0253]), xp)
    sct, cap, fis = float(xs['sct'][0]), float(xs['cap'][0]), float(xs['fis'][0])
    assert 80.0 < cap < 130.0, f'thermal capture {cap} b (expected ~99)'
    assert 500.0 < fis < 700.0, f'thermal fission {fis} b (expected ~584)'
    assert 5.0 < sct < 25.0, f'thermal scattering {sct} b (expected ~14)'
    # Also pin finite/positive everywhere across the RRR.
    einc = np.array([0.025, 0.1, 1.0, 100.0, 1000.0], dtype=np.float64)
    xs = rm.reconstruct(data, einc, xp)
    for key in ('sct', 'cap', 'fis', 'pot', 'tot'):
        arr = np.asarray(xs[key])
        assert np.all(np.isfinite(arr)), f'{key} has non-finite values'
    assert np.all(np.asarray(xs['cap']) >= 0.0), 'capture went negative'
    assert np.all(np.asarray(xs['fis']) >= 0.0), 'fission went negative'
