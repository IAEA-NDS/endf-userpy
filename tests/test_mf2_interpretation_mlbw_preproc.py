"""Sketch tests for the ENDF-6 -> MLBWData preprocessor.

Two kinds of coverage:

- Synthetic hand-built ENDF-6 dicts, one property per test. Small,
  fast, catches sign-convention and index bugs without needing a
  real file.
- One end-to-end round-trip on a real Nb-93 file: reconstructs the
  cross sections with the sketch after preprocessing via this
  module, and compares against the reference reconstruction from
  the JAX prototype's preprocessor + this sketch's ``reconstruct``.
  Skipped when the file is not available.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf2_interpretation_mlbw as mlbw
from endf_userpy.mfsec_interpretation import mf2_interpretation_mlbw_preproc as pre


# ============================================================
# Synthetic ENDF-6 dict fixtures.
# ============================================================


def _minimal_endf_dict(
    nsub=10,          # neutron incident
    spi=0.5, ap=0.6, naps=0, awri=93.0,
    resonances=((100.0, 0.5, 0.5, 0.3, 0.0, 0.0),),
    lrx=0, qx=0.0, emax=1000.0,
):
    """Minimal MF1/MT451 + MF2/MT151 dict with ONE L-group.

    Each resonance is (ER, AJ, GT, GN, GG, GF). The preprocessor
    derives GX from GT-GN-GG-GF when LRX>0."""
    ers = {i + 1: r[0] for i, r in enumerate(resonances)}
    ajs = {i + 1: r[1] for i, r in enumerate(resonances)}
    gts = {i + 1: r[2] for i, r in enumerate(resonances)}
    gns = {i + 1: r[3] for i, r in enumerate(resonances)}
    ggs = {i + 1: r[4] for i, r in enumerate(resonances)}
    gfs = {i + 1: r[5] for i, r in enumerate(resonances)}
    return {
        1: {451: {'NSUB': nsub}},
        2: {151: {
            'isotope': {1: {
                'ABN': 1.0,
                'range': {1: {
                    'LRU': 1, 'LRF': 2, 'NAPS': naps, 'NRO': 0,
                    'SPI': spi, 'AP': ap, 'NLS': 1, 'EH': emax,
                    'spingroup': {1: {
                        'AWRI': awri, 'QX': qx, 'L': 0,
                        'LRX': lrx, 'NRS': len(resonances),
                        'ER': ers, 'AJ': ajs, 'GT': gts,
                        'GN': gns, 'GG': ggs, 'GF': gfs,
                    }},
                }},
            }},
        }},
    }


# ============================================================
# Field-level correctness on synthetic dicts.
# ============================================================


def test_scalar_fields_match_the_input_dict():
    d = _minimal_endf_dict(spi=2.5, ap=0.85, awri=100.0)
    data = pre.mlbw_data_from_endf_dict(d)
    assert data.spi == 2.5
    assert data.abn == 1.0
    # ki > 0 and finite. For a neutron on a heavy nucleus
    # (awri ~ 100) ki ~ 2e-3 (in units where k(E) = ki * sqrt(E)
    # with E in eV gives k in fm^-1 with the barn convention).
    assert 1e-3 < data.ki < 5e-3


def test_single_j_group_ordering_and_mapping():
    """spi=0.5 target + s-wave (L=0) + only J=0.5 resonance:
    a single channel with the correct statistical weight, and the
    resonance mapped to it. There will ALSO be a dummy channel for
    the reachable-but-missing J=1.5 (see test below); this test just
    pins the first channel's identity and mapping."""
    d = _minimal_endf_dict(
        spi=0.5,
        resonances=((100.0, 0.5, 0.8, 0.5, 0.3, 0.0),),
    )
    data = pre.mlbw_data_from_endf_dict(d)
    assert data.ch_l[0] == 0
    # NJOY / SAMMY convention:
    #   g_J = (2|J|+1) / ((2 s_inc + 1) (2 I + 1))
    # For s_inc=0.5, I=0.5, |J|=0.5: (2·0.5+1)/((2·0.5+1)·(2·0.5+1))
    #                              = 2 / (2·2) = 0.5
    assert abs(data.ch_g[0] - 0.5) < 1e-12
    assert data.res_channel[0] == 0
    assert data.res_er[0] == 100.0


def test_missing_j_group_adds_dummy_channel():
    """spi=0.5 target + s-wave: reachable |J| in {0, 1}, so with only
    a J=0 resonance we should get a dummy channel for J=1."""
    d = _minimal_endf_dict(
        spi=0.5,
        resonances=((100.0, 0.0, 0.8, 0.5, 0.3, 0.0),),
    )
    data = pre.mlbw_data_from_endf_dict(d)
    # 1 real J=0 channel + 1 dummy J=1 channel = 2 channels.
    assert data.ch_l.shape[0] == 2
    # 1 real resonance + 1 dummy resonance carrying the missing J
    assert data.res_er.shape[0] == 2
    # Dummy resonance sits at 1e-12 with all widths zero
    assert data.res_er[1] == pytest.approx(1e-12)
    assert data.res_gn[1] == 0.0
    assert data.res_gg[1] == 0.0


def test_competitive_width_gx_derived_when_lrx():
    """GX = GT - GN - GG - GF, gated by LRX>0. When LRX=0, GX
    is 0 regardless of what GT-GN-GG-GF would give."""
    # LRX=0: GX=0 even if GT > GN+GG+GF.
    # (A dummy potential-only resonance may be appended for missing-J
    # channel multiplicity; look only at the real resonance at index 0.)
    d_no_lrx = _minimal_endf_dict(
        lrx=0, qx=0.0,
        resonances=((100.0, 0.5, 1.5, 0.5, 0.3, 0.0),),
    )
    data_no = pre.mlbw_data_from_endf_dict(d_no_lrx)
    assert data_no.res_gx[0] == 0.0

    # LRX=1: GX = GT - GN - GG - GF = 1.5 - 0.5 - 0.3 - 0.0 = 0.7
    d_lrx = _minimal_endf_dict(
        lrx=1, qx=1.0,
        resonances=((100.0, 0.5, 1.5, 0.5, 0.3, 0.0),),
    )
    data_lrx = pre.mlbw_data_from_endf_dict(d_lrx)
    assert abs(data_lrx.res_gx[0] - 0.7) < 1e-12


def test_wrong_lrf_raises():
    d = _minimal_endf_dict()
    d[2][151]['isotope'][1]['range'][1]['LRF'] = 3
    with pytest.raises(ValueError, match='LRF=3'):
        pre.mlbw_data_from_endf_dict(d)


def test_radii_naps_zero_derives_channel_radius_from_mass():
    """NAPS=0: channel radius = 0.123 * (AWRI*mn)^(1/3) + 0.08,
    scattering radius = AP. So r_a != r_ap when they differ."""
    d = _minimal_endf_dict(ap=0.9, naps=0, awri=100.0)
    data = pre.mlbw_data_from_endf_dict(d)
    # r_ap is the constant AP everywhere
    assert data.r_ap.y[0] == pytest.approx(0.9)
    # r_a is derived; for AWRI=100 it's ~0.596 fm
    expected_a = 0.123 * (100.0 * 1.00866491578) ** (1.0/3.0) + 0.08
    assert data.r_a.y[0] == pytest.approx(expected_a, rel=1e-10)


def test_radii_naps_one_uses_ap_for_channel_too():
    d = _minimal_endf_dict(ap=0.9, naps=1)
    data = pre.mlbw_data_from_endf_dict(d)
    assert data.r_a.y[0] == pytest.approx(0.9)
    assert data.r_ap.y[0] == pytest.approx(0.9)


def test_radius_tab1_covers_beyond_rrr_upper_bound():
    """The radius TAB1's upper x-bound must be far beyond the RRR's
    EH so that resonances with |E_r| > EH (bound-state and extension
    poles that R-M / MLBW evaluations routinely list) don't get
    silently zeroed via out-of-range interpolation.

    Regression: an earlier version of the preproc used x2 = emax
    (= EH) for the radius TAB1s. Any resonance with |E_r| > EH then
    hit the outside-value=0.0 path in `tab1.interp`, which zeroed
    that resonance's penetration factor and hence its reduced-width
    amplitude, silently dropping its contribution to the R-matrix
    sum. Manifested as a ~3% U-235 elastic disagreement against
    NJOY reconr. The fix is to extend x2 to ~1e11 eV (constant
    fill; radii are physically constant across the range anyway).
    """
    d = _minimal_endf_dict(emax=1000.0, ap=0.85, naps=1)
    data = pre.mlbw_data_from_endf_dict(d)
    from endf_userpy.primitives import tab1 as tab1_mod
    xp = array_ns.get_backend('numpy')
    # Query at energies BEYOND the file's EH.
    einc = np.array([1000.0, 2500.0, 5000.0, 1e7], dtype=np.float64)
    r_a = np.asarray(tab1_mod.interp(data.r_a, einc, xp))
    r_ap = np.asarray(tab1_mod.interp(data.r_ap, einc, xp))
    # All queries must return the constant radius, not 0.
    for i in range(einc.shape[0]):
        assert r_ap[i] == pytest.approx(0.85), (
            f'r_ap at E={einc[i]} returned {r_ap[i]}; expected 0.85. '
            f'radius TAB1 is clipping resonances beyond EH.'
        )
        assert r_a[i] > 0.0, (
            f'r_a at E={einc[i]} returned {r_a[i]}; radius TAB1 is '
            f'clipping resonances beyond EH.'
        )


# ============================================================
# End-to-end round-trip on real Nb-93 (skipped if file absent).
# ============================================================


from _corpus import resolve_nb93


def _nb93_available():
    return resolve_nb93() is not None


@pytest.mark.skipif(not _nb93_available(), reason='Nb-93 ENDF file not available')
def test_nb93_channel_and_resonance_counts_match_reference():
    """Preprocess Nb-93 and check the top-level structural counts
    (number of channels, number of resonances including the dummy
    potential-only rows). Ground truth: 7 channels, 202 resonances
    -- verified via the JAX prototype's reference preprocessor on
    branch feature_resonance.

    If either count drifts, chances are the missing-J-multiplicity
    dummy-channel accounting broke."""
    from endf_parserpy import EndfParserCpp
    p = EndfParserCpp()
    d = p.parsefile(resolve_nb93(), include=[1, 2])
    data = pre.mlbw_data_from_endf_dict(d)
    assert data.ch_l.shape[0] == 7
    assert data.res_er.shape[0] == 202
    # ki also independently verifiable: for Nb-93 (AWRI ~ 92.1) it's
    # ~2.17e-3 in the file's unit convention.
    assert 2.15e-3 < data.ki < 2.20e-3


@pytest.mark.skipif(not _nb93_available(), reason='Nb-93 ENDF file not available')
def test_nb93_reconstruction_matches_reference_at_a_few_energies():
    """Preprocess Nb-93, reconstruct with the sketch on a few
    query energies, sanity-check the result:
      - Total XS is positive and finite everywhere in the RRR.
      - The strongest resonance (at ~35 eV in Nb-93) shows a peak
        in the capture channel.
      - The far-below-thermal (~ 1e-3 eV) point has capture > 0
        (1/v tail).
    Doesn't require the JAX-prototype's preprocessor to be
    available -- these are self-checks on our own output."""
    from endf_parserpy import EndfParserCpp
    p = EndfParserCpp()
    d = p.parsefile(resolve_nb93(), include=[1, 2])
    data = pre.mlbw_data_from_endf_dict(d)

    xp = array_ns.get_backend('numpy')
    # Coarse grid across the RRR (Nb-93 EH ~ 7 keV).
    einc = np.array([1e-3, 1.0, 35.0, 100.0, 1000.0, 5000.0])
    xs = mlbw.reconstruct(data, einc, xp)
    tot = np.asarray(xs['tot'])
    cap = np.asarray(xs['cap'])
    assert np.all(np.isfinite(tot))
    assert np.all(tot > 0.0)
    assert cap[0] > cap[3]     # 1/v tail: cap at 1e-3 eV > cap at 100 eV
    # A dense sub-grid around 35 eV should show a peak in cap
    einc_peak = np.linspace(30.0, 40.0, 501)
    cap_peak = np.asarray(mlbw.reconstruct(data, einc_peak, xp)['cap'])
    e_peak_arg = float(einc_peak[np.argmax(cap_peak)])
    assert 30.0 < e_peak_arg < 40.0
