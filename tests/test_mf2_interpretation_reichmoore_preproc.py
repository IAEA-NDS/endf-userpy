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
    apl_per_L=None,    # dict {L: APL}, per-L override (else uses `apl`)
    emax=1000.0,
):
    """Minimal MF1/MT451 + MF2/MT151 LRF=3 dict.

    Each l_group is (L, resonances). Each resonance is a 6-tuple
    (ER, AJ, GN, GG, GFA, GFB). Optional ``apl_per_L`` overrides
    the single ``apl`` for specific L values (JEFF-4.0 Fe-56
    pattern)."""
    if l_groups is None:
        l_groups = [(0, [(100.0, 3.0, 0.001, 0.04, 0.0, 0.0)])]
    apl_per_L = apl_per_L or {}
    spingroups = {}
    for idx, (L, resonances) in enumerate(l_groups, start=1):
        ers = {i + 1: r[0] for i, r in enumerate(resonances)}
        ajs = {i + 1: r[1] for i, r in enumerate(resonances)}
        gns = {i + 1: r[2] for i, r in enumerate(resonances)}
        ggs = {i + 1: r[3] for i, r in enumerate(resonances)}
        gfas = {i + 1: r[4] for i, r in enumerate(resonances)}
        gfbs = {i + 1: r[5] for i, r in enumerate(resonances)}
        apl_this_L = apl_per_L.get(L, apl)
        spingroups[idx] = {
            'AWRI': awri, 'APL': apl_this_L, 'L': L,
            'NRS': len(resonances),
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
    """One L=0 resonance with GFA=GFB=0 -> one J·π group with
    the resonance, nfis=0. Uses spi=0 so the single-neutron
    channel spin S=1/2 admits only one J at each L; the preproc's
    phantom-group logic contributes nothing here."""
    d = _minimal_rm_endf_dict(
        spi=0.0,
        l_groups=[(0, [(100.0, 0.5, 0.001, 0.04, 0.0, 0.0)])],
    )
    data = pre.rm_data_from_endf_dict(d)
    assert data.group_l.tolist() == [0]
    assert data.group_nfis.tolist() == [0]
    # g_J = (2|J|+1) / ((2 s_inc + 1)(2 I + 1)) = 2 / (2·1) = 1
    assert abs(data.group_g[0] - 1.0) < 1e-12
    assert data.res_group.tolist() == [0]
    assert data.res_gf1[0] == 0.0
    assert data.res_gf2[0] == 0.0


def test_fission_channel_activation_via_nonzero_gfa_gfb():
    """nfis derived from whether any resonance in the group has a
    non-zero GFA (→ nfis>=1) or GFB (→ nfis=2). Filters phantom
    groups (empty resonance lists) out of the nfis assertion since
    phantoms are always nfis=0 and don't reflect the original
    resonances' fission channels."""
    # Only GFA non-zero -> nfis = 1
    d = _minimal_rm_endf_dict(
        l_groups=[(0, [
            (100.0, 3.0, 0.05, 0.04,  0.02, 0.0),
            (200.0, 3.0, 0.03, 0.04, -0.01, 0.0),
        ])],
    )
    data = pre.rm_data_from_endf_dict(d)
    # Real group (has resonances) at (L=0, |J|=3) with nfis=1;
    # any other groups are phantoms.
    real_groups = np.unique(data.res_group)
    assert real_groups.size == 1
    assert data.group_nfis[real_groups[0]] == 1
    # Signs of GFA preserved on the real resonances
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
    real_groups2 = np.unique(data2.res_group)
    assert real_groups2.size == 1
    assert data2.group_nfis[real_groups2[0]] == 2


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
    """L=0 and L=1 groups each with a J each -> two distinct REAL
    (L, |J|) groups. Uses spi=0 (single channel spin) and picks
    J values that saturate the allowed range at each L, so the
    preproc adds no phantom groups on top.

    spi=0, i=1/2 -> S=1/2 only, allowed J:
      L=0: J=1/2
      L=1: J=1/2, J=3/2
    Listing resonances at all three -> zero phantoms."""
    d = _minimal_rm_endf_dict(
        spi=0.0,
        l_groups=[
            (0, [(100.0, 0.5, 0.05, 0.04, 0.0, 0.0)]),
            (1, [
                (200.0, 0.5, 0.03, 0.04, 0.0, 0.0),
                (300.0, 1.5, 0.02, 0.04, 0.0, 0.0),
            ]),
        ],
    )
    data = pre.rm_data_from_endf_dict(d)
    # Real (resonance-carrying) groups: (L=0,J=1/2), (L=1,J=1/2),
    # (L=1,J=3/2). Ordering is by (L, |J|).
    real_groups = sorted(set(data.res_group.tolist()))
    real_Ls = [int(data.group_l[g]) for g in real_groups]
    assert real_Ls == [0, 1, 1]


def test_channel_spin_ambiguity_split_by_aj_sign():
    """When ENDF-6 LRF=3 encodes channel-spin ambiguity via the sign
    of AJ, the preproc must keep the two channel spins as separate
    groups. Merging on |AJ| would (i) mix R-matrix contributions
    from disjoint channel-spin blocks and (ii) drop one g_J from
    the potential-scattering sum. Bug it pins: sum of g_J at each
    L was 2 instead of the physical 2L+1=3 on K-39 (I=3/2, L=1)
    because AJ=+1 and AJ=-1 resonances (distinct channel spins
    S=1 and S=2 at J=1) collapsed into a single group.
    """
    # Two J-groups with same |J|=1 but opposite AJ sign -> two
    # distinct channel-spin groups.
    d = _minimal_rm_endf_dict(
        spi=1.5,
        l_groups=[(1, [
            (100.0,  1.0, 0.05, 0.04, 0.0, 0.0),   # AJ = +1 (channel spin A)
            (200.0, -1.0, 0.03, 0.04, 0.0, 0.0),   # AJ = -1 (channel spin B)
        ])],
    )
    data = pre.rm_data_from_endf_dict(d)
    # File resonances land in two REAL groups at (L=1, |J|=1), one
    # for each channel spin. Other L=1 J values physically allowed
    # (0, 2, 3) come back as phantom groups; the test focuses on
    # the file's own two groups being kept separate.
    real_g = sorted(set(data.res_group.tolist()))
    assert len(real_g) == 2
    for g in real_g:
        assert data.group_l[g] == 1
    # Both real groups get the SAME single-channel-spin g_J. The
    # sum-of-REAL-groups g_J = 2 * (2*1+1)/((2*0.5+1)(2*1.5+1)) =
    # 2 * 3/8 = 0.75, corresponding to two of the physical
    # (L=1,J=1) channel spins. If merged incorrectly, the pair
    # would collapse to one group with g_J = 3/8.
    import numpy as np
    real_gJ_sum = float(np.sum(data.group_g[real_g]))
    assert real_gJ_sum == pytest.approx(0.75)


def test_phantom_groups_added_for_missing_channel_spins():
    """When a file lists resonances for only one channel spin at
    an (L, |J|) that physically admits two, the preproc must add
    a phantom group with no resonances so the reconstruction
    emits the missing channel spin's hard-sphere phase. Mirrors
    NJOY reconr's csrmat kkkkkk=2 branch.

    Setup: Co-59-like target with spi=3.5. At L=1 the channel spins
    S=3 and S=4 admit J=3 and J=4 each with both S. Listing only
    AJ=+3 and AJ=+4 in the file leaves one channel spin unlisted
    at each of those two J values, so the preproc must add two
    phantom groups. J=2 (S=3 only) and J=5 (S=4 only) are boundary
    values where only one channel spin is physical; the file lists
    neither, so those add one phantom each (the whole physically-
    allowed hard-sphere phase for that J at L=1)."""
    d = _minimal_rm_endf_dict(
        spi=3.5,
        l_groups=[(1, [
            (100.0, 3.0, 0.05, 0.04, 0.0, 0.0),   # AJ = +3
            (200.0, 4.0, 0.03, 0.04, 0.0, 0.0),   # AJ = +4
        ])],
    )
    data = pre.rm_data_from_endf_dict(d)
    real = sorted(set(data.res_group.tolist()))
    assert len(real) == 2   # AJ=+3 and AJ=+4 stay separate real groups
    # Total groups: 2 real (J=3+ and J=4+) plus 4 phantoms
    # (missing-J=2, missing-J=3-, missing-J=4-, missing-J=5). We
    # care that the sum of g_J across ALL L=1 groups now hits
    # 2L+1 = 3 (the neutron statistical sum rule), which requires
    # every physically-allowed (J, S) coupling to be represented.
    import numpy as np
    L1_mask = data.group_l == 1
    total_gJ_L1 = float(np.sum(data.group_g[L1_mask]))
    gj_den = (2*0.5 + 1) * (2*3.5 + 1)   # 2 * 8 = 16
    assert total_gJ_L1 == pytest.approx((2*1 + 1) / gj_den * gj_den / 1.0)
    # Same thing spelled out: sum of (2J+1) over all (J,S) at L=1
    # divided by gj_den. For L=1 with S in {3,4}:
    #   S=3: J in {2,3,4},   contributions (2J+1) = 5, 7, 9  (sum 21)
    #   S=4: J in {3,4,5},   contributions (2J+1) = 7, 9, 11 (sum 27)
    # Total (2J+1) = 48. Divide by gj_den=16 -> 3.0 = 2L+1. Pin
    # to catch a regression that would drop any phantom.
    assert total_gJ_L1 == pytest.approx(3.0)


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


def test_per_L_apl_produces_per_group_radii_JEFF_Fe56_pattern():
    """Regression pinning: JEFF-4.0 Fe-56 has APL differing per L
    (L=0 uses range AP, L=1 uses APL=0.5002, etc.). Our preproc
    must fill per-group ``group_r_a`` / ``group_r_ap`` arrays so
    the reconstruction picks the right radius per L.

    Bug it pins: previously a single ``apl_ref`` (first non-zero
    APL) was applied to every group, giving an 83% error on
    Fe-56 elastic at interference-minimum energies.
    """
    d = _minimal_rm_endf_dict(
        ap=0.5444,
        apl_per_L={0: 0.0, 1: 0.5002, 2: 0.0},
        naps=1,
        l_groups=[
            (0, [(100.0, 0.5, 0.001, 0.04, 0.0, 0.0)]),
            (1, [(200.0, 1.5, 0.001, 0.04, 0.0, 0.0)]),
            (2, [(300.0, 2.5, 0.001, 0.04, 0.0, 0.0)]),
        ],
    )
    data = pre.rm_data_from_endf_dict(d)

    assert data.group_r_ap is not None, (
        'preproc must fill group_r_ap for per-L APL support'
    )
    assert data.group_r_a is not None, (
        'preproc must fill group_r_a for per-L APL support'
    )

    # 3 groups: L=0, L=1, L=2. Per-L APL override -> group r_ap
    # equals APL where non-zero, else range AP.
    L_to_expected_r_ap = {0: 0.5444, 1: 0.5002, 2: 0.5444}
    L_arr = np.asarray(data.group_l)
    for g in range(len(L_arr)):
        L = int(L_arr[g])
        assert float(data.group_r_ap[g]) == pytest.approx(
            L_to_expected_r_ap[L]
        ), (
            f'group {g} at L={L}: r_ap = {float(data.group_r_ap[g])}, '
            f'expected {L_to_expected_r_ap[L]}'
        )

    # NAPS=1 -> channel radius equals scattering radius. Same pattern.
    for g in range(len(L_arr)):
        L = int(L_arr[g])
        assert float(data.group_r_a[g]) == pytest.approx(
            L_to_expected_r_ap[L]
        )


def test_per_L_apl_reconstruction_uses_per_L_phase():
    """End-to-end: reconstruction with per-L APL gives a
    different result than the (buggy) single-APL-for-all-L
    approximation. Uses two L-groups with different APLs and
    checks that swapping the APLs changes the elastic XS
    outside the potential-only limit."""
    from endf_userpy.mfsec_interpretation import (
        mf2_interpretation_reichmoore as rm,
    )
    from endf_userpy.primitives import array_ns

    # Two groups at the same energy, with differing APLs. Choose
    # ap and E so rho ~ O(1) and the L=1 phase is not tiny.
    d_correct = _minimal_rm_endf_dict(
        ap=0.5,
        apl_per_L={0: 0.5, 1: 0.9},
        naps=1,
        l_groups=[
            (0, [(1e3, 0.5, 0.001, 0.04, 0.0, 0.0)]),
            (1, [(2e3, 1.5, 0.001, 0.04, 0.0, 0.0)]),
        ],
        emax=1e5,
    )
    d_wrong = _minimal_rm_endf_dict(
        ap=0.5,
        apl_per_L={0: 0.5, 1: 0.5},   # same APL for both L
        naps=1,
        l_groups=[
            (0, [(1e3, 0.5, 0.001, 0.04, 0.0, 0.0)]),
            (1, [(2e3, 1.5, 0.001, 0.04, 0.0, 0.0)]),
        ],
        emax=1e5,
    )
    data_correct = pre.rm_data_from_endf_dict(d_correct)
    data_wrong = pre.rm_data_from_endf_dict(d_wrong)

    xp = array_ns.get_backend('numpy')
    einc = np.array([5e4], dtype=np.float64)   # high enough for L=1 to matter
    xs_correct = rm.reconstruct(data_correct, einc, xp)
    xs_wrong = rm.reconstruct(data_wrong, einc, xp)

    # Elastic must differ non-trivially when the L=1 radius
    # changes from 0.5 to 0.9. Without the per-L fix both would
    # return the same number.
    sct_correct = float(xs_correct['sct'][0])
    sct_wrong = float(xs_wrong['sct'][0])
    assert sct_correct != pytest.approx(sct_wrong, rel=1e-6), (
        f'per-L APL fix inactive: elastic identical whether L=1 '
        f'uses r_ap=0.9 (correct={sct_correct}) or '
        f'r_ap=0.5 (wrong={sct_wrong}).'
    )


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

    We pass at 20% tolerance because the sketch uses the
    shift-absorbed ``L̃_c = i P_c`` approximation (SAMMY
    shift-eliminated boundary condition, ``B_c = S_c(|E_r|)``)
    which can nudge thermal-region values by a few percent, and
    the file's tabulated values are the RECONSTRUCTED ones
    anyway -- a discrepancy of a couple percent is expected and
    fine here.

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


# ============================================================
# Preproc scalar fields are pytree-shaped (np.asarray, not float).
# Regression on the small autodiff-enablement fix: fields like abn
# / spi / ki / ap must be numpy 0-d arrays or scalars so a caller
# can substitute a JAX tracer via dataclasses.replace(...) and have
# the reconstruction flow through it.
# ============================================================


def _jax_available():
    return 'jax' in array_ns.available_backends()


def test_scalar_fields_are_array_shaped_not_python_float():
    """RMData scalar fields (abn, spi, ap) should be numpy scalars
    or 0-d arrays after preproc, NOT Python ``float``. Storing as
    Python float destroys the substitution pathway that lets a
    caller swap in a JAX tracer for those fields."""
    d = _minimal_rm_endf_dict(
        l_groups=[(0, [(1.0, 0.5, 0.1, 0.05, 0.0, 0.0)])],
    )
    data = pre.rm_data_from_endf_dict(d)
    for field_name in ('abn', 'spi'):
        value = getattr(data, field_name)
        assert not isinstance(value, float), (
            f'RMData.{field_name} is a Python float ({value!r}); '
            f'expected a numpy scalar or 0-d array so JAX '
            f'substitution can flow through.'
        )
        # Must still be numerically usable as a scalar.
        assert float(value) == pytest.approx(float(value))


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_can_substitute_scalar_field_and_reconstruct():
    """Concrete JAX-substitution smoke test: replace RMData.abn with
    a jnp array, call reconstruct on the JAX backend, verify the
    output tracks the substituted value linearly (RM sct scales
    linearly with abn since abn multiplies every cross section)."""
    import dataclasses
    import jax.numpy as jnp
    d = _minimal_rm_endf_dict(
        l_groups=[(0, [(1.0, 0.5, 0.1, 0.05, 0.0, 0.0)])],
    )
    data = pre.rm_data_from_endf_dict(d)
    xp = array_ns.get_backend('jax')

    einc = jnp.array([0.5, 1.0, 1.5])
    xs_1 = np.asarray(rm.reconstruct(
        dataclasses.replace(data, abn=jnp.asarray(1.0)),
        einc, xp,
    )['sct'])
    xs_2 = np.asarray(rm.reconstruct(
        dataclasses.replace(data, abn=jnp.asarray(2.0)),
        einc, xp,
    )['sct'])
    np.testing.assert_allclose(xs_2, 2.0 * xs_1, rtol=1e-10,
                                err_msg='sct should scale linearly with abn')


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_can_substitute_group_r_ap_and_autodiff():
    """Substituting a JAX tracer into RMData.group_r_ap must flow
    through reconstruction. Pins that the per-L APL machinery is
    autodiff-friendly: no float() casts on group_g/group_r_a/group_r_ap
    inside the reconstruction, so users can fit per-group radii via
    jax.grad through elastic XS."""
    import dataclasses
    import jax
    import jax.numpy as jnp

    d = _minimal_rm_endf_dict(
        ap=0.5,
        apl_per_L={0: 0.5, 1: 0.9},
        naps=1,
        l_groups=[
            (0, [(1e3, 0.5, 0.001, 0.04, 0.0, 0.0)]),
            (1, [(2e3, 1.5, 0.001, 0.04, 0.0, 0.0)]),
        ],
        emax=1e5,
    )
    data = pre.rm_data_from_endf_dict(d)
    xp = array_ns.get_backend('jax')
    einc = jnp.array([5e4])

    base_r_ap = jnp.asarray(data.group_r_ap)
    base_r_a = jnp.asarray(data.group_r_a)

    def sct_of_r_ap_L1(r_ap_L1_scalar):
        new_r_ap = base_r_ap.at[1].set(r_ap_L1_scalar)
        new_r_a = base_r_a.at[1].set(r_ap_L1_scalar)   # NAPS=1
        d2 = dataclasses.replace(
            data,
            group_r_ap=new_r_ap,
            group_r_a=new_r_a,
        )
        return rm.reconstruct(d2, einc, xp)['sct'][0]

    grad = jax.grad(sct_of_r_ap_L1)(jnp.asarray(0.9))
    grad_val = float(grad)
    assert np.isfinite(grad_val), (
        f'jax.grad through group_r_ap must produce a finite gradient; '
        f'got {grad_val}. If NaN/inf, a float() cast (or non-jax op) '
        f'is severing the trace inside reconstruction.'
    )
    assert grad_val != 0.0, (
        'gradient of elastic w.r.t. L=1 group r_ap must be non-zero '
        'at r_ap=0.9 (potential elastic depends on it via sin(phi_1)).'
    )
