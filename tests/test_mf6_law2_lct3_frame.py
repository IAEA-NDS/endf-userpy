"""Regression test for MF6/LAW=2 in LCT=3 sections (issue #97).

PR #51 changed
`mf6_interpretation_subsecs.get_angdist_from_subsec_law2` to pass
an effective frame ``eff_lct`` to the Fortran evaluator instead of
the raw section-level ``LCT``:

- LCT=1 or LCT=2 -> eff_lct = LCT (frame is unambiguous).
- LCT=3 (CM for light ejectiles, LAB for heavy) ->
  eff_lct = 1 if AWP > 4 else 2.

Before PR #51, an LCT=3 subsection with ``awp <= 4`` (light
ejectile, typically a neutron) handed lct=3 to the Fortran, whose
``mf4lab2cm`` only performs the CM->LAB transformation when
``lct == 2``. Result: the angular distribution came back in CM
while being labelled LAB, which the caller then treats as LAB in
downstream compositions.

No file in the corpus (or in the wider fetch-on-demand set) has
an MF6 LAW=2 subsection with LCT=3 natively: files that use LCT=3
carry their LAW=2 in MF4 or use LAW=1 for the multi-particle-exit
subsections that need frame-mixing. Rather than pin the fix on a
"file we don't have", the test synthetically mutates Al-27 MT 51's
MF6/LAW=2 subsection from LCT=2 to LCT=3 and asserts:

1. The frame-aware code path treats a light-ejectile LCT=3
   subsection identically to a LCT=2 subsection (both should
   produce eff_lct=2 -> same CM->LAB transformation when
   ``to_lab=True``).
2. Passing ``to_lab=False`` produces a different angular
   distribution than ``to_lab=True`` (the CM<->LAB transformation
   actually matters at the elastic kinematics for Al-27
   MT 51 at 14 MeV; if the two agreed, the fix would be
   untested for real).

Pre-fix behaviour: LCT=3 without the eff_lct conversion would
have passed lct=3 to the Fortran, which doesn't transform, so
``to_lab=True`` on the mutated file would have equalled
``to_lab=False`` -- ill-formed. Post-fix, ``to_lab=True`` on the
mutated file matches ``to_lab=True`` on the unmutated file
(both go through eff_lct=2 CM->LAB).
"""
import copy
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation.mf6_interpretation_subsecs import (
    get_angdist_from_subsec_law2,
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


@pytest.fixture(scope='module')
def al27():
    """Al-27: MF6/MT 51 subsection 1 has LCT=2, LAW=2, ZAP=1
    (neutron ejectile), AWP=1 (light ejectile). Natural test bed
    for the frame-fix."""
    return _load('endfb81_n_Al-27.endf')


def _mutate_lct(endf, mt, new_lct):
    """Return a deep copy with MF6/MT's section-level LCT replaced."""
    mutated = copy.deepcopy(endf)
    mutated[6][mt]['LCT'] = new_lct
    return mutated


# ============================================================
# Rule: LCT=3 with light ejectile -> eff_lct = 2 (same as LCT=2).
# ============================================================


def test_lct3_light_ejectile_matches_lct2(al27):
    """Al-27 MT 51 subsection 1 has AWP=1 (neutron, light). With
    the eff_lct conversion in place (PR #51), mutating LCT from 2
    to 3 must produce the same angular distribution for
    to_lab=True: both paths compute eff_lct=2 and route through
    mf4lab2cm identically."""
    mt = 51
    subsec_num = 1
    einc = np.array([1.4e7])
    mus = np.linspace(-1.0, 1.0, 33)

    # Sanity: unmutated is LCT=2.
    assert al27[6][mt]['LCT'] == 2
    assert al27[6][mt]['subsection'][subsec_num]['AWP'] <= 4.0

    unmutated_lab = get_angdist_from_subsec_law2(
        al27, mt, subsec_num, einc, mus, to_lab=True,
    )

    mutated = _mutate_lct(al27, mt, 3)
    mutated_lab = get_angdist_from_subsec_law2(
        mutated, mt, subsec_num, einc, mus, to_lab=True,
    )

    # Both should be identical: same eff_lct=2, same Fortran path.
    np.testing.assert_allclose(
        mutated_lab, unmutated_lab, rtol=1e-10, atol=0.0,
        err_msg=(
            'Al-27 MT 51 MF6/LAW=2 with LCT=3 (light ejectile, '
            'AWP=1) must give the same to_lab=True angular '
            'distribution as the unmutated LCT=2 file. Pre-#51 the '
            'raw LCT=3 was passed straight to the Fortran, which '
            'skipped the CM->LAB transformation and returned the '
            'raw CM distribution -- differing from the LCT=2 output.'
        ),
    )


def test_lct3_light_ejectile_lab_differs_from_cm(al27):
    """Non-trivial fix: the to_lab=True result on the mutated
    LCT=3 file must actually differ from to_lab=False (pure CM
    output). If they agreed the fix would have nothing to test:
    the CM<->LAB transformation for elastic-like MT 51 at 14 MeV
    on Al-27 is non-trivial (few-percent-of-mu spread)."""
    mt = 51
    subsec_num = 1
    einc = np.array([1.4e7])
    mus = np.linspace(-1.0, 1.0, 33)

    mutated = _mutate_lct(al27, mt, 3)
    lab = get_angdist_from_subsec_law2(
        mutated, mt, subsec_num, einc, mus, to_lab=True,
    )
    cm = get_angdist_from_subsec_law2(
        mutated, mt, subsec_num, einc, mus, to_lab=False,
    )
    # The two must differ (frame transformation is non-trivial at
    # 14 MeV inelastic scattering on Al-27). If they agree, the
    # LCT=3 -> eff_lct=2 handling in this test is a no-op and the
    # fix is untested.
    max_diff = float(np.max(np.abs(lab - cm)))
    assert max_diff > 1e-6, (
        f'to_lab=True vs to_lab=False angular distributions agree '
        f'exactly (max diff {max_diff:.3e}) on Al-27 MT 51 LCT=3 '
        f'at 14 MeV: the frame transformation is a no-op and the '
        f'test cannot distinguish pre-fix from post-fix. Pick a '
        f'different Ein or a different discrete inelastic level.'
    )


def test_lct2_baseline_lab_differs_from_cm(al27):
    """Baseline sanity: on the unmutated Al-27 (LCT=2), to_lab=True
    also differs from to_lab=False. This confirms the reference
    behaviour for LCT=2 that the mutated LCT=3 case must match."""
    mt = 51
    subsec_num = 1
    einc = np.array([1.4e7])
    mus = np.linspace(-1.0, 1.0, 33)

    lab = get_angdist_from_subsec_law2(
        al27, mt, subsec_num, einc, mus, to_lab=True,
    )
    cm = get_angdist_from_subsec_law2(
        al27, mt, subsec_num, einc, mus, to_lab=False,
    )
    max_diff = float(np.max(np.abs(lab - cm)))
    assert max_diff > 1e-6, (
        f'Al-27 MT 51 (unmutated LCT=2) at 14 MeV: to_lab=True vs '
        f'to_lab=False differ by only {max_diff:.3e}, so the frame '
        f'transformation is negligible at this Ein and the '
        f'companion LCT=3 test cannot distinguish frames either.'
    )
