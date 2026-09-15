"""Tests for endf_userpy.primitives.properties.

Pinned by issue #32: `get_ejectile` used to assert
`len(ejectiles) == 1 or ejectiles[0][1] == 'n'`, which crashed with an
uninformative AssertionError on multi-ejectile MTs whose first
ejectile is not a neutron: MT 112 = (n,pα), MT 115 = (n,pd),
MT 116 = (n,pt), MT 117 = (n,dα). The same assert bled into
`get_ZAP` and `is_zap_consistent`, so anything asking "does MT emit
particle X" on these MTs blew up.

The fix: `get_ejectile` now raises an informative `ValueError` when
no unique ejectile exists, and `is_zap_consistent` consults the
reaction-string ejectile table via `primitives.reactions.contains_zap`
so multi-ejectile MTs are handled correctly.
"""
from pathlib import Path
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.primitives.properties import (
    get_ejectile, get_ZAP, is_zap_consistent,
)
from endf_userpy.primitives.physical_constants import PARTICLE_ZAP


DATA_DIR = Path(__file__).resolve().parent / 'data'
ADHOC_DATA_DIR = Path(__file__).resolve().parent / 'data_law1_adhoc'


@pytest.fixture(scope='module')
def fe56_endf_dict():
    """Fe-56 from PR #30's ad-hoc corpus. Fetched on demand via
    tests/data_law1_adhoc/fetch.sh, so this fixture skips gracefully
    when the file is not present."""
    fn = ADHOC_DATA_DIR / 'tendl21_n_Fe-56.endf'
    if not fn.exists():
        pytest.skip(
            f'{fn.name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh` to populate the corpus'
        )
    parser = EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    )
    return parser.parsefile(fn)


def test_get_ejectile_raises_informative_valueerror_on_multi_ejectile_mt(
    fe56_endf_dict,
):
    """MT 112 = (n,p a) has two ejectiles, neither is a neutron. The
    old code silently asserted; the new code raises a ValueError
    naming the ejectiles and the MT so the caller can react."""
    with pytest.raises(ValueError, match=r"MT=112.*multiple ejectiles.*'p'.*'a'"):
        get_ejectile(fe56_endf_dict, 112)


def test_get_ZAP_raises_informative_valueerror_on_multi_ejectile_mt(
    fe56_endf_dict,
):
    """get_ZAP wraps get_ejectile, same failure mode."""
    with pytest.raises(ValueError, match=r"MT=112.*multiple ejectiles"):
        get_ZAP(fe56_endf_dict, 112)


def test_is_zap_consistent_handles_multi_ejectile_mt(fe56_endf_dict):
    """is_zap_consistent must correctly identify each ejectile of a
    multi-ejectile MT (used to crash with AssertionError instead).
    Uses the reaction-string ejectile table rather than the single-
    ejectile get_ZAP for this check."""
    # MT 112 = (n,p a): emits proton, alpha, and (via cascades) gamma.
    # Neutrons and deuterons are not among the ejectiles.
    assert is_zap_consistent(fe56_endf_dict, 112, PARTICLE_ZAP['p']) is True
    assert is_zap_consistent(fe56_endf_dict, 112, PARTICLE_ZAP['a']) is True
    assert is_zap_consistent(fe56_endf_dict, 112, PARTICLE_ZAP['g']) is True
    assert is_zap_consistent(fe56_endf_dict, 112, PARTICLE_ZAP['n']) is False
    assert is_zap_consistent(fe56_endf_dict, 112, PARTICLE_ZAP['d']) is False


def test_is_zap_consistent_single_ejectile_still_works(fe56_endf_dict):
    """Regression: for single-ejectile MTs the answer is unchanged.
    MT 2 elastic emits one neutron; nothing else."""
    assert is_zap_consistent(fe56_endf_dict, 2, PARTICLE_ZAP['n']) is True
    # Gamma is always accepted by the "gammas accompany many reactions"
    # special case.
    assert is_zap_consistent(fe56_endf_dict, 2, PARTICLE_ZAP['g']) is True
    # Protons and other charged particles are not emitted by elastic.
    assert is_zap_consistent(fe56_endf_dict, 2, PARTICLE_ZAP['p']) is False


def test_is_zap_consistent_neutron_first_multi_ejectile(fe56_endf_dict):
    """For MTs like MT 16 = (n,2n) with a single ejectile species,
    is_zap_consistent still fires correctly."""
    # MT 16 emits 2 neutrons.
    assert is_zap_consistent(fe56_endf_dict, 16, PARTICLE_ZAP['n']) is True
    assert is_zap_consistent(fe56_endf_dict, 16, PARTICLE_ZAP['g']) is True
    assert is_zap_consistent(fe56_endf_dict, 16, PARTICLE_ZAP['p']) is False


def test_get_ejectile_single_ejectile_unchanged(fe56_endf_dict):
    """Regression: for MTs the function used to handle without
    asserting, get_ejectile returns the ejectile as before."""
    assert get_ejectile(fe56_endf_dict, 2) == 'n'
    assert get_ejectile(fe56_endf_dict, 16) == 'n'
    assert get_ejectile(fe56_endf_dict, 18) == 'n'  # fission special-case


def test_is_zap_consistent_unknown_mt_is_permissive(fe56_endf_dict):
    """For an MT not tabulated in the reaction table (unusual, but
    happens for some vendor-specific MTs), the function should be
    permissive (return True) so the caller can attempt reconstruction
    and fail with a domain-specific error if data really is missing.
    """
    # MT 42 doesn't appear in Fe-56 but is a valid reaction MT. If
    # is_zap_consistent doesn't know the ejectile list, it should
    # return True permissively rather than throw. Use a very large
    # MT number instead to ensure "unknown" behaviour.
    unknown_mt = 995
    assert is_zap_consistent(fe56_endf_dict, unknown_mt, PARTICLE_ZAP['n']) is True
