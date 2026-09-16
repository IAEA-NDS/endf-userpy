"""Tests for the per-MT MF6 gate in `get_emission_energies` (issue #77).

The top-level `selectors.contains_zap` admits any MT whose gamma
yields are declared in MF12 or MF13 (issue #29 rationale) even if
there is no MF6 subsection for that ZAP. `get_emission_energies`
then dispatched each admitted MT to `mf6_interp.get_emission_energies`,
which walks only MF6 and raised ``IndexError: subsection with
ZAP=0.0 not found in MF6/MT{mt}`` on the MT12/MF13-only gammas.

The fix filters `select_mts` with `has_mf6_mt` + `mf6_help.contains_zap`
so only MTs whose ZAP is actually in MF6 get dispatched. MTs whose
gamma content lives in MF12/MF13 contribute nothing to the mesh
today; wiring a full MF12/MF13 emission-mesh walker is separate
work tracked as issue #82 (item D7).
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_emission_energies


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


# ============================================================
# Regression: files that hit IndexError on `(n,total)+g` pre-fix.
# ============================================================


@pytest.mark.parametrize('fn', [
    'endfb81_n_Ni-58.endf',
    'endfb81_n_Pu-239.endf',
    'jeff40_n_Cu-63.endf',
])
def test_ntotal_gamma_does_not_raise_on_mf12_gamma_files(fn):
    """The three corpus files identified in the audit: MT51 has an
    MF6 neutron subsection but no gamma subsection; gamma yields
    are declared in MF12. Pre-fix the leaf raised IndexError; the
    new per-MT MF6 gate skips MT51 for the gamma query and the
    remaining admitted MTs contribute their MF6-carried gamma
    meshes."""
    endf = _load(fn)
    r = get_emission_energies(endf, '(n,total)', 'g')
    assert isinstance(r, np.ndarray)
    assert r.dtype == np.float64
    assert np.all(np.isfinite(r))
    assert np.all(np.diff(r) >= 0.0)


# ============================================================
# Non-regression: existing working queries still work with the
# same numeric answer.
# ============================================================


def test_ntotal_neutron_unchanged_on_fe56(fe56_tendl21):
    r = get_emission_energies(fe56_tendl21, '(n,2n)', 'n')
    assert len(r) > 0
    assert r[0] >= 0.0
    assert r[-1] > 1e7


@pytest.fixture(scope='module')
def fe56_tendl21():
    return _load('tendl21_n_Fe-56.endf')


