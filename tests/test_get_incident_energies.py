"""Tests for `get_incident_energies` in `endf_userpy.quantities`
(issue #74).

The previous implementation called `quant_mt_zap.get_incident_energies`
which does not exist on that module -- the helper lives on
`mfsec_interpretation.mf3_interpretation.get_incident_energies(mt)`.
Every call raised `AttributeError`, i.e. a documented public API
entry point was 100 % broken on every file.

These tests pin:

1. **Dispatch works** on the corpus files: for common reactions the
   call returns a non-empty ndarray.
2. **Physical shape** of the result: monotonically increasing,
   first entry non-negative.
3. **Bit-for-bit against `mf3interp.get_incident_energies`** for the
   single-MT case, since the public API is a union across MTs.
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_incident_energies
from endf_userpy.mfsec_interpretation import mf3_interpretation as mf3interp


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
def al27_endfb81():
    return _load('endfb81_n_Al-27.endf')


@pytest.fixture(scope='module')
def h1_endfb81():
    return _load('endfb81_n_H-1.endf')


@pytest.fixture(scope='module')
def fe56_tendl21():
    return _load('tendl21_n_Fe-56.endf')


# ============================================================
# Dispatch works: no AttributeError on any admitted reaction.
# ============================================================


@pytest.mark.parametrize('reaction', ['(n,total)', '(n,n_0)', '(n,g)'])
def test_dispatch_works_on_al27(al27_endfb81, reaction):
    """Regression guard for #74: the public API must not raise
    AttributeError on any reaction admitted by the file."""
    r = get_incident_energies(al27_endfb81, reaction)
    assert r.ndim == 1
    assert len(r) > 0


@pytest.mark.parametrize('reaction', ['(n,total)', '(n,n_0)', '(n,g)'])
def test_dispatch_works_on_h1(h1_endfb81, reaction):
    """H-1 is the smallest corpus file; a broad dispatch guard."""
    r = get_incident_energies(h1_endfb81, reaction)
    assert r.ndim == 1 and len(r) > 0


@pytest.mark.parametrize('reaction', ['(n,total)', '(n,n_0)', '(n,g)', '(n,2n)'])
def test_dispatch_works_on_fe56(fe56_tendl21, reaction):
    r = get_incident_energies(fe56_tendl21, reaction)
    assert r.ndim == 1 and len(r) > 0


# ============================================================
# Physical shape invariants.
# ============================================================


def test_monotone_and_nonnegative(fe56_tendl21):
    """Any incident-energy mesh must be strictly monotone and
    non-negative. Both are guaranteed by `np.unique` +
    non-negative-source-mesh, but pinning them here catches any
    future refactor that breaks either property."""
    for reaction in ('(n,total)', '(n,n_0)', '(n,g)', '(n,2n)'):
        r = get_incident_energies(fe56_tendl21, reaction)
        assert r[0] >= 0.0
        assert np.all(np.diff(r) > 0.0), (
            f'{reaction} mesh not strictly monotone'
        )


# ============================================================
# Bit-for-bit against the underlying mf3 helper for single-MT
# reactions (where the dispatcher's union across MTs collapses to
# a single mesh).
# ============================================================


def test_matches_mf3_single_mt_case(h1_endfb81):
    """For a reaction that resolves to exactly one MT and no
    admitted siblings, the top-level result must equal the leaf
    `mf3interp.get_incident_energies(mt)` result. Uses H-1 (n,g)
    which is MT=102 only."""
    from endf_userpy.primitives.reactions import (
        translate_reaction_string_to_mt,
    )
    mt = translate_reaction_string_to_mt('(n,g)')
    api = get_incident_energies(h1_endfb81, '(n,g)')
    leaf = mf3interp.get_incident_energies(h1_endfb81, mt)
    # Public API returns a unique-sorted union; leaf returns a
    # single mesh. The union of one mesh is that mesh sorted.
    np.testing.assert_allclose(api, np.unique(leaf), rtol=0, atol=0)
