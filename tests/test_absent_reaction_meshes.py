"""Tests for the empty-mesh return path of `get_incident_energies`
and `get_emission_energies` (issue #75).

Both functions used to unconditionally call
``np.concatenate(energy_meshes)``. When the requested reaction is
not present in the file (e.g. ``(n,f)`` on H-1, ``(n,p)`` on H-2),
or when the requested ejectile is not declared by any admitted MT
(e.g. ``get_emission_energies("(n,g)", "n")``), the mesh list is
empty and the call raised
``ValueError: need at least one array to concatenate`` -- a cryptic
message that hid a plain "reaction absent" answer behind an
exception.

Both functions now return an empty float ndarray in that case, so
introspection code can check ``len(result) == 0`` without a
try/except.
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_incident_energies,
    get_emission_energies,
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
def h1_endfb81():
    return _load('endfb81_n_H-1.endf')


@pytest.fixture(scope='module')
def h2_jeff40():
    return _load('jeff40_n_H-2.endf')


# ============================================================
# get_incident_energies: absent reaction returns empty ndarray.
# ============================================================


def test_get_incident_energies_absent_reaction_returns_empty_h1(h1_endfb81):
    """H-1 is not fissile, so ``(n,f)`` resolves to no MT."""
    r = get_incident_energies(h1_endfb81, '(n,f)')
    assert isinstance(r, np.ndarray)
    assert r.dtype == np.float64
    assert r.shape == (0,)


def test_get_incident_energies_absent_reaction_returns_empty_h2(h2_jeff40):
    r = get_incident_energies(h2_jeff40, '(n,p)')
    assert isinstance(r, np.ndarray)
    assert r.shape == (0,)


def test_get_incident_energies_present_reaction_unchanged(h1_endfb81):
    """Guard against a regression that would collapse the non-empty
    case to empty. ``(n,total)`` is present on every file."""
    r = get_incident_energies(h1_endfb81, '(n,total)')
    assert len(r) > 0


# ============================================================
# get_emission_energies: absent (reaction, particle) returns empty.
# ============================================================


def test_get_emission_energies_absent_particle_returns_empty_h1(h1_endfb81):
    """H-1 (n,g) exists but the neutron ejectile is not declared in
    MF6 for that MT -- prior behaviour was to raise; now returns
    empty."""
    r = get_emission_energies(h1_endfb81, '(n,g)', 'n')
    assert isinstance(r, np.ndarray)
    assert r.dtype == np.float64
    assert r.shape == (0,)


def test_get_emission_energies_absent_reaction_returns_empty_h2(h2_jeff40):
    r = get_emission_energies(h2_jeff40, '(n,p)', 'p')
    assert isinstance(r, np.ndarray)
    assert r.shape == (0,)


