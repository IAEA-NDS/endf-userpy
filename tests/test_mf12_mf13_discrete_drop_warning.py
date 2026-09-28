"""UserWarning when the unbroadened gamma DDX / dxs_dE dispatchers
drop MF12 or MF13 discrete photon-line content (issue #266).

Discrete gamma lines are Dirac deltas on the E_out axis. On the
caller's finite E_out grid they integrate to zero almost
everywhere, so the unbroadened path skips them; the warning names
the affected MTs and points to ``broadening=sigma_eV`` for a
physically meaningful kernel or ``broadening=0`` for an explicit
silence.

Uses ``tests/data/n-001_H_002.endf`` (single MF12 discrete gamma
at Eg=6.251 MeV in MT=102).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_ddxs,
    get_particle_production_dxs_dE,
)


@pytest.fixture(scope='module')
def h2_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        'tests/data/n-001_H_002.endf',
    )


def test_dxs_dE_warns_and_names_mt_when_broadening_none(h2_endf_dict):
    ein = np.array([1e6])
    eouts = np.linspace(0.0, 1e7, 51)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        get_particle_production_dxs_dE(
            h2_endf_dict, '(n,g)', 'g', ein, eouts, broadening=None,
        )
    msgs = [
        str(w.message) for w in caught
        if issubclass(w.category, UserWarning)
        and 'MF12/MF13' in str(w.message)
    ]
    assert len(msgs) == 1, f'expected one MF12/MF13 warning, got {msgs}'
    assert 'MT=102' in msgs[0]
    assert 'broadening=0' in msgs[0]


def test_ddxs_warns_and_names_mt_when_broadening_none(h2_endf_dict):
    ein = np.array([1e6])
    eouts = np.linspace(0.0, 1e7, 51)
    mus = np.array([0.0, 0.5])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        get_particle_production_ddxs(
            h2_endf_dict, '(n,g)', 'g', ein, eouts, mus, broadening=None,
        )
    msgs = [
        str(w.message) for w in caught
        if issubclass(w.category, UserWarning)
        and 'MF12/MF13' in str(w.message)
    ]
    assert len(msgs) == 1, f'expected one MF12/MF13 warning, got {msgs}'
    assert 'MT=102' in msgs[0]


def test_broadening_zero_silences_the_warning(h2_endf_dict):
    ein = np.array([1e6])
    eouts = np.linspace(0.0, 1e7, 51)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r1 = get_particle_production_dxs_dE(
            h2_endf_dict, '(n,g)', 'g', ein, eouts, broadening=0,
        )
    msgs = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and ('MF12/MF13' in str(w.message)
             or 'MF6/LAW=1' in str(w.message)
             or 'discrete two-body' in str(w.message))
    ]
    assert msgs == [], (
        f'broadening=0 should suppress the discrete-drop warnings, '
        f'got: {[str(w.message) for w in msgs]}'
    )
    # Numeric result: continuum only (no discrete-line placement).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r_none = get_particle_production_dxs_dE(
            h2_endf_dict, '(n,g)', 'g', ein, eouts, broadening=None,
        )
    np.testing.assert_allclose(np.asarray(r1), np.asarray(r_none),
                                rtol=0.0, atol=0.0)


def test_broadening_zero_silences_the_ddx_warning(h2_endf_dict):
    ein = np.array([1e6])
    eouts = np.linspace(0.0, 1e7, 51)
    mus = np.array([0.0, 0.5])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        get_particle_production_ddxs(
            h2_endf_dict, '(n,g)', 'g', ein, eouts, mus, broadening=0,
        )
    msgs = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and ('MF12/MF13' in str(w.message)
             or 'discrete two-body' in str(w.message))
    ]
    assert msgs == [], (
        f'broadening=0 should suppress the discrete-drop warnings, '
        f'got: {[str(w.message) for w in msgs]}'
    )


def test_negative_broadening_still_rejected():
    """Sanity check: sigma<0 is meaningless and still raises. Only
    sigma>=0 is accepted, with sigma=0 being the new silence hatch.
    """
    from endf_userpy.quantities import _normalize_broadening
    with pytest.raises(ValueError, match='non-negative'):
        _normalize_broadening(-1.0)


def test_broadening_zero_numeric_identical_to_none(h2_endf_dict):
    ein = np.array([1e6])
    eouts = np.linspace(0.0, 1e7, 51)
    mus = np.array([-0.5, 0.0, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r_none = get_particle_production_ddxs(
            h2_endf_dict, '(n,g)', 'g', ein, eouts, mus, broadening=None,
        )
        r_zero = get_particle_production_ddxs(
            h2_endf_dict, '(n,g)', 'g', ein, eouts, mus, broadening=0,
        )
    np.testing.assert_array_equal(np.asarray(r_zero), np.asarray(r_none))
