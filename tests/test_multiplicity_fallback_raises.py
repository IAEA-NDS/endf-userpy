"""Tests for the multiplicity-fallback ValueError in
`compute_yields` (issue #104 / audit D4).

`endf_userpy.quantities_mt_zap.quantities.compute_yields` has a
fallback branch for MTs that don't route through MF6 or the
MF12/MF13 gamma path: it looks up the ejectile multiplicity in
`reactions.REACTION_DICT` and constructs a constant-yield array
from it. When `reactions.get_multiplicity_for_zap` returned
`None` (unknown MT, mangled ejectile string, or an invalid level
suffix), the pre-fix code did
``np.full(len(energies_in), None, dtype=float)`` which produced
a silent NaN array -- the invariant violation propagated through
the whole cumulative-sum pipeline into the user's answer.

Fix: raise a clean `ValueError` naming the MT and ZAP. The path
is theoretically unreachable via the normal admission gates
(`selectors.contains_zap` filters unknown MTs upstream), so the
new raise is an internal-invariant check that fires only when
someone bypasses the admission gate directly (e.g., calling
`compute_yields` from a custom driver).

These tests use `unittest.mock.patch` to force the None branch --
no real ENDF file in the corpus triggers it via the normal API.
"""
from unittest import mock
import numpy as np
import pytest

from endf_userpy.quantities_mt_zap import quantities as qmt


def _minimal_endf():
    """Smallest endf_dict shape compute_yields' fallback branch
    accepts. The fallback branch only reaches `get_projectile`,
    which needs `endf_dict[1][451]['NSUB']` -- 10 for neutron."""
    return {
        1: {451: {'NSUB': 10}},
    }


def test_silent_nan_replaced_with_valueerror():
    """When `reactions.get_multiplicity_for_zap` returns `None`
    (e.g. because the MT isn't in the reactions table), the
    fallback branch must raise ValueError rather than emit a
    silent NaN array."""
    endf = _minimal_endf()
    einc = np.linspace(1e5, 1e7, 5)
    # Force the None return from get_multiplicity_for_zap.
    with mock.patch(
        'endf_userpy.quantities_mt_zap.quantities.reaction'
        '.get_multiplicity_for_zap',
        return_value=None,
    ):
        with pytest.raises(ValueError, match='Cannot derive multiplicity'):
            qmt.compute_yields(endf, mt=9999, zap=1.0, energies_in=einc)


def test_message_names_mt_and_zap():
    """The raised message names the offending MT and ZAP."""
    endf = _minimal_endf()
    einc = np.linspace(1e5, 1e7, 5)
    with mock.patch(
        'endf_userpy.quantities_mt_zap.quantities.reaction'
        '.get_multiplicity_for_zap',
        return_value=None,
    ):
        with pytest.raises(ValueError) as exc_info:
            qmt.compute_yields(endf, mt=9999, zap=1002.0, energies_in=einc)
    msg = str(exc_info.value)
    assert 'MT=9999' in msg
    assert 'ZAP=1002' in msg


def test_normal_multiplicity_returns_yields():
    """Non-regression: a valid (MT, ZAP) with a known multiplicity
    still constructs and returns a yield array of the requested
    length. Uses MT=16 (n,2n) with ZAP=1 (neutron), which has
    multiplicity 2 in the reactions table."""
    endf = _minimal_endf()
    einc = np.linspace(1e5, 1e7, 5)
    yields = qmt.compute_yields(endf, mt=16, zap=1.0, energies_in=einc)
    assert yields.shape == (5,)
    assert np.all(yields == 2.0)


def test_zero_multiplicity_still_returns_yields():
    """Non-regression: `get_multiplicity_for_zap` returns 0 (not
    None) for a valid MT with an ejectile the (MT, ZAP) pair
    doesn't contain. That case must produce an all-zero array,
    NOT hit the new raise."""
    endf = _minimal_endf()
    einc = np.linspace(1e5, 1e7, 5)
    # MT=2 (elastic) has multiplicity 1 for neutron ejectile; ask
    # for gamma ejectile (ZAP=0) -- multiplicity is 0, not None.
    yields = qmt.compute_yields(endf, mt=2, zap=0.0, energies_in=einc)
    assert yields.shape == (5,)
    assert np.all(yields == 0.0)
