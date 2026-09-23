"""MF12 LO=2 transition-probability -> yield: Python port
equivalence with the Fortran reference.

The port lives in :mod:`mf12_trans2yield_kernel` and is bit-
identical to the Fortran ``trans2yield`` / ``init_trans2yield``
because the algorithm is deterministic integer bookkeeping over
small-shape matrices (max 60 discrete levels) with no floating
adaptive iteration.

Pins:
- init state (``ee``, ``r``, ``a``) matches Fortran exactly.
- For every MT in a discrete inelastic series (Fe-56 (n,n')
  MT=51..80), the returned (photon_energy, level_energy, yield)
  arrays match Fortran bit-for-bit.
- Running state after walking the whole series matches Fortran
  exactly.
- Downstream API (compute_photon_yields_from_transition_probabilities)
  is unaffected by the port.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('endf_userpy.fortran.endf6')

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf12_interpretation as mf12_pub,
    mf12_interpretation_helpers as py_h,
    mf12_interpretation_helpers_fort as fort_h,
    mf12_trans2yield_kernel as _kernel,
)


ADHOC = Path(__file__).resolve().parent / 'data_law1_adhoc'


@pytest.fixture(scope='module')
def fe56_endf_dict():
    fn = ADHOC / 'tendl21_n_Fe-56.endf'
    if not fn.exists():
        pytest.skip(
            'tendl21_n_Fe-56.endf not present; run '
            '`bash tests/data_law1_adhoc/fetch.sh`'
        )
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(fn)


def test_init_trans2yield_matches_fortran(fe56_endf_dict):
    py_mts, py_state = py_h.init_trans2yield(fe56_endf_dict, 51)
    ft_mts, ft_state = fort_h.init_trans2yield_fort_wrapper(
        fe56_endf_dict, 51,
    )
    assert py_mts == ft_mts, 'available MTs mismatch'
    np.testing.assert_array_equal(py_state['ee'], ft_state['ee'])
    np.testing.assert_array_equal(py_state['r'], ft_state['r'])
    np.testing.assert_array_equal(py_state['a'], ft_state['a'])


def test_trans2yield_matches_fortran_full_series_fe56(fe56_endf_dict):
    """Walk the entire (n, n') discrete series and check every MT's
    photon lines and running state match the Fortran reference."""
    py_mts, py_state = py_h.init_trans2yield(fe56_endf_dict, 51)
    ft_mts, ft_state = fort_h.init_trans2yield_fort_wrapper(
        fe56_endf_dict, 51,
    )
    for mt in py_mts:
        py_out = py_h.trans2yield(fe56_endf_dict, mt, py_state)
        ft_out = fort_h.trans2yield_fort_wrapper(
            fe56_endf_dict, mt, ft_state,
        )
        assert py_out['photon_energy'].shape == ft_out['photon_energy'].shape
        np.testing.assert_array_equal(
            py_out['photon_energy'], ft_out['photon_energy'],
        )
        np.testing.assert_array_equal(
            py_out['level_energy'], ft_out['level_energy'],
        )
        np.testing.assert_array_equal(
            py_out['photon_yield'], ft_out['photon_yield'],
        )
        # And the running state must stay in lockstep.
        np.testing.assert_array_equal(py_state['ee'], ft_state['ee'])
        np.testing.assert_array_equal(py_state['r'], ft_state['r'])
        np.testing.assert_array_equal(py_state['a'], ft_state['a'])


def test_public_api_photon_yields_unchanged(fe56_endf_dict):
    """The public
    :func:`compute_photon_yields_from_transition_probabilities`
    (which composes init + trans2yield internally) returns the same
    result as computing each MT directly through the Fortran
    reference wrappers."""
    mts_to_query = [51, 60, 70, 80]
    py_all = mf12_pub.compute_photon_yields_from_transition_probabilities(
        fe56_endf_dict, mts_to_query,
    )
    ft_mts, ft_state = fort_h.init_trans2yield_fort_wrapper(
        fe56_endf_dict, 51,
    )
    ref = {}
    for mt in ft_mts:
        out = fort_h.trans2yield_fort_wrapper(
            fe56_endf_dict, mt, ft_state,
        )
        if mt in mts_to_query:
            ref[mt] = out
        if mt == max(mts_to_query):
            break
    for mt in mts_to_query:
        np.testing.assert_array_equal(
            py_all[mt]['photon_energy'], ref[mt]['photon_energy'],
        )
        np.testing.assert_array_equal(
            py_all[mt]['photon_yield'], ref[mt]['photon_yield'],
        )


def test_kernel_series_mt0_rejects_non_discrete():
    """MT values outside a discrete-inelastic series raise
    ``ValueError`` (matches the Fortran fatal-stop)."""
    with pytest.raises(ValueError, match='not a discrete'):
        _kernel._series_mt0(3)  # radiative capture, not discrete
    with pytest.raises(ValueError, match='not a discrete'):
        _kernel._series_mt0(50)  # ground-state boundary, excluded
    with pytest.raises(ValueError, match='not a discrete'):
        _kernel._series_mt0(91)  # continuum inelastic (n,n'), excluded


def test_kernel_yield_sums_to_expected_for_synthetic_two_level():
    """Synthetic 2-level system: single excited level -> ground.
    The (n, n') MT=51 populating the first excited state must
    produce exactly one photon line with yield 1 and energy equal
    to the level energy.
    """
    elis = 0.0
    qm = [-1.0e6]  # first excited level at 1 MeV above ground
    qi = [-1.0e6]
    ee, r, a = _kernel.init_trans2yield(elis, qm, qi, maxlevel=10)
    # MT=51, first excited: one transition to ground (esi=0), tp=1
    out = _kernel.trans2yield(
        mt=51, esns=1.0e6, esi=[0.0], tp=[1.0], gp=[1.0],
        ee=ee, r=r, a=a,
    )
    assert out['photon_energy'].shape == (1,)
    np.testing.assert_allclose(out['photon_energy'][0], 1.0e6)
    np.testing.assert_allclose(out['level_energy'][0], 1.0e6)
    np.testing.assert_allclose(out['photon_yield'][0], 1.0)


def test_kernel_yield_cascade_three_level():
    """3-level cascade: MT=52 populates level 2, which has 50%
    branching to level 1 (which decays to ground, y=1) and 50% to
    ground directly. Expect two photon lines from MT=52 with yields
    that sum to 1.5 (one 2->1 photon of yield 0.5, and one 2->0
    photon of yield 0.5, plus one 1->0 photon of yield 0.5 from the
    cascade). Order by descending photon energy.
    """
    elis = 0.0
    qm = [-1.0e6, -3.0e6]  # levels at 1 MeV, 3 MeV
    qi = [-1.0e6, -3.0e6]
    ee, r, a = _kernel.init_trans2yield(elis, qm, qi, maxlevel=10)
    # MT=51 (populate level 1, direct decay to ground)
    out51 = _kernel.trans2yield(
        mt=51, esns=1.0e6, esi=[0.0], tp=[1.0], gp=[1.0],
        ee=ee, r=r, a=a,
    )
    assert out51['photon_energy'].shape == (1,)
    # MT=52 (populate level 2, 50% -> level 1, 50% -> ground)
    out52 = _kernel.trans2yield(
        mt=52, esns=3.0e6, esi=[0.0, 1.0e6], tp=[0.5, 0.5], gp=[1.0, 1.0],
        ee=ee, r=r, a=a,
    )
    # Three photon lines expected from MT=52: 3->0 (3 MeV, y=0.5),
    # 3->1 (2 MeV, y=0.5), 1->0 cascade (1 MeV, y=0.5).
    assert out52['photon_energy'].shape == (3,)
    # Sorted descending in photon energy: 3, 2, 1 MeV.
    np.testing.assert_allclose(
        out52['photon_energy'], [3.0e6, 2.0e6, 1.0e6],
    )
    np.testing.assert_allclose(
        out52['photon_yield'], [0.5, 0.5, 0.5],
    )
    # Total yield sums to 1.5 (one 3-MeV emission per event; plus
    # one 2-MeV cascade with 50% probability, followed by a 1-MeV
    # cascade also with 50% probability = 1.5 photons per event).
    np.testing.assert_allclose(out52['photon_yield'].sum(), 1.5)
