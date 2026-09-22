"""Bit-for-bit equivalence between the pure-Python MF6 LAW=1
continuum reconstruction and the Fortran-backed reference.

Same pattern as the LAW=6 / LAW=2 / LAW=7 / LAW=1-discrete-lines
equivalence tests.

Coverage:
- ENDF/B-VIII.1 Al-27 MT=5 / 16 / 91 / 108 / 111 / 117 / 649
  (LAW=1 LANG=2 Kalbach-Mann; various neutron / alpha / proton /
  deuteron / triton ejectiles) -- adhoc corpus.
- Be-9 MT=701 (photon LAW=1 LANG=1) -- shipped corpus.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_subsecs as mf6py,
    mf6_interpretation_subsecs_fort as mf6fort,
    mf6_law1_helpers,
)

from _corpus import resolve_al27


DATA_DIR = Path(__file__).resolve().parent / 'data'


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip(
            'Al-27 ENDF file not available (set AL27_ENDF, run '
            'tests/data_law1_adhoc/fetch.sh, or place the file at '
            'tests/data_law1_adhoc/endfb81_n_Al-27.endf)'
        )
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def be9_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        DATA_DIR / 'n-004_Be_009.endf',
    )


@pytest.mark.parametrize('mt,zap_label', [
    (5, 'neutron continuum (n, anything) LANG=2 Kalbach'),
    (16, '(n, 2n) neutron ejectile LANG=2 Kalbach'),
    (91, '(n, n\') continuum LANG=2 Kalbach'),
    (108, '(n, 2alpha) alpha ejectile LANG=2 Kalbach'),
    (111, '(n, 2p) proton ejectile LANG=2 Kalbach'),
    (117, '(n, dalpha) deuteron ejectile LANG=2 Kalbach'),
    (649, '(n, p) proton ejectile LANG=2 Kalbach'),
])
def test_law1_continuum_python_fortran_equivalence_al27(al27_endf_dict, mt, zap_label):
    """Al-27 MF6 LAW=1 continuum reconstruction across a
    representative set of MTs (neutron / alpha / proton /
    deuteron / triton ejectiles, all LANG=2 Kalbach-Mann). Pure
    Python must match Fortran bit-for-bit on the reachable (E, E',
    mu) grid.

    Query energies are chosen inside the section's tabulated E
    range (E_first * 1.1 to E_last * 0.9) to avoid the
    pad-outside-values decorator zero-filling everything.
    """
    ss = al27_endf_dict[6][mt]['subsection'][1]
    E_first = list(ss['E'].values())[0]
    E_last = list(ss['E'].values())[-1]
    e_in = np.linspace(E_first * 1.1, min(E_last * 0.9, 2.0e7), 3)
    e_out = np.linspace(0.0, min(E_last, 2.0e7), 6)
    mu = np.linspace(-0.9, 0.9, 5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        res_py = np.asarray(mf6py.get_dist2d_from_subsec_law1(
            al27_endf_dict, mt, 1, e_in, e_out, mu, True,
        ))
        res_fort = np.asarray(mf6fort.get_dist2d_from_subsec_law1(
            al27_endf_dict, mt, 1, e_in, e_out, mu, True,
        ))
    assert res_py.shape == (len(e_in), len(e_out), len(mu))
    assert res_py.shape == res_fort.shape
    np.testing.assert_allclose(
        res_py, res_fort, rtol=1e-11, atol=1e-14, equal_nan=True,
    )


def test_law1_continuum_python_fortran_equivalence_be9_photon(be9_endf_dict):
    """Be-9 MT=701 (photon production) MF6 LAW=1 LANG=1 continuum:
    tests the Legendre-in-mu path (LANG=1) at photon ZAP=0."""
    ss = be9_endf_dict[6][701]['subsection'][3]
    E_first = list(ss['E'].values())[0]
    E_last = list(ss['E'].values())[-1]
    e_in = np.linspace(E_first * 1.1, E_last * 0.9, 3)
    e_out = np.linspace(0.0, 2.0e7, 6)
    mu = np.linspace(-0.9, 0.9, 5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        res_py = np.asarray(mf6py.get_dist2d_from_subsec_law1(
            be9_endf_dict, 701, 3, e_in, e_out, mu, True,
        ))
        res_fort = np.asarray(mf6fort.get_dist2d_from_subsec_law1(
            be9_endf_dict, 701, 3, e_in, e_out, mu, True,
        ))
    np.testing.assert_allclose(
        res_py, res_fort, rtol=1e-11, atol=1e-14, equal_nan=True,
    )


def test_mf6lab2cm_identity_for_lct1_or_heavy_ejectile():
    """Forward LAB->CM map: LCT=1 (LAB always) and LCT=3 with
    heavy ejectile (AWP>=4) return the identity. LCT=3 with light
    ejectile (AWP<4) triggers the CM shift. Companion pin to the
    inverse mf6cm2lab_disc branch selection."""
    awr, awi = 55.454, 1.0
    e, ep, u = 1.0e7, 5.0e5, 0.3
    tp, w, dinv = mf6_law1_helpers.mf6lab2cm(awr, awi, 1.0, 1, e, ep, u)
    assert tp == ep and w == u and dinv == 1.0
    tp, w, dinv = mf6_law1_helpers.mf6lab2cm(awr, awi, 4.0, 3, e, ep, u)
    assert tp == ep and w == u and dinv == 1.0
    tp, w, dinv = mf6_law1_helpers.mf6lab2cm(awr, awi, 1.0, 3, e, ep, u)
    assert not (tp == ep and w == u and dinv == 1.0)


def test_mf6lab2cm_returns_zero_dinv_at_ep_zero():
    """On the CM branch (LCT=2), ep <= 0 signals no physical
    outgoing energy: return dinv=0 sentinel."""
    tp, w, dinv = mf6_law1_helpers.mf6lab2cm(
        awr=1.0, awi=1.0, awp=1.0, lct=2, e=1.0e7, ep=0.0, u=0.5,
    )
    assert dinv == 0.0
    # tp = c0^2 * e = 0.25 * 1e7 = 2.5e6 as the sentinel
    assert tp == pytest.approx(2.5e6)
    assert w == -1.0
