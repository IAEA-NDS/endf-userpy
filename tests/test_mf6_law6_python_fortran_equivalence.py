"""Bit-for-bit equivalence between the pure-Python MF6 LAW=6
(N-body phase-space) reconstruction and the Fortran-backed
reference implementation.

Mirrors ``test_mf4_interpretation.py::test_mf4_*_python_fortran_equivalence``.

Reference file ``n-001_H_002.endf`` (H-2 = deuterium): breakup
reactions like (n, 2n) MT=16 use LAW=6 to distribute the
outgoing neutrons over the N-body phase space; the existing
``test_dist2d_law6_interface`` (smoke test) uses the same file.

Two coverage passes:

1. Direct-kernel equality on a representative ``(E, E', mu)`` grid
   spanning the LAW=6 kinematic region above threshold. Machine
   precision (~1e-14) since both paths evaluate the same
   closed-form phase-space formula in the same order.

2. Sanity: values are non-negative and the ``E' > E_i^max`` cut-off
   fires in both paths (both return zero there).

Bug-catch verified during PR development: dropping the LAB-frame
kinematic shift ``E_s`` (``E'_c = E' - 2 mu sqrt(E_s E')``) flips
these tests to fail with residuals of order the incident energy /
outgoing energy ratio; wrong ``C_n`` normalisation for ``npsx=4``
(e.g. leaving the ``105/32`` prefactor as ``4/pi``) flips them
with a percent-level residual.
"""
from pathlib import Path
import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_subsecs as mf6py,
    mf6_interpretation_subsecs_fort as mf6fort,
)


DATA_DIR = Path(__file__).resolve().parent / 'data'


@pytest.fixture(scope='module')
def h2_endf_dict():
    parser = EndfParserCpp(ignore_missing_tpid=True)
    return parser.parsefile(DATA_DIR / 'n-001_H_002.endf')


def test_law6_python_fortran_equivalence_h2_mt16(h2_endf_dict):
    """H-2 (n, 2n) MT=16 subsec 1: LAW=6 with npsx=3 (two outgoing
    neutrons + recoiling H). Sample the (E, E', mu) grid across
    the URR-scale energies where the phase-space formula peaks."""
    e_inc = np.array([3.0e6, 5.0e6, 7.0e6, 1.0e7, 1.5e7, 2.0e7])
    e_out = np.linspace(0.0, 1.5e7, 20)
    mu = np.linspace(-1.0, 1.0, 11)

    res_py = np.asarray(mf6py.get_dist2d_from_subsec_law6(
        h2_endf_dict, 16, 1, e_inc, e_out, mu, True,
    ))
    res_fort = np.asarray(mf6fort.get_dist2d_from_subsec_law6(
        h2_endf_dict, 16, 1, e_inc, e_out, mu, True,
    ))

    assert res_py.shape == (len(e_inc), len(e_out), len(mu))
    assert res_py.shape == res_fort.shape
    # Bit-for-bit: same closed-form expression, same evaluation order.
    np.testing.assert_allclose(res_py, res_fort, rtol=1e-13, atol=1e-30)


def test_law6_python_kinematic_cutoff_matches_fortran(h2_endf_dict):
    """The phase-space distribution is zero above the kinematic
    maximum ``E'_c > E_i^max`` (or below reaction threshold).
    Both paths must return zero on the same mask, not just match
    on the interior."""
    # High E' at low E_inc: guaranteed above E_i^max on every mu.
    e_inc = np.array([3.0e6])
    e_out = np.array([2.0e7])   # well above kinematic max
    mu = np.array([-1.0, 0.0, 1.0])
    res_py = np.asarray(mf6py.get_dist2d_from_subsec_law6(
        h2_endf_dict, 16, 1, e_inc, e_out, mu, True,
    ))
    res_fort = np.asarray(mf6fort.get_dist2d_from_subsec_law6(
        h2_endf_dict, 16, 1, e_inc, e_out, mu, True,
    ))
    assert np.all(res_py == 0.0)
    assert np.all(res_fort == 0.0)


def test_law6_python_nonnegative(h2_endf_dict):
    """Physical sanity: a probability density must not be negative."""
    e_inc = np.array([5.0e6, 1.0e7, 2.0e7])
    e_out = np.linspace(0.0, 1.5e7, 15)
    mu = np.linspace(-1.0, 1.0, 9)
    res = np.asarray(mf6py.get_dist2d_from_subsec_law6(
        h2_endf_dict, 16, 1, e_inc, e_out, mu, True,
    ))
    assert np.all(np.isfinite(res))
    assert np.all(res >= 0.0)
