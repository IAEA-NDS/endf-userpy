"""Bit-for-bit equivalence between the pure-Python MF6 LAW=7
(tabulated angle-energy double-differential) reconstruction and
the Fortran-backed reference implementation, plus a JAX-parity
test on the ported backend-agnostic path.

Mirrors ``test_mf6_law6_python_fortran_equivalence.py`` and
``test_mf6_law2_python_fortran_equivalence.py``.

Corpus: ``tests/data/n-004_Be_009.endf`` (Be-9). Be-9 MT=16
(n, 2n) writes both a neutron ejectile (ZAP=1, subsec 1) and an
alpha ejectile (ZAP=2004, subsec 2) as MF6 LAW=7 with tabulated
``f(Ep, mu | E)``. This is the shipped example for the LAW=7
smoke test (``test_dist2d_law7_interface``) and covers both
non-gamma ejectile ZAPs with the same code path.
"""
from pathlib import Path
import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_subsecs as mf6py,
    mf6_interpretation_subsecs_fort as mf6fort,
)
from endf_userpy.primitives import array_ns
from endf_userpy.primitives.helpers import deg2rad


DATA_DIR = Path(__file__).resolve().parent / 'data'


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def be9_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        DATA_DIR / 'n-004_Be_009.endf',
    )


@pytest.mark.parametrize('subsec_num,zap_label', [
    (1, 'neutron (ZAP=1)'),
    (2, 'alpha (ZAP=2004)'),
])
def test_law7_python_fortran_equivalence_be9_mt16(be9_endf_dict, subsec_num, zap_label):
    """Be-9 MT=16 subsec 1 (neutron) and subsec 2 (alpha), both
    LAW=7. Pure-Python path must match the Fortran path to machine
    precision on a representative ``(E, Ep, mu)`` grid.

    Both paths evaluate identical formulae: unit-base mu
    interpolation between adjacent mu subpanels within each
    incident-energy panel, then outer interpolation between the two
    bracketing E panels with the file's INT law. The only
    difference is which language does the arithmetic.
    """
    # Be-9 MT=16 LAW=7 E-mesh spans 15..20 MeV (reaction threshold
    # is above 1 MeV); sample query energies inside that window.
    e_in = np.array([1.55e7, 1.7e7, 1.85e7, 2.0e7])
    e_out = np.linspace(0.0, 5.0e5, 12)
    mu = np.cos(deg2rad(np.linspace(10.0, 170.0, 9)))

    res_py = np.asarray(mf6py.get_dist2d_from_subsec_law7(
        be9_endf_dict, 16, subsec_num, e_in, e_out, mu, True,
    ))
    res_fort = mf6fort.get_dist2d_from_subsec_law7(
        be9_endf_dict, 16, subsec_num, e_in, e_out, mu, True,
    )
    assert res_py.shape == (len(e_in), len(e_out), len(mu))
    assert res_py.shape == res_fort.shape
    np.testing.assert_allclose(
        res_py, res_fort, rtol=1e-12, atol=1e-14, equal_nan=True,
    )


def test_law7_python_returns_zero_outside_e_range(be9_endf_dict):
    """The ``pad_outside_dist2d_values`` decorator on the port
    zero-fills incident energies outside the tabulated range. Pins
    that behaviour (matches the pre-port Fortran path via the same
    decorator).
    """
    # Get the tabulated E range from the subsec.
    subsec = be9_endf_dict[6][16]['subsection'][1]
    ei_list = list(subsec['E'].values())
    e_min = min(ei_list)
    e_max = max(ei_list)
    # Query one point well below the mesh and one above.
    e_in = np.array([e_min * 0.5, e_max * 2.0])
    e_out = np.array([1.0e5])
    mu = np.array([0.0])
    res = np.asarray(mf6py.get_dist2d_from_subsec_law7(
        be9_endf_dict, 16, 1, e_in, e_out, mu, True,
    ))
    np.testing.assert_array_equal(res, 0.0)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law7_numpy_jax_parity(be9_endf_dict):
    """Full LAW=7 chain on Be-9: numpy vs JAX must return the same
    numbers to machine precision. Exercises unit-base mu
    interpolation + outer E interpolation via the backend-agnostic
    ``interp_tab2`` and ``_interp_two_point_columns`` primitives."""
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')
    e_in = np.array([1.55e7, 1.7e7, 1.85e7])
    e_out = np.linspace(0.0, 3.0e5, 8)
    mu = np.linspace(-0.9, 0.9, 7)
    res_np = np.asarray(mf6py.get_dist2d_from_subsec_law7(
        be9_endf_dict, 16, 1, e_in, e_out, mu, True, xp=xp_np,
    ))
    res_jax = np.asarray(mf6py.get_dist2d_from_subsec_law7(
        be9_endf_dict, 16, 1, e_in, e_out, mu, True, xp=xp_jax,
    ))
    np.testing.assert_allclose(res_np, res_jax, rtol=1e-12, atol=1e-14)
