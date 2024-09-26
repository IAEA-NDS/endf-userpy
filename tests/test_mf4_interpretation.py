from pathlib import Path
import pytest
import numpy as np
from endf_parserpy import EndfParserCpp
import endf_userpy.mf4_interpretation as mf4py
import endf_userpy.mf4_interpretation_fort as mf4fort
from endf_userpy.helpers import deg2rad
from endf_userpy.reactions import is_binary_reaction


@pytest.fixture(scope="module")
def myEndfParser(
    ignore_zero_mismatch,
    ignore_number_mismatch,
    ignore_varspec_mismatch,
    accept_spaces,
    ignore_blank_lines,
    ignore_send_records,
    ignore_missing_tpid,
):
    return EndfParserCpp(
        ignore_zero_mismatch=ignore_zero_mismatch,
        ignore_number_mismatch=ignore_number_mismatch,
        ignore_varspec_mismatch=ignore_varspec_mismatch,
        accept_spaces=accept_spaces,
        ignore_blank_lines=ignore_blank_lines,
        ignore_send_records=ignore_send_records,
        ignore_missing_tpid=ignore_missing_tpid,
    )


def test_mf4_legrepr_python_fortran_equivalence():
    parser = EndfParserCpp(ignore_missing_tpid=True)
    data_dir = Path(__file__).resolve().parent / 'data'
    endf_file = data_dir / 'jeff33_1-H-2g_mf4_mt2.endf'
    endf_dict = parser.parsefile(endf_file)
    energies = np.array([1e6, 2e6, 3e6])
    angcos = np.cos(deg2rad(np.linspace(0.0, 180.0, 5)))
    res_py = mf4py.compute_angdist(endf_dict, 2, energies, angcos)
    res_fort = mf4fort.compute_angdist(endf_dict, 2, energies, angcos)
    assert np.allclose(res_py, res_fort)


def test_mf4_tabulated_python_fortran_equivalence():
    parser = EndfParserCpp(ignore_missing_tpid=True)
    data_dir = Path(__file__).resolve().parent / 'data'
    endf_file = data_dir / 'jeff33_13-Al-27g_mf4_mt2.endf'
    endf_dict = parser.parsefile(endf_file)
    energies = np.array([1e6, 2e6, 3e6])
    angcos = np.cos(deg2rad(np.linspace(0.0, 180.0, 5)))
    res_py = mf4py.compute_angdist(endf_dict, 2, energies, angcos)
    res_fort = mf4fort.compute_angdist(endf_dict, 2, energies, angcos)
    assert np.allclose(res_py, res_fort)


def test_mf4_python_fortran_equivalence(endf_file, myEndfParser):
    parser = myEndfParser
    endf_dict = parser.parsefile(endf_file)
    energies = np.array([1e6, 2e6, 3e6])
    angcos = np.cos(deg2rad(np.linspace(10.0, 170.0, 5)))
    mf4sec = endf_dict[4]
    for mt in mf4sec:
        # TODO: Generalize to also include non-binary reactions
        if not is_binary_reaction('n', mt):
            continue
        curens = energies
        if not mf4py.has_isotropic_angdist_repr(mf4sec, mt):
            en_range = mf4py.get_energy_range(mf4sec, mt)
            en_diff = np.diff(en_range)
            curens = np.linspace(en_range[0], en_range[1], 5)
        res_py = mf4py.compute_angdist(endf_dict, mt, curens, angcos)
        res_fort = mf4fort.compute_angdist(endf_dict, mt, curens, angcos)
        # TODO: If impossible kinematic situation, return 0, not NaN.
        #       After implementation, make test more strict
        assert np.allclose(res_py, res_fort, equal_nan=True)
