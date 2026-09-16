from pathlib import Path
import pytest
import numpy as np
from endf_parserpy import EndfParserCpp
import endf_userpy.mfsec_interpretation.mf4_interpretation as mf4py
import endf_userpy.mfsec_interpretation.mf4_interpretation_fort as mf4fort
from endf_userpy.primitives.helpers import deg2rad


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
    res_py = mf4py.compute_angdist_values(endf_dict, 2, energies, angcos)
    res_fort = mf4fort.compute_angdist_values(endf_dict, 2, energies, angcos)
    assert np.allclose(res_py, res_fort)


def test_mf4_tabulated_python_fortran_equivalence():
    parser = EndfParserCpp(ignore_missing_tpid=True)
    data_dir = Path(__file__).resolve().parent / 'data'
    endf_file = data_dir / 'jeff33_13-Al-27g_mf4_mt2.endf'
    endf_dict = parser.parsefile(endf_file)
    energies = np.array([1e6, 2e6, 3e6])
    angcos = np.cos(deg2rad(np.linspace(0.0, 180.0, 5)))
    res_py = mf4py.compute_angdist_values(endf_dict, 2, energies, angcos)
    res_fort = mf4fort.compute_angdist_values(endf_dict, 2, energies, angcos)
    assert np.allclose(res_py, res_fort)


def test_mf4_python_fortran_equivalence(endf_file, myEndfParser):
    parser = myEndfParser
    endf_dict = parser.parsefile(endf_file)
    energies = np.array([1e6, 2e6, 3e6])
    angcos = np.cos(deg2rad(np.linspace(10.0, 170.0, 5)))
    mf4sec = endf_dict[4]
    for mt in mf4sec:
        curens = energies
        if not mf4py.has_isotropic_angdist_repr(endf_dict, mt):
            en_range = mf4py.get_incident_energy_range(endf_dict, mt)
            curens = np.linspace(en_range[0], en_range[1], 5)
        res_py = mf4py.compute_angdist_values(endf_dict, mt, curens, angcos)
        res_fort = mf4fort.compute_angdist_values(endf_dict, mt, curens, angcos)
        # Kinematically forbidden mu_lab (H-1 elastic backwards
        # angles etc.) used to produce NaN in the Python path via
        # `sqrt(z<0)` in `convert_angcos_to_cmsys`; the Fortran
        # returned 0. Since issue #45 the Python path clips NaN
        # and the small-negative Jacobian artefacts near the
        # forbidden boundary to 0, matching the Fortran, so this
        # assertion no longer needs `equal_nan=True`.
        assert np.allclose(res_py, res_fort)
