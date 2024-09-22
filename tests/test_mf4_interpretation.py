from pathlib import Path
import numpy as np
from endf_parserpy import EndfParserCpp
from endf_userpy.mf4_interpretation import compute_angdist
from endf_userpy.mf4_interpretation_fort import (
    compute_angdist as compute_angdist_fort
)
from endf_userpy.helpers import deg2rad


def test_mf4_legrepr_python_fortran_equivalence():
    parser = EndfParserCpp(ignore_missing_tpid=True)
    data_dir = Path(__file__).resolve().parent / 'data' 
    endf_file = data_dir / 'jeff33_1-H-2g_mf4_mt2.endf'
    endf_dict = parser.parsefile(endf_file)
    energies = np.array([1e6, 2e6, 3e6]) 
    angcos = np.cos(deg2rad(np.linspace(0.0, 180.0, 5)))
    res_py = compute_angdist(endf_dict, 2, energies, angcos)
    res_fort = compute_angdist(endf_dict, 2, energies, angcos)
    assert np.allclose(res_py, res_fort)
