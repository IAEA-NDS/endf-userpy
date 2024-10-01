from pathlib import Path
import pytest
import numpy as np
from endf_parserpy import EndfParserCpp
from endf_userpy.mf5_interpretation import compute_spectrum 
from endf_userpy.mf5_interpretation import get_incident_energy_range


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


def test_mf5_interpretation_never_fails(endf_file, myEndfParser):
    parser = myEndfParser 
    endf_dict = parser.parsefile(endf_file)
    mf5sec = endf_dict.get(5, dict())
    eout = np.linspace(1e-5, 20e6, 10)
    for mt in mf5sec:
        mtsec = mf5sec[mt]
        ein_min, ein_max = get_incident_energy_range(mtsec)
        ein = np.linspace(ein_min, ein_max,  10)
        compute_spectrum(endf_dict, mt, ein, eout) 
