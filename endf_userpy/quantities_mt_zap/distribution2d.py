import numpy as np
from ..mfsec_interpretation import mf4_interpretation_fort as mf4_interp 
from ..mfsec_interpretation import mf5_interpretation as mf5_interp
from ..mfsec_interpretation import mf6_interpretation as mf6_interp 
from ..primitives.physical_constants import get_zap_for_particle
from ..primitives.properties import (
    has_mf4_mt, has_mf5_mt, has_mf6_mt
)


def compute_dist2d_values_from_mf4_mf5(endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab=True):
    if to_lab is not True:
        raise ValueError(
            f"Double-differential distribution for MT={mt}, ZAP={zap} reconstruction "
            "from MF4 and MF5 only possible with `to_lab=True` argument."
        )
    if zap != get_zap_for_particle('n'):
        raise ValueError(
            f"Value of ZAP={zap} does not correspond to a neutron as ejectile. "
            "Reconstruction from MF4 and MF5 is only possible for neutrons."
        )
    energy_spect = mf5_interp.compute_spectrum(
        endf_dict, mt, energies_in, energies_out
    )
    angular_dist = mf4_interp.compute_angdist_values(
        endf_dict, mt, energies_in, angle_cosines_out, to_lab
    )
    energy_spect = energy_spect[:, :, np.newaxis]
    angular_dist = angular_dist[:, np.newaxis, :]
    dist2d_values = energy_spect * angular_dist
    return dist2d_values


def compute_dist2d_values_from_mf6(
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab=True
):
    return mf6_interp.compute_dist2d_values(
        endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab
    )


def compute_dist2d_values(
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab=True
):
    if has_mf6_mt(endf_dict, mt):
        func = compute_dist2d_values_from_mf6
        if mt == 18 and endf_dict[6][mt]['JP'] > 0:
            func = compute_dist2d_values_from_mf4_mf5
    elif has_mf4_mt(endf_dict, mt) and has_mf5_mt(endf_dict, mt):
        func = compute_dist2d_values_from_mf4_mf5
    else:
        raise IndexError(
            f"Cannot reconstruct double-differential distribution for MT={mt} "
            "because the required data is not available."
        )

    return func(
        endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab
    )
