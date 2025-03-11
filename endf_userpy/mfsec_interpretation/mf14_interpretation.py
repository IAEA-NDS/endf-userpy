import numpy as np
from ..primitives.helpers import (
    dict2array,
    find_indices_with_tol,
)
from ..primitives.interpolation import evaluate_interp_legendre_polynomials 
from .mf4_interpretation import _convert_legendre_to_numpy_array
import logging


def get_photon_energies(endf_dict, mt):
    mtsec = endf_dict[14][mt]
    if mtsec['LI'] == 1:
        # everything isotropic, photon energy not provided
        return None
    return dict2array(mtsec['EG'])


def compute_angdist_from_isotropic(
    endf_dict, mt, energies_in, photon_energies, angle_cosines
): 
    num_eincs_in = len(energies_in)
    num_angcos_out = len(photon_energies)
    num_photens_out = len(angle_cosines)
    return np.full((num_eincs_in, num_angcos_out, num_photens_out), 0.5)


def compute_angdist_from_legendre(
    endf_dict, mt, energies_in, photon_energies, angle_cosines
):
    mtsec = endf_dict[14][mt]
    nk = mtsec['NK']
    ni = mtsec['NI']
    eg = dict2array(mtsec['EG'])

    idcs = find_indices_with_tol(
        eg, photon_energies, atol=1e-4, rtol=4e-5
    )

    # >= 0 check to skip photon energies that do not exist
    isotropic_idcs = idcs[(idcs >= 0) & (idcs < ni)]
    anisotropic_idcs = idcs[idcs >= ni]

    isotropic_angdists = compute_angdist_from_isotropic(
        endf_dict, mt, energies_in, angle_cosines, photon_energies[isotropic_idcs] 
    )
    einc_interp_tables = mtsec['E_interpol']
    res_list = []
    for sel_idx in anisotropic_idcs:
        interp_table = einc_interp_tables[sel_idx+1]
        nbt_arr = np.array(interp_table['NBT'], dtype=int)
        int_arr = np.array(interp_table['INT'], dtype=int)
        einc_mesh = dict2array(mtsec['E'][sel_idx+1], dtype=float) 
        coeffs_arr = _convert_legendre_to_numpy_array(mtsec['a'][sel_idx+1])
        f = evaluate_interp_legendre_polynomials(
            energies_in, angle_cosines, einc_mesh, coeffs_arr, int_arr, nbt_arr
        )
        res_list.append(f)

    full_angdists = np.zeros(
        (len(energies_in), len(photon_energies), len(angle_cosines)), dtype=float
    )
    full_angdists[:,isotropic_idcs,:] = isotropic_angdists
    if len(res_list) > 0:
        anisotropic_angdists = np.stack(res_list, axis=1) 
        full_angdists[:,anisotropic_idcs,:] = anisotropic_angdists
    return full_angdists


def compute_angdist_from_tabulated(
    endf_dict, mt, energies_in, photon_energies, angle_cosines
):
    raise NotImplementedError(
        'Tabulated angular distribution of photons not implemented.'
    )


def compute_angdist_values(
    endf_dict, mt, energies_in, photon_energies, angle_cosines
):
    mtsec = endf_dict[14][mt]
    if mtsec['LI'] == 1:
        return compute_angdist_from_isotropic(
            endf_dict, mt, energies_in, photon_energies, angle_cosines
        )
    elif mtsec['LI'] == 0 and mtsec['LTT'] == 1:
        return compute_angdist_from_legendre(
            endf_dict, mt, energies_in, photon_energies, angle_cosines
        )
    elif mtsec['LI'] == 0 and mtsec['LTT'] == 2:
        return compute_angdist_from_tabulated(
                endf_dict, mt, energies_in, photon_energies, angle_cosines
        )

    li_val = mtsec['LI']
    ltt_val = mtsec['LTT']
    raise ValueError(
        f'Unknown LI/LTT combination (LI={li_val}, LTT={ltt_val}) in MF14/MT{mt}'
    )
