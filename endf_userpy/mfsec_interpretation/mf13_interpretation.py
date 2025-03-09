import numpy as np
from ..primitives.helpers import dict2array
from ..primitives.interpolation import endf_interp1d


def get_photon_energies(endf_dict, mt):
    mtsec = endf_dict[13][mt]
    pe = [s['EG'] for s in mtsec['subsection'].values()]
    return np.array(pe, dtype=float)


def _compute_total_production_xs(endf_dict, mt, energies_in):
    mtsec = endf_dict[13][mt]
    int_arr = np.array(mtsec['INT'])
    nbt_arr = np.array(mtsec['NBT'])
    einc_mesh = np.array(mtsec['E'])
    xs_mesh = np.array(mtsec['sigma_tot'])
    return endf_interp1d(
        energies_in, einc_mesh, xs_mesh, int_arr, nbt_arr, outside_value=0.0
    )


def compute_photon_production_xs(endf_dict, mt, energies_in):
    mtsec = endf_dict[13][mt]
    eincs = energies_in
    subsecs = list(mtsec['subsection'].values())
    level_energies = np.array([sec['ES'] for sec in subsecs])
    photon_energies = np.array([sec['EG'] for sec in subsecs])
    prod_xs = np.array(
        [
            endf_interp1d(
            eincs, t['E'], t['sigma'], t['INT'], t['NBT'], outside_value=0.0
            ) for t in subsecs
        ]
    ).T
    return {
        'level_energy': level_energies,
        'photon_energy': photon_energies,
        'photon_prodxs': prod_xs
    }


def compute_total_photon_production_xs(endf_dict, mt, energies_in):
    prod = compute_photon_production_xs(endf_dict, mt, energies_in)
    prod_xs = prod['photon_prodxs']
    total_prod_xs = np.sum(prod_xs, axis=1)
    if prod_xs.shape[0] > 1:
        check_tot_prod_xs = _compute_total_production_xs(
            endf_dict, mt, energies_in
        )
        if not np.allclose(total_prod_xs, check_tot_prod_xs):
            raise ValueError(
                'Total photon production cross section '
                'given in MF12/MT{mt} does not equal the ' 
                'the sum of the partial production cross '
                'sections.'
            )
    return total_prod_xs
