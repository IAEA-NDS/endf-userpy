import numpy as np
from ..primitives.helpers import (
    dict2array,
    find_indices_with_tol,
)
from ..primitives.interpolation import endf_interp1d
from .mf12_interpretation_helpers import (
    get_discrete_series_mts,
    init_trans2yield,
    trans2yield,
)


def get_photon_energies(endf_dict, mt):
    mtsec = endf_dict[12][mt]
    if mtsec['LO'] == 1:
        return dict2array(mtsec['Eg'])
    elif mtsec['LO'] == 2:
        res = compute_photon_yields_from_transition_probabilities(endf_dict, mt)
        return res['photon_energy']


def compute_photon_yields_from_transition_probabilities(endf_dict, mts):
    scalar_mt = not hasattr(mts, '__iter__')
    if scalar_mt:
        mts = [mts]
    series_mts = get_discrete_series_mts(
        endf_dict, mts[0], include_ground_state=False
    )
    if not np.all(np.isin(mts, series_mts)):
        raise ValueError(
            'All MT numbers must belong to the same discrete '
            ' MT series (e.g. fall in the range from 51 to 90 '
            'for inelastic neutron scattering.'
        )

    disc_mts, state_cache = init_trans2yield(endf_dict, mts[0]) 

    user_mts = set(mts)
    user_max_mt = max(user_mts)
    results = {} 
    for mt in disc_mts: 
        cur_result = (
            trans2yield(endf_dict, mt, state_cache)
        )
        if mt in user_mts:
            results[mt] = cur_result
            if mt == user_max_mt:
                break
    if scalar_mt:
        results = results[mts[0]]
    return results


def compute_photon_yields_from_tabulated_yields(endf_dict, mt, energies_in):
    mtsec = endf_dict[12][mt]
    if mtsec['LO'] != 1:
        raise ValueError(
            f'MT{mt} does not contain photon multiplicities'
        )
    eincs = energies_in
    num_eincs = len(energies_in)
    num_secs = mtsec['NK']
    tables = list(mtsec['table'].values())

    level_energies = dict2array(mtsec['ES']) 
    photon_energies = dict2array(mtsec['Eg'])
    photon_yields = np.array(
        [
            endf_interp1d(
            eincs, t['Eint'], t['y'], t['INT'], t['NBT'], outside_value=0.0
            ) for t in tables
        ]
    ).T
    return {
        'level_energy': level_energies, 
        'photon_energy': photon_energies,
        'photon_yield': photon_yields,
    }


def compute_photon_yields(endf_dict, mt, energies_in, photon_energies):
    mtsec = endf_dict[12][mt]
    LO_value = mtsec['LO']
    if LO_value == 1:
        res = compute_photon_yields_from_tabulated_yields(
            endf_dict, mt, energies_in
        )
    elif LO_value == 2:
        res = compute_photon_yields_from_transition_probabilities(
            endf_dict, mt
        )
        ones_vec = np.ones_like(energies_in).reshape(-1, 1)
        res['photon_yield'] = (
            res['photon_yield'].reshape(1,-1) * ones_vec
        )
    else:
        raise ValueError(
            'Invalid value LO={LO_value}'
        )

    # select the requested photon energies
    idcs = find_indices_with_tol(
        res['photon_energy'], photon_energies, atol=1e-4, rtol=1e-5
    )
    if np.any(idcs == -1):
        raise ValueError(
            'All user-supplied `photon_energies` must exist '
            f'in MF12/MT{mt} but this is not the case.'
        )
    return res['photon_yield'][:, idcs]
