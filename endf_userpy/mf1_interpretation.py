import numpy as np
from .helpers import dict2array
from .interpolation import endf_interp1d


def _compute_yields_from_polynomial(coefs, energies_in):
    return coefs * (energies_in**(np.arange(len(coefs))))


def compute_yields_from_mt452(endf_dict, energies_in):
    """Compute average total number of neutrons per fission."""
    mtsec = endf_dict[1][452]
    lnu = mtsec['LNU']
    ein = energies_in.reshape(-1)
    if lnu == 1:
        coefs = dict2array(mtsec['C'], dtype=float)
        return _compute_yields_from_polynomial(coefs, ein)
    elif lnu == 2:
        ep = np.array(mtsec['Eint'])
        nup = np.array(mtsec['nu'])
        int_arr = np.array(mtsec['INT'])
        nbt_arr = np.array(mtsec['NBT'])
        return endf_interp1d(
            ein, ep, nup, int_arr, nbt_arr, outside_value=0.0
        )
    raise ValueError('Invalid value LNU={lnu}')


def compute_yields_from_mt455(endf_dict, energies_in):
    """Compute average number of delayed neutrons per fission."""
    mtsec = endf_dict[1][455]
    lnu = mtsec['LNU']
    ein = energies_in.reshape(-1)
    if lnu == 1:
        coefs = dict2array(mtsec['nubar_d'], dtype=float)
        return _compute_yields_from_polynomial(coefs, ein)
    elif lnu == 2:
        ep = np.array(mtsec['Eint'])
        nup = np.array(mtsec['nubar_d'])
        int_arr = np.array(mtsec['INT'])
        nbt_arr = np.array(mtsec['NBT'])
        return endf_interp1d(
            ein, ep, nup, int_arr, nbt_arr, outside_value=0.0
        )
    raise ValueError('Invalid value LNU={lnu}')


def compute_yields_from_mt456(endf_dict, energies_in):
    """Compute average number of prompt neutrons per fission."""
    mtsec = endf_dict[1][456]
    lnu = mtsec['LNU']
    ein = energies_in.reshape(-1)
    if lnu == 1:
        return np.full_like(ein, float(mtsec['nubar_p']))
    elif lnu == 2:
        ep = np.array(mtsec['Eint'])
        nup = np.array(mtsec['nubar_p'])
        int_arr = np.array(mtsec['INT'])
        nbt_arr = np.array(mtsec['NBT'])
        return endf_interp1d(
            ein, ep, nup, int_arr, nbt_arr, outside_value=0.0
        )
    raise ValueError('Invalid value LNU={lnu}')


def compute_yields(endf_dict, mt, energies_in):
    """Compute total/delayed/prompt average number of neutrons per fission."""
    if mt == 452:
        func = compute_yields_from_mt452
    elif mt == 455:
        func = compute_yields_from_mt455
    elif mt == 456:
        func = compute_yields_from_mt456
    else:
        raise ValueError(f'Unsupported number MT={mt}')
    return func(endf_dict, energies_in)
