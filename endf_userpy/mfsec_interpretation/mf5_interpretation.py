import numpy as np
from ..primitives.helpers import (
    dict2array,
    erf,
)
from ..primitives.interpolation import (
    endf_interp1d,
    interp_tab1,
    interp_tab2,
)


def _compute_prob(contrib_sec, energies_in):
    ein = energies_in.reshape(-1)
    return interp_tab1(
        ein, contrib_sec['p_table'], 'E', 'p', outside_value=0.0
    ).reshape(-1, 1)


def _compute_theta(contrib_sec, energies_in):
    ein = energies_in.reshape(-1)
    return interp_tab1(
        ein, contrib_sec['theta_table'], 'E', 'theta', outside_value=0.0
    ).reshape(-1, 1)  


def get_incident_energies_of_contribution(contrib_sec):
    en_mesh = np.array(contrib_sec['p_table']['E'], dtype=float)
    return en_mesh


def get_incident_energy_range(endf_dict, mt):
    mtsec = endf_dict[5][mt]
    contributions = list(mtsec['contribution'].values())
    en_min = np.inf 
    en_max = -np.inf
    for contrib in contributions:
        ens = get_incident_energies_of_contribution(contrib)
        en_min = min(np.min(ens), en_min)
        en_max = max(np.max(ens), en_max)
    return (en_min, en_max)


def compute_tabulated_spectrum(
    contrib_sec, energies_in, energies_out
):
    ein = energies_in
    eout = energies_out
    int_arr = np.array(contrib_sec['E_interp']['INT'])
    nbt_arr = np.array(contrib_sec['E_interp']['NBT'])
    ein_mesh = dict2array(contrib_sec['E'])
    tab1_records = list(contrib_sec['spectrum'].values())
    f = interp_tab2(
        ein, eout, ein_mesh, int_arr, nbt_arr, tab1_records, 'Eout', 'g', outside_value=0.0
    )
    return f


def compute_general_evaporation_spectrum(
    contrib_sec, energies_in, energies_out
):
    ein = energies_in.reshape(-1, 1)
    eout = energies_out.reshape(1, -1)
    theta = _compute_theta(contrib_sec, ein) 
    g = eout / theta
    return g


def compute_simple_maxwellian_fission_spectrum(
    contrib_sec, energies_in, energies_out
):
    ein = energies_in.reshape(-1, 1)
    eout = energies_out.reshape(1, -1)
    theta = _compute_theta(contrib_sec, ein)
    U = contrib_sec['U']
    is_nonzero = eout <= U
    eout_nz = eout[is_nonzero]
    fnz_unnorm = np.sqrt(eout_nz) * np.exp(-eout_nz / theta) 
    # compute normalization constant
    z = np.sqrt((ein - U) / theta)
    z1 = (0.5*np.sqrt(np.pi)*erf(z) - z*np.exp(-z))
    I = theta * np.sqrt(theta)
    fnz = fnz_unnorm / I
    f = np.zeros((ein.shape[0], eout.shape[1]), dtype=float)
    f[:, is_nonzero.reshape(-1)] = fnz
    return f


def compute_evaporation_spectrum(
    contrib_sec, energies_in, energies_out
):
    ein = energies_in.reshape(-1, 1)
    eout = energies_out.reshape(1, -1)
    U = contrib_sec['U']
    is_nonzero = eout <= U
    eout_nz = eout[is_nonzero]
    theta = _compute_theta(contrib_sec, ein)
    fnz_unnorm = eout_nz * np.exp(-eout_nz / theta) 
    # compute normalization constant
    I = 1 - (1.0 + (ein - U) / theta)  
    I *= np.square(theta)
    fnz = fnz_unnorm / I
    f = np.zeros((ein.shape[0], eout.shape[1]), dtype=float)
    f[:, is_nonzero.reshape(-1)] = fnz
    return f


def compute_spectrum_contribution(contrib_sec, energies_in, energies_out):
    ein = energies_in
    eout = energies_out
    lf = contrib_sec['LF']
    func = None
    if lf == 1:
        func = compute_tabulated_spectrum
    elif lf == 5:
        func = compute_general_evaporation_spectrum
    elif lf == 7:
        func = compute_simple_maxwellian_fission_spectrum
    elif lf == 9:
        func = compute_evaporation_spectrum
    else:
        raise ValueError(f'Spectrum computation for LF={lf} not implemented.')

    return func(contrib_sec, ein, eout) 


def compute_spectrum(endf_dict, mt, energies_in, energies_out): 
    ein = energies_in
    eout = energies_out
    contributions = list(endf_dict[5][mt]['contribution'].values())
    res = 0.0
    for contrib in contributions:
        prob = _compute_prob(contrib, ein)
        res += prob * compute_spectrum_contribution(contrib, ein, eout)
    return res
