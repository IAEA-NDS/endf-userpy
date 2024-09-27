import numpy as np
from .interpolation import (
    evaluate_interp_legendre_polynomials,
    interp,
    endf_interp1d,
    interp_tab2,
)
from .conversion import (
    compute_r2,
    convert_angcos_to_cmsys,
    convert_angdist_to_labsys,
)
from .helpers import (
    deg2rad,
    dict2array,
    find_interval,
    convert_interp_repr,
)
from .properties import (
    get_AWR, get_AWI, get_AWP,
    get_QM, get_QI, get_LR,
)


def get_energy_range(mf4sec, mt):
    ens = list(mf4sec[mt]['E'].values())
    return (np.min(ens), np.max(ens)) 


def has_isotropic_angdist_repr(mf4sec, mt):
    return mf4sec[mt]['LTT'] == 0 and mf4sec[mt]['LI'] == 1


def _convert_legendre_to_numpy_array(coeffs_dict):
    num_energies = len(coeffs_dict)
    num_coeffs_per_energy = [len(v)+1 for _, v in coeffs_dict.items()]
    max_num_coeffs = np.max(num_coeffs_per_energy)
    coeffs_arr = np.zeros((num_energies, max_num_coeffs), dtype=float)
    for en_idx in range(num_energies):
        num_coeffs = num_coeffs_per_energy[en_idx]
        coeffs_arr[en_idx, 1:num_coeffs] = \
            dict2array(coeffs_dict[en_idx+1])
    coeffs_arr[:,0] = 1.0
    # apply factors (l+1/2)
    coeffs_arr *= np.arange(coeffs_arr.shape[1]).reshape(1,-1) + 0.5
    return coeffs_arr


def compute_angdist_from_isotropic(mf4sec, energies, angle_cosines):
    mu = angle_cosines
    mu = mu.reshape(1,-1) if mu.ndim == 1 else mu
    m = len(energies)
    n = mu.shape[1]
    return np.full((m, n), 0.5, dtype=float)


def compute_angdist_from_legrepr(mf4sec, energies, angle_cosines):
    mu = angle_cosines
    # get the energy mesh and bookkeeping information
    incident_energies = dict2array(mf4sec['E'])
    nbt_arr = np.array(mf4sec['NBT'], dtype=int)
    int_arr = np.array(mf4sec['INT'], dtype=int)
    # convert Legendre coefficients to numpy array
    coeffs_arr = _convert_legendre_to_numpy_array(mf4sec['a'])
    # compute the angular distribution in the laboratory system
    f = evaluate_interp_legendre_polynomials(
        energies, mu, incident_energies, coeffs_arr, int_arr, nbt_arr
    )
    return f


def compute_angdist_from_tabulated(mf4sec, energies, angle_cosines):
    en_mesh = dict2array(mf4sec['E'], dtype=float)
    nbt_arr = np.array(mf4sec['energy_table']['NBT'], dtype=int)
    int_arr = np.array(mf4sec['energy_table']['INT'], dtype=int)
    tab1_records = list(mf4sec['angtable'].values())
    return interp_tab2(
        energies, angle_cosines, en_mesh, int_arr, nbt_arr,
        tab1_records, 'mu', 'f'
    )


def compute_angdist_from_mixed(mf4sec, energies, angle_cosines):
    en_mesh = dict2array(mf4sec['E'], dtype=float)
    # prepare Legendre interpolation info for low energies
    num_ens1 = mf4sec['NE1']
    en_mesh1 = en_mesh[:num_ens1]
    nbt_arr1 = np.array(mf4sec['leg_int']['NBT'])
    int_arr1 = np.array(mf4sec['leg_int']['INT'])
    coeffs_arr = _convert_legendre_to_numpy_array(mf4sec['al'])
    assert num_ens1 == coeffs_arr.shape[0]
    # prepare tabulated iterpolation info for high energies
    num_ens2 = mf4sec['NE2']
    en_mesh2 = en_mesh[num_ens1-1:]
    assert num_ens2 == len(en_mesh2)
    nbt_arr2 = np.array(mf4sec['ang_int']['NBT'])
    int_arr2 = np.array(mf4sec['ang_int']['INT'])
    tab1_records = list(mf4sec['angtable'].values())
    assert num_ens2 == len(tab1_records)
    # interpolate according to region
    break_energy = en_mesh[num_ens1-1]
    mu = angle_cosines
    mu = mu.reshape(1, -1) if mu.ndim == 1 else mu
    is_lower = energies < break_energy
    energies_lower = energies[is_lower]
    mu_lower = mu[is_lower,:]
    f_lower = evaluate_interp_legendre_polynomials(
        energies_lower, mu_lower, en_mesh1, coeffs_arr, int_arr1, nbt_arr1
    )
    energies_upper = energies[~is_lower]
    mu_upper = mu[~is_lower,:]
    f_upper = interp_tab2(
        energies_upper, mu_upper, en_mesh2, int_arr2, nbt_arr2,
        tab1_records, 'mu', 'f'
    )
    # assemble the result
    f = np.zeros((len(energies), mu.shape[1]), dtype=float)
    f[is_lower] = f_lower
    f[~is_lower] = f_upper
    return f


def _compute_r2(endf_dict, mt, energies):
    awi = get_AWI(endf_dict)
    awr = get_AWR(endf_dict)
    awp = get_AWP(endf_dict, mt)
    q = get_QI(endf_dict, mt)
    return compute_r2(energies, awi, awr, awp, q)


def compute_angdist(endf_dict, mt, energies, angle_cosines):
    mf4sec = endf_dict[4][mt]
    ltt = mf4sec['LTT']
    li = mf4sec['LI']
    lct = mf4sec['LCT']
    # convert angle cosines to CM system if indicated
    mu = angle_cosines
    if lct == 1:
        mu_eff = mu
    elif lct == 2:
        r2 = _compute_r2(endf_dict, mt, energies)
        mu_eff = convert_angcos_to_cmsys(mu, r2)
    else:
        raise ValueError(f'Unknown reference system (LCT={lct}).')
    # perform the appropriate interpolation
    if ltt == 0 and li == 1:
        f_eff = compute_angdist_from_isotropic(mf4sec, energies, mu_eff)
    elif ltt == 1 and li == 0:
        f_eff = compute_angdist_from_legrepr(mf4sec, energies, mu_eff)
    elif ltt == 2 and li == 0:
        f_eff = compute_angdist_from_tabulated(mf4sec, energies, mu_eff)
    elif ltt == 3 and li == 0:
        f_eff = compute_angdist_from_mixed(mf4sec, energies, mu_eff)
    else:
        raise ValueError(
            'Unknown angular distribution representation '
            f'(MT={mt}, LTT={ltt}, LI={li}).'
        )
    # convert result to LAB system if required
    f_lab = f_eff if lct == 1 else convert_angdist_to_labsys(mu_eff, f_eff, r2)
    return f_lab
