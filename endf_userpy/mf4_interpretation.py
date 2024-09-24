import numpy as np
from .interpolation import (
    evaluate_interp_legendre_polynomials,
    interp,
    endf_interp1d,
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


def compute_angdist_from_legrepr(mf4sec, energies, angle_cosines):
    mu = angle_cosines
    # get the energy mesh and bookkeeping information
    incident_energies = dict2array(mf4sec['E'])
    nbt_arr = np.array(mf4sec['NBT'], dtype=int)
    int_arr = np.array(mf4sec['INT'], dtype=int)
    num_coeffs_per_energy = dict2array(mf4sec['NL'])
    max_num_coeffs = np.max(num_coeffs_per_energy)
    # convert Legendre coefficients to numpy array
    coeffs = mf4sec['a']
    coeffs_arr = np.zeros((len(incident_energies), max_num_coeffs+1), dtype=float)
    for en_idx in range(len(incident_energies)):
        num_coeffs = num_coeffs_per_energy[en_idx]
        coeffs_arr[en_idx, 1:(num_coeffs+1)] = dict2array(coeffs[en_idx+1])
    coeffs_arr[:,0] = 1.0
    # apply factors (l+1/2)
    coeffs_arr *= np.arange(coeffs_arr.shape[1]).reshape(1,-1) + 0.5
    # compute the angular distribution in the laboratory system
    f = evaluate_interp_legendre_polynomials(
        energies, mu, incident_energies, coeffs_arr, int_arr, nbt_arr
    )
    return f


def compute_angdist_from_tabulated(mf4sec, energies, angle_cosines):
    mu = angle_cosines
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    en_mesh = dict2array(mf4sec['E'], dtype=float)
    nbt_arr = np.array(mf4sec['energy_table']['NBT'], dtype=int)
    int_arr = np.array(mf4sec['energy_table']['INT'], dtype=int)
    idcs = find_interval(en_mesh, energies)
    interp_arr = convert_interp_repr(int_arr, nbt_arr)
    # interpolation between angles
    result_dim = (len(energies), mu.shape[1])
    result_arr = np.zeros(result_dim, dtype=float)
    tab1_records = mf4sec['angtable']
    for i, idx in enumerate(idcs):
        curtab1 = tab1_records[idx+1]
        mu_mesh1 = np.array(curtab1['mu'], dtype=float)
        f_mesh1 = np.array(curtab1['f'], dtype=float)
        int_arr1 = np.array(curtab1['INT'], dtype=int)
        nbt_arr1 = np.array(curtab1['NBT'], dtype=int)
        curtab2 = tab1_records[idx+2]
        mu_mesh2 = np.array(curtab2['mu'], dtype=float)
        f_mesh2 = np.array(curtab2['f'], dtype=float)
        int_arr2 = np.array(curtab2['INT'], dtype=int)
        nbt_arr2 = np.array(curtab2['NBT'], dtype=int)
        if mu.shape[0] == 1:
            curmu = mu[0,:]
        else:
            curmu = mu[i,:]

        f1 = endf_interp1d(
            curmu, mu_mesh1, f_mesh1, int_arr1, nbt_arr1
        )
        f2 = endf_interp1d(
            curmu, mu_mesh2, f_mesh2, int_arr2, nbt_arr2
        )
        interp_type = interp_arr[idx]
        red_en_mesh = en_mesh[idx:idx+2]
        red_f = np.vstack([f1, f2])
        curen = energies[i]
        curres = np.zeros(mu.shape[1], dtype=float)
        for j in range(mu.shape[1]):
            curres[j] = \
                interp(curen, red_en_mesh , red_f[:,j], interp_type)

        result_arr[i,:] = curres

    return result_arr


def _compute_r2(endf_dict, mt, energies):
    awi = get_AWI(endf_dict)
    awr = get_AWR(endf_dict)
    awp = get_AWP(endf_dict, mt)
    qm = get_QM(endf_dict, mt)
    qi = get_QI(endf_dict, mt)
    breakup_flag = get_LR(endf_dict, mt)
    q = qi if breakup_flag == 0 else qm
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
    if ltt == 1 and li == 0:
        f_eff = compute_angdist_from_legrepr(mf4sec, energies, mu_eff)
    elif ltt == 2 and li == 0:
        f_eff = compute_angdist_from_tabulated(mf4sec, energies, mu_eff)
    else:
        raise ValueError(f'Unknown angular distribution representation.')
    # convert result to LAB system if required
    f_lab = f_eff if lct == 1 else convert_angdist_to_labsys(mu_eff, f_eff, r2)
    return f_lab
