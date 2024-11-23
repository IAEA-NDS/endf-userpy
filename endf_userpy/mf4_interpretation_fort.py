import numpy as np
from .fortran.endf6 import mf4_get_leg, mf4_get_tab
from .interpolation import find_interval
from .properties import (
    get_AWI, get_AWR, get_AWP,
    get_QM, get_QI, get_LR,
)
from .helpers import (
    dict2array,
    convert_interp_repr,
)


def get_angdist_from_isotropic(endf_dict, mt, energies, angle_cosines, to_lab):
    mf4sec = endf_dict[4][mt]
    awi = get_AWI(endf_dict)
    awr = get_AWR(endf_dict)
    awp = get_AWP(endf_dict, mt)
    lct = mf4sec['LCT'] if to_lab else 1
    qm = get_QM(endf_dict, mt)
    qi = get_QI(endf_dict, mt)
    breakup_flag = get_LR(endf_dict, mt)
    # TODO: Is it this determination of q-value correct? 
    q = qi if breakup_flag == 0 else qm

    num_angles = len(angle_cosines)
    num_coeffs1 = 1
    num_coeffs2 = 1
    coeffs1 = np.array([0], dtype=float)
    coeffs2 = np.array([0], dtype=float)
    en1 = np.min(energies)
    en2 = np.max(energies)
    interp_type = 2

    result_dim = (len(energies), len(angle_cosines))
    result_arr = np.zeros(result_dim, dtype=float)
    for i in range(result_dim[0]):
        cur_res_arr = np.zeros((1, result_dim[1]) , dtype=float)
        energy_out = energies[i]
        cur_res = mf4_get_leg(
            awr, awi, awp, q, lct,
            en1, coeffs1, num_coeffs1,
            en2, coeffs2, num_coeffs2,
            interp_type,
            energy_out, #  num_energies, (removed because automatically inferred)
            angle_cosines, num_angles,
            cur_res_arr
        )
        result_arr[i,:] = cur_res_arr

    return result_arr


def get_angdist_from_legendre(endf_dict, mt, energies, angle_cosines, to_lab):
    mf4sec = endf_dict[4][mt]
    awi = get_AWI(endf_dict)
    awr = get_AWR(endf_dict)
    awp = get_AWP(endf_dict, mt)
    lct = mf4sec['LCT'] if to_lab else 1
    q = get_QI(endf_dict, mt)

    # get the energy mesh and bookkeeping information
    incident_energies = dict2array(mf4sec['E'])
    nbt_arr = np.array(mf4sec['NBT'])
    int_arr = np.array(mf4sec['INT'])
    interp_arr = convert_interp_repr(int_arr, nbt_arr)
    num_coeffs_per_energy = dict2array(mf4sec['NL'])
    max_num_coeffs = np.max(num_coeffs_per_energy)
    # convert Legendre coefficients to numpy array
    coeffs = mf4sec['a']
    coeffs_arr = np.zeros((len(incident_energies), max_num_coeffs), dtype=float)
    for en_idx in range(len(incident_energies)):
        num_coeffs = num_coeffs_per_energy[en_idx]
        coeffs_arr[en_idx, 0:num_coeffs] = dict2array(coeffs[en_idx+1])
    # find enclosing energy intervals
    idcs = find_interval(incident_energies, energies)

    # call the fortran routine
    result_dim = (len(energies), len(angle_cosines))
    result_arr = np.zeros(result_dim, dtype=float)
    for i in range(result_arr.shape[0]):
        curidx = idcs[i]
        en1 = incident_energies[curidx]
        en2 = incident_energies[curidx+1]
        coeffs1 = coeffs_arr[curidx]
        coeffs2 = coeffs_arr[curidx+1]
        num_coeffs1 = num_coeffs_per_energy[curidx]
        num_coeffs2 = num_coeffs_per_energy[curidx+1]
        interp_type = interp_arr[curidx]

        # TODO: group all energies falling in a certain bin
        #       together and pass them all at once to mf4_get_leg
        num_energies = 1
        num_angles = len(angle_cosines)
        energy_out = np.array([energies[i]], dtype=float)
        cur_res_arr = np.zeros((num_energies, result_dim[1]) , dtype=float)

        # call the fortran routine
        # NOTE: num_energies comment out because f2py
        #       derives ne argument from shape of f4
        cur_res = mf4_get_leg(
            awr, awi, awp, q, lct,
            en1, coeffs1, num_coeffs1,
            en2, coeffs2, num_coeffs2,
            interp_type,
            energy_out, #  num_energies, (removed because automatically inferred)
            angle_cosines, num_angles,
            cur_res_arr
        )
        # copy current result to overall result array
        result_arr[i:i+1,:] = cur_res_arr

    return result_arr


def get_angdist_from_tabulated(endf_dict, mt, energies, angle_cosines, to_lab):
    num_energies = len(energies)
    num_angle_cosines = len(angle_cosines)
    mf4sec = endf_dict[4][mt]
    awi = get_AWI(endf_dict)
    awr = get_AWR(endf_dict)
    awp = get_AWP(endf_dict, mt)
    lct = mf4sec['LCT'] if to_lab else 1
    qm = get_QM(endf_dict, mt)
    qi = get_QI(endf_dict, mt)
    breakup_flag = get_LR(endf_dict, mt)
    q = qi if breakup_flag == 0 else qm
    # get incident mesh and interpolation info
    incident_energies = dict2array(mf4sec['E'])
    int_arr = mf4sec['energy_table']['INT']
    nbt_arr = mf4sec['energy_table']['NBT']
    interp_arr = convert_interp_repr(int_arr, nbt_arr)
    # find enclosing tab1 records with tabulated angular distribution
    idcs = find_interval(incident_energies, energies)
    # call fortran routine
    result_dim = (len(energies), len(angle_cosines))
    result_arr = np.zeros(result_dim, dtype=float)
    for i in range(result_dim[0]):
        curens = energies[i:i+1]
        curidx = idcs[i]
        e1 = incident_energies[curidx]
        e2 = incident_energies[curidx+1]
        interp_type = interp_arr[curidx]

        lower_tab1 = mf4sec['angtable'][curidx+1]
        upper_tab1 = mf4sec['angtable'][curidx+2]

        u1 = np.array(lower_tab1['mu'])
        f1 = np.array(lower_tab1['f'])
        nbt1 = np.array(lower_tab1['NBT'])
        ibt1 = np.array(lower_tab1['INT'])
        np1 = len(u1)
        nr1 = len(nbt1)

        u2 = np.array(upper_tab1['mu'])
        f2 = np.array(upper_tab1['f'])
        nbt2 = np.array(upper_tab1['NBT'])
        ibt2 = np.array(upper_tab1['INT'])
        np2 = len(u2)
        nr2 = len(nbt2)

        cur_res_arr = np.zeros((1, num_angle_cosines), dtype=float)

        mf4_get_tab(
            awr, awi, awp, q, lct,
            e1, u1, f1, np1, nbt1, ibt1, nr1,
            e2, u2, f2, np2, nbt2, ibt2, nr2,
            interp_type, curens, #  num_energies, (automatically inferred)
            angle_cosines, num_angle_cosines,
            cur_res_arr
        )
        result_arr[i, :] = cur_res_arr
    return result_arr


def get_angdist_from_mixed(endf_dict, mt, energies, angle_cosines, to_lab):
    mu = angle_cosines
    mu = mu.reshape(1, -1) if mu.ndim == 1 else mu
    num_energies = len(energies)
    num_angle_cosines = mu.shape[1]
    mf4sec = endf_dict[4][mt]
    awi = get_AWI(endf_dict)
    awr = get_AWR(endf_dict)
    awp = get_AWP(endf_dict, mt)
    lct = mf4sec['LCT'] if to_lab else 1
    qm = get_QM(endf_dict, mt)
    qi = get_QI(endf_dict, mt)
    breakup_flag = get_LR(endf_dict, mt)
    q = qi if breakup_flag == 0 else qm
    # get the energy mesh and interpolation info
    en_mesh = dict2array(mf4sec['E'])
    num_ens1 = mf4sec['NE1']
    num_ens2 = mf4sec['NE2']
    break_energy = en_mesh[num_ens1-1]
    nbt_arr_leg = np.array(mf4sec['leg_int']['NBT'])
    int_arr_leg = np.array(mf4sec['leg_int']['INT'])
    interp_arr_leg = convert_interp_repr(int_arr_leg, nbt_arr_leg)
    nbt_arr_tab = np.array(mf4sec['ang_int']['NBT'])
    int_arr_tab = np.array(mf4sec['ang_int']['INT'])
    interp_arr_tab = convert_interp_repr(int_arr_tab, nbt_arr_tab)
    interp_arr = np.concatenate([interp_arr_tab, interp_arr_leg])
    # prepare Legendre array
    num_coeffs_per_energy = dict2array(mf4sec['NL'])
    max_num_coeffs = np.max(num_coeffs_per_energy)
    coeffs = mf4sec['al']
    coeffs_arr = np.zeros((num_ens1, max_num_coeffs), dtype=float)
    for en_idx in range(num_ens1):
        num_coeffs = num_coeffs_per_energy[en_idx]
        coeffs_arr[en_idx, 0:num_coeffs] = dict2array(coeffs[en_idx+1])

    # call fortran routine
    idcs = find_interval(en_mesh, energies)
    result_dim = (len(energies), len(angle_cosines))
    result_arr = np.zeros(result_dim, dtype=float)
    for i, idx in enumerate(idcs):
        curen = energies[i]
        curens = np.array([curen], dtype=float)
        en1 = en_mesh[idx]
        en2 = en_mesh[idx+1]
        interp_type = interp_arr[idx]
        # Legendre representation case
        if curen < break_energy:
            coeffs1 = coeffs_arr[idx]
            coeffs2 = coeffs_arr[idx+1]
            num_coeffs1 = num_coeffs_per_energy[idx]
            num_coeffs2 = num_coeffs_per_energy[idx+1]

            num_mus = mu.shape[1]
            cur_mus = mu[0,:]
            energy_out = np.array([energies[i]], dtype=float)
            cur_res_arr = np.zeros((1, result_dim[1]) , dtype=float)
            # call the fortran routine
            # NOTE: num_energies comment out because f2py
            #       derives ne argument from shape of f4
            mf4_get_leg(
                awr, awi, awp, q, lct,
                en1, coeffs1, num_coeffs1,
                en2, coeffs2, num_coeffs2,
                interp_type,
                energy_out, #  num_energies, (removed because automatically inferred)
                cur_mus, num_mus,
                cur_res_arr
            )
            # copy current result to overall result array
            result_arr[i:i+1,:] = cur_res_arr

        # Tabulated angular distribution case
        else:
            lower_tab1 = mf4sec['angtable'][idx+1]
            upper_tab1 = mf4sec['angtable'][idx+2]

            u1 = np.array(lower_tab1['mu'])
            f1 = np.array(lower_tab1['f'])
            nbt1 = np.array(lower_tab1['NBT'])
            ibt1 = np.array(lower_tab1['INT'])
            np1 = len(u1)
            nr1 = len(nbt1)

            u2 = np.array(upper_tab1['mu'])
            f2 = np.array(upper_tab1['f'])
            nbt2 = np.array(upper_tab1['NBT'])
            ibt2 = np.array(upper_tab1['INT'])
            np2 = len(u2)
            nr2 = len(nbt2)

            cur_res_arr = np.zeros((1, num_angle_cosines), dtype=float)

            mf4_get_tab(
                awr, awi, awp, q, lct,
                en1, u1, f1, np1, nbt1, ibt1, nr1,
                en2, u2, f2, np2, nbt2, ibt2, nr2,
                interp_type, curens, #  num_energies, (automatically inferred)
                angle_cosines, num_angle_cosines,
                cur_res_arr
            )
            result_arr[i, :] = cur_res_arr

    return result_arr


def compute_angdist_values(endf_dict, mt, energies, angle_cosines, to_lab=True):
    mu_lab = angle_cosines
    mf4sec = endf_dict[4][mt]
    ltt = mf4sec['LTT']
    li = mf4sec['LI']
    if ltt == 0 and li == 1:
        return get_angdist_from_isotropic(endf_dict, mt, energies, mu_lab, to_lab)
    elif ltt == 1 and li == 0:
        return get_angdist_from_legendre(endf_dict, mt, energies, mu_lab, to_lab)
    elif ltt == 2 and li == 0:
        return get_angdist_from_tabulated(endf_dict, mt, energies, mu_lab, to_lab)
    elif ltt == 3 and li == 0:
        return get_angdist_from_mixed(endf_dict, mt, energies, mu_lab, to_lab)

    raise TypeError(
        f'Interpretation of MF4/MT for LTT={ltt}, LI={li} not implemented.'
    )
