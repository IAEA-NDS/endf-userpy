import numpy as np
from .fortran.endf6 import mf4_get_leg

# helper functions

def deg2rad(values):
    return np.pi / 180.0 * np.array(values, copy=None)


def dict2array(obj, dtype=None):
    return np.array(list(v for v in obj.values()), dtype=dtype)


def find_interval(xmesh, x):
    # range checks
    min_xmesh = np.min(xmesh)
    max_xmesh = np.max(xmesh)
    if np.any(min_xmesh > x) or np.any(max_xmesh < x):
        raise IndexError('some user energies not in the range of energy mesh')
    # interval finding
    idcs = np.searchsorted(xmesh, x, side='right')
    idcs[x == max_xmesh] = len(xmesh) - 1
    return idcs


def convert_interpolation_representation(int_arr, nbt_arr):
    num_elements = nbt_arr[-1]
    interp_arr = np.zeros(num_elements, dtype=int)
    first_idx = 0
    for i in range(len(nbt_arr)):
        upper_idx = nbt_arr[i]
        interp_arr[first_idx:upper_idx] = int_arr[i]
        first_idx = upper_idx
    return interp_arr


# endf-6 specific functions

def get_QM(endf_dict, mt):
    return endf_dict[3][mt]['QM']


def get_QI(endf_dict, mt):
    return endf_dict[3][mt]['QI']


def get_LR(endf_dict, mt):
    return endf_dict[3][mt]['LR']


def get_AWR(endf_dict):
    return endf_dict[1][451]['AWR']


def get_AWI(endf_dict):
    # TODO: generalize to incident particles different from neutron 
    return 1.0


def get_AWP(endf_dict, mt):
    # TODO: generalize to reactions different from neutron elastic scattering
    return 1.0


def get_angdist_from_legendre(endf_dict, mt, energies, angle_cosines):
    mf4sec = endf_dict[4][mt]
    awi = get_AWI(endf_dict)
    awr = get_AWR(endf_dict)
    awp = get_AWP(endf_dict, mt)
    lct = mf4sec['LCT']
    qm = get_QM(endf_dict, mt)
    qi = get_QI(endf_dict, mt)
    breakup_flag = get_LR(endf_dict, mt)
    q = qi if breakup_flag == 0 else qm 

    # get the energy mesh and bookkeeping information
    incident_energies = dict2array(mf4sec['E'])
    nbt_arr = np.array(mf4sec['NBT'])
    int_arr = np.array(mf4sec['INT'])
    interp_arr = convert_interpolation_representation(int_arr, nbt_arr)
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

    return cur_res_arr
