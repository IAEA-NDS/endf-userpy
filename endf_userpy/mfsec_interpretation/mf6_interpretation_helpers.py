import numpy as np
from ..primitives.properties import (
    get_ZAP,
    is_zap_consistent,
)
from ..primitives.helpers import dict2array


def is_dist2d_law(law):
    return law in (1, 6, 7)


def contains_subsec_dist2d(endf_dict, mt, subsec_num):
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    law = subsec['LAW']
    return is_dist2d_law(law)


def get_zap_with_check(endf_dict, mt, zap):
    if zap is None:
        zap = get_ZAP(endf_dict, mt)
    elif not is_zap_consistent(endf_dict, mt, zap):
        raise ValueError(f'provided ZAP={zap} not consistent with MT={mt}')
    return zap


def check_mf6_exists(endf_dict):
    if 6 not in endf_dict:
        raise IndexError(
            f'No information on product-angle distributions found '
            f'(MF=6 section missing)'
        )


def check_mt_exists_in_mf6(endf_dict, mt):
    if mt not in endf_dict[6]:
        raise IndexError(
            f'No information on product-angle distributions found for MT={mt} '
            f'(MF6/MT{mt} missing)'
        )


def find_subsec_nums(endf_dict, mt, zap):
    sec = endf_dict[6][mt]
    idcs = tuple()
    for idx, subsec in sec['subsection'].items():
        if subsec['ZAP'] == zap:
            idcs += (idx,)
    if len(idcs) == 0:
        raise IndexError(
            f'subsection with ZAP={zap} not found in MF6/MT{mt}'
        )
    return idcs


def get_subsecs(endf_dict, mt, zap):
    subsec_nums = find_subsec_nums(endf_dict, mt, zap)
    subsecs = endf_dict[6][mt]['subsection']
    return tuple(subsecs[idx] for idx in subsec_nums)


def find_subsec_nums_by_laws(endf_dict, mt, laws):
    sec = endf_dict[6][mt]
    subsec_nums = []
    for idx, subsec in sec['subsection'].items():
        if subsec['LAW'] in laws:
            subsec_nums.append(idx)
    return subsec_nums


def find_subsec_nums_by_law_for_all_mts(endf_dict, laws):
    sec = endf_dict[6]
    res = dict()
    for mt in sec:
        cur_subsec_nums = find_subsec_nums_by_laws(endf_dict, mt, laws)
        if len(cur_subsec_nums) > 0:
            res[mt] = cur_subsec_nums
    return res


def get_zaps_for_mt(endf_dict, mt, dist2d_only=False):
    sec = endf_dict[6][mt]
    subsecs = sec['subsection']
    zaps = []
    for idx, subsec in subsecs.items():
        law = subsec['LAW']
        zap = subsec['ZAP']
        if dist2d_only and not is_dist2d_law(law):
            continue
        zaps.append(zap)
    return zaps


def contains_zap(endf_dict, mt, zap):
    zaps = get_zaps_for_mt(endf_dict, mt)
    return zap in zaps


def get_zaps_for_all_mts(endf_dict, dist2d_only=False):
    sec = endf_dict[6]
    res = dict()
    for mt in sec:
        cur_zaps = get_zaps_for_mt(endf_dict, mt, dist2d_only)
        if len(cur_zaps) > 0:
            res[mt] = cur_zaps
    return res


def has_disc_part(endf_dict, mt, zap):
    subsecs = get_subsecs(endf_dict, mt, zap)
    for subsec in subsecs:
        if law == 1:
            nd_arr = np.array(list(subsec['ND'].values()))
            nep_arr = np.array(list(subsec['NEP'].values()))
            return bool(np.any(nd_arr > 0))
    return False


def has_cont_part(endf_dict, mt, zap):
    subsecs = get_subsecs(endf_dict, mt, zap)
    for subsec in subsecs:
        law = subsec['LAW']
        if law in (6, 7):
            return True
        if law == 1:
            nd_arr = np.array(list(subsec['ND'].values()))
            nep_arr = np.array(list(subsec['NEP'].values()))
            return bool(np.any(nep_arr > nd_arr))
    return False


def has_angdist_part(endf_dict, mt, zap):
    subsecs = get_subsecs(endf_dict, mt, zap)
    for subsec in subsecs:
        law = subsec['LAW']
        if law in (2, 3, 4):
            return True
    return False


def pad_outside_dist2d_values(func):

    def wrapfunc(
        endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out, *args, **kwargs
    ):
        eincs = energies_in
        eouts = energies_out
        subsec = endf_dict[6][mt]['subsection'][subsec_num]
        ei_mesh = dict2array(subsec['E'], dtype=float)
        is_inside_x = (eincs >= np.min(ei_mesh)) & (eincs <= np.max(ei_mesh))
        is_inside_y = (eouts >= 1e-20)
        eincs_inside = eincs[is_inside_x]
        eouts_inside = eouts[is_inside_y]
        # special casing regular case with all points within mesh for small speedup
        if len(eincs_inside) == len(eincs) and len(eouts_inside) == len(eouts):
            return func(
                endf_dict, mt, subsec_num,
                eincs, eouts, angle_cosines_out,
                *args, **kwargs
            )
        # special treatment if some points outside mesh
        res_dim = (len(eincs), len(eouts), len(angle_cosines_out))
        res_arr = np.full(res_dim, 0.0, dtype=float)
        if len(eincs_inside) == 0 or len(eouts_inside) == 0:
            return res_arr
        arr_inside = func(
            endf_dict, mt, subsec_num,
            eincs_inside, eouts_inside, angle_cosines_out,
            *args, **kwargs
        )
        res_arr[np.ix_(is_inside_x, is_inside_y, np.array([True]))] =  arr_inside
        return res_arr

    return wrapfunc


def pad_outside_angdist_values(func):

    def wrapfunc(
        endf_dict, mt, subsec_num, energies_in, angle_cosines_out, *args, **kwargs
    ):
        eincs = energies_in
        mus_out = angle_cosines_out
        subsec = endf_dict[6][mt]['subsection'][subsec_num]
        ei_mesh = dict2array(subsec['E'], dtype=float)
        is_inside = (eincs >= np.min(ei_mesh)) & (eincs <= np.max(ei_mesh))
        eincs_inside = eincs[is_inside]
        # special casing regular case with all points within mesh for small speedup
        if len(eincs_inside) == len(eincs):
            return func(
                endf_dict, mt, subsec_num, eincs, mus_out, *args, **kwargs
            )
        # special treatment if some points outside mesh
        res_dim = (len(eincs), len(mus_out))
        res_arr = np.full(res_dim, 0.0, dtype=float)
        if len(eincs_inside) == 0:
            return res_arr
        arr_inside = func(
            endf_dict, mt, subsec_num, eincs_inside, mus_out, *args, **kwargs
        )
        res_arr[is_inside, :] =  arr_inside
        return res_arr

    return wrapfunc
