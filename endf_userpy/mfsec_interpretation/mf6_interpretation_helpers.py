import inspect
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


def has_subsecs_for_mt_zap(endf_dict, mt, zap):
    sec = endf_dict[6][mt]
    for subsec in sec['subsection'].values():
        if subsec['ZAP'] == zap:
            return True
    return False


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


def pad_outside_values(argnames, selectors):
    """Decorator factory to zero-pad results for invalid inputs."""
    def decorator(func):
        def get_arrays(args):
            all_argnames = list(inspect.signature(func).parameters.keys())
            return [
                args[all_argnames.index(p)] for p in argnames
            ]

        def replace_arrays(args, new_arrays):
            all_argnames = list(inspect.signature(func).parameters.keys())
            new_args = list(args)
            for i, an in enumerate(argnames):
                idx = all_argnames.index(an)
                new_args[idx] = new_arrays[i]
            return new_args

        def wrapfunc(*args, **kwargs):
            arr_list = get_arrays(args)
            inside_list = [
                s(a, *args, **kwargs) for s, a in zip(selectors, arr_list)
            ]
            if all(np.all(v) for v in inside_list):
                return func(*args, **kwargs)
            filtered_arrays = [
                a[f] for a, f in zip(arr_list, inside_list)
            ]
            new_args = replace_arrays(args, filtered_arrays)
            res_dim = [len(a) for a in arr_list]
            res_arr = np.zeros(res_dim, dtype=np.float64)
            res_arr[np.ix_(*inside_list)] = func(*new_args, **kwargs)
            return res_arr
        return wrapfunc
    return decorator


def _no_filter(arr, *args, **kwargs):
    return np.ones_like(arr, dtype=bool)


def _filter_energies_in(
    energies_in, endf_dict, mt, subsec_num,  *args, **kwargs
):
    subsec = endf_dict[6][mt]['subsection'][subsec_num]
    ei_mesh = dict2array(subsec['E'], dtype=float)
    eincs = energies_in
    return (eincs >= np.min(ei_mesh)) & (eincs <= np.max(ei_mesh))


def _filter_energies_out(energies_out, *args, **kwargs):
    return (energies_out >= 1e-20)


def pad_outside_dist2d_values(func):
    return pad_outside_values(
        ['energies_in', 'energies_out', 'angle_cosines_out'],
        [_filter_energies_in, _filter_energies_out, _no_filter]
    )(func)


def pad_outside_energydist_values(func):
    return pad_outside_values(
        ['energies_in', 'energies_out'],
        [_filter_energies_in, _filter_energies_out]
    )(func)


def pad_outside_angdist_values(func):
    return pad_outside_values(
        ['energies_in', 'angle_cosines_out'],
        [_filter_energies_in, _no_filter]
    )(func)
