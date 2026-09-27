import numpy as np
from ..primitives.properties import (
    get_ZAP,
    is_zap_consistent,
)
from ..primitives.helpers import (
    dict2array,
    pad_outside_values,
    no_filter,
)


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
            'No information on product-angle distributions found '
            '(MF=6 section missing)'
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
    if not contains_zap(endf_dict, mt, zap):
        return False
    subsecs = get_subsecs(endf_dict, mt, zap)
    for subsec in subsecs:
        law = subsec['LAW']
        if law == 1:
            nd_arr = np.array(list(subsec['ND'].values()))
            if bool(np.any(nd_arr > 0)):
                return True
    return False


def has_cont_part(endf_dict, mt, zap):
    if not contains_zap(endf_dict, mt, zap):
        return False
    subsecs = get_subsecs(endf_dict, mt, zap)
    for subsec in subsecs:
        law = subsec['LAW']
        if law in (6, 7):
            return True
        if law == 1:
            # daniel_notes.txt notes that a (MT, ZAP) can have more
            # than one subsection (e.g. same ZAP with different LIP)
            # that must be summed. Continue past a discrete-only
            # LAW=1 subsec here rather than returning False, so a
            # continuum-carrying subsec later in the list is not
            # silently masked. Mirrors the structure of has_disc_part
            # a few lines up.
            nd_arr = np.array(list(subsec['ND'].values()))
            nep_arr = np.array(list(subsec['NEP'].values()))
            if bool(np.any(nep_arr > nd_arr)):
                return True
    return False


def has_angdist_part(endf_dict, mt, zap):
    if not contains_zap(endf_dict, mt, zap):
        return False
    subsecs = get_subsecs(endf_dict, mt, zap)
    for subsec in subsecs:
        law = subsec['LAW']
        if law in (2, 3, 4):
            return True
    return False


def _filter_energies_in(
    energies_in, endf_dict, mt, subsec_num,  *args, **kwargs
):
    eincs = energies_in
    xp = kwargs.get('xp')
    if xp is not None and xp.name != 'numpy':
        # Under a non-numpy backend (e.g. JAX with a tracer mesh for
        # mesh-knot autodiff, or a tracer query for query-side
        # autodiff) the ``pad_outside_values`` decorator cannot
        # inspect a JAX bool mask via its ``np.all(v)`` fast-path
        # check. Short-circuit to "all inside"; the traced
        # reconstruction kernels below handle out-of-range via
        # their own ``outside_value`` fills. Mirrors the MF4
        # helper fix landed in PR #244.
        return np.ones(eincs.shape, dtype=bool)
    subsec = endf_dict[6][mt]['subsection'][subsec_num]
    ei_mesh = dict2array(subsec['E'], dtype=float)
    return (eincs >= np.min(ei_mesh)) & (eincs <= np.max(ei_mesh))


def _filter_energies_out(energies_out, *args, **kwargs):
    xp = kwargs.get('xp')
    if xp is not None and xp.name != 'numpy':
        # Same rationale as ``_filter_energies_in``: under a non-numpy
        # backend the eout may be a JAX tracer and the ``np.all(v)``
        # fast-path check in ``pad_outside_values`` cannot inspect a
        # tracer bool. Short-circuit to "all inside" (the traced
        # reconstruction handles physical Eout < 0 via its own
        # xp.where masks).
        return np.ones(energies_out.shape, dtype=bool)
    return (energies_out >= 1e-20)


def pad_outside_dist2d_values(func):
    return pad_outside_values(
        ['energies_in', 'energies_out', 'angle_cosines_out'],
        [_filter_energies_in, _filter_energies_out, no_filter]
    )(func)


def pad_outside_energydist_values(func):
    return pad_outside_values(
        ['energies_in', 'energies_out'],
        [_filter_energies_in, _filter_energies_out]
    )(func)


def pad_outside_angdist_values(func):
    return pad_outside_values(
        ['energies_in', 'angle_cosines_out'],
        [_filter_energies_in, no_filter]
    )(func)
