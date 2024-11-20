import numpy as np
from .properties import is_zap_consistent
from .interpolation import interp_tab1
from .mf6_interpretation_subsecs import (
    get_dist2d_from_subsec_law1,
    get_dist1d_from_subsec_law2,
    get_dist2d_from_subsec_law6,
    get_dist2d_from_subsec_law7,
)


def compute_yields_from_subsec(endf_dict, mt, subsec_num, energies_in):
    sec = endf_dict[6][mt]
    subsec = sec[subsec_num]
    yield_tab = subsec['yields']
    interp_yields = interp_tab1(
        energies_in, yield_tab, 'Eint', 'yi'
    )
    return interp_yields


def compute_dist1d_from_subsec(
    endf_dict, mt, subsec_num,
    energies_in, angle_cosines_out
):
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    law = subsec['LAW']
    if law == 2:
        return get_dist1d_from_subsec_law2(
            endf_dict, mt, subsec_num, energies_in, angle_cosines_out
        )
    else:
        raise NotImplementedError(
            f'Angular distribution interpretation for LAW={law} not implemented.'
        )


def compute_dist2d_from_subsec(
    endf_dict, mt, subsec_num,
    energies_in, energies_out, angle_cosines_out
):
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    law = subsec['LAW']
    if law == 1:
        return get_dist2d_from_subsec_law1(
            endf_dict, mt, subsec_num,
            energies_in, energies_out, angle_cosines_out
        )
    elif law == 6:
        return get_dist2d_from_subsec_law6(
            endf_dict, mt, subsec_num,
            energies_in, energies_out, angle_cosines_out
        )
    elif law == 7:
        return get_dist2d_from_subsec_law7(
            endf_dict, mt, subsec_num,
            energies_in, energies_out, angle_cosines_out
        )
    else:
        raise NotImplementedError(
            f'DDX interpretation for LAW={law} not implemented.'
        )


def compute_dist2d(
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out
):
    check_mf6_exists(endf_dict)
    check_mt_exists_in_mf6(endf_dict, mt)
    zap = get_zap_with_check(endf_dict, mt, zap)
    subsec_num = find_subsec_num(endf_dict, mt, zap)
    if not contains_subsec_dist2d(endf_dict, mt, subsec_num):
        raise ValueError(
            f'Found product ZAP={zap} in MF6/MT{mt}/subsection[{idx}] '
            f'but only an angular distribution is given (LAW={law})'
        )
    return compute_dist2d_from_subsec(
        endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out
    )


def compute_yields(endf_dict, mt, zap, energies_in):
    check_mf6_exists(endf_dict)
    check_mt_exists_in_mf6(endf_dict, mt)
    zap = get_zap_with_check(endf_dict, mt, zap)
    subsec_num = _find_subsec_num(endf_dict, mt, zap)
    if subsec_num is None:
        raise IndexError(
            f'yields not found for ZAP={zap} not found in MT={mt}'
        )
    return get_yields_from_subsec(
        endf_dict, mt, subsec_num, energies_in
    )
