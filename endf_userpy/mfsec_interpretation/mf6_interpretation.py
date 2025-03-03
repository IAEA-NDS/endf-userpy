import numpy as np
from ..primitives.properties import get_ZAP
from .mf6_interpretation_helpers import (
    check_mf6_exists,
    check_mt_exists_in_mf6,
    find_subsec_num,
    contains_subsec_dist2d,
)
from .mf6_interpretation_subsecs import (
    compute_dist2d_from_subsec,
    compute_dist1d_from_subsec,
    compute_yields_from_subsec,
)


def get_incident_energies(endf_dict, mt, zap):
    check_mf6_exists(endf_dict)
    check_mt_exists_in_mf6(endf_dict, mt)
    zap = zap if zap is not None else get_ZAP(endf_dict, mt)
    subsec_num = find_subsec_num(endf_dict, mt, zap)
    return get_incident_energies_from_subsec(
        endf_dict, mt, subsec_num
    )


def compute_dist2d_values(
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab=True
):
    check_mf6_exists(endf_dict)
    check_mt_exists_in_mf6(endf_dict, mt)
    zap = zap if zap is not None else get_ZAP(endf_dict, mt)
    subsec_num = find_subsec_num(endf_dict, mt, zap)
    if not contains_subsec_dist2d(endf_dict, mt, subsec_num):
        law = endf_dict[6][mt]['subsection'][subsec_num]['LAW']
        raise ValueError(
            f'Found product ZAP={zap} in MF6/MT{mt}/subsection[{subsec_num}] '
            f'but only an angular distribution is given (LAW={law})'
        )
    return compute_dist2d_from_subsec(
        endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out, to_lab
    )


def compute_yields(endf_dict, mt, zap, energies_in):
    check_mf6_exists(endf_dict)
    check_mt_exists_in_mf6(endf_dict, mt)
    zap = zap if zap is not None else get_ZAP(endf_dict, mt)
    subsec_num = find_subsec_num(endf_dict, mt, zap)
    if subsec_num is None:
        raise IndexError(
            f'yields not found for ZAP={zap} not found in MT={mt}'
        )
    return compute_yields_from_subsec(
        endf_dict, mt, subsec_num, energies_in
    )
