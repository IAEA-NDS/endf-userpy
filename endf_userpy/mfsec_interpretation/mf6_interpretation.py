import numpy as np
from ..primitives import properties as prop
from ..primitives.properties import get_ZAP
from .mf6_interpretation_helpers import (
    check_mf6_exists,
    check_mt_exists_in_mf6,
    find_subsec_nums,
    contains_subsec_dist2d,
)
from .mf6_interpretation_subsecs import (
    get_incident_energies_from_subsec,
    get_emission_energies_from_subsec,
    compute_dist2d_from_subsec,
    compute_angdist_from_subsec,
    compute_yields_from_subsec,
)
import logging


module_logger = logging.getLogger(__name__)


def get_incident_energies(endf_dict, mt, zap):
    check_mf6_exists(endf_dict)
    check_mt_exists_in_mf6(endf_dict, mt)
    zap = zap if zap is not None else get_ZAP(endf_dict, mt)
    subsec_nums = find_subsec_nums(endf_dict, mt, zap)
    energies = []
    for subsec_num in subsec_nums:
        cur_energies = get_incident_energies_from_subsec(
            endf_dict, mt, subsec_num
        )
        energies.extend(cur_energies)
    return np.unique(energies)


def get_emission_energies(endf_dict, mt, zap, nofail=False):
    if nofail and not prop.has_mf6_mt(endf_dict, mt):
        return np.array([], dtype=float)
    check_mf6_exists(endf_dict)
    check_mt_exists_in_mf6(endf_dict, mt)
    subsec_nums = find_subsec_nums(endf_dict, mt, zap)
    energies = []
    for subsec_num in subsec_nums:
        cur_energies = get_emission_energies_from_subsec(
            endf_dict, mt, subsec_num, nofail
        )
        energies.extend(cur_energies)
    return np.unique(energies)


def compute_angdist_values(
    endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True
):
    check_mf6_exists(endf_dict)
    check_mt_exists_in_mf6(endf_dict, mt)
    zap = zap if zap is not None else get_ZAP(endf_dict, mt)
    subsec_nums = find_subsec_nums(endf_dict, mt, zap)
    found_angdist = False
    angdist = 0.0  # will be broadcasted to correct shape
    for subsec_num in subsec_nums:
        if contains_subsec_dist2d(endf_dict, mt, subsec_num):
            continue
        found_angdist = True
        angdist += compute_angdist_from_subsec(
            endf_dict, mt, subsec_num, energies_in, angle_cosines_out, to_lab
        )
    if not found_angdist:
        law = endf_dict[6][mt]['subsection'][subsec_num]['LAW']
        raise ValueError(
            f'Found product ZAP={zap} in MF6/MT{mt}/subsection[{subsec_num}] '
            f'but only double-differential distribution is given (LAW={law})'
        )
    return angdist


def compute_dist2d_values(
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab=True
):
    check_mf6_exists(endf_dict)
    check_mt_exists_in_mf6(endf_dict, mt)
    zap = zap if zap is not None else get_ZAP(endf_dict, mt)
    subsec_nums = find_subsec_nums(endf_dict, mt, zap)
    found_dist2d = False
    dist2d = 0.0  # will be broadcasted to correct shape
    for subsec_num in subsec_nums:
        if not contains_subsec_dist2d(endf_dict, mt, subsec_num):
            continue
        found_dist2d = True
        dist2d += compute_dist2d_from_subsec(
            endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out, to_lab
        )
    if not found_dist2d:
        law = endf_dict[6][mt]['subsection'][subsec_num]['LAW']
        raise ValueError(
            f'Found product ZAP={zap} in MF6/MT{mt}/subsection[{subsec_num}] '
            f'but only an angular distribution is given (LAW={law})'
        )
    return dist2d


def compute_yields(endf_dict, mt, zap, energies_in, include_discrete=True, level=None):
    check_mf6_exists(endf_dict)
    check_mt_exists_in_mf6(endf_dict, mt)
    module_logger.debug(f'compute yields for MT={mt}, ZAP={zap} and level={level}')
    zap = zap if zap is not None else get_ZAP(endf_dict, mt)
    subsec_nums = find_subsec_nums(endf_dict, mt, zap)
    found_yields = False
    yields = 0.0
    for curlev, subsec_num in enumerate(subsec_nums):
        if level is not None and curlev != level:
            module_logger.debug(f'skipping level={curlev} because level={level} requested')
            continue
        if (not contains_subsec_dist2d(endf_dict, mt, subsec_num)
                and not include_discrete):
            continue
        found_yields = True
        yields += compute_yields_from_subsec(
            endf_dict, mt, subsec_num, energies_in
        )
    if not found_yields:
        raise IndexError(
            f'yields not found for ZAP={zap} not found in MT={mt}'
        )
    return yields
