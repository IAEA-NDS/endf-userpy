from ..primitives import properties as prop 
from ..primitives import reactions as reac
from ..primitives import physical_constants as physconst
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6help
from .quantities import (
    get_reaction_mt_numbers
)


def contains_zap(endf_dict, mt, zap):
    if mt in (18, 19, 20, 21, 38):
        return zap == PARTICLE_ZAP['n']
    elif prop.has_mf6_mt(endf_dict, mt):
        return mf6help.contains_zap(endf_dict, mt, zap)
    else:
        projectile = prop.get_projectile(endf_dict)
        ret = reac.contains_zap(projectile, mt, zap)
        return ret if ret is not None else False


def contains_zap_with_select_heuristic(endf_dict, mt, zap):

    if not contains_zap(endf_dict, mt, zap):
        return False

    if not reac.is_sum_mt(mt) and not reac.is_in_sum_mt(mt):
        return True

    sum_mt = mt if reac.is_sum_mt(mt) else reac.get_sum_mt_from_part_mt(mt)
    part_mts = reac.get_part_mts_from_sum_mt(sum_mt)

    exist_mts = set(get_reaction_mt_numbers(endf_dict))
    exist_part_mts = exist_mts.intersection(part_mts)

    exist_part_mts_in_mf4 = set(endf_dict.get(4, {})).intersection(part_mts)
    exist_part_mts_in_mf5 = set(endf_dict.get(5, {})).intersection(part_mts)
    exist_part_mts_in_mf6 = set(endf_dict.get(6, {})).intersection(part_mts)

    if (exist_part_mts_in_mf4 or exist_part_mts_in_mf5 or exist_part_mts_in_mf6):
        # distribution info exists so only the partial mts is selected
        return mt != sum_mt

    # distribution info does not exist so only the cumulative mt is selected
    return mt == sum_mt 
