from ..primitives import properties as prop 
from ..primitives import reactions as reac
from ..primitives import physical_constants as physconst
from ..mfsec_interpretation import mf3_interpretation as mf3interp
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6help
from .quantities import (
    get_reaction_mt_numbers
)


def contains_zap(endf_dict, mt, zap):
    if mt in (18, 19, 20, 21, 38):
        return zap == physconst.PARTICLE_ZAP['n']
    if prop.has_mf6_mt(endf_dict, mt):
        return mf6help.contains_zap(endf_dict, mt, zap)

    projectile = prop.get_projectile(endf_dict)
    ret = reac.contains_zap(projectile, mt, zap)
    if ret is True:
        return True
    if ret is None and not reac.is_sum_mt(mt):
        raise ValueError(
            "Unable to determine whether reaction associated with "
            f"MT={mt} contains a particle identified by ZAP={zap}."
        )
    # for some sum channels, it can't be determined
    # from the reaction string whether there is a zap ejectile
    # and we need to loop over the available partial mts to figure out.
    avail_mts = get_reaction_mt_numbers(endf_dict)
    part_mts = reac.get_part_mts_from_sum_mt(mt)
    loop_mts = set(avail_mts).intersection(part_mts)
    for part_mt in loop_mts:
        if contains_zap(projectile, part_mt, zap):
            return True
    return False


def satisfies_select_heuristic(endf_dict, mt, user_mts=None):
    if user_mts is not None:
        if not (hasattr(user_mts, '__iter__') or
                hasattr(user_mts, '__contains__')):
            user_mts = [user_mts]
        user_mts = set(user_mts)

    if not reac.is_sum_mt(mt) and not reac.is_in_sum_mt(mt):
        # no sum rule involved so we select the mt number
        # (and only if mt in user_mts, if provided)
        if user_mts is not None:
            return mt in user_mts
        return True

    # a sum rule is involved
    sum_mt = mt if reac.is_sum_mt(mt) else reac.get_sum_mt_from_part_mt(mt)
    part_mts = reac.get_part_mts_from_sum_mt(sum_mt)

    if user_mts is not None:
        if (sum_mt in user_mts and user_mts.intersection(part_mts)):
            raise ValueError(
                f'`user_mts` contains the sum_mt={sum_mt} but also '
                ' MT numbers which contribute to the cross section '
                ' associated with `sum_mt`. This is not allowed.'
            )
        if mt in part_mts:
            if mt in user_mts:
                return True
            if sum_mt not in user_mts:
                return False
        if mt == sum_mt and mt not in user_mts:
            return False

    # the user_mts is not given or
    # mt == sum_mt and the mt is in user_mts.
    # in either case, the following logic decides
    # whether all partial mts should be selected but not the sum_mt
    # or the sum_mt should be selected only and partial mts skipped.
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
