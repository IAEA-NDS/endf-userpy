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
    print('debug #1')

    if user_mts is not None:
        # the current mt has only a chance of being selected
        # if any ancestor mt is listed in user_mts or the
        # current mt is directly listed in user_mts
        has_user_mt_ancestor = reac.any_ancestor_in_mts(mt, user_mts)
        if not has_user_mt_ancestor and mt not in user_mts:
            return False

    print('debug #2')
    # check if detailed distribution info available
    # for child mts (determined by sum rules) of current mt
    child_mts_avail_mf4 = reac.exist_associated_child_mts(mt, endf_dict.get(4, {}))
    child_mts_avail_mf5 = reac.exist_associated_child_mts(mt, endf_dict.get(5, {}))
    child_mts_avail_mf6 = reac.exist_associated_child_mts(mt, endf_dict.get(6, {}))

    if (child_mts_avail_mf4 or child_mts_avail_mf5 or child_mts_avail_mf6):
        # even if current mt is in user_mts,
        # summing of child mts is preferred
        # if detailed distribution info available for them
        return False

    print('debug #3')
    # we arrive here only if summing of child mts not preferred
    # so if current mt in user_mts, we select it
    if user_mts is not None and mt in user_mts:
        return True

    print('debug #4')
    # if detailed distribution info is available for current mt
    # or there is no known parent mt, we select it
    mt_in_mf4 = mt in endf_dict.get(4, {})
    mt_in_mf5 = mt in endf_dict.get(5, {})
    mt_in_mf6 = mt in endf_dict.get(6, {})
    has_ancestor = reac.any_ancestor_in_mts(mt, endf_dict.get(3, {}))
    if mt_in_mf4 or mt_in_mf5 or mt_in_mf6 or not has_ancestor:
        return True

    print('debug #5')
    return False
