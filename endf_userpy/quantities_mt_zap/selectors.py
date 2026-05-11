from ..primitives import properties as prop
from ..primitives import reactions as reac
from ..primitives import physical_constants as physconst
from ..mfsec_interpretation import mf3_interpretation as mf3interp
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6help
from .quantities import (
    get_reaction_mt_numbers
)
import logging


module_logger = logging.getLogger(__name__)


def has_continuous_ddx(endf_dict, mt, zap):
    """Whether (MT, ZAP) carries continuum DDX information.

    Returns True if a true 2D distribution (continuous in both Eout
    and mu) can be reconstructed for this MT/ZAP, either from MF6
    (LAW=6/7, or LAW=1 with a continuum part beyond any discrete
    spikes) or from the MF4 angular distribution combined with the
    MF5 energy spectrum (neutron only). Returns False for channels
    that only carry kinematic-delta information (e.g. MT=2 elastic
    in a typical neutron-induced file, or MT=51..90 with MF6/LAW=2
    only). Used to decide which channels to admit into a cumulative
    DDX sum.
    """
    if prop.has_mf6_mt(endf_dict, mt):
        return mf6help.has_cont_part(endf_dict, mt, zap)
    if prop.has_mf4_mt(endf_dict, mt) and prop.has_mf5_mt(endf_dict, mt):
        return zap == physconst.get_zap_for_particle('n')
    return False


def contains_zap(endf_dict, mt, zap):
    if mt in (18, 19, 20, 21, 38):
        return zap == physconst.PARTICLE_ZAP['n']
    if prop.has_mf6_mt(endf_dict, mt):
        return mf6help.contains_zap(endf_dict, mt, zap)
    if reac.is_x_particle_production_mt(mt):
        # MT 201..207 aggregate production of one specific particle;
        # whether they actually contribute to a cumulative sum is
        # decided by satisfies_select_heuristic (partials win when
        # available).
        particle = reac.X_PRODUCTION_SUM_TO_PARTICLE[mt]
        return zap == physconst.get_zap_for_particle(particle)
    if not reac.is_known_reaction_mt(mt):
        # HEATR-injected MTs (301..450), dosimetry/bookkeeping MTs
        # (251..253, 451+), and similar non-reaction MF3 entries
        # carry no ejectile information.
        return False

    projectile = prop.get_projectile(endf_dict)
    ret = reac.contains_zap(projectile, mt, zap)
    if ret is True or ret is False:
        return ret
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
        if contains_zap(endf_dict, part_mt, zap):
            return True
    return False


def contains_residual_za(endf_dict, mt, residual_za):
    proj = prop.get_projectile(endf_dict)
    za_projectile = prop.get_ZAI(endf_dict)
    ejectiles = reac.get_ejectiles(proj, mt)
    if ejectiles is None:
        return False
    real_za_residual = prop.get_ZA(endf_dict) + za_projectile
    for mult, ejectile in ejectiles:
        real_za_residual -= mult * physconst.get_zap_for_particle(ejectile)
    return real_za_residual == residual_za


def contains_residual_za_and_lfs(endf_dict, mt, residual_za, lfs):
    """Whether MT produces residual nucleus (residual_za, lfs).

    Prefers MF8/MT as the authoritative listing of (ZAP, LFS) pairs
    produced by this MT. When MF8/MT is absent, falls back to MF6/MT
    if it carries ZAP-tagged subsections (the catch-all MT=5 case in
    photonuclear and proton-induced files), and finally to the
    reaction-string-based contains_residual_za. `lfs=None` means
    "any isomer state"; the MF6 and reaction-string fallbacks cannot
    resolve isomers, so they only apply when `lfs` is None or 0.
    """
    if 8 in endf_dict and mt in endf_dict[8]:
        for sub in endf_dict[8][mt]['subsection'].values():
            if sub['ZAP'] != residual_za:
                continue
            if lfs is None or sub['LFS'] == lfs:
                return True
        return False
    if lfs not in (None, 0):
        return False
    if 6 in endf_dict and mt in endf_dict[6]:
        if mf6help.contains_zap(endf_dict, mt, residual_za):
            return True
    return contains_residual_za(endf_dict, mt, residual_za)


def _partials_can_answer(endf_dict, partials, zap, op):
    """Predicate: can the partial-channel path satisfy the requested
    operation for the given particle?"""
    if not partials:
        return False
    if op == 'xs':
        # Every partial in the list is in MF3 by construction.
        return True
    if op in ('dexs', 'daxs'):
        return all(
            prop.has_mf6_mt(endf_dict, m)
            or (prop.has_mf4_mt(endf_dict, m) and prop.has_mf5_mt(endf_dict, m))
            for m in partials
        )
    if op == 'ddxs':
        return all(has_continuous_ddx(endf_dict, m, zap) for m in partials)
    return False


def _x_production_sum_can_answer(endf_dict, sum_mt, zap, op):
    """Whether the aggregate MT 201..207 entry has the data needed."""
    if op == 'xs':
        return sum_mt in endf_dict.get(3, {})
    if op in ('dexs', 'daxs'):
        return (
            prop.has_mf6_mt(endf_dict, sum_mt)
            or (prop.has_mf4_mt(endf_dict, sum_mt)
                and prop.has_mf5_mt(endf_dict, sum_mt))
        )
    if op == 'ddxs':
        return has_continuous_ddx(endf_dict, sum_mt, zap)
    return False


def _resolve_x_production(endf_dict, mt, zap, op):
    """Decide MT 201..207 vs. partials admission for `(zap, op)`.

    Returns False to explicitly exclude `mt` under this policy; None
    to defer to the existing reaction/sum-rule logic (so partial-MT
    selection still respects user_mts, ancestor rules, etc.); True
    only when admitting MT 201..207 as the active source. Never mix
    the two paths.
    """
    if zap is None or op is None:
        return None
    particle = physconst.ZAP_PARTICLE.get(zap)
    if particle is None:
        return None
    sum_mt = reac.get_x_production_sum_mt(particle)
    if sum_mt is None:
        return None

    proj = prop.get_projectile(endf_dict)
    avail_mts = list(endf_dict.get(3, {}).keys())
    partials = reac.get_x_production_partials(proj, sum_mt, avail_mts)

    is_sum = (mt == sum_mt)
    if not is_sum and mt not in partials:
        return None  # mt unrelated to this decision

    if _partials_can_answer(endf_dict, partials, zap, op):
        # Partials win. Drop the sum; defer partials to existing logic.
        return False if is_sum else None
    if is_sum:
        # Partials can't answer; use the sum if it has data.
        return _x_production_sum_can_answer(endf_dict, sum_mt, zap, op)
    # Partials can't answer and `mt` is a partial; exclude.
    return False


def satisfies_select_heuristic(endf_dict, mt, user_mts=None, *, op=None, zap=None):
    if user_mts is not None:
        if not (hasattr(user_mts, '__iter__') or
                hasattr(user_mts, '__contains__')):
            user_mts = [user_mts]
        user_mts = set(user_mts)

    x_decision = _resolve_x_production(endf_dict, mt, zap, op)
    if x_decision is not None:
        return x_decision

    if not reac.is_sum_mt(mt) and not reac.is_in_sum_mt(mt):
        # no sum rule involved so we select the mt number
        # (and only if mt in user_mts, if provided)
        if user_mts is not None:
            return mt in user_mts
        return True

    # a sum rule is involved
    module_logger.debug(f'sum rule involved for MT={mt}')

    if user_mts is not None:
        # the current mt has only a chance of being selected
        # if any ancestor mt is listed in user_mts or the
        # current mt is directly listed in user_mts
        module_logger.debug('user_mts provided')
        has_user_mt_ancestor = reac.any_ancestor_in_mts(mt, user_mts)
        if not has_user_mt_ancestor and mt not in user_mts:
            module_logger.debug(
                f'no ancestor in user_mts and MT={mt} not in user_mts, '
                f'hence skipping inclusion of MT={mt}.'
            )
            return False

    # check if detailed distribution info available
    # for child mts (determined by sum rules) of current mt
    module_logger.debug(f'check availability of distribution info for MT={mt}')
    child_mts_avail_mf4 = reac.exist_associated_child_mts(mt, endf_dict.get(4, {}))
    child_mts_avail_mf5 = reac.exist_associated_child_mts(mt, endf_dict.get(5, {}))
    child_mts_avail_mf6 = reac.exist_associated_child_mts(mt, endf_dict.get(6, {}))

    if (child_mts_avail_mf4 or child_mts_avail_mf5 or child_mts_avail_mf6):
        # even if current mt is in user_mts,
        # summing of child mts is preferred
        # if detailed distribution info available for them
        module_logger.debug(
            f'child mts with distribution info available for MT={mt}, '
            f'hence skipping inclusion of MT={mt}.'
        )
        return False

    module_logger.debug(f'no distribution for child mts available for MT={mt}')
    # we arrive here only if summing of child mts not preferred
    # so if current mt in user_mts, we select it
    if user_mts is not None and mt in user_mts:
        module_logger.debug(f'selecting MT={mt} because included in user_mts')
        return True

    # if detailed distribution info is available for current mt
    # or there is no known parent mt, we select it
    mt_in_mf4 = mt in endf_dict.get(4, {})
    mt_in_mf5 = mt in endf_dict.get(5, {})
    mt_in_mf6 = mt in endf_dict.get(6, {})
    has_ancestor = reac.any_ancestor_in_mts(mt, endf_dict.get(3, {}))
    if mt_in_mf4 or mt_in_mf5 or mt_in_mf6 or not has_ancestor:
        module_logger.debug(
            f'selecting MT={mt} because distribution available '
            'or it has no ancestor'
        )
        return True

    module_logger.debug(f'none of the selection rules applied, not selecting MT={mt}')
    return False
