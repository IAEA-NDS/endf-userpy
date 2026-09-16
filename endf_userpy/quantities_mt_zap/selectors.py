from ..primitives import properties as prop
from ..primitives import reactions as reac
from ..primitives import physical_constants as physconst
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6help
from .quantities import (
    get_reaction_mt_numbers
)
import logging
import warnings


module_logger = logging.getLogger(__name__)


# Per-(file, parent-MT) dedup for the "gappy child coverage" warning
# emitted by satisfies_select_heuristic below (issue #108 / audit D8).
# Keyed on (id(endf_dict), parent_mt) so the same call site does not
# storm its warning across every ejectile and every MT iteration
# within one query pass. Cleared for tests via
# `_gappy_children_warned.clear()`.
_gappy_children_warned = set()


def _warn_gappy_child_coverage(endf_dict, parent_mt):
    """Emit one summary UserWarning per (file, parent_mt) when the
    heuristic drops `parent_mt` in favour of children that carry
    detailed distributions in MF4/5/6 -- but at least one sibling
    child has an MF3 cross section without any MF4/5/6/12/13 detail,
    so its contribution is silently missed by the resulting
    child-summed query.

    Rare in modern evaluations (which enumerate the discrete-level
    child series completely), but silent under-count when it hits is
    hard to diagnose without knowing the heuristic. Issue #108
    (audit D8): behaviour is unchanged; the warning is signal-only.
    """
    if not reac.is_sum_mt(parent_mt):
        return
    mf3 = endf_dict.get(3, {})
    if parent_mt not in mf3:
        # Parent has no MF3 XS to fall back on either; there is
        # nothing to under-count against, so no signal to emit.
        return
    part_mts = reac.get_part_mts_from_sum_mt(parent_mt)
    mf4 = endf_dict.get(4, {})
    mf5 = endf_dict.get(5, {})
    mf6 = endf_dict.get(6, {})
    mf12 = endf_dict.get(12, {})
    mf13 = endf_dict.get(13, {})
    # A "gappy" child is one that (a) has an MF3 cross section but
    # NO MF4/5/6/12/13 detail AND (b) is itself a leaf in the sum
    # tree (not a sum-MT whose own children get iterated
    # separately). Skipping intermediate sum-MTs is essential:
    # e.g. MT3 is in SUM_RULES[1] but its own children (MT4, MT16,
    # ...) are admitted downstream, so MT3's own MF3 XS is NOT
    # silently missed even when the heuristic drops MT1 in favour
    # of MT2. Only terminal leaves (MT52 in SUM_RULES[4], MT602 in
    # SUM_RULES[103], ...) are genuinely lost.
    gappy = [
        pm for pm in part_mts
        if pm in mf3
        and pm not in mf4 and pm not in mf5 and pm not in mf6
        and pm not in mf12 and pm not in mf13
        and not reac.is_sum_mt(pm)
    ]
    if not gappy:
        return
    key = (id(endf_dict), int(parent_mt))
    if key in _gappy_children_warned:
        return
    _gappy_children_warned.add(key)
    gappy_str = ', '.join(f'MT={m}' for m in gappy[:10])
    if len(gappy) > 10:
        gappy_str += f', ... ({len(gappy)} total)'
    warnings.warn(
        f'Sum-MT admission heuristic dropped parent MT={parent_mt} '
        f'in favour of children with detailed MF4/5/6 distributions, '
        f'but the following sibling child MT(s) carry only an MF3 '
        f'cross section (no MF4/5/6/12/13 detail) and are silently '
        f'missed by the current query: {gappy_str}. To recover the '
        f'missing contribution, query MT={parent_mt} directly '
        f'(bypasses the heuristic) or ask for the missing child MTs '
        f'individually. See issue #108.',
        UserWarning,
        stacklevel=3,
    )


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


def has_mf6_law1_discrete_lines(endf_dict, mt, zap):
    """Whether (MT, ZAP) carries MF6/LAW=1 discrete-energy lines (ND>0).

    LAW=1 subsections can encode a fixed set of discrete outgoing
    energies (Dirac peaks at the first ND tabulated E_out points)
    in addition to a continuum. `compute_dist2d_values` currently
    silently drops the discrete part (issue #27); the broadening
    dispatchers use this predicate to warn users when the file
    they're processing has content that will be missing from the
    broadened result.
    """
    if not prop.has_mf6_mt(endf_dict, mt):
        return False
    return mf6help.has_disc_part(endf_dict, mt, zap)


def has_mf12_discrete_lines(endf_dict, mt, zap):
    """Whether (MT, ZAP) carries discrete photon lines in MF12.

    True if MF12 declares at least one photon at Eg > 0 for this MT
    (the Eg=0 entry, when present, is the continuum-spectrum
    placeholder handled separately via MF15). Gamma-only: the
    predicate returns False for any non-gamma ZAP.

    Used by the broadening dispatcher for gamma dxs/dE (and later
    DDX) to admit MF12-declaring MTs into an additional sum term
    that folds each discrete photon line with the kernel, mirroring
    the MF6/LAW=1 discrete-line path.
    """
    if zap != physconst.PARTICLE_ZAP['g']:
        return False
    if not prop.has_mf12_mt(endf_dict, mt):
        return False
    from ..mfsec_interpretation import mf12_interpretation as mf12_interp
    pes = mf12_interp.get_photon_energies(endf_dict, mt)
    if pes is None:
        return False
    import numpy as _np
    return bool(_np.any(_np.asarray(pes) > 0.0))


def has_mf13_discrete_lines(endf_dict, mt, zap):
    """Whether (MT, ZAP) carries discrete photon lines in MF13.

    True if MF13 declares at least one photon at Eg > 0 for this MT.
    A Eg=0 subsection in MF13, when present, is the continuum-
    spectrum placeholder combined with MF15 (analogous to the MF12
    Eg=0 placeholder convention). Gamma-only.

    Sibling of `has_mf12_discrete_lines`. Files that carry per-
    partial-channel gamma yields in MF13 (typical for ENDF/B-VIII
    medium/heavy nuclei) admit through this predicate for the
    broadened dxs/dE and DDX folders (issue #101).
    """
    if zap != physconst.PARTICLE_ZAP['g']:
        return False
    if not prop.has_mf13_mt(endf_dict, mt):
        return False
    from ..mfsec_interpretation import mf13_interpretation as mf13_interp
    pes = mf13_interp.get_photon_energies(endf_dict, mt)
    if pes is None:
        return False
    import numpy as _np
    return bool(_np.any(_np.asarray(pes) > 0.0))


def has_discrete_two_body_ddx(endf_dict, mt, zap):
    """Whether (MT, ZAP) carries a 2-body kinematic-delta distribution.

    Returns True for channels whose secondary distribution lives on a
    1D curve in (E_out, mu) determined by 2-body kinematics: MF6 with
    LAW=2/3/4 angular distributions, or MF4-only (typical for MT 2
    elastic in a neutron file). These channels are normally excluded
    from cumulative DDX sums because they cannot be represented on a
    finite (E_out, mu) grid; with a broadening kernel along E_out
    they become plottable.
    """
    if prop.has_mf6_mt(endf_dict, mt):
        return mf6help.has_angdist_part(endf_dict, mt, zap)
    if prop.has_mf4_mt(endf_dict, mt) and not prop.has_mf5_mt(endf_dict, mt):
        return zap == physconst.get_zap_for_particle('n')
    return False


def contains_zap(endf_dict, mt, zap):
    if mt in (18, 19, 20, 21, 38):
        return zap == physconst.PARTICLE_ZAP['n']
    # Photons for partial channels (typically inelastic MTs 51..90 and
    # capture-related MTs) are declared in MF12 (yields) or MF13
    # (production cross sections) in most modern libraries (ENDF/B-VIII.1,
    # JEFF-4.0, TENDL), independent of MF6 which usually carries only
    # the scattered neutron. Trust MF12/MF13 as authoritative for gamma
    # so those partial channels are admitted into the gamma production
    # sum (issue #29). JENDL-5-style files that put gamma yields in MF6
    # still take the MF6 branch below when no MF12/MF13 is present.
    if zap == physconst.PARTICLE_ZAP['g']:
        if prop.has_mf12_mt(endf_dict, mt) or prop.has_mf13_mt(endf_dict, mt):
            return True
    if prop.has_mf6_mt(endf_dict, mt):
        return mf6help.contains_zap(endf_dict, mt, zap)
    if not reac.is_known_reaction_mt(mt) or reac.is_x_particle_production_mt(mt):
        # HEATR-injected MTs (301..450), dosimetry/bookkeeping MTs
        # (251..253, 451+), and similar non-reaction MF3 entries carry
        # no ejectile information. MT 201..207 are X-particle
        # production sums that are redundant with the partial channels
        # when those exist; promoting them to a fallback source needs
        # sum-rule logic and is tracked separately.
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
        # Diagnostic (issue #108): warn if some sibling child of `mt`
        # has an MF3 cross section but no MF4/5/6/12/13 detail. That
        # child would be admitted only via the escape-hatch branch
        # below (`or not has_ancestor`), which fires only when no
        # ancestor is in MF3 -- and here the ancestor `mt` IS in MF3,
        # so the child is silently dropped. Warning is signal-only;
        # per-(file, parent) dedup so a run over many MTs and many
        # ejectiles emits at most one warning per parent per query.
        _warn_gappy_child_coverage(endf_dict, mt)
        return False

    module_logger.debug(f'no distribution for child mts available for MT={mt}')
    # we arrive here only if summing of child mts not preferred
    # so if current mt in user_mts, we select it
    if user_mts is not None and mt in user_mts:
        module_logger.debug(f'selecting MT={mt} because included in user_mts')
        return True

    # if detailed distribution info is available for current mt
    # or there is no known parent mt, we select it. MF12 and MF13
    # count as "detailed distribution info" for the purpose of this
    # check: they carry per-photon yields (MF12) or per-photon
    # production cross sections (MF13) that reconstruct the gamma
    # differential quantities, and dropping their MT from a sum
    # query (like `(n,total)` gamma) silently under-counts the file
    # (issue #53). MF14 and MF15 are companion sections that only
    # accompany MF12/MF13, so admitting on MF12/MF13 alone is
    # sufficient. The gate is zap-independent -- `contains_zap`
    # upstream still filters (mt, zap) pairs, so admitting on MF12
    # here cannot over-include a neutron/proton/etc. query on an
    # MT that only produces gammas.
    mt_in_mf4 = mt in endf_dict.get(4, {})
    mt_in_mf5 = mt in endf_dict.get(5, {})
    mt_in_mf6 = mt in endf_dict.get(6, {})
    mt_in_mf12 = mt in endf_dict.get(12, {})
    mt_in_mf13 = mt in endf_dict.get(13, {})
    has_ancestor = reac.any_ancestor_in_mts(mt, endf_dict.get(3, {}))
    if (mt_in_mf4 or mt_in_mf5 or mt_in_mf6
            or mt_in_mf12 or mt_in_mf13
            or not has_ancestor):
        module_logger.debug(
            f'selecting MT={mt} because distribution available '
            'or it has no ancestor'
        )
        return True

    module_logger.debug(f'none of the selection rules applied, not selecting MT={mt}')
    return False
