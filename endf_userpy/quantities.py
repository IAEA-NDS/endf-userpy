import numpy as np
import warnings
from .primitives import physical_constants as physconst
from .primitives import properties as prop
from .primitives import reactions as reac
from .primitives.helpers import unpack_za
from .quantities_mt_zap import quantities as quant_mt_zap
from .quantities_mt_zap import selectors
from .quantities_mt_zap import ddx_broadening as ddxb
import logging
# TODO: Remove direct use of mf6_interpretation module in this module
from .mfsec_interpretation import mf3_interpretation as mf3interp
from .mfsec_interpretation import mf5_interpretation as mf5interp
from .mfsec_interpretation import mf6_interpretation as mf6interp
from .mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from .mfsec_interpretation import mf8_interpretation as mf8interp
from .mfsec_interpretation import mf12_interpretation as mf12interp
from .mfsec_interpretation import mf13_interpretation as mf13interp
from .mfsec_interpretation import mf15_interpretation as mf15interp
from .mfsec_interpretation.mf3_interpretation import (
    above_range_ctx,
    resonance_range_ctx,
)
from .mfsec_interpretation.mf6_law7_integrals import (
    collect_law7_log_errors,
)


# Cache of (id(endf_dict), mt, zap, lfs) tuples we have already warned
# about, so the warning fires once per file+MT+residual+isomer rather
# than once per call. id() can be reused after garbage collection but
# the worst case is a missed warning, never a wrong result.
_isomer_warning_seen = set()


def _warn_if_missing_isomer_routing(
    endf_dict, residual_str, za_residual, lfs, mt5_contrib
):
    """Warn for MTs that could physically produce ``za_residual`` but
    do not resolve the requested isomer state. Called only when the
    user explicitly requested a non-ground LFS.

    Two cases each trigger a per-(file, MT, ZAP, LFS) warning:

    - **No MF8 entry for the MT at all** (the whole isomer routing
      is missing). Handled since the original implementation.
    - **MF8 present but does not declare the requested (ZAP, LFS)**
      (partial isomer coverage; e.g. MF8 declares the ground state
      but not the metastable a user just asked for). Issue #106
      /audit D6 -- the pre-fix code skipped MTs with any MF8 entry
      and silently returned zero for the missing LFS.
    """
    avail_mts = quant_mt_zap.get_reaction_mt_numbers(endf_dict)
    has_mf8 = (8 in endf_dict)
    for mt in sorted(avail_mts):
        if mt == 5 and not mt5_contrib:
            continue
        if not selectors.contains_residual_za(endf_dict, mt, za_residual):
            continue

        mt_has_mf8_section = has_mf8 and mt in endf_dict[8]
        if mt_has_mf8_section:
            # Partial-LFS case: MF8 has an entry for this MT, but
            # does it declare the requested (ZAP, LFS)?
            subsec_nums = mf8interp.find_subsec_nums(
                endf_dict, mt, za_residual, level=lfs,
            )
            if subsec_nums:
                # (ZAP, LFS) IS declared -- routing is available,
                # no need to warn.
                continue
            reason = (
                f"MT={mt} has MF8 routing but no subsection for "
                f"ZAP={int(za_residual)}, LFS={lfs}"
            )
        else:
            reason = f"MT={mt} has no MF8 isomer routing in this file"

        key = (id(endf_dict), mt, int(za_residual), int(lfs))
        if key in _isomer_warning_seen:
            continue
        _isomer_warning_seen.add(key)
        warnings.warn(
            f"{reason}; production of '{residual_str}' from MT={mt} "
            f"is unresolved at the LFS level. Returning 0 for this "
            f"isomer. The metastable may still be produced according "
            f"to other evaluations.",
            UserWarning,
            stacklevel=3,
        )


def _format_lfs_suffix(lfs):
    if lfs == 0:
        return 'g'
    if lfs == 1:
        return 'm'
    return f'm{lfs}'


def _format_residual(zap, lfs):
    z, a = unpack_za(zap)
    sym = physconst.ELEMENT_SYMBOLS[z]
    return f'{sym}-{a}{_format_lfs_suffix(lfs)}'


module_logger = logging.getLogger(__name__)


# TODO: check and complete


def get_available_reactions(endf_dict):
    mts = quant_mt_zap.get_reaction_mt_numbers(endf_dict)
    reacs = []
    for mt in mts:
        if not reac.is_known_reaction_mt(mt):
            module_logger.debug(
                f'skipping MT={mt} from available reactions '
                f'(not in reaction table; e.g. HEATR heating number)'
            )
            continue
        reacs.append(prop.get_reaction_string_for_mt(endf_dict, mt))
    return reacs


def get_declared_residuals(endf_dict):
    """List residuals declared via MF8, with isomer suffix.

    Returns a sorted list of strings like ``["Co-58g", "Co-58m",
    "Co-60g", "Co-60m"]``. Empty list if MF8 is absent. The result
    is what the file explicitly says, not what physics-based MT/ZAP
    accounting would predict.
    """
    pairs = mf8interp.get_declared_zap_lfs(endf_dict)
    return [
        _format_residual(zap, lfs)
        for zap, lfs in sorted(pairs)
    ]


def is_residual_declared(endf_dict, residual_nucleus):
    """True iff the residual (with isomer state) is declared in MF8.

    The residual string follows the same format as for
    ``get_residual_production_xs`` (e.g. ``"Co-58m"``,
    ``"27-Co-58g"``). If no isomer suffix is given, returns True iff
    the bare nucleus is declared in any state.
    """
    zap, lfs = physconst.get_za_for_residual_nucleus(residual_nucleus)
    if lfs is None:
        return len(mf8interp.get_declared_lfs_for_zap(endf_dict, zap)) > 0
    return mf8interp.is_zap_lfs_declared(endf_dict, zap, lfs)


def get_declared_isomer_states(endf_dict, residual_nucleus):
    """Isomer-state suffixes declared in MF8 for a given nucleus.

    Any isomer suffix on the input is ignored: the function asks
    "which states of this nucleus does the file declare?". For
    ``"Co-58"`` it returns e.g. ``["g", "m"]``. Empty list if the
    nucleus has no MF8 entry.
    """
    zap, _ = physconst.get_za_for_residual_nucleus(residual_nucleus)
    return [
        _format_lfs_suffix(lfs)
        for lfs in mf8interp.get_declared_lfs_for_zap(endf_dict, zap)
    ]


def get_incident_energies(endf_dict, reaction):
    """Sorted union of tabulated incident-energy meshes across every
    MT admitted for `reaction`.

    Returns an empty float ndarray if the reaction resolves to no MT
    present in the file (issue #75: previously raised
    ``ValueError: need at least one array to concatenate`` from the
    unguarded ``np.concatenate([])`` call). Callers using this
    function to introspect "does this file have this reaction" can
    now check ``len(result) == 0`` instead of wrapping in try/except.

    Uses `mf3_interpretation.get_incident_energies(mt)` per MT (issue
    #74: the previous callsite reached for a non-existent
    `quant_mt_zap.get_incident_energies` and raised `AttributeError`
    on every call).
    """
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    mts = quant_mt_zap.get_reaction_mt_numbers(endf_dict)
    select_mts = [
        mt for mt in mts
        if selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
    ]
    module_logger.debug('selected ' + ','.join(str(mt) for mt in select_mts))
    energy_meshes = [
        mf3interp.get_incident_energies(endf_dict, mt)
        for mt in select_mts
    ]
    if not energy_meshes:
        return np.array([], dtype=float)
    return np.unique(np.concatenate(energy_meshes))


def get_emission_energies(endf_dict, reaction, particle, nofail=False):
    """Sorted union of tabulated outgoing-energy meshes for `particle`
    across every MT admitted for `reaction`.

    Walks MF6 (all ZAPs, LAW=1/7 tabulated Ep meshes), MF12
    (discrete photon lines for gamma), MF13 (discrete photon lines
    for gamma), MF15 (continuous photon Eout mesh for gamma), and
    MF5 (LF=1 tabulated neutron Eout mesh) as appropriate for the
    requested ejectile. Photon Eg=0 placeholders in MF12/MF13 are
    treated as continuum markers and skipped; the MF15 tabulated
    mesh provides the corresponding continuum coverage.

    LAW=2/3/4/5/6 MF6 subsections do not tabulate an Ep mesh
    (angular-distribution-only or n-body phase-space); they
    contribute an empty mesh via the leaf walker (issue #87 mode A)
    rather than raising.

    Returns an empty float ndarray if no MT in the file both matches
    the reaction and declares the requested ejectile (issue #75).

    Historical note: before this expansion the function walked only
    MF6, which returned an empty mesh on files whose gamma content
    lives entirely in MF12/MF13/MF15 (typical for ENDF/B-VIII
    medium/heavy nuclei; issue #107 / audit D7) and for the MF4+MF5
    neutron representation. `nofail` is now vestigial (the leaf
    walker always returns empty for LAWs it doesn't handle) but is
    kept for signature compatibility.
    """
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    gamma_zap = physconst.get_zap_for_particle('g')
    neutron_zap = physconst.get_zap_for_particle('n')
    mts = quant_mt_zap.get_reaction_mt_numbers(endf_dict)
    select_mts = [
        mt for mt in mts
        if selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        and selectors.contains_zap(endf_dict, mt, zap)
    ]
    module_logger.debug('selected ' + ','.join(str(mt) for mt in select_mts))

    energy_meshes = []
    for mt in select_mts:
        # MF6 walker: LAW=1/7 tabulated Ep meshes. The
        # `contains_zap` gate skips MTs whose MF6 subsection lacks
        # the requested ZAP (issue #77); other MFs still contribute
        # via the branches below.
        if prop.has_mf6_mt(endf_dict, mt) and mf6_help.contains_zap(
            endf_dict, mt, zap,
        ):
            energy_meshes.append(
                mf6interp.get_emission_energies(endf_dict, mt, zap, nofail),
            )
        if zap == gamma_zap:
            # MF12: discrete photon lines. get_photon_energies
            # returns None when LI=1 (isotropic, no photon energies
            # provided) so guard.
            if prop.has_mf12_mt(endf_dict, mt):
                pes = mf12interp.get_photon_energies(endf_dict, mt)
                if pes is not None:
                    pes = np.asarray(pes, dtype=float)
                    energy_meshes.append(pes[pes > 0.0])
            # MF13: same discrete-line pattern.
            if prop.has_mf13_mt(endf_dict, mt):
                pes = mf13interp.get_photon_energies(endf_dict, mt)
                if pes is not None:
                    pes = np.asarray(pes, dtype=float)
                    energy_meshes.append(pes[pes > 0.0])
            # MF15: continuous photon spectrum's Eout mesh.
            if prop.has_mf15_mt(endf_dict, mt):
                energy_meshes.append(np.asarray(
                    mf15interp.get_photon_energies(endf_dict, mt),
                    dtype=float,
                ))
        elif zap == neutron_zap:
            # MF5: tabulated (LF=1) neutron outgoing-energy mesh.
            # Analytic LFs contribute empty (see the mf5interp
            # helper's docstring).
            if prop.has_mf5_mt(endf_dict, mt):
                energy_meshes.append(
                    mf5interp.get_emission_energies(endf_dict, mt),
                )

    # Drop any empty mesh (contributions from unimplemented
    # LAWs / LFs) before the union.
    energy_meshes = [m for m in energy_meshes if len(m) > 0]
    if not energy_meshes:
        return np.array([], dtype=float)
    return np.unique(np.concatenate(energy_meshes))


def get_reaction_xs(
    endf_dict, reaction, energies_in, mt5_contrib=True,
    above_range='warn_nan', resonance_range='warn',
):
    """Cross section for `reaction` on the incident energy grid.

    `above_range` (default ``'warn_nan'``) controls how the library
    handles incident energies above the file's upper Ein boundary.
    See ``mfsec_interpretation.mf3_interpretation.compute_cross_section``
    for the full set of policies (``'warn_nan' | 'nan' | 'warn_zero' |
    'zero' | 'raise'``). The setting is inherited by every internal
    call to ``compute_cross_section`` for the duration of this call.

    `resonance_range` (default ``'warn'``) controls the handling of
    Ein points inside the file's resolved-resonance region (LRU=1
    in MF2/MT151). MF3 there is a subtractive background that the
    library cannot reconstruct against MF2, so raw MF3 in the RRR
    is not the physical cross section and can be negative (issue
    #84). Policies: ``'warn' | 'warn_nan' | 'nan' | 'raise'``.
    Silent on files without an LRU=1 range.
    """
    with above_range_ctx(above_range), resonance_range_ctx(resonance_range):
        return _get_reaction_xs_impl(
            endf_dict, reaction, energies_in, mt5_contrib,
        )


def _get_reaction_xs_impl(endf_dict, reaction, energies_in, mt5_contrib):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    avail_mts = set(quant_mt_zap.get_reaction_mt_numbers(endf_dict))
    iter_mts = avail_mts.copy()
    iter_mts.update(user_mts)
    xs = np.zeros_like(energies_in, dtype=float)
    proj = prop.get_projectile(endf_dict)
    for mt in sorted(iter_mts):
        module_logger.debug(f'consider MT={mt} for reaction xs')
        should_select = selectors.satisfies_select_heuristic(
            endf_dict, mt, user_mts
        )
        mt_available = mt in avail_mts
        if should_select and mt_available:
            module_logger.debug(f'select MT={mt} for reaction xs')
            cur_xs = quant_mt_zap.compute_xs(endf_dict, mt, energies_in)
            xs += cur_xs

        # add associated MT5 component if available and permissible
        if (mt in user_mts
                and mt5_contrib
                and 5 not in user_mts
                and not reac.any_ancestor_in_mts(5, user_mts)
                and reac.is_unique_path_to_residual(proj, mt)):
            if not mt_available or not should_select:
                cur_xs = np.zeros_like(energies_in, dtype=float)
            # `mt_available` is a Python bool (from `mt in avail_mts`),
            # so the pre-fix `~np.bool(mt_available)` cast was doing
            # nothing useful (and `np.bool` was removed in numpy 1.24
            # anyway -- issue #15). Ternary form makes the two branches
            # explicit: if the MT is not in the file, include every
            # incident energy in the MT5 sum; otherwise include only
            # those where the MT's own XS is zero.
            eincs_sel = (
                (cur_xs == 0.0) if mt_available
                else np.ones_like(energies_in, dtype=bool)
            )
            mt5_xs = quant_mt_zap.compute_xs_mt5_contrib(
                endf_dict, mt, energies_in[eincs_sel]
            )
            xs[eincs_sel] += mt5_xs
            if np.any(mt5_xs != 0.0):
                module_logger.debug(f'include MF6/MT5 component for MT={mt}')
    return xs


def get_residual_production_xs(
    endf_dict, residual_nucleus, energies_in, mt5_contrib=True,
    above_range='warn_nan', resonance_range='warn',
):
    """Residual-production cross section for `residual_nucleus`.

    `above_range` (default ``'warn_nan'``) and `resonance_range`
    (default ``'warn'``) match ``get_reaction_xs``; see its
    docstring for the policy sets.
    """
    with above_range_ctx(above_range), resonance_range_ctx(resonance_range):
        return _get_residual_production_xs_impl(
            endf_dict, residual_nucleus, energies_in, mt5_contrib,
        )


def _get_residual_production_xs_impl(
    endf_dict, residual_nucleus, energies_in, mt5_contrib,
):
    za_residual, level = physconst.get_za_for_residual_nucleus(residual_nucleus)
    if level is not None:
        module_logger.debug(f'user requested isomeric state LFS={level}')
    if level not in (None, 0):
        _warn_if_missing_isomer_routing(
            endf_dict, residual_nucleus, za_residual, level, mt5_contrib
        )
    xs = quant_mt_zap.compute_cumulative_quantity(
        lambda endf_dict, mt: quant_mt_zap.compute_residual_xs(
            endf_dict, mt, za_residual, level, energies_in
        ),
        lambda endf_dict, mt: (
            (mt5_contrib or mt != 5) and
            selectors.contains_residual_za_and_lfs(
                endf_dict, mt, za_residual, level
            ) and
            # Use the residual-tuned admission rule rather than the
            # general-purpose satisfies_select_heuristic. The latter's
            # ancestor-check clause drops ejectile-conserving leaves
            # (MT28, MT107, ...) whose only representation is MF3
            # whenever their sum-tree parent (MT3, MT101) is also in
            # MF3, silently under-counting the residual sum in the
            # exclusive-channel energy range (issue #120). The
            # residual-tuned rule admits every leaf and drops sum-MTs
            # only when their children are present in MF3 (would
            # double-count), which is the only correctness concern
            # here.
            selectors.satisfies_residual_select(endf_dict, mt)
        ),
        endf_dict
    )
    if xs is None:
        xs = np.zeros_like(energies_in, dtype=float)
    return xs


def get_particle_production_xs(
    endf_dict, reaction, particle, energies_in,
    above_range='warn_nan', resonance_range='warn',
):
    """Particle-production cross section on the incident energy grid.

    `above_range` (default ``'warn_nan'``) and `resonance_range`
    (default ``'warn'``) match ``get_reaction_xs``; see its
    docstring for the policy sets.
    """
    with above_range_ctx(above_range), resonance_range_ctx(resonance_range):
        return _get_particle_production_xs_impl(
            endf_dict, reaction, particle, energies_in,
        )


def _get_particle_production_xs_impl(
    endf_dict, reaction, particle, energies_in,
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        quant_mt_zap.compute_prodxs,
        lambda endf_dict, mt, zap, energies_in: (
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
            and selectors.contains_zap(endf_dict, mt, zap)
        ),
        endf_dict, zap, energies_in
    )


def get_particle_production_dxs_dE(
    endf_dict, reaction, particle, energies_in, energies_out,
    broadening=None, above_range='warn_nan', resonance_range='warn',
):
    """Energy-differential cross section for particle production.

    Parameters
    ----------
    endf_dict, reaction, particle, energies_in, energies_out
        As before.
    broadening : None or float or (callable, float), optional
        If None (default), behaviour is unchanged: discrete-level
        channels appear as the kinematic-box shape produced by the
        Jacobian transformation in `compute_dexs`. If broadening is
        supplied, the same kernel is applied to every admitted MT
        along E_out, including discrete and continuum channels alike,
        which smooths the box-edge singularities and lets the
        spectrum be compared with finite-resolution measurements.

        Same accepted forms as `get_particle_production_ddxs`.
    above_range : str, default ``'warn_nan'``
        Policy for incident energies above the file's upper Ein
        boundary. Matches ``get_reaction_xs``; see its docstring.
    resonance_range : str, default ``'warn'``
        Policy for incident energies inside the file's
        resolved-resonance region. Matches ``get_reaction_xs``.
    """
    with above_range_ctx(above_range), resonance_range_ctx(resonance_range):
        return _get_particle_production_dxs_dE_impl(
            endf_dict, reaction, particle, energies_in, energies_out,
            broadening,
        )


def _get_particle_production_dxs_dE_impl(
    endf_dict, reaction, particle, energies_in, energies_out, broadening,
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    kernel, kernel_width = _normalize_broadening(broadening)

    def select(endf_dict, mt, zap, einc, eouts):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        )

    if kernel is None:
        _warn_law1_discrete_dropped_from_unbroadened_dxs_dE(
            endf_dict, zap, user_mts,
        )
        return quant_mt_zap.compute_cumulative_quantity(
            quant_mt_zap.compute_dexs, select,
            endf_dict, zap, energies_in, energies_out,
        )

    def cont_compute(endf_dict, mt, zap, einc, eouts):
        return ddxb.compute_dxs_dE_broadened(
            endf_dict, mt, zap, einc, eouts,
            kernel=kernel, kernel_width=kernel_width,
        )

    def law1_disc_compute(endf_dict, mt, zap, einc, eouts):
        return ddxb.compute_dxs_dE_law1_discrete_broadened(
            endf_dict, mt, zap, einc, eouts,
            kernel=kernel,
        )

    def law1_disc_select(endf_dict, mt, zap, einc, eouts):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf6_law1_discrete_lines(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        )

    def mf12_disc_compute(endf_dict, mt, zap, einc, eouts):
        return ddxb.compute_dxs_dE_mf12_discrete_broadened(
            endf_dict, mt, zap, einc, eouts,
            kernel=kernel,
        )

    def mf12_disc_select(endf_dict, mt, zap, einc, eouts):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf12_discrete_lines(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        )

    def mf13_disc_compute(endf_dict, mt, zap, einc, eouts):
        return ddxb.compute_dxs_dE_mf13_discrete_broadened(
            endf_dict, mt, zap, einc, eouts,
            kernel=kernel,
        )

    def mf13_disc_select(endf_dict, mt, zap, einc, eouts):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf13_discrete_lines(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        )

    cont = quant_mt_zap.compute_cumulative_quantity(
        cont_compute, select,
        endf_dict, zap, energies_in, energies_out,
    )
    law1_disc = quant_mt_zap.compute_cumulative_quantity(
        law1_disc_compute, law1_disc_select,
        endf_dict, zap, energies_in, energies_out,
    )
    mf12_disc = quant_mt_zap.compute_cumulative_quantity(
        mf12_disc_compute, mf12_disc_select,
        endf_dict, zap, energies_in, energies_out,
    )
    mf13_disc = quant_mt_zap.compute_cumulative_quantity(
        mf13_disc_compute, mf13_disc_select,
        endf_dict, zap, energies_in, energies_out,
    )
    parts = [p for p in (cont, law1_disc, mf12_disc, mf13_disc) if p is not None]
    if not parts:
        return None
    total = parts[0]
    for p in parts[1:]:
        total = total + p
    return total


def get_particle_production_dxs_dmu(
    endf_dict, reaction, particle, energies_in, angle_cosines_out,
    above_range='warn_nan', resonance_range='warn',
):
    """Angle-differential cross section for particle production.

    `above_range` (default ``'warn_nan'``) and `resonance_range`
    (default ``'warn'``) match ``get_reaction_xs``; see its
    docstring for the policy sets.
    """
    # `collect_law7_log_errors` aggregates the knot-aware LAW=7
    # integrator's per-segment error estimate across every MT hit
    # by this query, and emits one summary UserWarning on exit if
    # any bracketing table uses log-based E' interpolation
    # (INT>=3). Silent no-op for the INT=1/2 cases that cover the
    # whole current corpus (issue #71).
    with (
        above_range_ctx(above_range),
        resonance_range_ctx(resonance_range),
        collect_law7_log_errors(),
    ):
        return _get_particle_production_dxs_dmu_impl(
            endf_dict, reaction, particle, energies_in, angle_cosines_out,
        )


def _get_particle_production_dxs_dmu_impl(
    endf_dict, reaction, particle, energies_in, angle_cosines_out,
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        quant_mt_zap.compute_daxs,
        lambda endf_dict, mt, zap, energies_in, angle_cosines_out: (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        ),
        endf_dict, zap, energies_in, angle_cosines_out
    )


def get_particle_production_ddxs(
    endf_dict, reaction, particle, energies_in, energies_out, angle_cosines_out,
    broadening=None, above_range='warn_nan', resonance_range='warn',
):
    """Double-differential cross section for particle production.

    Parameters
    ----------
    endf_dict, reaction, particle, energies_in, energies_out, angle_cosines_out
        As before.
    broadening : None or float or (callable, float), optional
        If None (default), only channels with a true continuous
        (E_out, mu) distribution contribute: MF6/LAW=1 continuum,
        LAW=6, LAW=7, or MF4+MF5. Discrete two-body channels (MT 2
        elastic, MT 51..90 discrete inelastic) carry their outgoing
        energy as a kinematic delta at ``E' = E'_kin(mu, E_in)``
        which cannot be represented on the caller's finite E_out
        grid; those MTs are silently excluded from the sum and a
        UserWarning names them (issue #21). Pass a `broadening=`
        to include their peaks.

        If broadening is provided, the result sums (i) the continuous
        DDX folded with the kernel along E_out, and (ii) the discrete
        two-body channels with the kinematic delta replaced by the
        kernel, plus (iii) the MF6/LAW=1 discrete-line channels and
        (iv) the MF12 discrete-line gamma channels.

        Accepted forms:
          - scalar `sigma` (eV) -> Gaussian kernel of that width.
          - tuple `(kernel_callable, width)` -> custom kernel; the
            callable is `kernel(delta_E)` and `width` is its
            characteristic scale (passed to the FFT mesh control).
    above_range : str, default ``'warn_nan'``
        Policy for incident energies above the file's upper Ein
        boundary. Matches ``get_reaction_xs``; see its docstring.
    resonance_range : str, default ``'warn'``
        Policy for incident energies inside the file's
        resolved-resonance region. Matches ``get_reaction_xs``.
    """
    with above_range_ctx(above_range), resonance_range_ctx(resonance_range):
        return _get_particle_production_ddxs_impl(
            endf_dict, reaction, particle, energies_in, energies_out,
            angle_cosines_out, broadening,
        )


def _get_particle_production_ddxs_impl(
    endf_dict, reaction, particle, energies_in, energies_out,
    angle_cosines_out, broadening,
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)

    kernel, kernel_width = _normalize_broadening(broadening)
    if kernel is None:
        _warn_discrete_dropped_from_unbroadened_ddx(
            endf_dict, zap, user_mts,
        )
        # Unbroadened gamma DDX from MF15 continuum + MF14 angular
        # (issue #125). The general compute_ddxs path goes through
        # compute_dist2d_values which handles MF6 XOR MF4+MF5 but
        # has no MF15 branch, so files whose gamma content lives in
        # MF15 (Al-27 MT102, U-238 MT18 with the #126 fix, ...)
        # return zero from that branch. Sum in a separate MF15+MF14
        # contribution the same way the broadened dispatcher does
        # for compute_ddx_mf15_continuum_broadened.
        cont_unbroad = quant_mt_zap.compute_cumulative_quantity(
            quant_mt_zap.compute_ddxs,
            lambda endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out: (
                selectors.contains_zap(endf_dict, mt, zap) and
                selectors.has_continuous_ddx(endf_dict, mt, zap) and
                selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
            ),
            endf_dict, zap, energies_in, energies_out, angle_cosines_out
        )
        mf15_unbroad = quant_mt_zap.compute_cumulative_quantity(
            quant_mt_zap.compute_ddxs_from_mf15_mf14,
            lambda endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out: (
                selectors.contains_zap(endf_dict, mt, zap) and
                selectors.has_mf15_continuum(endf_dict, mt, zap) and
                selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
            ),
            endf_dict, zap, energies_in, energies_out, angle_cosines_out
        )
        parts = [p for p in (cont_unbroad, mf15_unbroad) if p is not None]
        if not parts:
            return None
        total = parts[0]
        for p in parts[1:]:
            total = total + p
        return total

    def cont_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_continuous_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel, kernel_width=kernel_width,
        )

    def cont_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_continuous_ddx(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        )

    def disc_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_discrete_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel,
        )

    def disc_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_discrete_two_body_ddx(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        )

    def law1_disc_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_law1_discrete_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel,
        )

    def law1_disc_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf6_law1_discrete_lines(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        )

    def mf12_disc_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_mf12_discrete_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel,
        )

    def mf12_disc_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf12_discrete_lines(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        )

    def mf13_disc_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_mf13_discrete_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel,
        )

    def mf13_disc_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf13_discrete_lines(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        )

    def mf15_cont_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_mf15_continuum_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel, kernel_width=kernel_width,
        )

    def mf15_cont_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf15_continuum(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        )

    # Sum-then-broaden path (issue #26): when 2+ MTs pass
    # cont_select, computing dist2d for the whole sum inside ONE
    # adaptive_convolve saves the FFT overhead of the per-MT
    # approach. Convolution is linear so the answer is identical up
    # to floating-point summation order. One MT: no gain, fall
    # through to the per-MT path.
    cont_mts = [
        mt for mt in quant_mt_zap.get_reaction_mt_numbers(endf_dict)
        if cont_select(endf_dict, mt, zap,
                       energies_in, energies_out, angle_cosines_out)
    ]
    if len(cont_mts) >= 2:
        cont = ddxb.compute_ddx_continuous_broadened_summed(
            endf_dict, cont_mts, zap,
            energies_in, energies_out, angle_cosines_out,
            kernel=kernel, kernel_width=kernel_width,
        )
    else:
        cont = quant_mt_zap.compute_cumulative_quantity(
            cont_compute, cont_select,
            endf_dict, zap, energies_in, energies_out, angle_cosines_out,
        )
    disc = quant_mt_zap.compute_cumulative_quantity(
        disc_compute, disc_select,
        endf_dict, zap, energies_in, energies_out, angle_cosines_out,
    )
    law1_disc = quant_mt_zap.compute_cumulative_quantity(
        law1_disc_compute, law1_disc_select,
        endf_dict, zap, energies_in, energies_out, angle_cosines_out,
    )
    mf12_disc = quant_mt_zap.compute_cumulative_quantity(
        mf12_disc_compute, mf12_disc_select,
        endf_dict, zap, energies_in, energies_out, angle_cosines_out,
    )
    mf13_disc = quant_mt_zap.compute_cumulative_quantity(
        mf13_disc_compute, mf13_disc_select,
        endf_dict, zap, energies_in, energies_out, angle_cosines_out,
    )
    mf15_cont = quant_mt_zap.compute_cumulative_quantity(
        mf15_cont_compute, mf15_cont_select,
        endf_dict, zap, energies_in, energies_out, angle_cosines_out,
    )
    parts = [
        p for p in (cont, disc, law1_disc, mf12_disc, mf13_disc, mf15_cont)
        if p is not None
    ]
    if not parts:
        return None
    total = parts[0]
    for p in parts[1:]:
        total = total + p
    return total


def _warn_discrete_dropped_from_unbroadened_ddx(endf_dict, zap, user_mts):
    """Emit a UserWarning if the unbroadened DDX call would silently
    drop discrete two-body MTs that pass every other admission check
    (issue #21). Elastic (MT 2) and discrete inelastic (MT 51..90 in
    a neutron file) carry their outgoing energy as a kinematic delta
    at ``E' = E'_kin(mu, E_in)``. A delta on a finite (E', mu) grid
    integrates to zero almost everywhere, so the unbroadened DDX
    dispatcher (which admits only MTs with a continuous DDX)
    correctly excludes them -- but the exclusion is silent, leaving
    users wondering why the (n,n_i) peaks their eye expects at
    ``E' ~ E_in - Q_i`` are missing. Users get real content by
    passing `broadening=` (see the two-body-discrete kernel folder
    in `ddx_broadening.compute_ddx_discrete_broadened`).
    """
    dropped = []
    for mt in quant_mt_zap.get_reaction_mt_numbers(endf_dict):
        if not selectors.contains_zap(endf_dict, mt, zap):
            continue
        if not selectors.satisfies_select_heuristic(endf_dict, mt, user_mts):
            continue
        if selectors.has_continuous_ddx(endf_dict, mt, zap):
            continue
        if not selectors.has_discrete_two_body_ddx(endf_dict, mt, zap):
            continue
        dropped.append(mt)
    if not dropped:
        return
    if len(dropped) > 12:
        mt_str = ', '.join(str(m) for m in dropped[:12]) + f', ... ({len(dropped)} total)'
    else:
        mt_str = ', '.join(str(m) for m in dropped)
    warnings.warn(
        f'get_particle_production_ddxs (unbroadened) dropped discrete '
        f'two-body MTs whose outgoing energy is a kinematic delta at '
        f"E' = E'_kin(mu, E_in): MT={mt_str}. To include their peaks "
        f'in the DDX, pass a `broadening=sigma_eV` (or a custom '
        f'(kernel, width) tuple) so the deltas are folded into a '
        f'finite kernel that plots on the E_out grid.',
        UserWarning, stacklevel=3,
    )


def _warn_law1_discrete_dropped_from_unbroadened_dxs_dE(
    endf_dict, zap, user_mts,
):
    """Emit a UserWarning if the unbroadened `dxs/dE` call would
    silently drop MF6/LAW=1 ND>0 discrete-line channels that pass
    every other admission check (issue #102 / audit D2).

    LAW=1 with ND>0 subsections encodes per-line emissions as
    Dirac deltas at fixed outgoing energies inside a MF6 subsection
    (JENDL-5 partial-inelastic gamma cascades and some capture
    channels use this layout). The unbroadened 1D dispatcher walks
    only the continuum part of each subsection; the discrete deltas
    are correctly excluded from the sum -- a delta on a finite
    grid integrates to zero almost everywhere -- but the exclusion
    is silent. Users see a continuum-only spectrum and no signal
    that the discrete-line content is being dropped.

    Mirrors ``_warn_discrete_dropped_from_unbroadened_ddx`` (issue
    #21) for the DDX case. Users get real content by passing
    ``broadening=sigma_eV``: the same MT then routes through
    ``ddx_broadening.compute_dxs_dE_law1_discrete_broadened`` which
    folds each delta with the kernel and adds it to the continuum
    contribution.
    """
    dropped = []
    for mt in quant_mt_zap.get_reaction_mt_numbers(endf_dict):
        if not selectors.contains_zap(endf_dict, mt, zap):
            continue
        if not selectors.satisfies_select_heuristic(endf_dict, mt, user_mts):
            continue
        if not selectors.has_mf6_law1_discrete_lines(endf_dict, mt, zap):
            continue
        dropped.append(mt)
    if not dropped:
        return
    if len(dropped) > 12:
        mt_str = (
            ', '.join(str(m) for m in dropped[:12])
            + f', ... ({len(dropped)} total)'
        )
    else:
        mt_str = ', '.join(str(m) for m in dropped)
    warnings.warn(
        f'get_particle_production_dxs_dE (unbroadened) dropped '
        f'MF6/LAW=1 ND>0 discrete-line content whose outgoing '
        f'energy is a Dirac delta on the E\' axis: MT={mt_str}. '
        f'The continuum part of these MTs IS included. To include '
        f'the discrete lines too, pass a `broadening=sigma_eV` (or '
        f'a custom (kernel, width) tuple) so the deltas are folded '
        f'into a finite kernel that plots on the E\' grid.',
        UserWarning, stacklevel=3,
    )


def _normalize_broadening(broadening):
    """Translate a user broadening spec into (kernel, width) for the
    low-level folders. None propagates as (None, None)."""
    if broadening is None:
        return None, None
    if isinstance(broadening, (int, float, np.integer, np.floating)):
        sigma = float(broadening)
        if sigma <= 0:
            raise ValueError("broadening sigma must be positive")
        norm = 1.0 / (sigma * np.sqrt(2 * np.pi))

        def gaussian_kernel(d):
            return norm * np.exp(-0.5 * (d / sigma) ** 2)
        return gaussian_kernel, sigma
    try:
        kernel, width = broadening
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "broadening must be None, a positive scalar sigma, or a "
            "(kernel_callable, width) tuple"
        ) from exc
    if not callable(kernel):
        raise ValueError("broadening tuple element 0 must be a callable kernel")
    width = float(width)
    if width <= 0:
        raise ValueError("broadening tuple element 1 (width) must be positive")
    return kernel, width


