import numpy as np
import warnings
from .primitives import array_ns
from .primitives import physical_constants as physconst
from .primitives import properties as prop
from .primitives import reactions as reac
from .primitives.helpers import unpack_za
from .run_options import RunOptions
from .quantities_mt_zap import quantities as quant_mt_zap
from .quantities_mt_zap import selectors
from .quantities_mt_zap import ddx_broadening as ddxb
from .quantities_mt_zap import discrete_gamma_lines as discrete_gamma
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
from .run_options import _QueryState, _emit_summary_warnings


# Cache of (id(endf_dict), mt, zap, lfs) tuples we have already warned
# about, so the warning fires once per file+MT+residual+isomer rather
# than once per call. id() can be reused after garbage collection but
# the worst case is a missed warning, never a wrong result.
_isomer_warning_seen = set()


def _check_particle_production_mode1(
    endf_dict, user_mts, zap, mts, _query_state,
):
    """Issue #311 mode 1 for particle-production entry points.

    Record the user-requested MT when the file carries no MT that
    the particle-production selector would admit for this
    ``(reaction, zap)`` pair. The admission heuristic itself lives
    at the selector layer
    (:func:`selectors.any_mt_admitted_for_particle_production`);
    this helper only decides whether to append to ``_query_state`` and
    stays in the top-level API layer alongside the other policy /
    warning-recording glue.
    """
    if _query_state is None or not user_mts:
        return
    avail_mts = set(quant_mt_zap.get_reaction_mt_numbers(endf_dict))
    missing_user = [mt for mt in user_mts if mt not in avail_mts]
    if not missing_user:
        return
    if selectors.any_mt_admitted_for_particle_production(
        endf_dict, user_mts, zap, mts,
    ):
        return
    _query_state.missing_user_mts.extend(missing_user)


def _check_fission_chance_breakdown_vs_mf2(
    endf_dict, user_mts, reaction, options,
):
    """Raise if the user's reaction string resolves to a chance-
    breakdown fission MT (19/20/21/38) and the file's MF2 carries
    fission widths. See #311.

    MF2 fission widths represent total fission (MT 18); composing
    them with a chance-breakdown MT would overstate the resonance
    contribution above threshold. The honest answer is to point the
    user at MT 18 or at ``include_resonance=False``.
    """
    if not options.include_resonance:
        return
    bad = [mt for mt in user_mts if mt in reac.CHANCE_BREAKDOWN_FISSION_MTS]
    if not bad or not prop.has_mf2_fission_widths(endf_dict):
        return
    raise ValueError(
        f'Reaction {reaction!r} resolves to MT {bad[0]} (chance-'
        f'breakdown fission: first/second/third/fourth-chance). '
        f'This file\'s MF2 carries fission widths that describe '
        f'TOTAL fission (ENDF-6: MT 18 = MT 19 + MT 20 + MT 21 + '
        f'MT 38); composing them with MT {bad[0]} would overstate '
        f'the resonance contribution above threshold. Query '
        f"'(n,fission)' for MT 18 (total fission), or set "
        f'RunOptions(include_resonance=False) to use raw MF3 only.'
    )


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


def _stage_energies(energies_in, options, query_state):
    """Under a ``jax.jit`` trace with a concrete (numpy) query mesh,
    convert the mesh to a jax array once, behind an optimisation
    barrier, and keep the host copy in ``query_state.host_energies``.

    A mesh closed over by a jitted function would otherwise reach every
    layer as a separate compile-time constant: XLA then tries to
    constant-fold everything computed from it that does not depend on
    the traced arguments (minutes of compile time, ~15 GB at 1M
    points), and the composition layer, seeing only tracers, evaluates
    each resonance range on the full mesh instead of its in-range
    points. Eager calls and traced meshes pass through unchanged.
    """
    xp = options.backend
    if getattr(xp, 'name', None) != 'jax' or not array_ns._inside_jit_trace():
        return energies_in
    try:
        e_host = np.asarray(energies_in, dtype=np.float64)
    except Exception:          # traced mesh: nothing to stage
        return energies_in
    import jax
    staged = jax.lax.optimization_barrier(xp.asarray(e_host))
    query_state.host_energies = (staged, e_host)
    return staged


def get_reaction_xs(
    endf_dict, reaction, energies_in, *, options=None,
):
    """Cross section for `reaction` on the incident energy grid.

    Runtime policies (backend, above-range fill, resonance-range
    handling, resonance composition, MT5 catch-all redistribution,
    URR quadrature, JIT mesh bounds) live on
    :class:`endf_userpy.run_options.RunOptions`. Pass
    ``options=RunOptions(...)`` to override any of them; the
    default ``options=None`` resolves to the physics-first default
    ``RunOptions()`` (see its docstring). Issue #143.
    """
    if options is None:
        options = RunOptions()
    query_state = _QueryState()
    energies_in = _stage_energies(energies_in, options, query_state)
    result = _get_reaction_xs_impl(
            endf_dict, reaction, energies_in, options=options,
         _query_state=query_state)
    _emit_summary_warnings(query_state, options)
    return result


def _get_reaction_xs_impl(
    endf_dict, reaction, energies_in, *, options, _query_state=None,
):
    xp = options.backend
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    _check_fission_chance_breakdown_vs_mf2(
        endf_dict, user_mts, reaction, options,
    )
    if _query_state is not None:
        _query_state.user_mts.update(user_mts)
    avail_mts = set(quant_mt_zap.get_reaction_mt_numbers(endf_dict))
    iter_mts = avail_mts.copy()
    iter_mts.update(user_mts)
    energies_in_xp = xp.asarray(energies_in)
    xs = xp.zeros_like(energies_in_xp, dtype=xp.float64)
    proj = prop.get_projectile(endf_dict)
    admitted_count = 0
    # Record top-down fallback if any user MT is not in MF3 (issue #135).
    if (options.aggregation == 'top_down'
            and _query_state is not None):
        mf3_present = endf_dict.get(3, {})
        for user_mt in user_mts:
            if user_mt not in mf3_present:
                _query_state.aggregation_fallback.append(user_mt)
    for mt in sorted(iter_mts):
        module_logger.debug(f'consider MT={mt} for reaction xs')
        should_select = selectors.satisfies_select_heuristic(
            endf_dict, mt, user_mts, aggregation=options.aggregation,
        )
        mt_available = mt in avail_mts
        if should_select and mt_available:
            module_logger.debug(f'select MT={mt} for reaction xs')
            # ``compute_xs`` takes xp and returns xp-native on the
            # resonance-composition branch; MF3-only branch materialises
            # numpy and gets promoted via xp.asarray inside compute_xs.
            cur_xs = quant_mt_zap.compute_xs(
                endf_dict, mt, energies_in,
                options=options, _query_state=_query_state,
            )
            if xp.name == 'jax':
                xs = xs + cur_xs
            else:
                xs += cur_xs     # xs is this function's own buffer
            admitted_count += 1

        # MT5 fallback component: adds a redistributed MT5
        # contribution at Es where the direct MT is zero. Only
        # relevant when the file actually carries MF6/MT=5 data.
        # Under jax the fallback stays xp-native end-to-end (issue
        # #215): ``compute_xs_mt5_contrib`` derives xp from options,
        # and the "only backfill where direct MT is zero" mask uses
        # ``xp.where`` so tracers survive.
        if (mt in user_mts
                and options.mt5_contrib
                and 5 not in user_mts
                and not reac.any_ancestor_in_mts(5, user_mts)
                and reac.is_unique_path_to_residual(proj, mt)
                and prop.has_mf6_mt(endf_dict, 5)):
            energies_in_xp = xp.asarray(energies_in)
            if not mt_available or not should_select:
                cur_xs_ref = xp.zeros_like(
                    energies_in_xp, dtype=xp.float64,
                )
            else:
                cur_xs_ref = cur_xs
            mt5_xs_all = quant_mt_zap.compute_xs_mt5_contrib(
                endf_dict, mt, energies_in,
                options=options, _query_state=_query_state,
            )
            # Backfill: add mt5_xs_all where the direct MT gave zero
            # (or wasn't selected), leave xs unchanged where the
            # direct MT already had a value. Compute the full MT5
            # array on all Es rather than slicing to concrete
            # indices; slice-selection would materialise ``cur_xs``
            # and lose tracer under xp=jax. The extra cost is one
            # MT5 evaluation on Es where the direct MT already
            # covered the reaction; MT5 backfill is only physically
            # meaningful where the direct MT is zero, so this
            # doesn't change the numerical result.
            xs = xp.where(
                cur_xs_ref == 0.0, xs + mt5_xs_all, xs,
            )
            # Debug log fires unconditionally; under jax jit the
            # `xp.any(mt5_xs != 0.0)` guard would trace into a
            # TracerBoolConversionError, and the message is already
            # filtered by the logging level.
            module_logger.debug(f'MF6/MT5 component considered for MT={mt}')
    # Mode 1 (#311): user asked for an MT the file does not carry
    # AND no admitted partial could synthesise it via the sum rule.
    # Signal the empty result so a silent-zero does not look like a
    # correct answer.
    if _query_state is not None and admitted_count == 0:
        for mt in user_mts:
            if mt not in avail_mts:
                _query_state.missing_user_mts.append(mt)
    return xs


def get_residual_production_xs(
    endf_dict, residual_nucleus, energies_in, *, options=None,
):
    """Residual-production cross section for `residual_nucleus`.

    Runtime policies live on
    :class:`endf_userpy.run_options.RunOptions`; see
    :func:`get_reaction_xs`.
    """
    if options is None:
        options = RunOptions()
    query_state = _QueryState()
    result = _get_residual_production_xs_impl(
            endf_dict, residual_nucleus, energies_in, options=options,
         _query_state=query_state)
    _emit_summary_warnings(query_state, options)
    return result


def _get_residual_production_xs_impl(
    endf_dict, residual_nucleus, energies_in, *, options, _query_state=None,
):
    za_residual, level = physconst.get_za_for_residual_nucleus(residual_nucleus)
    if level is not None:
        module_logger.debug(f'user requested isomeric state LFS={level}')
    if level not in (None, 0):
        _warn_if_missing_isomer_routing(
            endf_dict, residual_nucleus, za_residual, level,
            options.mt5_contrib,
        )
    # When the queried residual is the target in its ground state (or
    # the user omitted the isomer suffix, which carries the same
    # intent in practice), apply the activation-library convention:
    # drop target-conserving MTs. Isomer queries (level >= 1) always
    # keep every admitted MT because the isomer is a genuine
    # different nuclide. See issue #137.
    apply_target_conserving_filter = (
        not options.include_target
        and za_residual == prop.get_ZA(endf_dict)
        and (level is None or level == 0)
    )
    xs = quant_mt_zap.compute_cumulative_quantity(
        lambda endf_dict, mt: quant_mt_zap.compute_residual_xs(
            endf_dict, mt, za_residual, level, energies_in,
            options=options, _query_state=_query_state,
        ),
        lambda endf_dict, mt: (
            (options.mt5_contrib or mt != 5) and
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
            selectors.satisfies_residual_select(
                endf_dict, mt, aggregation=options.aggregation,
            ) and
            not (
                apply_target_conserving_filter and
                selectors.is_target_conserving_mt(endf_dict, mt)
            )
        ),
        endf_dict
    )
    if xs is None:
        xs = np.zeros_like(energies_in, dtype=float)
    return xs


def get_particle_production_xs(
    endf_dict, reaction, particle, energies_in, *, options=None,
):
    """Particle-production cross section on the incident energy grid.

    Runtime policies live on
    :class:`endf_userpy.run_options.RunOptions`; see
    :func:`get_reaction_xs`.
    """
    if options is None:
        options = RunOptions()
    query_state = _QueryState()
    energies_in = _stage_energies(energies_in, options, query_state)
    result = _get_particle_production_xs_impl(
            endf_dict, reaction, particle, energies_in, options=options,
         _query_state=query_state)
    _emit_summary_warnings(query_state, options)
    return result


def _get_particle_production_xs_impl(
    endf_dict, reaction, particle, energies_in, *, options, _query_state=None,
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    _check_fission_chance_breakdown_vs_mf2(
        endf_dict, user_mts, reaction, options,
    )
    if _query_state is not None:
        _query_state.user_mts.update(user_mts)
    zap = physconst.get_zap_for_particle(particle)
    # Widened iteration (union of MF3+MF12+MF13+MF15 keys) so MTs
    # that carry gamma production only in MF12/MF13/MF15 without an
    # MF3 entry (JENDL-5 N-14 MT 3 nonelastic) are visited by the
    # cumulative-sum iteration (issue #130). Non-gamma queries
    # over the wider list are still filtered correctly by contains_zap.
    mts = mf3interp.get_reaction_mts_widened(endf_dict)
    _check_particle_production_mode1(endf_dict, user_mts, zap, mts, _query_state)
    # Record top-down fallback if the user's MT is not in MF3
    # (issue #135). Scalar XS entry, so fallback is user-visible.
    if (options.aggregation == 'top_down'
            and _query_state is not None):
        mf3_present = endf_dict.get(3, {})
        for user_mt in user_mts:
            if user_mt not in mf3_present:
                _query_state.aggregation_fallback.append(user_mt)
    query_state_hits = _query_state
    return quant_mt_zap.compute_cumulative_quantity(
        lambda endf_dict, mt, zap, einc: quant_mt_zap.compute_prodxs(
            endf_dict, mt, zap, einc,
            options=options, _query_state=query_state_hits,
        ),
        lambda endf_dict, mt, zap, energies_in: (
            selectors.satisfies_particle_production_select(
                endf_dict, mt, user_mts, zap,
                aggregation=options.aggregation,
            )
            and selectors.contains_zap(endf_dict, mt, zap)
        ),
        endf_dict, zap, energies_in, mts=mts,
    )


def get_particle_production_dxs_dE(
    endf_dict, reaction, particle, energies_in, energies_out, *,
    broadening=None, options=None,
):
    """Energy-differential cross section for particle production.

    Parameters
    ----------
    endf_dict, reaction, particle, energies_in, energies_out
        Physics arguments (dict, MT/particle strings, query axes).
    broadening : None or float or (callable, float), optional
        Measurement-resolution kernel (per-call physics knob; not
        a run-time policy). Same accepted forms as
        :func:`get_particle_production_ddxs`. If None (default),
        Dirac-delta discrete-line content is dropped with a
        UserWarning naming the affected MTs; pass ``broadening=0``
        to suppress that warning while keeping the drop.
    options : RunOptions, optional
        Runtime policies (backend, above-range fill,
        resonance-range handling, MT5 catch-all, URR quadrature,
        broadening-mesh bounds for jit safety). Default None
        resolves to a physics-first
        :class:`endf_userpy.run_options.RunOptions`.
        ``broadening_mesh_bounds`` on the options object is
        required when ``energies_out`` is a jax tracer (see
        Phase 5 of #290).
    """
    if options is None:
        options = RunOptions()
    query_state = _QueryState()
    result = _get_particle_production_dxs_dE_impl(
            endf_dict, reaction, particle, energies_in, energies_out,
            broadening, options=options,
         _query_state=query_state)
    _emit_summary_warnings(query_state, options)
    return result


def _get_particle_production_dxs_dE_impl(
    endf_dict, reaction, particle, energies_in, energies_out, broadening,
    *, options, _query_state=None,
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    _check_fission_chance_breakdown_vs_mf2(
        endf_dict, user_mts, reaction, options,
    )
    if _query_state is not None:
        _query_state.user_mts.update(user_mts)
    zap = physconst.get_zap_for_particle(particle)
    xp = options.backend
    broadening_mesh_bounds = options.broadening_mesh_bounds
    broadening_wkw = options.broadening_window_kernel_widths
    kernel, kernel_width = _normalize_broadening(broadening, xp=xp)
    # Widened MT iteration (issue #130): see _get_particle_production_xs_impl.
    mts = mf3interp.get_reaction_mts_widened(endf_dict)
    _check_particle_production_mode1(endf_dict, user_mts, zap, mts, _query_state)

    def select(endf_dict, mt, zap, einc, eouts):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
        )

    if kernel is None:
        if not _broadening_explicit_no_kernel(broadening):
            _warn_law1_discrete_dropped_from_unbroadened_dxs_dE(
                endf_dict, zap, user_mts,
            )
            _warn_mf12_mf13_discrete_dropped(
                endf_dict, zap, user_mts,
                'get_particle_production_dxs_dE',
            )
        return quant_mt_zap.compute_cumulative_quantity(
            lambda endf_dict, mt, zap, einc, eouts:
                quant_mt_zap.compute_dexs(
                    endf_dict, mt, zap, einc, eouts,
                    options=options, _query_state=_query_state,
                ),
            select,
            endf_dict, zap, energies_in, energies_out,
            mts=mts,
        )

    def cont_compute(endf_dict, mt, zap, einc, eouts):
        return ddxb.compute_dxs_dE_broadened(
            endf_dict, mt, zap, einc, eouts,
            kernel=kernel, kernel_width=kernel_width, xp=xp,
            options=options, _query_state=_query_state,
            mesh_bounds=broadening_mesh_bounds,
            window_kernel_widths=broadening_wkw,
        )

    def _admitted_cont_mts():
        # Collect MTs the per-MT continuous folder would visit, so
        # the coalesced summed-FFT path (issue #277) can run one
        # adaptive_convolve across them instead of one per MT.
        return [
            mt for mt in mts
            if select(endf_dict, mt, zap, energies_in, energies_out)
        ]

    def law1_disc_compute(endf_dict, mt, zap, einc, eouts):
        return ddxb.compute_dxs_dE_law1_discrete_broadened(
            endf_dict, mt, zap, einc, eouts,
            kernel=kernel, xp=xp,
            options=options, _query_state=_query_state,
        )

    def law1_disc_select(endf_dict, mt, zap, einc, eouts):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf6_law1_discrete_lines(endf_dict, mt, zap) and
            selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
        )

    def mf12_disc_compute(endf_dict, mt, zap, einc, eouts):
        return ddxb.compute_dxs_dE_mf12_discrete_broadened(
            endf_dict, mt, zap, einc, eouts,
            kernel=kernel, xp=xp,
            options=options, _query_state=_query_state,
        )

    def mf12_disc_select(endf_dict, mt, zap, einc, eouts):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf12_discrete_lines(endf_dict, mt, zap) and
            selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
        )

    def mf13_disc_compute(endf_dict, mt, zap, einc, eouts):
        return ddxb.compute_dxs_dE_mf13_discrete_broadened(
            endf_dict, mt, zap, einc, eouts,
            kernel=kernel, xp=xp,
        )

    def mf13_disc_select(endf_dict, mt, zap, einc, eouts):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf13_discrete_lines(endf_dict, mt, zap) and
            selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
        )

    admitted_cont_mts = _admitted_cont_mts()
    if len(admitted_cont_mts) >= 2:
        # Coalesced-summed FFT: one adaptive_convolve for the sum
        # of every admitted MT's continuous dxs/dE, instead of one
        # per MT (issue #277). ~30-50x runtime cut on files where
        # widening the reaction string admits many MTs
        # (e.g. U-233 (n,g) with ~100+ partial channels).
        cont = ddxb.compute_dxs_dE_broadened_summed(
            endf_dict, admitted_cont_mts, zap,
            energies_in, energies_out,
            kernel=kernel, kernel_width=kernel_width, xp=xp,
            options=options, _query_state=_query_state,
            mesh_bounds=broadening_mesh_bounds,
            window_kernel_widths=broadening_wkw,
        )
    else:
        cont = quant_mt_zap.compute_cumulative_quantity(
            cont_compute, select,
            endf_dict, zap, energies_in, energies_out,
            mts=mts,
        )
    law1_disc = quant_mt_zap.compute_cumulative_quantity(
        law1_disc_compute, law1_disc_select,
        endf_dict, zap, energies_in, energies_out,
        mts=mts,
    )
    mf12_disc = quant_mt_zap.compute_cumulative_quantity(
        mf12_disc_compute, mf12_disc_select,
        endf_dict, zap, energies_in, energies_out,
        mts=mts,
    )
    mf13_disc = quant_mt_zap.compute_cumulative_quantity(
        mf13_disc_compute, mf13_disc_select,
        endf_dict, zap, energies_in, energies_out,
        mts=mts,
    )
    parts = [p for p in (cont, law1_disc, mf12_disc, mf13_disc) if p is not None]
    if not parts:
        return None
    total = parts[0]
    for p in parts[1:]:
        total = total + p
    return total


def get_particle_production_dxs_dmu(
    endf_dict, reaction, particle, energies_in, angle_cosines_out, *,
    options=None,
):
    """Angle-differential cross section for particle production.

    Runtime policies live on
    :class:`endf_userpy.run_options.RunOptions`; see
    :func:`get_reaction_xs`.
    """
    if options is None:
        options = RunOptions()
    query_state = _QueryState()
    result = _get_particle_production_dxs_dmu_impl(
            endf_dict, reaction, particle, energies_in, angle_cosines_out,
            options=options,
         _query_state=query_state)
    _emit_summary_warnings(query_state, options)
    return result


def _get_particle_production_dxs_dmu_impl(
    endf_dict, reaction, particle, energies_in, angle_cosines_out, *,
    options, _query_state=None,
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    _check_fission_chance_breakdown_vs_mf2(
        endf_dict, user_mts, reaction, options,
    )
    if _query_state is not None:
        _query_state.user_mts.update(user_mts)
    zap = physconst.get_zap_for_particle(particle)
    mts = mf3interp.get_reaction_mts_widened(endf_dict)
    _check_particle_production_mode1(endf_dict, user_mts, zap, mts, _query_state)
    return quant_mt_zap.compute_cumulative_quantity(
        lambda endf_dict, mt, zap, einc, mus:
            quant_mt_zap.compute_daxs(
                endf_dict, mt, zap, einc, mus,
                options=options, _query_state=_query_state,
            ),
        lambda endf_dict, mt, zap, energies_in, angle_cosines_out: (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
        ),
        endf_dict, zap, energies_in, angle_cosines_out,
        mts=mts,
    )


def get_particle_production_ddxs(
    endf_dict, reaction, particle, energies_in, energies_out,
    angle_cosines_out, *, broadening=None, options=None,
):
    """Double-differential cross section for particle production.

    Parameters
    ----------
    endf_dict, reaction, particle, energies_in, energies_out, angle_cosines_out
        Physics arguments (dict, MT/particle strings, query axes).
    broadening : None or float or (callable, float), optional
        Measurement-resolution kernel (per-call physics knob). Same
        accepted forms as :func:`get_particle_production_dxs_dE`.
    options : RunOptions, optional
        Runtime policies. Default None resolves to a physics-first
        :class:`endf_userpy.run_options.RunOptions`.
        ``broadening_mesh_bounds`` on the options object is
        required when ``energies_out`` is a jax tracer (Phase 5 of
        #290).
    """
    if options is None:
        options = RunOptions()
    query_state = _QueryState()
    result = _get_particle_production_ddxs_impl(
            endf_dict, reaction, particle, energies_in, energies_out,
            angle_cosines_out, broadening, options=options,
         _query_state=query_state)
    _emit_summary_warnings(query_state, options)
    return result


def get_particle_production_discrete_gamma_lines(
    endf_dict, reaction, energies_in, *,
    angle_cosines_out=None, options=None,
):
    """Return the discrete photon-line content for ``reaction``
    that the unbroadened DDX / dxs_dE paths drop.

    The gamma paths of :func:`get_particle_production_ddxs` and
    :func:`get_particle_production_dxs_dE` exclude MF12 and MF13
    discrete photon lines (their Dirac deltas integrate to zero on
    a finite E_out grid) and emit a UserWarning naming the
    affected MTs. This function exposes those lines directly so
    callers can render them as stems on top of the continuum
    spectrum, apply a custom broadening kernel, or fit their
    per-Ein yields.

    Parameters
    ----------
    endf_dict : the parsed ENDF-6 dict from ``endf_parserpy``.
    reaction : str
        Reaction shorthand (e.g. ``'(n,g)'``, ``'(n,inl)'``); the
        widened-MT selector picks every underlying MT that matches.
    energies_in : 1D array-like of incident energies (eV).
    angle_cosines_out : optional 1D array-like of mu targets.
        When passed, each record's ``angdist`` field is populated
        from MF14 (or the isotropic fallback). When ``None``, the
        angular column is left as ``None`` and no MF14 lookup runs.
    xp : optional backend adapter (:mod:`array_ns`). Under
        ``xp=jax`` each record's ``weight`` (and ``angdist``, when
        requested) carry tracers so ``jax.grad`` reaches through
        the MF12 yield tables, the MF3 cross section, the MF13
        per-line production XS, and any MF14 Legendre coefficients.
    above_range, resonance_range : passed through to the MF3
        cross-section lookup used by the MF12 branch. Same
        semantics as :func:`get_reaction_xs`.

    Returns
    -------
    list of :class:`endf_userpy.quantities_mt_zap.discrete_gamma_lines.DiscreteGammaLine`
        Sorted by ``Eg`` ascending. Empty list if the reaction has
        no discrete gamma-line content. The per-line extraction
        and physics-composition live in
        :mod:`endf_userpy.quantities_mt_zap.discrete_gamma_lines`;
        this wrapper handles reaction-string resolution and the
        ``above_range`` / ``resonance_range`` policy contexts.
    """
    if options is None:
        options = RunOptions()
    xp = options.backend
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    mts = mf3interp.get_reaction_mts_widened(endf_dict)
    query_state = _QueryState()
    result = discrete_gamma.extract_discrete_gamma_lines(
        endf_dict, mts, user_mts, energies_in,
        angle_cosines_out=angle_cosines_out, xp=xp,
    )
    _emit_summary_warnings(query_state, options)
    return result


def _get_particle_production_ddxs_impl(
    endf_dict, reaction, particle, energies_in, energies_out,
    angle_cosines_out, broadening, *, options, _query_state=None,
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    _check_fission_chance_breakdown_vs_mf2(
        endf_dict, user_mts, reaction, options,
    )
    if _query_state is not None:
        _query_state.user_mts.update(user_mts)
    zap = physconst.get_zap_for_particle(particle)
    xp = options.backend
    broadening_mesh_bounds = options.broadening_mesh_bounds
    broadening_wkw = options.broadening_window_kernel_widths
    mts = mf3interp.get_reaction_mts_widened(endf_dict)
    _check_particle_production_mode1(endf_dict, user_mts, zap, mts, _query_state)

    kernel, kernel_width = _normalize_broadening(broadening, xp=xp)
    if kernel is None:
        if not _broadening_explicit_no_kernel(broadening):
            _warn_discrete_dropped_from_unbroadened_ddx(
                endf_dict, zap, user_mts,
            )
            _warn_mf12_mf13_discrete_dropped(
                endf_dict, zap, user_mts,
                'get_particle_production_ddxs',
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
            lambda endf_dict, mt, zap, einc, eouts, mus:
                quant_mt_zap.compute_ddxs(
                    endf_dict, mt, zap, einc, eouts, mus,
                    options=options, _query_state=_query_state,
                ),
            lambda endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out: (
                selectors.contains_zap(endf_dict, mt, zap) and
                selectors.has_continuous_ddx(endf_dict, mt, zap) and
                selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
            ),
            endf_dict, zap, energies_in, energies_out, angle_cosines_out,
            mts=mts,
        )
        mf15_unbroad = quant_mt_zap.compute_cumulative_quantity(
            lambda endf_dict, mt, zap, einc, eouts, mus:
                quant_mt_zap.compute_ddxs_from_mf15_mf14(
                    endf_dict, mt, zap, einc, eouts, mus,
                    options=options, _query_state=_query_state,
                ),
            lambda endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out: (
                selectors.contains_zap(endf_dict, mt, zap) and
                selectors.has_mf15_continuum(endf_dict, mt, zap) and
                selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
            ),
            endf_dict, zap, energies_in, energies_out, angle_cosines_out,
            mts=mts,
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
            kernel=kernel, kernel_width=kernel_width, xp=xp,
            options=options, _query_state=_query_state,
            mesh_bounds=broadening_mesh_bounds,
            window_kernel_widths=broadening_wkw,
        )

    def cont_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_continuous_ddx(endf_dict, mt, zap) and
            selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
        )

    def disc_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_discrete_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel, xp=xp,
            options=options, _query_state=_query_state,
        )

    def disc_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_discrete_two_body_ddx(endf_dict, mt, zap) and
            selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
        )

    def law1_disc_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_law1_discrete_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel, xp=xp,
            options=options, _query_state=_query_state,
        )

    def law1_disc_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf6_law1_discrete_lines(endf_dict, mt, zap) and
            selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
        )

    def mf12_disc_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_mf12_discrete_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel, xp=xp,
            options=options, _query_state=_query_state,
        )

    def mf12_disc_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf12_discrete_lines(endf_dict, mt, zap) and
            selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
        )

    def mf13_disc_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_mf13_discrete_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel, xp=xp,
        )

    def mf13_disc_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf13_discrete_lines(endf_dict, mt, zap) and
            selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
        )

    def mf15_cont_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_mf15_continuum_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel, kernel_width=kernel_width, xp=xp,
            options=options, _query_state=_query_state,
            mesh_bounds=broadening_mesh_bounds,
            window_kernel_widths=broadening_wkw,
        )

    def mf15_cont_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_mf15_continuum(endf_dict, mt, zap) and
            selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap)
        )

    # Sum-then-broaden path (issue #26): when 2+ MTs pass
    # cont_select, computing dist2d for the whole sum inside ONE
    # adaptive_convolve saves the FFT overhead of the per-MT
    # approach. Convolution is linear so the answer is identical up
    # to floating-point summation order. One MT: no gain, fall
    # through to the per-MT path.
    cont_mts = [
        mt for mt in mts
        if cont_select(endf_dict, mt, zap,
                       energies_in, energies_out, angle_cosines_out)
    ]
    if len(cont_mts) >= 2:
        cont = ddxb.compute_ddx_continuous_broadened_summed(
            endf_dict, cont_mts, zap,
            energies_in, energies_out, angle_cosines_out,
            kernel=kernel, kernel_width=kernel_width, xp=xp,
            options=options, _query_state=_query_state,
            mesh_bounds=broadening_mesh_bounds,
            window_kernel_widths=broadening_wkw,
        )
    else:
        cont = quant_mt_zap.compute_cumulative_quantity(
            cont_compute, cont_select,
            endf_dict, zap, energies_in, energies_out, angle_cosines_out,
            mts=mts,
        )
    disc = quant_mt_zap.compute_cumulative_quantity(
        disc_compute, disc_select,
        endf_dict, zap, energies_in, energies_out, angle_cosines_out,
        mts=mts,
    )
    law1_disc = quant_mt_zap.compute_cumulative_quantity(
        law1_disc_compute, law1_disc_select,
        endf_dict, zap, energies_in, energies_out, angle_cosines_out,
        mts=mts,
    )
    mf12_disc = quant_mt_zap.compute_cumulative_quantity(
        mf12_disc_compute, mf12_disc_select,
        endf_dict, zap, energies_in, energies_out, angle_cosines_out,
        mts=mts,
    )
    mf13_disc = quant_mt_zap.compute_cumulative_quantity(
        mf13_disc_compute, mf13_disc_select,
        endf_dict, zap, energies_in, energies_out, angle_cosines_out,
        mts=mts,
    )
    mf15_cont = quant_mt_zap.compute_cumulative_quantity(
        mf15_cont_compute, mf15_cont_select,
        endf_dict, zap, energies_in, energies_out, angle_cosines_out,
        mts=mts,
    )
    parts = [
        p for p in (cont, disc, law1_disc, mf12_disc, mf13_disc, mf15_cont)
        if p is not None
    ]
    if not parts:
        return None
    # Broadened kernels are FFT-based numpy-only; lift to xp at the
    # boundary when a caller asked for a non-numpy backend.
    if xp is not None:
        parts = [xp.asarray(p) for p in parts]
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
        if not selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap):
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
        f'finite kernel that plots on the E_out grid. Pass '
        f'`broadening=0` to silence this warning while keeping the '
        f'drop.',
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
        if not selectors.satisfies_particle_production_select(endf_dict, mt, user_mts, zap):
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
        f'into a finite kernel that plots on the E\' grid. Pass '
        f'`broadening=0` to silence this warning while keeping the '
        f'drop.',
        UserWarning, stacklevel=3,
    )


def _warn_mf12_mf13_discrete_dropped(endf_dict, zap, user_mts, context):
    """Emit a UserWarning when the unbroadened dispatcher drops
    MF12 discrete gamma lines or MF13 per-line photon-production
    entries for a gamma-emission call (issue #266). Fires only for
    gamma ZAP; other ZAPs cannot carry MF12/MF13 discrete-line
    content by construction.

    ``context`` is the caller-facing function name that appears in
    the warning body (e.g. ``'get_particle_production_dxs_dE'`` or
    ``'get_particle_production_ddxs'``). The suppression hatch is
    ``broadening=0``, which the dispatcher detects via
    :func:`_broadening_explicit_no_kernel` before calling this
    helper.
    """
    if zap != physconst.get_zap_for_particle('g'):
        return
    dropped = []
    for mt in quant_mt_zap.get_reaction_mt_numbers(endf_dict):
        if not selectors.contains_zap(endf_dict, mt, zap):
            continue
        if not selectors.satisfies_particle_production_select(
            endf_dict, mt, user_mts, zap,
        ):
            continue
        if (
            selectors.has_mf12_discrete_lines(endf_dict, mt, zap)
            or selectors.has_mf13_discrete_lines(endf_dict, mt, zap)
        ):
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
        f'{context} (unbroadened) dropped MF12/MF13 discrete '
        f"gamma-line content whose outgoing energy is a Dirac "
        f"delta on the E' axis: MT={mt_str}. The continuum part of "
        f'these MTs IS included. To include the discrete lines too, '
        f'pass a `broadening=sigma_eV` (or a custom (kernel, width) '
        f'tuple) so the deltas are folded into a finite kernel that '
        f"plots on the E' grid. Pass `broadening=0` to silence this "
        f'warning while keeping the drop.',
        UserWarning, stacklevel=3,
    )


def _normalize_broadening(broadening, xp=None):
    """Translate a user broadening spec into ``(kernel, width)`` for
    the low-level folders.

    ``None`` (the default) and ``0`` both propagate as
    ``(None, None)``: no kernel is folded and any Dirac-delta
    content (discrete gamma lines, kinematic two-body deltas)
    integrates to zero on the caller's finite grid. The two forms
    differ only in whether the caller has explicitly acknowledged
    that drop; the dispatcher checks via
    :func:`_broadening_explicit_no_kernel` and suppresses the
    "discrete content dropped" UserWarnings in the ``broadening=0``
    case.

    ``xp`` is captured by the built-in-Gaussian kernel closure so
    ``xp.exp`` dispatches correctly under ``@jax.jit`` /
    ``jax.grad``. Defaults to numpy; caller-supplied
    ``(kernel_callable, width)`` tuples are used as-is and are the
    caller's responsibility to keep xp-consistent.
    """
    if xp is None:
        from .primitives import array_ns
        xp = array_ns.get_backend('numpy')
    if broadening is None:
        return None, None
    if isinstance(broadening, (int, float, np.integer, np.floating)):
        sigma = float(broadening)
        if sigma < 0:
            raise ValueError("broadening sigma must be non-negative")
        if sigma == 0.0:
            return None, None
        norm = 1.0 / (sigma * np.sqrt(2 * np.pi))

        def gaussian_kernel(d):
            # Bind ``xp.exp`` at closure creation so ``@jax.jit`` /
            # ``jax.grad`` on the broadening path uses ``jnp.exp``
            # on tracer ``d`` and numpy on concrete ``d``. The
            # captured ``xp`` comes from the caller's
            # ``_normalize_broadening(broadening, xp=xp)`` call.
            return norm * xp.exp(-0.5 * (d / sigma) ** 2)
        return gaussian_kernel, sigma
    try:
        kernel, width = broadening
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "broadening must be None, 0, a positive scalar sigma, or a "
            "(kernel_callable, width) tuple"
        ) from exc
    if not callable(kernel):
        raise ValueError("broadening tuple element 0 must be a callable kernel")
    width = float(width)
    if width <= 0:
        raise ValueError("broadening tuple element 1 (width) must be positive")
    return kernel, width


def _broadening_explicit_no_kernel(broadening):
    """True iff the caller explicitly asked for no kernel by passing
    ``broadening=0``, as opposed to leaving the default
    ``broadening=None``. Both drop discrete-delta content from the
    output, but the explicit form silences the accompanying
    UserWarnings.
    """
    if isinstance(broadening, (int, float, np.integer, np.floating)):
        return float(broadening) == 0.0
    return False


