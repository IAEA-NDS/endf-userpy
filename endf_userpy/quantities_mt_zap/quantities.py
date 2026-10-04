import numpy as np
from ..primitives import properties
from ..primitives import reactions as reaction
from ..primitives.physical_constants import get_zap_for_particle
from ..mfsec_interpretation import mf1_interpretation as mf1_interp
from ..mfsec_interpretation import mf3_interpretation as mf3_interp
from . import resonance_composition as _res_comp
from ..mfsec_interpretation import mf6_interpretation as mf6_interp
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from ..mfsec_interpretation import mf8_interpretation as mf8_interp
from ..mfsec_interpretation import mf9_interpretation as mf9_interp
from ..mfsec_interpretation import mf10_interpretation as mf10_interp
from ..mfsec_interpretation import mf12_interpretation as mf12_interp
from ..mfsec_interpretation import mf13_interpretation as mf13_interp
from ..mfsec_interpretation import mf14_interpretation as mf14_interp
from ..mfsec_interpretation import mf15_interpretation as mf15_interp
from .distribution1d import (
    compute_angdist_values,
    compute_energydist_values,
)
from .distribution2d import compute_dist2d_values
from . import discrete_quantities as discrete_quant
import logging


module_logger = logging.getLogger(__name__)


# functions borrowed as is
from ..mfsec_interpretation.mf3_interpretation import (
    get_reaction_mts as get_reaction_mt_numbers,
)


def compute_yields(
    endf_dict, mt, zap, energies_in, include_discrete=True, level=None,
    xp=None,
):
    """Ejectile yield ``y(E_in)`` for one ``(mt, zap)``.

    ``xp=None`` (default) is numpy and bit-identical to the pre-port
    behaviour. Passing an xp adapter threads tracers through each
    dispatch branch (MF1 nubar for fission neutrons, MF6 per-
    subsection yields, MF12/MF13 gamma yields, and the reaction-
    string multiplicity fallback), so ``jax.grad`` reaches file-side
    yield leaves end-to-end.
    """
    from ..primitives import array_ns
    if xp is None:
        xp = array_ns.get_backend('numpy')
    module_logger.debug(f'compute yields for MT={mt} and ZAP={zap} and level={level}')
    neutron_zap = get_zap_for_particle('n')
    gamma_zap = get_zap_for_particle('g')
    if mt == 18 and zap == neutron_zap:
        # Prompt neutron yield for MT18 (n,f) comes from MF1/MT456
        # nubar, not from any per-MT ejectile-multiplicity table.
        if level is not None:
            raise ValueError(
                'For fission, `level` argument must be `None`'
            )
        module_logger.debug(f'--> getting yields for MT={mt} and ZAP={zap} from MF1/MT456')
        yields = mf1_interp.compute_yields(endf_dict, 456, energies_in, xp=xp)
    elif mt == 18 and zap == gamma_zap and (
        properties.has_mf12_mt(endf_dict, mt)
        or properties.has_mf13_mt(endf_dict, mt)
    ):
        # Prompt fission gammas flow through the same MF12/MF13
        # gamma-yield path as inelastic partial channels (issue
        # #126). The fall-through-elif below would also match, but
        # spelling it out here makes the fission gamma routing
        # explicit and keeps the fallback branch's error message
        # accurate for the "no representable yield" case.
        if level is not None:
            raise ValueError(
                'For fission, `level` argument must be `None`'
            )
        module_logger.debug(
            f'--> getting fission-gamma yields for MT={mt} from MF12/MF13'
        )
        yields = discrete_quant.compute_total_gamma_yields(
            endf_dict, mt, energies_in, xp=xp,
        )
    elif mt == 18:
        # Fission with a ZAP that's neither neutron nor a
        # gamma-with-MF12/13. Charged fragments are not
        # representable via this route.
        raise ValueError(
            f'For fission (MT=18), only prompt-neutron yield '
            f'(ZAP={neutron_zap}) and prompt-gamma yield (ZAP='
            f'{gamma_zap}, when the file carries MF12 or MF13 for '
            f'MT=18) are supported; got ZAP={zap}'
        )
    elif (properties.has_mf6_mt(endf_dict, mt)
          and mf6_help.contains_zap(endf_dict, mt, zap)):
        module_logger.debug(f'--> getting yields for MT={mt} and ZAP={zap} from MF6/MT{mt}')
        yields = mf6_interp.compute_yields(
            endf_dict, mt, zap, energies_in, include_discrete, level, xp=xp,
        )
    elif (zap == get_zap_for_particle('g')
          and (properties.has_mf12_mt(endf_dict, mt)
               or properties.has_mf13_mt(endf_dict, mt))):
        # Photons for many partial channels (e.g. inelastic MTs 51..90
        # in ENDF/B-VIII.1, JEFF-4.0, TENDL) are stored in MF12 (yields)
        # or MF13 (production cross sections) rather than MF6, which
        # carries only the scattered neutron. Without this branch these
        # gamma contributions were silently omitted from the gamma
        # production cross section (issue #29).
        if level is not None:
            raise ValueError(
                f'`level` argument not supported for MF12/MF13 gamma yields '
                f'(MT={mt}).'
            )
        module_logger.debug(
            f'--> getting photon yields for MT={mt} from MF12/MF13'
        )
        # Total (discrete lines + continuum placeholder) so files that
        # split MT 102 into a low-energy discrete cascade and a high-
        # energy continuum spectrum in MF15 do not silently produce a
        # zero yield for the continuum-dominant region.
        yields = discrete_quant.compute_total_gamma_yields(
            endf_dict, mt, energies_in, xp=xp,
        )
    else:
        if level is not None:
            raise ValueError(
                f'Yield directly derived from MT number (MT={mt}, '
                '`level` argument must be `None`'
            )
        proj = properties.get_projectile(endf_dict)
        mult = reaction.get_multiplicity_for_zap(proj, mt, zap)
        module_logger.debug(
            f'--> getting yields for MT={mt} and ZAP={zap} from reaction string (yield={mult})'
        )
        if mult is None:
            # Fallback reaction-string lookup couldn't determine a
            # multiplicity for this (MT, ZAP): the MT is either not
            # in `reactions.REACTION_DICT` or its ejectile parsing
            # hit a defensive `return None` branch (mangled ejectile
            # string, unknown particle, invalid level suffix). Well-
            # formed admissible calls do not reach here because
            # `selectors.contains_zap` already filters unknown MTs.
            # Raise loudly rather than propagate `np.full(..., None)`
            # -> silent NaN through the rest of the pipeline
            # (issue #104 / audit D4).
            raise ValueError(
                f'Cannot derive multiplicity for MT={mt}, ZAP={zap}: '
                f'the MT is not in reactions.REACTION_DICT (or its '
                f'ejectile string could not be parsed), no MF6 '
                f'subsection carries the ZAP, and it is not a '
                f'gamma-in-MF12/MF13 case. Add the MT to the '
                f'reaction table or file an issue with the offending '
                f'MT and file.'
            )
        yields = xp.full(len(energies_in), mult, dtype=xp.float64)
    return yields


def compute_xs_mt5_contrib(
    endf_dict, mt, energies_in, *, options=None, _query_state=None,
):
    """MT5 backfill contribution for reactions with a unique-path-to-
    residual. Returns the redistributed cross section ``y(E) * σ_MT5(E)``
    where ``y`` is the MF6/MT=5 yield for the residual uniquely
    identified by ``mt``, or zeros if the file has no MF6/MT=5 or the
    reaction isn't unique-path-to-residual.

    Runtime policies + backend live on ``options``; the accumulator
    (``_query_state``) is threaded to the leaf reader so above-range
    hits contribute to the top-level summary UserWarning.
    """
    from ..run_options import RunOptions
    if options is None:
        options = RunOptions()
    xp = options.backend
    zero_xs_result = xp.zeros_like(xp.asarray(energies_in), dtype=xp.float64)
    if not properties.has_mf6_mt(endf_dict, 5):
        return zero_xs_result

    proj = properties.get_projectile(endf_dict)
    if not reaction.is_unique_path_to_residual(proj, mt):
        return zero_xs_result
    ejectile = reaction.get_unique_ejectile(proj, mt)

    mt5 = 5
    za_projectile = properties.get_ZAI(endf_dict)
    za_target = properties.get_ZA(endf_dict)
    za_ejectile = get_zap_for_particle(ejectile)
    m = reaction.get_multiplicity_for_zap(proj, mt, za_ejectile)
    if m is None:
        return zero_xs_result
    za_residual = za_target + za_projectile - m * za_ejectile

    if not mf6_help.has_subsecs_for_mt_zap(endf_dict, mt5, za_residual):
        return zero_xs_result
    yield_mt5 = compute_yields(
        endf_dict, mt5, za_residual, energies_in, include_discrete=True,
        xp=xp,
    )
    # Route through compute_xs for consistency (issue #304). MT5 is
    # a catch-all and doesn't carry an MF2 resonance range, so this
    # degenerates to raw MF3 in practice; keeping the composed path
    # means every XS read in the mid-layer goes through the same
    # door.
    xs_mt5 = compute_xs(
        endf_dict, mt5, energies_in,
        options=options, _query_state=_query_state,
    )
    return xs_mt5 * yield_mt5


def compute_xs(endf_dict, mt, energies_in, *, options=None, _query_state=None):
    """Cross section for one MT.

    ``options`` (see :class:`endf_userpy.run_options.RunOptions`)
    carries every runtime policy: ``above_range``, ``resonance_range``,
    ``include_resonance``, ``backend`` (with ``'auto'`` resolving
    to numba on resonance calls when available), and
    ``urr_quadrature``. ``options=None`` resolves to a default
    :class:`RunOptions` instance, so leaf callers get sensible
    physics-first defaults if they invoke this directly.

    ``_query_state`` is the private
    :class:`~endf_userpy.run_options._QueryState`
    accumulator the top-level ``endf_userpy.quantities`` entry
    points thread through internal callers so ONE summary
    UserWarning fires per top-level query (issue #143). ``None``
    (default) makes leaf readers fall back to per-call warnings.
    """
    from ..run_options import RunOptions
    if options is None:
        options = RunOptions()
    xp = options.backend
    if options.include_resonance:
        result = _res_comp.compute_reconstructed_cross_section(
            endf_dict, mt, energies_in, xp,
            urr_quadrature=options.urr_quadrature,
            _query_state=_query_state,
        )
        # The composition layer uses ``compute_cross_section_agnostic``
        # so it bypasses the MF3 policy machinery. Reapply the
        # above_range policy at the composed level so a top-level
        # ``options=RunOptions(above_range=...)`` still governs
        # what happens above the file's Ein mesh. The
        # ``resonance_range`` policy is intentionally a no-op here:
        # the whole point of ``include_resonance=True`` is that the
        # composed XS in the RRR is physical, so warning about it
        # would be misleading. If MT is not in MF3 at all (JENDL-5
        # MT3 shape, no MF3 entry) skip the mask -- composition
        # returns the pure MF2 contribution and there is no mesh
        # to bound.
        if mt in endf_dict.get(3, {}):
            from ..mfsec_interpretation.mf3_interpretation import (
                _handle_above_range,
            )
            e_mesh = np.asarray(endf_dict[3][mt]['xstable']['E'], dtype=float)
            e_max = float(e_mesh.max())
            # Under jax the tracer energies_in cannot be numpified;
            # do the mask xp-natively. The _handle_above_range hook
            # still needs a concrete count for the accumulator; on
            # tracers we skip that (no summary warning fires under
            # jit anyway).
            einc_xp = xp.asarray(energies_in)
            above_mask_xp = einc_xp > e_max
            # Concrete-path warning + raise policy live inside
            # ``_handle_above_range``. Only skip that call when
            # ``energies_in`` is a jax tracer (can't materialise).
            # Distinguish tracer via a targeted probe rather than
            # a broad ``except Exception`` which would swallow the
            # legitimate ``ValueError`` raised by policy='raise'.
            try:
                einc_arr = np.asarray(energies_in, dtype=float)
            except (TypeError, ValueError):
                einc_arr = None
            if einc_arr is not None:
                above_mask_np = einc_arr > e_max
                fill = _handle_above_range(
                    options.above_range, mt, e_max,
                    above_mask_np, einc_arr, hits=_query_state,
                )
            else:
                # tracer path: pick the fill from the policy name
                if options.above_range in ('warn_nan', 'nan'):
                    fill = float('nan')
                else:
                    fill = 0.0
            if options.above_range in ('warn_nan', 'nan'):
                result = xp.where(above_mask_xp, fill, result)
            elif options.above_range in ('warn_zero', 'zero'):
                result = xp.where(above_mask_xp, 0.0, result)
            # 'raise' already raised inside _handle_above_range on
            # the concrete path; no fill to apply here.
        # Preserve pre-port behaviour on the numpy path (return
        # numpy); on non-numpy xp keep the tracer alive so autodiff
        # works end-to-end.
        if xp.name == 'numpy':
            return np.asarray(result)
        return result
    xs = mf3_interp.compute_cross_section(
        endf_dict, mt, energies_in,
        above_range=options.above_range,
        resonance_range=options.resonance_range,
        xp=xp, _query_state=_query_state,
    )
    if xp.name != 'numpy':
        return xp.asarray(xs)
    return xs


def compute_prodxs(
    endf_dict, mt, zap, energies_in, *, options=None, _query_state=None,
):
    """Particle-production cross section for one (MT, ZAP).

    Runtime policies + backend live on ``options`` (issue #143).
    ``options=None`` resolves to a physics-first
    :class:`~endf_userpy.run_options.RunOptions`.

    ``_query_state`` is the private ``_QueryState`` accumulator
    threaded from the top-level entry points so above-range /
    resonance-range hits populate the ONE summary UserWarning per
    top-level query. ``None`` (default) triggers per-call warnings
    from the leaf.
    """
    from ..run_options import RunOptions
    if options is None:
        options = RunOptions()
    xp = options.backend
    if (
        zap == get_zap_for_particle('g')
        and mt not in endf_dict.get(3, {})
        and mt in endf_dict.get(13, {})
    ):
        return mf13_interp.compute_total_photon_production_xs(
            endf_dict, mt, energies_in, xp=xp,
        )
    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=True, xp=xp,
    )
    # Route through compute_xs so include_resonance=True composes
    # the MF2 reconstruction on top of MF3 (issue #304). On files
    # without MF2 this is a no-op; on MF2-bearing files inside the
    # resolved-resonance region it switches from raw-MF3 background
    # to the physical composed cross section.
    xs = compute_xs(
        endf_dict, mt, energies_in,
        options=options, _query_state=_query_state,
    )
    return yields * xs


def _is_mf13_only_gamma(endf_dict, mt, zap):
    """True iff `(mt, zap)` is a gamma-production case that the file
    carries via MF13 but not MF3. The standard `xs * yields`
    composition in compute_dexs / compute_daxs / compute_ddxs
    requires MF3, so these functions short-circuit to using MF13's
    prodxs directly (which IS the total gamma production XS for the
    MT). Issue #130.
    """
    return (
        zap == get_zap_for_particle('g')
        and mt not in endf_dict.get(3, {})
        and mt in endf_dict.get(13, {})
    )


def compute_daxs(
    endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True,
    *, options=None, _query_state=None,
):
    """Angular-differential cross section ``d sigma / d mu`` for one
    (MT, ZAP).

    Runtime policies + backend live on ``options`` (issue #143).
    """
    from ..run_options import RunOptions
    if options is None:
        options = RunOptions()
    xp = options.backend
    if _is_mf13_only_gamma(endf_dict, mt, zap):
        prodxs = mf13_interp.compute_total_photon_production_xs(
            endf_dict, mt, energies_in, xp=xp,
        ).reshape(-1, 1)
        angdist = compute_angdist_values(
            endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab, xp=xp,
        )
        return angdist * prodxs / (2 * np.pi)
    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=True, xp=xp,
    ).reshape(-1, 1)
    # Route through compute_xs for MF2 composition under
    # include_resonance=True (issue #304).
    xs = compute_xs(
        endf_dict, mt, energies_in,
        options=options, _query_state=_query_state,
    ).reshape(-1, 1)
    angdist = compute_angdist_values(
        endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab, xp=xp,
    )
    return angdist * yields * xs / (2 * np.pi)


def compute_dexs(
    endf_dict, mt, zap, energies_in, energies_out, to_lab=True,
    *, options=None, _query_state=None,
):
    """Energy-differential cross section ``d sigma / d E'`` for one
    (MT, ZAP).

    Runtime policies + backend live on ``options`` (issue #143).
    """
    from ..run_options import RunOptions
    if options is None:
        options = RunOptions()
    xp = options.backend
    module_logger.debug(f'compute dexs for MT={mt} and ZAP={zap}')
    # MF13-only gamma fast path (issue #130).
    if _is_mf13_only_gamma(endf_dict, mt, zap):
        prodxs = mf13_interp.compute_total_photon_production_xs(
            endf_dict, mt, energies_in, xp=xp,
        ).reshape(-1, 1)
        energydist = compute_energydist_values(
            endf_dict, mt, zap, energies_in, energies_out, to_lab, xp=xp,
        )
        return energydist * prodxs
    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=True, xp=xp,
    ).reshape(-1, 1)
    # Route through compute_xs for MF2 composition under
    # include_resonance=True (issue #304).
    xs = compute_xs(
        endf_dict, mt, energies_in,
        options=options, _query_state=_query_state,
    ).reshape(-1, 1)
    energydist = compute_energydist_values(
        endf_dict, mt, zap, energies_in, energies_out, to_lab, xp=xp,
    )
    module_logger.debug(f'average yield for MT={mt} and ZAP={zap}')
    return energydist * yields * xs


def compute_ddxs(
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out,
    to_lab=True, *, options=None, _query_state=None,
):
    """Double-differential cross section for one (MT, ZAP).

    Runtime policies + backend live on ``options`` (issue #143).
    """
    from ..run_options import RunOptions
    if options is None:
        options = RunOptions()
    xp = options.backend
    if _is_mf13_only_gamma(endf_dict, mt, zap):
        n_einc = np.asarray(energies_in).size
        n_eout = np.asarray(energies_out).size
        n_mu = np.asarray(angle_cosines_out).size
        return xp.zeros((n_einc, n_eout, n_mu), dtype=xp.float64)
    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=False, xp=xp,
    ).reshape(-1, 1, 1)
    # Route through compute_xs for MF2 composition under
    # include_resonance=True (issue #304).
    xs = compute_xs(
        endf_dict, mt, energies_in,
        options=options, _query_state=_query_state,
    ).reshape(-1, 1, 1)
    f = compute_dist2d_values(
        endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out,
        to_lab, xp=xp,
    )
    return f * yields * xs / (2 * np.pi)


def compute_ddxs_from_mf15_mf14(
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out,
    to_lab=True, *, options=None, _query_state=None,
):
    """Unbroadened DDX contribution from MF15 continuum gamma
    spectrum + MF14 angular. Gamma-only peer of
    ``ddx_broadening.compute_ddx_mf15_continuum_broadened`` without
    the ``adaptive_convolve`` step (kernel is a delta).

    Contribution per (E_in, E_out, mu)::

        DDX(E_in, E_out, mu) = sigma(E_in)
                             * y_cont(E_in)
                             * spec(E_out | E_in)
                             * f_cont(mu | E_in) / (2 pi)

    where ``sigma`` is MF3, ``y_cont`` is the MF12 Eg=0
    continuum-placeholder yield, ``spec`` is the raw (unbroadened)
    MF15 continuous spectrum, and ``f_cont`` is the MF14 continuum
    angular distribution (LI=0 Eg=0 entry, or isotropic 0.5 for
    LI=1 or absent MF14). Matches the composition of
    ``compute_ddx_mf15_continuum_broadened`` in the delta-kernel
    limit.

    Callers must gate on ``selectors.has_mf15_continuum(mt, zap)``;
    calling on a MT with no MF15 returns a zero DDX. If MF12 has no
    Eg=0 continuum placeholder, the contribution is dropped (same
    convention as the 1D dxs/dE path from issue #103; the warning
    from that path fires for the same file).

    Runtime policies + backend live on ``options`` (issue #143).

    Returns
    -------
    ddx : ndarray of shape ``(n_einc, n_eouts, n_mus)``. Same units
    and shape as ``compute_ddxs``.
    """
    from ..run_options import RunOptions
    if options is None:
        options = RunOptions()
    xp = options.backend
    if zap != get_zap_for_particle('g'):
        raise ValueError(
            'MF15 continuum unbroadened DDX is gamma-only; got '
            f'ZAP={zap}'
        )
    energies_in = np.asarray(energies_in, dtype=float)
    energies_out = np.asarray(energies_out, dtype=float)
    angle_cosines_out = np.asarray(angle_cosines_out, dtype=float)
    n_einc = len(energies_in)
    n_eouts = len(energies_out)
    n_mus = len(angle_cosines_out)
    result_zero = xp.zeros((n_einc, n_eouts, n_mus), dtype=xp.float64)

    if not properties.has_mf15_mt(endf_dict, mt):
        return result_zero
    # Weight: (xs * y_cont) product for the gamma-continuum
    # contribution. Two shapes are supported:
    #  - Standard MF12+MF15 MT: xs from MF3 times MF12 Eg=0
    #    continuum-placeholder yield.
    #  - MF13-only MT (JENDL-5 MT 3 style, issue #130): MF13 IS
    #    the total gamma production XS -- use it directly, no
    #    MF3/MF12 lookup.
    if mt not in endf_dict.get(3, {}) and mt in endf_dict.get(13, {}):
        weight = mf13_interp.compute_total_photon_production_xs(
            endf_dict, mt, energies_in, xp=xp,
        )   # (n_einc,)
    else:
        if not properties.has_mf12_mt(endf_dict, mt):
            return result_zero
        pes = np.asarray(
            mf12_interp.get_photon_energies(endf_dict, mt), dtype=float,
        )
        cont_mask = pes == 0.0
        if not np.any(cont_mask):
            return result_zero
        yields_all = mf12_interp.compute_photon_yields(
            endf_dict, mt, energies_in, pes, xp=xp,
        )
        cont_idcs = np.where(cont_mask)[0]
        y_cont = xp.sum(yields_all[:, cont_idcs], axis=1)   # (n_einc,)
        # Route through compute_xs for MF2 composition under
        # include_resonance=True (issue #304).
        xs = compute_xs(
            endf_dict, mt, energies_in,
            options=options, _query_state=_query_state,
        )   # (n_einc,), xp-native
        weight = xs * y_cont

    # Continuum angular from MF14 Eg=0 entry (LI=0), else isotropic.
    if properties.has_mf14_mt(endf_dict, mt) and endf_dict[14][mt]['LI'] == 0:
        cont_angdist = mf14_interp.compute_angdist_values(
            endf_dict, mt, energies_in,
            np.array([0.0]), angle_cosines_out, xp=xp,
        )
        f_cont = cont_angdist[:, 0, :]
    else:
        f_cont = xp.full((n_einc, n_mus), 0.5, dtype=xp.float64)

    spec = mf15_interp.compute_spectrum(
        endf_dict, mt, energies_in, energies_out, xp=xp,
    )   # (n_einc, n_eouts)
    ddx = (
        weight.reshape(-1, 1, 1)
        * spec.reshape(n_einc, n_eouts, 1)
        * f_cont.reshape(n_einc, 1, n_mus)
    )
    return ddx / (2 * np.pi)


def compute_cumulative_quantity(func, select, endf_dict, *args, mts=None, **kwargs):
    """Iterate over MTs (default: MF3 keys), apply ``select``, sum
    ``func``.

    Pass ``mts=`` explicitly to widen the iteration source; the
    particle-production dispatchers pass the MF3+MF12+MF13+MF15
    union so that gamma production from MTs with MF13-only content
    (e.g. JENDL-5 N-14 MT 3) is not silently skipped (issue #130).

    Backend-agnostic (issue #169): ``xp`` in ``kwargs`` is
    forwarded to ``func`` but stripped from the ``select`` call
    (select predicates operate on file state, not numeric backends).
    The initial-accumulator ``cum_res`` starts as the first ``func``
    output (preserving its backend); subsequent additions use
    ``cum_res = cum_res + ...`` so JAX tracers survive across the
    sum.
    """
    if mts is None:
        mt_list = get_reaction_mt_numbers(endf_dict)
    else:
        mt_list = mts
    # ``select`` predicates take (endf_dict, mt, zap, ...) positional
    # only and do not accept ``xp`` / ``options`` / ``_query_state``.
    # Strip them before passing.
    select_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in ('xp', 'options', '_query_state')
    }
    is_first = True
    cum_res = None
    for mt in mt_list:

        module_logger.debug(f'consider MT={mt} for inclusion in cumulative quantity')
        if select is not None:
            if not select(endf_dict, mt, *args, **select_kwargs):
                continue
        module_logger.debug(f'select MT={mt} for inclusion in cumulative quantity')
        cur_res = func(endf_dict, mt, *args, **kwargs)
        if is_first:
            cum_res = cur_res
            is_first = False
        else:
            cum_res = cum_res + cur_res

    return cum_res


def _compute_residual_xs_for_lfs(
    endf_dict, mt, za_residual, lfs, energies_in,
    *, options=None, _query_state=None,
):
    from ..run_options import RunOptions
    if options is None:
        options = RunOptions()
    lmf = mf8_interp.get_mf_switch(endf_dict, mt, za_residual, lfs)
    # LMF 3/6/9 all source the base cross section from MF3 (plus a
    # yield multiplier in LMF 6/9); route through compute_xs so MF2
    # composition applies under include_resonance=True (issue #304).
    # LMF 10 reads from MF10 directly (per-residual XS), which has
    # no MF2 composition pathway in the current scope.
    if lmf == 3:
        return compute_xs(
            endf_dict, mt, energies_in,
            options=options, _query_state=_query_state,
        )
    if lmf == 6:
        xs = compute_xs(
            endf_dict, mt, energies_in,
            options=options, _query_state=_query_state,
        )
        y = mf6_interp.compute_yields(
            endf_dict, mt, za_residual, energies_in,
            include_discrete=True, level=lfs,
        )
        return xs * y
    if lmf == 9:
        xs = compute_xs(
            endf_dict, mt, energies_in,
            options=options, _query_state=_query_state,
        )
        y = mf9_interp.compute_yields(
            endf_dict, mt, za_residual, energies_in, level=lfs
        )
        return xs * y
    if lmf == 10:
        return mf10_interp.compute_cross_section(
            endf_dict, mt, za_residual, energies_in, level=lfs,
            above_range=options.above_range, _query_state=_query_state,
        )
    raise ValueError(
        f'unsupported LMF={lmf} in MF8/MT={mt} for ZAP={za_residual}, LFS={lfs}'
    )


def compute_residual_xs(
    endf_dict, mt, za_residual, lfs, energies_in,
    *, options=None, _query_state=None,
):
    """Cross section for producing (za_residual, lfs) via reaction MT.

    Dispatches via MF8 LMF: 3 (MF3 only, no isomer split), 6
    (MF3/MT * MF6/MT yield resolved by LFS), 9 (MF3/MT * MF9/MT
    yield), 10 (MF10/MT directly). When MF8/MT is absent, falls back
    to MF3/MT (and refuses to resolve a non-zero LFS).

    If `lfs` is None, sums contributions from all LFS values present
    in MF8/MT for this ZAP.

    Runtime policies live on ``options`` (issue #143).
    """
    from ..run_options import RunOptions
    if options is None:
        options = RunOptions()
    # Route cross-section reads through compute_xs so MF2 composition
    # applies under include_resonance=True (issue #304). The MF8
    # dispatch below only kicks in when MF8/MT exists; the two
    # non-MF8 branches handled first here are the only MF3 reads.
    if not (8 in endf_dict and mt in endf_dict[8]):
        if (6 in endf_dict and mt in endf_dict[6]
                and mf6_help.contains_zap(endf_dict, mt, za_residual)):
            xs = compute_xs(
                endf_dict, mt, energies_in,
                options=options, _query_state=_query_state,
            )
            y = mf6_interp.compute_yields(
                endf_dict, mt, za_residual, energies_in,
                include_discrete=True, level=lfs,
            )
            return xs * y
        if lfs not in (None, 0):
            raise ValueError(
                f'no MF8 information for MT={mt}, cannot resolve isomer level '
                f'(requested LFS={lfs})'
            )
        return compute_xs(
            endf_dict, mt, energies_in,
            options=options, _query_state=_query_state,
        )

    available_lfs = sorted({
        sub['LFS'] for sub in endf_dict[8][mt]['subsection'].values()
        if sub['ZAP'] == za_residual
    })
    if not available_lfs:
        return np.zeros_like(energies_in, dtype=float)

    if lfs is not None:
        if lfs not in available_lfs:
            return np.zeros_like(energies_in, dtype=float)
        return _compute_residual_xs_for_lfs(
            endf_dict, mt, za_residual, lfs, energies_in,
            options=options, _query_state=_query_state,
        )

    total = np.zeros_like(energies_in, dtype=float)
    for cur_lfs in available_lfs:
        total = total + _compute_residual_xs_for_lfs(
            endf_dict, mt, za_residual, cur_lfs, energies_in,
            options=options, _query_state=_query_state,
        )
    return total
