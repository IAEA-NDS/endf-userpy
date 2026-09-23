import contextlib
import contextvars
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


def compute_yields(endf_dict, mt, zap, energies_in, include_discrete=True, level=None):
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
        yields = mf1_interp.compute_yields(endf_dict, 456, energies_in)
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
            endf_dict, mt, energies_in
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
            endf_dict, mt, zap, energies_in, include_discrete, level
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
            endf_dict, mt, energies_in
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
        yields = np.full(len(energies_in), mult, dtype=float)
    return yields


def compute_xs_mt5_contrib(endf_dict, mt, energies_in):
    zero_xs_result = np.zeros_like(energies_in, dtype=float)
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
        endf_dict, mt5, za_residual, energies_in, include_discrete=True
    )
    xs_mt5 = mf3_interp.compute_cross_section(endf_dict, mt5, energies_in)
    return xs_mt5 * yield_mt5


# Context-inherited flag: when True, `compute_xs` returns the
# physical cross section including MF2 resolved-resonance
# reconstruction (composed with MF3 as an additive background), on
# whichever backend was set via `resonance_backend_ctx`. Set by the
# top-level `get_*` APIs when the user passes `include_resonance=True`;
# defaults to False everywhere else so legacy callers keep the raw
# MF3 behaviour and the resonance-range policy machinery in
# `mf3_interpretation`.
_include_resonance_var = contextvars.ContextVar(
    '_include_resonance', default=False,
)
_resonance_backend_var = contextvars.ContextVar(
    '_resonance_backend', default=None,
)


@contextlib.contextmanager
def resonance_reconstruction_ctx(include, backend=None):
    """Context manager toggling MF2 resonance reconstruction inside
    every :func:`compute_xs` call in the ``with`` block. ``backend``
    is a backend name understood by
    :func:`endf_userpy.primitives.array_ns.get_backend`
    (``'numpy'`` / ``'numba'`` / ``'jax'``); ``None`` means numpy.
    """
    tok_i = _include_resonance_var.set(bool(include))
    tok_b = _resonance_backend_var.set(backend)
    try:
        yield
    finally:
        _include_resonance_var.reset(tok_i)
        _resonance_backend_var.reset(tok_b)


def compute_xs(endf_dict, mt, energies_in, xp=None):
    """Cross section for one MT.

    Backend-agnostic (issue #169): ``xp=None`` (default) is numpy
    and preserves the pre-port behaviour (raw MF3 returned as
    numpy, resonance composition returned as numpy via
    ``np.asarray`` at the boundary). Passing a JAX adapter, when
    combined with ``resonance_reconstruction_ctx(include=True,
    backend='jax')``, keeps the composed cross section xp-native
    so ``jax.grad`` reaches file-side leaves (MF2 dict-stored
    resonance parameters) through this entry.

    Note: only the resonance composition is xp-native today; the
    raw MF3 branch (used when ``include_resonance=False``) stays
    numpy internally and returns numpy. See tier-2 remaining work
    under issue #169 for the MF3 xp port.
    """
    from ..primitives import array_ns
    if _include_resonance_var.get():
        # Prefer the caller-provided xp; fall back to the legacy
        # ``resonance_backend`` context so pre-port callers keep
        # working. If both are set the caller's xp wins.
        if xp is None:
            backend_name = _resonance_backend_var.get() or 'numpy'
            xp_res = array_ns.get_backend(backend_name)
        else:
            xp_res = xp
        result = _res_comp.compute_reconstructed_cross_section(
            endf_dict, mt, energies_in, xp_res,
        )
        # Preserve pre-port behaviour on the numpy path (return
        # numpy); on non-numpy xp keep the tracer alive so autodiff
        # works end-to-end.
        if xp_res.name == 'numpy':
            return np.asarray(result)
        return result
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
    if xp is not None and xp.name != 'numpy':
        return xp.asarray(xs)
    return xs


def compute_prodxs(endf_dict, mt, zap, energies_in, xp=None):
    """Particle-production cross section for one (MT, ZAP).

    ``xp=None`` (default) preserves the pre-port numpy behaviour.
    Passing a JAX adapter promotes the numpy sub-components (MF13
    fast-path XS or MF3 x yields) to xp-native at the boundary so
    downstream callers on JAX see xp arrays. MF3 / MF6 yields stay
    numpy internally (their own port is tier-2).
    """
    from ..primitives import array_ns
    if xp is None:
        xp = array_ns.get_backend('numpy')
    if (
        zap == get_zap_for_particle('g')
        and mt not in endf_dict.get(3, {})
        and mt in endf_dict.get(13, {})
    ):
        return mf13_interp.compute_total_photon_production_xs(
            endf_dict, mt, energies_in, xp=xp,
        )
    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=True
    )
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
    result = xs * yields
    return xp.asarray(result) if xp.name != 'numpy' else result


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
    endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True, xp=None,
):
    """Angular-differential cross section ``d sigma / d mu`` for one
    (MT, ZAP).

    ``xp=None`` (default) is numpy. Passing a JAX adapter threads
    tracers through the angular-distribution reconstruction (MF4 or
    MF6 LAW=2 kernels) so ``jax.grad`` reaches file-side leaves.
    MF3 XS and yields stay numpy internally (their own xp port is
    tier-2); they are materialised at the boundary via
    ``xp.asarray`` before multiplication.
    """
    from ..primitives import array_ns
    if xp is None:
        xp = array_ns.get_backend('numpy')
    if _is_mf13_only_gamma(endf_dict, mt, zap):
        prodxs = mf13_interp.compute_total_photon_production_xs(
            endf_dict, mt, energies_in, xp=xp,
        ).reshape(-1, 1)
        angdist = compute_angdist_values(
            endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab, xp=xp,
        )
        return angdist * prodxs / (2 * np.pi)
    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=True
    ).reshape(-1, 1)
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in).reshape(-1, 1)
    angdist = compute_angdist_values(
        endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab, xp=xp,
    )
    return angdist * xp.asarray(yields) * xp.asarray(xs) / (2 * np.pi)


def compute_dexs(
    endf_dict, mt, zap, energies_in, energies_out, to_lab=True, xp=None,
):
    """Energy-differential cross section ``d sigma / d E'`` for one
    (MT, ZAP).

    ``xp=None`` (default) is numpy. Passing a JAX adapter threads
    tracers through the energy-distribution reconstruction
    (composition layer -> MF6 LAW=1 integrator) so ``jax.grad``
    reaches file-side leaves. MF3 cross-section and MF6 yields
    stay numpy internally (their own xp port is tier-2 in issue
    #169's sequencing); they are converted at the boundary via
    ``xp.asarray`` before multiplication.
    """
    from ..primitives import array_ns
    if xp is None:
        xp = array_ns.get_backend('numpy')
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
        endf_dict, mt, zap, energies_in, include_discrete=True
    ).reshape(-1, 1)
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in).reshape(-1, 1)
    energydist = compute_energydist_values(
        endf_dict, mt, zap, energies_in, energies_out, to_lab, xp=xp,
    )
    module_logger.debug(f'average yield is {np.mean(yields)} for MT={mt} and ZAP={zap}')
    return energydist * xp.asarray(yields) * xp.asarray(xs)


def compute_ddxs(
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out,
    to_lab=True, xp=None,
):
    """Double-differential cross section for one (MT, ZAP).

    ``xp=None`` (default) is numpy. Passing a JAX adapter threads
    tracers through the double-differential reconstruction (MF6
    LAW=1/2/6/7 or the MF4 x MF5 product) so ``jax.grad`` reaches
    file-side leaves. MF3 XS and yields stay numpy internally
    (materialised at the boundary via ``xp.asarray``).
    """
    from ..primitives import array_ns
    if xp is None:
        xp = array_ns.get_backend('numpy')
    if _is_mf13_only_gamma(endf_dict, mt, zap):
        n_einc = np.asarray(energies_in).size
        n_eout = np.asarray(energies_out).size
        n_mu = np.asarray(angle_cosines_out).size
        return xp.zeros((n_einc, n_eout, n_mu), dtype=xp.float64)
    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=False
    ).reshape(-1, 1, 1)
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in).reshape(-1, 1, 1)
    f = compute_dist2d_values(
        endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out,
        to_lab, xp=xp,
    )
    return f * xp.asarray(yields) * xp.asarray(xs) / (2 * np.pi)


def compute_ddxs_from_mf15_mf14(
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out,
    to_lab=True, xp=None,
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

    ``xp=None`` (default) is numpy; passing an xp adapter threads
    tracers through MF12/MF13/MF14/MF15 leaves so ``jax.grad``
    reaches the gamma-composition file-side parameters. MF3 xs
    stays numpy internally.

    Returns
    -------
    ddx : ndarray of shape ``(n_einc, n_eouts, n_mus)``. Same units
    and shape as ``compute_ddxs``.
    """
    from ..primitives import array_ns
    if xp is None:
        xp = array_ns.get_backend('numpy')
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
        xs = mf3_interp.compute_cross_section(
            endf_dict, mt, energies_in,
        )   # (n_einc,), numpy
        weight = xp.asarray(xs) * y_cont

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
    # ``select`` predicates take (endf_dict, mt, zap, ...) and do
    # not accept ``xp``. Strip it before passing.
    select_kwargs = {k: v for k, v in kwargs.items() if k != 'xp'}
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


def _compute_residual_xs_for_lfs(endf_dict, mt, za_residual, lfs, energies_in):
    lmf = mf8_interp.get_mf_switch(endf_dict, mt, za_residual, lfs)
    if lmf == 3:
        return mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
    if lmf == 6:
        xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
        y = mf6_interp.compute_yields(
            endf_dict, mt, za_residual, energies_in,
            include_discrete=True, level=lfs,
        )
        return xs * y
    if lmf == 9:
        xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
        y = mf9_interp.compute_yields(
            endf_dict, mt, za_residual, energies_in, level=lfs
        )
        return xs * y
    if lmf == 10:
        return mf10_interp.compute_cross_section(
            endf_dict, mt, za_residual, energies_in, level=lfs
        )
    raise ValueError(
        f'unsupported LMF={lmf} in MF8/MT={mt} for ZAP={za_residual}, LFS={lfs}'
    )


def compute_residual_xs(endf_dict, mt, za_residual, lfs, energies_in):
    """Cross section for producing (za_residual, lfs) via reaction MT.

    Dispatches via MF8 LMF: 3 (MF3 only, no isomer split), 6
    (MF3/MT * MF6/MT yield resolved by LFS), 9 (MF3/MT * MF9/MT
    yield), 10 (MF10/MT directly). When MF8/MT is absent, falls back
    to MF3/MT (and refuses to resolve a non-zero LFS).

    If `lfs` is None, sums contributions from all LFS values present
    in MF8/MT for this ZAP.
    """
    if not (8 in endf_dict and mt in endf_dict[8]):
        if (6 in endf_dict and mt in endf_dict[6]
                and mf6_help.contains_zap(endf_dict, mt, za_residual)):
            xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
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
        return mf3_interp.compute_cross_section(endf_dict, mt, energies_in)

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
            endf_dict, mt, za_residual, lfs, energies_in
        )

    total = np.zeros_like(energies_in, dtype=float)
    for cur_lfs in available_lfs:
        total = total + _compute_residual_xs_for_lfs(
            endf_dict, mt, za_residual, cur_lfs, energies_in
        )
    return total
