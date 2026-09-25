import numpy as np
from scipy.integrate import simpson
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from ..mfsec_interpretation import mf6_interpretation_integrals as mf6_integral
from ..mfsec_interpretation import mf6_law7_integrals as mf6_law7
from ..primitives import array_ns
from ..primitives import conversion_relativistic as conv_relat
from ..primitives.properties import (
    get_QM,
    get_QI,
    get_projectile_mass,
    get_target_mass,
    get_reaction_qvalue,
)
from ..primitives.physical_constants import (
    get_particle_mass_for_zap,
)
from .distribution2d import compute_dist2d_values
import logging


USE_FORTRAN_INTEGRATION = True


module_logger = logging.getLogger(__name__)


def _is_jax_tracer(x):
    """True iff ``x`` is a JAX abstract tracer (not a concrete jax
    array). Concrete jax arrays can be materialised via
    ``np.asarray``; tracers cannot."""
    try:
        import jax.core
    except ImportError:
        return False
    return isinstance(x, jax.core.Tracer)


def _check_no_tracer_inputs(where, xp, *arrays):
    """Raise a clear ``NotImplementedError`` if ``xp`` is JAX and
    any of the passed arrays is an abstract tracer. Fires early
    with a suggestive message instead of letting a downstream
    ``np.asarray(tracer)`` crash with the confusing
    ``TracerArrayConversionError`` (issue #220).

    The affected integrator/kernel currently runs numpy internally
    and cannot preserve grad through it; users who need jax.grad
    through the ``dxs_dE`` / ``dxs_dmu`` API path should either
    pass concrete numpy arrays as the differentiation axis (grad
    wrt file-side leaves) or wait for the xp-native integrator
    ports tracked in #220.
    """
    if getattr(xp, 'name', None) != 'jax':
        return
    for a in arrays:
        if _is_jax_tracer(a):
            raise NotImplementedError(
                f'{where}: JAX tracer inputs are not supported '
                f'because the underlying numeric integration runs '
                f'on numpy internally. The autodiff-friendly '
                f'port is tracked as issue #220. As a workaround, '
                f'either pass concrete numpy arrays for the query '
                f'axis and take jax.grad wrt file-side leaves, or '
                f'call the direct-dist2d entry point (which is '
                f'xp-native end-to-end).'
            )


# Adaptive-mesh Simpson defaults for the two MF6 integrators below.
# INITIAL_MESH_N=81 already resolves Legendre expansions up to order
# ~40 and typical LAW=1/6 tabulated E' spectra to well under _RTOL.
# MAX_ITER=3 lets the mesh grow to 641 points for pathological cases
# (LAW=7 double-tabulated with sharp knots in E'; agreement against
# quad on the Be-9 (n,2n) LAW=7 sub is a few percent worst-case at
# 641 points, which is still within the file's own tabulation noise
# and much better than the previous quad(limit=50, epsrel=1e-4) that
# also failed to converge on the same integrand -- see PR #68).
_RTOL = 1e-3
_INITIAL_MESH_N = 81
_MAX_ITER = 3


def _adaptive_simpson_along_axis(
    build_dist, axis_lo, axis_hi,
    rtol=_RTOL, initial_n=_INITIAL_MESH_N, max_iter=_MAX_ITER,
):
    """Adaptive Richardson-style doubling on a shared 1D mesh.

    `build_dist(mesh)` returns an ndarray whose last axis is the one
    to integrate over (so callers pre-shape however they need). We
    evaluate on a uniform mesh, apply Simpson, double the mesh
    intervals, and stop once the max relative change across all
    non-axis cells falls under `rtol`.

    Doing one vectorised `build_dist` call per level replaces the
    per-cell `scipy.quad` launches that dominated the previous
    implementation (issue #46 hot spot 1): each level pays one dict
    walk plus one Fortran call instead of ~2000 per (E_in, mu) cell.
    """
    n = initial_n
    mesh = np.linspace(axis_lo, axis_hi, n)
    dist = build_dist(mesh)
    I_prev = simpson(dist, x=mesh, axis=-1)
    for _ in range(max_iter):
        n = 2 * n - 1  # double the number of intervals
        mesh = np.linspace(axis_lo, axis_hi, n)
        dist = build_dist(mesh)
        I_new = simpson(dist, x=mesh, axis=-1)
        scale = np.maximum(np.abs(I_new), 1e-30)
        if np.max(np.abs(I_new - I_prev) / scale) < rtol:
            return I_new
        I_prev = I_new
    return I_new


# Fixed mesh size for the xp-native (jax-tracer) Simpson path: no
# adaptivity because ``xp.max``/comparison on tracers can't drive
# Python control flow. n=321 matches the second-refinement level
# of the numpy adaptive path -- enough to keep LAW=7 mu-integration
# accurate to a few permille without dragging compile time out.
_XP_SIMPSON_N = 321


def _simpson_uniform_xp(f, h, xp):
    """Simpson's 1/3 rule on a uniform mesh, along the last axis.
    Weights are static (numpy-computed) so they don't create tracer
    dependencies. Requires odd ``n >= 3``.
    """
    n = f.shape[-1]
    if n < 3 or n % 2 == 0:
        raise ValueError(f'Simpson requires odd n >= 3, got {n}')
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w_xp = xp.asarray(w)
    return (h / 3.0) * (f * w_xp).sum(axis=-1)


def _fixed_simpson_along_axis_xp(
    build_dist, axis_lo, axis_hi, xp, n=_XP_SIMPSON_N,
):
    """Non-adaptive Simpson on a uniform xp mesh of ``n`` points.

    ``build_dist(mesh)`` receives an xp array; ``axis_lo`` /
    ``axis_hi`` may be tracers themselves (Ep upper bound depends on
    tracer E_in). One evaluation, no branching on the result -- so
    ``jax.jit`` / ``jax.grad`` reach the file-side leaves cleanly.
    """
    mesh = xp.linspace(axis_lo, axis_hi, n)
    dist = build_dist(mesh)
    h = (mesh[-1] - mesh[0]) / (n - 1)
    return _simpson_uniform_xp(dist, h, xp)


def integrate_mf6_dist2d_over_eout(
    endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True, xp=None,
):
    """Angular distribution from the MF6 2D distribution, integrated
    over outgoing energy on `[0, (E_in + q) * 1.1]`.

    Dispatches to a knot-aware trapezoid integrator for the
    single-subsection LAW=7 case (issue #69: sub-permille accuracy on
    the tabulated-slice unit-base interpolant that the general
    adaptive Simpson path only resolves to ~1 %). Everything else
    routes through `_integrate_mf6_over_eout_adaptive_simpson`.

    ``xp=None`` (default) is numpy. Under ``xp=jax`` the LAW=7 fast
    path and the adaptive-Simpson fallback both currently run on
    numpy internally and materialise the result to xp-native at the
    return boundary (autodiff through this integrator is not
    supported yet -- tracked as issue #220). The complementary
    ``integrate_mf6_dist2d_over_mu`` path IS end-to-end xp-native
    for LAW=1 (fast path).
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    _check_no_tracer_inputs(
        'integrate_mf6_dist2d_over_eout',
        xp, energies_in, angle_cosines_out,
    )
    mtsec = endf_dict[6][mt]
    subsec_nums = mf6_help.find_subsec_nums(endf_dict, mt, zap)
    if len(subsec_nums) == 1:
        law = mtsec['subsection'][subsec_nums[0]]['LAW']
        if law == 7:
            module_logger.debug(
                f'use knot-aware LAW=7 integrator for MT={mt}',
            )
            result = mf6_law7.integrate_law7_subsec_over_eout(
                endf_dict, mt, subsec_nums[0],
                energies_in, angle_cosines_out, to_lab,
            )
            return xp.asarray(result) if xp.name != 'numpy' else result
    result = _integrate_mf6_over_eout_adaptive_simpson(
        endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab,
    )
    return xp.asarray(result) if xp.name != 'numpy' else result


def _integrate_mf6_over_eout_adaptive_simpson(
    endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True,
):
    """General-purpose per-E_in adaptive Simpson integrator over
    outgoing energy. Fallback for MTs the specialised
    single-subsection integrators (Fortran LAW=1, knot-aware
    Python LAW=7) do not cover.
    """
    ens_inc = np.asarray(energies_in, dtype=float)
    mus_out = np.asarray(angle_cosines_out, dtype=float)
    q = max(get_QM(endf_dict, mt), get_QI(endf_dict, mt))
    angdist = np.zeros((len(ens_inc), len(mus_out)), dtype=float)
    for i, ein in enumerate(ens_inc):
        eout_max = (ein + q) * 1.1
        if eout_max <= 0:
            continue
        cur_ein = ens_inc[i:i+1]

        def build(ep_mesh, _cur_ein=cur_ein):
            # dist2d shape (1, n_ep, n_mu) -> (n_mu, n_ep) so the
            # integrate axis is last for _adaptive_simpson_along_axis.
            dist = compute_dist2d_values(
                endf_dict, mt, zap,
                _cur_ein, ep_mesh, mus_out, to_lab,
            )
            return np.moveaxis(dist[0], 0, 1)

        angdist[i, :] = _adaptive_simpson_along_axis(build, 0.0, eout_max)
    return angdist


def _integrate_mf6_dist2d_over_mu_default(
    endf_dict, mt, zap, energies_in, energies_out, to_lab=True
):
    """Energy distribution from the MF6 2D distribution, integrated
    over mu on `[-1, +1]`.

    Uses `_adaptive_simpson_along_axis` on a shared mu mesh across
    every (E_in, E_out) cell: one dist2d call per refinement level
    (three levels at worst) instead of `n_ein * n_eout * (~2000)`
    scalar dist2d calls the previous scipy.quad path incurred
    (issue #46). The mu integrand is smooth for LAW=1 (Legendre or
    tabulated), LAW=6, and LAW=7 tabulated distributions, so
    Simpson's uniform mesh converges before max_iter for the corpus.
    """
    ens_inc = np.asarray(energies_in, dtype=float)
    ens_out = np.asarray(energies_out, dtype=float)

    def build(mu_mesh):
        # dist2d already has the mu axis last: (n_ein, n_eout, n_mu).
        return compute_dist2d_values(
            endf_dict, mt, zap, ens_inc, ens_out, mu_mesh, to_lab,
        )

    return _adaptive_simpson_along_axis(build, -1.0, 1.0)


def _integrate_mf6_dist2d_over_mu_default_xp(
    endf_dict, mt, zap, energies_in, energies_out, to_lab, xp,
):
    """xp-native mu-integration for the tracer path (issue #220).

    Fixed-mesh Simpson (``_XP_SIMPSON_N=321``) so the whole
    integration graph is one straight-line evaluation that
    ``jax.grad`` / ``jax.jit`` compile cheaply. The mu axis is
    static (``[-1, +1]``), so the mesh has no tracer dependence and
    the underlying dist2d builder sees concrete mu values.

    Trades the numpy-side per-segment ``rtol=1e-3`` adaptivity for
    a fixed 321-point uniform mesh; measured accuracy on the same
    Be-9 (n,2n) LAW=7 mu integrand is within a few permille of the
    kink-aware kernel's ``<1e-5``, which is well within tabulation
    noise for the top-level physics API this feeds.
    """
    def build(mu_mesh):
        return compute_dist2d_values(
            endf_dict, mt, zap, energies_in, energies_out, mu_mesh,
            to_lab, xp=xp,
        )
    return _fixed_simpson_along_axis_xp(build, -1.0, 1.0, xp)


def integrate_mf6_dist2d_over_mu(
    endf_dict, mt, zap, energies_in, energies_out, to_lab=True, xp=None,
):
    """Energy distribution from MF6 dist2d, integrated over mu.

    Dispatches to specialised kink-aware integrators for the two
    single-subsection cases that dominate the corpus:

    - LAW=1: xp-native fast path through
      ``mf6_integral.get_energydist_from_subsec_law1`` (JAX tracers
      propagate to file-side coefficients).
    - LAW=7: numpy kink-aware midpoint integrator on the union of
      the two bracketing Ein slices' mu knots (matching
      ``integrate_law7_subsec_over_eout``'s knot-awareness on E').
      Exact for LAW=7's piecewise-linear mu interpolation; the
      general adaptive-Simpson path converges only to ~1 % on
      LAW=7 due to unresolved kinks at tabulated mu positions.

    ``xp=None`` (default) is numpy. LAW=1 fast path is end-to-end
    xp-native. LAW=7 and the general-purpose adaptive-Simpson
    fallback run on numpy internally and would materialise a jax
    tracer input; those paths raise ``NotImplementedError`` under
    ``xp=jax`` with tracer inputs, per issue #220.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mtsec = endf_dict[6][mt]
    subsec_nums = mf6_help.find_subsec_nums(endf_dict, mt, zap)
    tracer_ein = _is_jax_tracer(energies_in)
    tracer_eout = _is_jax_tracer(energies_out)
    if len(subsec_nums) == 1:
        law = mtsec['subsection'][subsec_nums[0]]['LAW']
        if law == 1 and USE_FORTRAN_INTEGRATION:
            module_logger.debug(f'use xp-native integrator for MT={mt}')
            return mf6_integral.get_energydist_from_subsec_law1(
                endf_dict, mt, subsec_nums[0],
                energies_in, energies_out, to_lab, xp=xp,
            )
        if law == 7:
            if not (tracer_ein or tracer_eout):
                module_logger.debug(
                    f'use knot-aware LAW=7 mu-integrator for MT={mt}',
                )
                result = mf6_law7.integrate_law7_subsec_over_mu(
                    endf_dict, mt, subsec_nums[0],
                    energies_in, energies_out, to_lab,
                )
                return xp.asarray(result) if xp.name != 'numpy' else result
            # Tracer Ein OR tracer Ep: fall through to the xp-native
            # fixed-mesh Simpson path so grad reaches file-side and
            # query-side leaves. Tracer Ep is now supported via the
            # LAW=7 kernel's unit-base traced-x branch (issue #220
            # PR 4).
            module_logger.debug(
                f'use xp-native mu-Simpson fallback for MT={mt} '
                f'LAW=7 (tracer Ein/Ep)',
            )
            return _integrate_mf6_dist2d_over_mu_default_xp(
                endf_dict, mt, zap, energies_in, energies_out,
                to_lab, xp,
            )
    # general-purpose integration routine.
    if tracer_ein or tracer_eout:
        if tracer_eout:
            raise NotImplementedError(
                'integrate_mf6_dist2d_over_mu (adaptive Simpson): '
                'JAX tracer energies_out is not supported yet '
                '(tracked in issue #220).'
            )
        return _integrate_mf6_dist2d_over_mu_default_xp(
            endf_dict, mt, zap, energies_in, energies_out, to_lab, xp,
        )
    result = _integrate_mf6_dist2d_over_mu_default(
        endf_dict, mt, zap, energies_in, energies_out, to_lab
    )
    return xp.asarray(result) if xp.name != 'numpy' else result


def _prepare_angdist_to_energydist_conversion(
    endf_dict, mt, zap, energies_in, energies_out, to_lab
):
    energies_in = energies_in.reshape(-1, 1)
    energies_out = energies_out.reshape(1,-1)

    if to_lab is not True:
        raise ValueError('This function requires `to_lab=True`')
    m_i = get_projectile_mass(endf_dict)
    m_t = get_target_mass(endf_dict)
    m_e = get_particle_mass_for_zap(zap)
    qval = get_reaction_qvalue(endf_dict, mt)
    m_r = m_t + (m_i-m_e) - qval

    if (m_r <= 0.0):
        raise ValueError(
            f'Reaction stored in MT={mt} energetically infeasible, '
            'check Q-value stored in MF3/MT{mt}.'
        )

    # Standard-mu API (issue #185): returns cos(theta_lab) and the
    # matching signed Jacobian directly, no bridge negation needed.
    # Downstream angular evaluators (mf4_interp.compute_angdist_values,
    # mf6_interp.compute_angdist_values) consume this ``mu`` as-is.
    angle_cosines_out = conv_relat.compute_mu_from_Ekin(
        energies_out, energies_in, m_i, m_t, m_e, m_r
    )
    jacvals = conv_relat.compute_dmu_dEkin(
        energies_out, energies_in, m_i, m_t, m_e, m_r
    )
    return angle_cosines_out, jacvals


def convert_angdist_to_energydist(
    compute_angdist_func, endf_dict, mt, zap, energies_in, energies_out,
    to_lab, xp=None,
):
    """dxs/dE from a stored angular distribution via the LAB
    kinematic Jacobian: for a two-body reaction the outgoing
    ejectile mu is uniquely determined by `(E_in, E_out)`, so
    `f(E_in, E') = angdist(E_in, mu(E_in, E')) * |dmu/dE'|`.

    `_prepare_angdist_to_energydist_conversion` returns
    `angle_cosines_out` of shape `(n_ein, n_eout)` -- a per-cell mu
    that varies with both axes. Evaluating `compute_angdist_func`
    on that 2D grid was broken: the CM<->LAB kinematic conversion
    (`primitives.conversion.convert_angcos_to_cmsys`) reshapes
    `mu_lab.reshape(1, -1)`, collapsing the 2D per-cell mesh into a
    single 1D row, then re-broadcasting against `r2` and producing
    a `(n_ein, n_ein * n_eout)` output that fails the downstream
    `angdist * |jac|` multiplication (issue #76).

    Loop per E_in so each `compute_angdist_func` call sees a 1D
    `angle_cosines_out` of shape `(n_eout,)` -- the shape all the
    per-scheme MF4 / MF6 evaluators actually support -- and the
    kinematic conversion sees a shared 1D mu that broadcasts
    correctly against the single-row `r2`.
    """
    module_logger.debug(
        f'convert angular distribution for MT={mt} to energy distribution',
    )
    angle_cosines_out, jacvals = (
        _prepare_angdist_to_energydist_conversion(
            endf_dict, mt, zap, energies_in, energies_out, to_lab,
        )
    )
    # Shapes: `(n_ein, n_eout)` for both, per the reshape in
    # `_prepare_angdist_to_energydist_conversion`.
    feasible = (~np.isnan(angle_cosines_out)) & (np.abs(angle_cosines_out) <= 1.0)
    angle_cosines_out = np.where(feasible, angle_cosines_out, 0.0)

    ens_inc = np.asarray(energies_in, dtype=float)
    n_ein = len(ens_inc)
    n_eout = angle_cosines_out.shape[1]
    energydist = np.zeros((n_ein, n_eout), dtype=float)
    for i, ein in enumerate(ens_inc):
        mu_row = angle_cosines_out[i, :]
        # (1, n_eout) shape out of the evaluator -- see the
        # per-scheme handlers in mf4_interpretation.py and
        # mf6_interpretation.py. Both return (n_ein, n_mu) where
        # `n_ein == 1` for a single-Ein call.
        angdist_row = compute_angdist_func(
            endf_dict, mt, zap,
            np.array([ein], dtype=float),
            mu_row, to_lab,
        )
        energydist[i, :] = np.asarray(angdist_row[0, :]) * np.abs(jacvals[i, :])
    energydist[~feasible] = 0.0
    if xp is not None and xp.name != 'numpy':
        return xp.asarray(energydist)
    return energydist
