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


def _concrete_bool(x):
    """``bool(x)``, or None when ``x`` is abstract (under ``jax.jit``)."""
    try:
        return bool(x)
    except Exception as exc:
        try:
            import jax.errors
        except ImportError:
            raise exc
        if isinstance(exc, jax.errors.ConcretizationTypeError):
            return None
        raise


def _adaptive_simpson_along_axis_xp(
    build_dist, axis_lo, axis_hi, xp,
    rtol=_RTOL, initial_n=_INITIAL_MESH_N, max_iter=_MAX_ITER,
):
    """:func:`_adaptive_simpson_along_axis` for traced inputs (issue
    #220): the same meshes and the same stopping rule, with the
    data-dependent stop expressed as nested ``xp.cond`` (``lax.cond``)
    when it is abstract (under ``jax.jit``), so only the levels numpy
    would evaluate run.
    ``axis_lo`` / ``axis_hi`` may be traced; ``build_dist`` receives
    an xp mesh.
    """
    # Concrete bounds keep a numpy mesh: inside an ``xp.cond`` branch
    # even a constant ``jnp.linspace`` is a tracer, and some dist2d
    # kernels need the integration axis concrete (LAW=7 with traced E').
    concrete = not (_is_jax_tracer(axis_lo) or _is_jax_tracer(axis_hi))

    def level(n):
        if concrete:
            mesh = np.linspace(axis_lo, axis_hi, n)
        else:
            mesh = xp.linspace(axis_lo, axis_hi, n)
        h = (axis_hi - axis_lo) / (n - 1)
        return _simpson_uniform_xp(build_dist(mesh), h, xp)

    def refine(it, n_prev, I_prev):
        n = 2 * n_prev - 1
        I_new = level(n)
        if it == max_iter:
            return I_new
        scale = xp.maximum(xp.abs(I_new), 1e-30)
        converged = xp.max(xp.abs(I_new - I_prev) / scale) < rtol
        decided = _concrete_bool(converged)
        if decided is not None:
            # Eager (incl. jax.grad): branch in Python. A cond branch
            # would trace its body and turn concrete inputs into
            # tracers.
            return I_new if decided else refine(it + 1, n, I_new)
        return xp.cond(
            converged,
            lambda: I_new,
            lambda: refine(it + 1, n, I_new),
        )

    I_first = level(initial_n)
    if max_iter == 0:
        return I_first
    return refine(1, initial_n, I_first)


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

    ``xp=None`` (default) is numpy. Under ``xp=jax`` the LAW=7
    fast path is end-to-end xp-native and reverse-mode gradients
    flow through the ``f`` amplitudes and ``Ep`` knot positions
    of the underlying tables (issue #220). With concrete inputs the
    adaptive-Simpson fallback runs on numpy and materialises the
    result to xp-native at the return boundary; with traced
    incident energies (``jax.jit`` over them) both paths run
    xp-native and reproduce the numpy quadrature
    (:func:`_integrate_mf6_over_eout_adaptive_simpson_traced`,
    :mod:`mf6_law7_integrals_traced`); the same holds for traced
    ``angle_cosines_out``.

    The LAW=7 fast path fires whenever every MF6 subsection
    matching ``zap`` is LAW=7 (a common pattern for files that
    carry more than one channel contributing to the same
    ejectile, e.g. multiple reaction paths summed on one ZAP).
    Contributions from all matching subsections are summed
    panel-exactly. Mixed-LAW cases (LAW=7 alongside LAW=1/2/6)
    still route through the adaptive-Simpson fallback.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    traced = xp.name == 'jax' and (
        _is_jax_tracer(energies_in) or _is_jax_tracer(angle_cosines_out)
    )
    mtsec = endf_dict[6][mt]
    subsec_nums = mf6_help.find_subsec_nums(endf_dict, mt, zap)
    if subsec_nums and all(
        mtsec['subsection'][sn]['LAW'] == 7 for sn in subsec_nums
    ):
        module_logger.debug(
            f'use knot-aware LAW=7 integrator for MT={mt} '
            f'({len(subsec_nums)} subsection(s))',
        )
        total = mf6_law7.integrate_law7_subsec_over_eout(
            endf_dict, mt, subsec_nums[0],
            energies_in, angle_cosines_out, to_lab, xp=xp,
        )
        for sn in subsec_nums[1:]:
            total = total + mf6_law7.integrate_law7_subsec_over_eout(
                endf_dict, mt, sn,
                energies_in, angle_cosines_out, to_lab, xp=xp,
            )
        return total
    if traced:
        return _integrate_mf6_over_eout_adaptive_simpson_traced(
            endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab, xp,
        )
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


def _integrate_mf6_over_eout_adaptive_simpson_traced(
    endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab, xp,
):
    """:func:`_integrate_mf6_over_eout_adaptive_simpson` for traced
    incident energies and / or angle cosines (issue #220): one
    ``xp.map`` body per ``E_in`` running the same adaptive Simpson on
    ``[0, (E_in + q) * 1.1]``."""
    mus_out = xp.asarray(angle_cosines_out, dtype=xp.float64).reshape(-1)
    q = max(get_QM(endf_dict, mt), get_QI(endf_dict, mt))
    e_in = xp.asarray(energies_in, dtype=xp.float64).reshape(-1)

    def one_ein(ein):
        eout_max = (ein + q) * 1.1
        positive = eout_max > 0
        hi = xp.where(positive, eout_max, 1.0)

        def build(ep_mesh):
            dist = compute_dist2d_values(
                endf_dict, mt, zap, ein.reshape(1), ep_mesh, mus_out,
                to_lab, xp=xp,
            )
            return xp.moveaxis(dist[0], 0, 1)

        row = _adaptive_simpson_along_axis_xp(build, 0.0, hi, xp)
        return xp.where(positive, row, 0.0)

    return xp.map(one_ein, e_in)


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
    """:func:`_integrate_mf6_dist2d_over_mu_default` for traced
    incident energies (issue #220): the same adaptive Simpson, with
    the stop expressed through ``xp.cond``. The mu axis is static
    (``[-1, +1]``), so the meshes are concrete."""
    def build(mu_mesh):
        return compute_dist2d_values(
            endf_dict, mt, zap, energies_in, energies_out, mu_mesh,
            to_lab, xp=xp,
        )
    return _adaptive_simpson_along_axis_xp(build, -1.0, 1.0, xp)


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
    xp-native. With traced incident energies (``jax.jit`` over them)
    LAW=7 and the general-purpose adaptive Simpson reproduce the
    numpy quadrature xp-native (issue #220); traced
    ``energies_out`` reach the adaptive Simpson for LAW=7 only.
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
        if law == 7 and not tracer_eout:
            # Concrete or traced Ein (the latter through the
            # host-precomputed per-panel path, issue #220).
            module_logger.debug(
                f'use knot-aware LAW=7 mu-integrator for MT={mt}',
            )
            return mf6_law7.integrate_law7_subsec_over_mu(
                endf_dict, mt, subsec_nums[0],
                energies_in, energies_out, to_lab, xp=xp,
            )
        if law == 7:
            # Tracer Eout on the query side: the xp-native adaptive
            # Simpson over mu, so grad reaches query-side leaves.
            module_logger.debug(
                f'use xp-native mu-Simpson fallback for MT={mt} '
                f'LAW=7 (tracer Ep)',
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


def _two_body_masses(endf_dict, mt, zap, to_lab):
    """``(m_i, m_t, m_e, m_r)`` of the two-body kinematics of ``mt``
    with ejectile ``zap`` (residual mass from the reaction Q-value)."""
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
    return m_i, m_t, m_e, m_r


def _prepare_angdist_to_energydist_conversion(
    endf_dict, mt, zap, energies_in, energies_out, to_lab
):
    energies_in = energies_in.reshape(-1, 1)
    energies_out = energies_out.reshape(1,-1)
    m_i, m_t, m_e, m_r = _two_body_masses(endf_dict, mt, zap, to_lab)

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
    if getattr(xp, 'name', None) == 'jax' and _is_jax_tracer(energies_in):
        return _convert_angdist_to_energydist_traced(
            compute_angdist_func, endf_dict, mt, zap, energies_in,
            energies_out, to_lab, xp,
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


def _convert_angdist_to_energydist_traced(
    compute_angdist_func, endf_dict, mt, zap, energies_in, energies_out,
    to_lab, xp,
):
    """:func:`convert_angdist_to_energydist` for traced incident
    energies (``jax.jit`` over them; issue #220): the kinematics and
    the feasibility mask in ``xp``, the per-``E_in`` angular evaluation
    as one ``xp.map`` body (``lax.map`` inside a trace) instead of the
    Python loop. ``compute_angdist_func`` must accept ``xp=``.
    ``energies_out`` stays concrete."""
    m_i, m_t, m_e, m_r = _two_body_masses(endf_dict, mt, zap, to_lab)
    e_in = xp.asarray(energies_in, dtype=xp.float64).reshape(-1)
    e_out = xp.asarray(np.asarray(energies_out, dtype=float).reshape(1, -1))
    mu = conv_relat.compute_mu_from_Ekin(
        e_out, e_in.reshape(-1, 1), m_i, m_t, m_e, m_r, xp=xp,
    )                                                       # (n_ein, n_eout)
    jac = conv_relat.compute_dmu_dEkin(
        e_out, e_in.reshape(-1, 1), m_i, m_t, m_e, m_r, xp=xp,
    )
    feasible = (~xp.isnan(mu)) & (xp.abs(mu) <= 1.0)
    mu = xp.where(feasible, mu, 0.0)

    def one_ein(args):
        ein, mu_row = args
        return compute_angdist_func(
            endf_dict, mt, zap, ein.reshape(1), mu_row, to_lab, xp=xp,
        )[0, :]

    angdist = xp.map(one_ein, (e_in, mu))
    return xp.where(feasible, angdist * xp.abs(jac), 0.0)
