import numpy as np
from scipy.integrate import simpson
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from ..mfsec_interpretation import mf6_interpretation_integrals as mf6_integral
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


def integrate_mf6_dist2d_over_eout(
    endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True
):
    """Angular distribution from the MF6 2D distribution, integrated
    over outgoing energy on `[0, (E_in + q) * 1.1]`.

    Uses `_adaptive_simpson_along_axis` per E_in: the upper limit is
    E_in-dependent so the mesh cannot be shared across incident
    energies, but each E_in gets a single vectorised dist2d call per
    refinement level instead of `n_mu * (~2000)` scalar calls that
    the previous scipy.quad path incurred (issue #46).
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


def integrate_mf6_dist2d_over_mu(
    endf_dict, mt, zap, energies_in, energies_out, to_lab=True
):
    mtsec = endf_dict[6][mt]
    subsec_nums = mf6_help.find_subsec_nums(endf_dict, mt, zap)
    if len(subsec_nums) == 1:
        law = mtsec['subsection'][subsec_nums[0]]['LAW']
        if law == 1 and USE_FORTRAN_INTEGRATION:
            module_logger.debug(f'use Fortran integration routine for MT={mt}')
            return mf6_integral.get_energydist_from_subsec_law1(
                endf_dict, mt, subsec_nums[0], energies_in, energies_out, to_lab
            )
    # general-purpose integration routine
    return _integrate_mf6_dist2d_over_mu_default(
        endf_dict, mt, zap, energies_in, energies_out, to_lab
    )


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

    # conv_relat.compute_cos_phi_from_Ekin returns cos(pi - theta_lab),
    # i.e. the cosine of the angle between the ejectile and the
    # *opposite* of the beam direction. The downstream angular-
    # distribution evaluators (mf4_interp.compute_angdist_values,
    # mf6_interp.compute_angdist_values) take the standard
    # mu = cos(theta_lab); negate to bridge the conventions. The
    # Jacobian sign flips with it; np.abs() in the caller masks the
    # sign of jacvals, so this is purely cosmetic for the magnitude
    # but kept consistent with the cos negation.
    angle_cosines_out = -conv_relat.compute_cos_phi_from_Ekin(
        energies_out, energies_in, m_i, m_t, m_e, m_r
    )
    jacvals = -conv_relat.compute_dcos_phi_dEkin(
        energies_out, energies_in, m_i, m_t, m_e, m_r
    )
    return angle_cosines_out, jacvals


def convert_angdist_to_energydist(
    compute_angdist_func, endf_dict, mt, zap, energies_in, energies_out, to_lab
):
    module_logger.debug(f'convert angular distribution for MT={mt} to energy distribution')
    angle_cosines_out, jacvals = (
        _prepare_angdist_to_energydist_conversion(
            endf_dict, mt, zap, energies_in, energies_out, to_lab
        )
    )
    feasible = (~np.isnan(angle_cosines_out)) & (np.abs(angle_cosines_out) <= 1.0)
    angle_cosines_out[~feasible] = 0.0  # to avoid warnings
    angdist_values = compute_angdist_func(
        endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab
    )
    energydist = angdist_values * np.abs(jacvals) 
    energydist[~feasible] = 0.0
    return energydist
