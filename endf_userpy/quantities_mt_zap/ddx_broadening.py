"""Broadening of double-differential cross sections along E_out.

This module holds per-(MT, ZAP) routines that apply a 1D convolution
kernel along the outgoing-energy axis. The continuous-part routine
wraps `adaptive_convolve` around the existing 2D distribution
machinery; the discrete-part routine evaluates the kernel directly
at the kinematic locus E_out_kin(E_in, mu) without any convolution
because the underlying distribution is a Dirac line in (E_out, mu).

The dispatcher that combines both lives in `endf_userpy.quantities`.
"""
import numpy as np
from ..mfsec_interpretation import mf3_interpretation as mf3_interp
from ..mfsec_interpretation import mf4_interpretation as mf4_interp
from ..mfsec_interpretation import mf6_interpretation as mf6_interp
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from ..primitives import conversion_relativistic as conv_relat
from ..primitives import reactions as reactions
from ..primitives.convolution import adaptive_convolve
from ..primitives.physical_constants import get_particle_mass_for_zap
from ..primitives.properties import (
    get_projectile,
    get_projectile_mass,
    get_target_mass,
    get_reaction_qvalue,
    has_mf4_mt,
    has_mf5_mt,
    has_mf6_mt,
)
from .distribution2d import compute_dist2d_values
from .quantities import compute_yields
import logging


module_logger = logging.getLogger(__name__)


def compute_ddx_continuous_broadened(
    endf_dict, mt, zap,
    energies_in, energies_out, angle_cosines_out,
    kernel, kernel_width,
    to_lab=True,
    **convolve_kwargs,
):
    """DDX of (MT, ZAP) convolved with `kernel` along E_out.

    Only the continuous part of the distribution is evaluated; the
    underlying `compute_dist2d_values` returns whatever continuous
    distribution the MF6/MF4+MF5 source provides. Callers must gate
    this routine on `has_continuous_ddx` for the channel; calling it
    on an MT whose distribution is a pure kinematic delta produces
    a meaningless result (zeros, or a NaN, depending on how the
    source code treats off-curve evaluation).

    Parameters
    ----------
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab
        Same as `compute_ddxs`.
    kernel : callable
        Convolution kernel `kernel(delta_E)`. Should integrate to ~1
        over its support so the production cross section is conserved.
    kernel_width : float
        Characteristic kernel width in eV; passed to `adaptive_convolve`
        to set the initial internal-mesh spacing and the truncation
        range.
    **convolve_kwargs
        Forwarded to `adaptive_convolve` (e.g. `rtol`, `max_iter`,
        `richardson`).

    Returns
    -------
    ddx : ndarray
        Broadened DDX, shape `(n_einc, n_eouts, n_mus)`. Same units as
        `compute_ddxs`.
    """
    energies_in = np.asarray(energies_in, dtype=float)
    energies_out = np.asarray(energies_out, dtype=float)
    angle_cosines_out = np.asarray(angle_cosines_out, dtype=float)

    def f(eout_internal):
        # compute_dist2d_values returns (n_einc, n_eout, n_mus); we need
        # E_out on the last axis for adaptive_convolve, so move axis 1
        # to the back.
        dist2d = compute_dist2d_values(
            endf_dict, mt, zap,
            energies_in, eout_internal, angle_cosines_out, to_lab,
        )
        return np.moveaxis(dist2d, 1, -1)

    # shape (n_einc, n_mus, n_eouts) after adaptive_convolve
    broadened = adaptive_convolve(
        f, kernel, energies_out,
        kernel_width=kernel_width,
        **convolve_kwargs,
    )
    # restore E_out as middle axis: (n_einc, n_eouts, n_mus)
    ddx = np.moveaxis(broadened, -1, 1)

    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=False,
    ).reshape(-1, 1, 1)
    xs = mf3_interp.compute_cross_section(
        endf_dict, mt, energies_in,
    ).reshape(-1, 1, 1)
    return ddx * yields * xs / (2 * np.pi)


def compute_ddx_discrete_broadened(
    endf_dict, mt, zap,
    energies_in, energies_out, angle_cosines_out,
    kernel,
    to_lab=True,
):
    """DDX of the 2-body discrete-level part of (MT, ZAP), with the
    kinematic delta delta(E_out - E_out_kin(E_in, mu)) replaced by
    kernel(E_out - E_out_kin(E_in, mu)).

    No convolution is performed: the underlying distribution is a
    1D curve in (E_out, mu) space, so the kernel is evaluated
    pointwise at every grid cell. Callers must gate this routine on
    `has_discrete_two_body_ddx` for the channel.

    Parameters
    ----------
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab
        Same as `compute_ddxs`.
    kernel : callable
        Kernel `kernel(delta_E)`. Must accept multi-dim ndarrays of
        offsets and return values of the same shape (most numpy-based
        kernels satisfy this automatically). Should integrate to ~1
        over its support so the production cross section is conserved.

    Returns
    -------
    ddx : ndarray
        Broadened discrete DDX, shape `(n_einc, n_eouts, n_mus)`. Same
        units as `compute_ddxs`.
    """
    energies_in = np.asarray(energies_in, dtype=float)
    energies_out = np.asarray(energies_out, dtype=float)
    angle_cosines_out = np.asarray(angle_cosines_out, dtype=float)

    angdist = _compute_discrete_angdist(
        endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab,
    )  # (n_einc, n_mus)
    eout_kin = _compute_eout_kin(
        endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab,
    )  # (n_einc, n_mus)

    # K(E_out_j - E_out_kin(E_in_i, mu_k)) for every grid cell.
    delta = (
        energies_out.reshape(1, -1, 1)
        - eout_kin.reshape(eout_kin.shape[0], 1, eout_kin.shape[1])
    )
    feasible = np.isfinite(eout_kin) & (eout_kin >= 0.0)
    delta = np.where(feasible[:, None, :], delta, 0.0)
    kernel_vals = np.asarray(kernel(delta))
    kernel_vals = np.where(feasible[:, None, :], kernel_vals, 0.0)

    angdist_b = angdist.reshape(angdist.shape[0], 1, angdist.shape[1])
    yields = _compute_discrete_yields(
        endf_dict, mt, zap, energies_in,
    ).reshape(-1, 1, 1)
    xs = mf3_interp.compute_cross_section(
        endf_dict, mt, energies_in,
    ).reshape(-1, 1, 1)
    return kernel_vals * angdist_b * xs * yields / (2 * np.pi)


def _compute_discrete_angdist(
    endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab,
):
    """Angular distribution g(mu|E_in) for a 2-body discrete channel."""
    if has_mf6_mt(endf_dict, mt) and mf6_help.has_angdist_part(endf_dict, mt, zap):
        return mf6_interp.compute_angdist_values(
            endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab,
        )
    if has_mf4_mt(endf_dict, mt) and not has_mf5_mt(endf_dict, mt):
        return mf4_interp.compute_angdist_values(
            endf_dict, mt, energies_in, angle_cosines_out, to_lab,
        )
    raise ValueError(
        f"MT={mt}, ZAP={zap} has no 2-body discrete-level angular "
        "distribution (need MF6/LAW=2/3/4 or MF4-only)."
    )


def _compute_eout_kin(
    endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab,
):
    """E_out_kin(E_in, mu) for a 2-body reaction.

    The conversion_relativistic primitives use cos_phi = cos(pi -
    theta_lab) = -mu (see header of conversion_relativistic.py); we
    feed them -mu to bridge that convention.
    """
    if to_lab is not True:
        raise ValueError("compute_ddx_discrete_broadened requires to_lab=True")

    m_i = get_projectile_mass(endf_dict)
    m_t = get_target_mass(endf_dict)
    m_e = get_particle_mass_for_zap(zap)
    qval = get_reaction_qvalue(endf_dict, mt)
    m_r = m_t + (m_i - m_e) - qval
    if m_r <= 0.0:
        raise ValueError(
            f"Reaction stored in MT={mt} energetically infeasible "
            f"(m_r <= 0); check Q-value in MF3/MT{mt}."
        )

    eout = conv_relat.compute_Ekin_from_cos_phi(
        cos_phi=-angle_cosines_out.reshape(1, -1),
        Ekin_i=energies_in.reshape(-1, 1),
        m_i=m_i, m_t=m_t, m_e=m_e, m_r=m_r,
    )
    return eout  # shape (n_einc, n_mus)


def _compute_discrete_yields(endf_dict, mt, zap, energies_in):
    """Yield of the discrete two-body part of (MT, ZAP).

    For a 2-body discrete-level channel the outgoing-particle
    multiplicity is fixed by the reaction (1 for (n,n'), 1 for (n,p'),
    etc.), so we read it from the reaction-string table. This avoids
    depending on MF6 yield bookkeeping, which in some evaluations
    (e.g. JENDL-5 U-238 MT 51..76) only lists the heavy residual in
    its subsections and not the light ejectile.
    """
    proj = get_projectile(endf_dict)
    mult = reactions.get_multiplicity_for_zap(proj, mt, zap)
    if mult is None:
        raise ValueError(
            f"No multiplicity defined for projectile={proj}, MT={mt}, "
            f"ZAP={zap}; cannot compute discrete yield."
        )
    return np.full(len(energies_in), float(mult))
