"""Broadening of double-differential cross sections along E_out.

This module holds per-(MT, ZAP) routines that apply a 1D convolution
kernel along the outgoing-energy axis. The continuous-part routine
wraps `adaptive_convolve` around the existing 2D distribution
machinery; a separate discrete-part routine (to be added later) will
handle two-body discrete-level channels analytically.

The dispatcher that combines both lives in `endf_userpy.quantities`.
"""
import numpy as np
from ..mfsec_interpretation import mf3_interpretation as mf3_interp
from ..primitives.convolution import adaptive_convolve
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
