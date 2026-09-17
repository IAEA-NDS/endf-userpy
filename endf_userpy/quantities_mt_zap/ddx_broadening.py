"""Broadening of secondary-distribution cross sections along E_out.

Five per-(MT, ZAP) routines live here:

  - `compute_ddx_continuous_broadened`: 2D DDX continuum, convolved
    along E_out via `adaptive_convolve`.
  - `compute_ddx_discrete_broadened`: 2D DDX for 2-body discrete-
    level channels (MF6/LAW=2 or MF4-only), where the kernel replaces
    the kinematic delta pointwise on the 1D curve E_out = E_out_kin(mu).
  - `compute_ddx_law1_discrete_broadened`: 2D DDX for MF6/LAW=1
    subsections with ND>0, whose discrete-line positions depend on
    mu through the eval-frame -> LAB inverse map rather than through
    2-body kinematics. Each user (E_in, mu) sees a set of Dirac
    peaks at file-tabulated eval-frame energies, mapped to LAB via
    `mf6cm2lab_disc`.
  - `compute_dxs_dE_broadened`: 1D dxs/dE, convolved along E_out via
    `adaptive_convolve`. Continuous and 2-body discrete channels
    share this path because the integration over mu already turns
    the 2-body delta into a finite "kinematic box" with integrable
    singularities at E_out_min, E_out_max.
  - `compute_dxs_dE_law1_discrete_broadened`: 1D analogue of the
    LAW=1 DDX folder: computes the DDX on an internal mu grid,
    integrates over dOmega. For LCT=1 or gamma (light-ejectile)
    channels the discrete position is mu-invariant and the internal
    grid is essentially cosmetic; for heavy-ejectile LCT=2/3 cases
    the mu sweep naturally produces a "kinematic box" smearing of
    the discrete peak that the kernel then convolves.

The dispatchers that combine these into the public API live in
`endf_userpy.quantities`.
"""
import numpy as np
from ..mfsec_interpretation import mf3_interpretation as mf3_interp
from ..mfsec_interpretation import mf4_interpretation as mf4_interp
from ..mfsec_interpretation import mf6_interpretation as mf6_interp
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from ..mfsec_interpretation import mf12_interpretation as mf12_interp
from ..mfsec_interpretation import mf13_interpretation as mf13_interp
from ..mfsec_interpretation import mf14_interpretation as mf14_interp
from ..mfsec_interpretation import mf15_interpretation as mf15_interp
from ..primitives import conversion_relativistic as conv_relat
from ..primitives import reactions as reactions
from ..primitives.convolution import adaptive_convolve
from ..primitives.np_compat import trapezoid
from ..primitives.physical_constants import (
    get_particle_mass_for_zap,
    get_zap_for_particle,
)
from ..primitives.properties import (
    get_projectile,
    get_projectile_mass,
    get_target_mass,
    get_reaction_qvalue,
    has_mf4_mt,
    has_mf5_mt,
    has_mf6_mt,
    has_mf12_mt,
    has_mf13_mt,
    has_mf14_mt,
    has_mf15_mt,
)
from .distribution2d import compute_dist2d_values
from .quantities import compute_yields, compute_dexs
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
    # DDX of a physical distribution is non-negative; FFT roundoff in
    # adaptive_convolve can produce sub-eps negatives at the tails,
    # which trip users who assert non-negativity or plot on log axes.
    # Clip them here rather than in the generic primitive.
    return np.clip(ddx * yields * xs / (2 * np.pi), 0.0, None)


def compute_ddx_continuous_broadened_summed(
    endf_dict, mts, zap,
    energies_in, energies_out, angle_cosines_out,
    kernel, kernel_width,
    to_lab=True,
    **convolve_kwargs,
):
    """DDX of `sum_{MT in mts}` convolved with `kernel` along E_out
    in a single `adaptive_convolve` call (issue #26).

    Uses linearity of convolution: `sum_m conv(dist2d_m * y_m * xs_m)
    == conv(sum_m dist2d_m * y_m * xs_m)`, so instead of one FFT per
    MT, one FFT covers the sum. Each internal-mesh evaluation now
    computes the sum of `M = len(mts)` dist2d contributions rather
    than one, so the dist2d cost stays roughly the same; the FFT
    count drops by a factor of M and the Python outer-loop overhead
    of `adaptive_convolve` amortises across the whole sum.

    Callers must pre-filter `mts` to those that pass
    `has_continuous_ddx(mt, zap)`; discrete two-body / LAW=1 discrete
    / MF12 discrete channels go through their own kernel folders.

    A one-MT list is accepted but there is no gain over the per-MT
    routine, so the top-level dispatcher only routes through this
    function when `len(mts) >= 2`.

    Returns
    -------
    ddx : ndarray of shape `(n_einc, n_eouts, n_mus)`.
    """
    einc = np.asarray(energies_in, dtype=float)
    eouts = np.asarray(energies_out, dtype=float)
    mus = np.asarray(angle_cosines_out, dtype=float)

    if len(mts) == 0:
        return np.zeros(
            (len(einc), len(eouts), len(mus)), dtype=float,
        )

    # Pre-compute the per-MT (yield * xs) scaling once. These depend
    # only on Ein so they don't participate in the convolution and
    # don't need to be recomputed inside f_summed.
    scales = []
    for mt in mts:
        y = compute_yields(
            endf_dict, mt, zap, einc, include_discrete=False,
        )
        xs = mf3_interp.compute_cross_section(endf_dict, mt, einc)
        scales.append((y * xs).reshape(-1, 1, 1))

    def f_summed(eout_internal):
        total = None
        for mt, scale in zip(mts, scales):
            dist2d = compute_dist2d_values(
                endf_dict, mt, zap, einc, eout_internal, mus, to_lab,
            )
            # (n_einc, n_eout, n_mus) -> (n_einc, n_mus, n_eout)
            contrib = np.moveaxis(dist2d, 1, -1) * scale
            total = contrib if total is None else total + contrib
        return total

    broadened = adaptive_convolve(
        f_summed, kernel, eouts,
        kernel_width=kernel_width, **convolve_kwargs,
    )
    # (n_einc, n_mus, n_eouts) -> (n_einc, n_eouts, n_mus)
    ddx = np.moveaxis(broadened, -1, 1)
    return np.clip(ddx / (2 * np.pi), 0.0, None)


def compute_dxs_dE_broadened(
    endf_dict, mt, zap,
    energies_in, energies_out,
    kernel, kernel_width,
    to_lab=True,
    **convolve_kwargs,
):
    """1D dxs/dE for (MT, ZAP) convolved with `kernel` along E_out.

    Wraps `adaptive_convolve` around the existing `compute_dexs`. No
    separate discrete branch is needed: integrating the (E_out, mu)
    delta over mu already turns 2-body discrete-level channels into
    finite "kinematic boxes" with 1/sqrt singularities at E_out_min
    and E_out_max. Those singularities are integrable, the Gaussian
    kernel low-passes them, and the convolved result is smooth; the
    adaptive doubling may need a few extra iterations near the
    kinematic boundaries to converge.

    Parameters
    ----------
    endf_dict, mt, zap, energies_in, energies_out, to_lab
        Same as `compute_dexs`.
    kernel : callable
        `kernel(delta_E)`. Should integrate to ~1 over its support.
    kernel_width : float
        Characteristic kernel width in eV.
    **convolve_kwargs
        Forwarded to `adaptive_convolve`.

    Returns
    -------
    dxs_dE : ndarray
        Broadened dxs/dE, shape `(n_einc, n_eouts)`. Same units as
        `compute_dexs`.
    """
    energies_in = np.asarray(energies_in, dtype=float)
    energies_out = np.asarray(energies_out, dtype=float)

    def f(eout_internal):
        return compute_dexs(
            endf_dict, mt, zap, energies_in, eout_internal, to_lab,
        )

    try:
        # dxs/dE of a physical spectrum is non-negative; clip sub-eps
        # FFT-noise negatives from adaptive_convolve for the same
        # reason as in compute_ddx_continuous_broadened.
        return np.clip(adaptive_convolve(
            f, kernel, energies_out,
            kernel_width=kernel_width,
            **convolve_kwargs,
        ), 0.0, None)
    except (IndexError, AssertionError):
        # Defensive: earlier revisions caught IndexError from
        # compute_energydist_values falling through with only MF6/
        # LAW=1 ND>0 content (issue #31 has since been fixed and it
        # now returns zeros directly, so this specific case no longer
        # reaches us). The catch is still needed because
        #   * compute_yields (mf6_interpretation.compute_yields) can
        #     raise IndexError for MT/ZAP combinations where MF6 has
        #     no subsection carrying that ZAP;
        #   * primitives.properties.get_ejectile asserts on multi-
        #     ejectile MTs where the first ejectile is not a neutron
        #     (issue #32; e.g. Fe-56 MT 112 = (n,p a) with 'p' first).
        # Both mean the cont path has nothing to contribute for this
        # MT/ZAP; the LAW=1 discrete folder (dispatched separately)
        # or the 2-body folder handles the actual content. Return
        # zeros so cumulative summation is well-defined.
        return np.zeros(
            (len(energies_in), len(energies_out)), dtype=float,
        )


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


def compute_ddx_law1_discrete_broadened(
    endf_dict, mt, zap,
    energies_in, energies_out, angle_cosines_out,
    kernel,
    to_lab=True,
):
    """DDX contribution from MF6/LAW=1 discrete-energy lines (ND>0),
    with the kinematic delta at each line replaced by `kernel`.

    The line positions ep_disc_lab(E_in, mu) and eval-frame-weighted
    amplitudes amp(E_in, mu) come from
    `mf6_interp.compute_law1_discrete_lines`, which reports them per
    subsection carrying this ZAP and concatenates along the last axis.
    Each user grid cell (E_in, E_out, mu) accumulates
    `kernel(E_out - ep_disc_lab) * amp * xs * yield / (2 pi)` summed
    over lines. Cells where the CM->LAB inverse map has no physical
    solution are marked by amp == 0 in the wrapper output and
    contribute nothing.

    yield uses `compute_yields(include_discrete=True)`, i.e. the full
    MF6 yield attributed to this ZAP. Since `compute_yields` treats
    the ND>0 and continuum portions of a LAW=1 subsection as one
    lump, this same full yield also flows into the continuous folder
    (with `include_discrete=False` there, which is a no-op for LAW=1
    subsections). The b(k) amplitude carried inside amp partitions
    the distribution across discrete lines vs continuum, so the two
    folders together sum to the full production distribution without
    double-counting.

    Callers must gate on `has_mf6_law1_discrete_lines` for the
    channel; calling on a MT/ZAP with no LAW=1 ND>0 content returns
    a zero DDX.

    Returns
    -------
    ddx : ndarray of shape (n_einc, n_eouts, n_mus).
    """
    energies_in = np.asarray(energies_in, dtype=float)
    energies_out = np.asarray(energies_out, dtype=float)
    angle_cosines_out = np.asarray(angle_cosines_out, dtype=float)

    ep_disc_lab, amp_disc = mf6_interp.compute_law1_discrete_lines(
        endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab,
    )
    # Shapes: (n_einc, n_mus, K)
    ddx = np.zeros(
        (len(energies_in), len(energies_out), len(angle_cosines_out)),
        dtype=float,
    )
    if ep_disc_lab.shape[-1] == 0:
        return ddx

    # (1, n_eouts, 1) - (n_einc, 1, n_mus) -> (n_einc, n_eouts, n_mus)
    for k in range(ep_disc_lab.shape[-1]):
        pos = ep_disc_lab[:, np.newaxis, :, k]  # (n_einc, 1, n_mus)
        amp = amp_disc[:, np.newaxis, :, k]     # same shape
        # Cells where the map failed have amp == 0; the kernel value
        # at whatever pos happens to be there is multiplied by zero.
        delta = energies_out[np.newaxis, :, np.newaxis] - pos
        ddx += np.asarray(kernel(delta)) * amp

    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=True,
    ).reshape(-1, 1, 1)
    xs = mf3_interp.compute_cross_section(
        endf_dict, mt, energies_in,
    ).reshape(-1, 1, 1)
    return ddx * yields * xs / (2 * np.pi)


def compute_ddx_mf12_discrete_broadened(
    endf_dict, mt, zap,
    energies_in, energies_out, angle_cosines_out,
    kernel,
):
    """DDX contribution from MF12 discrete photon lines with the MF14
    angular distribution factored in, and each Dirac peak at Eg_i
    replaced by ``kernel`` along E_out.

    For each discrete photon line ``i`` at energy ``Eg_i`` in MF12,
    adds::

        sigma(Ein) * y_i(Ein) * kernel(Eout - Eg_i)
                * f_i(mu | Ein) / (2 pi)

    to the result, where ``f_i(mu | Ein)`` is the per-line angular
    distribution (integrating to 1 over mu) taken from MF14 when
    present, or isotropic 0.5 as the neutral fallback when MF14 is
    absent. The overall ``1/(2 pi)`` matches the convention of the
    other DDX folders here (``compute_ddx_law1_discrete_broadened``,
    ``compute_ddx_continuous_broadened``).

    Layouts:

    - MF14 LI=1 (fully isotropic; the common case): ``f_i(mu) = 0.5``
      independent of line index. The DDX is the D1 discrete-line
      folder times ``0.5 / (2 pi) = 1 / (4 pi)``.
    - MF14 LI=0 with Legendre coefficients: ``f_i(mu)`` varies per
      line and is looked up in MF14 for each ``Eg_i > 0``. The MF12
      Eg=0 continuum placeholder is dropped from this folder; its
      contribution is picked up by
      `compute_ddx_mf15_continuum_broadened` (issue #123).
    - No MF14 for this MT (unusual for a gamma-emitting MT): treat
      as isotropic.

    Callers must gate on ``has_mf12_discrete_lines(mt, zap)`` for the
    channel; calling on a MT/ZAP with no MF12 discrete lines returns
    a zero DDX.

    Returns
    -------
    ddx : ndarray of shape ``(n_einc, n_eouts, n_mus)``.
    """
    if zap != get_zap_for_particle('g'):
        raise ValueError(
            'MF12 discrete-line broadening is gamma-only; got '
            f'ZAP={zap}'
        )
    energies_in = np.asarray(energies_in, dtype=float)
    energies_out = np.asarray(energies_out, dtype=float)
    angle_cosines_out = np.asarray(angle_cosines_out, dtype=float)
    n_einc = len(energies_in)
    n_eouts = len(energies_out)
    n_mus = len(angle_cosines_out)
    result = np.zeros((n_einc, n_eouts, n_mus), dtype=float)
    if not has_mf12_mt(endf_dict, mt):
        return result

    photon_energies = mf12_interp.get_photon_energies(endf_dict, mt)
    if photon_energies is None:
        return result
    photon_energies = np.asarray(photon_energies, dtype=float)
    disc_mask = photon_energies > 0.0
    if not np.any(disc_mask):
        return result

    yields_all = mf12_interp.compute_photon_yields(
        endf_dict, mt, energies_in, photon_energies,
    )
    Eg_disc = photon_energies[disc_mask]
    yields_disc = yields_all[:, disc_mask]  # (n_einc, n_disc)

    xs = mf3_interp.compute_cross_section(
        endf_dict, mt, energies_in,
    )  # (n_einc,)
    weight_E = yields_disc * xs[:, np.newaxis]  # (n_einc, n_disc)

    # Per-line angular distribution f_i(mu | Ein), shape
    # (n_einc, n_disc, n_mus).
    if has_mf14_mt(endf_dict, mt):
        mtsec14 = endf_dict[14][mt]
        if mtsec14['LI'] == 1:
            per_line_angdist = np.full(
                (n_einc, len(Eg_disc), n_mus), 0.5, dtype=float,
            )
        else:
            per_line_angdist = mf14_interp.compute_angdist_values(
                endf_dict, mt, energies_in, Eg_disc, angle_cosines_out,
            )
    else:
        per_line_angdist = np.full(
            (n_einc, len(Eg_disc), n_mus), 0.5, dtype=float,
        )

    for k in range(len(Eg_disc)):
        e_kernel = np.asarray(kernel(energies_out - Eg_disc[k]))
        # (n_einc, 1, 1) * (1, n_eouts, 1) * (n_einc, 1, n_mus)
        result += (
            weight_E[:, k].reshape(-1, 1, 1)
            * e_kernel.reshape(1, -1, 1)
            * per_line_angdist[:, k, :].reshape(n_einc, 1, n_mus)
        )
    return np.clip(result / (2 * np.pi), 0.0, None)


def compute_dxs_dE_mf12_discrete_broadened(
    endf_dict, mt, zap, energies_in, energies_out, kernel,
):
    """1D dxs/dE contribution from discrete photon lines declared in
    MF12, with each Dirac peak at Eg_i replaced by ``kernel``.

    For gamma-only, ZAP-checked by the caller via the
    ``has_mf12_discrete_lines`` selector. The photon-line positions
    ``Eg_i`` are read from ``mf12_interp.get_photon_energies`` and the
    per-line yields ``y_i(Ein)`` from ``mf12_interp.compute_photon_yields``.
    A photon energy of 0 in MF12 is the continuum-spectrum placeholder
    (its shape lives in MF15) and is excluded here; the continuum
    contribution flows through the ordinary continuous folder via
    ``compute_dexs`` / ``compute_energydist_values``.

    Returns
    -------
    dxs_dE : ndarray of shape ``(n_einc, n_eouts)``. Units match
    ``compute_dexs``: barn / eV, without the ``1/(2pi)`` factor that
    the DDX folders apply.

    The full formula for the folded contribution is::

        dxs_dE(Ein, Eout) = sigma(Ein) * sum_i y_i(Ein)
                                * kernel(Eout - Eg_i)

    where ``sigma(Ein)`` is the MF3 cross section for this MT.

    Notes
    -----
    MF13 (per-line photon production cross section) is not yet
    handled here. Files that carry gamma yields only in MF13 will
    contribute zero from this folder; support can be added by
    substituting ``mf13_interp.compute_photon_production_xs`` for
    ``sigma * y_i`` line by line.
    """
    if zap != get_zap_for_particle('g'):
        raise ValueError(
            'MF12 discrete-line broadening is gamma-only; got '
            f'ZAP={zap}'
        )
    energies_in = np.asarray(energies_in, dtype=float)
    energies_out = np.asarray(energies_out, dtype=float)
    result = np.zeros(
        (len(energies_in), len(energies_out)), dtype=float,
    )
    if not has_mf12_mt(endf_dict, mt):
        return result

    photon_energies = mf12_interp.get_photon_energies(endf_dict, mt)
    if photon_energies is None:
        return result
    photon_energies = np.asarray(photon_energies, dtype=float)
    disc_mask = photon_energies > 0.0
    if not np.any(disc_mask):
        return result

    # compute_photon_yields returns shape (n_einc, n_photen). We
    # request the full set (including any Eg=0 placeholder) so the
    # underlying reader keeps a consistent index; we then slice out
    # the discrete rows.
    yields_all = mf12_interp.compute_photon_yields(
        endf_dict, mt, energies_in, photon_energies,
    )
    Eg_disc = photon_energies[disc_mask]
    yields_disc = yields_all[:, disc_mask]  # (n_einc, n_disc_lines)

    xs = mf3_interp.compute_cross_section(
        endf_dict, mt, energies_in,
    )  # (n_einc,)
    weight = (yields_disc * xs[:, np.newaxis])  # (n_einc, n_disc_lines)

    # Per-line kernel folding. Loop over the K discrete lines rather
    # than materialising a (n_einc, n_eouts, K) tensor -- K is small
    # for LO=2 partial channels (typically 1..a few) and moderate for
    # LO=1 capture files (~300 for Al-27) but the loop stays flat
    # anyway and keeps memory linear in n_eouts.
    for k in range(len(Eg_disc)):
        delta = energies_out - Eg_disc[k]  # (n_eouts,)
        result += (
            np.asarray(kernel(delta))[np.newaxis, :]
            * weight[:, k].reshape(-1, 1)
        )
    return np.clip(result, 0.0, None)


def compute_dxs_dE_mf13_discrete_broadened(
    endf_dict, mt, zap, energies_in, energies_out, kernel,
):
    """1D dxs/dE contribution from discrete photon lines declared in
    MF13, with each Dirac peak at ``Eg_i`` replaced by ``kernel``.

    MF13 stores the per-line photon-production cross section
    ``sigma_gamma_i(Ein)`` directly (issue #29's XS-side pattern),
    so the fold reduces to::

        dxs_dE(Ein, Eout) = sum_i sigma_gamma_i(Ein) * kernel(Eout - Eg_i)

    -- no multiplication by MF3 sigma or MF12 yields needed. This
    is the MF13 analogue of ``compute_dxs_dE_mf12_discrete_broadened``
    and closes the differential-side gap noted in that function's
    docstring (issue #101).

    Gamma-only. A photon energy of 0 in MF13 is the
    continuum-spectrum placeholder combined with MF15 (same
    convention as MF12) and is excluded here.
    """
    if zap != get_zap_for_particle('g'):
        raise ValueError(
            'MF13 discrete-line broadening is gamma-only; got '
            f'ZAP={zap}'
        )
    energies_in = np.asarray(energies_in, dtype=float)
    energies_out = np.asarray(energies_out, dtype=float)
    result = np.zeros(
        (len(energies_in), len(energies_out)), dtype=float,
    )
    if not has_mf13_mt(endf_dict, mt):
        return result

    photon_energies = mf13_interp.get_photon_energies(endf_dict, mt)
    if photon_energies is None or len(photon_energies) == 0:
        return result
    photon_energies = np.asarray(photon_energies, dtype=float)
    disc_mask = photon_energies > 0.0
    if not np.any(disc_mask):
        return result
    Eg_disc = photon_energies[disc_mask]

    # (n_einc, n_disc): per-line photon-production XS in barn.
    prod_xs = mf13_interp.compute_photon_production_xs(
        endf_dict, mt, energies_in, Eg_disc,
    )

    for k in range(len(Eg_disc)):
        delta = energies_out - Eg_disc[k]
        result += (
            np.asarray(kernel(delta))[np.newaxis, :]
            * prod_xs[:, k].reshape(-1, 1)
        )
    return np.clip(result, 0.0, None)


def compute_ddx_mf13_discrete_broadened(
    endf_dict, mt, zap,
    energies_in, energies_out, angle_cosines_out,
    kernel,
):
    """DDX contribution from MF13 discrete photon lines with the
    MF14 angular distribution factored in, and each Dirac peak at
    ``Eg_i`` replaced by ``kernel`` along E_out.

    Sibling of ``compute_ddx_mf12_discrete_broadened``; differs only
    in reading the per-line photon-production cross section from
    MF13 (``mf13_interp.compute_photon_production_xs``) rather than
    ``sigma * y_i`` from MF3 + MF12. The per-line angular
    distribution ``f_i(mu | Ein)`` is looked up the same way (MF14
    LI=1 -> isotropic, LI=0 -> per-line Legendre; no MF14 -> fall
    back to isotropic).

    Returns
    -------
    ddx : ndarray of shape ``(n_einc, n_eouts, n_mus)``.
    """
    if zap != get_zap_for_particle('g'):
        raise ValueError(
            'MF13 discrete-line broadening is gamma-only; got '
            f'ZAP={zap}'
        )
    energies_in = np.asarray(energies_in, dtype=float)
    energies_out = np.asarray(energies_out, dtype=float)
    angle_cosines_out = np.asarray(angle_cosines_out, dtype=float)
    n_einc = len(energies_in)
    n_eouts = len(energies_out)
    n_mus = len(angle_cosines_out)
    result = np.zeros((n_einc, n_eouts, n_mus), dtype=float)
    if not has_mf13_mt(endf_dict, mt):
        return result

    photon_energies = mf13_interp.get_photon_energies(endf_dict, mt)
    if photon_energies is None or len(photon_energies) == 0:
        return result
    photon_energies = np.asarray(photon_energies, dtype=float)
    disc_mask = photon_energies > 0.0
    if not np.any(disc_mask):
        return result
    Eg_disc = photon_energies[disc_mask]

    # (n_einc, n_disc): per-line photon-production XS in barn.
    prod_xs = mf13_interp.compute_photon_production_xs(
        endf_dict, mt, energies_in, Eg_disc,
    )

    # Per-line angular distribution f_i(mu | Ein), shape
    # (n_einc, n_disc, n_mus). Same MF14 selection as the MF12
    # sibling; MF14 doesn't distinguish MF12 from MF13 as its yield
    # source.
    if has_mf14_mt(endf_dict, mt):
        mtsec14 = endf_dict[14][mt]
        if mtsec14['LI'] == 1:
            per_line_angdist = np.full(
                (n_einc, len(Eg_disc), n_mus), 0.5, dtype=float,
            )
        else:
            per_line_angdist = mf14_interp.compute_angdist_values(
                endf_dict, mt, energies_in, Eg_disc, angle_cosines_out,
            )
    else:
        per_line_angdist = np.full(
            (n_einc, len(Eg_disc), n_mus), 0.5, dtype=float,
        )

    for k in range(len(Eg_disc)):
        e_kernel = np.asarray(kernel(energies_out - Eg_disc[k]))
        result += (
            prod_xs[:, k].reshape(-1, 1, 1)
            * e_kernel.reshape(1, -1, 1)
            * per_line_angdist[:, k, :].reshape(n_einc, 1, n_mus)
        )
    return np.clip(result / (2 * np.pi), 0.0, None)


def compute_ddx_mf15_continuum_broadened(
    endf_dict, mt, zap,
    energies_in, energies_out, angle_cosines_out,
    kernel, kernel_width,
    to_lab=True,
    **convolve_kwargs,
):
    """DDX contribution from the MF15 continuous gamma spectrum,
    convolved with `kernel` along `E_out`.

    For gamma emission, this is the companion of
    `compute_ddx_mf12_discrete_broadened` (which folds the MF12
    discrete photon lines). MF15 tabulates the continuous shape
    ``spec(E_out | E_in)`` normalised to unit integral over E_out
    at each E_in; MF12 carries the yield ``y_cont(E_in)`` of the
    continuum photon "line" (the Eg=0 placeholder subsection). The
    DDX contribution for a single (E_in, E_out, mu) is::

        DDX(E_in, E_out, mu) = sigma(E_in)
            * (y_cont(E_in) / Y_total(E_in))
            * spec(E_out | E_in)
            * f_cont(mu | E_in) / (2 pi)

    where ``sigma`` is the MF3 cross section for MT, and the
    continuum angular distribution ``f_cont`` comes from MF14 (Eg=0
    entry, LI=0 with LTT=1 Legendre) when present, or falls back to
    isotropic ``0.5`` when MF14 is fully isotropic (LI=1) or absent
    or has no distinct continuum entry.

    Callers must gate this routine on
    ``has_mf15_continuum(endf_dict, mt, zap)`` for the channel; a MT
    with no MF15 returns a zero DDX. When MF12 has no Eg=0 continuum
    placeholder (a rare corpus shape, see issue #103), MF15 has no
    normalising yield to weight against and the contribution is
    dropped -- the caller-facing warning in
    `distribution1d.compute_energydist_values` already fires for the
    1D counterpart of the same file, so no separate warning is
    emitted here.

    Parameters
    ----------
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab
        Same as `compute_ddx_continuous_broadened`.
    kernel : callable
        `kernel(delta_E)`. Should integrate to ~1 over its support.
    kernel_width : float
        Characteristic kernel width in eV; passed to
        `adaptive_convolve` to set the internal mesh and truncation.
    **convolve_kwargs
        Forwarded to `adaptive_convolve` (e.g. `rtol`, `max_iter`).

    Returns
    -------
    ddx : ndarray of shape ``(n_einc, n_eouts, n_mus)``. Units match
    `compute_ddx_continuous_broadened`: barn / eV / sr.
    """
    if zap != get_zap_for_particle('g'):
        raise ValueError(
            'MF15 continuum broadening is gamma-only; got '
            f'ZAP={zap}'
        )
    energies_in = np.asarray(energies_in, dtype=float)
    energies_out = np.asarray(energies_out, dtype=float)
    angle_cosines_out = np.asarray(angle_cosines_out, dtype=float)
    n_einc = len(energies_in)
    n_eouts = len(energies_out)
    n_mus = len(angle_cosines_out)
    result_zero = np.zeros((n_einc, n_eouts, n_mus), dtype=float)

    if not has_mf15_mt(endf_dict, mt):
        return result_zero

    # Compose the (xs * y_cont) weight for the gamma-continuum
    # contribution. Two shapes are supported (issue #130 for the
    # MF13-only branch):
    #  - Standard MF12+MF15 MT: xs from MF3 * MF12 Eg=0
    #    continuum-placeholder yield.
    #  - MF13-only MT (JENDL-5 MT 3 style, no MF3): MF13 IS the
    #    total gamma production XS -- use it directly, no MF3/MF12
    #    lookup needed.
    if mt not in endf_dict.get(3, {}) and mt in endf_dict.get(13, {}):
        weight = mf13_interp.compute_total_photon_production_xs(
            endf_dict, mt, energies_in,
        )   # (n_einc,)
    else:
        # Continuum yield from MF12 (the Eg=0 placeholder subsection).
        # If MF12 has no Eg=0 placeholder, the continuum yield is
        # undefined and we drop the MF15 contribution -- same
        # convention as the 1D path (issue #103).
        if not has_mf12_mt(endf_dict, mt):
            return result_zero
        pes = np.asarray(
            mf12_interp.get_photon_energies(endf_dict, mt), dtype=float,
        )
        cont_mask = pes == 0.0
        if not np.any(cont_mask):
            return result_zero
        yields_all = mf12_interp.compute_photon_yields(
            endf_dict, mt, energies_in, pes,
        )
        y_cont = yields_all[:, cont_mask].sum(axis=1)   # (n_einc,)
        xs = mf3_interp.compute_cross_section(
            endf_dict, mt, energies_in,
        )   # (n_einc,)
        weight = xs * y_cont
        # NOTE on normalisation: the 1D dxs/dE path composes
        # `compute_dexs = compute_yields * xs * compute_energydist_values`
        # where compute_yields returns Y_total (all photons: discrete
        # + continuum) and compute_energydist_values (MF15 branch)
        # returns spec * (y_cont / Y_total). Net contribution to the
        # 1D integral is therefore `xs * y_cont`. This folder computes
        # the DDX contribution directly, so we skip the y_cont /
        # Y_total scaling and use y_cont as the weight -- matching
        # the physical normalisation that the 2D DDX integrated over
        # (E_out, mu) returns `xs * y_cont`.

    # Continuum angular distribution f_cont(mu | Ein), shape
    # (n_einc, n_mus). MF14 LI=1 is fully isotropic (the common
    # case in every ad-hoc corpus file today). MF14 LI=0 with an
    # Eg=0 entry: use that entry as the continuum angular. Absent
    # MF14, fall back to isotropic (the neutral choice: MF14's
    # own default for LI=1 is isotropic).
    if has_mf14_mt(endf_dict, mt) and endf_dict[14][mt]['LI'] == 0:
        cont_angdist = mf14_interp.compute_angdist_values(
            endf_dict, mt, energies_in,
            np.array([0.0]), angle_cosines_out,
        )
        # Shape (n_einc, 1, n_mus) -> (n_einc, n_mus).
        f_cont = cont_angdist[:, 0, :]
    else:
        f_cont = np.full((n_einc, n_mus), 0.5, dtype=float)

    # Broaden the MF15 spectrum along E_out. adaptive_convolve
    # expects f(eout) returning an array with the E_out axis last;
    # MF15's compute_spectrum returns (n_einc, n_eout). No mu axis
    # -- f_cont is broadcast in afterwards.
    def f_spec(eout_internal):
        return mf15_interp.compute_spectrum(
            endf_dict, mt, energies_in, eout_internal,
        )

    broadened_spec = adaptive_convolve(
        f_spec, kernel, energies_out,
        kernel_width=kernel_width,
        **convolve_kwargs,
    )  # shape (n_einc, n_eouts)

    # Assemble: weight * spec * f_cont / (2 pi)
    #   (n_einc, 1, 1) * (n_einc, n_eouts, 1) * (n_einc, 1, n_mus)
    ddx = (
        weight.reshape(-1, 1, 1)
        * broadened_spec.reshape(n_einc, n_eouts, 1)
        * f_cont.reshape(n_einc, 1, n_mus)
    )
    # FFT roundoff can produce sub-eps negatives at the tails; clip
    # for consistency with the other broadened folders.
    return np.clip(ddx / (2 * np.pi), 0.0, None)


def compute_dxs_dE_law1_discrete_broadened(
    endf_dict, mt, zap,
    energies_in, energies_out,
    kernel,
    to_lab=True,
    n_mu_internal=64,
):
    """1D analogue of `compute_ddx_law1_discrete_broadened`: DDX of
    MF6/LAW=1 ND>0 discrete lines with the kinematic delta replaced
    by `kernel`, then integrated over the outgoing solid angle.

    The 1D projection reduces to
      dxs/dE(E_in, E_out) =
          xs * yield * integral_over_dOmega( kernel(E_out - ep_lab(mu))
                                             * amp(mu) )
    which we approximate by running the 2D DDX folder on an internal
    mu grid and using trapezoid over dOmega = 2 pi dmu.

    For LCT=1 subsections and for gamma emission (awp=0, so the
    LCT=2/3 CM->LAB mapping degenerates to identity), the discrete
    position ep_lab(mu) is mu-invariant and the projection is a
    pointwise kernel evaluation weighted by the isotropic-projection
    of the angular distribution. For LCT=2/3 with a heavy ejectile,
    ep_lab(mu) sweeps a kinematic range as mu moves over [-1, +1]:
    the integrand becomes a mu-parametrised curve and the projection
    is a "kinematic box" smearing of the discrete peak, convolved
    with the kernel. A modest internal mu grid captures both regimes.

    Parameters match `compute_dxs_dE_broadened` except for the extra
    `n_mu_internal` knob controlling the mu-quadrature density.

    Returns
    -------
    dxs_dE : ndarray of shape (n_einc, n_eouts). Same units as
    `compute_dexs`.
    """
    if n_mu_internal < 2:
        raise ValueError('n_mu_internal must be >= 2')
    mus = np.linspace(-1.0, 1.0, n_mu_internal)
    ddx = compute_ddx_law1_discrete_broadened(
        endf_dict, mt, zap,
        energies_in, energies_out, mus,
        kernel, to_lab=to_lab,
    )
    return trapezoid(ddx, mus, axis=-1) * (2 * np.pi)


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
