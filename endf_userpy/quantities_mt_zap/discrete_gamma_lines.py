"""Per-line records for the delta-shaped photon-emission content
that the unbroadened DDX / dxs_dE dispatchers cannot render on a
finite E_out grid (issue #271).

Three ENDF-6 conventions carry the same physics ("a photon emitted
at a fixed nuclear-transition energy Eg") and are extracted here
under one uniform record type:

- MF12 (with MF3 * yield) discrete photon multiplicities.
- MF13 per-line photon-production cross sections.
- MF6 LAW=1 ND>0 subsections with ZAP=0.

The user-facing entry point
:func:`endf_userpy.quantities.get_particle_production_discrete_gamma_lines`
resolves the reaction string, applies the widened-MT selector, and
delegates the per-MT / per-MF extraction to :func:`extract_discrete_gamma_lines`
in this module.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..primitives import array_ns
from ..primitives import physical_constants as physconst
from ..primitives import properties as prop
from ..mfsec_interpretation import mf3_interpretation as mf3_interp
from ..mfsec_interpretation import mf6_interpretation as mf6_interp
from ..mfsec_interpretation import mf12_interpretation as mf12_interp
from ..mfsec_interpretation import mf13_interpretation as mf13_interp
from ..mfsec_interpretation import mf14_interpretation as mf14_interp
from . import quantities as quant_mt_zap
from . import selectors


@dataclass
class DiscreteGammaLine:
    """One discrete photon line's per-(MT, Eg) record.

    Returned by :func:`extract_discrete_gamma_lines` and by the
    user-facing wrapper
    :func:`endf_userpy.quantities.get_particle_production_discrete_gamma_lines`.

    Fields
    ------
    mt : int
        Source MT number.
    mf : int
        Source MF section (12 for MF12+MF3, 13 for MF13, 6 for
        MF6/LAW=1 ND>0).
    Eg : float
        Photon energy in eV.
    weight : array shape ``(n_einc,)``
        Amplitude at each incident energy. For MF12 sources this
        is ``sigma(Ein) * y_i(Ein)``; for MF13 sources it is the
        MF13 per-line photon-production cross section directly;
        for MF6/LAW=1 sources it is
        ``int_mu(amp(Ein, mu, k)) * yield(Ein) * sigma(Ein)``.
        All in barn. Under ``xp=jax`` this is an xp array that
        carries tracers if the caller has swapped an MF3 / MF12 /
        MF13 / MF6 column for a jax array.
    angdist : optional array shape ``(n_einc, n_mus)``
        Per-line angular distribution ``f_i(mu | Ein)``. Populated
        only when ``angle_cosines_out`` is passed to the entry
        point. For MF12 / MF13 the shape follows MF14 (LI=1 gives
        the isotropic ``0.5``, LI=0 goes through
        :func:`mf14_interpretation.compute_angdist_values`). For
        MF6/LAW=1 it is the mu-normalised ``amp / int_mu(amp)``
        so that ``int_-1^+1 angdist dmu == 1``, matching the MF14
        convention.
    """
    mt: int
    mf: int
    Eg: float
    weight: object
    angdist: Optional[object] = None


_MF6_LAW1_EP_REL_TOL = 1e-3
"""Relative tolerance for treating MF6/LAW=1 ND>0 discrete-line Ep
values as constant across the (Ein, mu) grid. Real corpus files
store the same nuclear-transition Ep at every Ein slice with
per-mu Doppler shifts of at most a few ppm; anything above this
threshold likely indicates a non-transition delta (e.g. bremsstrahlung
tail) and is skipped with a UserWarning naming the MT and slot.
"""


def extract_discrete_gamma_lines(
    endf_dict, mts, user_mts, energies_in, *,
    angle_cosines_out=None, xp=None,
):
    """Iterate the widened-MT set and extract every discrete
    gamma-line record for gamma emission from MF12, MF13, and
    MF6/LAW=1 ND>0. Returns a list of :class:`DiscreteGammaLine`
    sorted by ``Eg`` ascending.

    ``mts`` is the widened-MT iteration set (typically
    ``mf3_interpretation.get_reaction_mts_widened(endf_dict)``);
    ``user_mts`` is the caller-selected MT list used by the
    particle-production admission filter.

    See the module docstring for the physics conventions and the
    user-facing wrapper for reaction-string / policy plumbing.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    zap = physconst.get_zap_for_particle('g')
    energies_in = np.asarray(energies_in, dtype=float)
    n_einc = len(energies_in)
    want_angdist = angle_cosines_out is not None
    if want_angdist:
        angle_cosines_out = np.asarray(angle_cosines_out, dtype=float)

    lines = []
    for mt in mts:
        if not selectors.contains_zap(endf_dict, mt, zap):
            continue
        if not selectors.satisfies_particle_production_select(
            endf_dict, mt, user_mts, zap,
        ):
            continue
        lines.extend(_extract_mf12_lines(
            endf_dict, mt, energies_in, angle_cosines_out,
            n_einc, want_angdist, xp,
        ))
        lines.extend(_extract_mf13_lines(
            endf_dict, mt, energies_in, angle_cosines_out,
            n_einc, want_angdist, xp,
        ))
        lines.extend(_extract_mf6_law1_lines(
            endf_dict, mt, energies_in, angle_cosines_out,
            n_einc, want_angdist, xp,
        ))
    lines.sort(key=lambda ln: ln.Eg)
    return lines


def _fetch_per_line_angdist(
    endf_dict, mt, energies_in, Eg_disc, angle_cosines_out,
    n_einc, n_mus, xp,
):
    """MF14 lookup for per-line angular distributions.

    MF14 LI=1 (isotropic) and no-MF14 both return the isotropic
    ``0.5`` fallback; MF14 LI=0 with LTT=1 (Legendre) goes through
    ``mf14_interpretation.compute_angdist_values``.
    """
    if prop.has_mf14_mt(endf_dict, mt):
        mtsec14 = endf_dict[14][mt]
        if mtsec14['LI'] == 1:
            return xp.full(
                (n_einc, len(Eg_disc), n_mus), 0.5, dtype=xp.float64,
            )
        return mf14_interp.compute_angdist_values(
            endf_dict, mt, energies_in, Eg_disc, angle_cosines_out,
            xp=xp,
        )
    return xp.full(
        (n_einc, len(Eg_disc), n_mus), 0.5, dtype=xp.float64,
    )


def _extract_mf12_lines(
    endf_dict, mt, energies_in, angle_cosines_out,
    n_einc, want_angdist, xp,
):
    if not selectors.has_mf12_discrete_lines(
        endf_dict, mt, physconst.get_zap_for_particle('g'),
    ):
        return []
    photon_energies = mf12_interp.get_photon_energies(endf_dict, mt)
    if photon_energies is None:
        return []
    photon_energies = np.asarray(photon_energies, dtype=float)
    disc_mask = photon_energies > 0.0
    if not np.any(disc_mask):
        return []
    Eg_disc = photon_energies[disc_mask]
    disc_idcs = np.where(disc_mask)[0]
    yields_all = mf12_interp.compute_photon_yields(
        endf_dict, mt, energies_in, photon_energies, xp=xp,
    )
    yields_disc = yields_all[:, disc_idcs]
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
    weight_E = yields_disc * xp.asarray(xs[:, None])
    ang = None
    if want_angdist:
        n_mus = len(angle_cosines_out)
        ang = _fetch_per_line_angdist(
            endf_dict, mt, energies_in, Eg_disc, angle_cosines_out,
            n_einc, n_mus, xp,
        )
    return [
        DiscreteGammaLine(
            mt=int(mt), mf=12, Eg=float(Eg_disc[k]),
            weight=weight_E[:, k],
            angdist=(ang[:, k, :] if ang is not None else None),
        )
        for k in range(len(Eg_disc))
    ]


def _extract_mf13_lines(
    endf_dict, mt, energies_in, angle_cosines_out,
    n_einc, want_angdist, xp,
):
    if not selectors.has_mf13_discrete_lines(
        endf_dict, mt, physconst.get_zap_for_particle('g'),
    ):
        return []
    photon_energies = mf13_interp.get_photon_energies(endf_dict, mt)
    if photon_energies is None or len(photon_energies) == 0:
        return []
    photon_energies = np.asarray(photon_energies, dtype=float)
    disc_mask = photon_energies > 0.0
    if not np.any(disc_mask):
        return []
    Eg_disc = photon_energies[disc_mask]
    prod_xs = mf13_interp.compute_photon_production_xs(
        endf_dict, mt, energies_in, Eg_disc, xp=xp,
    )
    ang = None
    if want_angdist:
        n_mus = len(angle_cosines_out)
        ang = _fetch_per_line_angdist(
            endf_dict, mt, energies_in, Eg_disc, angle_cosines_out,
            n_einc, n_mus, xp,
        )
    return [
        DiscreteGammaLine(
            mt=int(mt), mf=13, Eg=float(Eg_disc[k]),
            weight=prod_xs[:, k],
            angdist=(ang[:, k, :] if ang is not None else None),
        )
        for k in range(len(Eg_disc))
    ]


def _extract_mf6_law1_lines(
    endf_dict, mt, energies_in, angle_cosines_out,
    n_einc, want_angdist, xp,
):
    """MF6/LAW=1 ND>0 discrete gamma-line extraction.

    Unlike MF12/MF13, MF6 stores each discrete-line ``Ep`` per
    ``(Ein, mu)`` slice rather than as a scalar transition
    energy. For a real nuclear-de-excitation gamma line the values
    should be Ein/mu-independent to numerical noise; when the
    per-cell drift exceeds ``_MF6_LAW1_EP_REL_TOL`` this extractor
    warns and skips the line rather than fabricating a per-Ein
    scalar. This preserves the "MF6/LAW=1 ND>0 discrete gammas
    describe the same physics as MF12/MF13" invariant for the
    returned records.

    The per-line weight follows the same convention as the MF12
    branch: ``weight[Ein]`` is the total gamma-production cross
    section attributed to line k, i.e.
    ``int_mu(amp) * yield * sigma`` (barn). The per-line angular
    distribution is ``amp / int_mu(amp)`` normalised so that
    ``int_-1^+1 angdist dmu == 1``, matching the MF14 convention
    used by the MF12/MF13 records.
    """
    zap_g = physconst.get_zap_for_particle('g')
    if not selectors.has_mf6_law1_discrete_lines(
        endf_dict, mt, zap_g,
    ):
        return []
    mus_for_check = (
        np.asarray(angle_cosines_out, dtype=float)
        if want_angdist
        else np.linspace(-1.0, 1.0, 21)
    )
    ep_disc_lab, amp_disc = mf6_interp.compute_law1_discrete_lines(
        endf_dict, mt, zap_g, energies_in, mus_for_check, to_lab=True,
    )
    n_lines = ep_disc_lab.shape[-1]
    if n_lines == 0:
        return []
    yields = quant_mt_zap.compute_yields(
        endf_dict, mt, zap_g, energies_in, include_discrete=True,
        xp=xp,
    )
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
    yields_xs = yields * xp.asarray(xs)

    out = []
    for k in range(n_lines):
        ep_k = ep_disc_lab[:, :, k]
        amp_k = amp_disc[:, :, k]
        valid = amp_k > 0.0
        if not np.any(valid):
            continue
        ep_valid = ep_k[valid]
        ep_ref = float(ep_valid.flat[0])
        if ep_ref <= 0.0:
            continue
        drift = float(np.abs(ep_valid - ep_ref).max())
        rel_drift = drift / max(abs(ep_ref), 1e-30)
        if rel_drift > _MF6_LAW1_EP_REL_TOL:
            warnings.warn(
                f'MF6/LAW=1 MT={mt} discrete-line slot {k}: '
                f'per-(Ein, mu) Ep varies by rel {rel_drift:.2e} '
                f'(>{_MF6_LAW1_EP_REL_TOL:.0e}); does not look like '
                f'a fixed nuclear-transition line -- skipping. '
                f'Pass `broadening=` to include this content via the '
                f'kinematic-delta folder instead.',
                UserWarning, stacklevel=4,
            )
            continue
        amp_int_mu = np.trapezoid(amp_k, mus_for_check, axis=1)
        weight = xp.asarray(amp_int_mu) * yields_xs
        if want_angdist:
            angdist_np = amp_k / np.maximum(
                amp_int_mu[:, None], 1e-300,
            )
            angdist = xp.asarray(angdist_np)
        else:
            angdist = None
        out.append(DiscreteGammaLine(
            mt=int(mt), mf=6, Eg=ep_ref,
            weight=weight, angdist=angdist,
        ))
    return out
