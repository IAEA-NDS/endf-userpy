import numpy as np
from ..mfsec_interpretation import mf4_interpretation as mf4_interp
from ..mfsec_interpretation import mf5_interpretation as mf5_interp
from ..mfsec_interpretation import mf6_interpretation as mf6_interp
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from ..mfsec_interpretation import mf12_interpretation as mf12_interp
from ..mfsec_interpretation import mf14_interpretation as mf14_interp
from ..mfsec_interpretation import mf15_interpretation as mf15_interp
from ..primitives.physical_constants import get_zap_for_particle
from ..primitives.properties import (
    is_zap_consistent,
    has_mf4_mt,
    has_mf5_mt,
    has_mf6_mt,
    has_mf12_mt,
    has_mf14_mt,
    has_mf15_mt,
)


_N_ZAP = get_zap_for_particle('n')
_G_ZAP = get_zap_for_particle('g')
from .distribution1d_helpers import (
    integrate_mf6_dist2d_over_eout,
    integrate_mf6_dist2d_over_mu,
    convert_angdist_to_energydist,
)
import logging


module_logger = logging.getLogger(__name__)


def compute_angdist_values(endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True):
    if not is_zap_consistent(endf_dict, mt, zap):
        raise ValueError(f'MT={mt} and ZAP={zap} are not consistent')

    module_logger.debug(f'determine angular distribution for MT: {mt}')

    if has_mf4_mt(endf_dict, mt) and zap == _N_ZAP:
        module_logger.debug('--> found discrete LAW in MF4')
        return mf4_interp.compute_angdist_values(
            endf_dict, mt, energies_in, angle_cosines_out, to_lab
        )

    elif has_mf6_mt(endf_dict, mt):
        found_angdist = False
        angdist = 0.0
        if mf6_help.has_cont_part(endf_dict, mt, zap):
            module_logger.debug('--> integrate MF6')
            found_angdist = True
            angdist += integrate_mf6_dist2d_over_eout(
                endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab
            )
        if mf6_help.has_angdist_part(endf_dict, mt, zap):
            module_logger.debug('--> found discrete LAW in MF6')
            found_angdist = True
            angdist += mf6_interp.compute_angdist_values(
                endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab
            )
        if found_angdist:
            return angdist

    # Gamma with MF14 angular distribution (issue #36 D2). MF14
    # declares the angular distribution per photon line, and MF12
    # provides the per-line yields that combine them into the total
    # gamma angular distribution for this MT.
    if zap == _G_ZAP and has_mf14_mt(endf_dict, mt):
        module_logger.debug('--> found gamma angular distribution in MF14')
        return _compute_mf14_gamma_angdist(
            endf_dict, mt, energies_in, angle_cosines_out,
        )

    # No representable angular distribution for this (MT, ZAP): MF6
    # exists but has no subsection for this ZAP and no MF14 either,
    # or no relevant MF section at all. Return zeros so cumulative
    # summation over MTs stays well-defined; used to raise
    # IndexError, which crashed get_particle_production_dxs_dmu for
    # common gamma-production files after PR #35 admitted these
    # MTs into the sum.
    module_logger.debug(
        f'no angular distribution reconstructable for MT={mt}, '
        f'ZAP={zap}; returning zeros'
    )
    return np.zeros((len(energies_in), len(angle_cosines_out)), dtype=float)


def _compute_mf14_gamma_angdist(
    endf_dict, mt, energies_in, angle_cosines_out,
):
    """Yield-weighted gamma angular distribution from MF14 + MF12.

    Combines per-photon-line angular distributions from MF14 with
    per-photon-line yields from MF12 into a single, MU-normalised
    angular distribution ``f(mu | Ein)`` (i.e. ``int f dmu = 1``)
    suitable for consumption by ``compute_daxs``. Handles three
    layouts:

    - MF14 LI=1 (fully isotropic, the common case in every ad-hoc
      corpus file today): returns ``0.5`` regardless of yields or
      photon energies.
    - MF14 LI=0 with Legendre coefficients per photon line, plus a
      MF12 Eg=0 continuum placeholder: yield-weighted average of the
      MF14 discrete-line distributions plus an isotropic continuum
      contribution weighted by ``y_cont / Y_total``.
    - No MF12 (unusual for a gamma-emitting MT with MF14): treated
      as fully isotropic, since without yield information the only
      neutral choice is to trust MF14's LI flag.

    The MF14 tabulated (LTT=2) branch of the reader is not implemented
    upstream; a MT reaching that branch will raise NotImplementedError
    from ``mf14_interp.compute_angdist_values``.
    """
    energies_in = np.asarray(energies_in, dtype=float)
    angle_cosines_out = np.asarray(angle_cosines_out, dtype=float)
    n_einc = len(energies_in)
    n_mus = len(angle_cosines_out)

    mtsec = endf_dict[14][mt]
    if mtsec['LI'] == 1:
        # Fully isotropic: f(mu) = 1/2, integrates to 1 over mu.
        return np.full((n_einc, n_mus), 0.5, dtype=float)

    if not has_mf12_mt(endf_dict, mt):
        # LI=0 without MF12 yields is a degenerate case: fall back
        # to isotropic rather than guess a partition.
        return np.full((n_einc, n_mus), 0.5, dtype=float)

    pes = np.asarray(mf12_interp.get_photon_energies(endf_dict, mt), dtype=float)
    yields_all = mf12_interp.compute_photon_yields(
        endf_dict, mt, energies_in, pes,
    )
    y_total = yields_all.sum(axis=1)

    disc_mask = pes > 0.0
    disc_pes = pes[disc_mask]
    disc_yields = yields_all[:, disc_mask]
    y_cont = yields_all[:, ~disc_mask].sum(axis=1) if np.any(~disc_mask) else np.zeros(n_einc)

    if len(disc_pes) == 0:
        # Only continuum-placeholder photons (no MF14 per-line data
        # applies): isotropic.
        return np.full((n_einc, n_mus), 0.5, dtype=float)

    per_line = mf14_interp.compute_angdist_values(
        endf_dict, mt, energies_in, disc_pes, angle_cosines_out,
    )  # (n_einc, n_disc, n_mus)

    with np.errstate(divide='ignore', invalid='ignore'):
        weights_disc = disc_yields / y_total[:, np.newaxis]
        cont_frac = y_cont / y_total
    weights_disc = np.where(np.isnan(weights_disc), 0.0, weights_disc)
    cont_frac = np.where(np.isnan(cont_frac), 0.0, cont_frac)

    f_disc_avg = np.einsum('ei,eim->em', weights_disc, per_line)
    f_cont = 0.5 * cont_frac[:, np.newaxis]
    return f_disc_avg + f_cont


def compute_energydist_values(endf_dict, mt, zap, energies_in, energies_out, to_lab=True):
    if not is_zap_consistent(endf_dict, mt, zap):
        raise ValueError(f'MT={mt} and ZAP={zap} are not consistent')

    module_logger.debug(f'determine energy distribution for MT: {mt}')

    if has_mf5_mt(endf_dict, mt) and zap == _N_ZAP:
        if to_lab is not True:
            raise ValueError(
                f"Energy spectrum for MT={mt}, ZAP={zap} reconstruction "
                "from MF5 only possible with `to_lab=True` argument."
            )
        module_logger.debug('--> found energy spectrum in MF5')
        return mf5_interp.compute_spectrum(
            endf_dict, mt, energies_in, energies_out
        )

    elif has_mf4_mt(endf_dict, mt) and zap == _N_ZAP:
        module_logger.debug('--> found discrete angdist in MF4')
        return convert_angdist_to_energydist(
            lambda endf_dict, mt, _, energies_in, energies_out, to_lab: (
                mf4_interp.compute_angdist_values(
                    endf_dict, mt, energies_in, energies_out, to_lab
                )
            ),
            endf_dict, mt, zap, energies_in, energies_out, to_lab
        )

    elif has_mf6_mt(endf_dict, mt):
        found_energydist = False
        mtsec = endf_dict[6][mt]
        energydist = 0.0  # will be broadcasted to correct 2d shape
        if mf6_help.has_cont_part(endf_dict, mt, zap):
            module_logger.debug('--> integrate MF6')
            found_energydist = True
            energydist += integrate_mf6_dist2d_over_mu(
                endf_dict, mt, zap, energies_in, energies_out, to_lab
            )
        if mf6_help.has_angdist_part(endf_dict, mt, zap):
            module_logger.debug('--> found discrete angdist in MF6')
            found_energydist = True
            energydist += convert_angdist_to_energydist(
                mf6_interp.compute_angdist_values,
                endf_dict, mt, zap, energies_in, energies_out, to_lab
            )
        if found_energydist:
            return energydist

    elif has_mf15_mt(endf_dict, mt) and zap == get_zap_for_particle('g'):
        module_logger.debug('--> found continuous gamma energy spectrum in MF15')
        return mf15_interp.compute_spectrum(
            endf_dict, mt, energies_in, energies_out
        )

    # No representable continuous energy spectrum for this (MT, ZAP):
    # either MF6 with only LAW=1 ND>0 discrete-line content (handled
    # separately by the LAW=1 discrete-line folder in the broadening
    # pipeline, and by MF12/14 for the unbroadened case), MF15 without
    # gamma ZAP, or no relevant MF section at all. Return zeros so
    # cumulative summation over MTs is well-defined; used to raise
    # IndexError, which forced defensive try/except workarounds in
    # downstream callers (issue #31).
    module_logger.debug(
        f'no continuum or angdist energy spectrum reconstructable for '
        f'MT={mt}, ZAP={zap}; returning zeros'
    )
    return np.zeros((len(energies_in), len(energies_out)), dtype=float)
