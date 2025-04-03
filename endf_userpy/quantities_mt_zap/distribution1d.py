import numpy as np
from ..mfsec_interpretation import mf4_interpretation as mf4_interp
from ..mfsec_interpretation import mf5_interpretation as mf5_interp
from ..mfsec_interpretation import mf6_interpretation as mf6_interp
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from ..mfsec_interpretation import mf15_interpretation as mf15_interp
from ..primitives.properties import (
    is_zap_consistent,
    has_mf4_mt,
    has_mf5_mt,
    has_mf6_mt,
    has_mf15_mt,
)
from .distribution1d_helpers import (
    integrate_mf6_dist2d_over_eout,
    integrate_mf6_dist2d_over_mu,
    convert_angdist_to_energydist,
)
import logging


module_logger = logging.getLogger(__name__)


def compute_angdist_values(endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True):
    if not is_zap_consistent(endf_dict, mt, zap):
        raise ValueError(f'MT={mt} and ZAP={mt} are not consistent')

    module_logger.debug(f'determine angular distribution for MT: {mt}')

    if has_mf4_mt(endf_dict, mt):
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

    raise IndexError(
        f'Required data to reconstruct angular distribution '
        f'for MT={mt} not available.'
    )


def compute_energydist_values(endf_dict, mt, zap, energies_in, energies_out, to_lab=True):
    if not is_zap_consistent(endf_dict, mt, zap):
        raise ValueError(f'MT={mt} and ZAP={zap} are not consistent')

    module_logger.debug(f'determine energy distribution for MT: {mt}')

    if has_mf5_mt(endf_dict, mt):
        if to_lab is not True:
            raise ValueError(
                f"Energy spectrum for MT={mt}, ZAP={zap} reconstruction "
                "from MF5 only possible with `to_lab=True` argument."
            )
        module_logger.debug('--> found energy spectrum in MF5')
        return mf5_interp.compute_spectrum(
            endf_dict, mt, energies_in, energies_out
        )

    elif has_mf4_mt(endf_dict, mt):
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

    raise IndexError(
        f'Required data to reconstruct energy spectrum '
        f'for MT={mt} not available.'
    )
