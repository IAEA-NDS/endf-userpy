import numpy as np
from ..mfsec_interpretation import mf3_interpretation as mf3_interp
from ..mfsec_interpretation import mf12_interpretation as mf12_interp
from ..mfsec_interpretation import mf13_interpretation as mf13_interp
from ..mfsec_interpretation import mf14_interpretation as mf14_interp
from ..primitives.physical_constants import get_zap_for_particle
from ..primitives.properties import (
    has_mf6_mt,
    has_mf12_mt,
    has_mf13_mt,
    has_mf14_mt,
)


def get_outgoing_energies(endf_dict, mt, zap):
    gamma_zap = get_zap_for_particle('g')
    if zap != gamma_zap:
        raise NotImplementedError(
            'discrete distribution only implemented for gammas'
        )

    if has_mf12_mt(endf_dict, mt):
        return mf12_interp.get_photon_energies(endf_dict, mt)

    elif has_mf13_mt(endf_dict, mt):
        return mf13_interp.get_photon_energies(endf_dict, mt)

    raise IndexError(
        f'Neither MF12/MT{mt} nor MF13/MT{mt} available for '
        'retrieving discrete outgoing energies'
    )


def compute_angdist_values(
    endf_dict, mt, zap, energies_in, discrete_energies_out, angle_cosines_out, to_lab=True
):
    gamma_zap = get_zap_for_particle('g')
    if zap != gamma_zap:
        raise NotImplementedError(
            'discrete distribution interpretation only implemented for gammas'
        )

    if has_mf14_mt(endf_dict, mt):
        return mf14_interp.compute_angdist_values(
            endf_dict, mt, energies_in, discrete_energies_out, angle_cosines_out
        )

    elif has_mf6_mt(endf_dict, mt):
        raise NotImplementedError(
            'Reconstruction of angular distribution using '
            'discrete outgoing energy spectra in MF6 not implemented.'
        )

    raise IndexError(
        f'Required data to reconstruct angular distribution '
        'from discrete outgoing energy spectrum for MT={mt} not available.')


def compute_energydist_values(
    endf_dict, mt, zap, energies_in, discrete_energies_out, to_lab=True
):
    gamma_zap = get_zap_for_particle('g')
    if zap != gamma_zap:
        raise NotImplementedError(
            'discrete distribution interpretation only implemented for gammas'
        )

    if has_mf12_mt(endf_dict, mt):
        yields = mf12_interp.compute_photon_yields(
            endf_dict, mt, energies_in, discrete_energies_out
        )
        yields_sum = np.sum(yields, axis=1)
        energy_dist_values = yields / yields_sum.reshape(-1, 1) 
        return energy_dist_values

    elif has_mf13_mt(endf_dict, mt):
        prodxs = mf13_interp.compute_photon_production_xs(
            endf_dict, mt, energies_in, discrete_energies_out
        )
        xs = mf3_interp.compute_cross_section(
            endf_dict, mt, energies_in
        )
        yields = prodxs / xs.reshape(-1, 1)
        yields_sum = np.sum(yields, axis=1)
        energy_dist_values = yields / yields_sum.reshape(1, -1)
        return energy_dist_values

    raise IndexError(
        f'Required data to reconstruct discrete emission '
        'energy distribution for MT={mt} not available.'
    )
