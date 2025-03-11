import numpy as np
from ..primitives.physical_constants import get_zap_for_particle
from ..mfsec_interpretation import mf3_interpretation as mf3_interp
from ..mfsec_interpretation import mf12_interpretation as mf12_interp
from ..mfsec_interpretation import mf13_interpretation as mf13_interp
from ..primitives.properties import (
    has_mf6_mt,
    has_mf12_mt,
    has_mf13_mt,
    has_mf14_mt,
)
from .discrete_distribution1d import get_outgoing_energies 


def compute_yields(endf_dict, mt, zap, energies_in):
    gamma_zap = get_zap_for_particle('g')
    if zap != gamma_zap:
        raise NotImplementedError(
            'discrete distribution interpretation only implemented for gammas'
        )

    if has_mf12_mt(endf_dict, mt):
        photon_energies = mf12_interp.get_photon_energies(endf_dict, mt)
        # exclude continuum
        photon_energies = photon_energies[photon_energies != 0]
        yields = mf12_interp.compute_photon_yields(
            endf_dict, mt, energies_in, photon_energies
        )

    elif has_mf13_mt(endf_dict, mt):
        photon_energies = mf13_interp.get_photon_energies(endf_dict, mt)
        # exclude continuum
        photon_energies = photon_energies[photon_energies != 0]
        prodxs = mf13_interp.compute_photon_production_xs(
            endf_dict, mt, energies_in, discrete_energies_out
        )
        xs = mf3_interp.compute_cross_section(
            endf_dict, mt, energies_in
        )
        yields = prodxs / xs.reshape(-1, 1)

    else:
        raise IndexError(
            'Required data to determine photon yields not available.'
            f'At least one of MF12 or MF13 must exist for MT={mt}.'
        )

    yields_sum = np.sum(yields, axis=1)
    return yields_sum
