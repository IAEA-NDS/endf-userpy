import numpy as np
from scipy.integrate import quad
from ..mfsec_interpretation import mf4_interpretation as mf4_interp
from ..mfsec_interpretation import mf6_interpretation as mf6_interp
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


USE_FORTRAN_INTEGRATION = True


def integrate_mf6_dist2d_over_eout(
    endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True
):
    angdist = np.zeros((len(energies_in), len(angle_cosines_out)), dtype=float)
    ens_inc = energies_in
    mus_out = angle_cosines_out
    q = max(get_QM(endf_dict, mt), get_QI(endf_dict, mt))
    for i in range(len(energies_in)):
        for j in range(len(angle_cosines_out)):
            eout_max = (energies_in[i] + q) * 1.1
            if eout_max <= 0:
                angdist[i, j] = 0.0
                continue
            cur_ens_inc = ens_inc[i:i+1]
            cur_mus_out = mus_out[j:j+1]
            dist2d_func = lambda x: compute_dist2d_values(
                endf_dict, mt, zap,
                cur_ens_inc, np.array([x], dtype=float), cur_mus_out,
                to_lab
            )
            angdist[i, j] = quad(dist2d_func, 0.0, eout_max, epsrel=1e-4)[0]
    return angdist


def _integrate_mf6_dist2d_over_mu_default(
    endf_dict, mt, zap, energies_in, energies_out, to_lab=True
):
        energydist = np.zeros((len(energies_in), len(energies_out)), dtype=float)
        ens_inc = energies_in
        ens_out = energies_out
        for i in range(len(ens_inc)):
            for j in range(len(ens_out)):
                cur_ens_inc = ens_inc[i:i+1]
                cur_ens_out =  ens_out[j:j+1]
                dist2d_func = lambda x: compute_dist2d_values(
                    endf_dict, mt, zap,
                    cur_ens_inc, cur_ens_out, np.array([x], dtype=float),
                    to_lab
                )
                energydist[i, j] = quad(dist2d_func, -1.0, 1.0, epsrel=1e-4)[0]
        return energydist


def integrate_mf6_dist2d_over_mu(
    endf_dict, mt, zap, energies_in, energies_out, to_lab=True
):
    mtsec = endf_dict[6][mt]
    subsec_nums = mf6_help.find_subsec_nums(endf_dict, mt, zap)
    if len(subsec_nums) == 1:
        law = mtsec['subsection'][subsec_nums[0]]['LAW']
        if law == 1 and USE_FORTRAN_INTEGRATION:
            print('(using Fortran routine')  # debug
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

    angle_cosines_out = conv_relat.compute_cos_phi_from_Ekin(
        energies_out, energies_in, m_i, m_t, m_e, m_r
    )
    jacvals = conv_relat.compute_dcos_phi_dEkin(
        energies_out, energies_in, m_i, m_t, m_e, m_r
    )
    return angle_cosines_out, jacvals


def convert_angdist_to_energydist(
    compute_angdist_func, endf_dict, mt, zap, energies_in, energies_out, to_lab
):
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
