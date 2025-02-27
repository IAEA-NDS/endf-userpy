import numpy as np
from scipy.integrate import quad
from . import mf4_interpretation as mf4_interp
from .distribution2d import compute_dist2d_values
from .properties import (
    is_zap_consistent,
    has_mf4_mt,
    has_mf6_mt,
    get_QM,
    get_QI,
)


def compute_angdist_values(endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True):
    if not is_zap_consistent(endf_dict, mt, zap):
        raise ValueError(f'MT={mt} and ZAP={mt} are not consistent')

    if has_mf4_mt(endf_dict, mt):
        return mf4_interp.compute_angdist_values(
            endf_dict, mt, energies_in, angle_cosines_out, to_lab
        )

    elif has_mf6_mt(endf_dict, mt):
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
                dist2d_func = lambda x: compute_dist2d_values(
                    endf_dict, mt, zap, ens_inc[i:i+1], np.array([x], dtype=float), mus_out[j:j+1]
                )
                angdist[i, j] = quad(dist2d_func, 0.0, eout_max, epsrel=1e-4)[0]
        return angdist

    raise IndexError(
        f'Required data to reconstruct angular distribution '
        f'for MT={mt} not available.'
    )
