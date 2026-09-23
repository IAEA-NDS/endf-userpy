"""Composition-layer double-differential distribution dispatcher.

Chooses which MF section to source the ``(E_in, E_out, mu)``
distribution from: MF6 double-differential when present, else the
MF4 (angular) x MF5 (energy) product for neutron-only files.

Backend-agnostic since issue #169: pass
``xp=array_ns.get_backend(name)`` to route the whole reconstruction
through that backend. With ``xp=jax``, tracers stored at file-side
leaves flow all the way through the MF6 kernels (LAW=1 / 2 / 6 / 7)
or the MF4+MF5 product into the returned distribution.
"""
from ..mfsec_interpretation import mf4_interpretation as mf4_interp
from ..mfsec_interpretation import mf5_interpretation as mf5_interp
from ..mfsec_interpretation import mf6_interpretation as mf6_interp
from ..primitives import array_ns
from ..primitives.physical_constants import get_zap_for_particle
from ..primitives.properties import (
    has_mf4_mt, has_mf5_mt, has_mf6_mt
)


def compute_dist2d_values_from_mf4_mf5(
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out,
    to_lab=True, xp=None,
):
    if to_lab is not True:
        raise ValueError(
            f"Double-differential distribution for MT={mt}, ZAP={zap} reconstruction "
            "from MF4 and MF5 only possible with `to_lab=True` argument."
        )
    if zap != get_zap_for_particle('n'):
        raise ValueError(
            f"Value of ZAP={zap} does not correspond to a neutron as ejectile. "
            "Reconstruction from MF4 and MF5 is only possible for neutrons."
        )
    if xp is None:
        xp = array_ns.get_backend('numpy')
    energy_spect = mf5_interp.compute_spectrum(
        endf_dict, mt, energies_in, energies_out, xp=xp,
    )
    angular_dist = mf4_interp.compute_angdist_values(
        endf_dict, mt, energies_in, angle_cosines_out, to_lab, xp=xp,
    )
    energy_spect = energy_spect[:, :, None]
    angular_dist = angular_dist[:, None, :]
    dist2d_values = energy_spect * angular_dist
    return dist2d_values


def compute_dist2d_values_from_mf6(
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out,
    to_lab=True, xp=None,
):
    return mf6_interp.compute_dist2d_values(
        endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out,
        to_lab, xp=xp,
    )


def compute_dist2d_values(
    endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out,
    to_lab=True, xp=None,
):
    """Composition-layer double-differential distribution.

    ``xp=None`` (default) is numpy. Passing a JAX adapter threads
    tracers through the MF6 (LAW=1/2/6/7) or MF4-x-MF5 branches
    end-to-end.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    n_zap = get_zap_for_particle('n')
    zeros_shape = (len(energies_in), len(energies_out), len(angle_cosines_out))
    if has_mf6_mt(endf_dict, mt):
        if mt == 18 and endf_dict[6][mt]['JP'] > 0:
            # Fission neutrons via the MT18 fallback: MF4+MF5.
            if zap != n_zap:
                return xp.zeros(zeros_shape, dtype=xp.float64)
            func = compute_dist2d_values_from_mf4_mf5
        else:
            # MF6 branch: return zeros if MF6 does not declare this zap
            # (typical for gamma on inelastic MTs 51..90 where the
            # neutron is in MF6 and the gamma's distribution is in
            # MF12/MF14, not yet wired here -- tracked in issue #36).
            from ..mfsec_interpretation import mf6_interpretation_helpers as mf6_help
            if not mf6_help.contains_zap(endf_dict, mt, zap):
                return xp.zeros(zeros_shape, dtype=xp.float64)
            func = compute_dist2d_values_from_mf6
    elif has_mf4_mt(endf_dict, mt) and has_mf5_mt(endf_dict, mt):
        # MF4+MF5 is a neutron-only representation by convention. For
        # any other ejectile (typical: gamma) fall through to zeros
        # rather than trying to reconstruct with a wrong kinematic
        # ejectile mass.
        if zap != n_zap:
            return xp.zeros(zeros_shape, dtype=xp.float64)
        func = compute_dist2d_values_from_mf4_mf5
    else:
        # No representable double-differential distribution for this
        # (MT, ZAP): return zeros so cumulative sums stay well-defined.
        return xp.zeros(zeros_shape, dtype=xp.float64)

    return func(
        endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out,
        to_lab, xp=xp,
    )
