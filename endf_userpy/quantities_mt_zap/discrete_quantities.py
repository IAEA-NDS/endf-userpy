import numpy as np
from ..primitives import array_ns
from ..primitives.physical_constants import get_zap_for_particle
from ..mfsec_interpretation import mf3_interpretation as mf3_interp
from ..mfsec_interpretation import mf12_interpretation as mf12_interp
from ..mfsec_interpretation import mf13_interpretation as mf13_interp
from ..primitives.properties import (
    has_mf12_mt,
    has_mf13_mt,
)


def compute_yields(endf_dict, mt, zap, energies_in, xp=None):
    if xp is None:
        xp = array_ns.get_backend('numpy')
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
            endf_dict, mt, energies_in, photon_energies, xp=xp,
        )

    elif has_mf13_mt(endf_dict, mt):
        photon_energies = mf13_interp.get_photon_energies(endf_dict, mt)
        # exclude continuum
        photon_energies = photon_energies[photon_energies != 0]
        prodxs = mf13_interp.compute_photon_production_xs(
            endf_dict, mt, energies_in, photon_energies, xp=xp,
        )
        xs = mf3_interp.compute_cross_section(
            endf_dict, mt, energies_in
        )
        yields = prodxs / xp.asarray(xs.reshape(-1, 1))

    else:
        raise IndexError(
            'Required data to determine photon yields not available.'
            f'At least one of MF12 or MF13 must exist for MT={mt}.'
        )

    yields_sum = xp.sum(yields, axis=1)
    return yields_sum


def compute_total_gamma_yields(endf_dict, mt, energies_in, xp=None):
    """Sum of photon yields for all lines declared for (MT, gamma).

    Unlike ``compute_yields`` above, this sum INCLUDES any continuum
    placeholder line (Eg=0 in MF12 LO=1, or the continuum entry in
    MF13). The continuum placeholder carries most of the yield above
    a few tens of keV in typical capture files (e.g. Al-27 MT 102),
    so excluding it -- as ``compute_yields`` does, correctly, for its
    discrete-line angular-distribution purpose -- would drive the
    gamma production cross section to nearly zero for those files.
    Used from ``quantities_mt_zap.quantities.compute_yields`` when
    dispatching gamma production to MF12/MF13.

    For MF12 LO=2 (transition-probability representation) there is no
    continuum placeholder and this equals ``compute_yields`` above.

    ``xp=None`` (default) is numpy. Passing an xp adapter threads
    tracers through the MF12 / MF13 photon-yield reconstructions so
    ``jax.grad`` reaches file-side gamma yield leaves.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    if has_mf12_mt(endf_dict, mt):
        photon_energies = mf12_interp.get_photon_energies(endf_dict, mt)
        yields = mf12_interp.compute_photon_yields(
            endf_dict, mt, energies_in, photon_energies, xp=xp,
        )
        return xp.sum(yields, axis=1)

    if has_mf13_mt(endf_dict, mt):
        prodxs = mf13_interp.compute_total_photon_production_xs(
            endf_dict, mt, energies_in, xp=xp,
        )
        xs = mf3_interp.compute_cross_section(
            endf_dict, mt, energies_in
        )   # numpy
        # Where MF3 xs is zero the yield is undefined; return 0 so
        # the caller multiplication (yield * MF3 xs) stays 0 rather
        # than nan. Use xp.where with a safe denominator for JAX so
        # the trace does not divide-by-zero.
        xs_xp = xp.asarray(xs)
        safe_xs = xp.where(xs_xp > 0, xs_xp, 1.0)
        yields = xp.where(xs_xp > 0, prodxs / safe_xs, xp.asarray(0.0))
        return yields

    raise IndexError(
        f'Neither MF12/MT{mt} nor MF13/MT{mt} available for '
        f'total photon-yield summation.'
    )
