import numpy as np
from ..primitives import array_ns
from ..primitives.helpers import dict2array
from ..primitives.interpolation import (
    interp_tab1,
    interp_tab2,
)


def _compute_prob(contrib_sec, energies_in, xp=None):
    if xp is None:
        xp = array_ns.get_backend('numpy')
    # Preserve JAX tracers on the query axis (roadmap #198 Phase 3):
    # ``interp_tab1`` routes through ``endf_interp1d``'s traced-x
    # path when xp=jax.
    ein = xp.asarray(energies_in).reshape(-1)
    return interp_tab1(
        ein, contrib_sec['rtfm_tab1'], 'Eint', 'p',
        outside_value=0.0, xp=xp,
    ).reshape(-1, 1)


def compute_tabulated_spectrum(
    contrib_sec, energies_in, energies_out, xp=None,
):
    if xp is None:
        xp = array_ns.get_backend('numpy')
    ein = energies_in
    eout = energies_out
    int_arr = np.array(contrib_sec['INT'])
    nbt_arr = np.array(contrib_sec['NBT'])

    ein_mesh = dict2array(contrib_sec['E'])
    tab1_records = list(contrib_sec['rtfm1_tab'].values())
    f = interp_tab2(
        ein, eout, ein_mesh, int_arr, nbt_arr, tab1_records,
        'Egamma', 'g', outside_value=0.0, xp=xp,
    )
    return f


def compute_spectrum_contribution(
    contrib_sec, energies_in, energies_out, xp=None,
):
    ein = energies_in
    eout = energies_out
    lf = contrib_sec['LF']
    if lf == 1:
        return compute_tabulated_spectrum(contrib_sec, ein, eout, xp=xp)
    raise ValueError(f'Spectrum computation for LF={lf} not implemented.')


def get_photon_energies(endf_dict, mt):
    contribs = list(endf_dict[15][mt]['subsection'].values())
    photon_energies = np.unique(sum(
        [t['Egamma'] for c in contribs for t in c['rtfm1_tab'].values()],
        start=[]
    ))
    return photon_energies


def compute_spectrum(endf_dict, mt, energies_in, energies_out, xp=None):
    """MF15 continuous photon spectrum p(E' | E_in) summed over
    subsections.

    ``xp=None`` (default) is numpy and bit-identical to the pre-port
    behaviour. Passing an xp adapter threads tracers through the
    subsection probability weights and the TAB2 unit-base interp
    so ``jax.grad`` reaches file-side MF15 leaves.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    ein = energies_in
    eout = energies_out
    contributions = list(endf_dict[15][mt]['subsection'].values())
    res = None
    for contrib in contributions:
        prob = _compute_prob(contrib, ein, xp=xp)
        cur = prob * compute_spectrum_contribution(contrib, ein, eout, xp=xp)
        res = cur if res is None else res + cur
    if res is None:
        # No contributions: return a matching-shape zero. Use
        # ``xp.asarray`` so a tracer ``ein`` / ``eout`` doesn't get
        # materialised (though this path is rare; a well-formed
        # MF15 section always has at least one subsection).
        n_ein = int(xp.asarray(ein).shape[0])
        n_eout = int(xp.asarray(eout).shape[0])
        return xp.zeros((n_ein, n_eout), dtype=xp.float64)
    return res
