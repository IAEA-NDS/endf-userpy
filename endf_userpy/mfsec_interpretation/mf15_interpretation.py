import numpy as np
from ..primitives.helpers import dict2array
from ..primitives.interpolation import (
    interp_tab1,
    interp_tab2,
)


def _compute_prob(contrib_sec, energies_in):
    ein = energies_in.reshape(-1)
    return interp_tab1(
        ein, contrib_sec['rtfm_tab1'], 'Eint', 'p', outside_value=0.0
    ).reshape(-1, 1)


def compute_tabulated_spectrum(
    contrib_sec, energies_in, energies_out
):
    ein = energies_in
    eout = energies_out
    int_arr = np.array(contrib_sec['INT'])
    nbt_arr = np.array(contrib_sec['NBT'])

    ein_mesh = dict2array(contrib_sec['E'])
    tab1_records = list(contrib_sec['rtfm1_tab'].values())
    f = interp_tab2(
        ein, eout, ein_mesh, int_arr, nbt_arr, tab1_records, 'Egamma', 'g', outside_value=0.0
    )
    return f


def compute_spectrum_contribution(contrib_sec, energies_in, energies_out):
    ein = energies_in
    eout = energies_out
    lf = contrib_sec['LF']
    func = None
    if lf == 1:
        func = compute_tabulated_spectrum
    else:
        raise ValueError(f'Spectrum computation for LF={lf} not implemented.')

    return func(contrib_sec, ein, eout) 


def get_photon_energies(endf_dict, mt):
    contribs = list(endf_dict[15][mt]['subsection'].values())
    photon_energies = np.unique(sum(
        [t['Egamma'] for c in contribs for t in c['rtfm1_tab'].values()],
        start=[]
    ))
    return photon_energies


def compute_spectrum(endf_dict, mt, energies_in, energies_out): 
    ein = energies_in
    eout = energies_out
    contributions = list(endf_dict[15][mt]['subsection'].values())
    res = 0.0
    for contrib in contributions:
        prob = _compute_prob(contrib, ein)
        res += prob * compute_spectrum_contribution(contrib, ein, eout)
    return res
