import numpy as np
from ..primitives import array_ns
from ..primitives.helpers import (
    dict2array,
    find_indices_with_tol,
)
from ..primitives.interpolation import evaluate_interp_legendre_polynomials
from .mf4_interpretation import _convert_legendre_to_numpy_array


def get_photon_energies(endf_dict, mt):
    mtsec = endf_dict[14][mt]
    if mtsec['LI'] == 1:
        # everything isotropic, photon energy not provided
        return None
    return dict2array(mtsec['EG'])


def compute_angdist_from_isotropic(
    endf_dict, mt, energies_in, photon_energies, angle_cosines, xp=None,
):
    """Isotropic (LI=1) branch: `f(E_in, E_gamma, mu) = 0.5`.

    Returns shape `(len(energies_in), len(photon_energies),
    len(angle_cosines))` -- the same axis order every other
    `compute_angdist_from_*` helper in this module and every caller
    of `compute_angdist_values` expects. The local variable names
    are chosen to match the axis they describe (issue #80: the
    prior names `num_angcos_out` / `num_photens_out` had the axis
    labels swapped relative to the parameters and reinforced the
    call-site bug fixed below).
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    n_ein = len(energies_in)
    n_photen = len(photon_energies)
    n_mu = len(angle_cosines)
    return xp.full((n_ein, n_photen, n_mu), 0.5, dtype=xp.float64)


def compute_angdist_from_legendre(
    endf_dict, mt, energies_in, photon_energies, angle_cosines, xp=None,
):
    """Angular distribution at each of the user-requested
    `photon_energies` for a MF14 section that stores per-line
    Legendre coefficients (LI=0, LTT=1).

    ENDF-6 MF14 declares a total of NE photon lines, of which the
    first NI are stored as pure-isotropic (only the EG value is
    written; no Legendre data) and the remaining NE-NI carry
    per-line Legendre coefficients along the incident energy axis.
    The header dicts `mtsec['E_interpol']`, `mtsec['E']`, and
    `mtsec['a']` are keyed by the OVERALL EG position (1-indexed).

    For each user-requested photon `photon_energies[k]` we match it
    to a position in the MF14 EG list (`eg_idx`) and either take
    the isotropic 0.5 (if `eg_idx < NI`) or evaluate the per-line
    Legendre expansion at that `eg_idx`. Photons the file doesn't
    declare contribute zero (idcs == -1 case).

    The mapping between "position in the user's `photon_energies`
    list" (used to index axis 1 of the returned array) and
    "position in the MF14 EG list" (used to fetch the per-line
    Legendre data) has to be kept explicit, otherwise the
    per-line-Legendre index leaks into the assignment target and
    the assignment either out-of-bounds (issue #80) or writes to
    the wrong photon slot silently.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mtsec = endf_dict[14][mt]
    ni = mtsec['NI']
    eg = dict2array(mtsec['EG'])

    idcs = find_indices_with_tol(
        eg, photon_energies, atol=1e-4, rtol=4e-5
    )

    # Two distinct index arrays with different roles:
    #  - `*_user_positions`  = positions in `photon_energies`, used
    #                          to index axis 1 of `full_angdists`;
    #  - `aniso_eg_indices`  = positions in `eg`, used to look up
    #                          per-line Legendre data in
    #                          `mtsec['E_interpol']` / `['E']` /
    #                          `['a']`.
    # Photons that don't appear in the file at all (idcs == -1)
    # stay at the zero fill.
    iso_mask = (idcs >= 0) & (idcs < ni)
    aniso_mask = idcs >= ni

    iso_user_positions = np.where(iso_mask)[0]
    aniso_user_positions = np.where(aniso_mask)[0]
    aniso_eg_indices = idcs[aniso_mask]

    # Positional-arg order for the isotropic helper:
    #   (energies_in, photon_energies, angle_cosines)
    # returning shape `(n_ein, len(photon_energies), len(angle_cosines))`.
    # The previous call passed
    #   (energies_in, angle_cosines, photon_energies[isotropic_idcs])
    # -- axis-swapped -- so the returned isotropic block came out
    # as `(n_ein, n_mu, n_iso_photons)` and the assignment
    # `full_angdists[:, isotropic_idcs, :] = ...` (which expects
    # `(n_ein, n_iso_photons, n_mu)`) raised a shape mismatch
    # (issue #80).
    isotropic_angdists = compute_angdist_from_isotropic(
        endf_dict, mt, energies_in,
        photon_energies[iso_user_positions], angle_cosines,
        xp=xp,
    )

    einc_interp_tables = mtsec['E_interpol']
    res_list = []
    for eg_idx in aniso_eg_indices:
        interp_table = einc_interp_tables[eg_idx + 1]
        nbt_arr = np.array(interp_table['NBT'], dtype=int)
        int_arr = np.array(interp_table['INT'], dtype=int)
        einc_mesh = dict2array(mtsec['E'][eg_idx + 1], dtype=float)
        coeffs_arr = _convert_legendre_to_numpy_array(
            mtsec['a'][eg_idx + 1], xp=xp,
        )
        # Per-photon-line Ein mesh: pad with zeros outside the
        # tabulated range rather than raising. In files where the
        # user's `energies_in` spans a wider range than a specific
        # line's tabulation (JENDL-5 C-12 gamma dxs/dmu on a
        # 100 keV .. 14 MeV grid vs. a line only tabulated above
        # ~10 MeV -- issue #81), the natural physical meaning is
        # "this line contributes nothing at those Ein" rather
        # than an aborting error.
        f = evaluate_interp_legendre_polynomials(
            energies_in, angle_cosines,
            einc_mesh, coeffs_arr, int_arr, nbt_arr,
            outside_value=0.0, xp=xp,
        )
        res_list.append(f)

    n_ein = len(energies_in)
    n_photen = len(photon_energies)
    n_mu = len(angle_cosines)

    if xp.name == 'jax':
        full_angdists = xp.zeros((n_ein, n_photen, n_mu), dtype=xp.float64)
        if iso_user_positions.size > 0:
            full_angdists = full_angdists.at[:, iso_user_positions, :].set(
                isotropic_angdists
            )
        if len(res_list) > 0:
            anisotropic_angdists = xp.stack(res_list, axis=1)
            full_angdists = full_angdists.at[:, aniso_user_positions, :].set(
                anisotropic_angdists
            )
        return full_angdists

    full_angdists = xp.zeros((n_ein, n_photen, n_mu), dtype=xp.float64)
    full_angdists[:, iso_user_positions, :] = isotropic_angdists
    if len(res_list) > 0:
        # per-line f has shape (n_ein, n_mu); stack -> (n_ein, n_aniso, n_mu)
        anisotropic_angdists = xp.stack(res_list, axis=1)
        full_angdists[:, aniso_user_positions, :] = anisotropic_angdists
    return full_angdists


def compute_angdist_from_tabulated(
    endf_dict, mt, energies_in, photon_energies, angle_cosines, xp=None,
):
    raise NotImplementedError(
        'Tabulated angular distribution of photons not implemented.'
    )


def compute_angdist_values(
    endf_dict, mt, energies_in, photon_energies, angle_cosines, xp=None,
):
    """MF14 photon angular distribution f(mu | E_in, E_gamma).

    ``xp=None`` (default) is numpy and bit-identical to the pre-port
    behaviour. Passing an xp adapter threads tracers through the
    Legendre-branch reconstruction so ``jax.grad`` reaches file-side
    MF14 leaves.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mtsec = endf_dict[14][mt]
    if mtsec['LI'] == 1:
        return compute_angdist_from_isotropic(
            endf_dict, mt, energies_in, photon_energies, angle_cosines,
            xp=xp,
        )
    elif mtsec['LI'] == 0 and mtsec['LTT'] == 1:
        return compute_angdist_from_legendre(
            endf_dict, mt, energies_in, photon_energies, angle_cosines,
            xp=xp,
        )
    elif mtsec['LI'] == 0 and mtsec['LTT'] == 2:
        return compute_angdist_from_tabulated(
                endf_dict, mt, energies_in, photon_energies, angle_cosines,
                xp=xp,
        )

    li_val = mtsec['LI']
    ltt_val = mtsec['LTT']
    raise ValueError(
        f'Unknown LI/LTT combination (LI={li_val}, LTT={ltt_val}) in MF14/MT{mt}'
    )
