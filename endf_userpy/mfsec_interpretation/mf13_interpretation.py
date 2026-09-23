import numpy as np
from ..primitives import array_ns
from ..primitives.helpers import (
    find_indices_with_tol,
)
from ..primitives.interpolation import endf_interp1d


def get_photon_energies(endf_dict, mt):
    mtsec = endf_dict[13][mt]
    pe = [s['EG'] for s in mtsec['subsection'].values()]
    return np.array(pe, dtype=float)


def _compute_total_production_xs(endf_dict, mt, energies_in, xp=None):
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mtsec = endf_dict[13][mt]
    int_arr = np.array(mtsec['INT'])
    nbt_arr = np.array(mtsec['NBT'])
    einc_mesh = np.array(mtsec['E'])
    xs_mesh = mtsec['sigma_tot']
    if xp.name == 'numpy':
        xs_mesh = np.asarray(xs_mesh, dtype=float)
    return endf_interp1d(
        energies_in, einc_mesh, xs_mesh, int_arr, nbt_arr,
        outside_value=0.0, xp=xp,
    )


def compute_photon_production_xs(
    endf_dict, mt, energies_in, photon_energies, xp=None,
):
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mtsec = endf_dict[13][mt]
    eincs = energies_in
    subsecs = list(mtsec['subsection'].values())
    existing_photon_energies = np.array([sec['EG'] for sec in subsecs])
    idcs = find_indices_with_tol(
        existing_photon_energies, photon_energies, atol=1e-3, rtol=1e-5
    )
    if np.any(idcs == -1):
        raise ValueError(
            'Array `photon_energies` must only contain energies '
            'that are also present in the file.'
        )
    subsecs = [subsecs[idx] for idx in idcs]
    cols = [
        endf_interp1d(
            eincs, t['E'], t['sigma'], t['INT'], t['NBT'],
            outside_value=0.0, xp=xp,
        )
        for t in subsecs
    ]
    if len(cols) == 0:
        n_ein = np.asarray(eincs).size
        return xp.zeros((n_ein, 0), dtype=xp.float64)
    return xp.stack(cols, axis=1)


def compute_total_photon_production_xs(endf_dict, mt, energies_in, xp=None):
    """Total photon production cross section for `mt` as the sum of
    every subsection's partial. When the section stores an extra
    top-level total (`NK > 1` in ENDF-6), the two are compared as a
    consistency check.

    Per ENDF-6 the section-level totals header (`E`, `sigma_tot`,
    `INT`, `NBT`) is only written when `NK > 1`; for `NK == 1` the
    single subsection IS the total and no header is written. The
    previous guard was ``prod_xs.shape[0] > 1``, which is
    ``n_ein > 1``, not ``n_subs > 1`` -- so single-subsection
    sections with more than one requested Ein point (i.e. every
    real use) fell through to the consistency check that then
    raised ``KeyError: 'INT'`` on the missing totals header
    (issue #79; N-14 MT 32 and MT 105).

    ``xp=None`` (default) is numpy; passing a JAX adapter threads
    tracers through the per-subsection interp so gradients reach
    file-side MF13 ``sigma`` leaves.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    photon_energies = get_photon_energies(endf_dict, mt)
    prod_xs = compute_photon_production_xs(
        endf_dict, mt, energies_in, photon_energies, xp=xp,
    )
    total_prod_xs = xp.sum(prod_xs, axis=1)
    if endf_dict[13][mt]['NK'] > 1:
        check_tot_prod_xs = _compute_total_production_xs(
            endf_dict, mt, energies_in, xp=xp,
        )
        if not np.allclose(
            np.asarray(total_prod_xs), np.asarray(check_tot_prod_xs),
        ):
            raise ValueError(
                'Total photon production cross section '
                f'given in MF13/MT{mt} does not equal the '
                'sum of the partial production cross '
                'sections.'
            )
    return total_prod_xs
