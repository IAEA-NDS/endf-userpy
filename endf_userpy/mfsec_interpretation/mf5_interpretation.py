import numpy as np
from ..primitives.helpers import (
    dict2array,
    erf,
)
from ..primitives.interpolation import (
    interp_tab1,
    interp_tab2,
)
from ..primitives.np_compat import trapezoid


def _compute_prob(contrib_sec, energies_in):
    ein = energies_in.reshape(-1)
    return interp_tab1(
        ein, contrib_sec['p_table'], 'E', 'p', outside_value=0.0
    ).reshape(-1, 1)


def _compute_theta(contrib_sec, energies_in):
    ein = energies_in.reshape(-1)
    return interp_tab1(
        ein, contrib_sec['theta_table'], 'E', 'theta', outside_value=0.0
    ).reshape(-1, 1)  


def get_incident_energies_of_contribution(contrib_sec):
    en_mesh = np.array(contrib_sec['p_table']['E'], dtype=float)
    return en_mesh


def get_incident_energy_range(endf_dict, mt):
    mtsec = endf_dict[5][mt]
    contributions = list(mtsec['contribution'].values())
    en_min = np.inf 
    en_max = -np.inf
    for contrib in contributions:
        ens = get_incident_energies_of_contribution(contrib)
        en_min = min(np.min(ens), en_min)
        en_max = max(np.max(ens), en_max)
    return (en_min, en_max)


def get_emission_energies(endf_dict, mt):
    """Sorted union of tabulated outgoing-energy meshes across every
    LF=1 (tabulated-spectrum) contribution of MF5/MT.

    Contributions with an analytic LF (5 = general evaporation,
    7 = simple Maxwellian, 9 = evaporation, 11 = energy-dependent
    Watt, 12 = Madland-Nix) do not tabulate a discrete outgoing-
    energy mesh -- the spectrum is a closed-form function of Ein
    and Eout -- so those contributions add nothing to the union.
    Users needing a fine Eout grid on analytic-only files should
    pick one via the kinematic upper bound
    (``E_in - contribution['U']``) themselves.

    Returns an empty float ndarray if the file has no MF5/MT or if
    every contribution is analytic.
    """
    if 5 not in endf_dict or mt not in endf_dict[5]:
        return np.array([], dtype=float)
    eouts = []
    for contrib in endf_dict[5][mt]['contribution'].values():
        if contrib.get('LF') != 1:
            continue
        for tab in contrib.get('spectrum', {}).values():
            eouts.extend(float(x) for x in tab.get('Eout', []))
    if not eouts:
        return np.array([], dtype=float)
    return np.unique(np.asarray(eouts, dtype=float))


def compute_tabulated_spectrum(
    contrib_sec, energies_in, energies_out
):
    ein = energies_in
    eout = energies_out
    int_arr = np.array(contrib_sec['E_interp']['INT'])
    nbt_arr = np.array(contrib_sec['E_interp']['NBT'])
    ein_mesh = dict2array(contrib_sec['E'])
    tab1_records = list(contrib_sec['spectrum'].values())
    f = interp_tab2(
        ein, eout, ein_mesh, int_arr, nbt_arr, tab1_records, 'Eout', 'g', outside_value=0.0
    )
    return f


def compute_general_evaporation_spectrum(
    contrib_sec, energies_in, energies_out
):
    """MF5 LF=5 general evaporation spectrum (issue #41).

    Per ENDF-6, the spectrum is

        f(E, E') = g(E'/theta(E)) / (theta(E) * G(E))

    supported on ``0 <= E' <= E - U``, where ``g(x)`` is tabulated
    in the section's ``g_table`` and

        G(E) = integral from 0 to (E - U)/theta(E) of g(x) dx

    provides the normalisation. The pre-fix code returned just
    ``E'/theta`` (the reduced variable), never touching ``g_table``.

    The normalisation integral is computed by the numpy trapezoid
    integrator on the ``g_table`` mesh clipped to ``[0, x_max]``.
    This is exact for the common LIN-LIN (INT=2) interpolation;
    panel-exact integration for the four log/histogram INT values
    is a follow-up.
    """
    ein_arr = np.asarray(energies_in, dtype=float).reshape(-1)
    eout_arr = np.asarray(energies_out, dtype=float).reshape(-1)
    ein = ein_arr.reshape(-1, 1)
    theta = _compute_theta(contrib_sec, ein)  # (n_ein, 1)
    U = contrib_sec['U']

    g_tab = contrib_sec['g_table']
    x_mesh = np.asarray(g_tab['x'], dtype=float)
    g_mesh = np.asarray(g_tab['g'], dtype=float)

    n_ein = ein_arr.size
    n_eout = eout_arr.size
    f = np.zeros((n_ein, n_eout), dtype=float)

    for i in range(n_ein):
        E_minus_U = ein_arr[i] - U
        if E_minus_U <= 0:
            continue  # no allowed E'; leave row zero
        th = float(theta[i, 0])
        if th <= 0:
            continue
        x_max = E_minus_U / th

        # Interpolate g at x = E'/theta for the allowed E' (<= E - U).
        allowed = eout_arr <= E_minus_U
        if not allowed.any():
            continue
        x_query = eout_arr[allowed] / th
        g_query = interp_tab1(
            x_query, g_tab, 'x', 'g', outside_value=0.0,
        )

        # Normalisation: G = integral_{0}^{x_max} g(x) dx on the
        # tab1 mesh clipped to [0, x_max], with a final point at
        # x_max evaluated via interp_tab1 so the trapezoid captures
        # the right-edge sliver exactly.
        m_used = (x_mesh >= 0.0) & (x_mesh <= x_max)
        if m_used.any():
            x_int = x_mesh[m_used].astype(float)
            g_int = g_mesh[m_used].astype(float)
        else:
            x_int = np.array([0.0])
            g_int = np.array([
                float(interp_tab1(
                    np.array([0.0]), g_tab, 'x', 'g', outside_value=0.0,
                )[0])
            ])
        if x_int[0] > 0.0:
            g_at_zero = float(interp_tab1(
                np.array([0.0]), g_tab, 'x', 'g', outside_value=0.0,
            )[0])
            x_int = np.concatenate(([0.0], x_int))
            g_int = np.concatenate(([g_at_zero], g_int))
        if x_int[-1] < x_max:
            g_at_xmax = float(interp_tab1(
                np.array([x_max]), g_tab, 'x', 'g', outside_value=0.0,
            )[0])
            x_int = np.concatenate((x_int, [x_max]))
            g_int = np.concatenate((g_int, [g_at_xmax]))
        G = float(trapezoid(g_int, x_int))
        if G <= 0:
            continue

        f[i, allowed] = g_query / (th * G)
    return f


def compute_simple_maxwellian_fission_spectrum(
    contrib_sec, energies_in, energies_out
):
    """MF5 LF=7 simple Maxwellian fission spectrum (issue #41).

    Per ENDF-6, supported on ``0 <= E' <= E - U`` with

        f(E, E') = sqrt(E') * exp(-E'/theta) / I(E)
        I(E) = theta^(3/2) * [ sqrt(pi)/2 * erf(z) - z * exp(-z^2) ]
        z    = sqrt((E - U) / theta)

    Pre-fix bugs:
    - Support mask was ``eout <= U`` instead of ``eout <= E - U``.
      With U=0 the mask was always empty and every spectrum was
      identically zero.
    - Normalisation was ``I = theta^(3/2)`` (missing the bracketed
      erf/exp factor) and used ``exp(-z)`` instead of ``exp(-z^2)``.
    """
    ein = np.asarray(energies_in, dtype=float).reshape(-1, 1)
    eout = np.asarray(energies_out, dtype=float).reshape(1, -1)
    theta = _compute_theta(contrib_sec, ein)  # (n_ein, 1)
    U = contrib_sec['U']

    E_minus_U = ein - U                      # (n_ein, 1)
    valid = E_minus_U > 0.0                  # (n_ein, 1)
    allowed = (eout >= 0.0) & (eout <= E_minus_U)  # (n_ein, n_eout)

    # Normalisation constant per incident energy (n_ein, 1). Where
    # E - U <= 0 the spectrum is defined-zero; set I to 1 there
    # (unused; masked out below).
    with np.errstate(invalid='ignore'):
        z_sq = np.where(valid, E_minus_U / theta, 0.0)
        z = np.sqrt(z_sq)
        I = np.where(
            valid,
            (theta ** 1.5) * (
                0.5 * np.sqrt(np.pi) * erf(z) - z * np.exp(-z_sq)
            ),
            1.0,
        )
    valid_I = valid & (I > 0)

    # Unnormalised spectrum on the full (n_ein, n_eout) grid.
    with np.errstate(invalid='ignore'):
        raw = np.where(
            allowed,
            np.sqrt(np.maximum(eout, 0.0)) * np.exp(-eout / theta),
            0.0,
        )
    return np.where(valid_I & allowed, raw / I, 0.0)


def compute_evaporation_spectrum(
    contrib_sec, energies_in, energies_out
):
    """MF5 LF=9 evaporation spectrum (issue #41).

    Per ENDF-6, supported on ``0 <= E' <= E - U`` with

        f(E, E') = E' * exp(-E'/theta) / I(E)
        I(E) = theta^2 * [ 1 - exp(-(E-U)/theta) * (1 + (E-U)/theta) ]

    Pre-fix bugs:
    - Same wrong support mask ``eout <= U`` as LF=7.
    - Normalisation was ``I = theta^2 * (1 - (1 + (E-U)/theta))``
      which reduces to ``-theta * (E-U)`` -- negative for physical
      inputs, missing the ``exp(-(E-U)/theta)`` factor entirely.
    """
    ein = np.asarray(energies_in, dtype=float).reshape(-1, 1)
    eout = np.asarray(energies_out, dtype=float).reshape(1, -1)
    theta = _compute_theta(contrib_sec, ein)
    U = contrib_sec['U']

    E_minus_U = ein - U
    valid = E_minus_U > 0.0
    allowed = (eout >= 0.0) & (eout <= E_minus_U)

    with np.errstate(invalid='ignore'):
        y = np.where(valid, E_minus_U / theta, 0.0)
        I = np.where(
            valid,
            (theta ** 2) * (1.0 - np.exp(-y) * (1.0 + y)),
            1.0,
        )
    valid_I = valid & (I > 0)

    with np.errstate(invalid='ignore'):
        raw = np.where(
            allowed,
            np.maximum(eout, 0.0) * np.exp(-eout / theta),
            0.0,
        )
    return np.where(valid_I & allowed, raw / I, 0.0)


def compute_spectrum_contribution(contrib_sec, energies_in, energies_out):
    ein = energies_in
    eout = energies_out
    lf = contrib_sec['LF']
    func = None
    if lf == 1:
        func = compute_tabulated_spectrum
    elif lf == 5:
        func = compute_general_evaporation_spectrum
    elif lf == 7:
        func = compute_simple_maxwellian_fission_spectrum
    elif lf == 9:
        func = compute_evaporation_spectrum
    else:
        raise ValueError(f'Spectrum computation for LF={lf} not implemented.')

    return func(contrib_sec, ein, eout) 


def compute_spectrum(endf_dict, mt, energies_in, energies_out): 
    ein = energies_in
    eout = energies_out
    contributions = list(endf_dict[5][mt]['contribution'].values())
    res = 0.0
    for contrib in contributions:
        prob = _compute_prob(contrib, ein)
        res += prob * compute_spectrum_contribution(contrib, ein, eout)
    return res
