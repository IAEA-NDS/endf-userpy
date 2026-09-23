"""ENDF-6 MF5 secondary-particle energy-spectrum reconstruction.

Backend-agnostic since issue #169: every reconstruction function
accepts an optional ``xp=None`` argument that routes the numerics
through the caller's backend adapter (numpy default; JAX via
``array_ns.get_backend('jax')``). ``xp=jax`` enables ``jax.grad``
through file-side leaves that fitters typically care about:

- ``contrib['p_table']['p']`` -- per-contribution probability
  (mixture weights).
- ``contrib['theta_table']['theta']`` -- Maxwellian / evaporation
  temperature (spectral shape).
- ``contrib['U']`` -- kinematic offset.
- ``contrib['g_table']['g']`` -- LF=5 general-evaporation shape.
- ``contrib['spectrum'][row]['g']`` -- LF=1 tabulated spectrum
  values.

The LF=5 branch keeps its per-incident-energy Python loop so the
normalisation integral ``G(E)`` runs against a mesh clipped to
``[0, x_max]`` -- a JAX-native rewrite that scans over ``ein``
would keep it JAX-jittable but is out of scope for this port.
When called with concrete ``ein`` values and ``xp=jax``, the loop
body still traces cleanly and grads flow through ``theta`` and
``g_table`` values.
"""
import numpy as np

from ..primitives import array_ns
from ..primitives.helpers import (
    dict2array,
    erf,
)
from ..primitives.interpolation import (
    interp_tab1,
    interp_tab2,
)


def _compute_prob(contrib_sec, energies_in, xp=None):
    ein = energies_in.reshape(-1)
    return interp_tab1(
        ein, contrib_sec['p_table'], 'E', 'p',
        outside_value=0.0, xp=xp,
    ).reshape(-1, 1)


def _compute_theta(contrib_sec, energies_in, xp=None):
    ein = energies_in.reshape(-1)
    return interp_tab1(
        ein, contrib_sec['theta_table'], 'E', 'theta',
        outside_value=0.0, xp=xp,
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
    contrib_sec, energies_in, energies_out, xp=None,
):
    ein = energies_in
    eout = energies_out
    int_arr = np.array(contrib_sec['E_interp']['INT'])
    nbt_arr = np.array(contrib_sec['E_interp']['NBT'])
    ein_mesh = dict2array(contrib_sec['E'], xp=xp)
    tab1_records = list(contrib_sec['spectrum'].values())
    f = interp_tab2(
        ein, eout, ein_mesh, int_arr, nbt_arr,
        tab1_records, 'Eout', 'g',
        outside_value=0.0, xp=xp,
    )
    return f


def compute_general_evaporation_spectrum(
    contrib_sec, energies_in, energies_out, xp=None,
):
    """MF5 LF=5 general evaporation spectrum (issue #41).

    Per ENDF-6, the spectrum is

        f(E, E') = g(E'/theta(E)) / (theta(E) * G(E))

    supported on ``0 <= E' <= E - U``, where ``g(x)`` is tabulated
    in the section's ``g_table`` and

        G(E) = integral from 0 to (E - U)/theta(E) of g(x) dx

    provides the normalisation. The normalisation integral is
    computed by the trapezoid integrator on the ``g_table`` mesh
    clipped to ``[0, x_max]``. Exact for the common LIN-LIN
    (INT=2) interpolation; panel-exact integration for the four
    log/histogram INT values is a follow-up.

    Backend-agnostic: ``xp=None`` runs on numpy. Under xp=jax the
    per-incident-energy loop body traces cleanly (each mask
    ``allowed`` and mesh slice ``m_used`` is a concrete numpy
    computation from ``ein_arr[i]``, then only the actual g /
    normalisation arithmetic runs xp-native). Grad wrt tracers in
    ``contrib_sec['theta_table']['theta']`` and
    ``contrib_sec['g_table']['g']`` reaches back through the
    reconstruction end-to-end.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    ein_arr = np.asarray(energies_in, dtype=float).reshape(-1)
    eout_arr = np.asarray(energies_out, dtype=float).reshape(-1)
    ein = ein_arr.reshape(-1, 1)
    theta = _compute_theta(contrib_sec, ein, xp=xp)  # (n_ein, 1)
    U = contrib_sec['U']

    g_tab = contrib_sec['g_table']
    x_mesh = np.asarray(g_tab['x'], dtype=float)
    # ``g_tab['g']`` is a plain Python list in endf_parserpy's
    # output; use xp.asarray directly so JAX tracer scalars stored
    # in the list survive (dict2array would fail here because it
    # expects a dict).
    g_mesh = xp.asarray(g_tab['g'], dtype=xp.float64)

    n_ein = ein_arr.size
    n_eout = eout_arr.size
    dtype_ = theta.dtype
    f = xp.zeros((n_ein, n_eout), dtype=dtype_)

    for i in range(n_ein):
        E_minus_U = ein_arr[i] - U
        if float(np.asarray(E_minus_U)) <= 0:
            continue  # no allowed E'; leave row zero
        th = theta[i, 0]
        # ``th`` may be a jax tracer; skip only on the concrete
        # boundary case ``th <= 0`` where the closed form is
        # undefined (files never carry non-positive theta).
        x_max = E_minus_U / th

        allowed = eout_arr <= float(np.asarray(E_minus_U))
        if not allowed.any():
            continue
        x_query = eout_arr[allowed] / th
        g_query = interp_tab1(
            x_query, g_tab, 'x', 'g',
            outside_value=0.0, xp=xp,
        )

        # Normalisation: G = integral_{0}^{x_max} g(x) dx on the
        # tab1 mesh clipped to [0, x_max], with left/right endpoint
        # slivers evaluated via interp_tab1 so the trapezoid rule
        # captures them exactly.
        x_max_c = float(np.asarray(x_max))
        m_used = (x_mesh >= 0.0) & (x_mesh <= x_max_c)
        if m_used.any():
            x_int = x_mesh[m_used].astype(float)
            g_int = g_mesh[m_used]
        else:
            x_int = np.array([0.0])
            g_int = interp_tab1(
                np.array([0.0]), g_tab, 'x', 'g',
                outside_value=0.0, xp=xp,
            )
        if x_int[0] > 0.0:
            g_at_zero = interp_tab1(
                np.array([0.0]), g_tab, 'x', 'g',
                outside_value=0.0, xp=xp,
            )
            x_int = np.concatenate(([0.0], x_int))
            g_int = xp.concatenate([g_at_zero, g_int])
        if x_int[-1] < x_max_c:
            g_at_xmax = interp_tab1(
                np.array([x_max_c]), g_tab, 'x', 'g',
                outside_value=0.0, xp=xp,
            )
            x_int = np.concatenate((x_int, [x_max_c]))
            g_int = xp.concatenate([g_int, g_at_xmax])
        # trapezoid integrand: use xp so JAX tracers in g_int and
        # theta survive. x_int is concrete numpy but that's fine
        # for the sampling grid.
        G = xp.trapezoid(g_int, xp.asarray(x_int))
        # G > 0 for physical tabulations; skip the concrete-zero
        # case to avoid division by zero on the numpy path.
        if float(np.asarray(G)) <= 0:
            continue

        row = g_query / (th * G)
        # Scatter into the concrete-index positions on the eout
        # axis. Under numpy this is plain boolean-indexing assign;
        # under jax we use .at[].set(...).
        allowed_idx = np.where(allowed)[0]
        if xp.name == 'jax':
            f = f.at[i, allowed_idx].set(row)
        else:
            f[i, allowed_idx] = row
    return f


def compute_simple_maxwellian_fission_spectrum(
    contrib_sec, energies_in, energies_out, xp=None,
):
    """MF5 LF=7 simple Maxwellian fission spectrum (issue #41).

    Per ENDF-6, supported on ``0 <= E' <= E - U`` with

        f(E, E') = sqrt(E') * exp(-E'/theta) / I(E)
        I(E) = theta^(3/2) * [ sqrt(pi)/2 * erf(z) - z * exp(-z^2) ]
        z    = sqrt((E - U) / theta)
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    ein = xp.asarray(energies_in, dtype=xp.float64).reshape(-1, 1)
    eout = xp.asarray(energies_out, dtype=xp.float64).reshape(1, -1)
    theta = _compute_theta(contrib_sec, ein, xp=xp)  # (n_ein, 1)
    U = contrib_sec['U']

    E_minus_U = ein - U                              # (n_ein, 1)
    valid = E_minus_U > 0.0                          # (n_ein, 1)
    allowed = (eout >= 0.0) & (eout <= E_minus_U)    # (n_ein, n_eout)

    # Normalisation constant per incident energy (n_ein, 1). Where
    # E - U <= 0 the spectrum is defined-zero; set I to 1 there
    # (unused; masked out below). Safe-value pattern for xp.where
    # so no division-by-zero warning fires on numpy.
    theta_safe = xp.where(valid, theta, 1.0)
    z_sq = xp.where(valid, E_minus_U / theta_safe, 0.0)
    z = xp.sqrt(z_sq)
    I = xp.where(
        valid,
        (theta_safe ** 1.5) * (
            0.5 * xp.sqrt(xp.pi) * erf(z, xp=xp) - z * xp.exp(-z_sq)
        ),
        1.0,
    )
    valid_I = valid & (I > 0)

    eout_safe = xp.where(eout >= 0.0, eout, 0.0)
    raw = xp.where(
        allowed,
        xp.sqrt(eout_safe) * xp.exp(-eout / theta_safe),
        0.0,
    )
    return xp.where(valid_I & allowed, raw / I, 0.0)


def compute_evaporation_spectrum(
    contrib_sec, energies_in, energies_out, xp=None,
):
    """MF5 LF=9 evaporation spectrum (issue #41).

    Per ENDF-6, supported on ``0 <= E' <= E - U`` with

        f(E, E') = E' * exp(-E'/theta) / I(E)
        I(E) = theta^2 * [ 1 - exp(-(E-U)/theta) * (1 + (E-U)/theta) ]
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    ein = xp.asarray(energies_in, dtype=xp.float64).reshape(-1, 1)
    eout = xp.asarray(energies_out, dtype=xp.float64).reshape(1, -1)
    theta = _compute_theta(contrib_sec, ein, xp=xp)
    U = contrib_sec['U']

    E_minus_U = ein - U
    valid = E_minus_U > 0.0
    allowed = (eout >= 0.0) & (eout <= E_minus_U)

    theta_safe = xp.where(valid, theta, 1.0)
    y = xp.where(valid, E_minus_U / theta_safe, 0.0)
    I = xp.where(
        valid,
        (theta_safe ** 2) * (1.0 - xp.exp(-y) * (1.0 + y)),
        1.0,
    )
    valid_I = valid & (I > 0)

    eout_safe = xp.where(eout >= 0.0, eout, 0.0)
    raw = xp.where(
        allowed,
        eout_safe * xp.exp(-eout / theta_safe),
        0.0,
    )
    return xp.where(valid_I & allowed, raw / I, 0.0)


def compute_spectrum_contribution(
    contrib_sec, energies_in, energies_out, xp=None,
):
    ein = energies_in
    eout = energies_out
    lf = contrib_sec['LF']
    if lf == 1:
        return compute_tabulated_spectrum(contrib_sec, ein, eout, xp=xp)
    if lf == 5:
        return compute_general_evaporation_spectrum(
            contrib_sec, ein, eout, xp=xp,
        )
    if lf == 7:
        return compute_simple_maxwellian_fission_spectrum(
            contrib_sec, ein, eout, xp=xp,
        )
    if lf == 9:
        return compute_evaporation_spectrum(contrib_sec, ein, eout, xp=xp)
    raise ValueError(f'Spectrum computation for LF={lf} not implemented.')


def compute_spectrum(endf_dict, mt, energies_in, energies_out, xp=None):
    ein = energies_in
    eout = energies_out
    contributions = list(endf_dict[5][mt]['contribution'].values())
    res = 0.0
    for contrib in contributions:
        prob = _compute_prob(contrib, ein, xp=xp)
        res = res + prob * compute_spectrum_contribution(
            contrib, ein, eout, xp=xp,
        )
    return res
