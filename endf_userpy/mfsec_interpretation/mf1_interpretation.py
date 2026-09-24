import numpy as np
from ..primitives import array_ns
from ..primitives.helpers import dict2array
from ..primitives.interpolation import endf_interp1d


def _compute_yields_from_polynomial(coefs, energies_in, xp=None):
    """Evaluate the ENDF-6 MF1 polynomial nubar representation
    ``nu(E) = sum_k C_k * E**k`` at each of ``energies_in``.

    ``coefs`` has shape ``(NC,)`` and ``energies_in`` shape
    ``(n_ein,)``. Uses Horner's rule so a JAX tracer in ``coefs``
    (nubar polynomial coefficients) or ``energies_in`` (incident-
    energy grid) propagates to ``jax.grad``. Numpy default matches
    ``np.polynomial.polynomial.polyval``.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    if xp.name == 'numpy':
        return np.polynomial.polynomial.polyval(energies_in, coefs)
    # Horner's rule, ascending-degree order (matches polyval).
    coefs = xp.asarray(coefs)
    energies_in = xp.asarray(energies_in)
    result = xp.zeros_like(energies_in)
    for k in range(int(coefs.shape[0]) - 1, -1, -1):
        result = result * energies_in + coefs[k]
    return result


def compute_yields_from_mt452(endf_dict, energies_in, xp=None):
    """Compute average total number of neutrons per fission."""
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mtsec = endf_dict[1][452]
    lnu = mtsec['LNU']
    ein = np.asarray(energies_in).reshape(-1)
    if lnu == 1:
        coefs = dict2array(mtsec['C'], dtype=float)
        return _compute_yields_from_polynomial(coefs, ein, xp=xp)
    elif lnu == 2:
        ep = np.array(mtsec['Eint'])
        nup = mtsec['nu']
        int_arr = np.array(mtsec['INT'])
        nbt_arr = np.array(mtsec['NBT'])
        return endf_interp1d(
            ein, ep, nup, int_arr, nbt_arr, outside_value=0.0, xp=xp,
        )
    raise ValueError(f'Invalid value LNU={lnu}')


def compute_yields_from_mt455(endf_dict, energies_in, xp=None):
    """Compute average number of delayed neutrons per fission."""
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mtsec = endf_dict[1][455]
    lnu = mtsec['LNU']
    ein = np.asarray(energies_in).reshape(-1)
    if lnu == 1:
        coefs = dict2array(mtsec['nubar_d'], dtype=float)
        return _compute_yields_from_polynomial(coefs, ein, xp=xp)
    elif lnu == 2:
        ep = np.array(mtsec['Eint'])
        nup = mtsec['nubar_d']
        int_arr = np.array(mtsec['INT'])
        nbt_arr = np.array(mtsec['NBT'])
        return endf_interp1d(
            ein, ep, nup, int_arr, nbt_arr, outside_value=0.0, xp=xp,
        )
    raise ValueError(f'Invalid value LNU={lnu}')


def compute_yields_from_mt456(endf_dict, energies_in, xp=None):
    """Compute average number of prompt neutrons per fission."""
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mtsec = endf_dict[1][456]
    lnu = mtsec['LNU']
    ein = np.asarray(energies_in).reshape(-1)
    if lnu == 1:
        # nubar_p is a scalar under LNU=1; broadcast to Ein shape via xp
        nubar = mtsec['nubar_p']
        return xp.asarray(nubar) * xp.ones_like(xp.asarray(ein))
    elif lnu == 2:
        ep = np.array(mtsec['Eint'])
        nup = mtsec['nubar_p']
        int_arr = np.array(mtsec['INT'])
        nbt_arr = np.array(mtsec['NBT'])
        return endf_interp1d(
            ein, ep, nup, int_arr, nbt_arr, outside_value=0.0, xp=xp,
        )
    raise ValueError(f'Invalid value LNU={lnu}')


def compute_yields(endf_dict, mt, energies_in, xp=None):
    """Compute total/delayed/prompt average number of neutrons per
    fission.

    ``xp=None`` (default) is numpy and bit-identical to the pre-port
    behaviour. Passing an xp adapter threads tracers through the
    LNU=1 polynomial nubar (Horner) and the LNU=2 tabulated nubar
    (via ``endf_interp1d``), so ``jax.grad`` reaches file-side MF1
    ``C``/``nu``/``nubar_p``/``nubar_d`` leaves for fission yield
    fitting.
    """
    if mt == 452:
        func = compute_yields_from_mt452
    elif mt == 455:
        func = compute_yields_from_mt455
    elif mt == 456:
        func = compute_yields_from_mt456
    else:
        raise ValueError(f'Unsupported number MT={mt}')
    return func(endf_dict, energies_in, xp=xp)
