"""ENDF-6 MF4 angular-distribution reconstruction.

Backend-agnostic since issue #169: every reconstruction function
accepts an optional ``xp=None`` argument that routes the numerics
through the caller's backend adapter (numpy default; JAX via
``array_ns.get_backend('jax')``). With ``xp=jax``, JAX tracers
stored at file-side leaves -- Legendre coefficients under
``mf4sec['a']`` / ``mf4sec['al']``, tabulated probabilities under
``mf4sec['angtable'][row]['f'][mu_idx]`` -- propagate through
``dict2array`` and the primitives-layer helpers
(``evaluate_interp_legendre_polynomials``, ``interp_tab2``,
``convert_angcos_to_cmsys``, ``convert_angdist_to_labsys``,
``compute_r2``) into the returned distribution, so ``jax.grad``
reaches back to any tabulated angular parameter end-to-end.

Query-energy autodiff is a separate concern (query masking in
:func:`pad_outside_angdist_values` cannot pass tracers through the
outside-range branch). For the fully-inside case the decorator
short-circuits and lets tracers flow.
"""
import numpy as np

from ..primitives import array_ns
from ..primitives.interpolation import (
    evaluate_interp_legendre_polynomials,
    interp_tab2,
)
from ..primitives.conversion import (
    compute_r2,
    convert_angcos_to_cmsys,
    convert_angdist_to_labsys,
)
from ..primitives.helpers import (
    dict2array,
)
from ..primitives.properties import (
    get_AWR, get_AWI, get_AWP,
    get_QI,
)
from .mf4_interpretation_helpers import pad_outside_angdist_values


def get_incident_energies(endf_dict, mt):
    return list(endf_dict[4][mt]['E'].values())


def get_incident_energy_range(endf_dict, mt):
    ens = get_incident_energies(endf_dict, mt)
    return (np.min(ens), np.max(ens))


def has_isotropic_angdist_repr(endf_dict, mt):
    mf4sec = endf_dict[4]
    return mf4sec[mt]['LTT'] == 0 and mf4sec[mt]['LI'] == 1


def _convert_legendre_to_numpy_array(coeffs_dict, xp=None):
    """Convert the raw ENDF Legendre coefficient dict into the
    ``(n_energies, max_L+1)`` array the reconstruction consumes.

    The L=0 slot is set to 1.0 (ENDF omits it because it's fixed by
    the ``f(mu) = (1/2) sum_L (2L+1) a_L P_L(mu)`` normalisation),
    every row is padded to the max length across energies, and each
    column ``L`` is scaled by ``(L + 1/2)``.

    Backend-agnostic (issue #169): ``xp=None`` (default) is numpy;
    passing a JAX adapter routes each per-energy coefficient row
    through ``dict2array(..., xp=xp)`` so tracers stored at
    ``coeffs_dict[E_idx][L]`` propagate into the returned 2D
    array.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    num_energies = len(coeffs_dict)
    num_coeffs_per_energy = [len(v) + 1 for _, v in coeffs_dict.items()]
    max_num_coeffs = int(np.max(num_coeffs_per_energy))
    # Assemble each row through dict2array so JAX tracers at the
    # per-coefficient leaves survive; pad to max_num_coeffs with 0.
    rows = []
    for en_idx in range(num_energies):
        num_coeffs = num_coeffs_per_energy[en_idx]
        row_coeffs = dict2array(
            coeffs_dict[en_idx + 1], dtype=float, xp=xp,
        )
        # Prepend the fixed L=0 entry (1.0) and pad trailing zeros.
        pad_after = max_num_coeffs - num_coeffs
        pieces = [xp.asarray([1.0], dtype=xp.float64), row_coeffs]
        if pad_after > 0:
            pieces.append(xp.zeros(pad_after, dtype=xp.float64))
        rows.append(xp.concatenate(pieces))
    coeffs_arr = xp.stack(rows, axis=0)
    # Apply the (L + 1/2) factor column-wise.
    l_factor = xp.arange(max_num_coeffs, dtype=coeffs_arr.dtype) + 0.5
    return coeffs_arr * l_factor[None, :]


def compute_angdist_from_isotropic(
    endf_dict, mt, energies, angle_cosines, xp=None,
):
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mu = angle_cosines
    mu = mu.reshape(1, -1) if mu.ndim == 1 else mu
    m = len(energies)
    n = mu.shape[1]
    return xp.full((m, n), 0.5, dtype=xp.float64)


def compute_angdist_from_legrepr(
    endf_dict, mt, energies, angle_cosines, xp=None,
):
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mf4sec = endf_dict[4][mt]
    mu = angle_cosines
    incident_energies = dict2array(mf4sec['E'], dtype=float, xp=xp)
    nbt_arr = np.array(mf4sec['NBT'], dtype=int)
    int_arr = np.array(mf4sec['INT'], dtype=int)
    coeffs_arr = _convert_legendre_to_numpy_array(mf4sec['a'], xp=xp)
    f = evaluate_interp_legendre_polynomials(
        energies, mu, incident_energies, coeffs_arr,
        int_arr, nbt_arr, xp=xp,
    )
    return f


def compute_angdist_from_tabulated(
    endf_dict, mt, energies, angle_cosines, xp=None,
):
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mf4sec = endf_dict[4][mt]
    en_mesh = dict2array(mf4sec['E'], dtype=float, xp=xp)
    nbt_arr = np.array(mf4sec['energy_table']['NBT'], dtype=int)
    int_arr = np.array(mf4sec['energy_table']['INT'], dtype=int)
    tab1_records = list(mf4sec['angtable'].values())
    return interp_tab2(
        energies, angle_cosines, en_mesh, int_arr, nbt_arr,
        tab1_records, 'mu', 'f', xp=xp,
    )


def compute_angdist_from_mixed(
    endf_dict, mt, energies, angle_cosines, xp=None,
):
    """MF4 LTT=3 mixed Legendre + tabulated: evaluate the Legendre
    branch below the break energy and the tabulated branch at or
    above it, then select per-query with ``xp.where``.

    Both branches are evaluated on the FULL query grid (with
    ``outside_value=0.0`` so the branch that does not own a query
    contributes zero at that query) and the selection uses only
    xp operations, so ``jax.grad`` on a tracer ``energies`` flows
    through cleanly (issue #201 remaining half). The 2x
    per-query work vs the pre-fix branch-index scatter is a
    small constant factor and only bites when both branches
    would otherwise be dominated by dead work; on typical MF4
    LTT=3 files the Legendre branch is far cheaper than the
    tabulated one it complements so the doubling is not
    measurable in practice.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mf4sec = endf_dict[4][mt]
    en_mesh = dict2array(mf4sec['E'], dtype=float, xp=xp)
    # Legendre part covers energies below the split.
    num_ens1 = mf4sec['NE1']
    en_mesh1 = en_mesh[:num_ens1]
    nbt_arr1 = np.array(mf4sec['leg_int']['NBT'])
    int_arr1 = np.array(mf4sec['leg_int']['INT'])
    coeffs_arr = _convert_legendre_to_numpy_array(mf4sec['al'], xp=xp)
    assert num_ens1 == coeffs_arr.shape[0]
    # Tabulated part covers energies above the split.
    num_ens2 = mf4sec['NE2']
    en_mesh2 = en_mesh[num_ens1 - 1:]
    assert num_ens2 == len(en_mesh2)
    nbt_arr2 = np.array(mf4sec['ang_int']['NBT'])
    int_arr2 = np.array(mf4sec['ang_int']['INT'])
    tab1_records = list(mf4sec['angtable'].values())
    assert num_ens2 == len(tab1_records)
    break_energy = float(en_mesh[num_ens1 - 1])

    energies_xp = xp.asarray(energies)
    n_e = int(energies_xp.shape[0])
    mu = angle_cosines
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    n_mu = int(mu.shape[1])
    # Broadcast mu to (n_e, n_mu) so both branches see one shape.
    if mu.shape[0] == 1:
        mu = xp.broadcast_to(mu, (n_e, n_mu))

    # Legendre branch: zero outside ``en_mesh1``. For queries at or
    # above ``break_energy`` this returns 0 and the ``xp.where``
    # below drops it. Under xp=jax with a tracer ``energies``, this
    # call routes through ``evaluate_interp_legendre_polynomials``'s
    # traced-x path (verified in PR #234 pins).
    f_lower_full = evaluate_interp_legendre_polynomials(
        energies_xp, mu, en_mesh1, coeffs_arr,
        int_arr1, nbt_arr1, outside_value=0.0, xp=xp,
    )
    # Tabulated branch: zero outside ``en_mesh2``. For queries below
    # ``break_energy`` this returns 0.
    f_upper_full = interp_tab2(
        energies_xp, mu, en_mesh2, int_arr2, nbt_arr2,
        tab1_records, 'mu', 'f', outside_value=0.0, xp=xp,
    )
    is_lower = (energies_xp < break_energy).reshape(-1, 1)
    return xp.where(is_lower, f_lower_full, f_upper_full)


def _compute_r2(endf_dict, mt, energies, xp=None):
    awi = get_AWI(endf_dict)
    awr = get_AWR(endf_dict)
    awp = get_AWP(endf_dict, mt)
    q = get_QI(endf_dict, mt)
    return compute_r2(energies, awi, awr, awp, q, xp=xp)


@pad_outside_angdist_values
def compute_angdist_values(
    endf_dict, mt, energies, angle_cosines, to_lab=True, xp=None,
):
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mf4sec = endf_dict[4][mt]
    ltt = mf4sec['LTT']
    li = mf4sec['LI']
    lct = mf4sec['LCT'] if to_lab else 1
    # convert angle cosines to CM system if indicated
    mu = angle_cosines
    if lct == 1:
        mu_eff = mu
    elif lct == 2:
        r2 = _compute_r2(endf_dict, mt, energies, xp=xp)
        mu_eff = convert_angcos_to_cmsys(mu, r2, xp=xp)
    else:
        raise ValueError(f'Unknown reference system (LCT={lct}).')
    # perform the appropriate interpolation
    if ltt == 0 and li == 1:
        f_eff = compute_angdist_from_isotropic(
            endf_dict, mt, energies, mu_eff, xp=xp,
        )
    elif ltt == 1 and li == 0:
        f_eff = compute_angdist_from_legrepr(
            endf_dict, mt, energies, mu_eff, xp=xp,
        )
    elif ltt == 2 and li == 0:
        f_eff = compute_angdist_from_tabulated(
            endf_dict, mt, energies, mu_eff, xp=xp,
        )
    elif ltt == 3 and li == 0:
        f_eff = compute_angdist_from_mixed(
            endf_dict, mt, energies, mu_eff, xp=xp,
        )
    else:
        raise ValueError(
            'Unknown angular distribution representation '
            f'(MT={mt}, LTT={ltt}, LI={li}).'
        )
    # Convert result to LAB system if required. At LAB angles that
    # are kinematically forbidden (for equal-mass or heavy-ejectile
    # elastic like H-1, `mu_lab < mu_lab_min`), `convert_angcos_to_cmsys`
    # returned NaN via `sqrt(z)` with `z = mu_lab^2 + r^2 - 1 < 0`,
    # and that NaN propagated through the polynomial / tabulated
    # evaluation and the LAB Jacobian. Values just inside the
    # boundary picked up small-negative artefacts from the vanishing
    # Jacobian near the CM back-scatter singularity. Physics says
    # the angular distribution is zero at forbidden angles (no
    # scattering into unreachable angles), so we clip NaN and
    # negatives to zero here (issue #45). The Fortran reference
    # implementation does the same, so this brings the two backends
    # to bit-for-bit agreement on the H-1 elastic equivalence test
    # and removes the `equal_nan=True` workaround downstream.
    if lct == 1:
        f_lab = f_eff
    else:
        f_lab = convert_angdist_to_labsys(mu_eff, f_eff, r2, xp=xp)
        f_lab = xp.where(xp.isnan(f_lab), 0.0, f_lab)
        f_lab = xp.clip(f_lab, 0.0, None)
    return f_lab
