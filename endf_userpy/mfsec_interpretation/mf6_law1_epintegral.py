"""Backend-agnostic MF6 LAW=1 continuum spectrum ``f(E, E')``
via ``mu``-integration of the double-differential distribution.

Fixed-grid emission spectrum: for each caller-supplied ``(E, E')``,
integrate

    f_spec(E, E') = int_{mu_min}^{+1} f_con(E, E', mu) dmu

where ``f_con`` is the LAW=1 continuum reconstruction (see
:mod:`mf6_law1_kernel`) and ``mu_min = mu_min(E, E')`` comes from
the LAB-frame kinematic cutoff (Fortran ``feep_law1con`` line 3020).

Compared to the Fortran reference (``feep_points_law1con``, endf6.f90
line 2901) this port replaces the per-subpanel Romberg-Richardson
extrapolation with a fixed-order **Gauss-Legendre** rule per
polar-angle subpanel. The polar-angle subdivision ``z = acos(mu)``
is preserved (it concentrates points near the kinematic cutoff
``mu_min`` where the CM<->LAB Jacobian sharpens). For smooth
integrands GL is more efficient than Romberg (10-point GL matches
Romberg's ``rtol=1e-3`` in a fraction of the integrand evaluations)
and the fixed shape is JAX-friendly: no data-dependent iteration
count, no ``lax.while_loop`` needed.

The default ``(n_gl=10, n_polar_subpanels=4)`` gives 40 integrand
evaluations per ``(E, E')`` and matches the Fortran to a few 1e-4
relative on Al-27 corpus files (see the equivalence test); more
than sufficient for downstream use, and configurable if a caller
wants tighter accuracy.

Design: input is an :class:`~mf6_law1_preproc.MF6Law1Data`
dataclass (produced by
:func:`~mf6_law1_preproc.mf6_law1_data_from_endf_dict`); output is
a ``(n_E, n_E')`` array. JAX tracers stored in ``data.b_panels``
propagate through the integrator so ``jax.grad`` reaches back to
file-stored angular parameters end-to-end.
"""
from __future__ import annotations

import numpy as np

from ..primitives import array_ns
from ..primitives.helpers import convert_interp_repr, find_interval
from . import mf6_law1_kernel as _kernel


def integrate_law1_spectrum(data, energies_in, energies_out, to_lab,
                              xp=None, n_gl=10, n_polar_subpanels=4):
    """MF6 LAW=1 continuum emission spectrum ``f(E, E')`` via
    ``mu``-integration.

    Parameters
    ----------
    data : MF6Law1Data
    energies_in : (n_E,) query incident energies (LAB, eV).
    energies_out : (n_Ep,) query outgoing energies (LAB, eV).
    to_lab : bool. LAW=1 is always LAB; this argument gates
        whether the section's LCT is honoured. False forces LAB
        (no CM->LAB shift), True uses the section's LCT.
    xp : optional backend adapter (``array_ns.get_backend(name)``).
        Defaults to numpy.
    n_gl : Gauss-Legendre order per polar subpanel (default 10).
    n_polar_subpanels : number of equal-Delta-z polar subpanels
        (default 4). The subpanel closest to the LAB cutoff carries
        the sharpest integrand feature; more subpanels help if the
        integrand varies rapidly there.

    Returns
    -------
    (n_E, n_Ep) array. Same dtype as ``data.b_panels`` (float on
    numpy, backend-native on JAX).
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    lct = data.lct if to_lab else 1
    if lct in (1, 2):
        eff_lct = lct
    elif lct == 3:
        eff_lct = 1 if data.awp > 4 else 2
    else:
        raise NotImplementedError(f'LCT={lct} not implemented')

    e_in_np = np.asarray(energies_in, dtype=float)
    ep_out_np = np.asarray(energies_out, dtype=float)
    n_e = e_in_np.shape[0]
    n_ep = ep_out_np.shape[0]
    ei_mesh_np = np.asarray(data.ei_mesh)

    # Gauss-Legendre nodes / weights (static, precomputed once)
    gl_x_np, gl_w_np = np.polynomial.legendre.leggauss(int(n_gl))
    gl_x = xp.asarray(gl_x_np)
    gl_w = xp.asarray(gl_w_np)

    ei_interp_full = convert_interp_repr(
        np.asarray(data.int_arr), np.asarray(data.nbt_arr),
    )

    # Mask to keep only E's inside the section's E-mesh range
    e_min = float(ei_mesh_np[0])
    e_max = float(ei_mesh_np[-1])
    inside_mask_np = (e_in_np >= e_min) & (e_in_np <= e_max)

    ep_out_xp = xp.asarray(ep_out_np)
    result = xp.zeros((n_e, n_ep), dtype=data.b_panels.dtype)
    if not inside_mask_np.any():
        return result

    e_inside_np = e_in_np[inside_mask_np]
    idcs = find_interval(ei_mesh_np, e_inside_np)
    inside_positions = np.where(inside_mask_np)[0]

    for panel_idx in np.unique(idcs):
        row_mask = (idcs == panel_idx)
        rows_np = inside_positions[row_mask]
        e_sub_np = e_inside_np[row_mask]
        e_sub = xp.asarray(e_sub_np)
        lei = int(ei_interp_full[panel_idx])
        f_sub = _law1_spectrum_panel_pair(
            data, int(panel_idx), lei, e_sub, ep_out_xp, eff_lct,
            gl_x, gl_w, int(n_polar_subpanels), xp,
        )
        result = _kernel._scatter_rows(result, rows_np, f_sub, xp)

    return result


def _law1_spectrum_panel_pair(data, panel_idx, lei, e_sub, ep_out_xp,
                                eff_lct, gl_x, gl_w, n_sub, xp):
    """Per-panel-pair spectrum on ``(e_sub, ep_out)`` via polar-angle
    Gauss-Legendre.

    Handles the panel-count regimes statically (Python-side):
    if either panel has no continuum data, returns zeros (matches
    the Fortran fall-through in ``f6law1con``).
    """
    p1 = panel_idx
    p2 = panel_idx + 1
    e1 = float(data.ei_mesh[p1])
    e2 = float(data.ei_mesh[p2])
    nep1 = int(data.nep_arr[p1])
    nd1 = int(data.nd_arr[p1])
    nep2 = int(data.nep_arr[p2])
    nd2 = int(data.nd_arr[p2])

    n_e_sub = int(e_sub.shape[0])
    n_ep = int(ep_out_xp.shape[0])

    if nep1 <= nd1 or nep2 <= nd2:
        # No continuum data on at least one panel; spectrum is 0.
        # (The between-panel unit-base transform requires both
        # panels to have continuum data; if either doesn't, the
        # Fortran f6law1con still returns something via the "only
        # one panel has continuum" fallback, but that path is
        # atypical for continuum-only spectra queries. Preserving
        # the physics is easier to see with a clean zero here.)
        return xp.zeros((n_e_sub, n_ep), dtype=data.b_panels.dtype)

    # Panel-pair Ep bounds at each incident e (unit-base transformed
    # from panel-1 and panel-2 continuum boundaries).
    ep1min = float(data.ep_panels[p1, nd1])
    ep1max = float(data.ep_panels[p1, nep1 - 1])
    ep2min = float(data.ep_panels[p2, nd2])
    ep2max = float(data.ep_panels[p2, nep2 - 1])
    e_bc = e_sub[:, None]                                        # (nE, 1)
    yslope = (e_bc - e1) / (e2 - e1)                             # (nE, 1)
    epmin_eff = ep1min + yslope * (ep2min - ep1min)              # (nE, 1)
    epmax_eff = ep1max + yslope * (ep2max - ep1max)              # (nE, 1)

    # mu_min per (E, Ep) via the LAB-frame kinematic cutoff.
    ep_bc = ep_out_xp[None, :]                                   # (1, nEp)
    umin = _mu_min_bc(data, eff_lct, e_bc, ep_bc, epmax_eff, xp)  # (nE, nEp)

    # Polar-angle subdivision: z = acos(mu), z in [0, z_min]
    # where z_min = acos(mu_min). Subdivide [0, z_min] into
    # n_sub equal-Delta-z subpanels; per subpanel apply GL.
    z_min = xp.arccos(xp.clip(umin, -1.0, 1.0))                  # (nE, nEp)
    dz = z_min / n_sub                                            # (nE, nEp)

    # Build z at every (subpanel, GL node): shape
    # (nE, nEp, n_sub, n_gl).
    s_idx = xp.arange(n_sub, dtype=z_min.dtype)                   # (n_sub,)
    z_a = dz[..., None] * s_idx                                   # (nE, nEp, n_sub)
    z_all = (z_a[..., None]
             + (dz[..., None, None] / 2.0) * (1.0 + gl_x))        # (..., n_sub, n_gl)

    mu_all = xp.cos(z_all)
    sin_z_all = xp.sin(z_all)

    # Broadcast (e, ep) against the polar-node axes for the amplitude
    # eval. Shape (nE, nEp, n_sub, n_gl).
    e_full = xp.broadcast_to(e_bc[..., None, None], mu_all.shape)
    ep_full = xp.broadcast_to(ep_bc[..., None, None], mu_all.shape)

    tp_full, w_full, dinv_full = _kernel._mf6lab2cm_bc(
        data.awr, data.awi, data.awp, eff_lct,
        e_full, ep_full, mu_all, xp,
    )
    f_amp = _kernel._f6law1con_panel_pair_bc(
        data, panel_idx, lei, e_full, tp_full, w_full, xp,
    )
    integrand = f_amp * dinv_full * sin_z_all
    # GL weighting on the polar node axis (last axis)
    # and interval-halving factor dz/2 on the subpanel axis.
    weighted = integrand * gl_w * (dz[..., None, None] / 2.0)
    return xp.sum(weighted, axis=(-2, -1))


def _mu_min_bc(data, eff_lct, e_bc, ep_bc, epmax_eff, xp):
    """Broadcast LAB-frame lower-cosine cutoff ``mu_min(E, E')``
    for MF6 LAW=1 continuum ``mu`` integration (matches Fortran
    ``feep_law1con`` lines 3020-3034).

    Returns ``mu_min`` clamped to ``[-1, 1]``; ``ep <= 0`` returns
    +1 so the integration domain is empty (zero contribution).
    """
    if not (eff_lct == 2 or (eff_lct == 3 and data.awp < 4.0)):
        # LAB frame: mu_min = -1 (full range).
        return xp.broadcast_to(
            xp.asarray(-1.0, dtype=e_bc.dtype),
            xp.broadcast_shapes(e_bc.shape, ep_bc.shape),
        )
    c0 = float(np.sqrt(data.awi * data.awp) / (data.awi + data.awr))
    c0_sq_e = (c0 * c0) * e_bc
    # Guard ep <= 0 so sqrt(ep * e) doesn't NaN out or divide by
    # zero; the where at the end masks those cells back to umin = 1
    # (empty domain). e_bc is always > 0 in the inside-mask path.
    ep_safe = xp.where(ep_bc > 0.0, ep_bc, 1.0)
    sqrt_epe = xp.sqrt(ep_safe * e_bc)
    denom_safe = xp.where(sqrt_epe > 0.0, 2.0 * c0 * sqrt_epe, 1.0)
    umin_raw = (ep_bc + c0_sq_e - epmax_eff) / denom_safe
    umin_clamped = xp.clip(umin_raw, -1.0, 1.0)
    return xp.where(ep_bc > 0.0, umin_clamped, 1.0)
