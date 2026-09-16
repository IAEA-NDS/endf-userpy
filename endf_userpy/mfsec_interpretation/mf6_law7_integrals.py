"""Knot-aware integration of MF6 LAW=7 (double-tabulated
angle-energy distributions) over the outgoing energy axis
(issue #69, follow-up to #46).

Background
----------
LAW=7 stores an outgoing-energy pdf `f(E_in, mu, E')` as a
tabulation over `(E_in_i, mu_j)` slices, each carrying its own E'
mesh. At an interpolated `(E_in, mu)` the value is reconstructed
via **unit-base interpolation** between the four bracketing slices
(mirrored in `mf6_get_law7` and `unit_base_intp` in
`endf_userpy/fortran/endf6.f90`):

    yslope    = (mu - u_j) / (u_{j+1} - u_j)
    xlow      = x1low  + yslope * (x2low  - x1low)
    xhigh     = x1high + yslope * (x2high - x1high)
    xrange    = xhigh - xlow
    xslope    = (E' - xlow) / xrange
    f1x       = tab1intp(x1, f1, x1low + xslope * x1range)
    f2x       = tab1intp(x2, f2, x2low + xslope * x2range)
    f(mu, E') = yintp(u_j, f1x*x1range/xrange, u_{j+1}, f2x*x2range/xrange, ...)

This transform is **nonlinear in E'** at the interpolated point,
so the tabulated E' knots do not describe the effective interpolant's
kinks. Trapezoid on the raw union of tabulated knots gives ~30 %
error (issue #46 measured this). Uniform-mesh Simpson (what
`distribution1d_helpers._adaptive_simpson_along_axis` uses today)
converges slowly because the reconstructed function has kinks that
don't align with the mesh: measured 0.5-1 % of peak on the Be-9
MT16 (n,2n) boundary case.

Approach
--------
For each target `(E_in, mu)` cell:

1. Locate bracketing indices in the section's `E` and per-`E_in`
   `mu` meshes.
2. For each of the two bracketing `E_in` slices, project the raw
   tabulated `Ep[k]` knots of the two bracketing mu tables through
   the unit-base transform at target `mu` (`_project_ub_knots`) --
   this gives the effective knot mesh contribution from that
   slice. Union with the other slice's contribution.
3. Clip the effective-knot set to `[0, (E_in + q) * 1.1]`.
4. Evaluate `f` at the midpoint of every segment defined by
   consecutive effective knots (plus the integration endpoints).
5. Sum `width * f(midpoint)` per segment.

The midpoint rule is exact for piecewise-constant (`INT=1`) and
piecewise-linear (`INT=2`) E' interpolants -- the two schemes that
appear in the corpus (Be-9: `INT=1` only, Cu-63 JENDL-5: `INT=2`
only, surveyed in issue #69). For `INT=3/4/5` (log-based) the rule
has `O(h^3 * f'')` error per segment, which stays well under the
uniform-Simpson error at the same mesh size.

Only single-subsection LAW=7 currently routes here (matching the
existing single-subsection LAW=1 Fortran shortcut in
`distribution1d_helpers.integrate_mf6_dist2d_over_mu`).
Multi-subsection LAW=7 continues to use the general-purpose
adaptive Simpson integrator.
"""
import numpy as np

from ..primitives.helpers import dict2array, find_interval
from ..primitives.properties import get_QM, get_QI
from . import mf6_interpretation_subsecs as _subsec


def _project_ub_knots(y0, y1, y2, x1_knots, x2_knots):
    """Map the raw `x` knots of two tables (at `y1` and `y2`) to
    the effective `x`-axis at target `y0` in `[y1, y2]` via the
    LAW=7 unit-base transform.

    Returns the sorted union of the two projected knot sets. If
    either table has zero-width (`x*range == 0`, degenerate), its
    knots collapse to the effective `xlow`; the other set is
    returned unchanged plus that single point.
    """
    x1_knots = np.asarray(x1_knots, dtype=float)
    x2_knots = np.asarray(x2_knots, dtype=float)
    if y2 == y1:
        yslope = 0.0
    else:
        yslope = (y0 - y1) / (y2 - y1)
    x1low, x1high = x1_knots[0], x1_knots[-1]
    x2low, x2high = x2_knots[0], x2_knots[-1]
    x1range = x1high - x1low
    x2range = x2high - x2low
    xlow = x1low + yslope * (x2low - x1low)
    xhigh = x1high + yslope * (x2high - x1high)
    xrange = xhigh - xlow
    if xrange <= 0.0:
        return np.array([xlow], dtype=float)
    if x1range > 0.0:
        eff_from_1 = xlow + (x1_knots - x1low) * (xrange / x1range)
    else:
        eff_from_1 = np.array([xlow], dtype=float)
    if x2range > 0.0:
        eff_from_2 = xlow + (x2_knots - x2low) * (xrange / x2range)
    else:
        eff_from_2 = np.array([xlow], dtype=float)
    return np.unique(np.concatenate([eff_from_1, eff_from_2]))


def integrate_law7_subsec_over_eout(
    endf_dict, mt, subsec_num, energies_in, angle_cosines_out, to_lab=True,
):
    """Angular distribution `da(E_in, mu)` from a single MF6
    subsection with `LAW=7`, integrated over outgoing energy on
    `[0, (E_in + q) * 1.1]`.

    Uses knot-aware midpoint integration on the effective E' mesh
    derived from the section's tabulated knots and the LAW=7
    unit-base transform. See module docstring for the derivation.

    Parameters
    ----------
    endf_dict, mt, subsec_num : the endf dict and section pointers.
    energies_in : 1D array of incident energies (eV).
    angle_cosines_out : 1D array of mu targets.
    to_lab : accepted for signature compatibility; LAW=7 is always
        stored in the LAB frame.

    Returns
    -------
    ndarray of shape `(len(energies_in), len(angle_cosines_out))`.
    """
    sec = endf_dict[6][mt]
    sub = sec['subsection'][subsec_num]
    if sub['LAW'] != 7:
        raise ValueError(
            f'MT={mt} subsec_num={subsec_num} is LAW={sub["LAW"]}, '
            'not LAW=7',
        )
    einc = np.asarray(energies_in, dtype=float)
    mus = np.asarray(angle_cosines_out, dtype=float)
    q = max(get_QM(endf_dict, mt), get_QI(endf_dict, mt))
    ei_mesh = dict2array(sub['E'], dtype=float)

    out = np.zeros((len(einc), len(mus)), dtype=float)

    for i, e in enumerate(einc):
        # Kinematic bounds: outside the section's E_in range -> zero.
        if e < ei_mesh[0] or e > ei_mesh[-1]:
            continue
        eout_max = (e + q) * 1.1
        if eout_max <= 0.0:
            continue
        curidx = int(find_interval(ei_mesh, np.array([e]))[0])
        mu_mesh_e1 = dict2array(sub['mu'][curidx + 1], dtype=float)
        mu_mesh_e2 = dict2array(sub['mu'][curidx + 2], dtype=float)
        tables_e1 = sub['table'][curidx + 1]
        tables_e2 = sub['table'][curidx + 2]

        # Precompute per-mu-slot knot arrays for the two bracketing
        # E_in slices; per-cell projection then does one 1D union.
        for j, u in enumerate(mus):
            # Outside the section's mu range at either bracketing
            # E_in slice -> zero (matches `mf6_get_law7`).
            if u < mu_mesh_e1[0] or u > mu_mesh_e1[-1]:
                continue
            if u < mu_mesh_e2[0] or u > mu_mesh_e2[-1]:
                continue
            j1 = int(find_interval(mu_mesh_e1, np.array([u]))[0])
            j2 = int(find_interval(mu_mesh_e2, np.array([u]))[0])

            ep11 = np.asarray(tables_e1[j1 + 1]['Ep'], dtype=float)
            ep12 = np.asarray(tables_e1[j1 + 2]['Ep'], dtype=float)
            ep21 = np.asarray(tables_e2[j2 + 1]['Ep'], dtype=float)
            ep22 = np.asarray(tables_e2[j2 + 2]['Ep'], dtype=float)

            knots_e1 = _project_ub_knots(
                u, mu_mesh_e1[j1], mu_mesh_e1[j1 + 1], ep11, ep12,
            )
            knots_e2 = _project_ub_knots(
                u, mu_mesh_e2[j2], mu_mesh_e2[j2 + 1], ep21, ep22,
            )
            knots = np.unique(np.concatenate([knots_e1, knots_e2]))

            # Clip to the physical integration range and pad the
            # endpoints. The reconstructed f is zero outside the
            # effective [xlow, xhigh] envelopes, so the segments
            # before the first knot and after the last knot
            # naturally contribute zero (midpoint samples fall
            # outside the tabulation and return 0).
            knots = knots[(knots > 0.0) & (knots < eout_max)]
            mesh = np.concatenate([[0.0], knots, [eout_max]])

            # Midpoint rule per segment: exact for INT=1 and INT=2
            # (piecewise-constant / piecewise-linear reconstructed
            # f between effective knots), O(h^3 * f'') for INT>=3.
            mids = 0.5 * (mesh[:-1] + mesh[1:])
            widths = mesh[1:] - mesh[:-1]

            dist = _subsec.get_dist2d_from_subsec_law7(
                endf_dict, mt, subsec_num,
                np.array([e], dtype=float),
                mids,
                np.array([u], dtype=float),
                True,
            )
            # dist shape: (1, len(mids), 1)
            fvals = dist[0, :, 0]
            out[i, j] = float(np.sum(widths * fvals))

    return out
