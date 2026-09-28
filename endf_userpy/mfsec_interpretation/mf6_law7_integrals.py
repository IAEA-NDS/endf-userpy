"""Panel-exact integration of MF6 LAW=7 (double-tabulated
angle-energy distributions).

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
    f1x       = tab1intp(x1_knots, f1_vals, x1low + xslope * x1range)
    f2x       = tab1intp(x2_knots, f2_vals, x2low + xslope * x2range)
    f_slice   = yintp(u_j, f1x*x1range/xrange, u_{j+1}, f2x*x2range/xrange, ...)

Then the outer E_in-axis blends two adjacent-Ein-slice values via
:func:`_law7_outer_e_interp_row` (typically INT=2 lin-lin).

Panel-exact integration over E'
-------------------------------
For INT=2 uniformly on the outer E_in axis and INT=2 (base) on
each mu axis, the reconstruction reduces to

    f(E_in, mu, E') = (1 - eslope) * f_slice_e1(E')
                   +      eslope  * f_slice_e2(E')
    f_slice_ek(E') = (1 - yslope_ek) * f1x_ek(E') · x1range_ek/xrange_ek
                   +      yslope_ek  * f2x_ek(E') · x2range_ek/xrange_ek

The **four tab1 interpolants** f1x_e1, f2x_e1, f1x_e2, f2x_e2 are
each an INT-native interp of one table evaluated at a
linear-in-E' shifted position. Between consecutive **effective
knots** (the union of all four tables' projected knot sets) each
of the four sits inside a single panel, so each is a pure INT=k
expression on that segment.

The full E' integral is therefore

    integral_a^b f dE' = (1 - eslope) * [
        (1 - yslope_e1) * integrate_tab1_panels(table_1_e1_panel, xi_1_e1(a), xi_1_e1(b))
      +      yslope_e1  * integrate_tab1_panels(table_2_e1_panel, xi_2_e1(a), xi_2_e1(b))
    ] + eslope * [same for e2 slice]

where each xi_k(t) = x_low_k + (t - xlow_slice)/xrange_slice · x_range_k
is the linear map from the effective-E' axis to table k's own E' axis.
Jacobian (dE'/dxi_k = xrange_slice / x_range_k) exactly cancels the
per-slice unit-base weight x_range_k / xrange_slice, so the formula is
clean.

**This is exact for every INT code** (histogram, lin-lin, and all
three log laws) on the E' axis of every underlying table. It
replaces the pre-refactor midpoint rule, which was exact for INT=1
and INT=2 on real corpus files but only O(h^3 * f'') per segment
for INT=3/4/5 hypothetical files, and required a runtime
error-estimator warning (`Law7LogErrorAccumulator`, dropped in
this refactor).

Scope
-----
- Outer E_in-axis: only ``INT=1 / INT=2`` supported. Real corpus
  LAW=7 files use INT=2 exclusively.
- Mu axes (both slices): only base ``INT=1 / INT=2`` supported.
- E' axis of any underlying table: all five INT codes (1..5)
  handled panel-exact by :func:`~primitives.interpolation.integrate_tab1_panels`.

Files with unsupported outer or mu-axis INT raise
``NotImplementedError`` with a clear diagnostic.

Only single-subsection LAW=7 currently routes here (matching the
existing single-subsection LAW=1 Fortran shortcut in
``distribution1d_helpers.integrate_mf6_dist2d_over_mu``).
Multi-subsection LAW=7 continues to use the general-purpose
adaptive Simpson integrator.
"""
import numpy as np

from ..primitives.helpers import (
    convert_interp_repr, dict2array, find_interval,
)
from ..primitives.interpolation import integrate_tab1_panels
from ..primitives.properties import get_QM, get_QI
from . import mf6_interpretation_subsecs as _subsec  # noqa: F401


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


def _ub_slice_geometry(mu, mu1, mu2, ep1, ep2):
    """Unit-base geometry for one Ein-slice's mu-bracket.

    Given target mu and the mu-bracket (mu1, mu2), plus the two
    tables' E' knot arrays (ep1, ep2), return
    ``(yslope, xlow, xrange, x1low, x1range, x2low, x2range)`` for
    the LAW=7 unit-base transform. All are per-cell scalars.
    """
    if mu2 == mu1:
        yslope = 0.0
    else:
        yslope = (mu - mu1) / (mu2 - mu1)
    x1low, x1high = float(ep1[0]), float(ep1[-1])
    x2low, x2high = float(ep2[0]), float(ep2[-1])
    x1range = x1high - x1low
    x2range = x2high - x2low
    xlow = x1low + yslope * (x2low - x1low)
    xhigh = x1high + yslope * (x2high - x1high)
    xrange = xhigh - xlow
    return yslope, xlow, xrange, x1low, x1range, x2low, x2range


def _tab_panel_int(int_arr, nbt_arr, n_pts):
    """Per-panel INT code for a tab1 record. Length ``n_pts - 1``."""
    int_per_point = convert_interp_repr(
        np.asarray(int_arr, dtype=int),
        np.asarray(nbt_arr, dtype=int),
    )
    if int_per_point.shape[0] <= 1:
        return int_per_point
    # Panel between point i and i+1 uses INT at breakpoint i+1.
    return int_per_point[1:]


def _table_segment_areas_vec(ep_arr, f_arr, int_per_panel, xi_a_arr, xi_b_arr):
    """Vectorised panel-exact integrals of one table's tab1
    interpolant from ``xi_a`` to ``xi_b`` for a batch of segments.

    Each segment is assumed to lie inside a single panel of the
    table (guaranteed by the effective-knot mesh construction).
    Segments outside the table's domain or with non-positive
    width contribute 0.

    Panel indexing uses the segment midpoint - the segment
    endpoints ``xi_a`` come from a linear projection that can
    drift by a few ULP from an exact table knot, and would
    otherwise land on the wrong panel under ``searchsorted``.
    """
    xi_a_arr = np.asarray(xi_a_arr, dtype=float)
    xi_b_arr = np.asarray(xi_b_arr, dtype=float)
    n = xi_a_arr.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    # Clip to the table's own domain.
    lo, hi = float(ep_arr[0]), float(ep_arr[-1])
    xa = np.clip(xi_a_arr, lo, hi)
    xb = np.clip(xi_b_arr, lo, hi)
    active = (xi_b_arr > xi_a_arr) & (xi_a_arr < hi) & (xi_b_arr > lo)
    # Panel index by midpoint, clamped to a valid range.
    mid = 0.5 * (xa + xb)
    panels = np.clip(
        np.searchsorted(ep_arr, mid, side='right') - 1,
        0, ep_arr.shape[0] - 2,
    )
    x0 = ep_arr[panels]
    x1 = ep_arr[panels + 1]
    y0 = f_arr[panels]
    y1 = f_arr[panels + 1]
    codes = (
        int_per_panel[panels]
        if int_per_panel.shape[0]
        else np.full(n, 2, dtype=int)
    )
    areas = integrate_tab1_panels(x0, x1, y0, y1, xa, xb, codes)
    return np.where(active, areas, 0.0)


def _check_supported_int(int_arr, nbt_arr, axis_name):
    """Guard: this integrator supports only INT=1/2 on the outer
    Ein and mu axes (they blend the four table amplitudes; higher
    INT codes make the blend non-linear so the four-integral
    decomposition doesn't apply). Raises with a clear message.
    """
    codes = set(int(v) for v in np.asarray(int_arr, dtype=int))
    unsupported = codes - {1, 2}
    if unsupported:
        raise NotImplementedError(
            f'MF6 LAW=7 panel-exact integration requires '
            f'INT=1 or INT=2 on the {axis_name} axis; got '
            f'INT codes {sorted(unsupported)}. Real corpus files '
            f'use INT=2 exclusively on this axis; file this if a '
            f'real file hits it.'
        )
    del nbt_arr


def integrate_law7_subsec_over_eout(
    endf_dict, mt, subsec_num, energies_in, angle_cosines_out, to_lab=True,
):
    """Angular distribution ``da(E_in, mu)`` from a single MF6
    subsection with ``LAW=7``, integrated over outgoing energy on
    ``[0, (E_in + q) * 1.1]`` panel-exactly.

    Sums the analytic per-effective-segment integral of the
    LAW=7 unit-base reconstruction using
    :func:`~primitives.interpolation.integrate_tab1_panels` for
    each of the four bracketing tables' contributions. Exact for
    all five ENDF INT codes on the E' axis; requires INT=1/2 on
    the outer Ein and mu axes (all corpus files satisfy this).

    Parameters
    ----------
    endf_dict, mt, subsec_num : the endf dict and section pointers.
    energies_in : 1D array of incident energies (eV).
    angle_cosines_out : 1D array of mu targets.
    to_lab : accepted for signature compatibility; LAW=7 is always
        stored in the LAB frame.

    Returns
    -------
    ndarray of shape ``(len(energies_in), len(angle_cosines_out))``.
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

    # Guard on outer Ein-axis INT.
    _check_supported_int(
        sub['E_interpol']['INT'], sub['E_interpol']['NBT'],
        axis_name='outer Ein',
    )
    e_int_per_point = convert_interp_repr(
        np.asarray(sub['E_interpol']['INT'], dtype=int),
        np.asarray(sub['E_interpol']['NBT'], dtype=int),
    )

    out = np.zeros((len(einc), len(mus)), dtype=float)

    for i, e in enumerate(einc):
        # Kinematic bounds: outside the section's E_in range -> zero.
        if e < ei_mesh[0] or e > ei_mesh[-1]:
            continue
        eout_max = (e + q) * 1.1
        if eout_max <= 0.0:
            continue
        curidx = int(find_interval(ei_mesh, np.array([e]))[0])
        e1 = float(ei_mesh[curidx])
        e2 = float(ei_mesh[curidx + 1])
        # Outer Ein-blend base law (mod 10; unit-base 20+k unused here).
        lei_law = int(e_int_per_point[curidx + 1]) % 10
        if lei_law == 1:
            eslope_1 = 1.0
            eslope_2 = 0.0
        else:  # INT=2
            eslope_2 = (e - e1) / (e2 - e1) if e2 != e1 else 0.0
            eslope_1 = 1.0 - eslope_2

        # Per-Ein-slice: mu meshes, tables, mu-axis INT.
        mu_mesh_e1 = dict2array(sub['mu'][curidx + 1], dtype=float)
        mu_mesh_e2 = dict2array(sub['mu'][curidx + 2], dtype=float)
        tables_e1 = sub['table'][curidx + 1]
        tables_e2 = sub['table'][curidx + 2]
        _check_supported_int(
            sub['mu_interpol'][curidx + 1]['INT'],
            sub['mu_interpol'][curidx + 1]['NBT'],
            axis_name='mu (Ein slice 1)',
        )
        _check_supported_int(
            sub['mu_interpol'][curidx + 2]['INT'],
            sub['mu_interpol'][curidx + 2]['NBT'],
            axis_name='mu (Ein slice 2)',
        )
        mu_int_e1 = convert_interp_repr(
            np.asarray(sub['mu_interpol'][curidx + 1]['INT'], dtype=int),
            np.asarray(sub['mu_interpol'][curidx + 1]['NBT'], dtype=int),
        )
        mu_int_e2 = convert_interp_repr(
            np.asarray(sub['mu_interpol'][curidx + 2]['INT'], dtype=int),
            np.asarray(sub['mu_interpol'][curidx + 2]['NBT'], dtype=int),
        )

        for j, u in enumerate(mus):
            # Outside the section's mu range at either bracketing
            # E_in slice -> zero (matches `mf6_get_law7`).
            if u < mu_mesh_e1[0] or u > mu_mesh_e1[-1]:
                continue
            if u < mu_mesh_e2[0] or u > mu_mesh_e2[-1]:
                continue
            j1 = int(find_interval(mu_mesh_e1, np.array([u]))[0])
            j2 = int(find_interval(mu_mesh_e2, np.array([u]))[0])

            # Four bracketing tables (E_in-slice x mu-slot).
            t_e1_1 = tables_e1[j1 + 1]
            t_e1_2 = tables_e1[j1 + 2]
            t_e2_1 = tables_e2[j2 + 1]
            t_e2_2 = tables_e2[j2 + 2]

            ep11 = np.asarray(t_e1_1['Ep'], dtype=float)
            ep12 = np.asarray(t_e1_2['Ep'], dtype=float)
            ep21 = np.asarray(t_e2_1['Ep'], dtype=float)
            ep22 = np.asarray(t_e2_2['Ep'], dtype=float)
            f11 = np.asarray(t_e1_1['f'], dtype=float)
            f12 = np.asarray(t_e1_2['f'], dtype=float)
            f21 = np.asarray(t_e2_1['f'], dtype=float)
            f22 = np.asarray(t_e2_2['f'], dtype=float)
            int11 = _tab_panel_int(t_e1_1['INT'], t_e1_1['NBT'], ep11.shape[0])
            int12 = _tab_panel_int(t_e1_2['INT'], t_e1_2['NBT'], ep12.shape[0])
            int21 = _tab_panel_int(t_e2_1['INT'], t_e2_1['NBT'], ep21.shape[0])
            int22 = _tab_panel_int(t_e2_2['INT'], t_e2_2['NBT'], ep22.shape[0])

            # Per-slice unit-base geometry.
            yslope_e1, xlow_e1, xrange_e1, x1low_e1, x1range_e1, x2low_e1, x2range_e1 = (
                _ub_slice_geometry(u, mu_mesh_e1[j1], mu_mesh_e1[j1 + 1], ep11, ep12)
            )
            yslope_e2, xlow_e2, xrange_e2, x1low_e2, x1range_e2, x2low_e2, x2range_e2 = (
                _ub_slice_geometry(u, mu_mesh_e2[j2], mu_mesh_e2[j2 + 1], ep21, ep22)
            )
            # Mu-blend base law (mod 10). INT=1 -> use left value only.
            mu_int_e1_base = int(mu_int_e1[j1 + 1]) % 10
            mu_int_e2_base = int(mu_int_e2[j2 + 1]) % 10
            if mu_int_e1_base == 1:
                yslope_e1 = 0.0
            if mu_int_e2_base == 1:
                yslope_e2 = 0.0

            # Effective-knot mesh, clipped to physical support.
            knots_e1 = _project_ub_knots(
                u, mu_mesh_e1[j1], mu_mesh_e1[j1 + 1], ep11, ep12,
            )
            knots_e2 = _project_ub_knots(
                u, mu_mesh_e2[j2], mu_mesh_e2[j2 + 1], ep21, ep22,
            )
            knots = np.unique(np.concatenate([knots_e1, knots_e2]))
            knots = knots[(knots > 0.0) & (knots < eout_max)]
            mesh = np.concatenate([[0.0], knots, [eout_max]])

            # Vectorised per-segment integration: build one (n_seg,)
            # array of clipped bounds per slice, project to each
            # table's own xi axis, and call the panel-exact
            # integrator once per table.
            seg_a = mesh[:-1]
            seg_b = mesh[1:]

            def _slice_areas(
                a_bnd, b_bnd, xlow, xrange_, x1low, x1range,
                x2low, x2range, ep_a, f_a, int_a, ep_b_, f_b_, int_b_,
                yslope,
            ):
                if xrange_ <= 0.0:
                    return np.zeros(seg_a.shape[0], dtype=float)
                # Clip each segment to the slice's effective E' range.
                a_c = np.maximum(a_bnd, xlow)
                b_c = np.minimum(b_bnd, xlow + xrange_)
                active = b_c > a_c
                # Wherever inactive, set widths to 0 so integrate_tab1_
                # panels returns 0 anyway; keep the array shape stable.
                a_c = np.where(active, a_c, xlow)
                b_c = np.where(active, b_c, xlow)
                # Shifted x for table_a.
                xi_a_a = x1low + (a_c - xlow) / xrange_ * x1range
                xi_a_b = x1low + (b_c - xlow) / xrange_ * x1range
                # Shifted x for table_b.
                xi_b_a = x2low + (a_c - xlow) / xrange_ * x2range
                xi_b_b = x2low + (b_c - xlow) / xrange_ * x2range
                area_a = _table_segment_areas_vec(
                    ep_a, f_a, int_a, xi_a_a, xi_a_b,
                )
                area_b = _table_segment_areas_vec(
                    ep_b_, f_b_, int_b_, xi_b_a, xi_b_b,
                )
                return (1.0 - yslope) * area_a + yslope * area_b

            s1 = _slice_areas(
                seg_a, seg_b, xlow_e1, xrange_e1,
                x1low_e1, x1range_e1, x2low_e1, x2range_e1,
                ep11, f11, int11, ep12, f12, int12, yslope_e1,
            )
            s2 = _slice_areas(
                seg_a, seg_b, xlow_e2, xrange_e2,
                x1low_e2, x1range_e2, x2low_e2, x2range_e2,
                ep21, f21, int21, ep22, f22, int22, yslope_e2,
            )
            out[i, j] = float(np.sum(eslope_1 * s1 + eslope_2 * s2))

    return out


# Composite-Simpson node count per LAW=7 mu segment: 33 samples per
# segment means 32 sub-intervals of Simpson (must be even), enough
# to resolve intra-segment E'-tab1 sub-kinks that a low-order rule
# misses. Two-level Richardson-style test at n=17 and n=33 catches
# unconverged segments and bumps them to n=65 automatically.
_LAW7_MU_SEG_N_COARSE = 17
_LAW7_MU_SEG_N_FINE = 33
_LAW7_MU_SEG_N_REFINED = 65
_LAW7_MU_SEG_RTOL = 5e-5


def _simpson_1d(f_samples, h):
    """Composite Simpson on samples with uniform spacing ``h``, along
    the last axis. Requires ``f_samples.shape[-1]`` to be odd (i.e.,
    an even number of sub-intervals).
    """
    n = f_samples.shape[-1]
    assert n % 2 == 1, 'composite Simpson requires odd sample count'
    # Weights: 1, 4, 2, 4, 2, ..., 4, 1
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    return (h / 3.0) * (f_samples * w).sum(axis=-1)


def integrate_law7_subsec_over_mu(
    endf_dict, mt, subsec_num, energies_in, energies_out, to_lab=True,
):
    """Energy distribution ``de(E_in, E_out)`` from a single MF6
    subsection with ``LAW=7``, integrated over ``mu`` on ``[-1, +1]``.

    Kink-aware per-segment composite-Simpson integration on the mu
    axis, analogous to ``integrate_law7_subsec_over_eout`` for E'.
    The LAW=7 unit-base reconstruction has

      * genuine kinks at each tabulated mu knot of either bracketing
        Ein slice (the active table pair changes and the mu-interp
        weights reset), and
      * intra-segment sub-kinks whenever the target E' causes the
        tab1 argument ``x1low + xslope(mu) * x1range`` to cross a
        tabulated E' knot in one of the two per-slice tables.

    Segmenting on the mu-knot union and applying composite Simpson
    inside each segment gets both:

    * Segment edges land on the primary kinks, avoiding the numerical
      noise of a uniform mesh spanning them.
    * Simpson within a segment resolves the smooth-but-not-polynomial
      integrand (rational in mu, plus tab1 sub-kinks) via mesh
      density, with a two-level check that upgrades unconverged
      segments to a finer mesh.

    Measured on JEFF-4.0 H-2 (n,2n) MT16, the resulting max rel
    error vs a converged adaptive-Simpson reference is < 1e-5;
    the general-purpose adaptive Simpson (initial_n=81 max_iter=3)
    reaches ~4e-3 on the same integrand.

    Parameters
    ----------
    endf_dict, mt, subsec_num : the endf dict and section pointers.
    energies_in : 1D array of incident energies (eV).
    energies_out : 1D array of outgoing-energy targets (eV).
    to_lab : accepted for signature compatibility; LAW=7 is always
        stored in the LAB frame.

    Returns
    -------
    ndarray of shape ``(len(energies_in), len(energies_out))``.
    """
    sec = endf_dict[6][mt]
    sub = sec['subsection'][subsec_num]
    if sub['LAW'] != 7:
        raise ValueError(
            f'MT={mt} subsec_num={subsec_num} is LAW={sub["LAW"]}, '
            'not LAW=7',
        )
    einc = np.asarray(energies_in, dtype=float)
    eouts = np.asarray(energies_out, dtype=float)
    q = max(get_QM(endf_dict, mt), get_QI(endf_dict, mt))
    ei_mesh = dict2array(sub['E'], dtype=float)

    n_eout = len(eouts)
    out = np.zeros((len(einc), n_eout), dtype=float)

    for i, e in enumerate(einc):
        # Outside the section's Ein range -> zero.
        if e < ei_mesh[0] or e > ei_mesh[-1]:
            continue
        eout_max = (e + q) * 1.1
        if eout_max <= 0.0:
            continue
        curidx = int(find_interval(ei_mesh, np.array([e]))[0])
        mu_mesh_e1 = dict2array(sub['mu'][curidx + 1], dtype=float)
        mu_mesh_e2 = dict2array(sub['mu'][curidx + 2], dtype=float)

        # Effective kink set on mu for this Ein: union of both
        # bracketing slices' tabulated mu points, restricted to the
        # (-1, +1) open interval, padded with the physical endpoints.
        knots = np.unique(np.concatenate([mu_mesh_e1, mu_mesh_e2]))
        knots = knots[(knots > -1.0) & (knots < 1.0)]
        seg_edges = np.concatenate([[-1.0], knots, [1.0]])
        seg_lo = seg_edges[:-1]
        seg_hi = seg_edges[1:]
        n_seg = seg_edges.shape[0] - 1

        # Fine per-segment Simpson mesh (n=33 samples, 32 intervals).
        # Also grab the coarse (n=17) mesh implicitly via decimation
        # for the Richardson-style convergence check.
        n_fine = _LAW7_MU_SEG_N_FINE
        theta_fine = np.linspace(0.0, 1.0, n_fine)
        # sample_pts shape (n_seg, n_fine): each row is one segment's
        # uniform mesh.
        sample_pts_fine = (
            seg_lo[:, None] + (seg_hi - seg_lo)[:, None] * theta_fine[None, :]
        )
        mus_flat = sample_pts_fine.reshape(-1)

        dist = _subsec.get_dist2d_from_subsec_law7(
            endf_dict, mt, subsec_num,
            np.array([e], dtype=float),
            eouts,
            mus_flat,
            True,
        )
        # dist shape (1, n_eout, n_seg * n_fine).
        f_fine = dist[0].reshape(n_eout, n_seg, n_fine)

        # Composite Simpson per segment.
        h_fine = (seg_hi - seg_lo) / (n_fine - 1)   # (n_seg,)
        integ_fine = _simpson_1d(f_fine, 1.0) * h_fine[None, :]
        # Coarse check: decimate to every second sample (n_coarse=17).
        f_coarse = f_fine[:, :, ::2]
        h_coarse = 2 * h_fine
        integ_coarse = _simpson_1d(f_coarse, 1.0) * h_coarse[None, :]

        # Per-segment convergence: refine segments where the two
        # levels disagree by more than RTOL relative to the max fine
        # integral over eout for that segment.
        seg_max = np.abs(integ_fine).max(axis=0)   # (n_seg,)
        seg_scale = np.maximum(seg_max, 1e-30)
        seg_diff = np.abs(integ_fine - integ_coarse).max(axis=0)
        need_refine = seg_diff > _LAW7_MU_SEG_RTOL * seg_scale

        if np.any(need_refine):
            refine_idcs = np.where(need_refine)[0]
            n_ref = _LAW7_MU_SEG_N_REFINED
            theta_ref = np.linspace(0.0, 1.0, n_ref)
            ref_lo = seg_lo[refine_idcs]
            ref_hi = seg_hi[refine_idcs]
            sample_pts_ref = (
                ref_lo[:, None] + (ref_hi - ref_lo)[:, None] * theta_ref[None, :]
            )
            mus_ref = sample_pts_ref.reshape(-1)
            dist_ref = _subsec.get_dist2d_from_subsec_law7(
                endf_dict, mt, subsec_num,
                np.array([e], dtype=float),
                eouts,
                mus_ref,
                True,
            )
            f_ref = dist_ref[0].reshape(
                n_eout, len(refine_idcs), n_ref,
            )
            h_ref = (ref_hi - ref_lo) / (n_ref - 1)
            integ_ref = _simpson_1d(f_ref, 1.0) * h_ref[None, :]
            integ_fine[:, refine_idcs] = integ_ref

        out[i, :] = integ_fine.sum(axis=1)

    return out
