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

INT>=3 warning (issue #71)
--------------------------
When any bracketing table's E' interpolation law is `INT>=3`
(log-based) and a caller has opened `collect_law7_log_errors()`,
the integrator adds a per-segment error estimate on top of the
midpoint integral (two extra evaluations per segment at
`mid +- w/4`, second-difference `f''` estimate, `w^3/24 * |f''|`
per segment) and records it into the context-scoped accumulator.
The top-level API (`get_particle_production_dxs_dmu` in
`endf_userpy.quantities`) opens the context around its dispatch
and emits one summary `UserWarning` at the end of the call. On
INT=1/2 sections (100 % of the corpus surveyed to date) the code
path is inactive and no extra evaluations happen.
"""
import contextlib
import contextvars
import warnings
import numpy as np

from ..primitives.helpers import dict2array, find_interval
from ..primitives.properties import get_QM, get_QI
from . import mf6_interpretation_subsecs as _subsec


# Per-query accumulator for LAW=7 INT>=3 integration errors. Set
# by `collect_law7_log_errors()`; read by `integrate_law7_subsec_
# over_eout` to decide whether to do the extra work of estimating
# `f''` on each segment. Value is either None (no active context;
# skip the estimate) or an `Law7LogErrorAccumulator` instance.
_law7_log_error_accum = contextvars.ContextVar(
    '_law7_log_error_accum', default=None,
)


class Law7LogErrorAccumulator:
    """Per-query record of LAW=7 sections that hit log-based
    `INT>=3` E' interpolation, together with the computed
    integration-error estimate. Consumed by
    `collect_law7_log_errors()` on exit to emit one summary
    UserWarning for the query. Filled by
    `integrate_law7_subsec_over_eout`.
    """

    __slots__ = ('records',)

    def __init__(self):
        # {(mt, subsec_num): (int_laws_seen_set, max_abs_err, peak_val)}
        self.records = {}

    def record(self, mt, subsec_num, int_laws, max_abs_err, peak_val):
        key = (mt, subsec_num)
        prev = self.records.get(key)
        if prev is None:
            self.records[key] = (set(int_laws), max_abs_err, peak_val)
        else:
            prev_laws, prev_err, prev_peak = prev
            prev_laws.update(int_laws)
            self.records[key] = (
                prev_laws,
                max(prev_err, max_abs_err),
                max(prev_peak, peak_val),
            )

    def has_records(self):
        return bool(self.records)

    def format_summary(self):
        parts = []
        max_abs = 0.0
        max_rel = 0.0
        for (mt, subsec), (laws, err, peak) in sorted(self.records.items()):
            laws_str = ','.join(str(x) for x in sorted(laws))
            rel = err / peak if peak > 0 else 0.0
            parts.append(
                f'MT={mt} subsec={subsec} INT={{{laws_str}}} '
                f'|err|<={err:.3e} ({rel*100:.3f}% of peak)'
            )
            max_abs = max(max_abs, err)
            max_rel = max(max_rel, rel)
        return (
            f'MF6 LAW=7 knot-aware integrator saw log-based E\' '
            f'interpolation on {len(self.records)} subsection(s). The '
            f'midpoint rule is exact for INT=1/2 but has O(h^3) '
            f'per-segment error on log-based interpolants. '
            f'Estimated max absolute integration error this call: '
            f'{max_abs:.3e} ({max_rel*100:.3f}% of peak). Details: '
            + '; '.join(parts) + '. '
            'For sub-permille accuracy, pass explicit E\' grids or '
            'use scipy.integrate.quad(points=knots, ...) with the '
            'section\'s tabulated E\' knots as breakpoints. See '
            'issue #71.'
        )


@contextlib.contextmanager
def collect_law7_log_errors():
    """Context manager that turns on the per-segment error estimate
    inside the knot-aware LAW=7 integrator for any `INT>=3` cell it
    encounters, aggregates the estimates across every `(mt, subsec)`
    hit during the block, and emits one summary `UserWarning` on
    exit if anything was recorded.

    Opened by the top-level `get_particle_production_dxs_dmu` in
    `endf_userpy.quantities`. Also usable directly when calling the
    knot-aware integrator outside the top-level API. On INT=1/2
    sections (100 % of the corpus surveyed to date) the block is
    entered and exited with zero records; no warning is emitted.
    """
    accum = Law7LogErrorAccumulator()
    token = _law7_log_error_accum.set(accum)
    try:
        yield accum
    finally:
        _law7_log_error_accum.reset(token)
        if accum.has_records():
            warnings.warn(accum.format_summary(), UserWarning, stacklevel=2)


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

    # Optional per-segment integration-error estimate for cells with
    # log-based (INT>=3) E' interpolation on any bracketing table
    # (issue #71). Only fires when a caller has opened
    # `collect_law7_log_errors()`; on INT=1/2 sections the code path
    # never triggers because `_check_log_int` returns False.
    err_accum = _law7_log_error_accum.get()
    total_max_abs_err = 0.0
    total_peak_val = 0.0
    log_int_laws_seen = set()

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

            # Cell-level INT>=3 detection: only kick in the error
            # estimator when a table on this cell has a log-based
            # E' interpolation AND the caller is inside
            # `collect_law7_log_errors()`. Records the actual INT
            # values seen so the summary warning can name them.
            cell_log_laws = ()
            if err_accum is not None:
                cell_log_laws = tuple(sorted(
                    int(v) for v in set().union(
                        (int(x) for x in tables_e1[j1 + 1]['INT']),
                        (int(x) for x in tables_e1[j1 + 2]['INT']),
                        (int(x) for x in tables_e2[j2 + 1]['INT']),
                        (int(x) for x in tables_e2[j2 + 2]['INT']),
                    ) if int(v) >= 3
                ))

            if cell_log_laws:
                # Estimate per-segment error via second-difference of
                # `f` at `mid`, `mid +- w/4`. Midpoint rule error on
                # a smooth per-segment integrand is `(w^3/24)*f''(xi)`;
                # substituting `f''(mid) ~ (f(mid - h) - 2 f(mid) +
                # f(mid + h)) / h^2` with `h = w/4` gives per-segment
                # bound `(2 w / 3) * |Delta^2 f|`. Skip zero-width
                # segments defensively.
                h = widths / 4.0
                # Pack the three sample sets into one dist2d call to
                # keep the dispatch cost proportional to (n_seg * 3)
                # rather than three separate section walks.
                lo_pts = mids - h
                hi_pts = mids + h
                all_pts = np.concatenate([mids, lo_pts, hi_pts])
                dist = _subsec.get_dist2d_from_subsec_law7(
                    endf_dict, mt, subsec_num,
                    np.array([e], dtype=float),
                    all_pts,
                    np.array([u], dtype=float),
                    True,
                )
                fv = dist[0, :, 0]
                n = len(mids)
                f_mid = fv[:n]
                f_lo = fv[n:2 * n]
                f_hi = fv[2 * n:]
                out[i, j] = float(np.sum(widths * f_mid))
                d2 = np.abs(f_lo - 2.0 * f_mid + f_hi)
                seg_err = (2.0 * widths / 3.0) * d2
                cell_err = float(np.sum(seg_err))
                if cell_err > total_max_abs_err:
                    total_max_abs_err = cell_err
                log_int_laws_seen.update(cell_log_laws)
            else:
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
            if abs(out[i, j]) > total_peak_val:
                total_peak_val = abs(out[i, j])

    if err_accum is not None and log_int_laws_seen:
        err_accum.record(
            mt, subsec_num, log_int_laws_seen,
            total_max_abs_err, total_peak_val,
        )

    return out
