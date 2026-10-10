"""MF6 LAW=7 integrators for traced incident energies (issue #220).

Under ``jax.jit`` over the incident energies ``E_in`` is a tracer, so
the per-``E_in`` Python control flow of :mod:`mf6_law7_integrals`
(Ein-panel lookup, mu-bracket lookup, effective-knot mesh union)
cannot run on it. Everything those integrators derive from the Ein
*panel* but not from ``E_in`` itself is concrete, though, and a
subsection has only tens of panels. So the host precomputes it per
panel into padded arrays, and the traced body (one ``xp.map``
iteration per ``E_in``) gathers the row of the panel ``E_in`` falls in:

* over E' (:func:`integrate_law7_subsec_over_eout_traced`): per
  ``(panel, mu)`` the four bracketing tables (as padded table rows)
  and the unit-base slice geometry. The effective-knot mesh is built
  in the traced body, because ``eout_max`` cuts it: the padded knots
  are sorted instead of ``np.unique``-d, and the duplicates become
  zero-width segments that contribute exactly zero.
* over mu (:func:`integrate_law7_subsec_over_mu_traced`): the LAW=7
  amplitude blends the two bracketing Ein slices with an outer law
  (INT=1..3) that is linear in the slice amplitudes, and the slice
  amplitudes do not depend on ``E_in``. So the per-segment Simpson
  integrals of both slices on the panel's mu segments (coarse, fine
  and refined levels) are precomputed on the host, and the traced
  body blends them and applies numpy's per-segment refinement rule.

Both reproduce the numpy integrators to rounding. File-side jax
leaves (``f`` / ``Ep`` swapped for jax arrays) are not supported on
this path; the tables are read on the host.
"""
import numpy as np

from ..primitives.helpers import convert_interp_repr, dict2array, find_interval
from ..primitives.interpolation import integrate_tab1_panels
from ..primitives.properties import get_QM, get_QI
from . import mf6_interpretation_subsecs as _subsec
from . import mf6_law7_integrals as _law7


def _panel_index(ei_mesh, e, xp):
    """``find_interval`` for one traced energy inside the mesh."""
    n_panels = ei_mesh.shape[0] - 1
    k = xp.searchsorted(xp.asarray(ei_mesh), e, side='right') - 1
    return xp.clip(k, 0, n_panels - 1)


def _padded_tables(sub):
    """All ``(Ein slice, mu slot)`` E' tables of the subsection as
    padded rows: ``ep`` / ``f`` replicate their last value, the
    per-panel INT codes pad with 2. Padded panels have zero width and
    are never active. Returns ``(ids, ep, f, codes)`` with ``ids``
    mapping the 1-based dict keys ``(s, j)`` to a row."""
    ids, eps, fs, codes = {}, [], [], []
    for s in sorted(sub['table']):
        for j in sorted(sub['table'][s]):
            tab = sub['table'][s][j]
            ep = np.asarray(tab['Ep'], dtype=float)
            ids[(s, j)] = len(eps)
            eps.append(ep)
            fs.append(np.asarray(tab['f'], dtype=float))
            codes.append(_law7._tab_panel_int(tab['INT'], tab['NBT'], ep.shape[0]))
    n_pad = max(2, max(ep.shape[0] for ep in eps))
    ep_rows = np.empty((len(eps), n_pad))
    f_rows = np.empty((len(eps), n_pad))
    code_rows = np.full((len(eps), n_pad - 1), 2, dtype=int)
    for r, (ep, f, c) in enumerate(zip(eps, fs, codes)):
        ep_rows[r, :ep.shape[0]] = ep
        ep_rows[r, ep.shape[0]:] = ep[-1]
        f_rows[r, :f.shape[0]] = f
        f_rows[r, f.shape[0]:] = f[-1]
        if c.shape[0] == ep.shape[0] - 1:
            code_rows[r, :c.shape[0]] = c
    return ids, ep_rows, f_rows, code_rows


def _segment_areas_rows(ep, f, codes, xi_a, xi_b, xp):
    """:func:`mf6_law7_integrals._table_segment_areas_vec` with one
    padded table row per leading index: ``ep`` / ``f`` ``(..., P)``,
    ``codes`` ``(..., P - 1)``, segment bounds ``(..., n_seg)``."""
    lo = ep[..., :1]
    hi = ep[..., -1:]
    xa = xp.clip(xi_a, lo, hi)
    xb = xp.clip(xi_b, lo, hi)
    active = (xi_b > xi_a) & (xi_a < hi) & (xi_b > lo)
    mid = 0.5 * (xa + xb)
    # searchsorted(row, mid, 'right') - 1 for every row at once
    panels = xp.sum(ep[..., None, :] <= mid[..., :, None], axis=-1) - 1
    panels = xp.clip(panels, 0, ep.shape[-1] - 2)
    x0 = xp.take_along_axis(ep, panels, axis=-1)
    x1 = xp.take_along_axis(ep, panels + 1, axis=-1)
    y0 = xp.take_along_axis(f, panels, axis=-1)
    y1 = xp.take_along_axis(f, panels + 1, axis=-1)
    c = xp.take_along_axis(codes, panels, axis=-1)
    areas = integrate_tab1_panels(x0, x1, y0, y1, xa, xb, c, xp=xp)
    return xp.where(active, areas, 0.0)


def _slice_geometry(mus, mu_mesh, mu_int, tab_ids, s, ep_rows):
    """Concrete unit-base geometry of one Ein slice (dict key ``s``)
    for every mu target, as in ``_ub_slice_geometry`` /
    ``_project_ub_knots``. Returns ``(valid, tids, geom)`` with
    ``tids`` ``(n_mu, 2)`` and ``geom`` a dict of ``(n_mu,)`` arrays."""
    valid = (mus >= mu_mesh[0]) & (mus <= mu_mesh[-1])
    j = np.zeros(mus.shape[0], dtype=int)
    if np.any(valid):
        j[valid] = find_interval(mu_mesh, mus[valid])
    mu1 = mu_mesh[j]
    mu2 = mu_mesh[j + 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        yslope = np.where(mu2 == mu1, 0.0, (mus - mu1) / (mu2 - mu1))
    tids = np.stack([
        np.array([tab_ids[(s, jj + 1)] for jj in j]),
        np.array([tab_ids[(s, jj + 2)] for jj in j]),
    ], axis=-1)
    x1low = ep_rows[tids[:, 0], 0]
    x1high = ep_rows[tids[:, 0], -1]
    x2low = ep_rows[tids[:, 1], 0]
    x2high = ep_rows[tids[:, 1], -1]
    x1range = x1high - x1low
    x2range = x2high - x2low
    xlow = x1low + yslope * (x2low - x1low)
    xhigh = x1high + yslope * (x2high - x1high)
    geom = {
        # Mu-blend base law INT=1 keeps the lower table only; the
        # geometry itself uses the unmodified yslope.
        'yslope': np.where(mu_int[j + 1] % 10 == 1, 0.0, yslope),
        'xlow': xlow, 'xrange': xhigh - xlow,
        'x1low': x1low, 'x1range': x1range,
        'x2low': x2low, 'x2range': x2range,
    }
    return valid, tids, geom


def integrate_law7_subsec_over_eout_traced(
    endf_dict, mt, subsec_num, energies_in, angle_cosines_out, xp,
):
    """:func:`mf6_law7_integrals.integrate_law7_subsec_over_eout` for
    traced ``energies_in`` (concrete ``angle_cosines_out``)."""
    sub = endf_dict[6][mt]['subsection'][subsec_num]
    mus = np.asarray(angle_cosines_out, dtype=float).reshape(-1)
    q = max(get_QM(endf_dict, mt), get_QI(endf_dict, mt))
    ei_mesh = dict2array(sub['E'], dtype=float)
    n_panels = ei_mesh.shape[0] - 1
    _law7._check_supported_int(
        sub['E_interpol']['INT'], sub['E_interpol']['NBT'],
        axis_name='outer Ein',
    )
    e_int = convert_interp_repr(
        np.asarray(sub['E_interpol']['INT'], dtype=int),
        np.asarray(sub['E_interpol']['NBT'], dtype=int),
    )
    tab_ids, ep_rows, f_rows, code_rows = _padded_tables(sub)

    # Per (panel, mu) concrete data, slices e1 (key k + 1) and e2.
    valid = np.empty((n_panels, mus.shape[0]), dtype=bool)
    tids = np.empty((n_panels, mus.shape[0], 4), dtype=int)
    geom_keys = ('yslope', 'xlow', 'xrange', 'x1low', 'x1range',
                 'x2low', 'x2range')
    geom = {g: np.empty((n_panels, mus.shape[0], 2)) for g in geom_keys}
    for k in range(n_panels):
        for side, s in enumerate((k + 1, k + 2)):
            mu_interpol = sub['mu_interpol'][s]
            _law7._check_supported_int(
                mu_interpol['INT'], mu_interpol['NBT'],
                axis_name=f'mu (Ein slice {side + 1})',
            )
            mu_int = convert_interp_repr(
                np.asarray(mu_interpol['INT'], dtype=int),
                np.asarray(mu_interpol['NBT'], dtype=int),
            )
            mu_mesh = dict2array(sub['mu'][s], dtype=float)
            v, t, g = _slice_geometry(mus, mu_mesh, mu_int, tab_ids, s, ep_rows)
            valid[k] = v if side == 0 else (valid[k] & v)
            tids[k, :, 2 * side:2 * side + 2] = t
            for name in geom_keys:
                geom[name][k, :, side] = g[name]
    lei = e_int[1:] % 10                                   # (n_panels,)

    ei_xp = xp.asarray(ei_mesh)
    lei_xp = xp.asarray(lei)
    valid_xp = xp.asarray(valid)
    tids_xp = xp.asarray(tids)
    geom_xp = {name: xp.asarray(a) for name, a in geom.items()}
    ep_xp = xp.asarray(ep_rows)
    f_xp = xp.asarray(f_rows)
    code_xp = xp.asarray(code_rows)

    def one_ein(e):
        k = _panel_index(ei_mesh, e, xp)
        eout_max = (e + q) * 1.1
        ok = (e >= ei_mesh[0]) & (e <= ei_mesh[-1]) & (eout_max > 0.0)
        e1 = ei_xp[k]
        e2 = ei_xp[k + 1]
        lin = lei_xp[k] != 1
        de = xp.where(e2 != e1, e2 - e1, 1.0)
        eslope_2 = xp.where(lin & (e2 != e1), (e - e1) / de, 0.0)
        eslope_1 = xp.where(lin, 1.0 - eslope_2, 1.0)

        t = tids_xp[k]                                     # (n_mu, 4)
        ep = ep_xp[t]                                      # (n_mu, 4, P)
        f = f_xp[t]
        codes = code_xp[t]
        g = {name: a[k] for name, a in geom_xp.items()}    # (n_mu, 2)

        # Effective knots of both slices (``_project_ub_knots``).
        knots = []
        for side in (0, 1):
            xlow = g['xlow'][:, side:side + 1]
            xrange_ = g['xrange'][:, side:side + 1]
            for tab, low, rng in ((2 * side, 'x1low', 'x1range'),
                                  (2 * side + 1, 'x2low', 'x2range')):
                x_low = g[low][:, side:side + 1]
                x_rng = g[rng][:, side:side + 1]
                rng_safe = xp.where(x_rng > 0.0, x_rng, 1.0)
                eff = xlow + (ep[:, tab, :] - x_low) * (xrange_ / rng_safe)
                eff = xp.where((x_rng > 0.0) & (xrange_ > 0.0), eff, xlow)
                knots.append(eff)
        knots = xp.sort(xp.concatenate(knots, axis=-1), axis=-1)
        knots = xp.clip(knots, 0.0, eout_max)
        n_mu = knots.shape[0]
        mesh = xp.concatenate([
            xp.zeros((n_mu, 1)), knots,
            xp.broadcast_to(eout_max, (n_mu, 1)),
        ], axis=-1)
        seg_a = mesh[:, :-1]
        seg_b = mesh[:, 1:]

        def slice_areas(side):
            xlow = g['xlow'][:, side:side + 1]
            xrange_ = g['xrange'][:, side:side + 1]
            xr_safe = xp.where(xrange_ > 0.0, xrange_, 1.0)
            a_c = xp.maximum(seg_a, xlow)
            b_c = xp.minimum(seg_b, xlow + xrange_)
            active = b_c > a_c
            a_c = xp.where(active, a_c, xlow)
            b_c = xp.where(active, b_c, xlow)
            areas = []
            for tab, low, rng in ((2 * side, 'x1low', 'x1range'),
                                  (2 * side + 1, 'x2low', 'x2range')):
                x_low = g[low][:, side:side + 1]
                x_rng = g[rng][:, side:side + 1]
                xi_a = x_low + (a_c - xlow) / xr_safe * x_rng
                xi_b = x_low + (b_c - xlow) / xr_safe * x_rng
                areas.append(_segment_areas_rows(
                    ep[:, tab, :], f[:, tab, :], codes[:, tab, :],
                    xi_a, xi_b, xp,
                ))
            yslope = g['yslope'][:, side:side + 1]
            res = (1.0 - yslope) * areas[0] + yslope * areas[1]
            return xp.where(xrange_ > 0.0, res, 0.0)

        cells = xp.sum(eslope_1 * slice_areas(0) + eslope_2 * slice_areas(1),
                       axis=-1)
        return xp.where(valid_xp[k] & ok, cells, 0.0)

    e_in = xp.asarray(energies_in, dtype=xp.float64).reshape(-1)
    return xp.map(one_ein, e_in)


def _slice_segment_integrals(sub, s, seg_lo, seg_hi, eouts, eout_ok):
    """Per-segment Simpson integrals over mu of Ein slice ``s`` on the
    segments ``[seg_lo, seg_hi]``: ``(fine, coarse, refined)``, each
    ``(n_eout, n_seg)``, with the sample points, rules and step sizes
    of :func:`mf6_law7_integrals.integrate_law7_subsec_over_mu`."""
    n_seg = seg_lo.shape[0]
    out = []
    for n_pts in (_law7._LAW7_MU_SEG_N_FINE, _law7._LAW7_MU_SEG_N_REFINED):
        theta = np.linspace(0.0, 1.0, n_pts)
        pts = seg_lo[:, None] + (seg_hi - seg_lo)[:, None] * theta[None, :]
        vals = _subsec._law7_eval_mu_panel(
            sub, s, pts.reshape(-1), eouts, _law7._resolve_xp(None),
        )                                                  # (n_mu, n_eout)
        vals = np.where(eout_ok[None, :], vals, 0.0)
        out.append(vals.T.reshape(eouts.shape[0], n_seg, n_pts))
    f_fine, f_ref = out
    h_fine = (seg_hi - seg_lo) / (_law7._LAW7_MU_SEG_N_FINE - 1)
    h_ref = (seg_hi - seg_lo) / (_law7._LAW7_MU_SEG_N_REFINED - 1)
    fine = _law7._simpson_1d(f_fine, 1.0) * h_fine[None, :]
    coarse = _law7._simpson_1d(f_fine[:, :, ::2], 1.0) * (2 * h_fine)[None, :]
    refined = _law7._simpson_1d(f_ref, 1.0) * h_ref[None, :]
    return fine, coarse, refined


def integrate_law7_subsec_over_mu_traced(
    endf_dict, mt, subsec_num, energies_in, energies_out, xp,
):
    """:func:`mf6_law7_integrals.integrate_law7_subsec_over_mu` for
    traced ``energies_in`` (concrete ``energies_out``)."""
    sub = endf_dict[6][mt]['subsection'][subsec_num]
    eouts = np.asarray(energies_out, dtype=float).reshape(-1)
    q = max(get_QM(endf_dict, mt), get_QI(endf_dict, mt))
    ei_mesh = dict2array(sub['E'], dtype=float)
    n_panels = ei_mesh.shape[0] - 1
    # Outer law as in ``get_dist2d_from_subsec_law7`` (mod 10 of the
    # INT at the panel's lower point).
    ei_interp = convert_interp_repr(
        np.asarray(sub['E_interpol']['INT'], dtype=int),
        np.asarray(sub['E_interpol']['NBT'], dtype=int),
    )
    lei = ei_interp[:-1] % 10
    if np.any((lei < 1) | (lei > 3)):
        raise NotImplementedError(
            'MF6 LAW=7 mu-integration under jax.jit over the incident '
            'energies supports outer Ein INT=1..3 only (the log-y '
            f'laws do not blend linearly); got {sorted(set(lei))}.'
        )
    # ``pad_outside_dist2d_values`` zeroes E' < 1e-20.
    eout_ok = eouts >= 1e-20

    panel_segs = []
    cache = {}
    for k in range(n_panels):
        mu1 = dict2array(sub['mu'][k + 1], dtype=float)
        mu2 = dict2array(sub['mu'][k + 2], dtype=float)
        knots = np.unique(np.concatenate([mu1, mu2]))
        knots = knots[(knots > -1.0) & (knots < 1.0)]
        edges = np.concatenate([[-1.0], knots, [1.0]])
        ints = []
        for s in (k + 1, k + 2):
            key = (s, edges.tobytes())
            if key not in cache:
                cache[key] = _slice_segment_integrals(
                    sub, s, edges[:-1], edges[1:], eouts, eout_ok,
                )
            ints.append(cache[key])
        panel_segs.append(ints)
    max_seg = max(ints[0][0].shape[1] for ints in panel_segs)
    # (n_panels, 2 slices, 3 levels, n_eout, max_seg); zero-padded
    # segments never refine and add zero.
    tab = np.zeros((n_panels, 2, 3, eouts.shape[0], max_seg))
    for k, ints in enumerate(panel_segs):
        for side in (0, 1):
            for lev in range(3):
                a = ints[side][lev]
                tab[k, side, lev, :, :a.shape[1]] = a

    ei_xp = xp.asarray(ei_mesh)
    lei_xp = xp.asarray(lei)
    tab_xp = xp.asarray(tab)
    small = 1.0e-38

    def one_ein(e):
        k = _panel_index(ei_mesh, e, xp)
        eout_max = (e + q) * 1.1
        ok = (e >= ei_mesh[0]) & (e <= ei_mesh[-1]) & (eout_max > 0.0)
        e1 = ei_xp[k]
        e2 = ei_xp[k + 1]
        t = tab_xp[k]                                      # (2, 3, n_eout, n_seg)
        a, b = t[0], t[1]
        # ``_interp_two_point_columns`` per outer law
        lin = a + (e - e1) * (b - a) / (e2 - e1)
        e1s = xp.where(e1 != 0.0, e1, small)
        log = a + xp.log(e / e1s) * (b - a) / xp.log(e2 / e1s)
        law = lei_xp[k]
        levels = xp.where(law == 1, a, xp.where(law == 2, lin, log))
        fine, coarse, refined = levels[0], levels[1], levels[2]
        seg_scale = xp.maximum(xp.max(xp.abs(fine), axis=0), 1e-30)
        seg_diff = xp.max(xp.abs(fine - coarse), axis=0)
        need = seg_diff > _law7._LAW7_MU_SEG_RTOL * seg_scale
        row = xp.sum(xp.where(need[None, :], refined, fine), axis=1)
        return xp.where(ok, row, 0.0)

    e_in = xp.asarray(energies_in, dtype=xp.float64).reshape(-1)
    return xp.map(one_ein, e_in)
