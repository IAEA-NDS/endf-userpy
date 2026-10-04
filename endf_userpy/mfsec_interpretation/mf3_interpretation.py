import warnings
import numpy as np
from ..primitives.helpers import treat_duplicates
from ..primitives.interpolation import interp_tab1
from ..primitives.properties import (
    get_reaction_string_for_mt,
)


_ABOVE_RANGE_POLICIES = ('warn_nan', 'nan', 'warn_zero', 'zero', 'raise')
_RESONANCE_RANGE_POLICIES = ('warn', 'warn_nan', 'nan', 'raise')


def get_resolved_resonance_ranges(endf_dict):
    """Return list of `(EL, EH)` tuples for every resolved-resonance
    (LRU=1) range in the file, or `[]` if there is no MF2/MT151 or no
    LRU=1 range.

    Unresolved-resonance ranges (LRU=2) are skipped: MF3 in the URR
    is the average cross section, not a subtractive background, so
    it does not carry the negative-value or missing-reconstruction
    problem the LRU=1 case does.

    Files without MF2 at all (photonuclear, some light-nuclide
    evaluations that go straight to point-wise MF3) return `[]` and
    the resonance-range policy layer becomes a no-op on them.
    """
    if 2 not in endf_dict or 151 not in endf_dict[2]:
        return []
    mf2 = endf_dict[2][151]
    ranges = []
    for iso in mf2.get('isotope', {}).values():
        for rng in iso.get('range', {}).values():
            if rng.get('LRU') != 1:
                continue
            ranges.append((float(rng['EL']), float(rng['EH'])))
    return ranges


def _handle_resonance_range(policy, mt, einc_arr, rrr_ranges, hits=None):
    """Common resonance-range policy implementation shared by every
    XS reader. Returns `(in_rrr_mask, fill_value)`: `in_rrr_mask` is
    True at Ein positions inside any LRU=1 range; `fill_value` is
    `np.nan` for the nan policies and `None` (meaning don't overwrite)
    for the pure-warn / passthrough policies.

    Callers apply the mask themselves:

        if in_rrr_mask.any() and fill_value is not None:
            xs = np.where(in_rrr_mask, fill_value, xs)
    """
    if not rrr_ranges:
        # No LRU=1 ranges in the file: nothing to check.
        return None, None
    einc_arr = np.asarray(einc_arr, dtype=float)
    in_rrr_mask = np.zeros_like(einc_arr, dtype=bool)
    for el, eh in rrr_ranges:
        # Half-open [EL, EH): consistent with the resonance-composition
        # range convention (issue #149). At E == EH the URR (or MF3
        # above URR) owns the value, so a query there is not "inside
        # the RRR" and should not fire the RRR-warning/nan policy.
        in_rrr_mask |= (einc_arr >= el) & (einc_arr < eh)
    n_in = int(in_rrr_mask.sum())
    if n_in == 0:
        return None, None
    # Union bounds for the summary warning
    el_union = min(el for el, _ in rrr_ranges)
    eh_union = max(eh for _, eh in rrr_ranges)
    if policy == 'raise':
        raise ValueError(
            f'{n_in} of {len(einc_arr)} incident energies for MT={mt} '
            f'fall inside the file\'s resolved-resonance region '
            f'[{el_union:.6g}, {eh_union:.6g}] eV. MF3 there is a '
            f'subtractive background, not the physical cross section '
            f'(this library does not perform MF2 resonance '
            f'reconstruction).'
        )
    if policy in ('warn', 'warn_nan'):
        if hits is not None:
            prev = hits.resonance_range.get(mt)
            if prev is None or n_in > prev[2]:
                hits.resonance_range[mt] = (
                    el_union, eh_union, n_in, len(einc_arr),
                )
        else:
            # Called outside a ctx (leaf reader used directly): fall
            # back to a per-call warning so the caller still sees a
            # signal without needing to open the context manager.
            action = 'NaN' if policy == 'warn_nan' else 'the raw MF3 background'
            warnings.warn(
                f'{n_in} of {len(einc_arr)} incident energies for '
                f'MT={mt} fall inside the resolved-resonance region '
                f'[{el_union:.6g}, {eh_union:.6g}] eV; returning '
                f'{action}. MF3 there is a subtractive background '
                f'cross section (this library does not reconstruct '
                f'MF2 resonances).',
                UserWarning, stacklevel=3,
            )
    if policy in ('warn_nan', 'nan'):
        return in_rrr_mask, np.nan
    return in_rrr_mask, None  # 'warn' / passthrough


def _handle_above_range(policy, mt, e_max, above_mask, energies_in, hits=None):
    """Common implementation of the `above_range` policy shared by
    every XS reader (see issue #28). Returns the numeric-fill value
    the caller should place at `above_mask` positions in its result
    array (`0.0` or `np.nan`), after either raising per the policy
    or recording the hit in the caller-provided ``hits``
    accumulator (issue #143).

    ``hits`` is a :class:`~endf_userpy.run_options._QueryState`
    instance threaded down by top-level
    :mod:`endf_userpy.quantities` entry points, or ``None`` for
    direct leaf callers -- in which case ``warn_*`` variants fall
    back to a per-call UserWarning so the leaf stays usable in
    isolation.
    """
    if policy not in _ABOVE_RANGE_POLICIES:
        raise ValueError(
            f"above_range must be one of {_ABOVE_RANGE_POLICIES}; "
            f"got {policy!r}"
        )
    n_above = int(above_mask.sum())
    if n_above == 0:
        return 0.0  # unused; no positions to fill
    if policy == 'raise':
        raise ValueError(
            f'{n_above} of {len(energies_in)} incident energies exceed '
            f'MT={mt} upper mesh energy ({e_max:.6g} eV); cross section '
            f'is undefined above the evaluation range.'
        )
    if policy in ('warn_nan', 'warn_zero'):
        if hits is not None:
            # Inside a top-level call: record for the summary
            # UserWarning that _emit_summary_warnings drains on
            # the way back out. Keep the largest n_above per MT if
            # the same MT is queried more than once.
            prev = hits.above_range.get(mt)
            if prev is None or n_above > prev[1]:
                hits.above_range[mt] = (e_max, n_above, len(energies_in))
        else:
            # Called outside a ctx (leaf reader used directly).
            # Emit a per-call warning so the caller still sees a
            # signal without needing to open the context manager.
            warnings.warn(
                f'{n_above} of {len(energies_in)} incident energies '
                f'exceed MT={mt} upper mesh energy ({e_max:.6g} eV); '
                f'returned as {"NaN" if policy == "warn_nan" else "0"}. '
                f'Cross section is undefined above the evaluation range.',
                UserWarning, stacklevel=3,
            )
    return np.nan if policy in ('warn_nan', 'nan') else 0.0


def get_nominal_incident_energies(endf_dict, mt):
    sec = endf_dict[3][mt]
    xstab = sec['xstable']
    return np.array(xstab['E'], copy=True)


def get_nominal_incident_energy_range(endf_dict, mt):
    energies = get_nominal_incident_energies(endf_dict, mt)
    return (min(energies), max(energies))


def get_incident_energies(endf_dict, mt):
    """Nominal incident-energy mesh for MT trimmed to the nonzero
    cross-section block plus one bracketing zero on each side.

    Returns an empty float ndarray if the MT's cross section is
    all-zero (some libraries carry placeholder MT entries for MF6 or
    MF8 book-keeping with a zeroed MF3): callers use empty as the
    "no meaningful incident energies" signal (issue #96).
    """
    energies = get_nominal_incident_energies(endf_dict, mt)
    treat_duplicates(energies, inplace=True)
    xs = np.array(endf_dict[3][mt]['xstable']['xs'], dtype=float)
    idcs = np.nonzero(xs)[0]
    if idcs.size == 0:
        # All-zero cross section: no energy has a physical
        # reaction rate; return empty rather than the full mesh
        # (which would misleadingly imply coverage) or raise
        # IndexError from `idcs[0]` (issue #96, second bug).
        return energies[:0].copy()
    # Bracket the nonzero block by one point on each side (so the
    # cross section is guaranteed to start/end at zero for downstream
    # interpolation). Cap by the mesh length, not the nonzero-count:
    # the pre-fix `last_idx+1 < len(idcs)` compared against the
    # number of nonzero points (issue #96, first bug) and silently
    # dropped the trailing bracketing zero whenever the nonzero block
    # extended to within one index of the mesh end.
    first_idx = max(idcs[0] - 1, 0)
    last_idx = min(idcs[-1] + 1, len(xs) - 1)
    return energies[first_idx:last_idx+1].copy()


def get_incident_energy_range(endf_dict, mt):
    """(min, max) of `get_incident_energies(mt)`.

    Raises `ValueError` if the cross section is all-zero (empty
    mesh has no min/max). Callers that need to survive all-zero MTs
    should check `len(get_incident_energies(...)) == 0` first.
    """
    eincs = get_incident_energies(endf_dict, mt)
    if eincs.size == 0:
        raise ValueError(
            f'MT={mt} has an all-zero cross section; no incident '
            f'energy range defined'
        )
    return (min(eincs), max(eincs))


def get_reaction_mts(endf_dict):
    return list(endf_dict[3].keys())


def get_reaction_mts_widened(endf_dict):
    """Union of MTs across MF3 + MF12 + MF13 + MF15.

    Wider than `get_reaction_mts` (MF3-only): includes MTs that have
    photon-production data (MF12 yields, MF13 per-photon XS, MF15
    continuous spectra) without an MF3 cross section. Typical case
    is a file that puts gamma production only on a sum-MT (e.g.
    JENDL-5 N-14 puts MF13 on MT 3 nonelastic without tabulating
    MT 3 in MF3 -- MT 3 is implicitly MT 1 - MT 2 there).

    Used by the particle-production dispatchers when the caller's
    ejectile is photon so those MTs are visited by the
    cumulative-sum iteration. Non-gamma iterations that call this
    should either check MF3 presence themselves per MT or route
    through the standard `get_reaction_mts` (MF3-only).

    Issue #130.
    """
    mts = set()
    for mf in (3, 12, 13, 15):
        mts |= set(endf_dict.get(mf, {}).keys())
    return sorted(mts)


def get_reactions(endf_dict):
    mts = get_reaction_mts(endf_dict)
    reacs = [get_reaction_string_for_mt(endf_dict, m) for m in mts] 
    return reacs


def compute_cross_section(
    endf_dict, mt, energies_in, above_range='warn_nan',
    resonance_range='warn', xp=None, _query_state=None,
):
    """Cross section for MT evaluated on `energies_in`.

    `above_range` controls what happens at incident energies above
    the file's upper Ein boundary for this MT -- ENDF-6 makes no
    claim about the cross section there. When `None` (default), the
    policy comes from the context variable set by the top-level
    `get_*` APIs in `endf_userpy.quantities`; the ambient default
    when nothing is set is ``'warn_nan'``.

    Explicit `above_range` policies:

    - ``'warn_nan'`` (default): fill above-range points with NaN and
      emit a `UserWarning` naming the MT and the exceeded limit.
      NaN propagates through downstream arithmetic so the undefined
      region is visible everywhere it matters.
    - ``'nan'``: same as above but silent.
    - ``'warn_zero'``: fill with 0 and emit a UserWarning. Matches
      pre-issue-#28 numeric behaviour while surfacing a signal.
    - ``'zero'``: fill with 0, silent. Pre-issue-#28 default.
    - ``'raise'``: raise `ValueError` and refuse to return.

    `resonance_range` controls what happens at incident energies
    inside the file's resolved-resonance region (LRU=1 in
    MF2/MT151). ENDF-6 stores MF3 there as a subtractive
    **background** cross section that must be added to the resonance
    reconstruction from MF2; this library does not reconstruct
    resonances (README "Known limitations"), so raw MF3 in the RRR
    is not the physical cross section and can be negative (issue
    #84). Default ``'warn'`` returns the raw background with a
    summary UserWarning; ``'warn_nan'`` / ``'nan'`` fill with NaN;
    ``'raise'`` errors on any RRR hit. Files without MF2 or with
    only LRU=0 / LRU=2 ranges are unaffected (no LRU=1 range
    exists to check against).

    Below the MT-specific mesh (typically sub-threshold or below
    the reaction onset) the return is always 0 regardless of
    `above_range`: `interp_tab1`'s outside-value fill uses 0 there
    and the physics interpretation of "below the file's lowest
    tabulated Ein" is almost always "physically zero" (the mesh
    starts at or below the reaction threshold).
    """
    from ..primitives import array_ns
    if xp is None:
        xp = array_ns.get_backend('numpy')
    sec = endf_dict[3][mt]
    xstab = sec['xstable']
    # Route the mesh through xp so a tracer stored at ``xstab['E'][idx]``
    # (mesh-knot autodiff, issue #169) survives into the above-range
    # mask below. ``e_max`` stays as a backend-native 0-d array (a
    # numpy scalar or a JAX scalar); ``float(...)`` on a JAX tracer
    # would raise so we do not force it.
    e_mesh = xp.asarray(xstab['E'], dtype=xp.float64)
    e_max = e_mesh.max()
    xs = interp_tab1(
        energies_in, xstab, 'E', 'xs', outside_value=0.0, xp=xp,
    )
    if xp.name == 'jax':
        # Under jax: skip the numpy-side above-range warning machinery
        # (can't safely check ``.any()`` on a tracer). Apply the
        # policy's fill via ``xp.where`` on the tracer mask. RRR
        # policy defaults to a warning fired ONCE per top-level call
        # from a context var; skip that under trace too.
        einc_xp = xp.asarray(energies_in)
        above_mask = einc_xp > e_max
        # For the warn variants under trace, we cannot inspect the
        # mask; treat as the corresponding non-warn variant.
        policy = above_range
        if policy in ('warn_nan', 'nan'):
            fill = float('nan')
            xs = xp.where(above_mask, fill, xs)
        elif policy in ('warn_zero', 'zero'):
            xs = xp.where(above_mask, 0.0, xs)
        elif policy == 'raise':
            # Under trace we can't conditionally raise; leave xs
            # untouched. Callers who want the raise semantics should
            # extract concrete E outside the trace.
            pass
        rrr_ranges = get_resolved_resonance_ranges(endf_dict)
        if rrr_ranges and resonance_range in ('warn_nan', 'nan'):
            in_rrr = xp.zeros_like(einc_xp, dtype=bool)
            for lo, hi in rrr_ranges:
                in_rrr = in_rrr | ((einc_xp >= lo) & (einc_xp <= hi))
            overwrite_mask = in_rrr & (~above_mask)
            xs = xp.where(overwrite_mask, float('nan'), xs)
        return xs
    einc_arr = np.asarray(energies_in, dtype=float)
    above_mask = einc_arr > e_max
    fill_value = _handle_above_range(
        above_range, mt, e_max, above_mask, einc_arr, hits=_query_state,
    )
    if above_mask.any() and fill_value != 0.0:
        xs = np.where(above_mask, fill_value, xs)
    # Resonance-range policy (issue #84): applied after the
    # above-range fill so that Ein above the file's mesh keeps
    # the NaN/0 above-range fill even if it happened to also be
    # inside the RRR (in practice above_mask and RRR are disjoint;
    # the guard is defensive).
    rrr_ranges = get_resolved_resonance_ranges(endf_dict)
    if rrr_ranges:
        in_rrr_mask, rrr_fill = _handle_resonance_range(
            resonance_range, mt, einc_arr, rrr_ranges, hits=_query_state,
        )
        if in_rrr_mask is not None and rrr_fill is not None:
            # Only overwrite positions that are NOT already
            # above-range (above_range takes precedence).
            overwrite_mask = in_rrr_mask & ~above_mask
            if overwrite_mask.any():
                xs = np.where(overwrite_mask, rrr_fill, xs)
    return xs


def compute_cross_section_agnostic(
    endf_dict, mt, energies_in, xp, outside_value=0.0, side='right',
):
    """Backend-agnostic MF3 cross-section reconstruction.

    Same physics as :func:`compute_cross_section` (TAB1 interp)
    minus the above_range / resonance_range policies. Runs
    unchanged on the numpy, JAX, and numba backends via the
    :mod:`endf_userpy.primitives.array_ns` adapter; enables MF3
    to participate in sensitivity workflows (``jax.grad`` through
    tabulated cross-section values) and in composed pipelines with
    the array-agnostic MF2 reconstructions.

    The policy machinery (above_range, resonance_range, warnings,
    context-variable defaults) is intentionally not replicated
    here -- those are user-facing decisions that don't belong in
    a differentiable / jit-friendly numerical kernel. Callers
    that want them should stay on :func:`compute_cross_section`;
    callers that want backend-agnostic reconstruction (and are
    happy to enforce range policies themselves) use this.

    Parameters
    ----------
    endf_dict : dict
        Parsed ENDF-6 dict from ``endf_parserpy``.
    mt : int
        MT number to reconstruct. Raises ``KeyError`` if MF3/MT
        is not present.
    energies_in : array_like
        Incident-neutron energies. Cast to the backend's float64
        via ``xp.asarray`` inside :func:`primitives.tab1.interp`.
    xp : backend
        As returned by :func:`endf_userpy.primitives.array_ns.get_backend`.
    outside_value : float, optional
        Fill for query energies outside the tabulated mesh.
        Default 0.0 matches :func:`compute_cross_section` below
        the file's lowest tabulated Ein. Pass ``float('nan')`` for
        the flag convention used by the higher-level quantities
        API.
    side : {'right', 'left'}, optional
        Endpoint-side selection at a doubled-E discontinuity
        (see :func:`primitives.tab1.interp` and issue #136).
        Default ``'right'`` matches NJOY reconr's TAB1 lookup
        convention; away from doubled-E points the setting has no
        effect. Pass ``'left'`` to force the pre-discontinuity
        value at the exact-E query.
    """
    # Local import avoids pulling primitives.tab1 at module import
    # time; keeps the existing MF3 code path free of extra deps.
    from ..primitives import tab1 as tab1_mod
    sec = endf_dict[3][mt]['xstable']
    t = tab1_mod.from_endf_dict(sec, x_key='E', y_key='xs')
    return tab1_mod.interp(
        t, energies_in, xp,
        outside_value=outside_value, side=side,
    )
