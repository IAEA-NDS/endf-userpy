import contextlib
import contextvars
import warnings
import numpy as np
from ..primitives.helpers import treat_duplicates
from ..primitives.interpolation import interp_tab1
from ..primitives.properties import (
    get_reaction_string_for_mt,
)


_ABOVE_RANGE_POLICIES = ('warn_nan', 'nan', 'warn_zero', 'zero', 'raise')

# Context-inherited policy for how to handle incident energies above
# the file's upper Ein boundary (issue #28). Set by the top-level
# `get_*` functions in `endf_userpy.quantities` via `above_range_ctx`;
# read by the leaf `compute_cross_section` functions in mf3 / mf10.
# Default `'warn_nan'` gives NaN fill with a UserWarning; see the
# `compute_cross_section` docstring for the full set of policies.
_above_range_var = contextvars.ContextVar(
    '_above_range', default='warn_nan',
)

# Per-query accumulator for above-range hits. Populated by
# `_handle_above_range` and drained by `above_range_ctx.__exit__`,
# which emits ONE summary UserWarning for the whole query rather
# than one per (MT, call). Value is either None (outside a context
# that collects) or a dict {mt -> (e_max, n_above, n_total)}.
_above_range_accum = contextvars.ContextVar(
    '_above_range_accum', default=None,
)


@contextlib.contextmanager
def above_range_ctx(policy):
    """Context manager that sets the above-range fill policy for
    the duration of the `with` block. Nested calls inherit the
    innermost setting; the previous value is restored on exit.

    For the two ``'warn_*'`` policies this context also collects
    the per-MT above-range hits reported by leaf readers and, on
    successful exit, emits ONE summary UserWarning naming every
    affected MT and its exceeded mesh limit. That keeps a single
    top-level query from emitting hundreds of warnings when the
    admitted-MT list is large; users who want per-MT granularity
    can inspect the summary or call the leaf reader directly.

    Users normally pass `above_range=...` to a top-level API
    function in `endf_userpy.quantities`; that function opens this
    context manager internally so every descendant call to
    `compute_cross_section` picks the same policy up. Reach for
    this context manager directly only when calling a leaf reader
    (`mf3_interpretation.compute_cross_section` or
    `mf10_interpretation.compute_cross_section`) outside of the
    top-level API surface.
    """
    if policy not in _ABOVE_RANGE_POLICIES:
        raise ValueError(
            f"above_range must be one of {_ABOVE_RANGE_POLICIES}; "
            f"got {policy!r}"
        )
    policy_token = _above_range_var.set(policy)
    accum = {} if policy in ('warn_nan', 'warn_zero') else None
    accum_token = _above_range_accum.set(accum)
    try:
        yield
    finally:
        _above_range_var.reset(policy_token)
        _above_range_accum.reset(accum_token)
        if accum:  # non-empty dict
            fill_word = 'NaN' if policy == 'warn_nan' else '0'
            n_mts = len(accum)
            total_above = sum(n_above for _, n_above, _ in accum.values())
            mt_summary = ', '.join(
                f'MT={mt} (max {e_max:.6g} eV, {n_above} pts)'
                for mt, (e_max, n_above, _) in
                sorted(accum.items())
            )
            warnings.warn(
                f'above_range: {total_above} out-of-mesh points across '
                f'{n_mts} MTs returned as {fill_word}: {mt_summary}. '
                f'Cross section is undefined above the evaluation range.',
                UserWarning, stacklevel=2,
            )


def _handle_above_range(policy, mt, e_max, above_mask, energies_in):
    """Common implementation of the `above_range` policy shared by
    every XS reader (see issue #28). Returns the numeric-fill value
    the caller should place at `above_mask` positions in its result
    array (`0.0` or `np.nan`), after either raising per the policy
    or recording the hit in the per-query accumulator for a summary
    warning to emit on `above_range_ctx.__exit__`.

    Callers pass the already-computed boolean `above_mask` (True at
    positions where `energies_in > e_max`) rather than re-deriving
    it, so the mask never has to be built twice. When no
    `above_range_ctx` is active, the accumulator is `None` and
    warn-mode falls back to a per-call warning so the leaf reader
    stays usable in isolation.
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
        accum = _above_range_accum.get()
        if accum is not None:
            # Inside above_range_ctx: record for a summary warning
            # on context exit. Keep the largest n_above per MT if
            # the same MT is queried more than once in the ctx.
            prev = accum.get(mt)
            if prev is None or n_above > prev[1]:
                accum[mt] = (e_max, n_above, len(energies_in))
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
    energies = get_nominal_incident_energies(endf_dict, mt)
    treat_duplicates(energies, inplace=True)
    xs = np.array(endf_dict[3][mt]['xstable']['xs'], dtype=float)
    idcs = np.nonzero(xs)[0]
    first_idx = idcs[0]-1 if idcs[0] > 0 else 0
    last_idx = idcs[-1]+1 if idcs[-1]+1 < len(idcs) else idcs[-1]
    return energies[first_idx:last_idx+1].copy()


def get_incident_energy_range(endf_dict, mt):
    eincs = get_incident_energies(endf_dict, mt)
    return (min(eincs), max(eincs))


def get_reaction_mts(endf_dict):
    return list(endf_dict[3].keys())


def get_reactions(endf_dict):
    mts = get_reaction_mts(endf_dict)
    reacs = [get_reaction_string_for_mt(endf_dict, m) for m in mts] 
    return reacs


def compute_cross_section(endf_dict, mt, energies_in, above_range=None):
    """Cross section for MT evaluated on `energies_in`.

    `above_range` controls what happens at incident energies above
    the file's upper Ein boundary for this MT -- ENDF-6 makes no
    claim about the cross section there. When `None` (default), the
    policy comes from the context variable set by the top-level
    `get_*` APIs in `endf_userpy.quantities`; the ambient default
    when nothing is set is ``'warn_nan'``.

    Explicit policies (also accepted here for callers that go
    around the top-level API):

    - ``'warn_nan'`` (default): fill above-range points with NaN and
      emit a `UserWarning` naming the MT and the exceeded limit.
      NaN propagates through downstream arithmetic so the undefined
      region is visible everywhere it matters.
    - ``'nan'``: same as above but silent.
    - ``'warn_zero'``: fill with 0 and emit a UserWarning. Matches
      pre-issue-#28 numeric behaviour while surfacing a signal.
    - ``'zero'``: fill with 0, silent. Pre-issue-#28 default.
    - ``'raise'``: raise `ValueError` and refuse to return.

    Below the mesh (typically sub-threshold or below the reaction
    onset) the return is always 0 regardless of `above_range`:
    `interp_tab1`'s outside-value fill uses 0 there and the physics
    interpretation of "below the file's lowest tabulated Ein" is
    almost always "physically zero" (the mesh starts at or below
    the reaction threshold).
    """
    if above_range is None:
        above_range = _above_range_var.get()
    sec = endf_dict[3][mt]
    xstab = sec['xstable']
    e_mesh = np.asarray(xstab['E'], dtype=float)
    e_max = float(e_mesh.max())
    einc_arr = np.asarray(energies_in, dtype=float)
    above_mask = einc_arr > e_max
    fill_value = _handle_above_range(
        above_range, mt, e_max, above_mask, einc_arr,
    )
    xs = interp_tab1(energies_in, xstab, 'E', 'xs', outside_value=0.0)
    if above_mask.any() and fill_value != 0.0:
        xs = np.where(above_mask, fill_value, xs)
    return xs
