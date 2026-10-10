"""Frozen dataclass carrying the runtime policies of the top-level
``endf_userpy.quantities`` API (issue #143).

Replaces the three contextvar-driven policy channels
(``_above_range_var``, ``_resonance_range_var``, ``_include_resonance_var``
+ ``_resonance_backend_var`` + ``_urr_quadrature_var``) with an
explicit options object threaded down the call chain.

Design rationale is captured in issue #143; the summary is:

- Signatures should tell the truth about what influences behaviour.
  Contextvars hid three (later four, later five) knobs behind
  action-at-a-distance.
- ``RunOptions()`` gives sensible physics-first defaults so a
  user asking ``get_reaction_xs(dict, '(n,g)', energies)`` gets
  the best physics the file can express, not a legacy-truncated
  value.
- Overriding a single field is a one-liner:
  ``RunOptions(backend='jax', above_range='error')``. Frozen
  dataclass + ``dataclasses.replace()`` handles composition.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunOptions:
    """Runtime policies for the top-level XS / distribution API.

    Attributes
    ----------
    above_range : str, default ``'warn_nan'``
        Policy for incident energies above the file's upper Ein
        boundary. One of
        ``'warn_nan' | 'nan' | 'warn_zero' | 'zero' | 'raise'``.
        See ``mfsec_interpretation.mf3_interpretation.compute_cross_section``.
    resonance_range : str, default ``'warn'``
        Policy for Ein inside the file's resolved-resonance region
        when ``include_resonance=False`` (raw MF3 there is only the
        background). Ignored when ``include_resonance=True`` because
        the composed cross section is physical over the RRR. One of
        ``'warn' | 'warn_nan' | 'nan' | 'raise'``.
    include_resonance : bool, default ``True``
        When True (the default), compose the MF2 resolved-resonance
        reconstruction (MLBW / Reich-Moore / R-Matrix Limited KRM=3)
        with the MF3 background per the ENDF-6 additive convention.
        Files without MF2 silently no-op. When False, raw MF3 is
        returned (legacy behaviour).
    mt5_contrib : bool, default ``True``
        Whether to redistribute the MT5 catch-all contribution into
        specific residual products.
    include_target : bool, default ``False``
        Whether ``get_residual_production_xs`` includes
        target-conserving channels (elastic MT2, short-lived
        discrete-level inelastic MT51..MT90, continuum inelastic
        MT91, and the inelastic sum MT4) in the sum when the queried
        residual is the target nuclide in its ground state. ``False``
        (default) follows the activation-library convention
        (IRDFF / EAF / NJOY-ACTIVA): channels whose only residual is
        the target in LFS=0 do not count toward "production of the
        target", because elastic and short-lived inelastic
        de-excite back to the ground state and leave the nucleus
        unchanged. ``True`` restores the inclusive sum (every MT
        matching the queried residual), which is the right answer
        when the user wants event-classification bookkeeping rather
        than activation. Isomer queries (e.g. ``'Nb-93m'``) are
        unaffected because the metastable state is a genuine
        different nuclide; the specific MF8-declared isomer-
        producing MT stays in the sum regardless of
        ``include_target``.
    backend : {'auto', 'numpy', 'numba', 'jax'} or adapter object, default ``'auto'``
        Backend selection for the whole reconstruction.

        ``'auto'`` (default): numpy algebra end-to-end with
        opportunistic numba acceleration for MF2 resonance kernels
        (~30x speedup on real actinide files). Resolves to
        :class:`endf_userpy.primitives.array_ns.AutoBackend`; emits a
        single UserWarning per Python session on the first MF2
        reconstruction if numba is missing.

        ``'numpy'``: pure numpy, never accelerate.

        ``'numba'``: numba resonance kernels + numpy elsewhere.
        Raises ``RuntimeError`` at option construction if numba is
        not installed.

        ``'jax'``: JAX end-to-end via the array-agnostic path.
        Enables autodiff and ``@jax.jit``. Raises ``ValueError`` at
        option construction if jax is not installed.

        Advanced use: pass an ``array_ns`` adapter object directly.
    broadening_mesh_bounds : (float, float), optional
        Static ``(emin, emax)`` for ``adaptive_convolve``'s internal
        mesh. Required when the caller runs
        ``get_particle_production_dxs_dE`` or
        ``get_particle_production_ddxs`` under ``@jax.jit`` with a
        tracer ``energies_out``. Include the
        ``n_kernel_widths * kernel_width`` margin on both sides.
        See Phase 5 of #290.
    broadening_window_kernel_widths : float, optional
        Target span of each windowed sub-convolution inside
        ``adaptive_convolve``, in units of ``kernel_width``.
        Controls the memory-vs-compile-time trade-off for broadened
        DDX / dxs_dE queries. ``None`` (default) resolves inside
        ``adaptive_convolve`` to ``200`` for concrete ``energies_out``
        and ``inf`` (= single window) for tracer. The 200 default
        keeps the window count bounded (1-10 for most realistic
        queries), so jit workflows that capture ``energies_out`` as
        a Python closure get a small XLA trace while still enjoying
        most of the memory savings on wide, resonance-dense queries.

        Tune per workflow:

        * Set to ``50`` to maximise eager-memory savings on actinide
          broadening at the cost of a larger jit trace graph.
        * Set to ``inf`` to disable windowing entirely (single mesh
          over the full range; use when a jit workflow needs the
          smallest possible XLA graph).

        No effect unless a ``broadening=`` sigma is passed to the
        top-level entry point.
    urr_quadrature : {'gauss_legendre_32', 'ross_10'}, default ``'gauss_legendre_32'``
        LRU=2 URR fluctuation-integral quadrature. ``'ross_10'`` is
        the NJOY-unresr-parity choice; ``'gauss_legendre_32'``
        matches Monte Carlo on the χ² integrand to ~1e-7. Only
        effective when ``include_resonance=True`` and the file has
        an LSSF=0 URR range. See #299.
    aggregation : {'top_down', 'bottom_up'}, default ``'bottom_up'``
        Admission-direction policy for sum-MT queries on scalar
        cross-section APIs (``get_reaction_xs``,
        ``get_particle_production_xs``, ``get_residual_production_xs``).
        ``'bottom_up'`` (default) uses the most fine-grained MF3
        MTs available and sums them; this matches the pre-#135
        behaviour and is internally consistent with how differential
        queries must work (sum MTs do not carry per-ejectile
        distributions). ``'top_down'`` uses the most aggregated
        MF3 MT that can address the query; on files that tabulate
        both the parent sum-MT and its children, the parent is
        returned directly and interpolation-grid-mismatch error
        between separately-tabulated children is avoided. See
        issue #135.

        Differential APIs (``get_particle_production_dxs_dE``,
        ``_dxs_dmu``, ``_ddxs``) always use bottom-up regardless
        of this setting because the ENDF-6 data model forces it;
        the request is silently ignored for those APIs.

        Scalar XS APIs with ``aggregation='top_down'`` emit one
        summary UserWarning per top-level call if the user's
        requested MT is not tabulated in MF3 and the admission
        frontier had to descend to a lower level.
    """

    above_range: str = 'warn_nan'
    resonance_range: str = 'warn'
    include_resonance: bool = True
    mt5_contrib: bool = True
    include_target: bool = False
    backend: Any = 'auto'
    broadening_mesh_bounds: tuple[float, float] | None = None
    broadening_window_kernel_widths: float | None = None
    urr_quadrature: str = 'gauss_legendre_32'
    aggregation: str = 'bottom_up'

    def __post_init__(self):
        # Validate policy strings at construction so users see the
        # error immediately when they build the RunOptions rather
        # than deep inside a leaf reader mid-query.
        from .mfsec_interpretation.mf3_interpretation import (
            _ABOVE_RANGE_POLICIES, _RESONANCE_RANGE_POLICIES,
        )
        if self.above_range not in _ABOVE_RANGE_POLICIES:
            raise ValueError(
                f"above_range must be one of {_ABOVE_RANGE_POLICIES}; "
                f"got {self.above_range!r}"
            )
        if self.resonance_range not in _RESONANCE_RANGE_POLICIES:
            raise ValueError(
                f"resonance_range must be one of "
                f"{_RESONANCE_RANGE_POLICIES}; got {self.resonance_range!r}"
            )
        if self.aggregation not in ('top_down', 'bottom_up'):
            raise ValueError(
                f"aggregation must be one of ('top_down', 'bottom_up'); "
                f"got {self.aggregation!r}"
            )
        # Resolve all string aliases ('auto' / 'numpy' / 'numba' /
        # 'jax') to adapter objects at construction time, so that
        # downstream code can read ``options.backend`` and get a
        # live adapter without a conditional. Missing-optional-
        # dependency errors (``numba`` for 'numba', ``jax`` for
        # 'jax') surface here rather than deep in a resonance
        # dispatch site. ``'auto'`` resolves to AutoBackend, which
        # never probes numba at construction (its probe is deferred
        # to the first ``accelerator_available('numba')`` call).
        b = self.backend
        if isinstance(b, str):
            from .primitives.array_ns import get_backend
            object.__setattr__(self, 'backend', get_backend(b))


@dataclass
class _QueryState:
    """Per-top-level-call mutable scratch state threaded down the
    call chain by every ``endf_userpy.quantities`` entry point.

    Companion to :class:`RunOptions`: ``RunOptions`` is the frozen
    input policy, ``_QueryState`` is the per-call output / scratch
    state that leaf readers populate and that the top-level entry
    point drains on the way out. Threaded as an underscore-prefixed
    ``_query_state=`` kwarg through every internal function that
    transitively calls the leaves; ``_query_state=None`` (the leaf's
    default) preserves the pre-#143 per-call warning fallback, which
    keeps the leaf usable when called from outside a top-level
    query (unit tests, ad-hoc scripts).

    Fields hold summary-warning accumulators (above-range NaN fill,
    resonance-range raw-MF3 fallback, #311 silent-zero /
    silent-raw-MF3 visibility) and the per-call MF2 range
    reconstruction cache (``range_recon_cache``).

    Private implementation detail; not part of the public API.
    """
    above_range: dict = field(default_factory=dict)
    resonance_range: dict = field(default_factory=dict)
    aggregation_fallback: list = field(default_factory=list)
    """When ``options.aggregation == 'top_down'`` for a scalar XS
    query but the user-requested MT is not tabulated in MF3, the
    admission frontier descends below the user's root. Each entry
    is a user MT that triggered a fallback; drained into a single
    summary UserWarning per top-level call by
    :func:`_emit_summary_warnings`. Issue #135."""
    missing_user_mts: list = field(default_factory=list)
    """MTs the user requested (via reaction string) that are
    neither tabulated in the file nor synthesisable from admitted
    partials. Populated at the top-level ``_impl`` function after
    the cumulative-sum iteration completes. Mode 1 of #311."""
    unmapped_composition_mts: list = field(default_factory=list)
    """MTs the user requested that are in the file but have no
    entry in the active MF2 formalism's MT-to-partial-keys map, so
    the composition layer silently returned raw MF3 for them.
    Populated inside ``reconstruct_resonance_xs``, scoped to
    ``user_mts`` to avoid noise. Mode 2 of #311."""
    no_production: list = field(default_factory=list)
    """``(reaction, particle)`` pairs of a particle-production query
    for which no MT in the file was admitted, so the result is zeros.
    Populated by the four ``get_particle_production_*`` entry points."""
    user_mts: set = field(default_factory=set)
    """MTs the user explicitly requested at the top-level entry
    point. Threaded here so the composition layer can scope mode 2
    reporting to user-requested MTs only, not the widened
    iteration set in differential queries."""
    host_energies: tuple | None = None
    """``(staged, host)`` when the top-level call runs inside a
    ``jax.jit`` trace with a concrete query mesh (see
    ``quantities._stage_energies``): ``staged`` is the jax array every
    layer below receives, ``host`` the same energies as a numpy array,
    so the composition layer can still locate a resonance range's
    in-range points on the host and reconstruct only those. ``None``
    otherwise."""
    range_recon_cache: dict = field(default_factory=dict)
    """Per-call cache of MF2 range reconstructions (one dict of
    partial cross sections per resonance range). A top-level query
    composes several MTs (``(n,total)``: MT=1 / 2 / 18 / 102 ...)
    that each read a different key of the SAME range reconstruction;
    without the cache every range was reconstructed once per MT (2-3x
    per call). Filled and read by
    :func:`~endf_userpy.quantities_mt_zap.resonance_composition.reconstruct_resonance_xs`;
    entries hold a reference to the query-energy array they were
    computed for and are reused only for that same object. Lifetime:
    one top-level call (under ``jax.jit`` of the whole pipeline: one
    trace)."""


def _emit_summary_warnings(query_state, options):
    """Drain a :class:`_QueryState` into UserWarnings, at most one
    per policy family. Called by every top-level entry point in
    :mod:`endf_userpy.quantities` after its impl returns (see
    issue #143). Silent when the corresponding ``query_state``
    collection is empty or when the policy is a non-``warn_*``
    variant.
    """
    ar_policy = options.above_range
    rr_policy = options.resonance_range
    if query_state.above_range and ar_policy in ('warn_nan', 'warn_zero'):
        fill_word = 'NaN' if ar_policy == 'warn_nan' else '0'
        n_mts = len(query_state.above_range)
        total_above = sum(
            n_above for _, n_above, _ in query_state.above_range.values()
        )
        mt_summary = ', '.join(
            f'MT={mt} (max {e_max:.6g} eV, {n_above} pts)'
            for mt, (e_max, n_above, _) in
            sorted(query_state.above_range.items())
        )
        warnings.warn(
            f'above_range: {total_above} out-of-mesh points across '
            f'{n_mts} MTs returned as {fill_word}: {mt_summary}. '
            f'Cross section is undefined above the evaluation range.',
            UserWarning, stacklevel=3,
        )
    if query_state.resonance_range and rr_policy in ('warn', 'warn_nan'):
        action = 'NaN' if rr_policy == 'warn_nan' else 'raw MF3 background'
        n_mts = len(query_state.resonance_range)
        total_pts = sum(n for _, n, _, _ in query_state.resonance_range.values())
        details = ', '.join(
            f'MT={mt} ({n_in} of {n_total} pts in RRR '
            f'[{el:.3g}, {eh:.3g}] eV)'
            for mt, (el, eh, n_in, n_total) in
            sorted(query_state.resonance_range.items())
        )
        warnings.warn(
            f'resonance_range: {total_pts} in-RRR points across '
            f'{n_mts} MTs returned as {action}: {details}. '
            f'MF3 in the resolved-resonance region is a '
            f'subtractive background cross section; the physical '
            f'cross section requires resonance reconstruction from '
            f'MF2 (see README "Known limitations").',
            UserWarning, stacklevel=3,
        )
    if query_state.missing_user_mts:
        mt_list = ', '.join(
            f'MT={mt}' for mt in sorted(set(query_state.missing_user_mts))
        )
        warnings.warn(
            f'reaction string resolved to {mt_list}, which the '
            f'file does not tabulate; no admitted partial could '
            f'synthesise the requested cross section via the sum '
            f'rule, so the result is zero. Common cause: '
            f'ambiguous fission reaction spellings -- '
            f"'(n,fission)' resolves to MT 18 (total fission, "
            f"usually available), '(n,f)' to MT 19 (first-chance "
            f'fission, often absent). Use '
            f"endf_userpy.quantities_mt_zap.get_reaction_mt_numbers "
            f"to inspect what the file carries.",
            UserWarning, stacklevel=3,
        )
    if query_state.no_production:
        pairs = ', '.join(
            f'{particle!r} from {reaction!r}'
            for reaction, particle in sorted(set(query_state.no_production))
        )
        warnings.warn(
            f'no reaction in the file produces {pairs}: no MT carries '
            f'data for that ejectile (production is either absent from '
            f'the evaluation or lumped into an MT without it), so the '
            f'result is zero.',
            UserWarning, stacklevel=3,
        )
    if query_state.aggregation_fallback:
        mts = sorted(set(query_state.aggregation_fallback))
        mt_list = ', '.join(f'MT={m}' for m in mts)
        warnings.warn(
            f"aggregation='top_down': {mt_list} not tabulated in "
            f"MF3; the admission frontier descended to the top-most "
            f"MF3-present descendants. The returned cross section "
            f"is the sum of those descendants (equivalent to the "
            f"'bottom_up' answer when the parent is absent).",
            UserWarning, stacklevel=3,
        )
    if query_state.unmapped_composition_mts:
        mt_list = ', '.join(
            f'MT={mt}' for mt in sorted(set(query_state.unmapped_composition_mts))
        )
        warnings.warn(
            f'{mt_list} is in MF3 but has no entry in the active '
            f'MF2 formalism\'s MT-to-partial-keys map; the '
            f'composition layer returned raw MF3 only for these '
            f'MTs, without adding a resonance contribution. If '
            f'the file carries MF2 widths that should contribute '
            f'to these MTs, the returned cross section is '
            f'incomplete in the resolved-resonance region.',
            UserWarning, stacklevel=3,
        )
