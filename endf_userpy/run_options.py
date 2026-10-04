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
    urr_quadrature : {'gauss_legendre_32', 'ross_10'}, default ``'gauss_legendre_32'``
        LRU=2 URR fluctuation-integral quadrature. ``'ross_10'`` is
        the NJOY-unresr-parity choice; ``'gauss_legendre_32'``
        matches Monte Carlo on the χ² integrand to ~1e-7. Only
        effective when ``include_resonance=True`` and the file has
        an LSSF=0 URR range. See #299.
    """

    above_range: str = 'warn_nan'
    resonance_range: str = 'warn'
    include_resonance: bool = True
    mt5_contrib: bool = True
    backend: Any = 'auto'
    broadening_mesh_bounds: tuple[float, float] | None = None
    urr_quadrature: str = 'gauss_legendre_32'

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

    Fields today only hold summary-warning accumulators (above-range
    NaN fill, resonance-range raw-MF3 fallback, #311 silent-zero /
    silent-raw-MF3 visibility). Future per-call state belongs here
    too: see #306 for the planned ``reconstruction_cache`` field.

    Private implementation detail; not part of the public API.
    """
    above_range: dict = field(default_factory=dict)
    resonance_range: dict = field(default_factory=dict)
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
    user_mts: set = field(default_factory=set)
    """MTs the user explicitly requested at the top-level entry
    point. Threaded here so the composition layer can scope mode 2
    reporting to user-requested MTs only, not the widened
    iteration set in differential queries."""


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
