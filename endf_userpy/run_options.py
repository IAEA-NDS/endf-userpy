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

from dataclasses import dataclass
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
