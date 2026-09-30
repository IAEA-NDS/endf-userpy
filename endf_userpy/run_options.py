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


# One-shot flag guarding the "numba missing, would have been faster"
# warning emitted when a user leaves the default ``backend='auto'``
# and hits ``include_resonance=True`` on a file with MF2 data. Module-
# level so the warning fires at most once per Python session.
_WARNED_MISSING_NUMBA_ON_AUTO = False


def _warn_auto_backend_numba_missing_once():
    """Emit the one-shot ``backend='auto'`` no-numba warning."""
    global _WARNED_MISSING_NUMBA_ON_AUTO
    if _WARNED_MISSING_NUMBA_ON_AUTO:
        return
    _WARNED_MISSING_NUMBA_ON_AUTO = True
    warnings.warn(
        "backend='auto' fell back to numpy for MF2 resonance "
        "reconstruction because numba is not installed. Install "
        "with `pip install numba` for the ~30x speedup on real "
        "actinide files. Pass RunOptions(backend='numpy') to "
        "silence this warning.",
        UserWarning, stacklevel=2,
    )


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

        ``'auto'`` (default): pure numpy end-to-end, EXCEPT that the
        MF2 resonance reconstruction uses the numba kernels when
        numba is installed (~30x speedup on real actinide files).
        Emits a single UserWarning per Python session on the first
        ``include_resonance=True`` call if numba is missing.

        ``'numpy'``: pure numpy. No auto-selection, no warning.

        ``'numba'``: numba resonance kernels + numpy elsewhere.
        Raises ``ImportError`` at option construction if numba is
        not installed.

        ``'jax'``: JAX end-to-end via the array-agnostic path.
        Enables autodiff and ``@jax.jit``. Raises ``ImportError`` at
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
        # Resolve explicit string aliases at construction time so
        # missing-optional-dependency errors surface early. 'auto'
        # is deliberately deferred to first-use (resolve_backend);
        # otherwise every RunOptions() would trigger a numba probe.
        b = self.backend
        if isinstance(b, str) and b != 'auto':
            from .primitives.array_ns import get_backend
            object.__setattr__(self, 'backend', get_backend(b))

    def resolve_backend(self, *, is_resonance_call: bool):
        """Return the concrete array-ns adapter for the current
        call. Called by ``compute_xs`` (and any other site that
        needs to hand ``xp`` down to the leaves).

        If the field was already resolved to an adapter object in
        ``__post_init__``, return it. If it is the string
        ``'auto'``, pick numba on a resonance call when available,
        else numpy; emit the one-shot missing-numba warning only
        when numba was actually the preferred choice.
        """
        b = self.backend
        if not isinstance(b, str):
            return b
        # Only 'auto' should remain as a string past __post_init__.
        assert b == 'auto', f'unexpected string backend: {b!r}'
        from .primitives.array_ns import get_backend
        if is_resonance_call:
            try:
                return get_backend('numba')
            except Exception:
                _warn_auto_backend_numba_missing_once()
                return get_backend('numpy')
        return get_backend('numpy')
