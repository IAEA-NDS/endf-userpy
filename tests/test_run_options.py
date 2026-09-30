"""Unit tests for endf_userpy.run_options.RunOptions (issue #143)."""
from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError, replace

import pytest

from endf_userpy.run_options import (
    RunOptions,
    _warn_auto_backend_numba_missing_once,
)


def test_defaults_are_physics_first():
    """Default construction reflects the physics-first choice: MF2
    resonance composition and MT5 catch-all redistribution both on."""
    opts = RunOptions()
    assert opts.above_range == 'warn_nan'
    assert opts.resonance_range == 'warn'
    assert opts.include_resonance is True
    assert opts.mt5_contrib is True
    assert opts.backend == 'auto'
    assert opts.broadening_mesh_bounds is None
    assert opts.urr_quadrature == 'gauss_legendre_32'


def test_frozen_dataclass_rejects_mutation():
    opts = RunOptions()
    with pytest.raises(FrozenInstanceError):
        opts.above_range = 'raise'  # type: ignore[misc]


def test_dataclass_replace_composition():
    """replace() gives 'change one field, keep the rest' ergonomics."""
    base = RunOptions()
    tight = replace(base, above_range='raise')
    assert tight.above_range == 'raise'
    assert tight.include_resonance is True  # unchanged
    assert tight.backend == 'auto'          # unchanged


def test_backend_string_alias_numpy_resolves_at_construction():
    """RunOptions(backend='numpy') resolves to the numpy adapter
    at __post_init__ time (so missing-backend errors surface early)."""
    from endf_userpy.primitives import array_ns
    opts = RunOptions(backend='numpy')
    assert isinstance(opts.backend, type(array_ns.get_backend('numpy')))
    assert opts.backend.name == 'numpy'


def test_backend_auto_stays_string_until_resolve():
    """'auto' is deferred to resolve_backend() so RunOptions
    construction never probes numba availability."""
    opts = RunOptions()
    assert opts.backend == 'auto'


def test_resolve_backend_auto_non_resonance_call_gives_numpy():
    """On a non-resonance call, 'auto' resolves to numpy regardless
    of whether numba is installed. No warning emitted."""
    import endf_userpy.run_options as ro
    ro._WARNED_MISSING_NUMBA_ON_AUTO = False
    opts = RunOptions()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        resolved = opts.resolve_backend(is_resonance_call=False)
    assert resolved.name == 'numpy'


def test_resolve_backend_auto_resonance_call_prefers_numba_when_available():
    """On a resonance call, 'auto' prefers numba when installed,
    with no warning. Silently falls back to numpy with a one-shot
    warning otherwise."""
    from endf_userpy.primitives import array_ns
    import endf_userpy.run_options as ro
    ro._WARNED_MISSING_NUMBA_ON_AUTO = False

    try:
        array_ns.get_backend('numba')
        numba_available = True
    except Exception:
        numba_available = False

    opts = RunOptions()
    if numba_available:
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # no warnings expected
            resolved = opts.resolve_backend(is_resonance_call=True)
        assert resolved.name == 'numba'
    else:
        with pytest.warns(UserWarning, match=r"backend='auto' fell back to numpy"):
            resolved = opts.resolve_backend(is_resonance_call=True)
        assert resolved.name == 'numpy'


def test_auto_backend_missing_numba_warning_is_one_shot():
    """The 'auto' fallback warning fires at most once per process."""
    import endf_userpy.run_options as ro
    ro._WARNED_MISSING_NUMBA_ON_AUTO = False
    try:
        import numba  # noqa: F401
        pytest.skip('numba is installed; cannot exercise the one-shot fallback warning')
    except Exception:
        pass
    opts = RunOptions()
    with pytest.warns(UserWarning):
        opts.resolve_backend(is_resonance_call=True)
    # Second call must not re-emit.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        opts.resolve_backend(is_resonance_call=True)


def test_explicit_numba_raises_when_missing():
    """RunOptions(backend='numba') is a demand, not a hint: fail
    loud at construction if numba is unavailable so the user sees
    the error before any computation."""
    try:
        import numba  # noqa: F401
        pytest.skip('numba is installed; cannot exercise the missing-numba path')
    except Exception:
        pass
    with pytest.raises(Exception):
        RunOptions(backend='numba')


def test_explicit_jax_raises_when_missing():
    """Same demand-not-hint semantics for jax."""
    try:
        import jax  # noqa: F401
        pytest.skip('jax is installed; cannot exercise the missing-jax path')
    except Exception:
        pass
    with pytest.raises(Exception):
        RunOptions(backend='jax')


def test_adapter_object_passes_through():
    """Passing an already-resolved adapter object bypasses the
    string-resolution machinery."""
    from endf_userpy.primitives import array_ns
    numpy_backend = array_ns.get_backend('numpy')
    opts = RunOptions(backend=numpy_backend)
    assert opts.backend is numpy_backend
    # resolve_backend is a pass-through for non-string.
    assert opts.resolve_backend(is_resonance_call=True) is numpy_backend
    assert opts.resolve_backend(is_resonance_call=False) is numpy_backend
