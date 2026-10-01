"""Unit tests for endf_userpy.run_options.RunOptions and the
accelerator-policy methods on the array_ns backend classes (issue
#143 + AutoBackend refactor).
"""
from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError, replace

import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.array_ns import (
    AutoBackend, JaxBackend, NumbaBackend, NumpyBackend,
)
from endf_userpy.run_options import RunOptions


# ---------- RunOptions dataclass behaviour ----------


def test_defaults_are_physics_first():
    """Default construction reflects the physics-first choice: MF2
    resonance composition and MT5 catch-all redistribution both on."""
    opts = RunOptions()
    assert opts.above_range == 'warn_nan'
    assert opts.resonance_range == 'warn'
    assert opts.include_resonance is True
    assert opts.mt5_contrib is True
    assert isinstance(opts.backend, AutoBackend)
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
    assert tight.include_resonance is True              # unchanged
    assert isinstance(tight.backend, AutoBackend)       # unchanged


# ---------- Backend string resolution at construction ----------


def test_backend_string_numpy_resolves_to_numpy_backend():
    opts = RunOptions(backend='numpy')
    assert isinstance(opts.backend, NumpyBackend)
    assert not isinstance(opts.backend, AutoBackend)
    assert opts.backend.name == 'numpy'


def test_backend_string_auto_resolves_to_auto_backend():
    opts = RunOptions(backend='auto')
    assert isinstance(opts.backend, AutoBackend)
    # AutoBackend reports name='numpy' because its algebra IS numpy;
    # the accelerator preference is queried via wants_accelerator().
    assert opts.backend.name == 'numpy'


def test_explicit_numba_raises_when_missing():
    """RunOptions(backend='numba') is a demand, not a hint: fail
    loud at construction if numba is unavailable."""
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
    numpy_backend = array_ns.get_backend('numpy')
    opts = RunOptions(backend=numpy_backend)
    assert opts.backend is numpy_backend


# ---------- Accelerator-policy methods ----------


def test_numpy_backend_rejects_acceleration():
    """Explicit numpy backend never takes the accelerator branch."""
    xp = array_ns.get_backend('numpy')
    assert xp.wants_accelerator('numba') is False
    # Pure runtime probe is still honest (returns whatever numba's
    # import status is), but wants=False gates the actual dispatch.
    xp.raise_if_needed_but_missing('numba')   # no-op, must not raise


def test_auto_backend_wants_numba_without_requiring_it():
    """AutoBackend asks for numba but doesn't demand it."""
    xp = array_ns.get_backend('auto')
    assert xp.wants_accelerator('numba') is True
    xp.raise_if_needed_but_missing('numba')   # no-op, soft-use


def test_auto_backend_missing_numba_warning_is_one_shot():
    """AutoBackend.accelerator_available('numba') fires the one-shot
    missing-numba warning at most once per Python session."""
    AutoBackend._warned_missing_numba = False
    try:
        import numba  # noqa: F401
        pytest.skip('numba is installed; cannot exercise the one-shot fallback warning')
    except Exception:
        pass
    xp = array_ns.get_backend('auto')
    with pytest.warns(UserWarning, match=r"backend='auto' fell back to numpy"):
        xp.accelerator_available('numba')
    # Second probe must not re-emit.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        xp.accelerator_available('numba')


def test_auto_backend_present_numba_no_warning():
    """When numba IS installed, AutoBackend.accelerator_available
    returns True silently."""
    try:
        import numba  # noqa: F401
    except Exception:
        pytest.skip('numba is not installed; cannot exercise the present-numba path')
    AutoBackend._warned_missing_numba = False
    xp = array_ns.get_backend('auto')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert xp.accelerator_available('numba') is True


def test_numba_backend_wants_and_requires_numba():
    """NumbaBackend both wants and requires numba. ``require`` is
    enforced at construction time; the dispatch-site ``raise_if_...``
    is defensive."""
    try:
        import numba  # noqa: F401
    except Exception:
        pytest.skip('numba is not installed')
    xp = array_ns.get_backend('numba')
    assert isinstance(xp, NumbaBackend)
    assert xp.wants_accelerator('numba') is True
    assert xp.accelerator_available('numba') is True
    xp.raise_if_needed_but_missing('numba')   # numba present, no raise


def test_jax_backend_cannot_use_numba():
    """JaxBackend rejects numba at every method level: the algebras
    don't compose."""
    try:
        import jax  # noqa: F401
    except Exception:
        pytest.skip('jax is not installed')
    xp = array_ns.get_backend('jax')
    assert isinstance(xp, JaxBackend)
    assert xp.wants_accelerator('numba') is False
    # Pure availability probe from a jax backend reports False even
    # if numba is installed system-wide, because jax arrays cannot
    # be consumed by numba kernels.
    assert xp.accelerator_available('numba') is False
    xp.raise_if_needed_but_missing('numba')   # jax never requires numba


def test_unknown_accelerator_raises_valueerror():
    """Typo guard: unknown accelerator names raise at the method call."""
    xp = array_ns.get_backend('auto')
    with pytest.raises(ValueError, match='unknown accelerator'):
        xp.wants_accelerator('cuda')
    with pytest.raises(ValueError, match='unknown accelerator'):
        xp.accelerator_available('cuda')
    with pytest.raises(ValueError, match='unknown accelerator'):
        xp.raise_if_needed_but_missing('cuda')
