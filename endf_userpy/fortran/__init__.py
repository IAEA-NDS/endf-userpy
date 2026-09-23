"""Runtime-optional Fortran extension.

The Fortran extension (``endf6``) is opt-in at install time: the
default ``pip install .`` produces a pure-Python install with no
compiler dependency; the extension is built only when
``ENDF_USERPY_BUILD_FORTRAN=1`` is set (or a wheel that already
carries the extension is installed).

The runtime code paths in :mod:`endf_userpy.quantities` do not use
the extension. Only the parity-oracle ``_fort`` sibling modules
under :mod:`endf_userpy.mfsec_interpretation` import from it, and
those modules degrade gracefully via :func:`_stub` when the
extension is absent.
"""
from __future__ import annotations


try:
    from . import endf6 as _endf6  # noqa: F401
    HAS_FORTRAN = True
except ImportError:
    HAS_FORTRAN = False


_UNAVAILABLE_MSG = (
    'endf_userpy Fortran extension not built. This function is a '
    'Fortran-backed parity-oracle wrapper used only by the equivalence '
    'test suite; the runtime API in endf_userpy.quantities is pure '
    'Python and works without it. To build the extension, reinstall '
    'from source with `ENDF_USERPY_BUILD_FORTRAN=1 pip install -e .` '
    '(requires gfortran on Linux/macOS, ifx on Windows), or install a '
    'pre-built wheel that already ships it.'
)


def _stub(name):
    """Return a callable that raises ``RuntimeError`` when invoked.

    Use in ``_fort`` modules to substitute for Fortran symbols when
    the extension isn't built: importing the module then still
    succeeds (so tests can ``pytest.importorskip`` without triggering
    an ImportError chain), and only *calling* one of the wrappers
    surfaces the friendly install-hint message.
    """
    def _missing(*args, **kwargs):
        raise RuntimeError(f'{name}: {_UNAVAILABLE_MSG}')
    _missing.__name__ = f'{name}_stub'
    _missing.__qualname__ = _missing.__name__
    return _missing
