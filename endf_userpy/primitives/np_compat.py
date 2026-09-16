"""Numpy version compatibility helpers.

This module papers over a handful of numpy 2.0 API renames so the
rest of the package can be written against the modern spelling
but still run on numpy 1.x. Nothing here reimplements numpy; each
symbol is an alias for whichever spelling exists at import time.

Currently exports:

- ``trapezoid`` -- ``np.trapezoid`` on numpy >= 2.0, ``np.trapz``
  on numpy < 2.0. Same signature and semantics either way (numpy
  2.0 renamed ``np.trapz`` to ``np.trapezoid``, kept the old name
  as a deprecated alias in the 2.0.x series, and removed
  ``np.trapz`` outright in a later 2.x release).

Import as::

    from endf_userpy.primitives.np_compat import trapezoid

and use exactly as you would ``np.trapezoid``.
"""
# NOTE: we resolve the alias with `try/except ImportError` rather
# than `getattr(np, 'trapezoid', np.trapz)`. The `getattr` form
# evaluates its third argument unconditionally, so on any numpy
# release where `np.trapz` has been removed (>= 2.something) the
# import of this module would itself raise AttributeError even
# though `np.trapezoid` is available.
try:
    from numpy import trapezoid  # noqa: F401  (re-export)
except ImportError:  # numpy < 2.0
    from numpy import trapz as trapezoid  # noqa: F401  (re-export)


__all__ = ['trapezoid']
