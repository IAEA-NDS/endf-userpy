"""Numpy version compatibility helpers.

This module papers over a handful of numpy 2.0 API renames so the
rest of the package can be written against the modern spelling
but still run on numpy 1.x. Nothing here reimplements numpy; each
symbol is an alias for whichever spelling exists at import time.

Currently exports:

- ``trapezoid`` -- ``np.trapezoid`` on numpy >= 2.0, ``np.trapz``
  on numpy < 2.0. Same signature and semantics either way (numpy
  2.0 renamed ``np.trapz`` to ``np.trapezoid`` and kept the old
  name as a deprecated alias, which was removed in a later 2.x
  release).

Import as::

    from endf_userpy.primitives.np_compat import trapezoid

and use exactly as you would ``np.trapezoid``.
"""
import numpy as np

# np.trapezoid was added in numpy 2.0 as the replacement name
# for np.trapz. np.trapz was retained as a deprecated alias in
# 2.0 and later removed in a subsequent 2.x release. Fall back
# to np.trapz on any numpy version where np.trapezoid is absent
# (i.e. numpy < 2.0).
trapezoid = getattr(np, 'trapezoid', np.trapz)


__all__ = ['trapezoid']
