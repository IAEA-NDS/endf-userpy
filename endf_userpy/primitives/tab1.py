"""ENDF-6 TAB1 interpolation, backend-agnostic.

Small standalone module for TAB1 records represented as a dataclass
(``x``, ``y``, ``nbt``, ``intp`` numpy arrays). The
:func:`interp` entry point dispatches through the array-namespace
adapter (:mod:`endf_userpy.primitives.array_ns`) so the same
implementation runs on numpy and JAX; a numba backend can plug in
next to those without touching the physics that calls this.

Related helpers already in the package:

- :mod:`endf_userpy.primitives.interpolation` -- older / broader
  set of TAB1 helpers used by MF3, MF5, MF15. Those work on
  the ``endf_parserpy`` dict-of-arrays representation
  (``tab1['xy']['E']`` etc.) and are numpy-only. They remain the
  right choice for the existing MF3/5/15 code paths; the
  backend-agnostic API here is for the MF2 resonance-reconstruction
  work where the same physics code must run on multiple array
  backends. The :func:`from_endf_dict` helper below builds a
  :class:`TAB1` from the dict-of-arrays layout, so a caller that
  wants the array-agnostic path from an ENDF dict is one line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TAB1:
    """Natural-size ENDF-6 TAB1 record.

    No padding, no fixed capacity. Fields carried as numpy arrays
    because they are typically small (radii, background XS) and
    the interpolation runs on the backend of the caller via
    :func:`interp` -- so we cast on entry rather than committing
    to a backend at construction time.

    Fields (per ENDF-6 6.1.2):
        x:    (NP,) abscissae, monotone non-decreasing.
        y:    (NP,) ordinates.
        nbt:  (NR,) 0-indexed segment endpoints; nbt[-1] == NP - 1.
        intp: (NR,) interpolation-law codes 1..5:
              1 constant, 2 lin-lin, 3 lin-log, 4 log-lin, 5 log-log.
    """
    x: np.ndarray
    y: np.ndarray
    nbt: np.ndarray
    intp: np.ndarray


def interp(tab1: TAB1, x_query, xp, outside_value=0.0) -> Any:
    """ENDF-6 TAB1 interpolation.

    ``x_query`` can be a scalar or array of the same backend as
    ``xp``. Returns the same shape as ``x_query``.

    ENDF-6 interpolation-law dispatch (per subrange):
      0 (sentinel: query outside [x1, x2]) -> ``outside_value``
      1 (const): y1
      2 (lin-lin): y1 + (x - x1) * (y2 - y1) / (x2 - x1)
      3 (lin-log): y1 + log(x/x1) * (y2 - y1) / log(x2/x1)
      4 (log-lin): y1 * exp((x - x1) * log(y2/y1) / (x2 - x1))
      5 (log-log): y1 * exp(log(x/x1) * log(y2/y1) / log(x2/x1))

    Degenerate panels (x1==x2 or y1==y2) collapse to law 1
    (constant y1) to avoid divide-by-zero. Extrapolation returns
    ``outside_value`` (default 0.0).

    Note on ``outside_value``: pass ``float('nan')`` to flag
    out-of-mesh queries at the caller boundary (mirrors the
    warn_nan policy in :mod:`endf_userpy.quantities`). Raising
    on out-of-range is not supported here (backend-agnostic code
    can't raise cleanly under JAX tracing); use ``nan`` and
    check.
    """
    x = xp.asarray(x_query, dtype=xp.float64)

    tab_x = xp.asarray(tab1.x, dtype=xp.float64)
    tab_y = xp.asarray(tab1.y, dtype=xp.float64)
    tab_nbt = xp.asarray(tab1.nbt, dtype=xp.int32)
    tab_intp = xp.asarray(tab1.intp, dtype=xp.int32)

    # Locate the containing panel for each query point.
    i = xp.clip(xp.searchsorted(tab_x, x, side='left'), 1, len(tab_x) - 1)
    k = xp.searchsorted(tab_nbt, i, side='left')

    x1 = tab_x[i - 1]
    x2 = tab_x[i]
    y1 = tab_y[i - 1]
    y2 = tab_y[i]
    law = tab_intp[k] % 10

    # Collapse degenerate cases to constant y1.
    degenerate = (x1 == x2) | (y1 == y2)
    law = xp.where(degenerate, 1, law)

    # Out-of-range -> sentinel 0 -> `outside_value`.
    out_of_range = (x < x1) | (x > x2)
    law = xp.where(out_of_range, 0, law)

    return _apply_law_vectorised(
        law, x, x1, x2, y1, y2, xp, outside_value=outside_value,
    )


def from_endf_dict(
    tab1_dict, x_key: str = 'E', y_key: str = 'xs',
) -> TAB1:
    """Build a :class:`TAB1` from an ``endf_parserpy`` dict record.

    ``endf_parserpy`` renders a TAB1 as::

        tab1_dict = {'NBT': [...], 'INT': [...], x_key: [...], y_key: [...]}

    or (for cross sections) as::

        tab1_dict = {'xstable': {'NBT': ..., 'INT': ..., 'E': ..., 'xs': ...}}

    Both shapes work: if ``x_key`` / ``y_key`` sit under an
    ``'xstable'`` (or ``'xy'``) subdict, the helper drills in.

    The rendered ``NBT`` values are 1-based (ENDF-6 convention). We
    subtract 1 so the returned ``TAB1.nbt`` matches the 0-indexed
    convention that :func:`interp` expects.
    """
    if x_key not in tab1_dict:
        # Common wrapping in endf_parserpy: 'xstable' for MF3-style
        # cross sections, 'xy' for others.
        for wrapper in ('xstable', 'xy', 'xy_table'):
            if wrapper in tab1_dict and x_key in tab1_dict[wrapper]:
                tab1_dict = tab1_dict[wrapper]
                break
        else:
            raise KeyError(
                f"key {x_key!r} not found in TAB1 dict "
                f"(top-level keys: {list(tab1_dict.keys())})"
            )
    return TAB1(
        x=np.asarray(tab1_dict[x_key], dtype=np.float64),
        y=np.asarray(tab1_dict[y_key], dtype=np.float64),
        # NBT is 1-based in the ENDF-6 spec; subtract 1 to get
        # 0-indexed segment endpoints.
        nbt=np.asarray(tab1_dict['NBT'], dtype=np.int32) - 1,
        intp=np.asarray(tab1_dict['INT'], dtype=np.int32),
    )


def _apply_law_vectorised(law, x, x1, x2, y1, y2, xp, outside_value=0.0):
    """Vectorised element-wise dispatch across interpolation laws.

    Rather than `xp.switch` per element (branch-per-point, slow on
    every backend), compute all five candidate values and pick with
    `xp.select`. Wasted arithmetic on the branches not taken is
    dominated by the resonance-summation cost downstream and is
    the same order-of-magnitude as one extra ufunc pass.
    """
    EPS = 1e-30

    # Small guards to keep divisions/logs finite in the branches
    # that will be discarded by `select`. The `where` values only
    # affect the branch value, which is then not chosen.
    x1_safe = xp.where(x1 == 0.0, EPS, x1)
    x2_safe = xp.where(x2 == x1_safe, x1_safe * (1 + EPS), x2)
    y1_safe = xp.where(y1 == 0.0, EPS, y1)
    y2_safe = xp.where(y2 == 0.0, EPS, y2)

    const_law = y1
    lin_lin = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    # Silence divide-by-zero warnings from the branches that xp.select
    # discards. The result of the discarded branch is never used but
    # numpy still evaluates it; masking at the log inputs would work but
    # complicates the arithmetic. Numpy: use errstate. JAX: no warnings.
    if getattr(xp, 'name', None) == 'numpy':
        with xp._np.errstate(divide='ignore', invalid='ignore'):
            lin_log = y1 + xp.log(x / x1_safe) * (y2 - y1) / xp.log(x2_safe / x1_safe)
            log_lin = y1_safe * xp.exp((x - x1) * xp.log(y2_safe / y1_safe) / (x2 - x1))
            log_log = y1_safe * xp.exp(
                xp.log(x / x1_safe) * xp.log(y2_safe / y1_safe) / xp.log(x2_safe / x1_safe)
            )
    else:
        lin_log = y1 + xp.log(x / x1_safe) * (y2 - y1) / xp.log(x2_safe / x1_safe)
        log_lin = y1_safe * xp.exp((x - x1) * xp.log(y2_safe / y1_safe) / (x2 - x1))
        log_log = y1_safe * xp.exp(
            xp.log(x / x1_safe) * xp.log(y2_safe / y1_safe) / xp.log(x2_safe / x1_safe)
        )

    outside = xp.full_like(x, float(outside_value))
    return xp.select(
        [law == 0, law == 1, law == 2, law == 3, law == 4, law == 5],
        [outside, const_law, lin_lin, lin_log, log_lin, log_log],
        default=float(outside_value),
    )
