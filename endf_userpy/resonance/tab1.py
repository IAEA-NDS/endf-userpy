"""ENDF-6 TAB1 interpolation, backend-agnostic.

Small standalone module because the same TAB1 helpers are used by
every resonance formalism (channel-radius r_a, scattering-radius
r_ap, and later MF3 backgrounds). Kept out of the physics module
so the physics stays focused on formulas, not on
interpolation-law dispatch.
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


def interp(tab1: TAB1, x_query, xp) -> Any:
    """ENDF-6 TAB1 interpolation.

    ``x_query`` can be a scalar or array of the same backend as
    ``xp``. Returns the same shape as ``x_query``.

    ENDF-6 interpolation-law dispatch (per subrange):
      0 (sentinel: query outside [x1, x2]) -> 0
      1 (const): y1
      2 (lin-lin): y1 + (x - x1) * (y2 - y1) / (x2 - x1)
      3 (lin-log): y1 + log(x/x1) * (y2 - y1) / log(x2/x1)
      4 (log-lin): y1 * exp((x - x1) * log(y2/y1) / (x2 - x1))
      5 (log-log): y1 * exp(log(x/x1) * log(y2/y1) / log(x2/x1))

    Degenerate panels (x1==x2 or y1==y2) collapse to law 1
    (constant y1) to avoid divide-by-zero. Extrapolation returns 0.
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

    # Out-of-range -> sentinel 0 -> value 0.
    out_of_range = (x < x1) | (x > x2)
    law = xp.where(out_of_range, 0, law)

    return _apply_law_vectorised(law, x, x1, x2, y1, y2, xp)


def _apply_law_vectorised(law, x, x1, x2, y1, y2, xp):
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

    zero = xp.zeros_like(x)
    const_law = y1
    lin_lin = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    lin_log = y1 + xp.log(x / x1_safe) * (y2 - y1) / xp.log(x2_safe / x1_safe)
    log_lin = y1_safe * xp.exp((x - x1) * xp.log(y2_safe / y1_safe) / (x2 - x1))
    log_log = y1_safe * xp.exp(
        xp.log(x / x1_safe) * xp.log(y2_safe / y1_safe) / xp.log(x2_safe / x1_safe)
    )

    return xp.select(
        [law == 0, law == 1, law == 2, law == 3, law == 4, law == 5],
        [zero, const_law, lin_lin, lin_log, log_lin, log_log],
        default=0.0,
    )
