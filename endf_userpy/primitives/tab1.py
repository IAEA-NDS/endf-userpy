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


def interp(tab1: TAB1, x_query, xp, outside_value=0.0, side='right',
           x_host=None) -> Any:
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

    Degenerate panels (x1==x2 or y1==y2) collapse to law 1 to
    avoid divide-by-zero; the returned constant follows the
    ``side`` selection (``y1`` for ``side='left'``, ``y2`` for
    ``side='right'``). Extrapolation returns ``outside_value``
    (default 0.0).

    ``side`` — endpoint selection at a step discontinuity:
    ENDF-6 encodes a discontinuity by placing two adjacent
    x-values equal (with different y-values). A query at exactly
    that shared x is ambiguous: the left-limit is the y before
    the step, the right-limit is the y after. NJOY reconr's
    ``find_interval`` uses ``side='right'`` internally so its
    downstream lookups return the right-limit, and this library's
    own :func:`endf_interp1d` matches that convention already.
    ``primitives.tab1.interp`` now defaults to ``'right'`` as well
    so its behaviour agrees with NJOY at doubled-E points. Pass
    ``side='left'`` to force the left-limit (matches libnew and
    some other downstream tools; see issue #136). Away from
    doubled x-values the two settings produce identical results.

    Note on ``outside_value``: pass ``float('nan')`` to flag
    out-of-mesh queries at the caller boundary (mirrors the
    warn_nan policy in :mod:`endf_userpy.quantities`). Raising
    on out-of-range is not supported here (backend-agnostic code
    can't raise cleanly under JAX tracing); use ``nan`` and
    check.

    ``x_host`` (jax backend only): the query energies as a numpy array
    when ``x_query`` is a tracer of a concrete mesh, i.e. the staged
    mesh of a ``jax.jit`` trace (see ``quantities._stage_energies``).
    The panel lookup then runs on the host and only the int32 panel
    indices enter the traced program; without it a tracer query takes
    the fully traceable path below (device-side ``searchsorted``,
    ~2x slower at 1M points).
    """
    if side not in ('right', 'left'):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    x = xp.asarray(x_query, dtype=xp.float64)

    # Numba fast path (issue #349). The panel lookup
    # (`searchsorted`, `where`) stays on numpy because its C
    # implementation is already SIMD-fast; only the per-point law
    # dispatch is offloaded to numba. The pre-#349 vectorised path
    # computed all five candidate laws (log, exp included) for
    # every point even when a lin-lin branch was going to be
    # selected -- that is the hot spot on wide dense Ein meshes.
    # The numba kernel below branches per point and only evaluates
    # the law that applies.
    if xp.wants_accelerator('numba') and xp.accelerator_available('numba'):
        from . import tab1_numba
        x_arr = np.asarray(x, dtype=np.float64)
        orig_shape = x_arr.shape
        flat = x_arr.ravel()
        y_flat = tab1_numba._interp_full_numba(
            flat,
            np.asarray(tab1.x, dtype=np.float64),
            np.asarray(tab1.y, dtype=np.float64),
            np.asarray(tab1.nbt, dtype=np.int32),
            np.asarray(tab1.intp, dtype=np.int32),
            float(outside_value),
            side == 'left',
        )
        if orig_shape == ():
            return y_flat.reshape(())[()]
        return y_flat.reshape(orig_shape)

    # JAX with a concrete query mesh and concrete table abscissae:
    # panel lookup on numpy, value arithmetic in one jitted kernel
    # (see :mod:`tab1_jax`). Ordinates may be traced (gathered on jax).
    if getattr(xp, 'name', None) == 'jax':
        host = _host_arrays(x, tab1)
        if host is not None:
            from . import tab1_jax
            x_np, tab_x_np, tab_y_np = host
            i, law = tab1_jax.host_lookup(
                tab_x_np, np.asarray(tab1.nbt), np.asarray(tab1.intp),
                x_np, side,
            )
            if tab_y_np is not None:
                y1, y2 = tab_y_np[i - 1], tab_y_np[i]
            else:
                tab_y = xp.asarray(tab1.y, dtype=xp.float64)
                y1, y2 = tab_y[i - 1], tab_y[i]
            return tab1_jax.interp_from_lookup(
                law, x, tab_x_np[i - 1], tab_x_np[i], y1, y2,
                xp.asarray(outside_value, dtype=xp.float64),
                laws=_static_laws(tab1.intp), side=side,
            )
        if x_host is not None:
            try:
                tab_x_np = np.asarray(tab1.x, dtype=np.float64)
            except Exception:      # traced abscissae
                tab_x_np = None
            if tab_x_np is not None:
                return _interp_traced_query_host_lookup(
                    tab1, x, np.asarray(x_host, dtype=np.float64),
                    tab_x_np, xp, outside_value, side,
                )

    tab_x = xp.asarray(tab1.x, dtype=xp.float64)
    tab_y = xp.asarray(tab1.y, dtype=xp.float64)
    tab_nbt = xp.asarray(tab1.nbt, dtype=xp.int32)
    tab_intp = xp.asarray(tab1.intp, dtype=xp.int32)

    # Locate the containing panel for each query point. `side` picks
    # which of the two neighbouring panels a query that lands
    # exactly on a doubled x-value belongs to: 'right' -> next panel
    # (right-limit / NJOY convention), 'left' -> previous panel.
    i = xp.clip(xp.searchsorted(tab_x, x, side=side), 1, len(tab_x) - 1)
    k = xp.searchsorted(tab_nbt, i, side='left')

    x1 = tab_x[i - 1]
    x2 = tab_x[i]
    y1 = tab_y[i - 1]
    y2 = tab_y[i]
    law = tab_intp[k] % 10

    # Collapse degenerate panels to law 1. For `side='right'` the
    # constant is y2 (right-limit at a doubled x); for `side='left'`
    # it is y1 (left-limit). Encoded via a synthetic law code so the
    # dispatch stays vectorised.
    degen_y_ok = (y1 == y2)
    degen_x = (x1 == x2)
    degenerate = degen_x | degen_y_ok
    const_law_code = 1 if side == 'left' else 6
    law = xp.where(degenerate, const_law_code, law)

    # Out-of-range -> sentinel 0 -> `outside_value`.
    out_of_range = (x < x1) | (x > x2)
    law = xp.where(out_of_range, 0, law)

    return _apply_law_vectorised(
        law, x, x1, x2, y1, y2, xp, outside_value=outside_value,
        laws=_static_laws(tab1.intp),
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


_ALL_LAWS = (1, 2, 3, 4, 5)


def _interp_traced_query_host_lookup(tab1, x, x_host, tab_x_np, xp,
                                     outside_value, side):
    """jax: traced query ``x`` whose values are known on the host
    (``x_host``). Panel lookup on numpy; the int32 indices and law codes
    are the only query-sized constants in the traced program and sit
    behind an optimisation barrier so XLA does not constant-fold the
    interpolation at compile time. Panel values are gathered on jax, so
    traced ordinates keep their gradient."""
    import jax
    from . import tab1_jax
    i, law = tab1_jax.host_lookup(
        tab_x_np, np.asarray(tab1.nbt), np.asarray(tab1.intp), x_host, side,
    )
    i, law = jax.lax.optimization_barrier(
        (xp.asarray(i.astype(np.int32)), xp.asarray(law)))
    tab_x = xp.asarray(tab_x_np)
    tab_y = xp.asarray(tab1.y, dtype=xp.float64)
    return tab1_jax.interp_from_lookup(
        law, x, tab_x[i - 1], tab_x[i], tab_y[i - 1], tab_y[i],
        xp.asarray(outside_value, dtype=xp.float64),
        laws=_static_laws(tab1.intp), side=side,
    )


def _host_arrays(x, tab1):
    """``(x, tab1.x, tab1.y)`` as float64 numpy arrays when the query
    and the table abscissae are concrete (``tab1.y`` slot ``None`` if
    the ordinates are traced); ``None`` if either of the former is a
    tracer. Non-1-D queries keep the generic path."""
    if getattr(x, 'ndim', None) != 1:
        return None
    try:
        x_np = np.asarray(x, dtype=np.float64)
        tab_x = np.asarray(tab1.x, dtype=np.float64)
    except Exception:          # tracer
        return None
    try:
        tab_y = np.asarray(tab1.y, dtype=np.float64)
    except Exception:
        tab_y = None
    return x_np, tab_x, tab_y


def _static_laws(intp):
    """The ENDF interpolation laws a TAB1 can select, as a sorted tuple,
    when its INT codes are concrete; all five otherwise (traced codes
    never occur in practice, but stay correct)."""
    try:
        codes = np.unique(np.asarray(intp) % 10)
    except Exception:
        return _ALL_LAWS
    return tuple(int(c) for c in codes if 1 <= int(c) <= 5) or _ALL_LAWS


def _apply_law_vectorised(law, x, x1, x2, y1, y2, xp, outside_value=0.0,
                          laws=_ALL_LAWS):
    """Vectorised element-wise dispatch across interpolation laws.

    Rather than `xp.switch` per element (branch-per-point, slow on
    every backend), compute all five candidate values and pick with
    `xp.select`. Wasted arithmetic on the branches not taken is
    dominated by the resonance-summation cost downstream and is
    the same order-of-magnitude as one extra ufunc pass.

    Backward-pass safety
    --------------------

    Reverse-mode autodiff (`jax.grad`) still evaluates the gradient
    of every branch even for elements that `xp.select` discards, then
    zeros it via the branch-mask cotangent. If any discarded branch
    can produce ``Inf`` (log of 0, division by 0) the multiplication
    with the 0-cotangent gives ``NaN`` and pollutes the gradient of
    elements that were routed to a completely different, well-behaved
    branch.

    To keep this safe: replace every unsafe input to a log or a
    division by a strictly positive constant BEFORE the arithmetic,
    not just after the fact. `x` is clamped to `EPS` before entering
    any log ratio; `x1` and `x2` are separated by at least
    `x1_safe * (1 + EPS)`; y-values are floored to `EPS`. The
    discarded branch's numerical value doesn't matter (select masks
    it out); what matters is that its VJP graph never touches Inf.

    ``laws`` (static) lists the ENDF laws 1..5 the table can select
    (see :func:`_static_laws`); branches for absent laws are not
    evaluated. MF3 tables are mostly lin-lin only, so this skips the
    two logs and two exps per point the full dispatch pays. The
    sentinels 0 / 1 / 6 (outside, degenerate panel) are always kept.
    Results are identical: an absent law's condition never holds.
    """
    EPS = 1e-30

    # Strictly positive inputs for the log / division branches. `x`
    # can legitimately be 0 (below-mesh queries, out-of-range routed
    # to law=0); we still need a positive stand-in for the log
    # branch so the reverse-mode graph stays finite.
    need_log_x = 3 in laws or 5 in laws
    need_log_y = 4 in laws or 5 in laws
    if need_log_x:
        x_safe = xp.maximum(x, EPS)
    x1_safe = xp.where(x1 <= 0.0, EPS, x1)
    x2_safe = xp.where(
        x2 <= x1_safe, x1_safe * (1.0 + EPS), x2,
    )
    if need_log_y:
        y1_safe = xp.where(y1 <= 0.0, EPS, y1)
        y2_safe = xp.where(y2 <= 0.0, EPS, y2)
    conds = [law == 0, law == 1, law == 6]
    outside = xp.full_like(x, outside_value)     # may be a traced scalar
    vals = [outside, y1, y2]
    # Law code 6 is a private-to-this-module marker used by
    # `interp(side='right')` to pick the right-limit (y2) at a
    # doubled-x discontinuity instead of the default constant-y1
    # collapse of law 1. Not an ENDF-6 law code; never surfaces
    # to the caller.

    # For lin-lin the denominator `x2 - x1` is 0 only when the panel
    # is degenerate, which we route to law=1 or law=6 anyway; guard
    # for both forward NaN and backward Inf with a safe denominator.
    # Note: the `x2_safe = x1_safe * (1 + EPS)` nudge above collapses
    # to `x1_safe` in float64 when EPS is below machine epsilon, so
    # we clamp the derived denominators explicitly here rather than
    # relying on the input-side nudge.
    if 2 in laws or 4 in laws:
        x2m1_safe = xp.maximum(x2_safe - x1_safe, EPS)             # > 0
    if 2 in laws:
        conds.append(law == 2)
        vals.append(y1 + (x - x1) * (y2 - y1) / x2m1_safe)
    if need_log_x:
        log_x_over_x1 = xp.log(x_safe / x1_safe)
        log_x2_over_x1_safe = xp.maximum(
            xp.log(x2_safe / x1_safe), EPS,
        )                                                          # > 0
    if need_log_y:
        log_y2_over_y1 = xp.log(y2_safe / y1_safe)
    if 3 in laws:
        conds.append(law == 3)
        vals.append(y1 + log_x_over_x1 * (y2 - y1) / log_x2_over_x1_safe)
    if 4 in laws:
        conds.append(law == 4)
        vals.append(y1_safe * xp.exp(
            (x - x1) * log_y2_over_y1 / x2m1_safe,
        ))
    if 5 in laws:
        conds.append(law == 5)
        vals.append(y1_safe * xp.exp(
            log_x_over_x1 * log_y2_over_y1 / log_x2_over_x1_safe,
        ))
    return xp.select(conds, vals, default=outside)
