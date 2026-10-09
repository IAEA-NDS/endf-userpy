"""JAX fluctuation-integral moments for :mod:`mf2_interpretation_urr`.

The backend-agnostic ``reconstruct`` evaluates the default
Gauss-Legendre-32 fluctuation integral on dense ``(NE, nJ, Nq)``
arrays: eight ``_channel_factor`` calls, each computing both a
``pow`` and an ``exp`` at every node. Under eager JAX that
materialises a dozen ``(NE, nJ, 32)`` temporaries (~37 GB at 1 M
energies with 12 spin groups) and is transcendental-bound.
:func:`fluctuation_moments` replaces it on the ``'jax'`` backend:

- **Energy chunks.** ``lax.map`` over chunks of :data:`CHUNK`
  energies; each chunk's ``(CHUNK, nJ, Nq)`` block stays in cache
  and is reduced over the nodes before the next chunk starts, so
  memory is ``O(NE * nJ)``. The chunk body is ``jax.checkpoint``-ed
  so ``jax.grad`` recomputes chunks instead of storing every
  chunk's node-resolved residuals.
- **One power per channel.** The three moment orders share their
  base: ``g1 = g0 / base`` and ``g2 = (1 + 2/ν) g1 / base`` with
  ``g0 = base^(-ν/2)``, instead of three independent ``pow``.
- **Per-channel specialisation on the concrete DOF ν** (static,
  read from the file's AMUN / AMUG / AMUF / AMUX):

  - ``'exp'``: every group has ν = 0 (deterministic width; always
    the case for capture in ENDF/B-VIII.1): ``g = exp(-tΓ)``.
  - ``'int'`` / ``'int0'``: ν in {1, 2, 3, 4} (``'int0'``: some
    groups also ν = 0): ``base^(-ν/2) = rsqrt(base)^ν`` from one
    ``rsqrt`` and multiplies, no ``log`` / ``pow``.
  - ``'zero'``: the channel's width table is identically zero
    (fission and competitive in ~70% / ~80% of ENDF/B-VIII.1 URR
    ranges): every factor is 1 and the channel's own moment is 0,
    so its work is skipped. Only for a concrete table; under
    ``jax.grad`` wrt that table the full path runs.
  - ``'general'``: anything else (non-integer AMUX occurs in ~50
    ENDF/B-VIII.1 groups, or a traced ν):
    ``g0 = exp(-ν/2 · log(base))``.

The integrand and node set are the numpy path's; results agree with
it to ~1e-15 relative.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


CHUNK = 256
"""Energies per ``lax.map`` step. 256 x 12 groups x 32 nodes keeps a
block in L2; measured fastest of 64 / 256 / 1024 / 4096 on Nd-143."""

_EPS = 1e-38
_INT_NU = (1.0, 2.0, 3.0, 4.0)


def channel_mode(nu, table=None) -> str:
    """Static evaluation mode for one channel from its ``(nJ,)`` DOF
    row and, optionally, its ``(nJ, NE_tab)`` width table. See the
    module docstring."""
    from .mf2_interpretation_urr import _is_jax_tracer
    if (table is not None and not _is_jax_tracer(table)
            and np.all(np.asarray(table) == 0.0)):
        return 'zero'
    if _is_jax_tracer(nu):
        return 'general'
    v = np.asarray(nu, dtype=np.float64)
    if np.all(v == 0.0):
        return 'exp'
    if np.all(np.isin(v, _INT_NU)):
        return 'int'
    if np.all(np.isin(v, (0.0,) + _INT_NU)):
        return 'int0'
    return 'general'


def _channel_factors(alpha, nu, t, mode):
    """Order-0/1/2 fluctuation factors ``(g0, g1, g2)`` of one channel
    (same definitions as ``mf2_interpretation_urr._channel_factor``).
    ``alpha`` / ``nu`` broadcast against the node axis of ``t``."""
    if mode == 'zero':
        return 1.0, 1.0, 1.0
    if mode == 'exp':
        g = jnp.exp(-(t * alpha))
        return g, g, g
    is_zero = nu <= _EPS
    nu_safe = jnp.where(is_zero, 1.0, nu)
    # Divisions hoisted off the node axis: base = 1 + 2 t α / ν and
    # inv = 1 / base comes for free from rsqrt in the integer modes.
    base = 1.0 + t * (2.0 * alpha / nu_safe)
    if mode == 'general':
        inv = 1.0 / base
        g0 = jnp.exp(jnp.where(is_zero, -(t * alpha),
                               (-nu / 2.0) * jnp.log(base)))
    else:
        r = lax.rsqrt(base)
        inv = r * r
        g0 = jnp.where(nu == 1.0, r,
                       jnp.where(nu == 2.0, inv,
                                 jnp.where(nu == 3.0, inv * r, inv * inv)))
        if mode == 'int0':
            g0 = jnp.where(is_zero, jnp.exp(-(t * alpha)), g0)
    g1 = jnp.where(is_zero, g0, g0 * inv)
    g2 = jnp.where(is_zero, g0, (1.0 + 2.0 / nu_safe) * g1 * inv)
    return g0, g1, g2


def _moments_block(alpha_n, alpha_g, alpha_f, alpha_x,
                   nu_n, nu_g, nu_f, nu_x, t, w, modes):
    """Unnormalised moments ``(R_ncap, R_nfis, R_ncomp, R_nn)`` for an
    ``(n, nJ)`` block of energies, reduced over the ``Nq`` nodes."""
    def fac(a, nu, m):
        return _channel_factors(a[..., None], nu[..., None], t, m)

    _, g1_n, g2_n = fac(alpha_n, nu_n, modes[0])
    g0_g, g1_g, _ = fac(alpha_g, nu_g, modes[1])
    g0_f, g1_f, _ = fac(alpha_f, nu_f, modes[2])
    g0_x, g1_x, _ = fac(alpha_x, nu_x, modes[3])
    zero = jnp.zeros_like(alpha_n)

    def moment(a, b, m, channel_mode):
        if channel_mode == 'zero':
            return zero
        return a * b * jnp.sum(m * w, axis=-1)

    return (
        moment(alpha_n, alpha_g, g1_n * g1_g * g0_f * g0_x, modes[1]),
        moment(alpha_n, alpha_f, g1_n * g0_g * g1_f * g0_x, modes[2]),
        moment(alpha_n, alpha_x, g1_n * g0_g * g0_f * g1_x, modes[3]),
        moment(alpha_n, alpha_n, g2_n * g0_g * g0_f * g0_x, modes[0]),
    )


@partial(jax.jit, static_argnames=('modes', 'chunk'))
def fluctuation_moments(alpha_n, alpha_g, alpha_f, alpha_x,
                        nu_n, nu_g, nu_f, nu_x, t_nodes, w_t,
                        modes, chunk=CHUNK):
    """Fluctuation-averaged moments ``R_ncap, R_nfis, R_ncomp, R_nn``,
    each ``(NE, nJ)``, for mean widths ``alpha_*`` of shape
    ``(NE, nJ)`` and DOF rows ``nu_*`` of shape ``(1, nJ)``.

    ``modes`` is the static 4-tuple of :func:`channel_mode` for the
    (n, γ, f, x) channels.
    """
    ne, nj = alpha_n.shape
    chunk = max(1, min(chunk, ne))
    nc = -(-ne // chunk)
    pad = nc * chunk - ne

    def split(a):   # edge-pad so padded rows stay in-domain
        a = jnp.pad(a, ((0, pad), (0, 0)), mode='edge')
        return a.reshape(nc, chunk, nj)

    @jax.checkpoint
    def body(x):
        return _moments_block(*x, nu_n, nu_g, nu_f, nu_x,
                              t_nodes, w_t, modes)

    out = lax.map(body, tuple(split(a) for a in
                              (alpha_n, alpha_g, alpha_f, alpha_x)))
    return tuple(o.reshape(nc * chunk, nj)[:ne] for o in out)
