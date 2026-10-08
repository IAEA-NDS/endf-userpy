"""JAX R-matrix accumulation for :mod:`mf2_interpretation_reichmoore`.

The backend-agnostic ``_reconstruct_group`` builds the R-matrix
from a dense ``(ne, nres)`` complex ``1 / (E_r - E - i Γ_γ / 2)``
intermediate, once per J·π group with out-of-group resonances
masked to zero. On JAX that costs ``ngroups x ne x nres`` complex
divisions and the full intermediate in memory (51 GB at 1 M
energies on U-235).

:func:`accumulate_r_matrix` replaces that build on the ``'jax'``
backend, in the spirit of the numba kernel's per-resonance
accumulator loop:

- ``lax.scan`` over **blocks** of ``block`` resonances with an
  ``(ngroups * nch * nch, ne)`` accumulator carry, so memory is
  ``O(block * ne)`` instead of ``O(nres * ne)``.
- Each block's contribution is a real ``(ngroups * nch * nch,
  block) @ (block, ne)`` matmul on the real and imaginary parts of
  the inverse denominator. The weight matrix scatters every
  resonance into its own group's slot (one-hot over groups), so
  each resonance is evaluated once rather than once per group.
- The scan body is wrapped in ``jax.checkpoint``: under
  ``jax.grad`` the backward pass recomputes each block's inverse
  denominator instead of storing all of them, which would bring
  back the ``(nres, ne)`` footprint.
- The function itself is ``jax.jit``-compiled (static ``ngroups`` /
  ``block``), so eager callers pay the trace + compile once per
  shape instead of once per call.

Every intermediate has a shape known at trace time, so
``jax.jit(reconstruct)`` and ``jax.grad`` through the resonance
parameters, the widths and the energy mesh keep working.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


MAX_BLOCK = 256


def block_size(ne: int, max_bytes: int) -> int:
    """Resonances per scan step such that one ``(block, ne)`` float64
    intermediate stays under ``max_bytes``, capped at
    :data:`MAX_BLOCK` (larger blocks stop paying off on CPU)."""
    return max(1, min(MAX_BLOCK, max_bytes // (8 * max(ne, 1))))


@partial(jax.jit, static_argnames=('ngroups', 'block'))
def accumulate_r_matrix(e, res_er, res_gg, res_group, gammas,
                        ngroups: int, block: int):
    """Per-group Reich-Moore R-matrix at every energy.

    ``R_{g,cc'}(E) = Σ_{r in g} γ_{r,c} γ_{r,c'} / (E_r - E - i Γ_γ,r / 2)``

    Parameters
    ----------
    e : (ne,) float64
        Energies (non-negative; the caller masks ``E <= 0``).
    res_er, res_gg : (nres,) float64
        Resonance energies and capture widths.
    res_group : (nres,) int
        J·π group index of each resonance.
    gammas : (nch, nres) float64
        Reduced-width amplitudes of each resonance's OWN group, zero
        for channels that group does not have.
    ngroups : int
        Number of J·π groups (static).
    block : int
        Resonances per scan step (static). See :func:`block_size`.

    Returns
    -------
    (ngroups, nch, nch, ne) complex128
        Symmetric in ``(c, c')``.
    """
    nch, nres = gammas.shape
    ne = e.shape[0]
    npair = nch * nch
    nslot = ngroups * npair

    # Weight of resonance r in slot (g, c, c'): γ_c γ_c' if r in g.
    pair_w = (gammas[:, None, :] * gammas[None, :, :]).reshape(npair, nres)
    onehot = (
        res_group.reshape(1, -1) == jnp.arange(ngroups).reshape(-1, 1)
    ).astype(pair_w.dtype)                                     # (ngroups, nres)
    w = (onehot[:, None, :] * pair_w[None, :, :]).reshape(nslot, nres)

    # Pad nres to a multiple of block with zero-weight resonances; a
    # unit capture width keeps their denominator away from zero.
    nblk = -(-nres // block)
    pad = nblk * block - nres
    er_b = jnp.pad(res_er, (0, pad)).reshape(nblk, block)
    hg_b = (0.5 * jnp.pad(res_gg, (0, pad), constant_values=1.0)
            ).reshape(nblk, block)
    w_b = jnp.pad(w, ((0, 0), (0, pad))).reshape(nslot, nblk, block)
    w_b = jnp.transpose(w_b, (1, 0, 2))                        # (nblk, nslot, block)

    @jax.checkpoint
    def body(carry, xs):
        acc_re, acc_im = carry
        er_k, hg_k, w_k = xs
        # 1 / (dE - i hg) = (dE + i hg) / (dE^2 + hg^2)
        de = er_k.reshape(-1, 1) - e.reshape(1, -1)            # (block, ne)
        hg = hg_k.reshape(-1, 1)
        den = de * de + hg * hg
        return (acc_re + w_k @ (de / den),
                acc_im + w_k @ (hg / den)), None

    zero = jnp.zeros((nslot, ne), dtype=e.dtype)
    (acc_re, acc_im), _ = lax.scan(body, (zero, zero), (er_b, hg_b, w_b))
    return (acc_re + 1j * acc_im).reshape(ngroups, nch, nch, ne)
