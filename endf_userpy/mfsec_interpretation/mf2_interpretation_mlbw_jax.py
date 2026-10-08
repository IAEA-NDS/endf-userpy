"""JAX per-channel resonance sums for :mod:`mf2_interpretation_mlbw`.

The backend-agnostic ``reconstruct`` evaluates every per-resonance
term on a dense ``(ne, nres)`` grid, including the penetration and
shift factors ``P_L(rho(E))`` / ``S_L(rho(E))`` once per resonance
even though they only depend on the resonance's ``L``.
:func:`accumulate_channel_sums` replaces that on the ``'jax'``
backend, following :mod:`mf2_interpretation_reichmoore_jax`:

- The caller evaluates the E-dependent factors once per distinct
  ``L`` (shape ``(nL, ne)``); each scan step gathers the rows for
  its resonances.
- ``lax.scan`` over blocks of ``block`` resonances with a
  ``(5, nch, ne)`` accumulator carry for the per-channel ``A``,
  ``B``, capture, fission and competitive sums. Each term's block
  contribution is a real ``(nch, block) @ (block, ne)`` matmul
  against the resonance-to-channel one-hot (five plain matmuls
  measured ~30% faster than one batched einsum). Memory is
  ``O(block * ne)`` instead of ``O(nres * ne)``.
- The scan body is wrapped in ``jax.checkpoint`` so ``jax.grad``
  recomputes each block instead of storing all of them, and the
  function is ``jax.jit``-compiled with static ``(nch, block)``.

Shapes are all known at trace time, so ``jax.jit(reconstruct)``
and ``jax.grad`` through the resonance parameters, the radii and
the energy mesh keep working.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

_EPS = 1e-38


@partial(jax.jit, static_argnames=('nch', 'block'))
def accumulate_channel_sums(e, res_er, gn0, res_gg, res_gf, gx0, shf_r,
                            res_li, res_lxi, res_channel,
                            pnt_e_l, shf_e_l, pntx_e_l,
                            nch: int, block: int):
    """MLBW per-channel resonance sums at every energy.

    With ``Γ_n(E) = P_L(E) γ⁰_n``, ``Γ_x(E) = P_lx(E) γ⁰_x``,
    ``Γ = Γ_n + Γ_γ + Γ_f + Γ_x``, ``E_r' = E_r + (S_L(|E_r|) -
    S_L(E)) γ⁰_n / 2``, ``d = 2 (E - E_r')`` and ``ratio = 2 Γ_n /
    (Γ² + d²)``, returns the sums over each channel's resonances of
    ``ratio * (Γ, d, Γ_γ, Γ_f, Γ_x)``.

    Parameters
    ----------
    e : (ne,) float64
        Non-negative energies.
    res_er, gn0, res_gg, res_gf, gx0, shf_r : (nres,) float64
        Resonance energy, reduced neutron width, capture / fission
        widths, reduced competitive width, shift factor at ``|E_r|``.
    res_li, res_lxi : (nres,) int
        Row of each resonance's ``L`` / competitive ``lx`` in the
        per-L tables below.
    res_channel : (nres,) int
        Channel index of each resonance.
    pnt_e_l, shf_e_l : (nL, ne) float64
        ``P_L(rho(E))``, ``S_L(rho(E))`` per distinct ``L``.
    pntx_e_l : (nLx, ne) float64
        Competitive-channel penetration per distinct ``lx``.
    nch, block : int
        Number of channels and resonances per scan step (static).

    Returns
    -------
    (5, nch, ne) float64
        Per-channel sums of ``A``, ``B``, capture, fission,
        competitive terms.
    """
    nres = res_er.shape[0]
    ne = e.shape[0]
    onehot = (
        res_channel.reshape(1, -1) == jnp.arange(nch).reshape(-1, 1)
    ).astype(e.dtype)                                          # (nch, nres)

    # Pad to a multiple of block with resonances whose reduced neutron
    # width is 0: their ratio, hence every contribution, vanishes.
    nblk = -(-nres // block)
    pad = nblk * block - nres

    def _blk(a, fill=0):
        return jnp.pad(a, (0, pad), constant_values=fill).reshape(nblk, block)

    xs = (
        _blk(res_er), _blk(gn0), _blk(res_gg, 1.0), _blk(res_gf),
        _blk(gx0), _blk(shf_r), _blk(res_li), _blk(res_lxi),
        jnp.transpose(
            jnp.pad(onehot, ((0, 0), (0, pad))).reshape(nch, nblk, block),
            (1, 0, 2),
        ),                                                     # (nblk, nch, block)
    )

    @jax.checkpoint
    def body(acc, x):
        er, gn0_k, gg, gf, gx0_k, shf_r_k, li, lxi, oh = x
        col = lambda a: a.reshape(-1, 1)                       # noqa: E731
        gn_e = pnt_e_l[li] * col(gn0_k)                        # (block, ne)
        gx_e = pntx_e_l[lxi] * col(gx0_k)
        gt = gn_e + col(gg) + col(gf) + gx_e
        erp = col(er) + 0.5 * (col(shf_r_k) - shf_e_l[li]) * col(gn0_k)
        de = 2.0 * (e.reshape(1, -1) - erp)
        denom = gt * gt + de * de
        denom_safe = jnp.where(denom > _EPS, denom, 1.0)
        ratio = jnp.where(denom > _EPS, 2.0 * gn_e / denom_safe, 0.0)
        # One (nch, block) @ (block, ne) matmul per term.
        return acc + jnp.stack([
            oh @ (ratio * gt), oh @ (ratio * de), oh @ (ratio * col(gg)),
            oh @ (ratio * col(gf)), oh @ (ratio * gx_e),
        ]), None

    acc0 = jnp.zeros((5, nch, ne), dtype=e.dtype)
    acc, _ = lax.scan(body, acc0, xs)
    return acc
