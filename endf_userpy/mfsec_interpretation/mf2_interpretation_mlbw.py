"""Multi-Level Breit-Wigner (MLBW) reconstruction for MF2/MT151.

Port of the JAX MLBW kernel (``physics.py:mlbw`` on branch
``feature_resonance``) to a pure-numpy natural-size implementation
that runs unchanged on the JAX backend via the
:mod:`endf_userpy.primitives.array_ns` adapter. Deliberately
avoids the JAX-only design pressures:

- No pre-allocated padding (``NRS_SIZE`` / ``MAX_CHN`` / ``NP_TAB1``).
  Resonance and channel arrays are their natural sizes.
- No integer-key channel encoding (``CHN_STRIDE * L + CHN_OFFSET
  + 2 * J``). Channels are indexed 0..nch-1 directly; the
  resonance-to-channel map is a plain integer array
  ``res_channel[nres]``.
- No ``lax.switch`` for per-element branching. Vectorised
  ``xp.select`` handles interpolation-law dispatch;
  :func:`factors.pnt_shf` handles the L-dependence via a
  Newton recurrence.
- Per-channel aggregation uses a single matmul against a small
  ``(nres, nch)`` one-hot matrix instead of a Python loop over
  energies calling :func:`segment_sum` per row. Same result;
  turns O(NE) Python dispatch into one BLAS call and drops the
  per-row copy of the ``(NE, nch)`` output matrix that made the
  loop's cost superlinear in NE.

Formalism (ENDF-6 D.1.3.4, MLBW):

    sigma_sct(E) = pi / k^2 * sum_c g_c ((1 - cos 2phi_c - A_c)^2
                                         + (sin 2phi_c + B_c)^2)
    sigma_cap(E) = 2 pi / k^2 * sum_c g_c C_c
    sigma_fis(E) = 2 pi / k^2 * sum_c g_c F_c
    sigma_pot(E) = 2 pi / k^2 * sum_c g_c (1 - cos 2phi_c)
    sigma_rxx(E) = 2 pi / k^2 * sum_c g_c X_c    (competitive)

Per-resonance ratio ``r_r = 2 Gamma_n / ((E - E_r')^2 + Gamma^2 / 4)``
with shifted resonance energy ``E_r' = E_r + (S_r - S_e) Gamma_n^0 /
2``. Channel contributions ``A_c / B_c / C_c / F_c / X_c`` accumulate
resonance ratios weighted by widths, summed via segment-sum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..primitives import tab1
from . import mf2_interpretation_factors as factors


@dataclass
class MLBWData:
    """Natural-size MLBW input for one isotope / one energy range.

    Fields:
        abn:  isotopic abundance in the material.
        spi:  target spin I (used for statistical weight and lx).
        ki:   wavenumber coefficient: k(E) = ki * sqrt(E).
        qx:   Q-value for the competitive width (0 if absent).
        r_a:  channel radius as TAB1 vs E (or single-panel constant).
        r_ap: scattering radius as TAB1 vs E.
        ch_l:      (nch,) L per channel.
        ch_g:      (nch,) statistical weight g(J) per channel.
        res_channel: (nres,) channel index per resonance.
        res_l:  (nres,) L per resonance (== ch_l[res_channel]; kept
                explicit for clarity and to avoid one indirection
                inside vectorised loops).
        res_er, res_gn, res_gg, res_gf, res_gx: (nres,)
                resonance energy, neutron / gamma / fission /
                competitive widths.
    """
    abn: float
    spi: float
    ki: float
    qx: float
    r_a: tab1.TAB1
    r_ap: tab1.TAB1
    ch_l: np.ndarray
    ch_g: np.ndarray
    res_channel: np.ndarray
    res_l: np.ndarray
    res_er: np.ndarray
    res_gn: np.ndarray
    res_gg: np.ndarray
    res_gf: np.ndarray
    res_gx: np.ndarray


# Machine-eps guards for the "resonance width is zero" and
# "penetration factor is zero" edge cases. Both branches would
# blow up as 0/0; we clamp to zero via `where` masks below.
_EPS = 1e-38


def _lx_values(l_vec, spi, xp):
    """Angular-momentum values for the competitive width.

    Approximation used by all major reconstruction codes (SAMMY,
    NJOY, PREPRO): for spin-zero targets the competitive channel
    is assumed to be l = |L - 2|; otherwise the same L.
    """
    return l_vec + (xp.abs(l_vec - 2) - l_vec) * (spi == 0)


_EPS_SQRT = 1e-38


def _safe_sqrt(x, xp):
    """Backward-safe ``sqrt``: returns 0 at x <= 0 and its gradient
    is 0 there too. Plain ``xp.sqrt(x)`` produces Inf backward at
    x=0 which combines with the zero subgradient of ``max(x, 0)``
    upstream to give NaN under ``jax.grad``.
    """
    x_pos = x > _EPS_SQRT
    x_safe = xp.where(x_pos, x, 1.0)
    return xp.where(x_pos, xp.sqrt(x_safe), 0.0)


# Memory-safety cap on the (chunk, nres) complex128 intermediate the
# numpy path materialises inside reconstruct. 512 MB → peak working set
# of ~2-3 GB with all the transient tensors (inv_denom, ratio, A_r, ...)
# on top; fits any reasonable machine and never freezes a laptop. Users
# who want tighter control can monkey-patch this or pass a custom cap
# via the private `_max_intermediate_bytes` arg of ``reconstruct``.
NUMPY_MAX_INTERMEDIATE_BYTES = 512 * 1024 * 1024


def _numpy_chunk_size(ne: int, nres: int, max_bytes: int) -> int:
    """Chunk size in NE that keeps `(chunk, nres) complex128` under
    `max_bytes`. Returns `ne` (no chunking) if the whole grid fits.
    """
    bytes_per_row = max(nres * 16, 1)
    if bytes_per_row * ne <= max_bytes:
        return ne
    return max(1, max_bytes // bytes_per_row)


def _rho(e, ki, r_tab, xp):
    """Channel radius parameter rho = k(|E|) * radius(|E|)."""
    ee = xp.abs(e)
    return ki * _safe_sqrt(ee, xp) * tab1.interp(r_tab, ee, xp)


def _rho_competitive(e, qx, ki, r_a_tab, xp):
    """Rho for the competitive channel: uses E + Q shifted argument."""
    ee = xp.maximum(e + qx, 0.0)
    return ki * _safe_sqrt(ee, xp) * tab1.interp(r_a_tab, ee, xp)


def _channel_sums_dense(e_safe, rho_e, rho_xe, er, gn0, gg, gf, gx0, shf_r,
                        lr, lx, ich, nch, xp, nl_max):
    """Per-channel sums of the MLBW ``A``, ``B``, capture, fission and
    competitive terms, each ``(ne, nch)``, from a dense ``(ne, nres)``
    evaluation of every per-resonance term."""
    # Penetration / shift at the query energy, per resonance's L:
    # rho_e is a per-energy scalar; L varies per resonance. Broadcast
    # rho_e to (ne, 1), lr to (1, nres); output (ne, nres).
    pnt_e, shf_e = factors.pnt_shf(
        rho_e.reshape(-1, 1), lr.reshape(1, -1), xp, nl_max,
    )   # (ne, nres)
    pntx_e, _ = factors.pnt_shf(
        rho_xe.reshape(-1, 1), lx.reshape(1, -1), xp, nl_max,
    )   # (ne, nres)

    # Widths at the query energy.
    gn_e = pnt_e * gn0.reshape(1, -1)                       # (ne, nres)
    gx_e = pntx_e * gx0.reshape(1, -1)                      # (ne, nres)
    gg_e = gg.reshape(1, -1)                                 # (ne, nres)
    gf_e = gf.reshape(1, -1)                                 # (ne, nres)
    gt = gn_e + gg_e + gf_e + gx_e

    # Shifted resonance energy.
    erp = er.reshape(1, -1) + 0.5 * (
        shf_r.reshape(1, -1) - shf_e
    ) * gn0.reshape(1, -1)

    de = 2.0 * (e_safe.reshape(-1, 1) - erp)
    denom = gt ** 2 + de ** 2
    # Same double-where guard as for gn0/gx0 above: keeps `where`'s
    # backward pass finite when denom happens to hit zero (dummy
    # resonances with all widths and de=0).
    denom_safe = xp.where(denom > _EPS, denom, 1.0)
    ratio = xp.where(denom > _EPS, 2.0 * gn_e / denom_safe, 0.0)

    A_r = ratio * gt
    B_r = ratio * de
    cap_r = ratio * gg_e
    fis_r = ratio * gf_e
    rxx_r = ratio * gx_e

    # --- Sum per-resonance contributions into per-channel arrays. ---
    #
    # Build the one-hot channel-membership matrix once. `G[r, c] = 1`
    # iff resonance r sits in channel c. Per-channel aggregation is
    # then one matmul per partial cross section:
    #     (ne, nres) @ (nres, nch)  ->  (ne, nch)
    # For an MLBW range with a handful of channels and a few hundred
    # resonances the one-hot is tiny (nch * nres); the matmul reduces
    # to one BLAS call on numpy and traces cleanly on JAX.

    G = (xp.arange(nch).reshape(1, -1) == ich.reshape(-1, 1)).astype(
        A_r.dtype,
    )   # (nres, nch)

    A_ch = A_r @ G
    B_ch = B_r @ G
    cap_ch = cap_r @ G
    fis_ch = fis_r @ G
    rxx_ch = rxx_r @ G
    return A_ch, B_ch, cap_ch, fis_ch, rxx_ch


def _channel_sums_jax(data, e_safe, rho_e, rho_xe, er, gn0, gg, gf, gx0,
                      shf_r, lr, lx, ich, nch, xp, nl_max,
                      max_intermediate_bytes=None):
    """JAX counterpart of :func:`_channel_sums_dense`: the E-dependent
    penetration / shift factors are evaluated once per distinct L and
    the resonance sums accumulated block by block by
    :func:`mf2_interpretation_mlbw_jax.accumulate_channel_sums`.
    ``L`` is concrete (channel bookkeeping), so its distinct values
    are known at trace time. The competitive ``lx`` is ``|L - 2|`` or
    ``L`` depending on ``spi == 0``, and ``spi`` may be a tracer (dict
    leaf traced for autodiff, or the dataclass built inside a
    ``jax.jit``), so both candidates are tabulated and each
    resonance's row is selected with ``xp.where``."""
    from . import mf2_interpretation_mlbw_jax as _jax
    from . import mf2_interpretation_reichmoore_jax as _rm_jax
    l_np = np.asarray(data.res_l, dtype=np.int32)
    if l_np.shape[0] == 0:
        z = xp.zeros((e_safe.shape[0], int(nch)), dtype=xp.float64)
        return z, z, z, z, z
    l_u = np.unique(l_np)
    lx_alt = np.abs(l_np - 2)                   # lx for a spin-0 target
    lx_u = np.unique(np.concatenate([l_u, np.abs(l_u - 2)]))
    pnt_e_l, shf_e_l = factors.pnt_shf(
        rho_e.reshape(1, -1), xp.asarray(l_u).reshape(-1, 1), xp, nl_max,
    )   # (nL, ne)
    pntx_e_l, _ = factors.pnt_shf(
        rho_xe.reshape(1, -1), xp.asarray(lx_u).reshape(-1, 1), xp, nl_max,
    )   # (nLx, ne)
    max_bytes = (max_intermediate_bytes
                 if max_intermediate_bytes is not None
                 else NUMPY_MAX_INTERMEDIATE_BYTES)
    acc = _jax.accumulate_channel_sums(
        e_safe, er, gn0, gg, gf, gx0, shf_r,
        xp.asarray(np.searchsorted(l_u, l_np)),
        xp.where(xp.asarray(data.spi) == 0,
                 xp.asarray(np.searchsorted(lx_u, lx_alt)),
                 xp.asarray(np.searchsorted(lx_u, l_np))),
        ich, pnt_e_l, shf_e_l, pntx_e_l,
        nch=int(nch),
        block=_rm_jax.block_size(int(e_safe.shape[0]), max_bytes),
    )   # (5, nch, ne)
    return tuple(xp.transpose(acc[q]) for q in range(5))


def reconstruct(
    data: MLBWData, energies_in, xp, nl_max: int = 8,
    _max_intermediate_bytes: int = None,
    _skip_chunk: bool = False,
):
    """MLBW cross sections at ``energies_in`` (any 1D array).

    Returns a dict with keys sct, cap, fis, pot, rxx, tot, each of
    shape ``(len(energies_in),)``, in barn (with the ki convention).

    Parameters
    ----------
    data : MLBWData
        Preprocessed MLBW input for the range (one isotope, one
        energy range). See :class:`MLBWData`.
    energies_in : array_like
        Incident-neutron energies. Cast to the backend's float64
        via ``xp.asarray``.
    xp : backend
        As returned by :func:`array_ns.get_backend`.
    nl_max : int, optional
        Cap on the L-value Newton recurrence for penetration and
        shift factors. Default 8 matches the JAX prototype. Files
        with higher L in their resonance table (rare) need this
        raised. Ignored on the numba backend, which supports L<=5
        only (closed forms; raises for higher L).

    Notes
    -----
    On the numpy backend, ``reconstruct`` automatically chunks
    ``energies_in`` if the ``(NE, nres)`` complex128 intermediate
    the vectorised formula would materialise exceeds
    :data:`NUMPY_MAX_INTERMEDIATE_BYTES` (default 512 MB).
    Transparent to the caller; result is bit-comparable to a
    single-shot reconstruction on a machine with enough RAM.
    Neither the JAX nor the numba path triggers this: numba
    processes per-energy in parallel with no ``(NE, nres)``
    materialisation, and on JAX the per-channel resonance sums are
    accumulated by a blocked ``lax.scan`` over the resonance axis
    (:func:`mf2_interpretation_mlbw_jax.accumulate_channel_sums`)
    whose memory is ``O(block * NE)``; there
    ``_max_intermediate_bytes`` caps one ``(block, NE)`` float64
    block.

    Private args
    ------------
    ``_max_intermediate_bytes`` and ``_skip_chunk`` are for
    testing the chunking behaviour with a small threshold; users
    shouldn't rely on them.
    """
    xp.raise_if_needed_but_missing('numba')
    if xp.wants_accelerator('numba') and xp.accelerator_available('numba'):
        # Route to the hand-written @njit kernel. Same physics, same
        # inputs, same outputs. See
        # :mod:`mf2_interpretation_mlbw_numba` for the trade-offs.
        from . import mf2_interpretation_mlbw_numba as _numba
        return _numba.reconstruct(data, energies_in)

    e = xp.asarray(energies_in, dtype=xp.float64)

    # Numpy memory safety: split NE into slices whose (chunk, nres)
    # intermediate fits under the cap, recurse into ourselves with
    # `_skip_chunk` to bypass the wrapper on each slice.
    if not _skip_chunk and getattr(xp, 'name', None) == 'numpy':
        max_bytes = (_max_intermediate_bytes
                     if _max_intermediate_bytes is not None
                     else NUMPY_MAX_INTERMEDIATE_BYTES)
        nres = int(data.res_er.shape[0])
        chunk = _numpy_chunk_size(int(e.shape[0]), nres, max_bytes)
        if chunk < e.shape[0]:
            parts: dict = {}
            for start in range(0, int(e.shape[0]), chunk):
                sub = reconstruct(
                    data, e[start:start + chunk], xp, nl_max,
                    _max_intermediate_bytes=max_bytes,
                    _skip_chunk=True,
                )
                for k, v in sub.items():
                    parts.setdefault(k, []).append(v)
            return {k: np.concatenate(v) for k, v in parts.items()}

    er = xp.asarray(data.res_er, dtype=xp.float64)
    gn = xp.asarray(data.res_gn, dtype=xp.float64)
    gg = xp.asarray(data.res_gg, dtype=xp.float64)
    gf = xp.asarray(data.res_gf, dtype=xp.float64)
    gx = xp.asarray(data.res_gx, dtype=xp.float64)
    lr = xp.asarray(data.res_l, dtype=xp.int32)
    ich = xp.asarray(data.res_channel, dtype=xp.int32)
    ch_l = xp.asarray(data.ch_l, dtype=xp.int32)
    ch_g = xp.asarray(data.ch_g, dtype=xp.float64)
    nch = ch_l.shape[0]

    # --- Per-resonance quantities that depend only on E_r. ---
    #
    # Penetration and shift factors P_L(rho(E_r)) and S_L(rho(E_r));
    # the reduced widths gamma^0_n / gamma^0_x are (Gamma / P), used
    # below to build the energy-dependent widths P(E) * gamma^0.

    rho_r = _rho(er, data.ki, data.r_a, xp)                # (nres,)
    pnt_r, shf_r = factors.pnt_shf(rho_r, lr, xp, nl_max)  # (nres,)

    lx = _lx_values(lr, data.spi, xp)
    rho_xr = _rho_competitive(er, data.qx, data.ki, data.r_a, xp)
    pntx_r, _ = factors.pnt_shf(rho_xr, lx, xp, nl_max)     # (nres,)

    # `xp.where(cond, a/b, 0)` leaks NaN through the backward pass
    # when `b==0` even where `cond` is False -- the "unused" branch
    # of `where` still contributes to the gradient. Standard "double
    # where" idiom: replace the divisor with 1 wherever the guard
    # would discard the result anyway.
    pnt_r_safe = xp.where(pnt_r > _EPS, pnt_r, 1.0)
    pntx_r_safe = xp.where(pntx_r > _EPS, pntx_r, 1.0)
    gn0 = xp.where(pnt_r > _EPS, gn / pnt_r_safe, 0.0)
    gx0 = xp.where(pntx_r > _EPS, gx / pntx_r_safe, 0.0)

    # --- Per-energy factors (broadcast to (ne, nres) below). ---

    e_safe = xp.maximum(e, 0.0)
    e_pos = e > 0.0                                          # (ne,)
    k_e = data.ki * xp.sqrt(xp.where(e_pos, e_safe, 1.0))    # (ne,)
    inv_k2 = xp.where(e_pos, xp.pi / (k_e * k_e), 0.0)       # (ne,)
    pi_k2 = data.abn * inv_k2                                # (ne,)
    twopi_k2 = 2.0 * pi_k2                                   # (ne,)

    rho_e = _rho(e_safe, data.ki, data.r_a, xp)              # (ne,)
    rho_xe = _rho_competitive(e_safe, data.qx, data.ki, data.r_a, xp)   # (ne,)
    if getattr(xp, 'name', None) == 'jax':
        A_ch, B_ch, cap_ch, fis_ch, rxx_ch = _channel_sums_jax(
            data, e_safe, rho_e, rho_xe, er, gn0, gg, gf, gx0, shf_r,
            lr, lx, ich, nch, xp, nl_max, _max_intermediate_bytes,
        )
    else:
        A_ch, B_ch, cap_ch, fis_ch, rxx_ch = _channel_sums_dense(
            e_safe, rho_e, rho_xe, er, gn0, gg, gf, gx0, shf_r,
            lr, lx, ich, nch, xp, nl_max,
        )

    # --- Hard-sphere potential-scattering phase per channel. ---
    rho_s = _rho(e_safe, data.ki, data.r_ap, xp)            # (ne,)
    phi2 = 2.0 * factors.phase(
        rho_s.reshape(-1, 1), ch_l.reshape(1, -1), xp, nl_max,
    )   # (ne, nch)
    pot_ch = 1.0 - xp.cos(phi2)                             # (ne, nch)

    # --- Assemble by channel and reduce over channels. ---
    sct_ch = (pot_ch - A_ch) ** 2 + (xp.sin(phi2) + B_ch) ** 2   # (ne, nch)
    gmask = ch_g.reshape(1, -1)                             # (1, nch)

    sct = xp.sum(sct_ch * gmask, axis=1) * pi_k2
    cap = xp.sum(cap_ch * gmask, axis=1) * twopi_k2
    fis = xp.sum(fis_ch * gmask, axis=1) * twopi_k2
    pot = xp.sum(pot_ch * gmask, axis=1) * twopi_k2
    rxx = xp.sum(rxx_ch * gmask, axis=1) * twopi_k2
    tot = sct + cap + fis + rxx

    return {'sct': sct, 'cap': cap, 'fis': fis, 'pot': pot,
            'rxx': rxx, 'tot': tot}
