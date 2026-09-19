"""Reich-Moore (LRF=3) resonance reconstruction for MF2/MT151.

Same 3-backend design as :mod:`mf2_interpretation_mlbw`: physics
written once against the :mod:`endf_userpy.primitives.array_ns`
adapter, so numpy and JAX share one implementation via
``get_backend('numpy'|'jax')``; a hand-written ``@njit`` kernel for
the ``'numba'`` backend follows in a sibling module (planned).

Formulation
-----------

Follows the SAMMY / NJOY-reconr convention with boundary condition
``B_c = 0`` (LSSF=0, which is the default and covers the majority
of ENDF-6 R-M evaluations). Per J·π group at energy E:

- **Reduced-width amplitudes.** For the elastic channel (c=0),
  ``γ_{r,0} = sign(GN_r) * sqrt(GN_r / (2 * P_L(|E_r|)))``.
  For each fission channel c (c=1..nfis),
  ``γ_{r,c} = sign(GF_{r,c}) * sqrt(|GF_{r,c}| / 2)``  (P=1
  for fission channels by convention).

- **R-matrix.** Complex ``(nch, nch)`` matrix,
  ``R_{cc'}(E) = Σ_r γ_{r,c} γ_{r,c'} / (E_r - E - i Γ_γ,r / 2)``.
  Capture is folded into the imaginary part of the denominator
  (the Reich-Moore approximation) instead of appearing as an
  explicit channel.

- **U-matrix.** ``X = (I - i R P)^{-1} R``,
  ``U_{cc'} = Ω_c Ω_{c'} [δ_{cc'} + 2 i √P_c √P_{c'} X_{cc'}]``,
  where ``Ω_c = exp(-i φ_c)`` for the elastic channel and
  ``Ω_c = 1`` for fission channels (no hard-sphere phase).

- **Cross sections** (Lane-Thomas):
  ``σ_scat_group = (π/k²) g_J |1 - U_{00}|²``
  ``σ_fis_group  = (π/k²) g_J Σ_{c∈fis} |U_{0c}|²``
  ``σ_abs_group  = (π/k²) g_J [1 - Σ_c |U_{0c}|²]``
  ``σ_cap_group  = σ_abs_group - σ_fis_group``
  Per-group results are summed with the group's statistical weight
  ``g_J = (2J + 1) / (2 (2I + 1))``.

Not covered by this sketch (deliberate scope):

- **Shift-factor subtraction ``S_c(E) - S_c(|E_r|)``**: the sketch
  uses ``L̃_c(E) = i P_c(E)`` (LSSF=0-with-shift-absorbed
  approximation). Adequate for narrow resonances / regions far
  from strong s-wave interferences; can be extended by adding a
  per-resonance ``E_r``-anchored shift when a real case shows a
  visible discrepancy against NJOY / SAMMY.
- **``LSSF != 0``**: alternate boundary condition. Adds one term to
  the L-matrix diagonal; can be added when a real case demands it.
- **URR (LRU=2)**: unresolved region; separate module.
- **ENDF-6 -> RMData preprocessing**: the caller supplies the
  natural-size dataclass. A helper that lifts ``d2_151`` into
  ``RMData`` is the natural next step.

JIT / autodiff notes
--------------------

The per-J·π group loop uses a fixed-shape ``(nres, ngroups)``
indicator mask (Option A) rather than Python-side boolean
indexing to select each group's resonances. That is what makes
``jax.jit(reconstruct)`` and ``jax.grad(...)`` work as expected:
every intermediate has a shape known at trace time. Trade-off:
every resonance contributes to every group's R-matrix sum (masked
to zero for non-members), so per-(c,c') work grows by a factor of
``ngroups`` (2-3 for typical actinide RM files). Memory for the
``(ne, nres)`` inv_denom intermediate is the same either way.

Sensitivity workflows built on top of this (``jax.grad`` /
``jax.jacrev`` for parameter fits, ``chunked_chi2`` for
memory-bounded gradients under fitting loops) therefore work
without a fork of the physics code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..primitives import tab1
from . import mf2_interpretation_factors as factors


_EPS = 1e-38


@dataclass
class RMData:
    """Natural-size Reich-Moore input for one isotope / one range.

    The layout mirrors :class:`~mf2_interpretation_mlbw.MLBWData`:
    scalars for the range, small per-channel arrays, per-resonance
    arrays. Reich-Moore has one extra structural axis (channels are
    grouped by J·π) which we encode with a per-resonance
    ``res_group`` index and per-group scalars.

    Fields:

    Scalars (per range):
        ``abn``:  isotopic abundance in the material.
        ``spi``:  target spin ``I`` (for the statistical weight).
        ``ki``:   wavenumber coefficient: ``k(E) = ki * sqrt(E)``.
        ``r_a``:  channel radius as TAB1 vs E.
        ``r_ap``: scattering radius as TAB1 vs E.

    Per J·π group (shape ``(ngroups,)``):
        ``group_l``:    elastic channel L per group. All elastic
                        channels within a group share this L (they
                        differ only in ``J``, which is absorbed into
                        ``g_J``).
        ``group_g``:    statistical weight ``g_J`` per group.
        ``group_nfis``: number of fission channels in the group
                        (0, 1, or 2).

    Per resonance (shape ``(nres,)``):
        ``res_group``: group index in ``[0, ngroups)``.
        ``res_er``:    resonance energy ``E_r``.
        ``res_gn``:    ``Γ_n`` at ``|E_r|`` (signed; sign carries
                       the reduced-width-amplitude sign).
        ``res_gg``:    radiative capture width ``Γ_γ``.
        ``res_gf1``:   ``Γ_{f,1}`` (signed; 0 if group has no
                       fission channels).
        ``res_gf2``:   ``Γ_{f,2}`` (signed; 0 if group has < 2
                       fission channels).
    """
    abn: float
    spi: float
    ki: float
    r_a: tab1.TAB1
    r_ap: tab1.TAB1
    group_l: np.ndarray
    group_g: np.ndarray
    group_nfis: np.ndarray
    res_group: np.ndarray
    res_er: np.ndarray
    res_gn: np.ndarray
    res_gg: np.ndarray
    res_gf1: np.ndarray
    res_gf2: np.ndarray


def _signed_sqrt(x, xp):
    """``sign(x) * sqrt(|x|)``: reduced-width amplitude from a signed
    width. Zero-safe."""
    ax = xp.abs(x)
    return xp.sign(x) * xp.sqrt(ax)


def _rho(e, ki, r_tab, xp):
    """Channel radius parameter ``rho = k(|E|) * radius(|E|)``."""
    ee = xp.abs(e)
    return ki * xp.sqrt(ee) * tab1.interp(r_tab, ee, xp)


def _reconstruct_group(
    e_safe, e_pos, k_e2, pi_k2,
    group_l, group_g, group_nfis,
    res_er, res_gn, res_gg, res_gf1, res_gf2,
    group_mask,
    ki, r_a, r_ap, xp,
):
    """Reconstruct one J·π group's contribution to sct/cap/fis.

    ``res_*`` arrays here are the FULL-LENGTH ``(nres,)`` per-resonance
    arrays across all groups. ``group_mask`` is a ``(nres,)`` indicator
    (0.0 / 1.0) marking which resonances belong to THIS group; the
    reduced-width amplitudes for out-of-group resonances get zeroed
    via this mask so their R-matrix contributions vanish.

    Keeping full-length arrays here (instead of Python-side boolean
    indexing to slice out this group's rows) is what lets the whole
    ``reconstruct`` traced under ``jax.jit`` -- data-dependent shapes
    from a boolean mask are the exact thing JAX's tracer refuses.

    ``group_l``, ``group_g``, ``group_nfis`` are group scalars
    (Python-level ints/floats at trace time).

    Returns three ``(ne,)`` arrays: ``sct``, ``cap``, ``fis``, each
    already multiplied by the group statistical weight and the
    ``π/k²`` prefactor.
    """
    ne = e_safe.shape[0]
    L = int(group_l)
    nfis = int(group_nfis)
    nch = 1 + nfis
    L_scalar = xp.asarray(L)   # 0-d array; factors.pnt_shf broadcasts against it

    # --- Elastic-channel factors at E and at |E_r|. ---
    rho_e = _rho(e_safe, ki, r_a, xp)                          # (ne,)
    p_e, _ = factors.pnt_shf(rho_e, L_scalar, xp)              # (ne,)

    rho_r = _rho(xp.abs(res_er), ki, r_a, xp)                  # (nres,)
    p_r, _ = factors.pnt_shf(rho_r, L_scalar, xp)              # (nres,)

    # Elastic reduced-width amplitude gamma_n0. Sign of GN matters.
    # gamma_{r,0} = sign(gn) * sqrt(|gn| / (2 * P_L(|E_r|))).
    #
    # Multiply by `group_mask` at the end: for resonances not in this
    # group we compute P_L with this group's L and (potentially wrong)
    # r_a, so the intermediate gamma is meaningless, but the mask
    # zeros it out before it can pollute the R-matrix sum. The extra
    # `xp.where` on the mask side lets us avoid dividing by whatever
    # tiny P_L we computed for out-of-group rows.
    denom = xp.where(p_r > _EPS, 2.0 * p_r, 1.0)
    gamma0 = xp.where(
        p_r > _EPS,
        _signed_sqrt(res_gn, xp) / xp.sqrt(denom),
        0.0,
    )   # (nres,)
    gamma0 = gamma0 * group_mask

    # --- Fission reduced-width amplitudes (P=1 for fission channels). ---
    # Build as list to keep the code readable at nch=1..3. Fission
    # widths are per-channel and each is also masked to this group.
    gammas = [gamma0]
    if nfis >= 1:
        gammas.append(
            (_signed_sqrt(res_gf1, xp) / xp.sqrt(2.0)) * group_mask
        )
    if nfis >= 2:
        gammas.append(
            (_signed_sqrt(res_gf2, xp) / xp.sqrt(2.0)) * group_mask
        )
    # gammas: list of (nres,) arrays, length nch

    # --- R-matrix (ne, nch, nch) complex. ---
    #   R_{cc'}(E) = Σ_r gamma_{r,c} gamma_{r,c'} / (E_r - E - i Γ_γ / 2)
    # Build the (ne, nres) denominator once and pool contributions per
    # (c, c') pair. Complex arithmetic throughout.
    e_col = e_safe.reshape(-1, 1)                              # (ne, 1)
    er_row = res_er.reshape(1, -1)                             # (1, nres)
    gg_row = res_gg.reshape(1, -1)
    # denom_er[i, r] = E_r - E_i - i * Gamma_gamma_r / 2
    denom_er = (er_row - e_col) - 1j * 0.5 * gg_row            # (ne, nres) complex
    inv_denom = 1.0 / denom_er                                 # (ne, nres)

    R = xp.zeros((ne, nch, nch), dtype=xp.complex128)
    for c in range(nch):
        for cp in range(c, nch):
            # (nres,) * (nres,) * (ne, nres) -> reduce over nres.
            w = (gammas[c] * gammas[cp]).reshape(1, -1)         # (1, nres)
            R_ccp = xp.sum(w * inv_denom, axis=1)              # (ne,)
            R = _put_2d(R, c, cp, R_ccp, xp)
            if cp != c:
                R = _put_2d(R, cp, c, R_ccp, xp)

    # --- Penetration diag P = diag(P_c(E)). Fission: P=1. ---
    P_diag = xp.ones((ne, nch), dtype=xp.float64)
    P_diag = _put_col(P_diag, 0, p_e, xp)

    # --- W = I - i R P, X = W^{-1} R. ---
    # Do the multiplication R * diag(P) as a broadcast.
    RP = R * P_diag.reshape(ne, 1, nch)                        # (ne, nch, nch)
    I_ = xp.eye(nch, dtype=xp.complex128).reshape(1, nch, nch)
    W = I_ - 1j * RP
    # Small-matrix solve, (ne, nch, nch) inverse-solve applied to R.
    X = xp.linalg.solve(W, R)

    # --- U-matrix. Only need the row U_{0, :} (incident = elastic). ---
    # Ω_c = exp(-i phi_L(rho_{ap}(E))): elastic gets the hard-sphere
    # phase evaluated at the SCATTERING radius R' (r_ap), NOT at the
    # channel radius a (r_a). The two coincide for NAPS in {0, 1} but
    # differ for NAPS=2, where a is derived from AWRI while R'=AP is
    # tabulated separately. Fission channels have no hard-sphere phase
    # in the external region -- they get Ω = 1.
    rho_ap = _rho(e_safe, ki, r_ap, xp)                        # (ne,)
    phi_e = factors.phase(rho_ap, L_scalar, xp)                # (ne,)
    omega_c = xp.exp(-1j * phi_e)                              # (ne,)
    # omega_row for the fission channels are 1 (no hard-sphere phase).
    # U_{0,c} = omega_0 * omega_c * [δ_{0c} + 2 i sqrt(P_0) sqrt(P_c) X_{0,c}]
    sqrt_P0 = xp.sqrt(p_e)                                     # (ne,)
    # For fission channels, sqrt(P_c) = 1.
    U_row = xp.zeros((ne, nch), dtype=xp.complex128)
    for c in range(nch):
        sqrt_Pc = sqrt_P0 if c == 0 else xp.ones_like(p_e)
        omega_prod = omega_c * (omega_c if c == 0 else 1.0)
        # 2 i sqrt(P_0) sqrt(P_c) X_{0,c}
        term = 2j * sqrt_P0 * sqrt_Pc * X[:, 0, c]
        if c == 0:
            term = 1.0 + term
        # Multiply by omega_prod: only elastic-elastic (c=0) gets the
        # exp(-2i phi); fission channels get exp(-i phi).
        U_row = _put_col(U_row, c, omega_prod * term, xp)

    # --- Cross sections. ---
    U00 = U_row[:, 0]
    sct = pi_k2 * group_g * xp.abs(1.0 - U00) ** 2

    # |U_{0,c}|^2 sum, splitting fission and total.
    sumsq = xp.abs(U_row) ** 2
    sumsq_total = xp.sum(sumsq, axis=1)                        # (ne,)
    if nfis > 0:
        sumsq_fis = xp.sum(sumsq[:, 1:1 + nfis], axis=1)
    else:
        sumsq_fis = xp.zeros_like(sumsq_total)
    fis = pi_k2 * group_g * sumsq_fis
    abs_ = pi_k2 * group_g * (1.0 - sumsq_total)
    cap = abs_ - fis

    # Positive-energy mask: below zero energy, contributions are 0.
    zero = xp.zeros_like(sct)
    sct = xp.where(e_pos, sct, zero)
    cap = xp.where(e_pos, cap, zero)
    fis = xp.where(e_pos, fis, zero)
    return sct, cap, fis


def _put_2d(mat, i, j, val, xp):
    """`mat[:, i, j] = val` in a backend-friendly way."""
    if getattr(xp, 'name', None) == 'jax':
        return mat.at[:, i, j].set(val)
    mat = mat.copy()
    mat[:, i, j] = val
    return mat


def _put_col(mat, j, val, xp):
    """`mat[:, j] = val` in a backend-friendly way."""
    if getattr(xp, 'name', None) == 'jax':
        return mat.at[:, j].set(val)
    mat = mat.copy()
    mat[:, j] = val
    return mat


def reconstruct(data: RMData, energies_in, xp):
    """Reich-Moore cross sections at ``energies_in`` (any 1D array).

    Returns a dict with keys ``sct``, ``cap``, ``fis``, ``pot``,
    ``tot``, each of shape ``(len(energies_in),)``, in barn (with the
    ``ki`` convention).

    Parameters
    ----------
    data : RMData
        Preprocessed R-M input for the range. See :class:`RMData`.
    energies_in : array_like
        Incident-neutron energies. Cast to the backend's float64 via
        ``xp.asarray``.
    xp : backend
        As returned by :func:`endf_userpy.primitives.array_ns.get_backend`.

    Notes
    -----
    The potential-scattering cross section ``pot`` is computed
    separately as ``σ_pot = 4π / k² (Σ_J g_J
    sin²(φ_{L(J)}(E)))`` -- the same hard-sphere-phase
    expression used by MLBW, kept as a diagnostic output.

    ``tot`` is ``sct + cap + fis`` (no competitive channel in R-M).
    """
    if getattr(xp, 'name', None) == 'numba':
        # Route to the hand-written @njit kernel. Same physics, same
        # inputs, same outputs. See
        # :mod:`mf2_interpretation_reichmoore_numba` for the trade-offs.
        from . import mf2_interpretation_reichmoore_numba as _numba
        return _numba.reconstruct(data, energies_in)

    e = xp.asarray(energies_in, dtype=xp.float64)
    e_pos = e > 0.0
    e_safe = xp.maximum(e, 0.0)
    k_e2 = (data.ki ** 2) * e_safe                             # (ne,)
    inv_k2 = xp.where(e_pos, xp.pi / xp.where(k_e2 > 0, k_e2, 1.0), 0.0)
    pi_k2 = data.abn * inv_k2                                  # (ne,)

    ngroups = data.group_l.shape[0]
    res_group = xp.asarray(data.res_group, dtype=xp.int32)
    res_er = xp.asarray(data.res_er, dtype=xp.float64)
    res_gn = xp.asarray(data.res_gn, dtype=xp.float64)
    res_gg = xp.asarray(data.res_gg, dtype=xp.float64)
    res_gf1 = xp.asarray(data.res_gf1, dtype=xp.float64)
    res_gf2 = xp.asarray(data.res_gf2, dtype=xp.float64)

    sct_tot = xp.zeros_like(e_safe)
    cap_tot = xp.zeros_like(e_safe)
    fis_tot = xp.zeros_like(e_safe)
    pot_tot = xp.zeros_like(e_safe)

    # --- Potential scattering: sum over groups of g_J * sin^2(phi_L). ---
    rho_ap = _rho(e_safe, data.ki, data.r_ap, xp)              # (ne,)
    for g in range(ngroups):
        L_g = xp.asarray(int(data.group_l[g]))
        g_J = float(data.group_g[g])
        phi_L = factors.phase(rho_ap, L_g, xp)                 # (ne,)
        pot_tot = pot_tot + g_J * xp.sin(phi_L) ** 2
    pot_tot = 4.0 * pi_k2 * pot_tot

    # --- Per-group loop for the resonant contribution. ---
    # Fixed-shape membership mask: `res_mask[r, g] == 1.0` iff
    # resonance `r` belongs to group `g`. Keeping all `(nres,)` arrays
    # full-length inside the group loop (instead of Python-side boolean
    # indexing to slice each group's rows out) is what makes this
    # `reconstruct` traceable under `jax.jit`; JAX refuses the
    # data-dependent shape a boolean mask would produce. The extra
    # cost is `ngroups`x more per-(c,c') work in the R-matrix sum,
    # since every resonance contributes to every group's sum (zeroed
    # out via the mask for non-members); for the typical actinide RM
    # file with 2-3 J·π groups this is a small constant factor.
    group_idx = xp.arange(ngroups, dtype=xp.int32)
    res_mask = (
        res_group.reshape(-1, 1) == group_idx.reshape(1, -1)
    ).astype(xp.float64)                                       # (nres, ngroups)

    for g in range(ngroups):
        sct_g, cap_g, fis_g = _reconstruct_group(
            e_safe, e_pos, k_e2, pi_k2,
            data.group_l[g], data.group_g[g], data.group_nfis[g],
            res_er, res_gn, res_gg, res_gf1, res_gf2,
            res_mask[:, g],
            data.ki, data.r_a, data.r_ap, xp,
        )
        sct_tot = sct_tot + sct_g
        cap_tot = cap_tot + cap_g
        fis_tot = fis_tot + fis_g

    tot = sct_tot + cap_tot + fis_tot
    return {
        'sct': sct_tot, 'cap': cap_tot, 'fis': fis_tot,
        'pot': pot_tot, 'tot': tot,
    }
