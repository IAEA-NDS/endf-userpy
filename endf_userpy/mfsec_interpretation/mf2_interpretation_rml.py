"""ENDF-6 MF2/MT151 R-Matrix Limited (LRF=7) resolved-resonance
reconstruction — dataclass and (upcoming) reconstruction entry
point.

Scope of the LRF=7 arc:

- Roadmap issue: #198 (Phase 2). Arc tracker: #228. MF2 LRF=7
  R-Matrix Limited is common in modern evaluations
  (ENDF/B-VIII.1, JEFF-4.0). The arc lands in four/five PRs
  matching the pattern set by MLBW (LRF=2) and Reich-Moore
  (LRF=3):

    1. preproc + this dataclass (**this PR**)
    2. array-agnostic (numpy + JAX via ``array_ns``) reconstruction
       for the KRM=3 Reich-Moore variant
    3. numba backend for the hot loop
    4. wire into :mod:`resonance_composition` behind the existing
       ``include_resonance=True`` API
    5. corpus end-to-end verification (Rh-103, Pu-239, Cu-63)

Formalism initial scope (this arc):

- ``KRM=3`` Reich-Moore approximation applied to the full R-Matrix
  Limited channel bookkeeping. Every real ``LRU=1, LRF=7`` file in
  the ad-hoc corpus uses ``KRM=3``. Other ``KRM`` values (1=SLBW,
  2=MLBW, 4=full R-matrix) raise ``NotImplementedError`` from the
  preprocessor; SLBW/MLBW are effectively the older ``LRF=1/2``
  paths, and full R-matrix (``KRM=4``) is out of scope pending a
  concrete need.
- ``KRL=0`` (non-relativistic kinematics).
- ``IFG=0`` (widths, not reduced-width amplitudes). ``IFG=1`` is
  rare in modern files and can be added later without dataclass
  churn: the preprocessor would convert to ``IFG=0`` widths at
  ingest time.
- ``NRO=0`` (energy-independent scattering radius). ``NRO=1``
  needs a TAB1 read for the range-level radius; adding it later is
  a small change to the preprocessor.
- ``KBK=0`` (no background R-matrix contribution per J-group).
  Rare in modern files.
- ``KPS=0`` (no additional hard-sphere phase shift beyond the
  standard).

The dataclass keeps things flat and shape-uniform (channels padded
per J-group to the group's max NCH; resonances padded per group to
the group's max NRS), so the eventual reconstruction can run under
``jax.jit`` without data-dependent shapes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..primitives import tab1
from . import mf2_interpretation_factors as factors


_EPS = 1e-38


@dataclass
class RMLData:
    """Natural-size R-Matrix Limited (LRF=7) input for one isotope
    / one range.

    Layout follows the LRF=7 nesting: range-level scalars, per
    particle pair arrays, per J-group arrays, per-(group, channel)
    padded 2-D arrays, per resonance flat arrays, and per-(group,
    channel, resonance) widths padded to the group's ``max(NCH)``.
    """

    # ---------- Range-level scalars ----------
    abn: float
    """Isotopic abundance in the material."""
    spi: float
    """Target spin (incident-channel target spin) used in ``g_J``
    denominator. Sourced from MF2/MT151 SPI, not from the
    per-pair ``IB`` (they agree for the elastic pair)."""
    ki: float
    """Wavenumber coefficient of the ELASTIC channel:
    ``k_elastic(E) = ki * sqrt(E)`` in ``cm^-1`` (with ``E`` in eV,
    ``k`` in cm^-1). Kept scalar because every non-elastic pair
    gets its own ``ki`` derived from ``pp_ma`` / ``pp_mb`` in the
    reconstruction; this ``ki`` is the shortcut for the elastic
    (incident) pair, which drives the ``π/k²`` prefactor of the
    partial cross sections."""
    r_ap: tab1.TAB1
    """Range-level scattering radius as a TAB1 vs E. Constant
    (``NRO=0``) in the initial scope: two-point ``[EL, 1e11]``
    with the range's ``AP``. Per-channel radii ``ch_ape`` /
    ``ch_apt`` override this in the KRM=3 reconstruction."""
    krm: int
    """R-Matrix approximation flag. Only ``KRM=3`` (Reich-Moore)
    is supported by the reconstruction in the initial scope."""
    ifg: int
    """Reduced-width flag. Only ``IFG=0`` (widths) in the initial
    scope."""
    krl: int
    """Relativistic-kinematics flag. Only ``KRL=0`` (non-rel) in
    the initial scope."""
    naps: int
    """Scattering-radius handling flag: 0 = derive channel radius
    from mass, 1 = use ``AP``, 2 = ``AP`` is scattering radius but
    channel radius is mass-derived. See
    :func:`mf2_interpretation_mlbw_preproc._channel_radius`."""

    # ---------- Per particle pair (shape (npp,)) ----------
    pp_ma: np.ndarray
    """Mass of particle A in each pair, in neutron-mass units.
    Convention: A is typically the "emitted" particle (n, gamma,
    fission fragment)."""
    pp_mb: np.ndarray
    """Mass of particle B in each pair, in neutron-mass units."""
    pp_za: np.ndarray
    """Charge of particle A in each pair (elementary units)."""
    pp_zb: np.ndarray
    """Charge of particle B in each pair (elementary units)."""
    pp_ia: np.ndarray
    """Spin of particle A in each pair."""
    pp_ib: np.ndarray
    """Spin of particle B in each pair."""
    pp_q: np.ndarray
    """Reaction Q value per pair (eV, from ENDF ``Q``)."""
    pp_pnt: np.ndarray
    """Penetrability flag per pair: 0 = default (yes if applicable),
    1 = compute penetrabilities, -1 = penetrability = 1
    (photon / fission)."""
    pp_shf: np.ndarray
    """Shift-factor flag per pair: 0 = default (yes if applicable),
    1 = compute shift factor, -1 = shift factor = 0."""
    pp_mt: np.ndarray
    """Reaction MT per pair. MT=2 → elastic, MT=18 → fission,
    MT=102 → capture, ..."""
    pp_pa: np.ndarray
    """Parity of particle A per pair. 0 encodes the neutral case
    used for spinless / undefined parity in ENDF."""
    pp_pb: np.ndarray
    """Parity of particle B per pair."""
    pp_incident_idx: int
    """1-based index into the per-pair arrays of the incident
    (elastic) pair. Identified by ``MT=2`` and non-zero neutron
    mass. Kept as a scalar because every LRU=1 range has exactly
    one incident pair by construction."""

    # ---------- Per J-group (shape (ngroups,)) ----------
    group_aj: np.ndarray
    """Signed J value per group; sign encodes parity for KPS=0
    files (matches the sign convention on ``AJ`` in the ENDF-6
    LRF=7 records). Reconstruction takes ``|group_aj|`` for the
    ``g_J`` weight and reads the parity from ``group_pj`` (or,
    equivalently, ``sign(group_aj)``)."""
    group_pj: np.ndarray
    """Parity per group. 0 in most files (parity carried via the
    sign of ``AJ`` above); stored verbatim from ENDF ``PJ``."""
    group_g: np.ndarray
    """Statistical weight ``g_J = (2|J| + 1) / (2(2I + 1))`` per
    group. Precomputed here so the reconstruction stays formula-
    free on the J side."""
    group_nch: np.ndarray
    """Number of channels in each group. ``ch_*`` per-channel
    arrays are padded per group up to ``max(group_nch)``; this
    array carries the true count for masking."""

    # ---------- Per-(group, channel) padded (shape (ngroups, max_nch)) ----------
    ch_ppi: np.ndarray
    """1-based particle-pair index for each channel. Padded rows
    beyond ``group_nch[g]`` carry 0 (unused)."""
    ch_l: np.ndarray
    """Orbital angular momentum ``L`` for each channel. Padded
    rows carry 0."""
    ch_sch: np.ndarray
    """Channel spin ``S`` for each channel. Padded rows carry
    0.0."""
    ch_bnd: np.ndarray
    """Boundary condition ``B_c`` for each channel. Typically 0
    in modern files."""
    ch_ape: np.ndarray
    """Effective channel radius ``a_e`` (used for the hard-sphere
    phase). Fm."""
    ch_apt: np.ndarray
    """True channel radius ``a_t`` (used for the penetration
    factor). Fm."""
    ch_active: np.ndarray
    """Boolean mask ``(ngroups, max_nch)`` marking real vs padded
    channels. True where the channel is a genuine LRF=7 channel."""

    # ---------- Per resonance (shape (nres,)) ----------
    res_group: np.ndarray
    """0-based group index for each resonance (``[0, ngroups)``)."""
    res_er: np.ndarray
    """Resonance energy ``E_r`` (eV, signed; can be negative for
    bound / sub-threshold poles)."""

    # ---------- Per-(resonance, channel) padded (shape (nres, max_nch)) ----------
    res_gam: np.ndarray
    """Per-channel width ``Γ_{r,c}`` for each resonance and each
    channel of the resonance's group. Widths are signed in ENDF-6
    LRF=7 (the sign is the reduced-width-amplitude sign that the
    reconstruction eventually needs after the
    ``γ_c = sign(Γ) sqrt(|Γ| / (2 P_c(|E_r|)))`` mapping).
    Padded channel slots beyond the resonance's group ``NCH``
    carry 0.0.

    Note: this is the raw ``GAM`` from the file when ``IFG=0``.
    ``IFG=1`` files (reduced-width amplitudes) are not supported
    in the initial scope; when added, the preprocessor should
    convert them to widths so the reconstruction sees a single
    convention."""

    def n_pp(self) -> int:
        """Number of particle pairs."""
        return int(self.pp_ma.shape[0])

    def n_groups(self) -> int:
        """Number of J-groups."""
        return int(self.group_aj.shape[0])

    def n_res(self) -> int:
        """Total number of resonances across all groups."""
        return int(self.res_er.shape[0])

    def max_nch(self) -> int:
        """Padded channel dimension of the per-(group, channel)
        and per-(resonance, channel) arrays."""
        return int(self.ch_ppi.shape[1])


def _signed_sqrt(x, xp):
    """``sign(x) * sqrt(|x|)``. Zero-safe reduced-width amplitude
    from a signed width."""
    ax = xp.abs(x)
    return xp.sign(x) * xp.sqrt(ax)


def _put_2d(mat, i, j, val, xp):
    """``mat[:, i, j] = val`` backend-agnostically. Copy for numpy,
    ``.at[]`` for JAX."""
    if getattr(xp, 'name', None) == 'jax':
        return mat.at[:, i, j].set(val)
    mat = mat.copy()
    mat[:, i, j] = val
    return mat


def _put_col(mat, j, val, xp):
    """``mat[:, j] = val`` backend-agnostically."""
    if getattr(xp, 'name', None) == 'jax':
        return mat.at[:, j].set(val)
    mat = mat.copy()
    mat[:, j] = val
    return mat


def _channel_kind(data: RMLData, ppi: int) -> str:
    """Classify a channel by its particle-pair index (1-based).

    - ``'gamma'`` if the pair has particle-A mass 0 (radiation).
      Eliminated in the KRM=3 Reich-Moore approximation.
    - ``'fission'`` if the pair's MT is 18 (or the pair's PNT is
      ``-1`` with a non-gamma mass — kept for future coverage).
    - ``'elastic'`` if the pair's MT is 2.
    - ``'other'`` for everything else (inelastic to excited
      levels, charged particle exit, ...); the initial scope
      routes these through the particle branch with the standard
      elastic-k penetration, which is not the correct kinematics
      in general.  Files that need non-elastic particle channels
      are out of scope for this arc.
    """
    ma = float(data.pp_ma[ppi - 1])
    mt = int(round(float(data.pp_mt[ppi - 1])))
    if ma == 0.0:
        return 'gamma'
    if mt == 18:
        return 'fission'
    if mt == 2:
        return 'elastic'
    return 'other'


def _classify_group_channels(data: RMLData, g: int):
    """Return ``(kind_list, particle_channels, gamma_channels,
    elastic_slot)`` for group ``g``.

    - ``kind_list``: per-slot channel kind, length ``max_nch`` (the
      padded per-channel array width). Padded slots carry
      ``'padded'``.
    - ``particle_channels``: list of 0-based slot indices that are
      explicit in the R-matrix (elastic, fission, other).
    - ``gamma_channels``: list of 0-based slot indices that are
      eliminated (radiation).
    - ``elastic_slot``: index (into ``particle_channels``) of the
      elastic channel that carries the incident wave. When a group
      has more than one elastic channel (different (L, S) coupling
      to the same J), the FIRST one in the padded layout is
      chosen; multi-elastic-channel groups fall through the
      standard sum-over-elastic-channels aggregation in the
      caller.
    """
    max_nch = data.max_nch()
    nch_true = int(data.group_nch[g])
    kinds = []
    particle = []
    gamma = []
    for c in range(max_nch):
        if c >= nch_true:
            kinds.append('padded')
            continue
        ppi = int(round(float(data.ch_ppi[g, c])))
        kind = _channel_kind(data, ppi)
        kinds.append(kind)
        if kind == 'gamma':
            gamma.append(c)
        else:
            particle.append(c)
    elastic_slot = None
    for i, c in enumerate(particle):
        if kinds[c] == 'elastic':
            elastic_slot = i
            break
    if elastic_slot is None:
        raise ValueError(
            f'LRF=7 group {g} has no elastic (MT=2) channel — '
            f'kinds seen: {kinds[:nch_true]}. Non-elastic-incident '
            f'groups are out of the initial scope of this arc.'
        )
    return kinds, particle, gamma, elastic_slot


def _pair_uses_penetration(data: RMLData, ppi: int) -> bool:
    """Whether the reconstruction should compute ``P_L(rho)`` for
    the pair. False for gamma / fission / any ``PNT=-1`` pair;
    True for elastic-like massive pairs."""
    if _channel_kind(data, ppi) in ('gamma', 'fission'):
        return False
    pnt = int(round(float(data.pp_pnt[ppi - 1])))
    if pnt == -1:
        return False
    return True


def _reconstruct_group(
    data: RMLData, g: int, e_safe, e_pos, pi_k2, xp,
):
    """Reconstruct one J-group's contribution to (elastic, capture,
    fission).

    Uses the KRM=3 Reich-Moore approximation: gamma channels are
    eliminated by folding their reduced-width amplitudes into a
    scalar ``Γ_γ_r`` per resonance that lives in the imaginary
    part of the R-matrix denominator. The R-matrix operates only
    over the particle channels.

    Returns three ``(ne,)`` arrays: ``sct``, ``cap``, ``fis``,
    each with the group's statistical weight and the ``π/k²``
    prefactor already folded in.
    """
    kinds, particle, gamma, elastic_slot = _classify_group_channels(
        data, g,
    )
    npart = len(particle)
    ne = int(e_safe.shape[0])

    g_J = data.group_g[g]
    ki = data.ki

    res_er = data.res_er
    res_group_arr = xp.asarray(data.res_group, dtype=xp.int32)
    group_mask = (res_group_arr == g).astype(xp.float64)         # (nres,)
    nres = int(res_er.shape[0])
    if nres == 0:
        z = xp.zeros_like(e_safe)
        return z, z, z

    # --- Per-channel P_c(E) and P_c(|E_r|), and phi_c(E) for
    # phase-carrying channels. ---
    # For the initial scope every channel uses the elastic-pair
    # wavenumber ``ki`` (see `_channel_kind` docstring). Rho is
    # computed per channel with its own APT (penetration) or APE
    # (phase).
    e_col = e_safe.reshape(-1, 1)                                 # (ne, 1)
    er_row = res_er.reshape(1, -1)                                # (1, nres)
    er_abs = xp.abs(res_er)                                        # (nres,)
    sqrt_e = xp.sqrt(e_safe)                                       # (ne,)
    sqrt_er = xp.sqrt(er_abs)                                      # (nres,)

    # --- Per-resonance elimination sum: Γ_γ_r = 2 * Σ_{c ∈ gamma}
    # γ_{r,c}^2 (P=1 convention for gamma channels). Equivalent to
    # summing GAM values of gamma channels for each resonance
    # (since γ^2 = |Γ| / (2 P) with P=1 for gamma; sign washes out
    # of the sum because we square). ---
    gam_full = data.res_gam                                        # (nres, max_nch)
    gamma_gg = xp.zeros(nres, dtype=xp.float64)
    for c in gamma:
        gamma_gg = gamma_gg + xp.abs(gam_full[:, c])
    # Mask out-of-group resonances so they don't contribute
    # spuriously via the shared full-length arrays.
    gamma_gg = gamma_gg * group_mask

    # --- Particle channels: reduced-width amplitudes γ_{r,c}, phases. ---
    # For each particle channel c: compute L, APT, APE, PNT-flag.
    # gamma_pc[c] shape (nres,) is the reduced-width amplitude
    # γ_{r, particle_channel_c} for that channel.
    gamma_pc = []          # list of (nres,) per-particle-channel
    p_e_list = []          # list of (ne,) per-particle-channel penetration
    phase_list = []        # list of (ne,) per-particle-channel phi
    for c in particle:
        ppi = int(round(float(data.ch_ppi[g, c])))
        L_c = int(round(float(data.ch_l[g, c])))
        L_arr = xp.asarray(L_c)
        apt_c = float(data.ch_apt[g, c])
        ape_c = float(data.ch_ape[g, c])

        # Rho at E and at |E_r|.
        rho_e = ki * sqrt_e * apt_c                                # (ne,)
        rho_r = ki * sqrt_er * apt_c                               # (nres,)
        if _pair_uses_penetration(data, ppi):
            p_e, _ = factors.pnt_shf(rho_e, L_arr, xp)
            p_r, _ = factors.pnt_shf(rho_r, L_arr, xp)
        else:
            p_e = xp.ones_like(e_safe)
            p_r = xp.ones_like(er_abs)
        p_e_list.append(p_e)

        # Hard-sphere phase only for elastic (particle A of pair is
        # incident with non-zero charge/mass; fission gives zero
        # scattering amplitude contribution).
        if _channel_kind(data, ppi) == 'elastic':
            rho_e_hat = ki * sqrt_e * ape_c
            phi_c = factors.phase(rho_e_hat, L_arr, xp)
        else:
            phi_c = xp.zeros_like(e_safe)
        phase_list.append(phi_c)

        # Reduced-width amplitude γ_{r,c}.
        denom = xp.where(p_r > _EPS, 2.0 * p_r, 1.0)
        signed = _signed_sqrt(gam_full[:, c], xp)
        gam_c = xp.where(
            p_r > _EPS,
            signed / xp.sqrt(denom),
            signed / xp.sqrt(xp.asarray(2.0)),
        )
        # Mask out-of-group resonances.
        gam_c = gam_c * group_mask
        gamma_pc.append(gam_c)

    # --- Build the R-matrix R_{cc'}(E) over particle channels only. ---
    # Denominator: (E_r - E - i Γ_γ_r / 2), broadcast (ne, nres).
    gg_row = gamma_gg.reshape(1, -1)                               # (1, nres)
    denom_er = (er_row - e_col) - 1j * 0.5 * gg_row                # (ne, nres)
    inv_denom = 1.0 / denom_er

    R = xp.zeros((ne, npart, npart), dtype=xp.complex128)
    for a in range(npart):
        for b in range(a, npart):
            w = (gamma_pc[a] * gamma_pc[b]).reshape(1, -1)          # (1, nres)
            R_ab = xp.sum(w * inv_denom, axis=1)                    # (ne,)
            R = _put_2d(R, a, b, R_ab, xp)
            if b != a:
                R = _put_2d(R, b, a, R_ab, xp)

    # --- W = I - i R P, X = W^{-1} R. ---
    P_diag = xp.zeros((ne, npart), dtype=xp.float64)
    for a in range(npart):
        P_diag = _put_col(P_diag, a, p_e_list[a], xp)
    RP = R * P_diag.reshape(ne, 1, npart)                          # (ne, npart, npart)
    I_ = xp.eye(npart, dtype=xp.complex128).reshape(1, npart, npart)
    W = I_ - 1j * RP
    X = xp.linalg.solve(W, R)                                       # (ne, npart, npart)

    # --- U-row for the elastic (incident) channel. ---
    # U_{ec, c} = Ω_ec Ω_c [ δ_{ec, c} + 2 i sqrt(P_ec) sqrt(P_c) X_{ec, c} ]
    # where Ω_c = exp(-i phi_c) for elastic-like channels and 1 for
    # non-phase (fission) channels.
    ec = elastic_slot
    sqrt_pe_ec = xp.sqrt(p_e_list[ec])
    phi_ec = phase_list[ec]
    omega_ec = xp.exp(-1j * phi_ec)

    U_row = xp.zeros((ne, npart), dtype=xp.complex128)
    for c in range(npart):
        sqrt_pe_c = xp.sqrt(p_e_list[c])
        phi_c = phase_list[c]
        omega_c = xp.exp(-1j * phi_c)
        term = 2j * sqrt_pe_ec * sqrt_pe_c * X[:, ec, c]
        if c == ec:
            term = 1.0 + term
        U_row = _put_col(U_row, c, omega_ec * omega_c * term, xp)

    # --- Cross sections (Lane-Thomas). ---
    U_ee = U_row[:, ec]
    sct = pi_k2 * g_J * xp.abs(1.0 - U_ee) ** 2
    sumsq = xp.abs(U_row) ** 2
    sumsq_total = xp.sum(sumsq, axis=1)
    fis_slots = [
        i for i, c in enumerate(particle)
        if kinds[c] == 'fission'
    ]
    if fis_slots:
        sumsq_fis = xp.sum(
            xp.stack([sumsq[:, i] for i in fis_slots], axis=1),
            axis=1,
        )
    else:
        sumsq_fis = xp.zeros_like(sumsq_total)
    fis = pi_k2 * g_J * sumsq_fis
    cap = pi_k2 * g_J * (1.0 - sumsq_total)

    zero = xp.zeros_like(sct)
    sct = xp.where(e_pos, sct, zero)
    cap = xp.where(e_pos, cap, zero)
    fis = xp.where(e_pos, fis, zero)
    return sct, cap, fis


def reconstruct(data: RMLData, energies_in, xp):
    """KRM=3 R-Matrix Limited reconstruction (ENDF-6 LRF=7).

    Returns a dict with keys ``sct``, ``cap``, ``fis``, ``pot``,
    ``tot``, each of shape ``(len(energies_in),)``, in barn (using
    the ``ki`` convention baked into :class:`RMLData`).

    Parameters
    ----------
    data : RMLData
        Preprocessed LRF=7 input for the range.
    energies_in : array_like
        Incident-neutron energies (eV, lab-frame).
    xp : backend
        As returned by :func:`endf_userpy.primitives.array_ns.get_backend`.
        Numpy and JAX supported in this PR; numba routes through
        a future sibling module.
    """
    if data.krm != 3:
        raise NotImplementedError(
            f'LRF=7 KRM={data.krm} not supported by this arc; '
            f'only KRM=3 (Reich-Moore approximation) is '
            f'implemented in the initial scope.'
        )
    if getattr(xp, 'name', None) == 'numba':
        from . import mf2_interpretation_rml_numba as _numba
        return _numba.reconstruct(data, energies_in)
    e = xp.asarray(energies_in, dtype=xp.float64)
    e_pos = e > 0.0
    e_safe = xp.maximum(e, 0.0)
    k_e2 = (data.ki ** 2) * e_safe                                # (ne,)
    inv_k2 = xp.where(
        e_pos, xp.pi / xp.where(k_e2 > 0, k_e2, 1.0), 0.0,
    )
    pi_k2 = data.abn * inv_k2                                     # (ne,)

    sct_tot = xp.zeros_like(e_safe)
    cap_tot = xp.zeros_like(e_safe)
    fis_tot = xp.zeros_like(e_safe)
    pot_tot = xp.zeros_like(e_safe)

    ngroups = data.n_groups()
    for g in range(ngroups):
        # Potential scattering contribution: sum g_J sin^2(phi_e_L)
        # over groups, using the elastic channel's APE. Multi-
        # elastic-channel groups pick the first elastic channel's
        # radius for this diagnostic; a follow-up can refine.
        _, particle, _, elastic_slot = _classify_group_channels(
            data, g,
        )
        ec_slot = particle[elastic_slot]
        L_e = int(round(float(data.ch_l[g, ec_slot])))
        ape_e = float(data.ch_ape[g, ec_slot])
        rho_e_hat = data.ki * xp.sqrt(e_safe) * ape_e
        phi_e = factors.phase(rho_e_hat, xp.asarray(L_e), xp)
        pot_tot = pot_tot + data.group_g[g] * xp.sin(phi_e) ** 2

        # Resonant contribution.
        sct_g, cap_g, fis_g = _reconstruct_group(
            data, g, e_safe, e_pos, pi_k2, xp,
        )
        sct_tot = sct_tot + sct_g
        cap_tot = cap_tot + cap_g
        fis_tot = fis_tot + fis_g

    pot_tot = 4.0 * pi_k2 * pot_tot
    tot = sct_tot + cap_tot + fis_tot
    return {
        'sct': sct_tot, 'cap': cap_tot, 'fis': fis_tot,
        'pot': pot_tot, 'tot': tot,
    }
