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


def reconstruct(*args, **kwargs):
    """Placeholder. The array-agnostic KRM=3 Reich-Moore
    reconstruction is planned for a follow-up PR (see the arc
    outline in this module's docstring)."""
    raise NotImplementedError(
        'MF2 LRF=7 (R-Matrix Limited) reconstruction is not yet '
        'implemented. The preprocessor and dataclass landed first '
        '(this PR); the KRM=3 Reich-Moore reconstruction is the '
        'next PR in the arc. Tracked under the LRF=7 roadmap item.'
    )
