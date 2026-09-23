"""MF2 (resolved + unresolved) resonance reconstruction composed
with the MF3 background.

ENDF-6 stores the cross section in the resonance region as two
additive parts: MF2 (resonance parameters, reconstructed
point-wise for LRU=1 or as an average for LRU=2 LSSF=0) plus MF3
(a smooth "background" subtractive residual). The physical cross
section on a query grid is the sum. Outside every resonance
range the MF2 contribution is zero and the sum reduces to the
MF3 interpolation.

Two entry points:

- :func:`reconstruct_resonance_xs` returns only the MF2 partial
  for a given MT, summed over every LRU=1 range and every
  LRU=2 LSSF=0 URR range. Zero outside every range.
- :func:`compute_reconstructed_cross_section` returns
  ``MF3(mt, E) + MF2_partial(mt, E)`` -- the composed physical
  cross section.

Both are backend-agnostic: pass an ``xp`` returned by
:func:`endf_userpy.primitives.array_ns.get_backend` to run on
numpy, numba, or JAX. The default (``xp=None``) resolves to
numpy.

Supported LRU=1 formalisms: LRF=2 (MLBW) and LRF=3 (Reich-Moore).
Supported LRU=2 formalisms: LRF=2 with INT=2 (Case C
energy-dependent widths, lin-lin table interpolation).

Ranges the current implementation cannot reconstruct
(Adler-Adler LRF=4, R-matrix limited LRF=7, URR with non-INT=2
tables, ...) contribute zero. If a file has no supported range
at all, or has an LSSF=0 URR range the reconstructor cannot
handle, a :class:`UserWarning` names the specific formalism /
INT code seen.
"""

import warnings

from ..mfsec_interpretation import mf3_interpretation
from ..mfsec_interpretation import mf2_interpretation_mlbw
from ..mfsec_interpretation import mf2_interpretation_mlbw_preproc
from ..mfsec_interpretation import mf2_interpretation_reichmoore
from ..mfsec_interpretation import mf2_interpretation_reichmoore_preproc
from ..mfsec_interpretation import mf2_interpretation_urr
from ..mfsec_interpretation import mf2_interpretation_urr_preproc
from ..primitives import array_ns


# MT number -> which keys of the reconstruction result to sum.
# The reconstruction returns 'sct', 'cap', 'fis', 'pot', 'tot' (+ 'rxx'
# for MLBW and URR). 'pot' is a diagnostic already included in 'sct',
# so it never appears here; 'tot' is the aggregate already summed.
_MLBW_MT_TO_KEYS = {
    1: ('tot',),                     # total
    2: ('sct',),                     # elastic
    3: ('cap', 'fis', 'rxx'),        # nonelastic (= total - elastic)
    18: ('fis',),                    # fission
    27: ('cap', 'fis'),              # absorption
    102: ('cap',),                   # radiative capture
}

_RM_MT_TO_KEYS = {
    1: ('tot',),
    2: ('sct',),
    3: ('cap', 'fis'),               # R-M has no competitive channel
    18: ('fis',),
    27: ('cap', 'fis'),
    102: ('cap',),
}

# URR partials have the MLBW shape (sct / cap / fis / rxx / pot / tot),
# so the same MT-to-keys map applies.
_URR_MT_TO_KEYS = _MLBW_MT_TO_KEYS


def _iter_lru1_ranges(endf_dict):
    """Yield ``(iso_idx, rng_idx, rng_dict)`` for every LRU=1 range.

    Silent no-op on files with no MF2/MT151, no isotopes, or no
    LRU=1 range.
    """
    if 2 not in endf_dict or 151 not in endf_dict[2]:
        return
    isotopes = endf_dict[2][151].get('isotope', {})
    for iso_i in sorted(isotopes):
        d_iso = isotopes[iso_i]
        for rng_i in sorted(d_iso.get('range', {})):
            rng = d_iso['range'][rng_i]
            if int(rng.get('LRU', 0)) != 1:
                continue
            yield iso_i, rng_i, rng


def _iter_lru2_lssf0_ranges(endf_dict):
    """Yield ``(iso_idx, rng_idx, rng_dict)`` for every LRU=2 URR
    range with ``LSSF=0``. These are the URR ranges that need
    reconstruction to produce a physical XS (LSSF=1 URR is
    silently correct: MF3 already carries the average XS).
    """
    if 2 not in endf_dict or 151 not in endf_dict[2]:
        return
    isotopes = endf_dict[2][151].get('isotope', {})
    for iso_i in sorted(isotopes):
        d_iso = isotopes[iso_i]
        for rng_i in sorted(d_iso.get('range', {})):
            rng = d_iso['range'][rng_i]
            if int(rng.get('LRU', 0)) != 2:
                continue
            if int(rng.get('LSSF', 0)) != 0:
                continue
            yield iso_i, rng_i, rng


def _reconstruct_lru1_range(endf_dict, iso_i, rng_i, rng, energies, xp):
    """Preprocess + reconstruct a single LRU=1 range. Returns
    ``(recon_dict, mt_to_keys_map)`` or ``(None, None)`` if the
    LRF is not supported (caller then skips it)."""
    lrf = int(rng.get('LRF', 0))
    if lrf == 2:
        data = mf2_interpretation_mlbw_preproc.mlbw_data_from_endf_dict(
            endf_dict, isotope_idx=iso_i, range_idx=rng_i,
        )
        recon = mf2_interpretation_mlbw.reconstruct(data, energies, xp)
        return recon, _MLBW_MT_TO_KEYS
    if lrf == 3:
        # Thread xp into the preproc so JAX tracers stored at
        # per-resonance dict leaves (ER / GN / GG) survive into
        # the R-matrix reconstruction (issue #159).
        data = mf2_interpretation_reichmoore_preproc.rm_data_from_endf_dict(
            endf_dict, isotope_idx=iso_i, range_idx=rng_i, xp=xp,
        )
        recon = mf2_interpretation_reichmoore.reconstruct(data, energies, xp)
        return recon, _RM_MT_TO_KEYS
    return None, None


def _reconstruct_urr_range(endf_dict, iso_i, rng_i, rng, energies, xp):
    """Preprocess + reconstruct a single LRU=2 LSSF=0 URR range.
    Returns ``(recon_dict, mt_to_keys_map)`` on success, or
    ``(None, reason_str)`` on failure (caller propagates the
    reason into the unsupported-URR warning). Supported URR
    formalism is LRF=2 with INT=2 (lin-lin) tables.

    Any preproc / kernel exception is caught here and turned into
    a graceful "URR reconstruction unavailable for this range"
    signal: the composition layer falls back to MF3-only for the
    URR energy window and emits one summary warning that names
    every unhandled range with its specific failure reason. This
    prevents a malformed or unsupported URR range from crashing
    an otherwise-valid cross-section query.
    """
    lrf = int(rng.get('LRF', 0))
    if lrf != 2:
        return None, f'LRF={lrf} (only LRF=2 supported)'
    try:
        data = mf2_interpretation_urr_preproc.urr_data_from_endf_dict(
            endf_dict, isotope_idx=iso_i, range_idx=rng_i,
        )
    except Exception as exc:
        return None, f'preproc failed: {type(exc).__name__}: {exc}'
    try:
        recon = mf2_interpretation_urr.reconstruct(data, energies, xp)
    except Exception as exc:
        return None, f'kernel failed: {type(exc).__name__}: {exc}'
    return recon, _URR_MT_TO_KEYS


def reconstruct_resonance_xs(endf_dict, mt, energies_in, xp=None):
    """MF2 partial-XS contribution for ``mt``, summed over every
    supported resonance range in ``endf_dict``. Zero at query
    energies outside every range: each range is treated as the
    half-open interval ``[EL, EH)`` (lower endpoint inclusive,
    upper endpoint exclusive), so at the standard ENDF-6 seam
    energies (RRR ``EH == URR EL``; URR ``EH ==`` first
    MF3-above-URR knot) the higher-energy range owns the value
    and no double-count occurs. Matches PR #146's tab1
    ``side='right'`` default for the analogous MF3 doubled-x
    transitions (issue #149).

    ``mt`` outside the supported set
    ``{1, 2, 3, 18, 27, 102}`` returns zero without a warning:
    higher-MT reactions like MT=51 discrete inelastic or MT=16
    (n,2n) legitimately have no MF2 contribution and are handled
    entirely by MF3.

    LRU=1 (RRR): supported LRFs are 2 (MLBW) and 3 (Reich-Moore).
    Unsupported LRFs (Adler-Adler LRF=4, R-matrix limited LRF=7)
    are skipped. If the file has no supported LRU=1 range at all,
    a :class:`UserWarning` names the formalism seen.

    LRU=2 (URR): LSSF=1 URR ranges never contribute (MF3 already
    carries the physical average XS -- correct as-is). LSSF=0 URR
    ranges are reconstructed via the chi-squared width-fluctuation
    kernel in :mod:`mf2_interpretation_urr` when the format is
    supported (LRF=2 with INT=2 lin-lin tables); ranges the
    kernel cannot handle contribute zero and produce a
    UserWarning naming the specific reason (unsupported LRF,
    variable-NE across groups, non-lin-lin INT code, ...).
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    e = xp.asarray(energies_in, dtype=xp.float64)
    total = xp.zeros_like(e)

    # ---- LRU=1 (RRR): MLBW / Reich-Moore.
    saw_lru1_supported = False
    lru1_unsupported = []
    for iso_i, rng_i, rng in _iter_lru1_ranges(endf_dict):
        lrf = int(rng.get('LRF', 0))
        if lrf not in (2, 3):
            lru1_unsupported.append((iso_i, rng_i, lrf))
            continue
        saw_lru1_supported = True
        keys = _MLBW_MT_TO_KEYS.get(mt) if lrf == 2 else _RM_MT_TO_KEYS.get(mt)
        if not keys:
            continue
        el = float(rng['EL'])
        eh = float(rng['EH'])
        # Half-open [EL, EH): at the seam E == EH the next range (URR
        # or MF3 above URR) owns the value. Matches NJOY's right-limit
        # convention at doubled-x range transitions (issue #149; same
        # side selection as PR #146 for MF3 tab1 lookups).
        in_range = (e >= el) & (e < eh)
        try:
            any_in = bool(in_range.any())
        except Exception:
            any_in = True    # JAX tracer: always run
        if not any_in:
            continue
        recon, _ = _reconstruct_lru1_range(
            endf_dict, iso_i, rng_i, rng, e, xp,
        )
        if recon is None:
            continue
        contrib = xp.zeros_like(e)
        for k in keys:
            if k in recon:
                contrib = contrib + recon[k]
        total = total + xp.where(in_range, contrib, xp.zeros_like(e))

    if lru1_unsupported and not saw_lru1_supported:
        parts = ', '.join(f'iso={i} rng={j} LRF={f}'
                          for i, j, f in lru1_unsupported)
        warnings.warn(
            f'reconstruct_resonance_xs: no supported LRF (2 MLBW / 3 '
            f'Reich-Moore) LRU=1 range found (saw: {parts}); returning '
            f'zero resolved-resonance contribution. Adler-Adler '
            f'(LRF=4) and R-matrix limited (LRF=7) are not '
            f'implemented yet.',
            UserWarning, stacklevel=2,
        )

    # ---- LRU=2 LSSF=0 (URR needing reconstruction).
    urr_unsupported = []
    keys_urr = _URR_MT_TO_KEYS.get(mt)
    for iso_i, rng_i, rng in _iter_lru2_lssf0_ranges(endf_dict):
        if not keys_urr:
            # MT does not receive an MF2 contribution in the URR
            # either; skip the reconstruction work.
            continue
        el = float(rng['EL'])
        eh = float(rng['EH'])
        # Half-open [EL, EH): at E == EH the MF3 above-URR tabulation
        # owns the value. See issue #149.
        in_range = (e >= el) & (e < eh)
        try:
            any_in = bool(in_range.any())
        except Exception:
            any_in = True
        if not any_in:
            continue
        recon, err = _reconstruct_urr_range(
            endf_dict, iso_i, rng_i, rng, e, xp,
        )
        if recon is None:
            urr_unsupported.append((iso_i, rng_i, err))
            continue
        contrib = xp.zeros_like(e)
        for k in keys_urr:
            if k in recon:
                contrib = contrib + recon[k]
        total = total + xp.where(in_range, contrib, xp.zeros_like(e))

    if urr_unsupported:
        parts = ', '.join(
            f'iso={i} rng={j} ({err})' for i, j, err in urr_unsupported
        )
        warnings.warn(
            f'reconstruct_resonance_xs: file has LRU=2 LSSF=0 URR '
            f'range(s) the reconstruction kernel could not handle '
            f'({parts}). MF3 in those energy ranges is a background '
            f'to an unresolved-region reconstruction, so the '
            f'returned XS is only the MF3 background there, not the '
            f'physical average XS. For LSSF=1 URR (MF3 already '
            f'carries the average XS) the result would be correct '
            f'without any URR reconstruction.',
            UserWarning, stacklevel=2,
        )

    return total


def compute_reconstructed_cross_section(
    endf_dict, mt, energies_in, xp=None,
):
    """Physical cross section for ``mt`` per ENDF-6:

        sigma(E) = MF3(mt, E) + sum over supported resonance ranges of
                                 MF2_partial(mt, E)

    Supported ranges include LRU=1 RRR (MLBW / Reich-Moore) and
    LRU=2 LSSF=0 URR (via the chi-squared width-fluctuation
    kernel).

    Backend-agnostic: pass ``xp`` from
    :func:`endf_userpy.primitives.array_ns.get_backend` to run on
    numpy, numba, or JAX. Default ``xp=None`` resolves to numpy.

    The MF3 term is the raw TAB1 interpolation (via
    :func:`mfsec_interpretation.mf3_interpretation.compute_cross_section_agnostic`),
    without the ``above_range`` / ``resonance_range`` policy
    machinery of the policy-driven
    :func:`~mf3_interpretation.compute_cross_section` -- the whole
    point of composing MF2 in is that the "raw MF3 is a background,
    warn about it" policy is no longer needed. Callers who still
    want above-range policy on top of the composed cross section
    should apply it themselves.

    See :func:`reconstruct_resonance_xs` for supported LRF list
    and warning behaviour on unsupported formalisms.

    Sum-MTs (MT=1, 3, 4, 27, ...) are returned verbatim from the
    file's own tabulation for those MT numbers; no sum-rule
    enforcement is applied here. If the file's raw MF3/MT=1 grid
    or INT law differs from its partial tabulations, the value
    this function returns for MT=1 can differ from the sum of
    partials at intermediate interpolation points (observed at
    ~3 mbarn on ~12 barn total in JEFF-4.0 U-235 above the URR;
    less than 1e-6 relative in the resolved-resonance region where
    MF3 is zero). NJOY reconr in contrast re-enforces the sum rule
    by writing MT=1 as the sum of reconstructed partials on its
    output grid. Callers who want the NJOY-consistent, sum-rule-
    enforced total should use :func:`endf_userpy.quantities.get_reaction_xs`,
    whose selector heuristic drops sum-MTs in favour of their
    leaf children when those children carry detailed data.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mf3_xs = mf3_interpretation.compute_cross_section_agnostic(
        endf_dict, mt, energies_in, xp,
    )
    resonance_xs = reconstruct_resonance_xs(
        endf_dict, mt, energies_in, xp,
    )
    return xp.asarray(mf3_xs) + xp.asarray(resonance_xs)
