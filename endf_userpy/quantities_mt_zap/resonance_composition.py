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

Supported LRU=1 formalisms: LRF=2 (MLBW), LRF=3 (Reich-Moore),
and LRF=7 R-Matrix Limited with the KRM=3 Reich-Moore
approximation. Supported LRU=2 formalisms: LRF=2 with INT=2
(Case C energy-dependent widths, lin-lin table interpolation).

Ranges the current implementation cannot reconstruct
(Adler-Adler LRF=4, LRF=7 with KRM other than 3 or with
KBK / KPS / IFG / NRO out of the initial-scope
zero-values, URR with non-INT=2 tables, ...) contribute zero. If
a file has no supported range at all, or has an LSSF=0 URR range
the reconstructor cannot handle, a :class:`UserWarning` names
the specific formalism / INT code seen.
"""

import warnings

import numpy as np

from ..mfsec_interpretation import mf3_interpretation
from ..mfsec_interpretation import mf2_interpretation_mlbw
from ..mfsec_interpretation import mf2_interpretation_mlbw_preproc
from ..mfsec_interpretation import mf2_interpretation_reichmoore
from ..mfsec_interpretation import mf2_interpretation_reichmoore_preproc
from ..mfsec_interpretation import mf2_interpretation_rml
from ..mfsec_interpretation import mf2_interpretation_rml_preproc
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

# LRF=7 KRM=3 partials mirror the Reich-Moore shape: gamma is
# eliminated, so 'cap' aggregates radiative capture and any other
# implicit-channel contribution; 'fis' aggregates every explicit
# fission channel; there is no competitive channel by construction.
_RML_MT_TO_KEYS = _RM_MT_TO_KEYS

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
        # Thread xp into the preproc so JAX tracers stored at
        # per-resonance dict leaves (ER / GN / GG / GF / GT / QX)
        # survive into the reconstruction (issue #159).
        data = mf2_interpretation_mlbw_preproc.mlbw_data_from_endf_dict(
            endf_dict, isotope_idx=iso_i, range_idx=rng_i, xp=xp,
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
    if lrf == 7:
        # LRF=7 R-Matrix Limited (KRM=3 Reich-Moore variant); the
        # preproc rejects KRM != 3 with a clear NotImplementedError,
        # so the composition layer never silently returns something
        # wrong. Threading xp preserves JAX tracers on ER / GAM
        # leaves the same way as LRF=3.
        try:
            data = mf2_interpretation_rml_preproc.rml_data_from_endf_dict(
                endf_dict, isotope_idx=iso_i, range_idx=rng_i, xp=xp,
            )
        except NotImplementedError:
            # Out-of-scope KRM / KRL / IFG / NRO / KBK / KPS: fall
            # through to the caller's unsupported-LRF handling so
            # the user sees a single summary warning instead of a
            # deep traceback.
            return None, None
        recon = mf2_interpretation_rml.reconstruct(data, energies, xp)
        return recon, _RML_MT_TO_KEYS
    return None, None


def _reconstruct_urr_range(endf_dict, iso_i, rng_i, rng, energies, xp,
                           urr_quadrature='gauss_legendre_32'):
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
        # Thread xp so JAX tracers stored at per-J-group dict
        # leaves (ES / D / GN0 / GG / GF / GX) survive into the URR
        # kernel (issue #159).
        data = mf2_interpretation_urr_preproc.urr_data_from_endf_dict(
            endf_dict, isotope_idx=iso_i, range_idx=rng_i, xp=xp,
        )
    except Exception as exc:
        return None, f'preproc failed: {type(exc).__name__}: {exc}'
    try:
        recon = mf2_interpretation_urr.reconstruct(
            data, energies, xp, quadrature=urr_quadrature,
        )
    except Exception as exc:
        return None, f'kernel failed: {type(exc).__name__}: {exc}'
    return recon, _URR_MT_TO_KEYS


# JAX eager mode compiles every op once per new array shape. Slicing
# to an in-range subset introduces new shapes, which costs ~0.8 s of
# first-call compilation for a Reich-Moore range; only slice when that
# skips at least this many out-of-range energies (measured on U-235:
# 1K-point meshes are faster unsliced on the first call, 10K+ sliced).
_JAX_SLICE_MIN_SKIPPED = 2048


def _accumulate_range_contrib(total, e, in_range, keys, xp, reco_fn):
    """Add one resonance range's contribution onto ``total``.

    Numpy-backend fast path (issue #307): materialise the in-range
    indices, slice ``e`` to that subset, run ``reco_fn`` on the
    slice, scatter the result back. Avoids evaluating the formalism
    on Ein points that would be zeroed by the ``xp.where`` mask
    (log-spaced XS queries that straddle an actinide RRR waste 25
    to 94 pct of the work on points above the RRR upper bound).

    JAX: when the energies are concrete (the eager case; the
    in-range mask converts to numpy) and slicing skips at least
    :data:`_JAX_SLICE_MIN_SKIPPED` points, the same slice runs on the
    jax arrays and the slice result is scatter-added back with
    ``.at[idx].add`` -- differentiable wrt every file-side leaf the
    formalism reads. When the energies are traced (``jax.jit`` over
    ``e``, ``jax.grad`` wrt ``e``) slicing by a traced mask would
    give a data-dependent shape, so the fixed-shape pre-#307
    "evaluate full ``e``, mask after" path runs instead.

    Returns the new running total, or ``None`` when the formalism
    could not reconstruct the range (``reco_fn`` returned ``None``);
    callers propagate that into their own error-tracking.
    """
    if xp.name == 'jax':
        try:
            in_range_np = np.asarray(in_range)
        except Exception:      # traced energies: fixed-shape path
            in_range_np = None
        if (in_range_np is not None
                and in_range_np.size - np.count_nonzero(in_range_np)
                < _JAX_SLICE_MIN_SKIPPED):
            # Too little to skip: the slice's new array shape would
            # cost more in first-call op compilation than it saves.
            in_range_np = None
        if in_range_np is None:
            recon = reco_fn(e)
            if recon is None:
                return None
            contrib = xp.zeros_like(e)
            for k in keys:
                if k in recon:
                    contrib = contrib + recon[k]
            return total + xp.where(in_range, contrib, xp.zeros_like(e))
        idx_in = np.where(in_range_np)[0]
        if idx_in.size == 0:
            return total
        recon = reco_fn(e[idx_in])
        if recon is None:
            return None
        contrib_in = xp.zeros((idx_in.size,), dtype=e.dtype)
        for k in keys:
            if k in recon:
                contrib_in = contrib_in + recon[k]
        return total.at[idx_in].add(contrib_in)

    in_range_np = np.asarray(in_range)
    idx_in = np.where(in_range_np)[0]
    if idx_in.size == 0:
        return total
    e_np = np.asarray(e)
    e_slice = e_np[idx_in]
    recon = reco_fn(e_slice)
    if recon is None:
        return None
    contrib_in = np.zeros_like(e_slice)
    for k in keys:
        if k in recon:
            contrib_in = contrib_in + np.asarray(recon[k])
    total_np = np.asarray(total).copy()
    total_np[idx_in] += contrib_in
    return total_np


def _per_call_cached(query_state, key, energies_in, reco_fn):
    """Wrap ``reco_fn`` (one range's reconstruction on the slice of
    ``energies_in`` the caller selects) with the per-call cache on
    ``query_state.range_recon_cache``.

    The MTs of one top-level query each read different keys of the
    same range reconstruction; the cache computes it once. An entry
    stores ``energies_in`` itself and is reused only when the next
    call passes that same object (``is``), so an id-recycled or
    different mesh can never hit; keeping the reference also keeps
    the array alive for the lifetime of the cache (one top-level
    call). Without a ``query_state`` (leaf-level callers) there is no
    cache, as before.
    """
    if query_state is None:
        return reco_fn
    cache = query_state.range_recon_cache

    def cached(e_slice):
        hit = cache.get(key)
        if hit is not None and hit[0] is energies_in:
            return hit[1]
        out = reco_fn(e_slice)
        cache[key] = (energies_in, out)
        return out

    return cached


def _host_energies(energies_in, query_state):
    """The query energies as a float64 numpy array: directly when they
    are concrete, from ``query_state.host_energies`` when they are the
    staged mesh of a ``jax.jit`` trace, else ``None`` (traced)."""
    staged = getattr(query_state, 'host_energies', None)
    if staged is not None and energies_in is staged[0]:
        return staged[1]
    try:
        return np.asarray(energies_in, dtype=np.float64)
    except Exception:          # tracer
        return None


def reconstruct_resonance_xs(endf_dict, mt, energies_in, xp=None,
                             urr_quadrature='gauss_legendre_32',
                             _query_state=None):
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

    LRU=1 (RRR): supported LRFs are 2 (MLBW), 3 (Reich-Moore),
    and 7 (R-Matrix Limited with the KRM=3 Reich-Moore
    approximation, KRL=0, IFG=0, NRO=0, KBK=0, KPS=0).
    Unsupported LRFs (Adler-Adler LRF=4, LRF=7 outside the
    initial scope) are skipped. If the file has no supported
    LRU=1 range at all, a :class:`UserWarning` names the
    formalism seen.

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
    total = _resonance_xs(
        endf_dict, mt, energies_in, xp,
        urr_quadrature=urr_quadrature, _query_state=_query_state,
    )
    if total is None:
        return xp.zeros_like(xp.asarray(energies_in, dtype=xp.float64))
    return total


def _resonance_xs(endf_dict, mt, energies_in, xp,
                  urr_quadrature='gauss_legendre_32', _query_state=None):
    """Body of :func:`reconstruct_resonance_xs`, but returns ``None``
    instead of an all-zero array when no supported range contributes
    to ``mt`` at any query energy (the common case: most MTs have no
    MF2 contribution), so the composition can skip allocating and
    adding zeros on the full mesh."""
    e = xp.asarray(energies_in, dtype=xp.float64)
    total = None
    # In-range masks on the host whenever the energies are concrete --
    # also for the staged mesh of a jax.jit trace -- so
    # ``_accumulate_range_contrib`` reconstructs only the in-range
    # points; traced energies keep the xp-native full-mesh mask.
    e_mask = _host_energies(energies_in, _query_state)
    if e_mask is None:
        e_mask = e

    # ---- LRU=1 (RRR): MLBW / Reich-Moore / R-Matrix Limited.
    saw_lru1_supported = False
    lru1_unsupported = []
    for iso_i, rng_i, rng in _iter_lru1_ranges(endf_dict):
        lrf = int(rng.get('LRF', 0))
        if lrf not in (2, 3, 7):
            lru1_unsupported.append((iso_i, rng_i, lrf))
            continue
        saw_lru1_supported = True
        if lrf == 2:
            keys = _MLBW_MT_TO_KEYS.get(mt)
        elif lrf == 3:
            keys = _RM_MT_TO_KEYS.get(mt)
        else:
            keys = _RML_MT_TO_KEYS.get(mt)
        if not keys:
            # Mode 2 (#311): if the user asked for this MT at the
            # top-level entry point, record it so the summary
            # warning fires. Scoped to user_mts so the widened MT
            # iteration in differential queries doesn't emit a
            # warning for every incidentally-visited MT.
            if (
                _query_state is not None
                and mt in getattr(_query_state, 'user_mts', set())
                and mt in endf_dict.get(3, {})
            ):
                _query_state.unmapped_composition_mts.append(mt)
            continue
        el = float(rng['EL'])
        eh = float(rng['EH'])
        # Half-open [EL, EH): at the seam E == EH the next range (URR
        # or MF3 above URR) owns the value. Matches NJOY's right-limit
        # convention at doubled-x range transitions (issue #149; same
        # side selection as PR #146 for MF3 tab1 lookups).
        in_range = (e_mask >= el) & (e_mask < eh)
        try:
            any_in = bool(in_range.any())
        except Exception:
            any_in = True    # JAX tracer: always run
        if not any_in:
            continue
        new_total = _accumulate_range_contrib(
            xp.zeros_like(e) if total is None else total,
            e, in_range, keys, xp,
            _per_call_cached(
                _query_state, ('lru1', iso_i, rng_i, xp.name), energies_in,
                lambda e_slice: _reconstruct_lru1_range(
                    endf_dict, iso_i, rng_i, rng, e_slice, xp,
                )[0],
            ),
        )
        if new_total is None:
            continue
        total = new_total

    if lru1_unsupported and not saw_lru1_supported:
        parts = ', '.join(f'iso={i} rng={j} LRF={f}'
                          for i, j, f in lru1_unsupported)
        warnings.warn(
            f'reconstruct_resonance_xs: no supported LRF (2 MLBW / 3 '
            f'Reich-Moore / 7 R-Matrix Limited KRM=3) LRU=1 range '
            f'found (saw: {parts}); returning zero resolved-resonance '
            f'contribution. Adler-Adler (LRF=4) is not implemented.',
            UserWarning, stacklevel=2,
        )

    # ---- LRU=2 LSSF=0 (URR needing reconstruction).
    urr_unsupported = []
    keys_urr = _URR_MT_TO_KEYS.get(mt)
    for iso_i, rng_i, rng in _iter_lru2_lssf0_ranges(endf_dict):
        if not keys_urr:
            # MT does not receive an MF2 contribution in the URR
            # either; skip the reconstruction work. Mode 2 (#311):
            # if the user asked for this MT and it is in MF3, note
            # the gap so the summary warning fires.
            if (
                _query_state is not None
                and mt in getattr(_query_state, 'user_mts', set())
                and mt in endf_dict.get(3, {})
            ):
                _query_state.unmapped_composition_mts.append(mt)
            continue
        el = float(rng['EL'])
        eh = float(rng['EH'])
        # Half-open [EL, EH): at E == EH the MF3 above-URR tabulation
        # owns the value. See issue #149.
        in_range = (e_mask >= el) & (e_mask < eh)
        try:
            any_in = bool(in_range.any())
        except Exception:
            any_in = True
        if not any_in:
            continue

        def _reco_urr(e_slice):
            # (recon or None, failure reason) cached together so a
            # cache hit still reports the reason.
            return _reconstruct_urr_range(
                endf_dict, iso_i, rng_i, rng, e_slice, xp,
                urr_quadrature=urr_quadrature,
            )

        _err_box = [None]
        cached_urr = _per_call_cached(
            _query_state, ('urr', iso_i, rng_i, xp.name, urr_quadrature),
            energies_in, _reco_urr,
        )

        def _reco_urr_recon(e_slice, _err=_err_box):
            recon_, err_ = cached_urr(e_slice)
            _err[0] = err_
            return recon_

        new_total = _accumulate_range_contrib(
            xp.zeros_like(e) if total is None else total,
            e, in_range, keys_urr, xp, _reco_urr_recon,
        )
        if new_total is None:
            # Reconstruction kernel refused the range; carry the error
            # out and keep the running total untouched.
            urr_unsupported.append((iso_i, rng_i, _err_box[0]))
            continue
        total = new_total

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
    urr_quadrature='gauss_legendre_32',
    _query_state=None,
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
    resonance_xs = _resonance_xs(
        endf_dict, mt, energies_in, xp,
        urr_quadrature=urr_quadrature,
        _query_state=_query_state,
    )
    if resonance_xs is None:
        # No range contributes to this MT: MF3 + 0 is MF3; skip the
        # full-mesh zeros and the add.
        return xp.asarray(mf3_xs)
    return xp.asarray(mf3_xs) + xp.asarray(resonance_xs)
