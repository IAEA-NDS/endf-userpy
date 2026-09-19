"""MF2 resolved-resonance reconstruction composed with MF3 background.

ENDF-6 stores the cross section in the resolved-resonance region as
two additive parts: MF2 (resonance parameters, reconstructed
point-wise by MLBW / R-M / etc.) plus MF3 (a smooth "background"
subtractive residual). The physical cross section on a query grid
is the sum. Outside every resolved range the MF2 contribution is
zero and the sum reduces to the MF3 interpolation.

Two entry points:

- :func:`reconstruct_resonance_xs` returns only the MF2 partial for
  a given MT, summed over every LRU=1 range in the file. Zero
  outside every range.
- :func:`compute_reconstructed_cross_section` returns
  ``MF3(mt, E) + MF2_partial(mt, E)`` — the composed physical
  cross section.

Both are backend-agnostic: pass an ``xp`` returned by
:func:`endf_userpy.primitives.array_ns.get_backend` to run on
numpy, numba, or JAX. The default (``xp=None``) resolves to
numpy.

Supported formalisms: LRF=2 (MLBW) and LRF=3 (Reich-Moore).
Unresolved-resonance ranges (LRU=2), Adler-Adler (LRF=4), and
R-matrix limited (LRF=7) contribute zero and, if no supported
range exists in the file at all, produce a :class:`UserWarning`.
"""

import warnings

from ..mfsec_interpretation import mf3_interpretation
from ..mfsec_interpretation import mf2_interpretation_mlbw
from ..mfsec_interpretation import mf2_interpretation_mlbw_preproc
from ..mfsec_interpretation import mf2_interpretation_reichmoore
from ..mfsec_interpretation import mf2_interpretation_reichmoore_preproc
from ..primitives import array_ns


# MT number -> which keys of the reconstruction result to sum.
# The reconstruction returns 'sct', 'cap', 'fis', 'pot', 'tot' (+ 'rxx'
# for MLBW). 'pot' is a diagnostic already included in 'sct', so it
# never appears here; 'tot' is the aggregate already summed for us.
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


def _reconstruct_range(endf_dict, iso_i, rng_i, rng, energies, xp):
    """Preprocess + reconstruct a single supported range.

    Returns ``(recon_dict, mt_to_keys_map)`` or ``(None, None)`` if
    the range is not a supported formalism (caller then skips it).
    """
    lrf = int(rng.get('LRF', 0))
    if lrf == 2:
        data = mf2_interpretation_mlbw_preproc.mlbw_data_from_endf_dict(
            endf_dict, isotope_idx=iso_i, range_idx=rng_i,
        )
        recon = mf2_interpretation_mlbw.reconstruct(data, energies, xp)
        return recon, _MLBW_MT_TO_KEYS
    if lrf == 3:
        data = mf2_interpretation_reichmoore_preproc.rm_data_from_endf_dict(
            endf_dict, isotope_idx=iso_i, range_idx=rng_i,
        )
        recon = mf2_interpretation_reichmoore.reconstruct(data, energies, xp)
        return recon, _RM_MT_TO_KEYS
    return None, None


def reconstruct_resonance_xs(endf_dict, mt, energies_in, xp=None):
    """MF2 partial-XS contribution for ``mt``, summed over every
    LRU=1 range in ``endf_dict``. Zero at query energies outside
    every range (per convention: the resonance representation is
    defined only on ``[EL, EH]``).

    ``mt`` outside the supported set
    ``{1, 2, 3, 18, 27, 102}`` returns zero without a warning:
    higher-MT reactions like MT=51 discrete inelastic or MT=16
    (n,2n) legitimately have no MF2 contribution and are handled
    entirely by MF3.

    Ranges with unsupported LRF (Adler-Adler, R-matrix limited)
    are skipped. If the file has no supported range at all, a
    :class:`UserWarning` is emitted (once per call) naming which
    formalism was seen.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    e = xp.asarray(energies_in, dtype=xp.float64)
    total = xp.zeros_like(e)

    saw_supported = False
    unsupported = []
    for iso_i, rng_i, rng in _iter_lru1_ranges(endf_dict):
        lrf = int(rng.get('LRF', 0))
        if lrf not in (2, 3):
            unsupported.append((iso_i, rng_i, lrf))
            continue
        saw_supported = True
        keys = _MLBW_MT_TO_KEYS.get(mt) if lrf == 2 else _RM_MT_TO_KEYS.get(mt)
        if not keys:
            # Supported formalism but MT does not receive an MF2
            # contribution (e.g. MT=51). Reconstruction would still
            # compute correctly; skip it to avoid the work.
            continue
        el = float(rng['EL'])
        eh = float(rng['EH'])
        in_range = (e >= el) & (e <= eh)
        # Guard the reconstruction call itself with a zero-early-out
        # when the query grid does not touch the range at all: the
        # preproc + kernel would still work, but we can save the cost.
        try:
            any_in = bool(in_range.any())
        except Exception:
            any_in = True    # JAX tracer: always run
        if not any_in:
            continue
        recon, _ = _reconstruct_range(endf_dict, iso_i, rng_i, rng, e, xp)
        if recon is None:
            continue
        contrib = xp.zeros_like(e)
        for k in keys:
            if k in recon:
                contrib = contrib + recon[k]
        total = total + xp.where(in_range, contrib, xp.zeros_like(e))

    if unsupported and not saw_supported:
        parts = ', '.join(f'iso={i} rng={j} LRF={f}'
                          for i, j, f in unsupported)
        warnings.warn(
            f'reconstruct_resonance_xs: no supported LRF (2 MLBW / 3 '
            f'Reich-Moore) LRU=1 range found (saw: {parts}); returning '
            f'zero resonance contribution. Adler-Adler (LRF=4) and '
            f'R-matrix limited (LRF=7) are not implemented yet.',
            UserWarning, stacklevel=2,
        )
    return total


def compute_reconstructed_cross_section(
    endf_dict, mt, energies_in, xp=None,
):
    """Physical cross section for ``mt`` per ENDF-6:

        sigma(E) = MF3(mt, E) + sum over LRU=1 ranges of MF2_partial(mt, E)

    Backend-agnostic: pass ``xp`` from
    :func:`endf_userpy.primitives.array_ns.get_backend` to run on
    numpy, numba, or JAX. Default ``xp=None`` resolves to numpy.

    The MF3 term is the raw TAB1 interpolation (via
    :func:`mfsec_interpretation.mf3_interpretation.compute_cross_section_agnostic`),
    without the ``above_range`` / ``resonance_range`` policy
    machinery of the policy-driven
    :func:`~mf3_interpretation.compute_cross_section` — the whole
    point of composing MF2 in is that the "raw MF3 is a background,
    warn about it" policy is no longer needed. Callers who still
    want above-range policy on top of the composed cross section
    should apply it themselves.

    See :func:`reconstruct_resonance_xs` for supported LRF list and
    warning behaviour on unsupported formalisms.
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
