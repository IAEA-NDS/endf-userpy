import numpy as np
from ..primitives import array_ns
from ..primitives.helpers import (
    dict2array,
    find_indices_with_tol,
)
from ..primitives.interpolation import endf_interp1d
from .mf12_interpretation_helpers import (
    get_discrete_series_mts,
    init_trans2yield,
    trans2yield,
)


def _is_jax_tracer(x):
    """Lazy-import isinstance(x, jax.core.Tracer). Returns False
    when jax is not installed (numpy-only environments)."""
    try:
        import jax  # noqa: F401
    except Exception:
        return False
    from jax.core import Tracer
    return isinstance(x, Tracer)


def get_photon_energies(endf_dict, mt):
    mtsec = endf_dict[12][mt]
    if mtsec['LO'] == 1:
        return dict2array(mtsec['Eg'])
    elif mtsec['LO'] == 2:
        res = compute_photon_yields_from_transition_probabilities(endf_dict, mt)
        return res['photon_energy']


def compute_photon_yields_from_transition_probabilities(
    endf_dict, mts, xp=None,
):
    if xp is None:
        xp = array_ns.get_backend('numpy')
    scalar_mt = not hasattr(mts, '__iter__')
    if scalar_mt:
        mts = [mts]
    series_mts = get_discrete_series_mts(
        endf_dict, mts[0], include_ground_state=False
    )
    if not np.all(np.isin(mts, series_mts)):
        raise ValueError(
            'All MT numbers must belong to the same discrete '
            ' MT series (e.g. fall in the range from 51 to 90 '
            'for inelastic neutron scattering.'
        )

    disc_mts, state_cache = init_trans2yield(endf_dict, mts[0], xp=xp)

    user_mts = set(mts)
    user_max_mt = max(user_mts)
    results = {}
    for mt in disc_mts:
        cur_result = (
            trans2yield(endf_dict, mt, state_cache, xp=xp)
        )
        if mt in user_mts:
            results[mt] = cur_result
            if mt == user_max_mt:
                break
    if scalar_mt:
        results = results[mts[0]]
    return results


def _as_1d_list(obj):
    """Normalise ``endf_parserpy``'s per-table columns to a flat
    Python list. The parser yields ``dict`` for some fields and
    ``list`` for others; both routes need a plain sequence for the
    bucket-signature key and for ``np.asarray``.
    """
    if hasattr(obj, 'values'):
        return list(obj.values())
    return list(obj)


def _table_signature(t):
    """(Eint, INT, NBT) tuple key for LO=1 bucket grouping."""
    return (
        tuple(_as_1d_list(t['Eint'])),
        tuple(_as_1d_list(t['INT'])),
        tuple(_as_1d_list(t['NBT'])),
    )


def _compute_photon_yields_bucketed_jax(tables, eincs, xp):
    """Batched LO=1 tabulated-yield interpolation for xp=jax.

    Groups tables sharing ``(Eint, INT, NBT)`` and evaluates each
    group with one ``endf_interp1d`` call over a stacked ``y[K_b, N]``
    array (Lever C of #290). For Al-27 MT102 the 291 photon-line
    tables collapse to 3 buckets, cutting the cold-jit graph of
    ``compute_photon_yields`` from ~10 s to ~0.25 s (only 3 XLA
    interp graphs to lower/compile instead of 291) and cutting the
    warm-time from ~0.11 ms to ~0.03 ms (one vectorised gather +
    arithmetic per bucket instead of per line).

    The scalar per-table code path is preserved for xp=numpy in the
    caller; there is no compile-time cost to save, and per-table
    numpy loops keep the read-through pattern of the original.
    """
    from collections import defaultdict
    buckets = defaultdict(list)                 # signature -> [table_idx]
    for k, t in enumerate(tables):
        buckets[_table_signature(t)].append(k)

    bucket_outputs = []
    bucket_cols = []
    for (ei_tup, int_tup, nbt_tup), idxs in buckets.items():
        ei_np = np.asarray(ei_tup, dtype=float)
        int_np = np.asarray(int_tup, dtype=int)
        nbt_np = np.asarray(nbt_tup, dtype=int)
        y_stack = xp.stack(
            [xp.asarray(_as_1d_list(tables[k]['y']), dtype=xp.float64)
             for k in idxs],
            axis=0,
        )                                       # (K_b, N)
        # ``endf_interp1d`` under xp=jax dispatches to
        # ``_endf_interp1d_traced_x``, which takes ``fp`` along the
        # last axis, so ``y_stack`` (K_b, N) yields (K_b, n_ein).
        y_at_ein = endf_interp1d(
            eincs, ei_np, y_stack, int_np, nbt_np,
            outside_value=0.0, xp=xp,
        )                                       # (K_b, n_ein)
        bucket_outputs.append(y_at_ein)
        bucket_cols.extend(idxs)

    combined = xp.concatenate(bucket_outputs, axis=0)   # (K, n_ein)
    # Reorder rows so row j has the original table-index-j yield.
    perm = np.asarray(bucket_cols, dtype=np.int64)
    inv_perm = np.argsort(perm)
    reordered = xp.take(combined, xp.asarray(inv_perm), axis=0)
    return reordered.T                                  # (n_ein, K)


def compute_photon_yields_from_tabulated_yields(
    endf_dict, mt, energies_in, xp=None,
):
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mtsec = endf_dict[12][mt]
    if mtsec['LO'] != 1:
        raise ValueError(
            f'MT{mt} does not contain photon multiplicities'
        )
    eincs = energies_in
    tables = list(mtsec['table'].values())

    level_energies = dict2array(mtsec['ES'])
    photon_energies = dict2array(mtsec['Eg'])
    if len(tables) == 0:
        n_ein = np.asarray(eincs).size
        photon_yields = xp.zeros((n_ein, 0), dtype=xp.float64)
    elif xp.name == 'jax':
        # Signature-bucketed batched interpolation (Lever C of #290).
        # Reduces the traced-graph size from O(K) per-line XLA subgraphs
        # to O(num-distinct-signatures) batched calls; on Al-27 MT102
        # (K=291 lines, 3 signatures) cold jit drops from ~10 s to
        # ~0.25 s and warm time from ~0.11 ms to ~0.03 ms. Numerically
        # bit-identical to the per-table path on real corpus files.
        photon_yields = _compute_photon_yields_bucketed_jax(
            tables, eincs, xp,
        )
    else:
        cols = [
            endf_interp1d(
                eincs, t['Eint'], t['y'], t['INT'], t['NBT'],
                outside_value=0.0, xp=xp,
            )
            for t in tables
        ]
        photon_yields = xp.stack(cols, axis=1)
    return {
        'level_energy': level_energies,
        'photon_energy': photon_energies,
        'photon_yield': photon_yields,
    }


def compute_photon_yields(
    endf_dict, mt, energies_in, photon_energies, xp=None,
):
    """MF12 photon yields y_gamma(E_in, E_gamma) for one MT.

    ``xp=None`` (default) is numpy and bit-identical to the pre-port
    behaviour. Passing an xp adapter threads tracers through both
    the LO=1 tabulated-yields interpolation and the LO=2
    transition-probability cascade so ``jax.grad`` reaches every
    file-side MF12 leaf (LO=1: ``y``; LO=2: ``TP``, ``GP``,
    ``ES``, ``ES_NS``, and the upstream ``ELIS``/``QM``/``QI``).
    Both ``jax.grad(jax.jit(f))`` and ``jax.jit(jax.grad(f))``
    compose end-to-end.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    mtsec = endf_dict[12][mt]
    LO_value = mtsec['LO']
    if LO_value == 1:
        res = compute_photon_yields_from_tabulated_yields(
            endf_dict, mt, energies_in, xp=xp,
        )
    elif LO_value == 2:
        res = compute_photon_yields_from_transition_probabilities(
            endf_dict, mt, xp=xp,
        )
        ones_vec = xp.ones_like(xp.asarray(energies_in)).reshape(-1, 1)
        res['photon_yield'] = (
            res['photon_yield'].reshape(1, -1) * ones_vec
        )
    else:
        raise ValueError(
            f'Invalid value LO={LO_value}'
        )

    # Select the requested photon energies. The concrete tolerance
    # matcher preserves the pre-port error message when it can
    # materialise ``res['photon_energy']`` (numpy, or xp=jax with
    # concrete file leaves). When ``ES_NS``/``ELIS``/``QM``/``QI``
    # tracers make ``photon_energy`` non-concrete under
    # ``jax.jit``/``jax.grad``, an xp-native ``argmin`` fallback
    # supplies the indices; the user is responsible for passing
    # ``photon_energies`` that actually exist in the file (an
    # equivalent Python-level ``raise`` would break the jit trace
    # anyway).
    photon_energies_np = np.asarray(photon_energies)
    if _is_jax_tracer(res['photon_energy']):
        diffs = xp.abs(
            res['photon_energy'][:, None]
            - xp.asarray(photon_energies_np)[None, :]
        )
        idcs = xp.argmin(diffs, axis=0)
    else:
        photon_energy_np = np.asarray(res['photon_energy'])
        idcs = find_indices_with_tol(
            photon_energy_np, photon_energies_np,
            atol=1e-4, rtol=1e-5,
        )
        if np.any(idcs == -1):
            raise ValueError(
                'All user-supplied `photon_energies` must exist '
                f'in MF12/MT{mt} but this is not the case.'
            )
    return res['photon_yield'][:, idcs]
