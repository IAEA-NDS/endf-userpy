"""Compare our resonance reconstruction against NJOY reconr output.

Parses a PENDF file (from ``run_reconr.sh``) and re-runs our own
reconstruction on the exact same energy grid using the numba
backend. Reports per-MT median / mean / max-scaled relative
difference, the point of maximum disagreement, and wall-clock
timing. See README.md.

Deliberately backend-neutral in the script (numba by default) so
million-point actinide grids don't OOM.
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np


def _sketch_reconstruct(formalism, source_endf, energies, backend='numba'):
    """Preprocess + reconstruct via the sketch, returning the same
    dict of {sct, cap, fis, pot, tot} that the sketch produces."""
    from endf_parserpy import EndfParserCpp
    from endf_userpy.primitives import array_ns

    d = EndfParserCpp().parsefile(source_endf, include=[1, 2])
    if formalism == 'mlbw':
        from endf_userpy.mfsec_interpretation import (
            mf2_interpretation_mlbw as fmt,
            mf2_interpretation_mlbw_preproc as pre,
        )
        data = pre.mlbw_data_from_endf_dict(d)
    elif formalism == 'rm':
        from endf_userpy.mfsec_interpretation import (
            mf2_interpretation_reichmoore as fmt,
            mf2_interpretation_reichmoore_preproc as pre,
        )
        data = pre.rm_data_from_endf_dict(d)
    else:
        raise ValueError(f'formalism must be mlbw or rm, got {formalism!r}')
    xp = array_ns.get_backend(backend)
    # Small warm-up so timings measure steady state (jit compile, etc.).
    _ = fmt.reconstruct(data, energies[:16], xp)
    t0 = time.perf_counter()
    xs = fmt.reconstruct(data, energies, xp)
    return {k: np.asarray(v) for k, v in xs.items()}, time.perf_counter() - t0


def _load_pendf_mt(pendf_path, mt):
    from endf_parserpy import EndfParserCpp
    d_pendf = EndfParserCpp().parsefile(pendf_path, include=[3])
    if mt not in d_pendf[3]:
        return None, None
    xstab = d_pendf[3][mt]['xstable']
    return (
        np.asarray(xstab['E'], dtype=np.float64),
        np.asarray(xstab['xs'], dtype=np.float64),
    )


# Map MT numbers to the sketch's output dict key.
MT_TO_KEY = {2: 'sct', 18: 'fis', 102: 'cap', 1: 'tot'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True,
                    help='path to source ENDF-6 file')
    ap.add_argument('--pendf', required=True,
                    help='path to NJOY PENDF file (typically tape21)')
    ap.add_argument('--formalism', choices=('mlbw', 'rm'), required=True)
    ap.add_argument('--rrr-hi', type=float, required=True,
                    help='RRR upper bound in eV (queries are clipped to '
                         '[1e-5, rrr_hi]; U-235 = 2250, Nb-93 = 7000)')
    ap.add_argument('--mts', default='1,2,102',
                    help='comma-separated MT list; 1,2,18,102 typical for '
                         'fissile actinides, 1,2,102 for non-fissile')
    ap.add_argument('--backend', default='numba',
                    choices=('numpy', 'numba', 'jax'),
                    help='reconstruction backend; numba is memory-safe and '
                         'the default')
    args = ap.parse_args()

    print(f'source ENDF: {args.source} ({os.path.getsize(args.source)/1e6:.2f} MB)')
    print(f'PENDF:       {args.pendf}    ({os.path.getsize(args.pendf)/1e6:.2f} MB)')
    print(f'formalism:   {args.formalism}   RRR upper = {args.rrr_hi} eV')
    print(f'backend:     {args.backend}')
    print()

    mts = [int(x) for x in args.mts.split(',')]

    for mt in mts:
        if mt not in MT_TO_KEY:
            print(f'MT={mt}: unsupported (only {sorted(MT_TO_KEY)} handled); skip')
            continue
        key = MT_TO_KEY[mt]

        e, sig_njoy = _load_pendf_mt(args.pendf, mt)
        if e is None:
            print(f'MT={mt}: not present in PENDF; skip')
            continue

        mask = (e >= 1e-5) & (e <= args.rrr_hi)
        e_r = e[mask]; sig_r = sig_njoy[mask]

        xs, t = _sketch_reconstruct(args.formalism, args.source, e_r,
                                     backend=args.backend)
        sig_ours = xs[key]

        abs_diff = np.abs(sig_ours - sig_r)
        scale = float(np.max(np.abs(sig_r)))
        rel = abs_diff / np.maximum(np.abs(sig_r), 1e-30)
        i = int(np.argmax(abs_diff))

        print(f'MT={mt:>3} ({key}):  {e_r.shape[0]:>7} points in RRR')
        print(f'  peak σ         = {scale:>11.4e} b')
        print(f'  median rel err = {float(np.median(rel)):>11.4e}')
        print(f'  mean rel err   = {float(np.mean(rel)):>11.4e}')
        print(f'  max abs / peak = {float(np.max(abs_diff))/scale if scale>0 else 0.0:>11.4e}')
        print(f'    at E = {float(e_r[i]):>10.4e} eV: '
              f'ours={float(sig_ours[i]):.6e}  njoy={float(sig_r[i]):.6e}')
        print(f'  reconstruct wall ({args.backend}): {t*1000:>8.1f} ms  '
              f'({t/e_r.shape[0]*1e6:.3f} µs/energy)')
        print()


if __name__ == '__main__':
    main()
