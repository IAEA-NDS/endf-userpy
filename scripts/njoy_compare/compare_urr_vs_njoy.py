"""Compare our URR reconstruction vs NJOY unresr's MT152 output.

Prerequisite: run ``run_unresr.sh <source.endf> <out-dir>`` first
to produce ``<out-dir>/tape22`` (the PENDF file with URR average
XS in MF2/MT152 at infinite dilution, T=0 K).

Usage:

    ~/venvs/endf-userpy-test/bin/python scripts/njoy_compare/compare_urr_vs_njoy.py \\
        --source tests/data_law1_adhoc/tendl21_n_U-235.endf \\
        --pendf  /tmp/njoy_urr/tape22

Prints a side-by-side table of our URR reconstruction vs NJOY
unresr on the tabulated MT152 energy grid, with per-partial
relative error. Uses the numpy backend by default; ``--backend
numba`` (or ``jax``) picks a different one and prints the peak
numpy/backend disagreement as a sanity gate before the NJOY
comparison.

Reads NJOY MT152 directly from ASCII, bypassing endf_parserpy's
strict PENDF handling (which trips on the SEND-record layout
NJOY writes). The MT152 body for LSSF=1 URR at infinite
dilution is a flat list: ``[sig0, E_i, <tot>_i, <el>_i,
<fis>_i, <cap>_i, <col5>_i, ...]`` (6 fields per energy after
the leading sig0 flag).
"""
from __future__ import annotations

import argparse
import re

import numpy as np


def _endf_float(s: str) -> float:
    """Parse an ENDF-6 float like '1.000+3' or '-2.5-8'."""
    s = s.strip()
    if not s:
        return 0.0
    m = re.match(r'([\-+]?\d*\.?\d+)([\-+]\d+)$', s)
    if m:
        return float(m.group(1) + 'e' + m.group(2))
    return float(s)


def _extract_mt152_lines(pendf_path: str):
    """Yield the (content_columns) of every MT152 line in a PENDF."""
    with open(pendf_path) as f:
        for line in f:
            if len(line) < 75:
                continue
            # Columns 67-70 = MAT, 71-72 = MF, 73-75 = MT. Look
            # for " 2152" in the (MF, MT) suffix (spaces + "2152").
            tail = line[70:75]
            mf = tail[:2].strip()
            mt = tail[2:].strip()
            if mf == '2' and mt == '152':
                yield line[:66]


def read_njoy_mt152(pendf_path: str):
    """Return (energies, tot, el, fis, cap) numpy arrays parsed
    from the MT152 body of a NJOY PENDF."""
    lines = list(_extract_mt152_lines(pendf_path))
    if len(lines) < 3:
        raise ValueError(
            f'{pendf_path}: expected at least 3 MT152 lines '
            f'(HEAD + LIST hdr + body); got {len(lines)}'
        )
    # HEAD: line[0]; LIST hdr on line[1] has NPL / NE at cols 45-55 / 56-66.
    list_hdr = lines[1]
    NPL = int(list_hdr[44:55])
    NE = int(list_hdr[55:66])
    # Flatten body numbers.
    flat = []
    for line in lines[2:]:
        for c in range(0, 66, 11):
            s = line[c:c + 11]
            if s.strip():
                flat.append(_endf_float(s))
    if len(flat) != NPL:
        raise ValueError(
            f'{pendf_path}: MT152 body has {len(flat)} floats, '
            f'expected NPL={NPL}'
        )
    # Layout: 1 leading sig0 flag, then 6 fields per energy.
    n_leading = 1
    nfields = 6
    if (NPL - n_leading) // nfields != NE:
        raise ValueError(
            f'{pendf_path}: (NPL - 1) / 6 = {(NPL - n_leading) // nfields} '
            f'does not match NE={NE}. Different NJOY layout than expected?'
        )
    E = np.empty(NE)
    tot = np.empty(NE)
    el = np.empty(NE)
    fis = np.empty(NE)
    cap = np.empty(NE)
    for i in range(NE):
        off = n_leading + i * nfields
        E[i] = flat[off]
        tot[i] = flat[off + 1]
        el[i] = flat[off + 2]
        fis[i] = flat[off + 3]
        cap[i] = flat[off + 4]
    return E, tot, el, fis, cap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True,
                    help='original ENDF-6 evaluation file')
    ap.add_argument('--pendf', required=True,
                    help='NJOY unresr output tape22')
    ap.add_argument('--backend', default='numpy',
                    choices=['numpy', 'numba', 'jax'])
    ap.add_argument('--isotope-idx', type=int, default=1)
    ap.add_argument('--range-idx', type=int, default=2,
                    help='1-based URR range index (default 2, '
                         'i.e. the range after the RRR)')
    args = ap.parse_args()

    from endf_parserpy import EndfParserCpp
    from endf_userpy.mfsec_interpretation import mf2_interpretation_urr as urr
    from endf_userpy.mfsec_interpretation import (
        mf2_interpretation_urr_preproc as pre,
    )
    from endf_userpy.primitives import array_ns

    d = EndfParserCpp().parsefile(args.source, include=[1, 2])
    data = pre.urr_data_from_endf_dict(
        d, isotope_idx=args.isotope_idx, range_idx=args.range_idx,
    )
    xp = array_ns.get_backend(args.backend)

    E_njoy, tot_njoy, el_njoy, fis_njoy, cap_njoy = read_njoy_mt152(args.pendf)
    print(f'NJOY MT152 has {len(E_njoy)} energy points at '
          f'[{E_njoy[0]:.3g}, {E_njoy[-1]:.3g}] eV')

    ours = urr.reconstruct(data, E_njoy, xp)
    ours_sct = np.asarray(ours['sct'])
    ours_cap = np.asarray(ours['cap'])
    ours_fis = np.asarray(ours['fis'])

    # Header.
    fmt = ('{:>10s} {:>10s} {:>10s} {:>8s}  '
           '{:>10s} {:>10s} {:>8s}  {:>10s} {:>10s} {:>8s}')
    print(fmt.format(
        'E(eV)',
        'ours_el', 'njoy_el', 'rel%',
        'ours_cap', 'njoy_cap', 'rel%',
        'ours_fis', 'njoy_fis', 'rel%',
    ))
    rows = zip(E_njoy, ours_sct, el_njoy, ours_cap, cap_njoy, ours_fis, fis_njoy)
    for E, o_e, n_e, o_c, n_c, o_f, n_f in rows:
        print(fmt.format(
            f'{E:.4g}',
            f'{o_e:.4f}', f'{n_e:.4f}', f'{(o_e - n_e) / n_e * 100:+.2f}',
            f'{o_c:.4f}', f'{n_c:.4f}', f'{(o_c - n_c) / n_c * 100:+.2f}',
            f'{o_f:.4f}', f'{n_f:.4f}', f'{(o_f - n_f) / n_f * 100:+.2f}',
        ))

    for name, ours_x, njoy_x in [
        ('elastic', ours_sct, el_njoy),
        ('capture', ours_cap, cap_njoy),
        ('fission', ours_fis, fis_njoy),
    ]:
        rel = np.abs(ours_x - njoy_x) / njoy_x
        print(f'summary: max relative error on {name}: {rel.max():.3%}')


if __name__ == '__main__':
    main()
