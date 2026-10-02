"""Benchmark `benchmark_seed_strategies` across available actinide
corpus files, logging progress line-by-line to a file that can be
tailed live.

Writes to the LOG path below. Each entry is flushed immediately so
`tail -f` sees updates in real time.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

LOG = os.environ.get(
    'BENCH_LOG',
    os.path.join(os.path.dirname(__file__), 'benchmark_actinides.log'),
)

from endf_parserpy import EndfParserCpp  # noqa: E402

from endf_userpy.linearization import benchmark_seed_strategies  # noqa: E402
from endf_userpy.primitives import array_ns  # noqa: E402
from endf_userpy.run_options import RunOptions  # noqa: E402


# (label, path, reaction_strings, (e_min, e_max))
# Th-232 and Pu-241 are not in fetch.sh; running on the four available
# actinide files only. Pu-239 has both ENDF/B-VIII.1 and CENDL-3.2
# copies; using ENDF/B-VIII.1 for apples-to-apples alongside the others.
TARGETS = [
    ('U-233 ENDF/B-VIII.1',
     'tests/data_law1_adhoc/endfb81_n_U-233.endf'),
    ('U-235 TENDL-2021',
     'tests/data_law1_adhoc/tendl21_n_U-235.endf'),
    ('U-238 JENDL-5',
     'tests/data_law1_adhoc/jendl5_n_U-238.endf'),
    ('Pu-239 ENDF/B-VIII.1',
     'tests/data_law1_adhoc/endfb81_n_Pu-239.endf'),
]
# Elastic scattering = MT=2 → '(n,n_0)' resolves to it via the alias
# fallback. '(n,g)' = capture (MT=102). '(n,fission)' = total fission
# (MT=18), avoiding the '(n,f)' alternative which maps to the
# first-chance variant MT=19.
REACTIONS = [
    ('elastic   ', '(n,n_0)'),
    ('capture   ', '(n,g)'),
    ('fission   ', '(n,fission)'),
]
E_RANGE = (1.0, 2000.0)   # eV; covers the resonance-rich RRR window
TOL_REL = 1e-3


def log(msg: str, f=None):
    stamp = datetime.now().strftime('%H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if f is not None:
        f.write(line + '\n')
        f.flush()


def main():
    with open(LOG, 'w') as f:
        log(f'benchmark log: {LOG}', f)

        try:
            xp_numba = array_ns.get_backend('numba')
            log(
                f'numba backend available: {type(xp_numba).__name__}'
                f' (name={xp_numba.name!r})', f,
            )
        except Exception as exc:
            log(f'numba backend UNAVAILABLE ({exc!r}); aborting', f)
            return

        options = RunOptions(backend=xp_numba)
        log(f'using options: {options}', f)
        log(f'energy range: {E_RANGE} eV, tol_rel={TOL_REL}', f)

        # One warm-up call so the numba JIT compile cost does not fall
        # on the first timed run.
        d_warm = EndfParserCpp().parsefile(TARGETS[0][1])
        log('warming up numba kernels (one-off JIT cost)...', f)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _ = benchmark_seed_strategies(
                d_warm, '(n,g)', 1.0, 10.0, tol_rel=TOL_REL,
                options=options, verify_mesh_size=1000,
            )
        log(f'warm-up done in {time.perf_counter() - t0:.1f}s', f)

        rows = []
        total_runs = len(TARGETS) * len(REACTIONS)
        done = 0

        for label, path in TARGETS:
            if not os.path.exists(path):
                log(f'SKIP {label}: file not found at {path}', f)
                continue
            log(f'parsing {label} from {path}', f)
            t0 = time.perf_counter()
            d = EndfParserCpp().parsefile(path)
            log(f'  parse took {time.perf_counter() - t0:.1f}s', f)

            for rname, rstr in REACTIONS:
                done += 1
                log(
                    f'[{done}/{total_runs}] {label}: {rname} '
                    f'reaction={rstr} range={E_RANGE[0]:g}..{E_RANGE[1]:g} eV',
                    f,
                )
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        b = benchmark_seed_strategies(
                            d, rstr, E_RANGE[0], E_RANGE[1],
                            tol_rel=TOL_REL, options=options,
                        )
                except Exception as exc:
                    log(
                        f'  FAILED ({type(exc).__name__}: {exc})', f,
                    )
                    continue
                for r in b.results:
                    log(
                        f'  {r.strategy:22s}  seed={r.seed_size:6d}  '
                        f'final={r.final_size:6d}  iter={r.iterations:3d}  '
                        f'wall={r.wall_time_s:6.2f}s  '
                        f'rel_err={r.verify_max_rel_err:.3e}  {r.status}',
                        f,
                    )
                    rows.append({
                        'file': label, 'reaction': rname,
                        **r.__dict__,
                    })

        log('=' * 70, f)
        log('SUMMARY', f)
        log('=' * 70, f)
        log(
            f'{"file":22s} {"reaction":10s} {"strategy":22s} '
            f'{"seed":>6} {"final":>6} {"iter":>4} {"wall":>7} {"rel_err":>10}',
            f,
        )
        for row in rows:
            log(
                f'{row["file"]:22s} {row["reaction"]:10s} '
                f'{row["strategy"]:22s} '
                f'{row["seed_size"]:6d} {row["final_size"]:6d} '
                f'{row["iterations"]:4d} '
                f'{row["wall_time_s"]:6.2f}s '
                f'{row["verify_max_rel_err"]:.3e}',
                f,
            )
        log('benchmark complete', f)


if __name__ == '__main__':
    main()
