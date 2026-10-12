"""Log-log plots per material: elastic / capture / fission cross
sections on a shared axis.

Uses the composed `get_reaction_xs` on a dense log-spaced mesh as
ground truth. Overlays the adaptive mesh from
`linearize_reaction_xs(seed_strategy='territory_adaptive')` as
scatter markers so the knot density is visible against the
underlying curve.
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT_DIR = os.environ.get('PLOT_OUT_DIR', os.path.dirname(__file__))
LOG = os.path.join(OUT_DIR, 'plot_actinides.log')

from endf_parserpy import EndfParserCpp  # noqa: E402

from endf_userpy.linearization import linearize_reaction_xs  # noqa: E402
from endf_userpy.primitives import array_ns  # noqa: E402
from endf_userpy.quantities import get_reaction_xs  # noqa: E402
from endf_userpy.run_options import RunOptions  # noqa: E402


TARGETS = [
    ('U-233', 'endfb81', 'tests/data_law1_adhoc/endfb81_n_U-233.endf'),
    ('U-235', 'tendl21', 'tests/data_law1_adhoc/tendl21_n_U-235.endf'),
    ('U-238', 'jendl5',  'tests/data_law1_adhoc/jendl5_n_U-238.endf'),
    ('Pu-239', 'endfb81', 'tests/data_law1_adhoc/endfb81_n_Pu-239.endf'),
]
REACTIONS = [
    ('elastic', '(n,n_0)', 'tab:blue'),
    ('capture', '(n,g)',   'tab:orange'),
    ('fission', '(n,fission)', 'tab:green'),
]
E_RANGE = (1e-5, 2000.0)   # extended lower bound: thermal + 1/v
TOL_REL = 1e-3


def log(msg, f):
    stamp = datetime.now().strftime('%H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    f.write(line + '\n')
    f.flush()


def main():
    with open(LOG, 'w') as f:
        log(f'plot log: {LOG}', f)
        xp = array_ns.get_backend('numba')
        options = RunOptions(backend=xp)

        # Shared verification mesh (dense, log-spaced) for the smooth
        # ground-truth curves.
        e_check = np.geomspace(E_RANGE[0], E_RANGE[1], 20_000)

        for isotope, library, path in TARGETS:
            log(f'=== {isotope} ({library}) ===', f)
            if not os.path.exists(path):
                log('  SKIP: file not found', f)
                continue
            d = EndfParserCpp().parsefile(path)

            fig, ax = plt.subplots(figsize=(10, 6))
            for rname, rstr, color in REACTIONS:
                log(f'  {rname} ({rstr})', f)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        xs_truth = np.asarray(
                            get_reaction_xs(
                                d, rstr, e_check, options=options,
                            ),
                            dtype=float,
                        )
                        res = linearize_reaction_xs(
                            d, rstr, E_RANGE[0], E_RANGE[1],
                            tol_rel=TOL_REL, options=options,
                            seed_strategy='territory_adaptive',
                            return_diagnostics=True,
                        )
                except Exception as exc:
                    log(f'    FAILED ({type(exc).__name__}: {exc})', f)
                    continue

                # Ground-truth curve (log-log).
                ax.loglog(
                    e_check, xs_truth, '-', color=color, lw=1.0,
                    alpha=0.85,
                    label=f'{rname} (truth, 20k pts)',
                )
                # Adaptive knots overlaid as markers so sampling density
                # against the resonance structure is visible.
                ax.loglog(
                    res.mesh, res.sigma, '.', color=color, ms=2.0,
                    alpha=0.5,
                    label=(
                        f'{rname} adaptive  '
                        f'({res.mesh.size} pts, {res.iterations} it, '
                        f'{res.status})'
                    ),
                )
                log(
                    f'    truth min/max = {xs_truth.min():.3e} / '
                    f'{xs_truth.max():.3e} barn, adaptive '
                    f'mesh size = {res.mesh.size}, iter = '
                    f'{res.iterations}, status = {res.status}',
                    f,
                )

            ax.set_xlabel('Incident neutron energy (eV)')
            ax.set_ylabel('Cross section (barn)')
            ax.set_title(
                f'{isotope} ({library}): elastic / capture / fission '
                f'on {E_RANGE[0]:g} to {E_RANGE[1]:g} eV\n'
                f'(adaptive mesh via territory_adaptive seed, '
                f'tol_rel = {TOL_REL}, numba backend)'
            )
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(loc='best', fontsize=8)
            fig.tight_layout()
            out = os.path.join(OUT_DIR, f'xs_{isotope.replace("-", "")}_{library}.png')
            fig.savefig(out, dpi=140)
            plt.close(fig)
            log(f'  saved {out}', f)

        log('plot script complete', f)


if __name__ == '__main__':
    main()
