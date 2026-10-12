"""Per-(material, reaction) comparison plots. Two panels each:
top shows the composed cross section (ground truth) with both
strategies' adaptive meshes overlaid as markers; bottom shows the
relative linear-interpolation error of each strategy against the
ground truth.
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
LOG = os.path.join(OUT_DIR, 'plot_strategy_comparison.log')

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
    ('elastic', '(n,n_0)'),
    ('capture', '(n,g)'),
    ('fission', '(n,fission)'),
]
STRATEGIES = [
    ('windowed_fixed',     'tab:blue',    'o'),
    ('territory_adaptive', 'tab:orange',  '^'),
]
E_RANGE = (1e-5, 2000.0)
TOL_REL = 1e-3
VERIFY_N = 50_000


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
        e_check = np.geomspace(E_RANGE[0], E_RANGE[1], VERIFY_N)

        for isotope, library, path in TARGETS:
            if not os.path.exists(path):
                log(f'SKIP {isotope}: file not found', f)
                continue
            log(f'=== parsing {isotope} ({library}) ===', f)
            d = EndfParserCpp().parsefile(path)

            for rname, rstr in REACTIONS:
                log(f'  {isotope} {rname} ({rstr})', f)
                # Ground truth on the dense verification mesh.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    xs_truth = np.asarray(
                        get_reaction_xs(d, rstr, e_check, options=options),
                        dtype=float,
                    )

                # Run both strategies, interp to verification mesh,
                # compute relative error.
                strat_data = []
                for strat_name, color, marker in STRATEGIES:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            res = linearize_reaction_xs(
                                d, rstr, E_RANGE[0], E_RANGE[1],
                                tol_rel=TOL_REL, options=options,
                                seed_strategy=strat_name,
                                return_diagnostics=True,
                            )
                    except Exception as exc:
                        log(
                            f'    {strat_name} FAILED '
                            f'({type(exc).__name__}: {exc})',
                            f,
                        )
                        continue
                    xs_interp = np.interp(e_check, res.mesh, res.sigma)
                    denom = np.maximum(np.abs(xs_truth), 1e-30)
                    rel_err = np.abs(xs_truth - xs_interp) / denom
                    strat_data.append({
                        'name': strat_name, 'color': color, 'marker': marker,
                        'res': res, 'rel_err': rel_err,
                    })
                    log(
                        f'    {strat_name:22s} mesh={res.mesh.size:6d}  '
                        f'iter={res.iterations:3d}  '
                        f'max_rel_err={float(rel_err.max()):.3e}  '
                        f'{res.status}',
                        f,
                    )

                if not strat_data:
                    continue

                # ---- Figure: two stacked panels sharing the x axis. ----
                fig, (ax_top, ax_err) = plt.subplots(
                    2, 1, figsize=(11, 7),
                    gridspec_kw={'height_ratios': [3, 1.5], 'hspace': 0.05},
                    sharex=True,
                )
                # Top: ground truth + both strategies' knots.
                ax_top.loglog(
                    e_check, xs_truth, '-', color='black', lw=0.9,
                    alpha=0.75, label='ground truth (50k log-spaced)',
                )
                for sd in strat_data:
                    r = sd['res']
                    ax_top.loglog(
                        r.mesh, r.sigma, sd['marker'],
                        color=sd['color'], ms=2.5, alpha=0.55,
                        label=(
                            f'{sd["name"]}  '
                            f'({r.mesh.size} pts, {r.iterations} it, {r.status})'
                        ),
                    )
                ax_top.set_ylabel('Cross section (barn)')
                ax_top.set_title(
                    f'{isotope} ({library}): {rname}  reaction={rstr}  '
                    f'[{E_RANGE[0]:g}, {E_RANGE[1]:g}] eV  '
                    f'tol_rel={TOL_REL}  numba'
                )
                ax_top.grid(True, which='both', alpha=0.3)
                ax_top.legend(loc='best', fontsize=8)

                # Bottom: relative error vs E for each strategy.
                for sd in strat_data:
                    ax_err.loglog(
                        e_check, sd['rel_err'],
                        '-', color=sd['color'], lw=0.9, alpha=0.85,
                        label=f'{sd["name"]}  '
                              f'(max {float(sd["rel_err"].max()):.2e})',
                    )
                ax_err.axhline(
                    TOL_REL, color='gray', ls='--', lw=0.8,
                    label=f'tol_rel = {TOL_REL:g}',
                )
                ax_err.set_xlabel('Incident neutron energy (eV)')
                ax_err.set_ylabel('relative error  |σ − interp| / |σ|')
                ax_err.grid(True, which='both', alpha=0.3)
                ax_err.legend(loc='best', fontsize=8)
                # Clip the error axis range so sub-eps "zeros" at the
                # mesh knots don't blow up the plot.
                ax_err.set_ylim(1e-10, 1.0)

                fig.tight_layout()
                out = os.path.join(
                    OUT_DIR,
                    f'cmp_{isotope.replace("-", "")}_{rname}_{library}.png',
                )
                fig.savefig(out, dpi=140)
                plt.close(fig)
                log(f'    saved {out}', f)

        log('comparison plots complete', f)


if __name__ == '__main__':
    main()
