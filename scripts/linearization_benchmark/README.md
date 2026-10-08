# linearization_benchmark

Benchmark + plotting scripts for the mesh-densification prototype in
`endf_userpy.linearization` and the parallel curvature-equidistribution
driver in `endf_userpy.linearization_curvature`.

Shelved alongside the prototype itself; see
`docs/notes_mesh_densification.md` for the design findings and the
list of known defects (issues #312, #313) that need to land before
the output here can be trusted at the stated tolerance.

## Scripts

- `benchmark_actinides.py` — runs `benchmark_seed_strategies` on the
  four actinides available in `tests/data_law1_adhoc/` (U-233, U-235,
  U-238, Pu-239) across elastic / capture / fission at
  `tol_rel = 1e-3`, numba backend. Writes a live-tailable log.
- `plot_actinides.py` — one log-log plot per material with elastic /
  capture / fission on shared axes, `territory_adaptive` mesh knots
  overlaid against a dense ground-truth curve.
- `plot_strategy_comparison.py` — two-panel plot per
  (material, reaction): top = XS with both strategies' meshes;
  bottom = relative error vs ground truth with `tol_rel` as a
  reference line.

## Usage

```
# Fetch the corpus files once if not already done.
bash tests/data_law1_adhoc/fetch.sh

# Numba backend is used by all three scripts.
pip install numba

# From anywhere; scripts self-locate the repo via __file__.
python scripts/linearization_benchmark/benchmark_actinides.py
python scripts/linearization_benchmark/plot_actinides.py
python scripts/linearization_benchmark/plot_strategy_comparison.py
```

Plots and logs land next to the script by default; set
`PLOT_OUT_DIR` (or `BENCH_LOG` for the benchmark log) to redirect.
