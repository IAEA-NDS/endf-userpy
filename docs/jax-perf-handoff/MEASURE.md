# Measurement harness

Benchmarks + correctness checks the agent runs before and after every
change. The three recipes below should fit in ~5-10 minutes total on
the standard reproducer.

## Data file

`tests/data_law1_adhoc/endfb81_n_U-235.endf` (ENDF/B-VIII.1 U-235, MAT
9228), pinned in `tests/data_law1_adhoc/fetch.sh`:

```bash
bash tests/data_law1_adhoc/fetch.sh     # or set U235_ENDFB81_ENDF
```

The recipes below still read `tests/data_temp/n_092-U-235_9228.dat`
(same file, byte-identical); point them at either path. If the IAEA
mirror is behind Cloudflare for your sandbox, ship the ~14 MB zip
alongside the handoff folder. Any Reich-Moore (LRU=1 LRF=3) actinide
file reproduces the architectural numbers.

## Methodology: alternate, and keep a control column

Measured on a shared host, sequential "main then branch" runs swung by
up to +70 % on the *unchanged* numpy path. Only trust comparisons that

- run main and the branch in **separate processes, alternating**
  (main, branch, main, branch, ...; 3 rounds, report medians), from a
  detached `main` worktree next to the branch checkout (symlink the
  gitignored corpus into it);
- carry the **numpy backend as a control column**: if numpy moves, the
  host moved, not the code;
- report **first call and steady state** separately (jax first calls
  are dominated by per-shape compilation).

## Recipe 1: `ppxs_scan` - end-to-end wall + correctness, all backends

```python
"""Scan backends x mesh sizes for get_particle_production_xs."""
import os, sys, time, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, '.')  # from repo root

import numpy as np
from endf_parserpy import EndfParserFactory
from endf_userpy.quantities import get_particle_production_xs
from endf_userpy.run_options import RunOptions

d = EndfParserFactory.create().parsefile(
    'tests/data_temp/n_092-U-235_9228.dat'
)

sizes = [1_000, 10_000, 100_000, 1_000_000]
# Prime once with small input so numba / jax compile at a tiny shape.
e_warm = np.linspace(1e-5, 1e7, 100)
for b in ('numpy', 'numba'):
    _ = get_particle_production_xs(
        d, '(n,total)', 'n', e_warm, options=RunOptions(backend=b),
    )

print(f"{'N':>10} | {'numpy':>8} | {'numba':>8} | "
      f"{'jax 1st':>10} | {'jax ss':>8} | {'jax rel':>10}")
print('-' * 72)
for N in sizes:
    e = np.linspace(1e-5, 1e7, N)
    t0 = time.monotonic()
    xs_np = get_particle_production_xs(
        d, '(n,total)', 'n', e, options=RunOptions(backend='numpy'))
    t_np = time.monotonic() - t0
    t0 = time.monotonic()
    xs_nb = get_particle_production_xs(
        d, '(n,total)', 'n', e, options=RunOptions(backend='numba'))
    t_nb = time.monotonic() - t0
    try:
        t0 = time.monotonic()
        xs_jax = get_particle_production_xs(
            d, '(n,total)', 'n', e, options=RunOptions(backend='jax'))
        xs_jax = np.asarray(xs_jax)
        t_first = time.monotonic() - t0
        t0 = time.monotonic()
        xs_jax2 = get_particle_production_xs(
            d, '(n,total)', 'n', e, options=RunOptions(backend='jax'))
        np.asarray(xs_jax2)
        t_ss = time.monotonic() - t0
        rel = float(np.max(np.abs(xs_jax - np.asarray(xs_nb))
                           / np.maximum(np.abs(np.asarray(xs_nb)), 1e-30)))
        print(f'{N:>10} | {t_np:>8.3f} | {t_nb:>8.3f} | '
              f'{t_first:>10.3f} | {t_ss:>8.3f} | {rel:>10.2e}')
    except Exception as ex:
        print(f'{N:>10} | {t_np:>8.3f} | {t_nb:>8.3f} | '
              f'{"FAIL":>10} | {"FAIL":>8} | {type(ex).__name__}')
```

**Pass criteria for every row**: `jax rel <= 1e-10`.

**Target**: `jax ss` strictly less than the previous-attempt value;
`numpy` and `numba` within +/- 5 percent of the handoff baseline in
`FINDINGS.md`.

## Recipe 2: `reconstruct_only` - kernel-level isolation

```python
"""Isolate Reich-Moore reconstruct wall time per backend."""
import warnings, time, numpy as np, jax
warnings.simplefilter('ignore')
from endf_parserpy import EndfParserFactory
from endf_userpy.mfsec_interpretation.mf2_interpretation_reichmoore_preproc import (
    rm_data_from_endf_dict,
)
from endf_userpy.mfsec_interpretation import mf2_interpretation_reichmoore as rm
from endf_userpy.primitives import array_ns

d = EndfParserFactory.create().parsefile(
    'tests/data_temp/n_092-U-235_9228.dat'
)
e = np.linspace(1e-5, 2e3, 10_000)  # 10 K points inside RRR

def _timed(fn):
    t0 = time.monotonic()
    r = fn()
    # Force materialisation
    if isinstance(r, dict):
        for v in r.values():
            np.asarray(v)
    return time.monotonic() - t0

for name in ('numba', 'numpy', 'jax'):
    xp = array_ns.get_backend(name)
    data = rm_data_from_endf_dict(d, xp=xp)
    _ = rm.reconstruct(data, e[:100], xp)  # warm
    best = min(
        _timed(lambda: rm.reconstruct(data, e, xp)) for _ in range(3)
    )
    print(f'  {name:7} reconstruct @10K: {best:.3f} s')
```

**Target**: `jax` within 2x of `numba` on the reconstruct body.
Current state: 24x.

## Recipe 3: autodiff smoke

```bash
# Must still pass after any change.
pytest tests/test_mf2_interpretation_mlbw.py \
       tests/test_mf2_interpretation_reichmoore.py \
       tests/test_jit_composition.py \
       tests/test_mf2_resonance_autodiff_wrt_E.py \
       tests/test_mf2_rml_autodiff.py \
       tests/test_mf2_urr_autodiff.py \
       tests/test_tab1_numba.py \
       tests/test_tab1_interp_discontinuity.py \
       --tb=short -q
```

Known pre-existing failures on main at `333eb62` (not caused by
handoff work; do not fix in this arc; status after 2026-10-09: not
reproduced in the sandbox with the corpus present, re-check on CI):

- `tests/test_resonance_composition_rml.py::test_reconstruct_resonance_xs_pu239_fission`
- `tests/test_resonance_composition_rml.py::test_reconstruct_resonance_xs_lrf7_seam_is_zero_above_eh`

Explicitly stash with `git stash push -- <file>` and re-run the
handoff tests to confirm your change is responsible for any new
failure, not something pre-existing.

## Recipe 4: memory probe (optional, for H3)

```python
"""Approximate peak RSS during a jax.grad backward pass."""
import jax, jax.numpy as jnp, numpy as np, os, resource
# ... setup data as in Recipe 2 ...

def loss(params):
    # params = data.res_er, say
    data_pert = dataclasses.replace(data, res_er=params)
    out = rm.reconstruct(data_pert, e, xp_jax)
    return jnp.sum(out['tot'])

grad = jax.grad(loss)
_ = grad(data.res_er)  # trace + run
peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f'peak RSS: {peak_kb / 1024:.1f} MB')
```

Compare before (unchunked / un-checkpointed) and after any
scan+checkpoint change. Should drop proportionally with the chunk
count (chunked_chi2 saw 11x reduction).

## Ruff gate

```bash
ruff check --output-format github .     # what CI runs
# or, tracked files only (needs --force-exclude for pyproject excludes):
git ls-files '*.py' | xargs ruff check --force-exclude --output-format=github
```

Must be clean before any merge.

## Bug-catching verification (CLAUDE.md convention)

For every regression test added to pin a behaviour change:

```bash
# Stage the fix only.
git stash push -- <fix-file>.py
pytest <new-test>.py --tb=no -q    # confirm it fails without the fix
git stash pop
pytest <new-test>.py --tb=no -q    # confirm it passes with the fix
```
