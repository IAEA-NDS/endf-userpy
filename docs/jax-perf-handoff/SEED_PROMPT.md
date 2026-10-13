# Seed prompt

Paste the block below (or adapt it) as the first message to the
sandboxed agent. The goal is to point it at the handoff files and
give it the operating rules once; the markdown files carry the
substance.

---

```
You are continuing an in-flight performance / memory optimisation
effort on the `endf-userpy` repo (branch: main). Your target is the
JAX backend's throughput and memory on resonance reconstruction +
MF3 interpolation, while preserving `jax.jit` + `jax.grad`
compatibility.

Everything you need is in `docs/jax-perf-handoff/`. Read in order:

1. `docs/jax-perf-handoff/README.md`
2. `docs/jax-perf-handoff/MISSION.md`
3. `docs/jax-perf-handoff/FINDINGS.md`
4. `docs/jax-perf-handoff/PLAYBOOK.md`
5. `docs/jax-perf-handoff/MEASURE.md`

Then look at `docs/jax-perf-handoff/PROGRESS.md` for what has already
been tried.

Workflow:

- Run the baseline from `MEASURE.md::Recipe 1` and
  `MEASURE.md::Recipe 2`. Confirm your numbers are in the ballpark
  of the `FINDINGS.md` table. If they diverge by more than 50%,
  stop and investigate the environment (CPU model, numba thread
  count, jax version, available RAM).
- Pick the top unaddressed item in `PLAYBOOK.md`. If H1 is still
  open, start there.
- Branch: `perf/jax-<slug>` from `main`.
- Attempt the change; measure with the recipe named in the playbook
  entry; log the result in `PROGRESS.md` whether it worked or not.
- Only when the acceptance criteria in `MISSION.md` are fully met,
  push the branch and open a PR. The PR body should link to the
  `PROGRESS.md` entry.
- Never merge PRs on your own; leave them open for the human to
  review.
- If an attempt fails, don't delete the exploration — commit a short
  write-up to `PROGRESS.md` of why it failed so the next attempt
  avoids the same path.

Hard constraints:

- No regressions on numpy / numba paths (within +/- 5% wall time).
- No change to the public API of `endf_userpy.quantities`.
- `jax.jit(reconstruct)` and `jax.grad(reconstruct)` must still work
  for every formalism (MLBW / Reich-Moore / RML / URR). The tests
  listed in `MISSION.md::acceptance` must all pass.
- Measure every change; include before/after numbers in `PROGRESS.md`.
- Architectural respect: every new function goes in its correct
  layer and matches the local module's naming + signature
  conventions. Before adding a function anywhere, run the four-step
  placement checklist in `MISSION.md::Architectural respect` (which
  module, which layer, matching name, matching signature). If a
  function sits in the wrong module or carries a leading underscore
  for something that is a public selector, that is a regression to
  be fixed before merge, not a style preference. Read the root
  `CLAUDE.md` for the three-layer architecture if any placement
  decision feels unclear.

Start now by running Recipe 1 to confirm the environment, then
report the measured baseline table before touching any code.
```

---

## Operational notes for the human running the sandbox

- **Repo copy**: give the agent a fresh clone, not a shared working
  tree. It will create branches freely.
- **Network**: U-235 ENDF file needs to be either pre-shipped in
  `tests/data_temp/` or re-fetchable. The IAEA mirror is behind
  Cloudflare in some regions; shipping the file is more reliable.
- **GPU**: not required. The baselines were measured on CPU-only
  JAX. If a GPU is available, numbers will shift; the architectural
  conclusions still hold.
- **Memory**: baseline sandbox was 64 GB RAM. 1 M-point calls
  pre-chunking need ~55 GB. For a tighter machine, start the agent
  with the H5 prototype branch merged (see `FINDINGS.md`).
- **Session cadence**: the agent is designed to run autonomously
  across many iterations. Point it back at the folder each session;
  it should read `PROGRESS.md` to pick up where the previous
  session stopped.
- **Human review loop**: PRs are opened by the agent but merged by
  you. The `PROGRESS.md` entries are the audit trail for the
  decisions behind each change.

## Alternative: direct CLAUDE.md hook

If the agent's startup auto-reads `CLAUDE.md` in the working
directory, you can save pasting the prompt by placing a short
`CLAUDE.md` at the repo root (or in `docs/jax-perf-handoff/`
depending on which the agent reads first) that just says:

```
See docs/jax-perf-handoff/README.md for the active task and
operating rules.
```

The seed prompt above still works; the CLAUDE.md hook is a
convenience for recurring sessions.
