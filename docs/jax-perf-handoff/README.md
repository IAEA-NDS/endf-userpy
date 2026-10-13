# JAX performance / memory handoff for `endf-userpy`

This folder is a long-running-agent brief for improving the JAX
backend's throughput and memory footprint on the resonance
reconstruction and MF3 interpolation hot paths.

## Order to read

1. **`MISSION.md`** - what you are trying to achieve, scope, acceptance
   criteria, and what is explicitly out of scope.
2. **`FINDINGS.md`** - measured baselines, the architectural map that
   produces the current numbers, and prior work in the repo that is
   already relevant.
3. **`PLAYBOOK.md`** - ranked hypotheses with concrete file pointers,
   expected wins, risks, and templates to copy from.
4. **`MEASURE.md`** - the exact benchmark + correctness harness. Run it
   before you change anything (baseline) and after every change.
5. **`PROGRESS.md`** - running log. Append one dated entry per attempt
   with: what you tried, measurement, decision. This is what gives the
   next agent (or the human) a timeline when they pick this up.
6. **`SEED_PROMPT.md`** - the kickoff message. Not meant to be read
   directly by the agent; it is the text the human pastes to start a
   session.

## Working principles

- Measure before and after every change. Never ship a change you have
  not benchmarked.
- Preserve correctness: bit-exact or within `1e-10` relative to the
  numba reference on the standard reproducer. The numba path is
  authoritative.
- Preserve `jax.jit` / `jax.grad` compatibility. Several tests pin
  these; see `MISSION.md` for the specific test file list.
- Do not change the public API of `endf_userpy.quantities`.
- Keep numpy + numba paths untouched or faster. If a change benefits
  JAX at the cost of numpy / numba, discuss in `PROGRESS.md` before
  merging.
- **Architectural respect**: every new function goes in its correct
  layer (primitives / mfsec_interpretation / quantities_mt_zap /
  quantities) with a name and signature that match the local module's
  conventions. Before writing a new function, run the four-step
  placement checklist in `MISSION.md::Architectural respect`. A
  mis-placed function is a regression to fix before merge, not a
  style preference. If a reviewer or user asks "is this the right
  place for it", the answer should already be yes.

## File layout

```
docs/jax-perf-handoff/
  README.md       <- this file
  MISSION.md      <- goal + scope + acceptance
  FINDINGS.md     <- baselines, architecture, prior art
  PLAYBOOK.md     <- ranked ideas to try
  MEASURE.md      <- benchmark harness
  PROGRESS.md     <- append-only agent log
  SEED_PROMPT.md  <- kickoff text (for the human to paste)
```
