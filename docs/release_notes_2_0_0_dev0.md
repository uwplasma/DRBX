# Release Notes: 2.0.0.dev0

Development series for the v2 research-grade program defined in
[`plan_jax_drb.md`](https://github.com/uwplasma/jax_drb/blob/main/plan_jax_drb.md).

Landed so far (Phase 0):

- CI runs the full fast test suite on Python 3.10-3.12 instead of a
  seven-file slice.
- Coverage is a single whole-package branch-coverage number over
  `src/jax_drb` (baseline 86%); the curated closeout/promoted coverage
  gates were removed.
- Dependency metadata corrected: unused `diffrax` and `equinox` removed,
  `lineax` exposed as an optional extra.
- The Alfven-wave benchmark input is a committed fixture instead of a
  machine-specific absolute path.

## Validation

The v2 validation program is the benchmark ladder B1-B10 in
`plan_jax_drb.md`. In this dev series the previously shipping gates remain
in force (operator kernels vs scalar references, MMS convergence order,
golden-array parity for the drift-wave/neutral-mixed/recycling/tokamak
families, Alfven-wave phase-speed check); the ladder rungs land phase by
phase and are recorded here as they do.

## Current Boundary

The stable full output-window recycling BDF default remains in force for the
heavy recycling lanes; the JAX-linearized and JVP paths stay opt-in research
lanes until they match it at the same fidelity and cost (re-evaluated at the
plan's Phase 5 exit). Differentiability claims apply only to pure-JAX paths
with derivative tests; host-side SciPy paths are labeled as such.

The 2.0.0 release ships at the end of the plan's Phase 8 with the full
hermes-3 capability matrix, the literature-anchored benchmark ladder, the
closed/open tokamak/stellarator example matrix, SOLVAX-backed solvers, and
strong-scaling evidence.
