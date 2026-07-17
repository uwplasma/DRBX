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
  the perpendicular-Laplacian GMRES now runs on `solvax` (the earlier
  `lineax` backend and its optional extra were removed).
- The Alfven-wave benchmark input is a committed fixture instead of a
  machine-specific absolute path.
- Phase 1 slice 1: the mocked reference-report generators, orphan campaign
  modules, meeting demos, profiler audits, and one-off diagnose scripts were
  removed from the package (−10.4k lines).
- Incorporated the FCI operator/sharding stack from
  [PR #3](https://github.com/uwplasma/jax_drb/pull/3) by Aiken Xie:
  cell-centered FCI geometry with shard/halo layouts, consistent
  finite-volume operators, halo exchange, 2-field/4-field/electromagnetic
  models, an RK4 integrator, and slab/shifted-torus MMS, operator,
  domain-decomposition, halo, and multigrid verification suites. See the
  README section "Incorporated FCI/Sharding Stack (from PR #3)" for the
  exact scope.

## Validation

The v2 validation program is the benchmark ladder B1-B10 in
`plan_jax_drb.md`. In this dev series the previously shipping gates remain
in force (operator kernels vs scalar references, MMS convergence order,
Alfven-wave phase-speed check); the ladder rungs land phase by phase and are
recorded here as they do.

## Current Boundary

Differentiability claims apply only to pure-JAX paths with derivative tests;
host-side SciPy paths are labeled as such.

The 2.0.0 release ships at the end of the plan's Phase 8 with the full
native capability matrix, the literature-anchored benchmark ladder, the
closed/open tokamak/stellarator example matrix, SOLVAX-backed solvers, and
strong-scaling evidence.
