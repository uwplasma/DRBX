# Release Notes: 2.0.0

The v2.0.0 release of the research-grade program tracked in the project
planning notes. This is the first stable release under the **DRBX** name
(the project was previously developed as `jax_drb`).

Headline capabilities in 2.0.0:

- end-to-end differentiable drift-reduced Braginskii turbulence on closed and
  open field lines, in tokamak and stellarator (FCI) geometry;
- Hasegawa-Wakatani drift-wave flagship with gradient-based optimization
  through saturated turbulence; linear dispersion solver (drift-wave,
  shear-Alfven, interchange) verified to machine precision;
- rotating-ellipse and island-divertor stellarators, plus imported ESSOS coil
  and VMEC (VMEX) equilibria, including four-field turbulence on the
  Landreman-Paul configuration with a sheath-drained scrape-off layer;
- hermes-3 neutral model (packaged AMJUEL rates) with a self-consistent
  detaching SOL (SD1D rollover);
- structured solves via `solvax` (the perpendicular-Laplacian GMRES potential
  inversion, tridiagonal, Fourier-Helmholtz); the sync-free RHS and solvax
  GMRES roughly halved the single-CPU four-field step;
- multi-device `shard_map` FCI stepping, bit-exact against single-device
  execution.

Landed (Phase 0):

- CI runs the full fast test suite on Python 3.10-3.12 instead of a
  seven-file slice.
- Coverage is a single whole-package branch-coverage number over
  `src/drbx` (baseline 86%); the curated closeout/promoted coverage
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
  [PR #3](https://github.com/uwplasma/drbx/pull/3) by Aiken Xie:
  cell-centered FCI geometry with shard/halo layouts, consistent
  finite-volume operators, halo exchange, 2-field/4-field/electromagnetic
  models, an RK4 integrator, and slab/shifted-torus MMS, operator,
  domain-decomposition, halo, and multigrid verification suites. See the
  README section "Incorporated FCI/Sharding Stack (from PR #3)" for the
  exact scope.

## Validation

The v2 validation program is the benchmark ladder B1-B10 tracked in the
project planning notes. In this dev series the previously shipping gates remain
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
