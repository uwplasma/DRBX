# Release Notes

## 2.2.0

Since 2.1.0 the HSX FCI Braginskii backend is coupled to the wall, neutrals and
the magnetic field, differentiable end to end, and faster per step.

- **Sheath wall closure**: cell-by-cell sheath current (`sheath_potential = "current"`)
  with a wall energy ledger; 32^3 HSX runs from t = 0 to 0.6.
- **Fluid neutrals** (AMJUEL rates, momentum, kinetic correction) on SOLVAX flat FGMRES;
  the 1D leg agrees with the SD1D scan within 0.34%.
- **Electromagnetic four-field closure** (`[physics] electromagnetic`, `beta`): A_par,
  flutter and electron inertia in the four-field model and the HSX backend. Shear-Alfven
  and drift-Alfven frequencies match analytic within 2.5% and 0.5%; the energy ledger closes.
- **Reverse-mode gradients** through the GMRES, potential and Ampere solves
  (implicit-function VJPs); driver gradients with plasma and neutrals agree with
  central differences.
- **Performance**: parallel operator pairs and the potential solver's -L_perp as
  sparse rows assembled once per run; inner tolerance tied to the step; one fewer
  potential solve per step; device-side GPU loops. CPU 16^3 step 0.17 -> 0.07 s,
  32^3 1.16 -> 0.57 s; GPU about -18%.
- **Leaner package**: pillow dropped from the base dependencies, five stellarator
  campaign modules removed; requires SOLVAX >= 0.36.1.
- README validation figure (operators, waves, neutrals, wall ledger, gradients).

**New: every model runs from a TOML deck** (`[model] backend = "<name>"`).
Hasegawa-Wakatani (with an autodiff gradient), the open-field-line SOL flux tube
with sheath/recycling accounting, the recycling SOL, SD1D detachment, four-field
FCI turbulence on rotating-ellipse, shifted-torus and island-divertor geometry,
and the differentiable FCI case join the HSX Braginskii deck. Tables are the
Python configuration objects (one set of defaults), unknown keys are errors, and
`docs/decks.md` documents every key and output, checked against the code by a
test. `gmres_preconditioner` (`line-u` or `none`) is now an explicit HSX config
field, deck key and script option.

## 2.1.0

**New: experimental HSX FCI Braginskii backend** (`drbx.fci_braginskii`, #6).
A seven-field electrostatic Braginskii model on stellarator-following FCI
geometry for the HSX scrape-off layer, with an implicit current/potential stage,
owner-cell agglomeration at the wall, sharded GMRES, a geometry builder, and
`drbx run <deck>.toml` input decks. Wall-limited: no sheath, neutral or
recycling coupling yet. Verified: stage continuity of the returned state
~1e-16, bitwise restart, equal GMRES counts on 1/2/4 devices, normalization
tests, real-geometry tests. Measured temporal order: ~2 for vorticity, 1.4-1.6
for n, T, Vi, ~1 for phi and Ve (stiff electrons at the physical mass ratio).

**New: SD1D-matched 1D divertor model** (#18). Steady-state Newton solve of the
SD1D equations; against the published SD1D hydrogen scan the target
temperature agrees within 0.35% and the target flux within 0.15% (800 cells);
particle/power ledgers close; implicit-function gradients of target quantities.
Replaces the previous reduced detachment model, which was not grid-converged.

**Physics fixes**
- Neutral reaction sources conserve thermal plus kinetic energy (2/3
  energy-to-pressure conversion, receiver mixing heat) (#8).
- 1D conduction is driven by the temperature gradient; CX damping uses K_cx n_n (#8).
- Implicit current/potential stage uses one owner-level gradient in the solve
  and in Ve, consistent under agglomeration (#13).

**Solvers and performance**
- GMRES keeps the requested restart length (it was silently reduced to
  gcd(restart, maxiter)); strict vs relaxed acceptance reported (#10, #12).
- GPU: small batched eigen-solves no longer serialize on the device; an HSX
  step on an RTX A4000 went from ~555 s to ~35 s (#33).

**Data and provenance**: hydrogen AMJUEL fits pinned to the January 13, 2020
report (#16).

**Footprint**: duplicated helpers, pytree boilerplate and duplicated backend
modules removed (#9, #17, #22-#26, #30-#32), test-only plotting code removed;
the README follows the feature-section layout with a sourced comparison table (#15).

## 2.0.0

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
- hermes-3 neutral model (packaged AMJUEL rates) and an SD1D-matched 1D
  plasma-neutral SOL (matches SD1D's 13.6 eV case; no flux rollover without
  impurity radiation);
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

### Validation

The v2 validation program is the benchmark ladder B1-B10 tracked in the
project planning notes. In this dev series the previously shipping gates remain
in force (operator kernels vs scalar references, MMS convergence order,
Alfven-wave phase-speed check); the ladder rungs land phase by phase and are
recorded here as they do.

### Current Boundary

Differentiability claims apply only to pure-JAX paths with derivative tests;
host-side SciPy paths are labeled as such.

The 2.0.0 release ships at the end of the plan's Phase 8 with the full
native capability matrix, the literature-anchored benchmark ladder, the
closed/open tokamak/stellarator example matrix, SOLVAX-backed solvers, and
strong-scaling evidence.
