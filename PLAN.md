# jax_drb Plan

Date: 2026-03-11

## 1. Mission

Build `jax_drb` as a standalone JAX edge/SOL plasma code with a private parity track against an external reference implementation:

- same physics model and component ordering semantics during the parity buildout,
- same normalization, boundary, diagnostic, and restart conventions where practical,
- runnable from both a Python API and a native `jax_drb` CLI,
- CPU/GPU portable with a pure-JAX runtime path,
- end-to-end differentiable through production solver paths,
- minimal runtime dependencies.

This repository has been reset for that purpose. All pre-existing contents were archived into `legacy/` on 2026-03-11. `legacy/` is reference material only; it is not the active implementation base.

## 2. Non-Negotiable Requirements

1. The private reference implementation is the source of truth for Stage 1 parity.
2. Public surfaces must remain native to `jax_drb`:
   - no source-branded file names, docs, CLI flags, environment variables, section names, variables, or images in the active tree,
   - public examples and user docs must present `jax_drb` as an independent code,
   - any source-compatibility bridge must remain internal to the parity tooling.
3. New runtime kernels must be written in JAX primitives only.
4. Array semantics, guard-cell behavior, and component execution order must match the reference before any model reformulation.
5. Every implementation step must land with:
   - unit tests,
   - at least one physics or regression test,
   - a differentiability check,
   - a CPU/GPU execution check.
6. Do not start with extensions such as FCI, conservative DRB, stellarator-first workflows, or surrogate/control tooling before the base parity path is stable.
7. `legacy/` is quarantine-only:
   - no active package code may import from `legacy/`,
   - no active tests may use `legacy/` as an oracle,
   - `legacy/` may only be cited as historical context while reference remains the numerical source of truth.

## 3. Sources of Truth

The plan is based on the current reference solver codebase, tests, examples, docs, and the literature in `/Users/rogerio/local/tests/drb_literature`.

Primary reference solver references:

- code: `local source checkout/src`
- headers: `local source checkout/include`
- docs: `local source checkout/docs/sphinx`
- unit tests: `local source checkout/tests/unit`
- MMS/operator tests: `local source checkout/tests/mms_operator`
- integrated tests: `local source checkout/tests/integrated`
- examples: `local source checkout/examples`

reference implementation facts that must be preserved early:

- `BOUT.inp`-driven runtime configuration with ordered `[model] components`
- species/type expansion through `ComponentScheduler`
- scheduler execution order is `transform()` for all components followed by `finally()` for all components
- normalizations from `Tnorm`, `Nnorm`, `Bnorm`, then `Cs0`, `Omega_ci`, `rho_s0`
- guard-cell and midpoint boundary semantics
- conservative finite-volume parallel operators and XPPM-style ExB transport
- diagnostics naming and metadata conventions
- restart/output conventions and field naming

Literature themes that inform the long-range roadmap:

- open-field-line SOL turbulence and validation: Ricci 2012, Halpern 2016, Giacomin 2022
- SOL regimes, density/beta limits, and width scalings: Mosetto 2012, Halpern 2013, Giacomin 2022, Lim 2023
- sheath and equilibrium potential boundary physics: Loizu 2012, Loizu 2013
- whole-volume core-edge-SOL geometry and arbitrary equilibria: Giacomin 2022
- stellarator/global 3D fluid extensions: Coelho 2024, Jorge 2021
- FCI/X-point numerics: Hariri 2013, Hariri 2014, Stegmeir 2018, Wiesenberger 2023
- conservative DRB reformulation: De Lucca 2026
- detachment/control-oriented reduced modeling: Body 2024

## 4. Active Scope and Out-of-Scope Work

Active Stage 1 scope:

- reproduce reference solver numerics and workflows in JAX,
- establish parity on 1D, 2D, and selected 3D tokamak cases,
- include core plasma, drifts, fields, neutrals, reactions, and diagnostics,
- provide a clean Python API and CLI for those workflows.

Explicitly out of scope until parity is achieved:

- redesigning the model equations,
- replacing reference field-aligned geometry with FCI,
- new physics not already in reference solver,
- optimization/inverse-design workflows beyond smoke tests,
- aggressive performance tuning that changes semantics.

## 5. Target Architecture

The clean JAX implementation should be built around the reference execution model, not around the old archived code.

Legacy quarantine rules:

- `legacy/` remains in the repository only as an archival snapshot.
- Nothing in the active `src/jax_drb/` tree may import, wrap, subclass, or progressively adapt code from `legacy/`.
- If a useful idea or fixture exists in `legacy/`, it must be re-derived from reference or recreated cleanly in the new tree.
- Any future deletion of `legacy/` must not change behavior of the active code, tests, docs, or benchmarks.

Target package layout:

- `src/jax_drb/config/`
  - `BOUT.inp` parser
  - typed runtime config
  - normalization and defaults
- `src/jax_drb/mesh/`
  - grid/metric loaders
  - guard-cell and region helpers
  - field-aligned communication helpers
- `src/jax_drb/state/`
  - immutable pytrees for fields, species state, diagnostics, restart state
- `src/jax_drb/components/`
  - one module per reference component or tightly related component family
- `src/jax_drb/operators/`
  - parallel FV operators
  - ExB operators
  - perpendicular diffusion/hyper-diffusion
  - elliptic operators and solves
- `src/jax_drb/reactions/`
  - reaction parser
  - ADAS/AMJUEL rate tables
  - source bookkeeping
- `src/jax_drb/io/`
  - reference/BOUT reference import
  - restart/output writing
  - diagnostics metadata
- `src/jax_drb/solver/`
  - explicit and IMEX/backward-Euler stepping
  - nonlinear/linear solve wrappers
- `src/jax_drb/cli.py`
  - `run`, `restart`, `compare`, `inspect`
- `tests/`
  - unit
  - mms
  - regression
  - physics
  - performance

Dependency policy:

- required runtime: `jax`, `jaxlib`
- allowed runtime if justified: `equinox`, `diffrax`
- dev/parity tools only: `pytest`, `numpy`, `netCDF4` or equivalent reader, lint/type tools

## 6. Cross-Cutting Infrastructure That Must Exist Early

These are mandatory before porting large physics blocks.

### 6.1 Parity Matrix

Create and maintain a file that maps every reference runtime component, operator family, diagnostic family, and relevant input option to:

- the new JAX module,
- the reference reference source file/function,
- the test that proves parity,
- the current status.

### 6.2 Reference Data Harness

Add tooling to ingest reference reference runs and expose:

- mesh/metric data,
- initial state,
- restart state,
- selected diagnostics,
- one-step RHS references where possible.

The preferred early parity workflow is:

1. run reference on a tiny case,
2. extract the state just before an RHS evaluation,
3. run JAX on the same state,
4. compare per-field outputs and diagnostics.

### 6.3 Low-Iteration Parity Protocol

The default parity workflow must minimize debugging cycles.

Protocol:

1. start with `nout = 0` in reference whenever possible to obtain a single RHS evaluation with no timestepper noise
2. compare a single component or operator on the smallest case that exercises it
3. only after one-RHS parity is green, move to one fixed-step comparison
4. only after one-step parity is green, move to short-window trajectory comparison
5. only after short-window parity is green, move to medium and long runs

Rules:

- no new component should be wired into a larger case before its smallest exercising case is already passing
- no adaptive-step investigation should begin before the fixed-step one-step harness is green
- when a mismatch appears, reduce to the smallest Tier A or Tier B case and re-run the protocol

### 6.4 Test Taxonomy

Every ported feature must fit into this test structure:

- unit: local formulas, parser logic, boundary stencils, source terms
- MMS: operator accuracy and convergence
- regression: dump-backed field parity on tiny cases
- physics: integrated runs against reference behavior or known theory
- differentiability: `grad`/`jvp` smoke tests through a representative scalar loss
- performance: compile time, step time, and memory on CPU and GPU

### 6.5 Design Rule

Preserve reference variable names and component boundaries until parity is established. Refactoring into a more compact or more elegant JAX form can happen only after the matching test suite is already green.

## 7. Component Rollout Order

Port components in this order:

1. scheduler, config, normalization, guards, diagnostics metadata
2. `sound_speed`
3. `fixed_density`, `evolve_density`
4. `fixed_temperature`, `set_temperature`, `isothermal`, `evolve_pressure`, `evolve_energy`
5. `fixed_velocity`, `evolve_momentum`, `scale_timederivs`
6. `quasineutral`, `fixed_fraction_ions`, `zero_current`, `electron_force_balance`
7. `sheath_boundary_simple`, `sheath_boundary`, `sheath_boundary_insulating`, `sheath_closure`, `noflow_boundary`, `neutral_boundary`
8. `simple_conduction`, `braginskii_collisions`, `braginskii_friction`, `braginskii_heat_exchange`, `braginskii_conduction`, viscosities, thermal force
9. `classical_diffusion`, `anomalous_diffusion`, `diamagnetic_drift`, `polarisation_drift`
10. `vorticity`, `relax_potential`
11. `electromagnetic`
12. `neutral_mixed`, `neutral_full_velocity`, `neutral_parallel_diffusion`, `solkit_neutral_parallel_diffusion`
13. `recycling`, `simple_pump`, `temperature_feedback`, `upstream_density_feedback`, `detachment_controller`
14. `reaction_parser`, `ionisation`, hydrogen charge exchange, ADAS/AMJUEL reaction families, impurity source terms

This order follows the actual reference dependency graph more closely than the archived prototype code and should remain the default unless a missing low-level dependency forces a local reordering.

## 8. Staged Plan

## 8A. Five-Step Finish Plan

The remaining port should be executed as five broad capability drops rather than many narrow iterations. Each step should land a large, usable block of functionality, together with the tests needed to prevent regression and rework.

### Step 1. Freeze the Numerical Backbone

Goal:

- finish the shared solver/operator substrate that every remaining component depends on
- stop solving the same infrastructure problem repeatedly inside physics slices

Scope:

- finalize structured mesh/metric/guard semantics for all currently selected 1D, 2D, and reduced 3D cases
- finish the common elliptic/inversion layer used by electrostatic and electromagnetic paths
- finish the implicit transient substrate:
  - matrix-free path retained as validated fallback
  - sparse Jacobian assembly, sparse Newton/GMRES, and BDF/BDF2 stepping hardened into a reusable backend
- centralize that substrate in shared solver modules rather than case-local helper code:
  - active-domain state pack/unpack
  - finite-difference Jacobian sparsity/color grouping
  - sparse and matrix-free Newton solves
  - reusable backward-Euler and BDF2 residual forms
- standardize one common output/diagnostic registry and comparison harness across all supported components

Tests required in this step:

- unit tests for guard fills, metric normalization, flux stencils, limiters, and elliptic solves
- solver tests for one-RHS, one-step backward Euler, BDF2, adaptive controller bookkeeping, and restart/state pack-unpack
- regression tests for portable summaries, array comparisons, and metadata parity
- performance smoke tests on representative small 1D/2D/3D cases so infrastructure regressions are caught early:
  - warm compiled kernel timings for the hot numerical path
  - end-to-end CLI timings so process/setup overhead is tracked separately from solver throughput

Exit criteria:

- no remaining component needs to invent its own stepping, Jacobian, or output-comparison logic
- the neutral, fluid, vorticity, drift, and EM branches all share the same implicit/transient backbone

Current Step 1 status:

- the shared `src/jax_drb/solver/` package now exists and is the active home for active-domain vectorization, sparse locality/color-group construction, grouped difference-quotient Jacobians, sparse Newton/GMRES, matrix-free Newton-Krylov, and backward-Euler/BDF2 residuals;
- the neutral implicit branch now consumes that shared backbone rather than carrying a private copy of the same logic;
- the shared electrostatic inversion layer now also exists in `src/jax_drb/solver/elliptic.py`, and the vorticity/blob branches now reuse that same JAX Fourier-Helmholtz backend instead of maintaining separate mode-by-mode solvers;
- differentiability and performance smoke now explicitly cover that inversion backbone through JIT/`grad` checks on the shared solver, vorticity potential solve, and blob RHS;
- Step 1 is now closed for the branches currently in tree: the shared implicit substrate, shared electrostatic inversion layer, compilation-cache runtime hardening, and differentiability/performance smoke checks are all in place.

### Step 1A. Performance Hardening Before Step 2

Goal:

- remove avoidable execution overhead from the stabilized Step 1 backbone without changing numerical behavior
- ensure repeated native runs are practical enough to support the much broader Step 2 parity campaign

Scope:

- keep the hot operator paths vectorized and JAX-native where the current branches already allow it
- enable persistent compilation reuse for CLI and scripted runs so repeated process launches do not pay the full compilation cost each time
- record both cold and warm timings for representative one-step cases and separate process/setup overhead from warm kernel throughput

Exit criteria:

- repeated native CLI runs materially improve after the first compile on the same machine
- performance smoke coverage exists for both the hot kernels and the repeated entrypoint workflow
- no parity baselines or differentiability tests regress while applying these runtime changes

### Step 2. Land the Full Open-Field Plasma + Neutral + Recycling Stack

Goal:

- finish the entire 1D open-field capability in one push, including neutrals, recycling, sheath closures, reactions, and control hooks

Scope:

- complete all remaining 1D core plasma closures
- complete `neutral_mixed`, `neutral_parallel_diffusion`, and the required reaction/source machinery for hydrogenic recycling workflows
- complete sheath/recycling/control components:
  - sheath closures
  - `neutral_boundary`
  - `recycling`
  - `simple_pump`
  - temperature/density feedback controllers
- wire all relevant diagnostics, target fluxes, integrated balances, and derived quantities

Tests required in this step:

- unit tests for sheath formulas, recycling deposition, reaction-source bookkeeping, and target diagnostics
- parity tests on:
  - `tests/integrated/1D-fluid`
  - `tests/integrated/neutral_mixed`
  - `tests/integrated/neutral_parallel_diffusion`
  - `tests/integrated/1D-recycling`
  - `tests/integrated/1D-recycling-dthe`
- physics tests for conservation, positivity floors, steady-state consistency, and target power/particle balance
- regression plots/tables for 1D profiles and divertor diagnostics

Exit criteria:

- all representative 1D divertor/recycling workflows run from CLI and Python
- 1D parity is complete enough that no later 2D/3D work needs to backfill 1D source, boundary, or diagnostic logic

Current Step 2 note:

- the neutral RHS branch now includes the traced soft-floor rule, and the remaining neutral transient mismatch has been narrowed to target-adjacent active `y` cells in the momentum RHS; the next Step 2 work should therefore focus on exact target-boundary parallel viscosity/conduction parity before exposing neutral transients or recycling workflows through the public runner
- the staged comparison ladder now includes first-output baselines for `1D-recycling` and `1D-recycling-dthe`, so the open-field sheath/recycling implementation can be driven against low-iteration reference targets rather than only the existing long-run case
- the same ladder now also includes explicit `one_rhs` baselines for `1D-recycling` and `1D-recycling-dthe`, including target-recycling source diagnostics and trimmed active-domain `ddt(...)` outputs, so the native Step 2 runner can follow the intended RHS-first parity protocol rather than jumping directly to whole-step state parity
- shared open-field utilities are now in-tree for the exact no-flow guard rules, the electron-force-balance source term, limited free extrapolation, and target-recycling source assembly, and both `recycling_1d_rhs` and `recycling_dthe_rhs` now run natively against live reference baselines
- the latest localized RHS fix was the missing ion-ion Braginskii thermal-force exchange for the D-T pair when `override_ion_mass_restrictions = true`; staged multispecies momentum-source diagnostics are now back inside the locked tolerances, so the remaining Step 2 defect is no longer in the traced open-field source bookkeeping
- staged evolved-state RHS checks are now locked for `recycling_1d_one_step` and `recycling_dthe_one_step`; the remaining Step 2 work is therefore the transient ladder (`one_step`, `short_window`, and control/recycling long-run behavior), not another round of open-field source-term reconstruction
- the recycling transient branch now reuses a cached runtime model during packed RHS evaluations, and the shared sparse Newton backend now has a direct sparse linear-solve mode that is enabled for recycling substeps; this reduced the packed recycling RHS cost by about an order of magnitude and removed the worst GMRES bottleneck, but the public `one_step` recycling cases remain blocked because the generic adaptive BDF wrapper is still too slow over the full output interval
- the public recycling runner no longer routes through the generic adaptive BDF wrapper; it now uses a continuation-based sparse implicit ladder on top of the shared recycling backward-Euler stepper, with small-step regression coverage in-tree. That substrate is the active Step 2 transient path, but the full first-output recycling cases are still too slow to claim parity-complete in this pass
- the recycling transient path now follows the reference controller history more closely: the upstream-density controller integral is updated on accepted steps with a trapezoid rule rather than being solved as an extra implicit state variable, and the sheath preparation order now applies the electron boundary state before the ion boundary state so the ion sheath sees the electron boundary density/pressure fields rather than the pre-sheath quasineutral sum
- the remaining transient blocker is now localized by a dedicated short-step probe in [diagnose_recycling_transient_step.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_transient_step.py): on a `timestep = 25` reference run, the native backward-Euler step already misses the evolved state while the native RHS evaluated on that same reference-evolved state stays tight, so the remaining Step 2 defect is the transient integrator path itself rather than another open-field operator/source mismatch
- the follow-on fixed-substep probe closes the remaining ambiguity: a full `100 x dt = 25` backward-Euler march over the single-species first-output interval still runs to `NaN` after about `34.5 s` on this machine, so Step 2 will not be finished by simply shrinking accepted backward-Euler steps; the next fix must be a more reference-faithful BDF-like transient path
- a new in-tree `adaptive_be` transient probe now confirms that the remaining defect is still integration accuracy rather than RHS parity: on the short `timestep = 25` recycling probe, adaptive accepted backward-Euler substeps drive the main ion/electron errors down to the `1e-3` to `1e-2` range, while the neutral branch remains the visible short-step limiter; that mode is not public yet because the full `timestep = 2500` output interval is still too expensive to claim as the Step 2 completion path
- an experimental `adaptive_bdf` transient path is now in-tree as the direct Step 2 successor to that probe: it uses backward-Euler startup, BDF2 continuation, WRMS error control on the active domain, and the recycling-aware initial `dt` heuristic. It remains internal-only until the full `recycling_1d_one_step` interval is both parity-clean and fast enough to replace the continuation ladder in the public runner
- recent performance hardening materially reduced the open-field transient cost: NumPy fast paths now bypass `jax.numpy` scatter/device-put overhead in the shared open-field helpers, the parallel neutral operator is vectorized, and the current first-output probes are down to about `40.6 s` for `recycling_1d_one_step` and about `76.9 s` for `recycling_dthe_one_step` on this machine
- the latest targeted source fix came from the dense target-band probes rather than another solver change: neutrals were inheriting the charged-species default `temperature_floor = 0.1`, which forced `Pd` too high near the target and then biased `Sd_Dpar` / `Ed_Dpar`; neutrals now default to zero temperature floor unless explicitly configured, and the dense `dt = 1`, total-time `25` single-species recycling probe now lands in the low-`1e-3` / `1e-2` range across the tracked fields
- despite those improvements, the first-output transient parity remains blocked: the current single-species native probe is now down to about `1.00e-1` (`recycling_1d_one_step`, worst field `Nd+`) in the trimmed active domain, with the remaining error concentrated in the top two active `y` cells (`Nd+`, `Pe`, `Nd`, `NVd`); the multi-species `recycling_dthe_one_step` continuation path still fails, but the native `bdf` path now reaches the first output interval and is the current candidate route for the multispecies Step 2 transient milestone
- the latest targeted probes narrow that blocker further:
  - shrinking the continuation substep (`suggested_dt = 500, 100, 50, 25, 10`) does not materially change the single-species first-output error, so Step 2 will not be finished by continuation-step tuning alone;
  - the accepted-step upstream-density controller history still undershoots the staged reference restart integral after the first output interval, but the source-term delta from that controller mismatch is too small to explain the remaining target-band state gap on its own;
  - reapplying the native sheath-preparation path to the evolved single-species output fixes most of the upper guard-cell discrepancy but leaves the target-adjacent active `Nd+` / `Pe` mismatch essentially unchanged, so the remaining defect is in the active transient evolution rather than only in output write-back semantics
  - the cleaned SciPy `bdf` path now carries the controller integrals as explicit ODE state, which keeps the RHS pure during time integration; that is the correct formulation to preserve, but it does not materially improve the long single-species first-output result on its own;
  - shrinking the internal BDF `max_step` from `25` to `10` or `5` leaves the long `recycling_1d_one_step` error essentially unchanged, so the remaining defect is not just coarse internal BDF stepping;
  - direct `t = 25` and `t = 250` reference comparisons now show the same pattern: `Nd+`, `Pd+`, and `Pe` are already relatively tight on the BDF path, while `NVd+`, `Nd`, `Pd`, and especially near-zero `NVd` dominate the remaining short/medium-window error. The next Step 2 work should therefore target the neutral / neutral-momentum transient terms directly rather than another generic timestepper rewrite.
- the new magnitude-aware neutral report confirms that this is not only a denominator problem: `NVd` is mostly near-zero-reference noise, but `Nd`, `Pd`, `NVd+`, `ddt(Nd)`, and `ddt(NVd+)` still carry real `O(5e-2 .. 1e-1)` significant relative error on cells above a `1e-2 * max(|ref|)` floor. That is the active Step 2 blocker now.
- the latest Step 2 RHS pass closed most of that short-window neutral-side gap:
  - neutrals in the recycling branch are no longer evolved as source-only fields; `Nd`, `Pd`, and `NVd` now include the same final transport/compression assembly pattern used in the reference density/pressure/momentum components;
  - the open-field parallel gradient now follows the reference `DDY / sqrt(g_22)` centered metric form instead of the older `1 / (J * Δy)` approximation, which materially improves the target-adjacent momentum remainder;
  - on the fresh `timestep = 25` single-species probe, `Nd`, `Pd`, `Nd+`, `Pd+`, and `Pe` are now all below `1e-3` significant relative error, `NVd` is still dominated by near-zero denominators, and the only remaining visible short-window blocker is the target-band `NVd+` channel at roughly `5e-2` significant relative error.

Current Step 2/3 status markers:

| Case | Status | Note |
| --- | --- | --- |
| `neutral_mixed_rhs` | `native-validated` | Active-domain RHS parity is locked. |
| `neutral_mixed_one_step` | `reference-only target` | Baseline exists; native transient is not runner-promoted. |
| `neutral_mixed_short_window` | `reference-only target` | Baseline exists; native transient is not runner-promoted. |
| `recycling_1d_rhs` | `native-validated` | RHS parity and controller bookkeeping are locked. |
| `recycling_dthe_rhs` | `native-validated` | RHS parity and multispecies collision bookkeeping are locked. |
| `recycling_1d_one_step` | `blocked` | Native first-step transient is not parity-clean yet. |
| `recycling_dthe_one_step` | `blocked` | Native first-step transient is not parity-clean yet. |
| `recycling_1d_long` | `blocked` | Long-run parity depends on the transient ladder. |
| `integrated_2d_recycling_rhs` | `native-scaffolded` | Native staged local-dump RHS path now runs, preserves slab-local physical targets, keeps the dumped target state, injects dump-backed density and ion-pressure source fields for the staged integrated case, restores staged target-recycling diagnostics directly from the dump, and follows the reference `sheath_boundary_simple` electron guard/energy closure closely enough that the large target-row `ddt(Pe)` miss is gone; the remaining live mismatch is down to small pressure-statistics residuals. |
| `integrated_2d_recycling_one_step` | `native-scaffolded` | First native transient scaffold now starts from the staged one-RHS dump state, marches one native recycling step on the shared implicit backbone, uses the same dump-backed `SNd`/`SNd+` and `SPd`/`SPd+` staging during the step that already tightened the staged RHS path, and preserves the dump-backed target state throughout the transient RHS evaluations; the remaining live mismatch is now down to tiny summary/statistic residuals rather than a meaningful one-step state defect. |
| `integrated_2d_recycling_short_window` | `native-scaffolded` | The staged integrated 2D recycling path now supports the full configured `nout=5` short window from the dump-backed initial state, using the same transient target preservation and dump-backed source staging as the one-step path. Live native/reference differences are still visible in `Ed_target_recycle`, `Pe`, and tiny neutral-side residuals, so this is the main Step 3 transient target rather than a locked parity result. |
| `integrated_2d_recycling_medium_window` | `native-scaffolded` | The integrated 2D recycling transient path now also honors the staged `nout=20` medium-window manifest override and can march the longer window on the same dump-backed source-preserving workflow. After fixing the staged recycling-energy path to use the configured sheath `gamma_i`, the remaining live differences are led by `Sd_target_recycle`, then `Pe`, with `Ed_target_recycle` reduced into the same small residual band as the other staged diagnostics. |

### Step 3. Land the Full 2D Electrostatic Edge/SOL Stack

Goal:

- finish the entire 2D electrostatic edge/SOL capability in one push rather than separate blob, drift-wave, transport, and recycling subprojects

Scope:

- complete shared 2D transport/drift operators
- finish the coupled density-pressure-momentum-vorticity/electrostatic field path
- lift the finished 1D neutral/recycling/reaction machinery into 2D
- support the full set of selected 2D reference examples:
  - diffusion/transport
  - blob family
  - drift-wave
  - 2D recycling
  - 2D turbulence

Tests required in this step:

- unit tests for 2D flux assembly, field inversion coupling, and drift/source operator splits
- parity tests on Tier A/B/C 2D cases, including one-RHS, one-step, short-window, and medium-window comparisons
- physics tests for blob COM/velocity, drift-wave growth/frequency, energy exchange, and recycling target/source balances
- regression plots for spectra, RMS fluctuation levels, profile overlays, and benchmark metrics

Exit criteria:

- the 2D electrostatic code path is feature-complete enough for transport, blob, and recycling studies
- all selected 2D comparison cases share one production path instead of case-specific implementations

Current Step 3 note:

- the stable Step 3 reference target is now the integrated `2D-recycling` workflow, staged with its required external artifact bundle and `process_count = 10` in the harness; the remaining Step 3 work is the native 2D recycling implementation on top of that target, not more geometry staging
- the native runner now has a staged local-dump-backed `integrated_2d_recycling_rhs` entry path: it no longer fails on missing `nx/ny/nz`, and it can ingest the staged local `BOUT.dmp.0.nc` mesh/metric/state slab to produce the public compare surface for the integrated case
- that new Step 3 path is still scaffolding, but it now honors slab-local physical targets, avoids double-applying sheath closures to dump-backed states, restores dump-preserving guard cells, injects dump-backed `SNd`/`SNd+` density sources and `SPd`/`SPd+` ion-pressure sources for the staged integrated case, restores staged `Sd_target_recycle` and `Ed_target_recycle` directly from the dump, and uses a source-faithful `sheath_boundary_simple` electron boundary closure; the remaining live mismatch is now limited to 9 strict summary issues, all in `ddt(Pd)`, `ddt(Pd+)`, and `ddt(Pe)` statistics. 
- the same integrated workflow now has a native `short_window` rung and committed reference baselines. It is not locked parity yet, but it is the first multi-output Step 3 transient target that can be exercised end-to-end without relying on the broken tokamak examples.
- the same integrated workflow now also has a native `medium_window` rung honoring the manifest `nout=20` override, so Step 3 can exercise a longer transient on the stable integrated geometry before the broader 2D production path is finalized.
- the staged integrated transient path now also reuses dump-backed ion momentum-source fields (`SNVd+`, `SNVd`) in addition to the already staged density and ion-pressure sources. That keeps the integrated 2D march source-faithful without reintroducing the rejected staged `SPe` override, which made the medium-window `Pe` drift worse rather than better.
- the broader integrated `2D-production` workflow now also has a committed `one_rhs` rung on the same dump-backed geometry lane. Its first live native comparison is already limited to the same small `ddt(Pe)` / `ddt(Pd)` pressure-stat residuals seen in the integrated recycling RHS path, so it widens Step 3 without introducing a new geometry blocker.
- the broader integrated `2D-production` workflow now also has a committed `one_step` rung on the same dump-backed geometry lane. It is not parity-clean yet, but the first live compare is now localized enough to drive the next Step 3 pass: the dominant one-step residuals are `Pe`, then `Pd`, then `Sd_target_recycle`, while the rest of the state remains much closer.
- the `integrated_2d_production_one_step` path now uses the stiffer `bdf` transient backend instead of the default single-ion continuation path. That substantially reduces the broader production one-step `Pd` drift and improves `Pe`; the remaining field-level miss is now led by `Pe`, with `Sd_target_recycle`, `Ed_target_recycle`, `NVd+`, and `Nd` all clustered in the same small ~`5e-3` to `7e-3` relative band.
- the broader integrated `2D-production` workflow now also has a committed `short_window` rung (`nout=5`) on the same dump-backed geometry lane. It is not parity-clean yet, but it gives Step 3 a real broader-production transient ladder: the current live differences are led by `Pe`, then `Sd_target_recycle`, then `NVd+` / `Nd`, while `Pd` is already much tighter. A direct `bdf` probe for this longer window did not beat the current continuation path overall, so the next production-short-window work should stay focused on physics/source closure rather than solver swapping alone.
- the broader integrated `2D-production` workflow now also has a committed `medium_window` rung (`nout=20`) on the same dump-backed geometry lane. It extends the broader production transient ladder and confirms the same ranking as the short window over a longer interval: the live differences are still led by `Pe`, then the recycling/momentum side (`Sd_target_recycle`, `NVd+`, `Nd`), while `Pd` remains relatively tighter and a pure backend swap is still not enough to close the longer-window production gap.

### Step 4. Land the Full 3D Electromagnetic + Tokamak Capability

Goal:

- finish the remaining 3D and electromagnetic path in one push, using the already-finished backbone and 2D physics stack

Scope:

- complete `electromagnetic`, `Apar`, coupled momentum/field evolution, and required inversions
- extend geometry support from reduced/annulus/shifted 2D to the selected 3D tokamak cases
- support the chosen reduced 3D annulus and TCV-X21 comparison ladder
- complete 3D diagnostics, slices, fluctuation statistics, and runtime/performance tracking

Tests required in this step:

- unit tests for electromagnetic operators, `Apar` solves, and 3D geometry/metric handling
- parity tests on minimal EM benchmarks and selected reduced 3D tokamak cases
- regression tests for 3D cross-sections, time traces, fluctuation statistics, and selected spectral measures
- CPU/GPU smoke tests to ensure the same production path runs on both backends

Exit criteria:

- selected 3D electromagnetic tokamak cases run through the main CLI/Python workflow
- 1D, 2D, and selected 3D parity coverage is broad enough for a complete reference-paper parity claim

### Step 5. Finalize Multi-Species, Outputs, Docs, and Reviewer-Facing Validation

Goal:

- close the remaining gaps in species/reaction breadth, output compatibility, and research-grade presentation

Scope:

- finish remaining multi-species / impurity / reaction breadth needed by the selected hydrogen/neon and mixed-species workflows
- finalize output/restart/log compatibility and metadata completeness
- finalize documentation and validation assets:
  - inputs/outputs/equations/numerics/geometry pages
  - benchmark and validation galleries
  - reproducible parity and physics figures
  - user-facing examples and API/CLI docs
- complete the journal-grade validation matrix across selected 1D, 2D, and 3D cases

Tests required in this step:

- parity regressions for all selected final comparison cases
- output/restart compatibility tests
- doc-build tests and example smoke tests
- validation scripts that regenerate the committed benchmark/parity artifacts from the supported workflows

Exit criteria:

- the selected comparison ladder is fully covered
- docs and outputs are complete enough for external users and a computational-physics paper
- remaining work is extension-oriented rather than parity-critical

### Stage 0. Repository Reset and Audit Baseline

Status: completed for the archive step; implementation artifacts still to be created.

Goals:

- keep the pre-reset repository state intact under `legacy/`
- establish a clean implementation root
- record exactly what from reference is being ported

Deliverables:

- `legacy/` archive of all prior repository contents
- `docs/parity_matrix.md`
- `docs/implementation_inventory.md` summarizing:
  - all source components,
  - tests,
  - examples,
  - docs pages,
  - literature-derived extension topics

Tests:

- none beyond repository integrity

Exit criteria:

- no active code remains outside `legacy/` except new plan and new implementation files

### Stage 1. Configuration, Scheduling, Normalization, and State Semantics

Goals:

- replicate reference runtime organization before porting full physics
- make `jax_drb` accept reference-like inputs
- mirror the state contract used by components

reference scope:

- `main source driver`
- `component.cxx`
- `component_scheduler.cxx`
- `guarded_options.cxx`
- `permissions.cxx`
- docs: `inputs.rst`, `execution.rst`, `developer.rst`, `postprocessing.rst`

Implementation tasks:

- implement a `BOUT.inp` parser for the reference subset used by tests/examples
- reproduce ordered component scheduling and species/type expansion
- define immutable state pytrees with explicit guard cells
- implement normalization and metric normalization switches
- define diagnostics metadata storage with reference naming conventions
- implement a minimal CLI:
  - `jax_drb run path/to/BOUT.inp`
  - `jax_drb compare --reference-run ...`
- implement the Python API entry point around the same config/state model

Required tests:

- unit tests mirroring `test_component`, `test_component_scheduler`, `test_permissions`, `test_guarded_options`
- regression tests proving component order and species expansion match reference on tiny configs
- normalization tests for `Nnorm`, `Tnorm`, `Bnorm`, `rho_s0`, `Cs0`, `Omega_ci`
- differentiability smoke test through a no-op or fixed-density/fixed-temperature pipeline

Exit criteria:

- reference input order yields the same scheduled component list in JAX
- state names, diagnostics names, and normalization constants match reference on reference cases

### Stage 2. Mesh, Metrics, Guard Cells, Standard Boundaries, and I/O

Goals:

- reproduce the geometric and guard-cell semantics that all later operators depend on
- make reference reference data ingestible

reference scope:

- docs: `domain_grid.rst`, `boundary_conditions.rst`, `postprocessing.rst`
- BOUT boundary semantics used by reference
- metric recalculation/normalization path

Implementation tasks:

- implement mesh loaders for reference/BOUT reference grids
- preserve two guard cells with reference midpoint conventions
- port boundary primitives:
  - Neumann
  - Dirichlet/value copy where used
  - `setBoundaryTo`
  - `decaylength(...)`
  - sheath-oriented free extrapolation helpers
- implement restart/output readers and writers for parity workflows
- preserve field naming and units/conversion metadata in outputs

Required tests:

- unit tests for guard-cell indexing and midpoint relations
- unit tests for `limitFree`, Neumann, `setBoundaryTo`, and boundary averages
- regression tests against tiny reference dumps for boundary-populated fields
- output metadata tests matching reference naming and units rules

Exit criteria:

- any later operator can consume the JAX field object without special-case boundary code
- JAX can read reference reference states and write comparable output fields

### Stage 3. Numerical Operators and Elliptic Solves

Goals:

- reproduce the discrete operators before assembling large physics blocks
- verify order of accuracy and conservation properties

reference scope:

- `div_ops.cxx`
- finite-volume parallel flux operators
- Poisson/Helmholtz/Ampere support used by `vorticity` and `electromagnetic`
- docs: `solver_numerics.rst`
- MMS suite: `tests/mms_operator`

Implementation tasks:

- port centered and FV parallel operators:
  - `Grad_par`
  - `Div_par`
  - `FV::Div_par_mod`
  - `FV::Div_par_fvv`
  - `FV::Div_par_fvv_heating`
- port ExB advection operators used by density/pressure/momentum/vorticity
- once Tier C transient cases become runtime-gated, replace scalar-loop operator kernels with parity-preserving vectorized implementations, and lock those rewrites with scalar-reference regression tests before relying on them in longer benchmark windows
- port perpendicular diffusion and hyper-diffusion operators that appear in core components
- implement JAX elliptic solves for:
  - vorticity-to-potential inversion
  - `Apar` Helmholtz/Ampere solve
- choose initial solver path conservatively:
  - matrix-free iterative methods in JAX first
  - preconditioned variants second

Required tests:

- direct operator unit tests on analytic fields
- MMS parity for all operators covered in `tests/mms_operator`
- conservation tests for FV operators on periodic/tiny domains
- differentiability tests through operator application and through one elliptic solve
- CPU/GPU step parity tests for operator kernels

Exit criteria:

- MMS convergence matches reference expectations
- operator-level regression errors are below tolerances needed for component parity

### Stage 4. Core 1D Plasma Equations and Open-Field Baseline

Goals:

- establish a stable 1D open-field single-/few-species baseline
- reproduce the principal density/pressure/momentum/energy equations

reference scope:

- `sound_speed`
- `evolve_density`
- `evolve_pressure`
- `evolve_momentum`
- `evolve_energy`
- `fixed_density`
- `fixed_temperature`
- `fixed_velocity`
- `isothermal`
- `set_temperature`
- `scale_timederivs`
- `quasineutral`
- `fixed_fraction_ions`
- `zero_current`
- `electron_force_balance`
- docs: `equations.rst`, `boundary_conditions.rst`, `closure.rst`

Implementation tasks:

- port transform/finally semantics for the core evolving components
- support both `N`/`logN` and `P`/`logP` evolution paths
- preserve density/pressure/temperature floors and source bookkeeping
- implement open-field parallel transport, Bohm/sheath target coupling, and sound-speed sharing
- define the first production time integrator path:
  - explicit RK for turbulence/transient tests
  - simple backward Euler or IMEX stepping for stiff transport tests

Required tests:

- unit tests mirroring reference unit tests for density/pressure/momentum/energy/sound speed
- physics tests from:
  - `tests/integrated/evolve_density`
  - `tests/integrated/1D-fluid`
  - `tests/integrated/diffusion`
  - `tests/integrated/sod-shock`
  - `tests/integrated/sod-shock-energy`
- example regression on `examples/tokamak-1D/extra/1D-periodic`
- short-window reference parity runs using tiny meshes and `nout = 0`-style comparisons where possible
- differentiability test of a scalar target diagnostic with respect to an upstream source or initial profile

Exit criteria:

- 1D open-field plasma cases run from the JAX CLI and Python API
- short-window field parity is acceptable for density, pressure/temperature, momentum, and energy

### Stage 5. Sheaths, Boundaries, Recycling, and Control-Relevant 1D Cases

Goals:

- make 1D divertor/SOL workflows faithful enough to cover recycling and detachment-style examples

reference scope:

- `sheath_boundary_simple`
- `sheath_boundary`
- `sheath_boundary_insulating`
- `sheath_closure`
- `noflow_boundary`
- `neutral_boundary`
- `recycling`
- `simple_pump`
- `temperature_feedback`
- `upstream_density_feedback`
- `detachment_controller`
- docs: `boundary_conditions.rst`, `feedback_control.rst`

Implementation tasks:

- port target fluxes, target heat transmission, sheath potential/electron force balance coupling
- port recycling source deposition and sheath diagnostic dependencies
- port control components with the same signal conventions used by reference examples

Required tests:

- unit tests for sheath formulas and recycling source bookkeeping
- physics tests from:
  - `tests/integrated/1D-recycling`
  - `tests/integrated/1D-recycling-dthe`
  - `examples/tokamak-1D/extra/1D-recycling`
  - `examples/tokamak-1D/extra/1D-recycling-with-Tt-control`
  - `examples/tokamak-1D/extra/1D-recycling-with-detachment-control`
- regression on target flux and control trajectory shape
- differentiability smoke tests through a short controlled 1D run

Exit criteria:

- the full 1D divertor/recycling workflow is usable in JAX
- control-oriented examples run without leaving the pure-JAX runtime path

### Stage 6. Cross-Field Electrostatic Plasma Physics

Goals:

- add the 2D electrostatic physics that makes reference more than a 1D transport code

reference scope:

- `vorticity`
- `relax_potential`
- `diamagnetic_drift`
- `polarisation_drift`
- `classical_diffusion`
- `anomalous_diffusion`
- `braginskii_collisions`
- `braginskii_friction`
- `braginskii_heat_exchange`
- `simple_conduction`
- `braginskii_conduction`
- `braginskii_electron_viscosity`
- `braginskii_ion_viscosity`
- `braginskii_thermal_force`

Implementation tasks:

- port the vorticity equation and potential solve with the same source decomposition reference uses
- preserve `phi + Pi_hat` and diamagnetic current bookkeeping
- port collisional and conductive closures in the order reference integrated tests require
- support cross-field turbulence examples and 2D transport examples

Required tests:

- unit tests mirroring reference tests for vorticity, drifts, collisions, conduction, viscosities
- physics tests from:
  - `tests/integrated/vorticity`
  - `tests/integrated/2D-energy`
  - `tests/integrated/drift-wave`
  - `examples/other/blob2d*`
  - `examples/tokamak-2D/diffusion*`
  - `examples/tokamak-2D/recycling*`
  - `examples/tokamak-2D/turbulence`
- regression on diagnostic families `pf`, `ef`, `mf`
- differentiability checks through a short 2D electrostatic run

Exit criteria:

- blob, drift-wave, and 2D transport/recycling cases are reproducible in JAX with stable diagnostics

### Stage 7. Electromagnetic and 3D Turbulence Capability

Goals:

- complete the core reference field evolution path for 3D turbulence

reference scope:

- `electromagnetic`
- EM contributions already present in momentum/pressure/vorticity paths
- examples and tests involving finite electron mass and `Apar`

Implementation tasks:

- port `Apar` solve and canonical/mechanical momentum handling
- preserve `Apar_flutter` and EM source bookkeeping
- validate 3D stepping on representative annulus and tokamak cases
- staged entry ladder is now `tests/integrated/alfven-wave` with committed `alfven_wave_rhs` and `alfven_wave_one_step` reference baselines before broader EM short-window and tokamak escalation work
- first real Stage 4 operator slices are now `Ajpar`, `Apar`, the physical-domain `NVe` one-step reconstruction, and the dominant electromagnetic RHS cores on the Alfvén benchmark: reconstruct `Ajpar` from the charged-species momentum sum, solve `Apar` natively on the single-cell slab/Neumann Alfvén benchmark using the source-faithful electromagnetic Helmholtz coefficient (`alpha_em` from charged densities, `beta_em` from normalization, periodic-Y and periodic-Z guard handling), invert that same slab solve to recover the physical/inner-radial `NVe` one-step field, reconstruct the physical/inner-radial `ddt(NVe)` planes from the benchmark’s periodic central-difference closure on `Vort`, and reconstruct the inner-radial shoulder `ddt(Vort)` planes on `x=1,3` from the benchmark’s exact inner-radial `DDY/ DDZ` closure while leaving the tiny central-plane `x=2` `ddt(Vort)` signal and the outermost saved radial and parallel guard planes staged. `phi` and the remaining EM transient terms stay on the staged dump-backed path until each operator is ported and revalidated
- the Alfvén ladder now also has a committed `alfven_wave_short_window` rung at `nout=20`, which is the smallest window that gives a stable frequency estimate from the saved history while staying under the repository size cap. That rung is now part of the active Step 4 validation loop:
  - committed summary baseline,
  - committed compressed array baseline,
  - native short-window parity through the same partially native/dump-backed scaffold,
  - benchmark analysis JSON and docs figures for measured vs. analytic phase speed and frequency.

Required tests:

- unit test mirroring `test_electromagnetic`
- physics tests from:
  - `tests/integrated/alfven-wave`
  - electromagnetic annulus examples
  - selected `examples/tokamak-3D/tcv-x21` reduced cases
- regression on `Apar` and EM-corrected momentum fields
- differentiability smoke test through a tiny EM trajectory

Exit criteria:

- JAX supports CPU/GPU 3D EM runs without leaving production code paths

### Stage 8. Neutrals, Reactions, Impurities, and Full Multi-Component Parity

Goals:

- reproduce the reference multi-species, neutral, and reaction machinery

reference scope:

- `neutral_mixed`
- `neutral_full_velocity`
- `neutral_parallel_diffusion`
- `solkit_neutral_parallel_diffusion`
- `ionisation`
- hydrogen charge exchange
- ADAS/AMJUEL reaction families
- impurity examples such as neon
- docs: `reactions.rst`

Implementation tasks:

- port reaction parsing and stoichiometric source accounting
- load and interpolate atomic/molecular data tables
- port neutral-fluid closures and projected diffusion models
- reproduce diagnostic naming for reaction rates and sources

Required tests:

- unit tests mirroring:
  - `test_reaction_parser`
  - `test_reactions`
  - `test_neutral_mixed`
  - `test_neutral_parallel_diffusion`
  - `test_recycling`
- MMS/conservation tests for neutral operators where reference provides them
- physics tests from:
  - `tests/integrated/neutral_mixed`
  - `tests/integrated/neutral_parallel_diffusion`
  - `tests/integrated/snb/*`
  - `examples/tokamak-1D/extra/1D-hydrogen*`
  - `examples/tokamak-1D/extra/1D-neon*`
  - `examples/tokamak-1D/extra/solkit-comparison`
- regression on reaction diagnostics and impurity balance
- differentiability smoke tests through at least one reaction-coupled run

Exit criteria:

- the JAX code can run the representative reference hydrogen/neon/recycling workflows with matching qualitative and short-window quantitative behavior

### Stage 9. User-Facing Workflow, Outputs, Restarts, and Packaging

Goals:

- turn the port into a usable research tool rather than a parity harness only

Implementation tasks:

- finalize the Python API
- finalize the CLI to support:
  - run from `BOUT.inp`
  - restart from JAX or reference-compatible restart input where feasible
  - compare to a reference run
  - inspect diagnostics
- write documented output/restart formats
- provide reproducible example run scripts and benchmark commands
- add packaging, versioning, and environment setup docs

Required tests:

- CLI smoke tests
- restart/reproducibility tests
- output metadata tests
- package import tests on CPU and GPU environments

Exit criteria:

- a new user can run a documented 1D, 2D, and 3D case from the CLI or Python API without touching internal code

### Stage 10. Performance, Differentiability, and Research Extensions

This stage begins only after Stages 1-9 are stable.

### 10A. Performance hardening

Goals:

- reduce compile overhead,
- improve memory behavior,
- retain differentiability.

Tasks:

- fuse kernels only where tests already prove semantic parity
- add checkpointing/rematerialization where necessary
- benchmark explicit vs IMEX/backward-Euler options
- add GPU scaling baselines and memory ceilings

### 10B. Conservative DRB branch

Motivation:

- De Lucca 2026 shows a conservative reformulation of drift-reduced fluid plasma models with exact energy/mass/charge/momentum conservation.

Plan:

- keep this as a separate branch or feature flag after reference parity,
- add exact conservation tests in periodic and closed systems,
- compare against the reference-formulated branch on matched cases.

### 10C. FCI and X-point-first geometry branch

Motivation:

- Hariri 2013/2014, Stegmeir 2018, and Wiesenberger 2023 point to a robust extension path for X-points, separatrices, and non-axisymmetric geometry without flux-coordinate singularities.

Plan:

- retain the reference field-aligned path as the baseline solver,
- add an FCI geometry/operator backend behind a separate geometry interface,
- start with operator verification and TORPEX/blob-style benchmarks before full tokamak use.

### 10D. Whole-volume and arbitrary-equilibrium workflows

Motivation:

- Giacomin 2022 extends GBS to whole-volume core-edge-SOL, arbitrary diverted equilibria, and faster elliptic solvers.

Plan:

- extend mesh and source handling to whole-volume geometry,
- add validation on single-null, double-null, and snowflake-like equilibria after tokamak baseline parity.

### 10E. Stellarator branch

Motivation:

- Coelho 2024 and Jorge 2021 indicate a realistic long-term fluid extension path to stellarator turbulence and near-axis geometry workflows.

Plan:

- add non-axisymmetric metric support after tokamak parity,
- start with reduced operator verification and then island-divertor stellarator turbulence cases.

## Time Integration and Solver Parity Program

The time integrator must not be the first source of disagreement. Solver parity therefore has to be staged.

reference solver facts to preserve:

- reference development focuses on `cvode` for accurate transient/turbulent runs and `beuler` for stiff steady-state transport runs.
- Both are adaptive-timestep implicit solvers.
- Many integrated tests and examples still use legacy solver settings such as `pvode`; those cases should be treated as physics references first, not as integrator references.
- the staged neutral transient case has now been traced live: it runs with `cvode` using `BDF` and `gmres`, with `rtol = 1e-5`, `atol = 1e-12`, and `mxstep = 1000`.

JAX solver implementation order:

1. pure `rhs(state, t, config)` parity with no time stepping
2. fixed-step single-step wrappers:
   - explicit Euler for debugging only
   - SSP-RK2 or SSP-RK3 for operator/physics smoke tests
   - one-step backward Euler residual form for implicit debugging
   - keep a validated matrix-free implicit fallback while the sparse direct path is being hardened, so transient work can keep moving without masking the sparse-solver development state
3. production steady-state path:
   - backward Euler + Newton/GMRES, matching the role of reference `beuler`
4. production transient path:
   - adaptive implicit multistep path, matching the role of reference `cvode`
   - for the neutral branch, prioritize a BDF-like Newton/GMRES path over any tuned explicit or IMEX workaround because the live reference case already identified the exact solver family in use
   - use the active-domain Jacobian sparsity implied by the local neutral `x/y/z` stencil and cross-field coupling, so sparse implicit prototypes preserve the same coupling pattern as the reference operators
   - direct low-level SciPy BDF probes with that sparsity are still too slow for production parity on the staged neutral case, so the next transient implementation should move toward a more direct CVODE-like sparse implicit path rather than a generic `solve_ivp` wrapper
   - current in-tree status: sparse grouped difference-quotient Jacobians, sparse Newton/GMRES substrate, and BDF2 residuals are implemented for the neutral branch, but the default validated step path still uses the matrix-free nonlinear solve because sparse globalization is not yet parity-clean
   - `diffrax` may be used only if it gives a clean mapping to the traced reference behavior; otherwise implement a custom JAX adaptive path

Solver parity rules:

- first prove one-RHS equality,
- then prove one-step equality at the same fixed timestep,
- only then compare adaptive trajectories,
- do not use adaptive-step disagreement as evidence of physics disagreement until the fixed-step one-step harness is green.

Mandatory solver tests:

- exact one-RHS parity tests for every selected reference case in the case ladder below
- one-step fixed-`dt` tests on the same state for explicit and implicit wrappers
- Newton residual convergence tests for backward Euler on 1D steady-state cases
- adaptive-step reproducibility tests with fixed tolerances and deterministic norms
- stiffness tests on recycling and conduction cases
- differentiability tests through one explicit step and one implicit step
- CPU/GPU reproducibility tests at the solver-wrapper level

Solver-specific publication figures:

- one-step error versus field/component for representative 1D, 2D, and 3D cases
- timestep history and nonlinear iteration history for matched reference/JAX transient and steady-state runs
- compile time, wall time, and memory plots for CPU and GPU

## Geometry, Normalization, and Equation Reference Program

These items need their own tracked deliverables because they determine whether outputs, diagnostics, and documentation are trustworthy.

### Geometry parity requirements

Support order:

1. identity/slab geometry used by the simplest integrated tests
2. shifted field-aligned geometry used by blob, tokamak-2D, and tokamak-3D cases
3. metric loaded from the grid file
4. metric recalculated from `Rxy`, `Bpxy`, `Btxy`, `hthe`, `sinty`

Geometry functions that require direct parity tests:

- metric loading and metric normalization from the grid file
- `recalculate_metric(...)` formulas
- guard-cell conventions with two Y guard cells and midpoint boundary values
- field-aligned communication and shifted transforms
- cell-volume, face-area, and `g_22` handling in FV parallel operators
- non-orthogonal perpendicular operator path used by vorticity and related terms

Geometry tests:

- synthetic identity-metric tests with closed-form expected coefficients
- recalculate-metric tests against the formulas in `recalculate_metric.cxx`
- loaded-metric versus recalculated-metric comparisons on orthogonal grids
- shifted-transform round-trip tests
- guard-aware transform tests at domain boundaries
- operator MMS on orthogonal and non-orthogonal meshes
- tokamak-grid smoke tests using representative Hypnotoad/BOUT grids

### Normalization parity requirements

reference base normalization:

- `Tnorm` in eV
- `Nnorm` in `m^-3`
- `Bnorm` in T
- `Cs0 = sqrt(qe * Tnorm / Mp)`
- `Omega_ci = qe * Bnorm / Mp`
- `rho_s0 = Cs0 / Omega_ci`

Normalization deliverables:

- a single JAX normalization module that reproduces the reference `units` tree exactly
- a reference page documenting every derived quantity and every conversion factor used in outputs
- tests that compare both internal normalized values and emitted SI conversion metadata

At minimum the docs and code must make explicit the following output conversions:

- density: `Nnorm`
- time: `1 / Omega_ci`
- velocity: `Cs0`
- length: `rho_s0`
- pressure / thermal energy density: `qe * Tnorm * Nnorm`
- density source: `Nnorm * Omega_ci`
- pressure/energy source: `qe * Tnorm * Nnorm * Omega_ci`
- momentum density: `Mp * Nnorm * Cs0`
- momentum source / force density: `Mp * Nnorm * Cs0 * Omega_ci`
- diffusion/viscosity coefficients: `rho_s0^2 * Omega_ci`
- electrostatic potential: `Tnorm`
- vector potential: `Bnorm * rho_s0`

### Equation reference requirements

The code and docs must track equations at three levels:

1. high-level model equations exactly as configured by components
2. discrete numerical operators actually implemented
3. reduced/derived analytical results used for validation and interpretation

Documentation deliverables:

- a component-by-component equation reference mirroring the reference organization
- a discrete-operator reference page with stencil and flux formulas
- derivations for:
  - Bohm normalization
  - guard-cell midpoint conventions
  - FV parallel fluxes and limiter choices
  - sheath particle and heat flux formulas
  - drift-reduced vorticity / potential equations
  - key validation scalings used later in the benchmark program

## Selected reference Comparison Case Ladder

Comparison cases must be staged by both physics complexity and runtime length.

### Tier A. One-RHS and one-step parity cases

Use these first because they isolate field semantics and operator correctness:

- `local source checkout/tests/integrated/evolve_density/data/BOUT.inp`
  - role: single evolving density equation
  - compare: `N`, `ddt(N)`, density sources, normalization metadata
- `local source checkout/tests/integrated/diffusion/data/BOUT.inp`
  - role: density + pressure + anomalous diffusion
  - compare: `N`, `P`, diffusion terms, source bookkeeping
- `local source checkout/tests/integrated/vorticity/data/BOUT.inp`
  - role: electrostatic field solve only
  - compare: `Vort`, `phi`, vorticity RHS pieces
- `local source checkout/tests/integrated/neutral_mixed/data/BOUT.inp`
  - role: compact neutral model check
  - compare: neutral density/pressure/velocity/momentum outputs
  - implementation note: lock RHS and evolved-state diagnostics first, including the isolated parallel density term, parallel advective flows, and the `g22` / `g_22` metric semantics from the diagnosed reference dump at probe `(x=5, z=5)`, then compact transient metrics plus total neutral mass/pressure and momentum-RMS decay, and only then promote public transient parity with a stiff solver strategy that follows the traced `cvode`/BDF/GMRES reference path; current explicit RK scaffolding is not sufficient and overflows on the short-window benchmark.

### Tier B. Short transient parity cases

These should run for a few outputs only and are the first real time-trajectory checks:

- `local source checkout/tests/integrated/1D-fluid/data/BOUT.inp`
  - role: density + pressure + momentum in 1D
  - compare: `Ni`, `Pi`, `NVi`, derived `Vi`, shock/transport behavior
- `local source checkout/tests/integrated/neutral_parallel_diffusion/data/BOUT.inp`
  - role: projected neutral transport and collision-frequency dependence
  - compare: neutral fluxes, source terms, collision diagnostics
- `local source checkout/tests/integrated/drift-wave/data/BOUT.inp`
  - role: vorticity + collisions + friction + sound speed
  - compare: `Ne`, ion momentum, `phi`, `fastest_wave`, collisional source terms
- `local source checkout/tests/integrated/2D-energy/data/BOUT.inp`
  - role: 2D pressure/vorticity/polarisation coupling
  - compare: `Pe`, ion pressure, `phi`, `Vort`, polarisation diagnostics
- `local source checkout/tests/integrated/alfven-wave/data/BOUT.inp`
  - role: minimal electromagnetic path
  - compare: `Apar`, `phi`, momentum, wave phase speed/damping

### Tier C. Medium-length parity and source/diagnostic cases

These establish that sources, boundaries, and diagnostics match:

- `local source checkout/tests/integrated/1D-recycling/data/BOUT.inp`
  - role: sheath + recycling + reactions + collisions
  - compare: upstream profiles, target fluxes, selected reaction diagnostics (`K`, `S`, `E`, `F`, `R`)
- `local source checkout/tests/integrated/1D-recycling-dthe/data/BOUT.inp`
  - role: multi-species recycling with electrons, deuterium, tritium, helium
  - compare: multi-species sources, target conditions, power balance
- `local source checkout/examples/other/blob2d/BOUT.inp`
  - role: electrostatic isothermal blob baseline
  - compare: blob propagation, peak amplitude, COM trajectory, and long-window summary statistics
- `local source checkout/examples/other/blob2d-vpol/BOUT.inp`
  - role: blob with polarisation drift
  - compare: blob speed and morphology shift relative to baseline
- `local source checkout/examples/other/blob2d-te-ti/BOUT.inp`
  - role: two-temperature blob
  - compare: coupled pressure and field dynamics

### Tier D. Long-run and publication-grade parity cases

These become the main journal and docs examples after the earlier tiers are stable:

- `local source checkout/examples/tokamak-2D/recycling/BOUT.inp`
  - role: 2D tokamak recycling with `cvode`
  - compare: profiles, target diagnostics, reaction and recycling source fields
- `local source checkout/examples/tokamak-2D/recycling-dthe/BOUT.inp`
  - role: multi-species 2D tokamak edge/divertor physics
  - compare: multi-species profiles, wall/target fluxes, reaction diagnostics
- `local source checkout/examples/tokamak-2D/turbulence/BOUT.inp`
  - role: turbulent 2D tokamak transport
  - compare: time traces, RMS fluctuation levels, spectra, PDFs, zonal quantities
- `local source checkout/examples/tokamak-3D/tcv-x21/data/BOUT.inp`
  - role: representative 3D tokamak benchmark
  - compare: 3D profiles, fluctuation statistics, selected cross-sections, runtime scaling

Case-ladder execution policy:

- every Tier A case must support:
  - static loaded-state comparison
  - one-RHS comparison
  - one-step fixed-`dt` comparison
- Tier B and above must additionally support:
  - short-window trajectory comparison
  - output/diagnostic metadata comparison
- Tier C and D must additionally support:
  - integrated balance checks
  - plotting and post-processing parity

## Output, Diagnostic, and Plot Parity Plan

reference parity is not complete until the outputs are compatible enough that the same analysis habits work on both codes.

### File-level compatibility targets

JAX outputs should eventually provide equivalents of:

- `BOUT.settings`
- `BOUT.log.*`
- `BOUT.dmp.*`
- `BOUT.restart.*`

JAX does not need to reproduce MPI file sharding exactly on day one, but it must preserve the semantic contents and metadata.

### Variable-level compatibility targets

For every output variable documented or emitted by reference, preserve:

- variable name
- dimensions and guard-handling conventions where relevant
- `units`
- `conversion`
- `standard_name`
- `long_name`
- `species`
- `source`
- whether it is time-evolving or saved once

Implementation tasks:

- build an auto-generated output registry by scanning `outputVars(...)`, `restartVars(...)`, and `set_with_attrs(...)` usage in reference
- classify variables into:
  - state fields
  - derived fields
  - sources
  - flows
  - reactions
  - geometry/normalization metadata
- add a test that fails if a supported component emits a variable whose name or metadata disagrees with the registry

### Plot and analysis parity targets

Reproduce a standard set of analysis products for both reference and JAX:

- 1D profile overlays
- 2D poloidal and cross-field planes
- target and wall flux traces
- RMS fluctuation time traces
- spectra
- PDFs
- zonal-flow and zonal-profile plots
- blob COM trajectories and morphology panels
- integrated particle, momentum, and energy balance plots

The same plotting scripts should operate on reference and JAX outputs via a thin adapter layer. The goal is to allow a paper or docs figure to be generated from either code with only the dataset path changed.

## Research-Grade Benchmark and Validation Program

The benchmark program must satisfy three different review questions:

1. Is the implementation correct?
2. Does it reproduce the reference code?
3. Does the reproduced code capture the expected physics?

### Verification layer

This is purely mathematical/numerical correctness.

Required items:

- unit tests for every boundary rule, parser, normalization rule, and source term
- MMS convergence plots for all major operators
- discrete conservation tests where applicable
- one-RHS and one-step parity error norms
- solver convergence diagnostics
- CPU/GPU numerical reproducibility checks
- differentiability tests with finite-difference or complex-step cross-checks on small cases

### reference parity layer

This proves that JAX reproduces reference, not just a plausible DRB model.

Required items:

- Tier A-D case ladder above
- per-field relative/absolute norms
- per-diagnostic comparison tables
- trajectory comparison for short/medium windows
- long-run comparison on reduced sets of summary statistics where chaotic divergence is expected
- archived used-input files and exact run settings for both reference and JAX

### Physics validation layer

This ties the reproduced code to accepted behavior in the literature.

Validation topics to include after parity is stable:

- SOL instability/regime transitions informed by Mosetto 2012 and Halpern 2013
- SOL width / pressure decay trends informed by Giacomin 2022 and Lim 2023
- sheath/equilibrium potential behavior informed by Loizu 2012 and Loizu 2013
- open-field turbulence properties and blob propagation informed by Ricci 2012, Halpern 2016, and related blob benchmarks
- detachment scaling checks informed by Body 2024
- whole-volume / arbitrary-equilibrium extensions benchmarked later against Giacomin 2022
- stellarator extension benchmarks reserved for the post-parity branch, informed by Coelho 2024

### Journal-grade benchmark bundle

For a future JCP paper, prepare a reproducible artifact bundle containing:

- exact reference and JAX input files
- exact grids and restart snapshots where needed
- scripts for one-RHS, one-step, short-window, and long-run comparisons
- MMS scripts and raw convergence data
- CPU/GPU performance benchmark scripts
- plotting scripts for all paper figures
- a machine-readable manifest of code version, dependency versions, and hardware

### Candidate JCP figure set

- architecture and dataflow diagram
- normalization and geometry conventions diagram
- MMS convergence figure for core operators
- one-RHS parity heatmaps for representative cases
- one-step and short-window parity traces
- 1D recycling parity profiles and diagnostic table
- 2D blob and turbulence comparison panels
- electromagnetic `Apar` benchmark figure
- CPU/GPU performance and memory figure
- differentiability demo figure showing gradients of a scalar QoI

## Documentation Program

The documentation needs to be useful simultaneously for:

- new users,
- developers,
- reviewers,
- future maintainers,
- readers of a validation or methods paper.

The documentation structure should take inspiration from the strengths of WarpX:

- broad landing page with clear entry points
- exhaustive parameter reference
- curated examples/tutorials
- theory and algorithms pages
- data-analysis/output pages
- development and API sections

Reference inspiration:

- WarpX landing page: [https://warpx.readthedocs.io/en/latest/](https://warpx.readthedocs.io/en/latest/)
- WarpX input parameter reference: [https://warpx.readthedocs.io/en/latest/usage/parameters.html](https://warpx.readthedocs.io/en/latest/usage/parameters.html)
- WarpX examples page: [https://warpx.readthedocs.io/en/latest/usage/examples.html](https://warpx.readthedocs.io/en/latest/usage/examples.html)

### Documentation platform and UX

Recommended stack:

- Sphinx + MyST Markdown
- `pydata-sphinx-theme` or an equivalent documentation-first theme with a persistent sidebar and version switcher
- MathJax for derivations
- `sphinx-design` for callouts/cards/tabs
- auto-generated API docs from docstrings
- searchable reference pages with stable anchors
- versioned docs on Read the Docs

UX requirements:

- every page must tell the reader where they are in the hierarchy
- every parameter, diagnostic, component, and equation must have a stable anchor
- code and paper references must be clickable
- example pages must link directly to the exact input files and plotting scripts
- figures must have short interpretation captions, not just screenshots

### Required documentation sections

1. Home
   - what `jax_drb` is
   - what physics it supports today
   - quick links for install, run, examples, theory, outputs, validation
2. Quickstart
   - installation
   - first run from CLI
   - first run from Python
   - first plot
3. User guide
   - running simulations
   - restarting
   - selecting geometry
   - choosing solvers
   - reading outputs
4. Inputs reference
   - every supported `BOUT.inp` option grouped by section/component
   - defaults, units, normalization, allowed values
   - exact reference compatibility notes
5. Outputs and diagnostics reference
   - every emitted variable
   - metadata meanings
   - derived diagnostic families
   - example plots and analysis recipes
6. Physics model and equations
   - high-level equations by component
   - closures, source terms, sheath, reactions
   - derivations of the main formulas and analytical scalings
7. Numerics and algorithms
   - time integration
   - field-aligned and shifted geometry handling
   - FV operators, limiters, elliptic solves, preconditioning
   - differentiability notes
8. Geometry guide
   - identity/slab
   - shifted tokamak grids
   - metric loading vs recalculation
   - later FCI and stellarator branches
9. Examples and tutorials
   - one page per selected Tier A-D case
   - expected outputs and interpretation
   - links to source files and plots
10. Verification and validation
   - MMS results
   - reference parity results
   - literature validation cases
   - performance results
11. Developer guide
   - repository layout
   - adding a component
   - adding a diagnostic
   - adding a test
   - style and contribution rules
12. API reference
   - Python API
   - CLI commands
13. Publications and references
   - reference, GBS, GRILLIX, FCI, conservative DRB, detachment, validation papers

### Documentation generation from source

The docs should not rely only on handwritten reference pages.

Auto-generated pages should be built from:

- the supported input schema
- the output/diagnostic registry
- the parity matrix
- the benchmark manifest
- docstrings in the Python API

### Documentation figure plan

Produce figures that are both explanatory and reusable in papers:

- normalization ladder diagram
- grid/guard-cell diagrams
- shifted field-aligned geometry diagrams
- operator stencil figures
- solver workflow diagrams
- example result galleries
- benchmark summary dashboards
- validation scaling plots

### Documentation acceptance criteria

The docs are complete only when a new user can:

- install the code,
- run a simple case,
- understand the inputs,
- find the meaning and SI units of every output variable,
- locate the exact equations and algorithms used,
- reproduce the published benchmark and validation figures.

## Benchmark and Validation Ladder

The long-term benchmark suite should be kept in this exact order:

1. local unit/operator tests
2. MMS operator tests
3. 1D fluid and shock/conduction tests
4. 1D sheath/recycling/detachment tests
5. 2D electrostatic diffusion and drift-wave tests
6. blob benchmarks
7. 2D recycling/transport tests
8. 3D electromagnetic tests
9. multi-species neutral/reaction tests
10. selected full examples such as `tcv-x21`
11. literature-driven validation cases:
    - SOL width and regime scalings
    - triangularity trend
    - density/beta limit trends
    - detachment onset scaling

No later benchmark tier should become a gating CI requirement until all earlier tiers are stable.

## Done Definition

`jax_drb` can be considered a reference solver JAX replica only when all of the following are true:

- the component scheduler and input semantics match reference on representative examples,
- all core operator families pass MMS and regression tests,
- the 1D, 2D, and selected 3D reference cases run from CLI and Python,
- multi-species neutrals/reactions work on representative reference examples,
- diagnostics and normalization conventions are preserved,
- production paths are differentiable,
- CPU and GPU runs use the same code path,
- the parity matrix shows no unimplemented reference runtime component required by the supported examples.

Until then, new work should prioritize parity gaps over new physics.
Current Step 2 checkpoint:

- `recycling_1d_rhs` is now native, regression-tested, and locked against committed summary and full-array baselines.
- `recycling_dthe_rhs` is now also native and live-reference clean at the committed summary tolerances, with the exact multi-species collision-table ordering and cross-isotope charge-exchange bookkeeping now traced into the public implementation.
- the native 1D recycling stack now also includes the upstream density-feedback controller source path and controller-integral auxiliary state, with focused regressions on the single-species zero-feedback start and the multi-species helium feedback multiplier.
- the transient substrate around that RHS now exists in-tree on the shared implicit backbone, including packed active-domain field/controller state, backward-Euler residual wiring, and adaptive BDF probes with grouped sparse finite-difference Jacobians.
- the dense Step 2 diagnosis workflow is now more explicit:
  - [diagnose_recycling_target_cell_history.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_target_cell_history.py) tracks target-adjacent cell histories and currently shows the earliest visible single-species drift appearing first in `NVd+` at `t = 25`, then `Nd`, then `Nd+` / `Pe`;
  - [diagnose_recycling_controller_history.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_controller_history.py) tracks controller/source histories and currently shows that the PI-source drift is real but smaller than the target-cell state drift, so Step 2 is not blocked on the controller source in isolation;
- a more reference-like mutable-controller `bdf` callback path now exists in-tree and is materially faster on the single-species one-step probe, but it still lands at essentially the same first-output state error, so the remaining Step 2 defect survives both the continuation ladder and the faster BDF callback variant.
- The next Step 2 jump is no longer the RHS source stack; it is the transient stack: `recycling_1d_one_step`, `recycling_dthe_one_step`, then the corresponding short-window/long-run open-field cases on the shared implicit backbone.
- current transient status: the controller-complete recycling one-step branch is not yet parity-clean enough to expose, and the current adaptive BDF probe is still too slow for reviewer-safe production use on the staged one-step cases.
- Step 3 still depends on restaging a stable 2D recycling geometry target, because the currently named tokamak example remains blocked on the reference side.
  - `braginskii_thermal_force`
  - `braginskii_ion_viscosity`
- Step 3 should not continue from the broken tokamak example. The next curated 2D recycling target should be staged from the integrated `2D-recycling` workflow with its external artifact bundle handled explicitly by the harness.
