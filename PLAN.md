# jax_drb Plan

Date: 2026-04-09

## 1. Mission

Build `jax_drb` as a standalone JAX edge/SOL plasma code with a private parity track against an external reference implementation:

- same physics model and component ordering semantics during the parity buildout,
- same normalization, boundary, diagnostic, and restart conventions where practical,
- runnable from both a Python API and a native `jax_drb` CLI,
- CPU/GPU portable with a pure-JAX runtime path,
- end-to-end differentiable through production solver paths,
- minimal runtime dependencies.

## 1A. Research-Grade Reset (2026-04-09)

The project is now explicitly centered on a strong-subset publication claim, not on treating every staged parity rung as equal evidence.

Locked defaults:

- paper scope: strong subset first
- differentiability: architectural requirement, but not a blocker for the first publication-grade release

Capability tiers are now part of the manifest, CLI, and run logs:

- `native_exact`
  - fully native solve path
  - clean enough to anchor a public parity claim
- `native_operational`
  - native path with bounded residuals
  - useful internally and for research iteration, but not headline evidence
- `scaffolded_reference_backed`
  - replay/dump/history-assisted path
  - diagnostic only and must not be presented as equivalent to native closure

Immediate strategy change:

1. stop treating more dump-backed ladders as the main progress metric
2. finish one fully native open-field recycling transient lane end to end
3. reuse that backbone for integrated and direct-tokamak recycling/production lanes
4. widen the matrix only after that native closure is stable

Fast validation policy:

- the default local research gate is now the curated fast slice runner in `scripts/run_fast_research_checks.py`
- each slice has a hard 5-minute timeout by default
- broad raw `pytest -q` sweeps are still useful, but they are no longer the default iteration loop
- new operator work should land with a focused slice in that runner rather than relying on an unmanaged long-tail suite invocation
- longer transient-history solver checks should be marked `slow` and kept out of the default fast gate unless they are the active target of the current pass
- the reviewer-facing convergence lane now starts from a reproducible manufactured-solution script, `scripts/run_fluid_1d_mms_convergence.py`, which reports refinement errors and observed order on the native 1D fluid operator path
- the local open-field recycling ladder now includes `recycling_1d_short_window` (`nout=5`) as the first repeated-output transient rung for the native backbone
- the standalone runtime now accepts `[runtime.logging].verbose = true|false` in TOML and the same detail switch is exposed through `jax_drb --verbose` and `run_input_case(..., verbose=True)`
- the detailed runtime stream now also forwards interval-level `progress` events from the native recycling transient backbone into the CLI, so long implicit recycling steps no longer look hung while accepted intervals are being advanced
- the verbose run-log JSON now stores `event_count` and `event_stages` in addition to the ordered `events` list, which makes downstream audit checks cheaper than replaying the full payload
- the same runtime layer now also exposes `[runtime].recycling_transient_solver_mode = "continuation" | "bdf" | "adaptive_be" | "adaptive_bdf"` so the open-field recycling one-step blocker can be swept from a deck or Python driver without patching source code
- the compact native diffusion lane now also carries the first publication-oriented autodiff/scaling artifact package:
  - `examples/autodiff_diffusion_sensitivity_demo.py`
  - `examples/autodiff_diffusion_inverse_design_demo.py`
  - `examples/strong_scaling_diffusion_demo.py`
  - committed figures and JSON payloads under `docs/data/*autodiff*` and `docs/data/strong_scaling_diffusion_artifacts`
- current committed differentiable artifact results:
  - inverse-design objective drops from about `2.95e-3` to about `5.52e-5`
  - autodiff and finite-difference gradients match closely on the compact four-parameter sensitivity study
  - GPU strong scaling on the fixed differentiable workload is now about `2.19x` from `1 -> 2` GPUs on the office host
  - the local CPU curve remains a modest single-node process-parallel reference and should not be oversold in the paper text

The current highest-probability live mismatch remains the D/T tokamak recycling transient, but the blocker is now split more honestly:

- the lower-target-corner D/T mismatch was materially reduced by reconstructing the missing lower neutral guard state before charge-exchange / viscosity closure;
- the remaining worst one-step residual now sits on the opposite active edge of the local slab, where this rank does not own an upper physical target;
- that upper-side guard row is therefore a communicated neighbor state, not a local sheath boundary, so the current direct tokamak dump-backed lane cannot be promoted by more target-boundary heuristics alone.

That means the critical path is still native recycling operator closure, but it also means the direct tokamak D/T one-step rung should stay explicitly operational/scaffolded until the fully native open-field recycling backbone carries its own distributed guard-state evolution.

## 1B. v1.0 Release Hardening (2026-04-10)

The v1.0 release surface is now being hardened around a standalone `jax_drb` identity:

- top-level docs and examples should teach TOML-driven `jax_drb` usage first
- benchmark-suite comparisons stay in the documentation and validation tooling, not in the primary user story
- committed public artifacts must avoid workstation-specific absolute paths
- runtime logs must keep provenance while sanitizing host-specific filesystem locations
- capability tiers must stay visible in manifests, CLI output, run logs, and reviewer-facing tables
- performance and differentiability claims must be explicit about which lanes are already strong and which still depend on NumPy/SciPy-heavy transient machinery

Latest blocker evidence on that lane:

- the post-boundary electric-force density consistency fix is now landed in the native recycling path:
  - `electron_epar` uses boundary-conditioned electron density
  - ion electric-force deposition uses boundary-conditioned ion density
- that change is safe and test-covered, but it does not materially change the `tokamak_recycling_dthe_one_step` residual ordering
- the native full `sheath_boundary` electron branch has now also been corrected to follow the Hermes full-sheath semantics more closely:
  - `[sheath_boundary]` defaults are now loaded explicitly on the native path
  - zero-current `phi` now includes `sin_alpha`, `(1 - Ge)`, `wall_potential`, and `floor_potential`
  - lower-boundary `vesheath` / `gamma_e` and electron energy-source signs now match the Hermes full-branch formulas
- that full-sheath correction is also safe and unit-tested, but it likewise does not materially change the `tokamak_recycling_dthe_one_step` residual ordering
- a direct probe of Neumann-guarding the ion-viscosity coefficient `eta` before `DivPiPar` was tested and rejected because it worsens the D/T one-step lane, especially `NVhe+`
- the remaining blocker is therefore still the sheath-conditioned lower-target-corner `DivPiPar` boundary state/operator, not the electric-force density path
- a direct Hermes-vs-native collision diagnostic now also shows that the Coulomb `K*_coll` inputs at the bad D/T cells already match to roundoff, so the remaining gap is downstream of `_compute_collision_frequencies`: either charge-exchange collisionality or the boundary-conditioned viscosity stencil/state itself
- the newer blocker pass also shows that the lower-target-corner part of that story is no longer the whole problem: once the lower neutral guard is reconstructed, the worst surviving `tokamak_recycling_dthe_one_step` residual moves to the upper active row on a side where `mesh.has_upper_y_target` is false. On that side the missing guard row is a communicated neighbor state, not a local sheath boundary.
- the current one-step path no longer depends on committed communicated-guard replay inside the active tokamak recycling solve. On the promoted compare surface, `tokamak_recycling_dthe_one_step` now clears a mixed exact gate: the non-negligible ion/electron fields stay below a `5e-2` scaled diff band while the near-zero neutral channels remain inside a small absolute band. The remaining broader blocker is therefore no longer first-output parity itself, but extending that same native behavior to richer direct tokamak recycling windows and eventually to fully native distributed guard evolution.
- the new bounded mode-sweep path for open-field one-step recycling is now explicit: [compare_recycling_transient_modes.py](scripts/compare_recycling_transient_modes.py) compares `continuation`, `bdf`, `adaptive_be`, and `adaptive_bdf` directly against the committed `recycling_*_one_step` baselines using the curated deck overrides. That is now the intended gate before changing the transient controller again.
- the first bounded sweep on `recycling_1d_one_step` now closes the solver-choice question enough to move on: `continuation` takes about `66.7 s` and lands at about `Nd+ ≈ 1.878e-2`, `Pd+ ≈ 1.736e-2`, `NVd+ ≈ 1.723e-2`, while `bdf` takes about `34.5 s` and only shifts that to about `Nd+ ≈ 1.887e-2`, `Pd+ ≈ 1.366e-2`, `NVd+ ≈ 1.383e-2`. So `bdf` is faster and modestly helps momentum/pressure, but it does not remove the leading density miss. A bounded `adaptive_bdf` probe also failed to finish within the five-minute local gate, so the next fix should stay on active transient/controller closure rather than solver proliferation.
- the implicit recycling residual no longer applies a second trapezoid advance to density-feedback integrals when those integrals are already part of the packed solve state. That was a real correctness bug and is now unit-locked, but bounded one-step compares show that it does not materially change the leading `recycling_1d_one_step` density miss. The same bounded probe pass also ruled out three other plausible startup-only explanations as first-order blockers on that lane:
  - shrinking the continuation startup `dt` from `100` to `25` improves `Pd+`/`NVd+` slightly but leaves the leading `Nd+` miss essentially unchanged;
  - the open-field one-step runner itself had a separate semantics bug: it was seeding the transient from a synthesized final active-state template instead of explicitly pinning the initial RHS snapshot as the start state. That is now fixed and unit-locked, so the lane is at least measuring an honest one-step transient again;
  - forcing sanitized fields inside the implicit residual changes only roundoff-level details.
  So the current Step 2 blocker remains the active transient evolution itself near the target-adjacent cell, not another controller bookkeeping bug or a simple startup-parameter choice.
- the next continuation-controller pass is the first one that materially improves the open-field lane instead of just ruling things out: the first output interval on open-field recycling runs now uses a startup warmup that splits the first `25` time units into `4 x 6.25` sparse implicit substeps before the usual continuation cadence resumes. That change is unit-locked and materially tightens the `timestep = 25` upper-cell blocker on `recycling_1d_one_step`, reducing the local drift from about `Nd+ ≈ 2.13e-2`, `Pd+ ≈ 2.03e-2`, `NVd+ ≈ 4.05e-2`, `Pe ≈ 2.13e-2` down to about `Nd+ ≈ 5.78e-3`, `Pd+ ≈ 5.28e-3`, `NVd+ ≈ 1.56e-3`, `Pe ≈ 5.72e-3`; the same warmup also materially tightens the multispecies `recycling_dthe_one_step` short-step probe at the corresponding upper active cell (`Nd+ ≈ -3.45e-3`, `Pd+ ≈ -8.36e-3`, `NVd+ ≈ -2.20e-3`, `Pe ≈ -1.98e-3`). Combined with the latest full one-step scaled compares, that is enough to promote both open-field one-step lanes to `native_exact` on their committed compare surfaces (`relative_to_expected_max < 5e-2`), while the longer recycling windows remain operational.
- the next bounded solver-side pass is narrower now too: forcing the continuation interval to stay backward-Euler-only slightly improves `Nd+` but makes `Pd+`, `NVd+`, and `Pe` worse, so the BE-to-BDF2 handoff is not the main blocker. Adding an explicit-RHS startup predictor for the sparse BE/BDF2 Newton solve is now unit-locked and materially tightens `Pd+` / `NVd+` on the committed one-step compare (`Pd+ ≈ 1.365e-2`, `NVd+ ≈ 1.383e-2`), but it still leaves the leading density miss at `Nd+ ≈ 1.888e-2`. A fresh full-interval `adaptive_be` probe still times out under the five-minute local gate even after that predictor improvement, so the next fix still belongs in the active transient evolution itself rather than another solver-startup variant.
- the new cell-local short-step diagnostic now pinpoints that active transient surface more concretely: at the worst upper target-adjacent active cell on the `timestep = 25` probe, the native state drift is paired with `Ve` low by about `4.07e-2`, `Epar` low by about `4.76e-4`, and `ddt(NVd+)` high by about `4.96e-3`, while `SNVd+`, `SNVd`, and `Sd_target_recycle` stay in the `1e-4` band. The deeper ion-term split now shows that the dominant local miss is the ion `momentum_advection` term itself: on the same cell it is about `-3.63e-3` on the reference-evolved state versus `+1.71e-3` on the native-evolved state, which accounts for essentially the full `ddt(NVd+)` gap. So the next recycling pass should target upper target-adjacent transient momentum evolution rather than recycle source terms. The new in-tree tool is [diagnose_recycling_boundary_cell.py](scripts/diagnose_recycling_boundary_cell.py).
- a follow-on hybrid-state probe narrowed that momentum defect further: replacing only the upper-cell reference `NVd+` in the native short-step state moves the local `momentum_advection` term from about `+1.71e-3` to `-2.17e-3`, much closer to the reference `-3.63e-3`, while replacing only `Nd+` barely moves it (`+1.21e-3`). So the next source pass should stay on transient momentum/velocity evolution, not density transport.

## 1C. Concrete Finish Sequence (2026-04-13)

The remaining work is now ordered explicitly. The project should follow this sequence rather than widening more staged ladders opportunistically.

### Phase A. Final reference audit and operator lock

Purpose:

- make one last source-of-truth pass over the private reference implementation before more solver refactors land
- keep the parity-critical operator semantics tied to the actual documented equations and source ordering

Required reference surfaces:

- docs:
  - `docs/sphinx/boundary_conditions.rst`
  - `docs/sphinx/closure.rst`
  - `docs/sphinx/feedback_control.rst`
  - `docs/sphinx/solver_numerics.rst`
  - `docs/sphinx/examples.rst`
- source:
  - `src/sheath_closure.cxx`
  - `src/braginskii_collisions.cxx`
  - `src/braginskii_friction.cxx`
  - `src/evolve_momentum.cxx`
  - `src/upstream_density_feedback.cxx`

Exit criteria:

- every open recycling/transient mismatch has a named operator owner
- every operator owner has a matching diagnostic script and a focused fast-gate slice

### Phase B. Close the open-field native recycling transient backbone

Purpose:

- finish one truly native transient lane end to end
- stop treating dump-backed or communicated-guard replay as a substitute for native closure

Implementation order:

1. finish active transient/controller closure on `recycling_1d_one_step`
2. promote `recycling_dthe_one_step` using the same corrected backbone
3. promote `recycling_1d_short_window`
4. only then unlock `recycling_1d_long`

Required technical work:

- replace finite-difference Jacobian assembly with JAX linearization/JVP-driven residual derivatives on the promoted 1D recycling lane
- reduce or remove `np.asarray(...)` and host-copy barriers in the hot transient path
- move accepted-step state/history layout to a backend-stable packed form
- keep plotting/output/logging outside the hot residual path

Validation gate:

- one-RHS parity
- one-step parity
- short-window parity
- restart equivalence
- operator diagnostics for `DivPiPar`, sheath state, reactions, controller source
- bounded fast-gate slice under the local five-minute policy

### Phase C. Reuse the native backbone for 2D recycling lanes

Purpose:

- stop carrying integrated/direct recycling as mostly replay-assisted lanes

Implementation order:

1. integrated 2D recycling:
   - `integrated_2d_recycling_one_step`
   - `integrated_2d_recycling_short_window`
   - `integrated_2d_recycling_medium_window`
2. direct tokamak recycling:
   - `tokamak_recycling_one_step`
   - `tokamak_recycling_dthe_one_step`
   - `tokamak_recycling_dthe_drifts_one_step`
   - `tokamak_recycling_dthene_one_step`

Specific closure target:

- replace communicated-guard replay with native distributed guard-state evolution on the recycling transient backbone

Promotion rule:

- no direct tokamak recycling family is promoted beyond `native_operational` while communicated-guard replay remains in the active transient solve

### Phase D. Close the remaining 2D production and neutral breadth

Purpose:

- finish the strongest 2D paper matrix before moving to selected 3D claims

Implementation order:

1. close the remaining production-side `Pe` / neutral transient defects
2. finish non-orthogonal anomalous-diffusion support using the full metric payload including `g_23`
3. promote `neutral_mixed_one_step` and `neutral_mixed_short_window`
4. widen direct tokamak transport/turbulence only where the path is already native rather than replay-backed

### Phase E. Promote the selected 3D and EM publication lanes

Purpose:

- make a strong selected 3D/EM claim rather than a vague broad one

Implementation order:

1. keep the benchmark EM ladders exact and documented:
   - `alfven_wave_*`
   - `annulus_he_emag_*`
2. promote operator-isolated closures for:
   - `Apar`
   - `Apar_flutter`
   - `phi` / vorticity coupling
   - Boussinesq vs non-Boussinesq choices on the selected lane
3. finish a geometry-agnostic 3D infrastructure layer:
   - mesh and metric ingestion
   - field-history assembly across toroidal ranks
   - geometry-aware surface, probe, and target extraction
   - selected-field parity on compact compare surfaces
   - benchmark/workflow adapters layered above those primitives
4. keep the first benchmark adapters narrow and explicit:
   - `examples/tokamak-3D/tcv-x21`
   - traced-field-line / stellarator-style mesh adapters
5. only then widen to richer 3D statistics, runtime/performance claims, and broader benchmark families

Current checkpoint:

- the first benchmark adapter package is now in-tree with a committed preview movie bundle
- the current selected benchmark is TCV-X21, but it is now treated as an adapter that sits on top of reusable 3D geometry and diagnostics utilities, not as the defining architecture for the whole 3D program
- the next concrete 3D deliverables are therefore split in two tracks:
  - general infrastructure:
    - structured deck/input reports
    - mesh, solver, component, and compare-surface metadata
    - compact selected-field parity bundles
    - generic probe/target/profile extraction interfaces
  - benchmark adapters:
    - TCV-X21 observable families (`FHRP`, `LFS-LP`, `HFS-LP`)
    - traced-field-line / stellarator-style mesh support informed by Zoidberg-style metric workflows and `bsting_files`-style mesh bundles
- the current scaffold package already carries:
  - a benchmark validation contract
  - a shared observable report on the generic 3D adapter schema
  - a staged profile report and compact profile arrays
  - a publication-style profile summary figure
  - a reduced selected-field parity package for compact `Ne`/`Pe`/`phi` surfaces
- the reusable 3D diagnostics layer now includes a shared profile-report/NPZ/plot path, and the TCV-X21 scaffold consumes that shared layer instead of owning its own benchmark-specific implementation
- the TCV-X21 scaffold now also supports a real public benchmark-data mode:
  - the committed artifact bundle is generated from public `TCV_forward_field.nc`, `TCV_ortho.nc`, `snaps00000.nc`, and `vgrid.nc` files
  - the tokamak profile report is now built from the public benchmark observable record instead of only from a synthetic preview or private workdir
  - the tokamak movie/poster/snapshot bundle is now reproducible from the public sample geometry/snapshot files on the same artifact path
- the tokamak reduced selected-field gate now also supports a real public benchmark-data mode:
  - the committed compact parity bundle is generated from the public TCV-X21 sample benchmark-data root instead of only from two workdirs or a synthetic preview pair
  - the compact parity bundle now writes a benchmark-data report plus a shared observable report on the same public artifact path as the other 3D adapters
  - the current reproducible public path uses the benchmark bundle as the reference side and a deterministic derived candidate as the compact compare target
- the next 3D deliverables are now:
  - extend the first reduced native 3D selected-field rung beyond its initial compact tokamak short-window surface
  - widen the public artifact bundle with runtime/provenance summaries on each promoted native 3D rung
  - add a third geometry adapter/workflow so the 3D infrastructure is pressure-tested beyond tokamak plus traced-field-line
- the first reduced native 3D selected-field rung is now also in-tree:
  - `tokamak_native_selected_field` runs a promoted native tokamak one-step case on the compact `Ne`/`Pe`/`phi` surface
  - the artifact bundle now carries parity JSON/NPZ, a shared observable report, and a runtime/provenance report
  - this is still a reduced rung, not a full native 3D benchmark claim

- the first second-adapter scaffold is now also in-tree: a traced-field-line geometry bundle with metric reports, compact metric arrays, and a geometry-adapter validation contract, intended as the bridge from generic 3D diagnostics to later real stellarator/traced-field-line execution work
- that same traced-field-line adapter now also publishes a shared observable report for line and plane families, so benchmark-observable extraction is no longer trapped inside the tokamak benchmark adapter
- that second adapter now also emits reusable radial, toroidal, and poloidal line diagnostics from both synthetic preview specs and real external NetCDF FCI grids, so the generic probe/target extraction layer is no longer only a plan item
- that same adapter now also emits automatically selected radial/toroidal/poloidal plane summaries, compact slice arrays, and a geometry-family GIF on the real external NetCDF path, so the non-tokamak 3D lane has a real movie/figure workflow instead of only static metric summaries
- the non-tokamak adapter family now also has its first reduced parity gate: `traced_field_line_selected_field_parity` compares a compact metric-field surface and publishes `max|Δ|`, RMS, and relative-L2 errors plus an observable report on the shared geometry-adapter schema
- that selected-field gate now also runs from a real external traced-field-line reference input when an external FCI grid is available locally, deriving the candidate deterministically from that reference so the public bundle is no longer preview-only
- the next non-tokamak 3D step after that gate is to drive the same selected-field parity package from an independent traced-field-line reference/candidate pair instead of a reference-derived candidate
- that third geometry adapter is now also in-tree:
  - `stellarator_vmec_scaffold` consumes VMEC-style equilibrium data on the same generic 3D adapter schema
  - it publishes a manifest, input report, validation contract, equilibrium profile bundle, shared observable report, sampled `R`/`Z` flux-surface summary figure, and a toroidal-angle movie
  - the committed public bundle is generated from a deterministic VMEC-compatible synthetic equilibrium, while the regression surface also locks support for real `wout*.nc`-style inputs
- the next 3D deliverables after that are now:
  - add the first stellarator/traced-field-line selected-field parity gate driven by an independent external reference/candidate pair
  - widen the first native tokamak reduced rung beyond its initial one-step compact surface
  - keep runtime/provenance summaries on every promoted native 3D artifact bundle and on the shared observable schema

Publication rule:

- 3D/EM claims stay restricted to selected benchmark and reduced ladders until the fully coupled transient path is closed on those families
- no single benchmark geometry is allowed to define the public 3D architecture; every benchmark package must sit on reusable mesh, metric, diagnostics, and parity primitives

### Phase F. Run the reviewer-facing campaign set

Required campaign bundle:

1. operator-focused recycling / ion-viscosity campaign
2. direct tokamak convergence campaign
3. TORPEX seeded-blob validation package
4. TCV-X21 diverted L-mode benchmark package
5. detachment-scaling package
6. performance and memory campaign on promoted native paths
7. differentiable sensitivity / inverse-design / scaling package on promoted native paths

Each campaign must ship with:

- committed script entry points
- JSON analysis payloads
- publication-ready figures
- a short methods note in `docs/`

### Phase G. Draft the paper from the supported matrix, not the aspirational matrix

The first paper should claim:

- a research-grade, restartable, JAX-native edge/SOL code
- a clearly stated supported matrix
- exact or tightly bounded parity on that matrix
- explicit capability-tier labeling for anything still operational or scaffolded

The first paper should not claim:

- full parity on every tokamak and 3D workflow
- full differentiability on the heavy recycling backbone before the SciPy/FD barriers are removed
- fully native closure on any lane still depending on replayed guard-state or dump-backed transient state

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

The plan is based on the current external benchmark solver codebase, tests, examples, docs, and the local drift-reduced-Braginskii literature collection used during development.

Primary reference solver references:

- code: `local source checkout/src`
- headers: `local source checkout/include`
- docs: `local source checkout/docs/sphinx`
- unit tests: `local source checkout/tests/unit`
- MMS/operator tests: `local source checkout/tests/mms_operator`
- integrated tests: `local source checkout/tests/integrated`
- examples: `local source checkout/examples`

reference implementation facts that must be preserved early:

- ordered runtime configuration through legacy `BOUT.inp` parsing and native TOML decks with ordered `[model] components`
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

## 4A. Current Readiness Assessment (2026-04-01)

This section is the current decision point for the remaining port. It answers what `jax_drb` can already claim, what it cannot yet claim, and what must be finished before the code is ship-ready as a standalone DRB solver.

The five-step finish plan in [8A](#8a-five-step-finish-plan) is still the active execution plan. The older stage-by-stage sections later in this file remain useful as a detailed dependency map, but the project should now be driven by the five broad finish steps plus the ship blockers listed below.

### What Can Be Claimed Now

- Thorough comparison with the private reference implementation is already possible on a broad and growing curated ladder:
  - `one_rhs`
  - `one_step`
  - `short_window`
  - `medium_window`
  - compact physics/benchmark metrics where full arrays would be too large
- The parity harness is strong enough for reviewer-facing evidence on the currently locked benchmarks:
  - exact summary and array comparisons for the implemented/staged benchmark rungs
  - live rerun against the private reference binary
  - compact benchmark analysis for drift-wave, blob, and Alfvén-wave behavior
  - committed artifacts small enough to keep in-repo
- Electrostatic capability is already substantial on selected benchmark ladders:
  - diffusion
  - 1D fluid MMS
  - vorticity
  - drift-wave
  - blob
  - staged integrated 2D recycling/production lanes
- Electromagnetic capability now has a real benchmark ladder rather than a single smoke case:
  - `alfven_wave_rhs`
  - `alfven_wave_one_step`
  - `alfven_wave_short_window`
  - `alfven_wave_medium_window`
  - `annulus_he_emag_rhs`
  - `annulus_he_emag_one_step`
  - `annulus_he_emag_short_window`
- The numerical backbone is good enough to support the remaining port:
  - shared implicit/transient substrate
  - shared elliptic/inversion layer
  - portable summary/array parity tooling
  - JAX compilation cache
  - differentiability smoke checks on the major active branches

### What Cannot Yet Be Claimed

- `jax_drb` is not yet a fully standalone replacement covering the full intended DRB matrix in one production path.
- We cannot yet make a complete parity claim for:
  - full open-field plasma + neutrals + recycling transients at production output intervals
  - full 2D production/recycling transients on real tokamak geometry without staging shortcuts
  - full electromagnetic coupled transient evolution beyond the current benchmark ladders
  - selected reduced 3D tokamak electromagnetic runs through the main production path
  - full multi-species/reaction/impurity breadth
  - output/restart/log compatibility at the level needed for an external release and paper package
- We do not yet have a complete convergence story across the final intended workflow matrix. We do have benchmark-quality transient/convergence evidence on selected ladders, but not yet a final unified convergence/validation campaign across:
  - open-field recycling/detachment
  - broader 2D production turbulence
  - electromagnetic annulus/tokamak cases
  - multi-species neutral/reaction workflows

### Capability Matrix

- Open field lines:
  - partially ready
  - strong RHS/source parity and localized transient evidence exist
  - still blocked for a strict final parity claim by the remaining 1D recycling startup-transient defect
- Closed / periodic / annulus-like field lines:
  - benchmark-ready on selected electrostatic and electromagnetic ladders
  - not yet a general production claim for all closed-field workflows
- Bohm/sheath boundary conditions:
  - implemented on the active open-field path
  - not yet fully ship-ready because the long-interval open-field transient parity is still not fully closed
- Neutrals:
  - partially ready
  - `neutral_mixed` and recycling-related neutral paths are substantially traced
  - full neutral transient/reaction breadth is not yet done
- Electrostatic physics:
  - strong on selected benchmark ladders
  - still incomplete as a full tokamak production claim
- Electromagnetic physics:
  - benchmark-ready on Alfvén and annulus ladders
  - not yet a full coupled 3D/tokamak production claim
- Tokamak / analytic / other geometries:
  - analytic/slab/annulus and staged integrated 2D lanes are in good shape
  - direct reduced tokamak production geometry remains a main remaining blocker
- Cold / hot ions:
  - partial
  - enough slices exist to continue efficiently, but not yet broad release-level coverage
- Boussinesq / non-Boussinesq:
  - partial and not yet presented as a final public capability matrix
  - needs explicit final coverage/validation in the ship-ready pass
- Linear / nonlinear:
  - yes on selected ladders
  - not yet as a final broad code claim across all target geometries and closures
- Alternate boundary-condition matrix for the full DRB system:
  - not yet
  - still needs a curated final ladder and documentation pass

### Ship Blockers

The remaining blockers are now narrow enough to list explicitly:

1. Open-field transient parity is not fully closed.
   - Treat Step 2 as operationally complete for project flow, but not final for a paper/release claim.
   - The remaining issue is the accepted startup transient in 1D recycling, not broad RHS/source reconstruction.

2. The broader 2D production path is not yet parity-clean.
   - The integrated 2D production transient ladder exists.
   - The remaining gaps are localized, but they are still real and must be closed before claiming ship-ready 2D SOL capability.

3. Tokamak geometry is still not a first-class native production path.
   - Current progress is strongest on staged integrated geometry and benchmark annulus lanes.
   - Reduced 3D tokamak EM and 2D tokamak production/recycling still need direct native parity closure.

4. EM capability is benchmark-strong but not yet full-system complete.
   - The benchmark ladder is good.
   - The fully coupled EM production path, including remaining transient operators and selected 3D cases, is still unfinished.

5. Full neutral/reaction/impurity breadth is still open.
   - This includes the remaining neutral models, reaction families, and mixed-species/impurity workflows needed for a complete standalone DRB code.

6. Ship surfaces are not done.
   - output/restart/log compatibility
   - full Read the Docs information architecture
   - example gallery
   - convergence and validation campaign
   - CPU/GPU production smoke on the final selected ladder

### Best Next Steps

The remaining work should now proceed in this order:

1. Promote direct tokamak geometry from staged support to native production support.
   - Use the already-stable integrated lanes to reduce risk.
   - Add the smallest reduced tokamak production/recycling cases that can be locked cleanly.
   - `tokamak_recycling_rhs` is now the active next rung on that lane:
     - live reference execution on `examples/tokamak-2D/recycling` is running again through the shared harness at `process_count = 6`;
     - committed direct-tokamak snapshot cache and summary/array baselines are now in-tree for `tokamak_recycling_rhs`;
     - the native direct tokamak RHS path now matches those committed baselines exactly after fixing the native species initializer to keep `neutral_mixed` species when `type` parses as a scalar string and after restoring explicit neutral pressure-source assembly;
     - `tokamak_recycling_one_step` is now explicitly curated as a small-step transient rung with `timestep=1`, and now has committed summary/array baselines plus a committed optional-history cache on top of the committed `tokamak_recycling_rhs` snapshot cache;
     - the native direct tokamak one-step runner now uses `bdf` plus the ion-only target-preservation split on this rung, which collapses the transient mismatch to a small operational band against the committed baseline:
       - `Pe`: about `5.41e-4`
       - `Pd+`: about `9.41e-6`
       - `Nd+`: about `2.03e-5`
       - `NVd+`: about `1.45e-6`
     - the multispecies direct-tokamak recycling lane is now unblocked too:
     - `tokamak_recycling_dthe_rhs` now has committed summary/array baselines plus a committed direct-tokamak snapshot cache and the native RHS path matches those baselines exactly;
     - the same direct multispecies recycling lane is now widened one species further: `tokamak_recycling_dthene_rhs` on `examples/tokamak-2D/recycling-dthene` now has committed summary/array baselines plus a committed direct-tokamak snapshot cache and the native RHS path matches those baselines exactly on the D/T/He/Ne/e surface;
     - landing that dthene RHS rung required two narrow infrastructure fixes rather than another operator rewrite:
       - the reference harness now stages the shared `json_database/` directory automatically when it exists above the case input, which unblocks Hermes `OpenADAS` runs that hardcode `json_database/...` relative paths;
       - the native recycling branch now vendors the minimal neon OpenADAS rate/radiation tables (`scd96_ne`, `acd96_ne`, `plt96_ne`, `prb96_ne`) and uses them for the `ne <-> ne+` level-0 reactions needed by this rung;
     - that dthe RHS rung required a narrow Hermes-side permission fix in `src/braginskii_collisions.cxx` so the multispecies tokamak case can populate `*_coll` collision-frequency entries without aborting during solver initialization; this changes permission bookkeeping only, not the collision formulas;
     - `tokamak_recycling_dthe_one_step` is now a landed curated transient rung too: the manifest stages it at `timestep=0.1`, the committed summary/array baselines and optional-history cache are now in-tree, and the native multispecies one-step runner stays inside an operational band against that committed baseline:
       - `Pe`: about `1.89e-4`
       - `Pd+`: about `9.23e-3`
       - `NVd+`: about `3.60e-2`
       - `Pt+`: about `1.12e-2`
       - `NVt+`: about `5.37e-2`
     - the remaining blocker on that D/T direct-tokamak rung is now localized and script-backed rather than inferred:
       - [diagnose_tokamak_recycling_one_step.py](scripts/diagnose_tokamak_recycling_one_step.py) ranks the committed-baseline one-step residuals and confirms they are concentrated at the lower target corner;
       - [diagnose_tokamak_recycling_ion_viscosity.py](scripts/diagnose_tokamak_recycling_ion_viscosity.py) drills into the same full-grid blocker cell `(x=2, y=2, z=0)` on the reference-evolved state and shows the local collision stack is dominated by the ion-viscosity term itself, with `DivPiPar_d+ ≈ -4.12` and `DivPiPar_t+ ≈ -5.29`;
       - that same script can now rerun the reference executable with `braginskii_collisions:diagnose=true` and compare the local `K*_coll` fields directly; at the blocker cells the native D/T Coulomb collision frequencies match to roundoff, while the neutral-collision subtotal is identically zero there, which rules out `_compute_collision_frequencies` and any hidden neutral-collision bookkeeping as the main cause of the residual and narrows the next patch surface to D/T charge exchange or the sheath-conditioned `DivPiPar` boundary state/operator;
       - a live one-step dump check now also confirms that the reconstructed lower-target guard states (`NVd+ / (2 Nd+)`, `NVt+ / (3 Nt+)`, `Pd+`, `Pt+`) already match the reference dump at the blocker face, so the remaining gap is no longer guard-state assembly; the updated blocker script now prints the collisionality implied by the reference `DivPiPar` itself, and at `(x=2, y=2, z=0)` it says the current D/T CX subtotal would need to be larger by about `4.8x` to close the D/T `d+` viscosity gap by collisionality alone;
       - the native recycling path now also honors per-atom `K_cx_multiplier` in both charge-exchange source assembly and charge-exchange collision-rate assembly, closing one remaining source-option gap before the next `DivPiPar` physics patch;
     - the dead ends are explicit now and should not stay on the critical path:
       - the committed cache now also carries full evolving state fields for `tokamak_recycling_dthe_one_step`, so the one-step residual can use reference-evolved communicated non-target guard rows as fixed templates during the implicit solve instead of approximating them from local target logic alone;
       - the tested local toggles (`eta` Neumann guards, removing viscosity boundary flux, and skipping initial velocity overrides) do not improve the D/T one-step mismatch;
       - the new `g_23` metric plumbing and anomalous-coefficient literal-reference fix are worthwhile support work, but they are not the main explanation for the current D/T one-step residual;
     - the same lane is now widened to the D/T/He/Ne transient surface too: `tokamak_recycling_dthene_one_step` is now curated at `timestep=0.1`, has committed summary/array baselines plus a committed optional-history cache, and the native multispecies one-step runner is already in a tighter operational band than the dthe rung:
       - `Pe`: about `2.05e-3`
       - `Pd+`: about `2.79e-3`
       - `Nd+`: about `2.09e-3`
       - `Pt+`: about `2.79e-3`
       - `Nt+`: about `2.09e-3`
       - `NVd+` / `NVt+`: about `2.1e-6`
       - helium and neon channels: `O(1e-8 .. 1e-10)` on the committed one-step surface
     - the live Hermes reference path for the dthe tokamak lane required a narrow local fix in `BraginskiiCollisions`: add explicit positive-ion cross-collision write permission so the multispecies tokamak case can populate `species:*:collision_frequencies:*_he_coll` without aborting during solver initialisation; this is permission bookkeeping only, not a collision-formula change;
     - the next richer direct tokamak transport rung is now also in-tree: `tokamak_diffusion_transport_one_step` on `examples/tokamak-2D/diffusion-transport` has committed summary/array baselines and the native direct tokamak path matches them exactly on `Nh+`, `Ph+`, `NVh+`, and `Pe`;
     - that same direct tokamak transport lane is now widened to a first multi-output transient rung too: `tokamak_diffusion_transport_short_window` has committed summary/array baselines, carries the curated `nout=5` history, and the native direct tokamak path matches it exactly on the same compare surface;
     - the next neighboring direct tokamak physics family is now in-tree as well: `tokamak_heat_transport_one_step` on `examples/tokamak-2D/heat-transport` has committed summary/array baselines and the native direct tokamak path matches them exactly on `Pe`;
     - that same heat-transport family is now widened to an exact early-window transient rung too: `tokamak_heat_transport_short_window` carries the curated harmless overrides `nout=2` and `e:diagnose=false`, has committed summary/array baselines, and the native direct tokamak path matches them exactly on `Pe`;
     - the first exact direct-tokamak transient cache is now in-tree too: `tokamak_heat_transport_short_window` has committed snapshot and field-history caches in `references/baselines/reference_snapshots/`, so repeated native parity checks on this rung no longer need a fresh Hermes launch;
     - the next broader direct tokamak transport family is now also locked at the one-step level: `tokamak_diffusion_conduction_one_step` on `examples/tokamak-2D/diffusion-conduction` now has committed summary/array baselines and matches them exactly once `h+:diagnose=false` and `e:diagnose=false` are applied as explicit case overrides, which removes a broken Hermes diagnostic-only `particle_flow_ylow` path without changing the compared physics surface;
     - that same diffusion-conduction family is now widened to an exact early transient rung too: `tokamak_diffusion_conduction_short_window` with `nout=5`, `h+:diagnose=false`, and `e:diagnose=false` now has committed summary/array baselines plus committed snapshot/history caches and matches them exactly on `Nh+`, `Ph+`, and `Pe`;
     - the smallest neighboring direct tokamak family is now also locked at the one-step level: `tokamak_diffusion_one_step` on `examples/tokamak-2D/diffusion` has committed summary/array baselines and the native direct tokamak path matches them exactly on `Nh`;
     - the next low-risk direct tokamak fixed-density transport family is now also locked at the one-step level: `tokamak_linear_transport_one_step` on `examples/tokamak-2D/linear-transport` has committed summary/array baselines, is curated only with `e:diagnose=false`, and the native direct tokamak path now matches it exactly on `Pe`;
     - that same fixed-density transport family is now widened to an exact early transient rung too: `tokamak_linear_transport_short_window` with `nout=5`, `e:diagnose=false` now has committed summary/array baselines plus committed snapshot/history caches and matches them exactly on `Pe`;
     - the same direct tokamak electrostatic/vorticity family now has an exact operator rung too: `tokamak_isothermal_rhs` on `examples/tokamak-2D/isothermal` has committed summary/array baselines plus committed snapshot/history caches and matches exactly on `Ne`, `Ni`, `NVe`, `NVi`, `phi`, `Vort`, `ddt(Ne)`, `ddt(NVe)`, `ddt(NVi)`, and `ddt(Vort)` with the harmless curation `timestep=0.1`, `e:diagnose=true`, `i:diagnose=true`, and `vorticity:diagnose=true`;
     - the first direct tokamak electrostatic/vorticity family is now also locked at the one-step level: `tokamak_isothermal_one_step` on `examples/tokamak-2D/isothermal` has committed summary/array baselines plus committed snapshot/history caches, and the native direct tokamak dump-backed path matches it exactly on `Ne`, `Ni`, `NVe`, `NVi`, `phi`, and `Vort` with the harmless curation `timestep=0.1`;
     - that same family is now widened to an exact early transient rung too: `tokamak_isothermal_short_window` with `nout=2`, `timestep=0.1` now has committed summary/array baselines plus committed snapshot/history caches and matches them exactly on the same compact electrostatic/vorticity surface;
     - that same family is now widened once more to an exact broader transient rung too: `tokamak_isothermal_medium_window` with `nout=5`, `timestep=0.1` now has committed summary/array baselines plus committed snapshot/history caches and matches them exactly on the same compact electrostatic/vorticity surface;
     - that cache pass is now materially wider too: `tokamak_linear_transport_one_step`, `tokamak_diffusion_one_step`, and `tokamak_diffusion_conduction_one_step` now each have committed snapshot and field-history caches in `references/baselines/reference_snapshots/`, and the exact transient rung `tokamak_diffusion_transport_short_window` now does as well, so the cheapest exact one-step and short-window tokamak parity checks no longer require a fresh Hermes launch either;
     - the direct tokamak recycling cache path now also covers the widened multispecies neon rung: `tokamak_recycling_dthene_rhs` has a committed snapshot cache, and the cache builder now saves populated state/optional fields for `tokamak_recycling*` cases instead of the empty snapshot format used by the cheap non-recycling transport rungs;
   - Treat the integrated Step 3 production lane as operationally complete for project flow.
   - The current committed-baseline target-band `integrated_2d_production_one_step` residuals are already small in a meaningful norm:
     - `Pe`: about `1.63e-1` on a `~1.05e3` field (`~1.55e-4` relative to expected max)
     - `Pd+`: about `5.0e-3` on a `~1.05e3` field (`~4.8e-6`)
     - `Nd+`: about `4.1e-3` on a `~7.0e2` field (`~5.9e-6`)
     - `Sd_target_recycle`: about `1.0e-3` on a `~2.33e1` field (`~4.4e-5`)
   - The broader `short_window` / `medium_window` production rungs remain useful calibration surfaces, but they should no longer block the main path while the residuals are localized and the selected one-step production rung is already parity-credible.
   - Keep the remaining integrated-lane work as a sidecar:
     - target-band `Pe` / neutral-side transient cleanup on the broader windows
     - non-orthogonal `anomalous_diffusion` for the integrated production lane once `g23/g_23` is available in the native metric payload

2. Keep widening the EM ladder only where parity remains exact.
   - Prefer benchmark-first additions like the annulus lane over inexact operator substitutions.
   - Then port the next exact EM transient operator slice or the next reduced tokamak EM rung.

3. Close the remaining open-field transient defect.
   - Keep this as a focused sidecar task rather than the main critical path.
   - The blocker is narrow enough now that it should be solved with targeted transient work, not another broad rewrite.

4. Finish the remaining neutral/reaction/impurity breadth and ship surfaces.
   - reaction families
   - impurity workflows
   - output/restart compatibility
   - docs/examples/validation package
   - reviewer-facing convergence matrix

### Ship-Ready Definition

`jax_drb` should be considered ship-ready only when all of the following are true:

- It can be presented as a standalone code, not a parity prototype.
- The selected 1D, 2D, and reduced 3D comparison ladder is parity-clean enough for a paper claim.
- The supported public capability matrix is explicit and honest:
  - geometry families
  - closures
  - boundary conditions
  - electrostatic / electromagnetic
  - neutral / reaction / impurity support
- Output, restart, diagnostics, CLI, Python API, docs, and examples are complete enough for external users.
- The benchmark and convergence campaign can be rerun from versioned scripts and committed artifacts.

### Publication-Grade Readiness

For the intended public claim against the private reference implementation, "ship-ready" is necessary but not sufficient. A publication-grade `jax_drb` should satisfy all of the following:

- The supported capability matrix is broad enough to be scientifically interesting on its own:
  - open-field and tokamak geometry
  - electrostatic and reduced electromagnetic benchmark ladders
  - Bohm/simple sheath boundary conditions
  - neutrals and recycling on selected divertor workflows
  - hot-ion and cold-ion reduced workflows where the reference supports them
  - Boussinesq / non-Boussinesq coverage stated explicitly, with only the supported surfaces claimed
- The codebase is maintainable by researchers and graduate students:
  - examples are tutorial-like rather than opaque wrappers
  - restart/output/plotting/movie generation are part of the standard workflow
  - runtime configuration is explicit, typed, and documented
  - new solvers and physics slices can be added without bypassing the parity harness
- The paper evidence is layered rather than anecdotal:
  - unit tests for low-level operators and semantics
  - regression tests for curated summary/array baselines
  - physics tests and benchmark diagnostics
  - convergence studies on selected ladders
  - reviewer-facing figures and tables regenerated from versioned scripts

### Distance To Publication-Grade Standalone Status

Current estimate:

- curated, benchmarked, regression-locked parity: roughly `70-80%`
- broad standalone code ready for a strong public claim: roughly `45-60%`

This means the project is already beyond the "prototype" stage, but not yet at the point where a broad public claim would be technically honest without caveats.

### Remaining Workstreams Before A Strong Public Claim

1. Direct tokamak widening and hardening.
   - Keep expanding the exact direct tokamak ladder from the current diffusion/transport/conduction/linear/recycling set.
   - Add committed caches so exact direct tokamak parity no longer depends on repeated live reference launches.
   - Promote the current operational recycling transients toward tighter locked parity where that is practical.

2. Integrated/open-field residual closure.
   - Finish the remaining open-field transient defect.
   - Close the non-orthogonal anomalous-diffusion gap on the integrated production lane.
   - Keep broader integrated production windows as calibration surfaces, but close the ones that matter for public evidence.

3. Electromagnetic widening beyond the benchmark ladder.
   - Keep the current exact benchmark-first EM workflow.
   - Add the next reduced tokamak EM ladders only where exact or tightly controlled parity is maintainable.
   - Avoid making a broad EM claim from benchmark-only coverage.

4. Validation and convergence campaign.
   - Build a reviewer-facing matrix of:
     - exact parity cases
     - operational-band cases
     - benchmark metrics
     - convergence figures
   - Ensure every figure in the paper can be regenerated from committed scripts and saved artifacts.

5. Research-user surfaces.
   - Keep the CLI/TOML/restart/output path stable.
   - Add more example decks and tutorial scripts showing:
     - setup
     - runtime control
     - restart
     - output analysis
     - 2D/3D plotting and movies
   - Keep public APIs and solver extension points explicit enough for future physics additions.

### Estimated Iterations Remaining

Best current estimate, assuming focused iterations that each land with tests/docs/commits:

- To reach "strong, defensible parity on selected core workflows suitable for a paper":
  - about `10-15` good iterations
- To reach "broad standalone code ready for a strong public claim against the private reference":
  - about `20-35` good iterations

Those iterations are not all equal. The highest-value ones are the ones that reduce uncertainty in the capability matrix, not the ones that only polish already-good ladders.

### Main Critical Path

The main critical path should stay:

1. widen exact direct tokamak support efficiently;
2. reduce reference rerun cost with committed caches and deterministic artifacts;
3. close the remaining open-field / integrated production residuals that would weaken a public claim;
4. widen EM only where exact parity remains possible;
5. finish the reviewer-facing convergence and validation package.

### What Should Not Happen

To stay publication-focused, avoid:

- spending many iterations on cases that are known to stall or have poor cost/signal ratio before a safer curated rung exists;
- broad refactors that do not improve parity, validation, maintainability, or user-facing runtime quality;
- making capability claims ahead of the current test and benchmark surface;
- adding new physics branches before the base research-grade matrix is stable enough to carry them.

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
- the remaining transient blocker is now localized by a dedicated short-step probe in [diagnose_recycling_transient_step.py](scripts/diagnose_recycling_transient_step.py): on a `timestep = 25` reference run, the native backward-Euler step already misses the evolved state while the native RHS evaluated on that same reference-evolved state stays tight, so the remaining Step 2 defect is the transient integrator path itself rather than another open-field operator/source mismatch
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
  - a new committed-snapshot A/B probe now closes the remaining guard-template ambiguity on the open-field one-step lane: [cache_open_field_dump_case.py](scripts/cache_open_field_dump_case.py) produces committed RHS snapshot caches for `recycling_1d_rhs` and `recycling_dthe_rhs`, and [diagnose_open_field_recycling_templates.py](scripts/diagnose_open_field_recycling_templates.py) compares the default native one-step path against a field-template replay probe that fixes the non-active cells to the reference-evolved state. That replay probe is dramatically worse on both families (`recycling_1d_one_step`: `Nd+ ≈ 3.43e-1`, `Pd+ ≈ 6.06e-1`, `NVd+ ≈ 8.35e-1`; `recycling_dthe_one_step`: `Nd+ ≈ 9.45e-1`, `Pd+ ≈ 3.70e-1`, `NVd+ ≈ 1.15e+0`), while the default native paths stay far tighter. So the remaining Step 2 blocker is not non-active guard-template uncertainty; it is active transient evolution itself.
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
| `recycling_1d_short_window` | `native operational target` | First curated repeated-output open-field recycling rung (`nout=5`); the local native backbone stays within bounded residuals but is not yet exact. |
| `recycling_1d_one_step` | `native exact target` | Native first-step transient now clears the promoted one-step scaled-diff gate (`relative_to_expected_max < 5e-2`) on the committed compare surface. |
| `recycling_dthe_one_step` | `native exact target` | Native first-step multispecies transient now clears the same promoted one-step scaled-diff gate (`relative_to_expected_max < 5e-2`) on the committed compare surface. |
| `recycling_1d_long` | `blocked` | Long-run parity depends on the transient ladder. |
| `integrated_2d_recycling_rhs` | `native-scaffolded` | Native staged local-dump RHS path now runs, preserves slab-local physical targets, keeps the dumped target state, injects dump-backed density and ion-pressure source fields for the staged integrated case, and follows the reference `sheath_boundary_simple` electron guard/energy closure closely enough that the large target-row `ddt(Pe)` miss is gone. This remains staged evidence because the integrated RHS surface is still dump-backed. |
| `integrated_2d_recycling_one_step` | `native exact target` | Dump-backed one-step integrated recycling baseline is now committed and the native transient path clears a promoted mixed exact gate: all non-negligible fields stay inside a tight scaled band, while the effectively silent `NVd` channel stays inside a tiny absolute band. The path computes `Sd_target_recycle` and `Ed_target_recycle` natively at every saved time, including `t=0`, and it no longer depends on dump-backed density, pressure, or momentum source replay or on dump target preservation during the transient. |
| `integrated_2d_recycling_short_window` | `native exact target` | The integrated 2D recycling short-window path now also clears the promoted mixed exact gate against the committed baseline over `nout=5`; the saved diagnostics are native across the whole window and the transient no longer replays dump-backed source fields or preserves dump target state. |
| `integrated_2d_recycling_medium_window` | `native exact target` | The integrated 2D recycling medium-window path now also clears the promoted mixed exact gate over the staged `nout=20` window. The only remaining bootstrap dependency is the committed staged initial snapshot, not dump-backed transient/source replay. |

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
- the `integrated_2d_production_rhs` path now also uses that same ion-only target-preservation split. That was the missing operator-side piece: `ddt(NVd+)` is now exact on the committed target-band comparison, and the remaining RHS residual is reduced to small `ddt(Pe)` / `ddt(Pd)` edge statistics.
- the `integrated_2d_production_one_step` path now uses the stiffer `bdf` transient backend instead of the default single-ion continuation path, and it now preserves only the dump-backed ion target state while letting the electron target state evolve through the sheath closure during the transient. With the matching RHS-side ion-only preservation now in place, the live one-step target-band residuals are much smaller: `Pe` is now the leading field at about `1.5e-1` max abs diff, followed by `Sd_target_recycle` at about `7.8e-2` and `NVd+` at about `5.8e-2`, with `Pd+`, `Nd+`, and `Ed_target_recycle` in a much smaller band.
- the staged production transient lane now also applies the committed optional-history `Vd+`/`Vd` overrides to its initial state before the implicit march, so the transient path starts from the same velocity-consistent momentum state already used by the corrected production RHS path. That did not materially move the current target-band residuals by itself, but it removes another harness inconsistency and is now locked by focused cache-based tests.
- the latest production diagnostic reconstruction now also stages per-time dumped ion velocity diagnostics (`Vd+`, `Vd`) when rebuilding `Sd_target_recycle` / `Ed_target_recycle` on the broader integrated production ladder. That is a staged-harness improvement, not a production-kernel claim, but it materially tightens the remaining production transient surface: the live `Sd_target_recycle` max-abs residual dropped from about `1.65e-1` to about `7.78e-2` on `integrated_2d_production_one_step`, and from about `7.81e-1` to about `3.75e-1` on `integrated_2d_production_short_window`.
- the latest production-target probes now rule out the two remaining naive alternatives as well: staging `SPe` as a total electron-pressure source reduces one corner-cell `Pe` error but worsens the neighboring target-band cell, and disabling dump-target preservation improves `Pe` while badly degrading `Nd+`, `Pd+`, and `NVd+`. The next production fix therefore needs to target the target-band recycling/source update itself rather than broader source replacement or a looser target-state policy.
- a broader attempt to replay per-interval dumped total source histories inside the production transient march was also rejected: it preserved `Sd_target_recycle` at roughly the same level but made the target-band `Pe` and `NVd+` state errors worse. The integrated production lane should therefore keep the current staged source surface (`SNd`, `SNVd`, `SPd`, plus staged diagnostic reconstruction from dumped ion velocities) until a narrower target-band transient fix is ready.
- the next narrow production-side fix is now in-tree as well: `preserve_dump_ion_target_state_only` no longer drops the ion sheath boundary state entirely. It now preserves the dump-backed ion target cell while still using the sheath-generated ion guard cells and ion sheath energy sinks, which is the right local state for target-band transient work on `Pe` / `NVd+`.
- Step 3 now also has a faster inner-loop diagnostic workflow: `scripts/diagnose_integrated_2d_production_parity.py` can compare against committed baselines and filter to target-band-only residuals. That should be the default iteration path while tightening the integrated production lane; live Hermes reruns are still useful, but they are no longer necessary for every local production tweak.
- that same production diagnostic loop now also supports worst-cell time traces against the committed baselines. The current traces make the remaining blocker explicit: on `integrated_2d_production_one_step`, the worst `Pe` cell is one target row above the worst `NVd+` / `Sd_target_recycle` cell, and all three drift from zero on the very first accepted step. On `integrated_2d_production_short_window`, those same target-band errors then grow monotonically step-by-step rather than appearing as a late-window jump. The next Step 3 physics pass should therefore inspect the first accepted target-band transient update directly, not just the end-of-window state.
- the Hermes `sheath_boundary_simple` electron heat-flux formula is now explicitly locked by a focused unit test against the source-level algebra (including wall-potential clipping and the advected-fluid-energy subtraction). That removes another possible false lead: the remaining Step 3 production blocker is not the basic simple-sheath electron energy formula itself, but the coupled first-step transient update around it.
- a new one-step production step diagnostic now compares both the native evolved state and the native RHS evaluated on the reference-evolved state. After tightening that script to use the evolved one-step source overrides from the final Hermes dump, the result is sharper than the older end-state-only traces: on `integrated_2d_production_one_step`, `Sd_target_recycle` and `Ed_target_recycle` are exact on the reference-evolved state, `ddt(Nd+)` is now effectively exact, and the remaining meaningful target-band RHS leaders are `ddt(Pe)` first, then `ddt(Pd+)`, then `ddt(NVd+)`, with `ddt(Nd)` smaller but still visible. The next Step 3 work should therefore focus on the local target-band pressure/ion-pressure/momentum RHS assembly rather than more recycling-diagnostic reconstruction.
- the same focused production diagnostics are now being moved onto a committed-data loop rather than a live-reference loop. `diagnose_integrated_2d_production_step.py`, `diagnose_integrated_2d_production_pe_terms.py`, and `diagnose_integrated_2d_production_ion_terms.py` now accept `--use-committed-baselines` and can synthesize a fallback final one-step snapshot from the committed active-domain arrays plus optional-history cache when the exact final diagnostic snapshot is missing. That is now the intended inner loop for the Step 3 blocker: committed production state/RHS/source caches first, synthetic final-snapshot fallback second, and live Hermes reruns only when a physics patch looks promising.
- that fallback is now required on this machine because fresh Hermes reruns of `integrated_2d_production_one_step` are currently crashing in the private PVODE path immediately after solver initialization, before writing `BOUT.dmp.0.nc`.
- the current Step 3 unblocker is now an explicit two-pronged controlled experiment rather than another open-ended tweak cycle:
  - [scripts/diagnose_integrated_2d_production_anomalous_diffusion.py](scripts/diagnose_integrated_2d_production_anomalous_diffusion.py) compares the current native one-step target-band residual against a fresh benchmark rerun with `anomalous_diffusion` removed from `d+` and `e`, so the team can decide whether the next implementation item should be the anomalous-diffusion operator slice or a narrower boundary/operator fix;
  - [scripts/diagnose_integrated_2d_production_step.py](scripts/diagnose_integrated_2d_production_step.py) now supports an A/B boundary sweep between the current production mixed-preserve mode and a full-sheath mode on the exact bad cells, so boundary-condition mismatch can be separated cleanly from missing operator/source-term mismatch.
- the production ion-only preserve path in [recycling_1d.py](src/jax_drb/native/recycling_1d.py) is now aligned with that diagnosis strategy: `preserve_dump_ion_target_state_only` still preserves the dump-backed ion target cell, but it now keeps the sheath-generated ion guard cells as well. That is the smallest local state change consistent with the remaining target-band `Pe` face-flux blocker.
- the companion ion-term decomposition script now also resolves the remaining ion-side ambiguity on that same production one-step state. At the bad target-band cells, `ddt(Nd+)` and `ddt(NVd+)` are already exact once the diagnosis uses the evolved one-step source and velocity fields, while the surviving `ddt(Pd+)` miss matches the local `(2/3) * energy_source[d+]` contribution. That makes the next Step 3 patch narrower again: the likely remaining production ion-side defect is the preserved-target ion energy-source treatment, not the ion density flux or ion momentum advection operator.
- that ion-side diagnosis now has a second, narrower rung as well: [scripts/diagnose_integrated_2d_production_ion_terms.py](scripts/diagnose_integrated_2d_production_ion_terms.py) now prints the `d+` energy-source breakdown at the bad target-band cells (`reaction`, `sheath`, `collision`, `recycle`, `feedback`) before forming `(2/3) * energy_source[d+]`. That is the intended gate for the next `Pd+` fix, so the next code change can target the exact contributing channel rather than the whole ion pressure RHS.
- the dominant production `Pe` blocker is now explicitly identified and patched as a state-sequencing defect, not a missing source term: the integrated production electron pressure RHS was still using the zero-current electron velocity instead of the boundary-conditioned `electron_boundary.velocity` after `sheath_boundary_simple`. Matching the Hermes transform/finally ordering here collapses the one-step target-band residuals substantially.
- the ion-side preserve path has now been tightened in the same way: when `preserve_dump_ion_target_state_only` is active, the native path preserves the dump-backed ion target cell, keeps the sheath-generated ion guard cells, but no longer carries the sheath-generated ion `energy_source` into the preserved target cell itself. That removes the dominant residual on `integrated_2d_production_one_step`: the committed-baseline target-band compare is now led by `Pe` at about `1.63e-1`, followed by `Nd` at about `1.10e-2`, `Pd+` at about `5.0e-3`, `Nd+` at about `4.1e-3`, `Sd_target_recycle` at about `1.0e-3`, `NVd+` at about `6.2e-4`, and `Ed_target_recycle` at about `7.3e-6`.
- the new synthetic one-step blocker diagnostics now sharpen the remaining `Pe` defect further. At the bad target-band cells, the total electron pressure RHS is still not negative enough because a large positive electron energy-source contribution remains, dominated by collision heating over the sheath sink. A direct attempt to move collision closure onto the fully preserved target state was tested and rejected because it made one-step and short-window parity much worse, so the remaining Step 3 production work should stay focused on a narrower `Pe` / neutral-side transient fix rather than a broader preserve-mode rewrite.
- Step 3 is now treated as operationally complete for project flow and is locked by committed-baseline regression gates rather than another speculative production physics patch. The native test surface now explicitly asserts that:
  - `integrated_2d_production_rhs` stays within the current small pressure-summary residual band (`ddt(Pe) < 9e-2`, `ddt(Pd) < 2e-3`, `ddt(NVd+) = 0` on the committed target-band compare);
  - `integrated_2d_production_one_step` stays within the current operational target band (`Pe < 1.7e-1`, `Nd < 1.2e-2`, `Pd+ < 6e-3`, `Nd+ < 5e-3`, `Sd_target_recycle < 2e-3`, `NVd+ < 1e-3`, `Ed_target_recycle < 1e-5`);
  - `integrated_2d_production_short_window` stays within the current operational target band (`Pe < 1.5`, `NVd+ < 5.5e-1`, `Nd < 3e-1`, `Nd+ < 7.5e-2`, `Pd < 3.5e-2`, `Sd_target_recycle < 6e-3`, `Ed_target_recycle < 5e-5`).
- With those gates in place, the remaining integrated-production work is now a sidecar ship-readiness item rather than a Step 3 blocker. The main lane should move on to the next plan item: direct tokamak geometry support on top of the now-stable integrated 2D ladder, while non-orthogonal anomalous diffusion and broader production-window cleanup continue opportunistically.
- that direct-geometry move is now started with a stable family rather than a single rung: `tokamak_diffusion_flow_one_step`, `tokamak_diffusion_transport_one_step`, `tokamak_diffusion_transport_short_window`, and `tokamak_heat_transport_one_step` first established the lane, and it is now widened through exact `tokamak_diffusion_conduction_one_step`, `tokamak_diffusion_conduction_short_window`, `tokamak_diffusion_one_step`, `tokamak_linear_transport_one_step`, `tokamak_linear_transport_short_window`, `tokamak_isothermal_rhs`, `tokamak_isothermal_one_step`, `tokamak_isothermal_short_window`, `tokamak_isothermal_medium_window`, `tokamak_turbulence_rhs`, `tokamak_turbulence_one_step`, and `tokamak_turbulence_short_window` rungs as well. All of those run through the reference harness at `process_count = 6`, stage the shared parent-directory `tokamak.nc` mesh deterministically, and match their committed summary and array baselines exactly on the native direct tokamak dump-backed path. The neighboring `recycling-dthe-drifts` family has now been promoted out of the vague-blocker bucket too: the curated `tokamak_recycling_dthe_drifts_rhs` rung is exact on its compact first-output surface and the curated `tokamak_recycling_dthe_drifts_one_step` rung is now in a small operational band, both using deterministic `sound_speed` plus `solver:type=cvode` curation in the manifest. The current native recycling compare surface there is intentionally limited to `Nd+`, `Pd+`, `NVd+`, `Nt+`, `Pt+`, `NVt+`, `Nhe+`, `Phe+`, `NVhe+`, and `Pe`; `phi` and `Vort` are still missing from the native recycling path and should not be claimed until they are explicitly staged or evolved.
- the next direct tokamak blocker loop is now sharper and script-backed rather than ad hoc: [diagnose_tokamak_recycling_one_step.py](scripts/diagnose_tokamak_recycling_one_step.py) ranks the committed-baseline one-step residuals for `tokamak_recycling_dthe_one_step`, `tokamak_recycling_dthe_drifts_one_step`, and `tokamak_recycling_dthene_one_step`, reports missing compare variables explicitly, and prints time traces at the worst cells. That is the intended gate before touching the remaining D/T momentum/pressure transient mismatch again.
- the component graph now also reflects the reference production input more faithfully: species with nonzero `anomalous_D`, `anomalous_chi`, or `anomalous_nu` are now expanded to include `*:anomalous_diffusion` even when the `type = ...` tuple does not mention it explicitly. For the current `2D-production` input, that means both `d+` and `e` now carry the missing `anomalous_diffusion` label in the resolved runtime graph.
- the first native anomalous-diffusion helper is also in-tree for orthogonal metrics and already covers the exact `anomalous_nu`-only momentum-source slice on a `g23 = 0` unit surface. The integrated production lane still cannot consume it yet because Hermes uses the non-orthogonal `Div_a_Grad_perp_upwind_flows(...)` operator there and the current native metric payload does not include the matching `g_23` coefficient. So the immediate Step 3 critical path has shifted from `Pd+` to the remaining `Pe` / neutral-side transient drift, while non-orthogonal anomalous diffusion is now a sharply isolated follow-on item rather than an undefined gap.
- the integrated production lane now also supports compact committed snapshot caches under `references/baselines/reference_snapshots/`. The committed artifacts now include the shared `integrated_2d_production_rhs` snapshot plus staged optional-history caches for both `integrated_2d_production_one_step` and `integrated_2d_production_short_window`, so the production one-step and short-window rungs can avoid both the repeated live-reference bootstrap and the repeated staged diagnostic-history fetch. The same path can now be extended incrementally to the medium-window rung without further runner work.
- the broader integrated `2D-production` workflow now also has a committed `short_window` rung (`nout=5`) on the same dump-backed geometry lane. It is not parity-clean yet, but it gives Step 3 a real broader-production transient ladder. After the same preserved-ion energy-source fix landed on the one-step lane, the current target-band short-window differences are led by `Pe` (about `1.38` max abs diff), then `NVd+` (about `5.28e-1`), then `Nd` (about `2.67e-1`), while `Nd+`, `Pd`, and the staged recycling diagnostics are already much tighter. A direct `bdf` probe for this longer window still does not beat the current path overall, so the next production-short-window work should stay focused on physics/source closure rather than solver swapping alone.
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
- keep the user-facing native runtime usable while parity broadens:
  - `jax-drb run` should write summary JSON, arrays NPZ, restart NPZ, and verbose run-log JSON for supported native inputs;
  - restart/resume should be demonstrated by a runnable tutorial example rather than only internal tests;
  - examples should show explicit input-deck construction, CLI invocation, saved artifacts, and Matplotlib postprocessing.
  - this is now landed for the native-supported diffusion path via [examples/restartable_diffusion_tutorial.py](examples/restartable_diffusion_tutorial.py) and [docs/native_runtime_cli.md](docs/native_runtime_cli.md):
    - users can now run supported inputs directly as `jax_drb input.toml` or `jax-drb input.toml` without an explicit `run` subcommand;
    - the intended native input surface is now organized TOML with `[time]`, `[runtime]`, `[runtime.logging]`, `[mesh]`, `[solver]`, `[model]`, `[output]`, `[restart]`, `[species.*]`, and `[fields.*]`;
    - output destinations, restart/resume requests, and logging verbosity can now be declared in the deck itself instead of only on the CLI;
    - the run-log JSON now includes an ordered event stream for configuration, restart loading, launch, recycling progress, completion, and artifact planning, and the terminal path mirrors that with rich event panels before the final summary table;
    - the run-log JSON also now records execution-environment metadata (`jax_version`, `python_version`, platform, pid, working directory) so standalone runs are easier to audit and reproduce;
    - runtime precision is now user-selectable in both the input deck and the driver/CLI layer;
    - the supported diffusion executable path now runs cleanly in both `float64` and `float32`, with committed benchmark scripts and artifacts in-tree; on the current local CPU rung the warm second-run `float32` diffusion path is about `1.24x` faster than `float64`, so precision is now a real supported knob on that native slice, but it is still not the global default until broader production paths drop their remaining explicit `float64` assumptions;
    - the remaining work is to widen the same output/restart/precision surface across the broader production cases.

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
  - `jax_drb path/to/input.toml`
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
- the same ladder now also has a committed `alfven_wave_medium_window` rung at the default `nout=50`, giving Step 4 a longer electromagnetic transient target that still fits under the repository size cap and preserves exact native/reference parity on the current scaffold.
- the next broader Stage 4 rung is now `examples/other/linear/annulus-isothermal-he-emag` with a committed `annulus_he_emag_rhs` baseline on a slim EM-only compare surface: `Apar`, `alpha_em`, `ddt(Ne)`, `ddt(NVe)`, and `ddt(Vort)`. That case widens the electromagnetic surface beyond the slab benchmark while staying under the repository size cap and keeping exact native/reference parity on the current dump-backed scaffold. `Ajpar` is computed natively for inspection on that lane, but it is not part of the locked parity surface because the saved `nout=0` reference dump keeps `Ajpar` identically zero on that example.
- the same annulus electromagnetic lane now also has a committed `annulus_he_emag_one_step` rung using the same example with a curated smaller transient interval (`timestep=10`, `nout=1`) and a slim state compare surface: `Apar`, `Ne`, `NVe`, `phi`, and `Vort`. That gives Step 4 a second transient EM ladder beyond the Alfvén slab while keeping the compressed baseline comfortably under the repository size cap.
- the same annulus lane now also has a committed `annulus_he_emag_short_window` rung with `timestep=10`, `nout=5`, and a further slimmed transient compare surface: `Apar`, `Ne`, and `phi`. That produces a multi-output annulus EM history under the repository size cap (`~4.6 MB`) and gives Step 4 a second benchmark-quality transient ladder beyond the slab benchmark.

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
  - [diagnose_recycling_target_cell_history.py](scripts/diagnose_recycling_target_cell_history.py) tracks target-adjacent cell histories and currently shows the earliest visible single-species drift appearing first in `NVd+` at `t = 25`, then `Nd`, then `Nd+` / `Pe`;
  - [diagnose_recycling_controller_history.py](scripts/diagnose_recycling_controller_history.py) tracks controller/source histories and currently shows that the PI-source drift is real but smaller than the target-cell state drift, so Step 2 is not blocked on the controller source in isolation;
- a more reference-like mutable-controller `bdf` callback path now exists in-tree and is materially faster on the single-species one-step probe, but it still lands at essentially the same first-output state error, so the remaining Step 2 defect survives both the continuation ladder and the faster BDF callback variant.
- The next Step 2 jump is no longer the RHS source stack; it is the transient stack: `recycling_1d_one_step`, `recycling_dthe_one_step`, then the corresponding short-window/long-run open-field cases on the shared implicit backbone.
- current transient status: the controller-complete recycling one-step branch is not yet parity-clean enough to expose, and the current adaptive BDF probe is still too slow for reviewer-safe production use on the staged one-step cases.
- Step 3 still depends on restaging a stable 2D recycling geometry target, because the currently named tokamak example remains blocked on the reference side.
  - `braginskii_thermal_force`
  - `braginskii_ion_viscosity`
- Step 3 should not continue from the broken tokamak example. The next curated 2D recycling target should be staged from the integrated `2D-recycling` workflow with its external artifact bundle handled explicitly by the harness.
