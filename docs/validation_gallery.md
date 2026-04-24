# Validation Gallery

This page collects the current public validation figures from the active
validation ladder. Each figure is generated from the same committed baseline
artifacts used by the regression harness, so the visuals and the automated
checks stay in sync.

The figure classes are chosen to match the main literature patterns used in
verification and edge/SOL validation papers: convergence curves and observed
orders in the style of Roy 2005 and the GBS parallel-gradient work, profile and
target comparisons in the style of TCV-X21, SOLPS-ITER, and Hermes-3 against
TCV-X21, and differentiable-science summaries closer to JAX-Fluids and related
JAX-based solver papers.

## Figure Status

| Figure | Status | Meaning |
| --- | --- | --- |
| `Diffusion Short Window` | `native-validated` | Transport baseline is locked. |
| `Electrostatic Vorticity Short Window` | `native-validated` | Electrostatic benchmark is locked. |
| `Coupled Drift-Wave One Step` | `native-validated` | First coupled 2D transient is locked. |
| `Drift-Wave Short-Window Parity` | `native-validated` | Drift-wave benchmark history is locked. |
| `Drift-Wave Short-Window Benchmark` | `native-validated` | Benchmark analysis is locked. |
| `Blob2d Short-Window Parity` | `native-validated` | Blob benchmark history is locked. |
| `Blob2d Meeting Movie` | `saved-payload visualization` | Fast one-step Blob2D visualization from a committed `.npz` payload. |
| `Diverted Tokamak Geometry Movie` | `benchmark-backed visualization` | Full-domain stitched tokamak turbulence movie with LCFS, wall, and divertor overlays. |
| `TCV-X21 Tokamak Scaffold` | `scaffolded_reference_backed` | First 3D tokamak kickoff package with a manifest-resolved preview path. |
| `Neutral Mixed Short-Window Benchmark Target` | `reference-only target` | Review artifact is staged; native transient is not yet promoted. |
| `Alfven-Wave Short-Window Benchmark` | `native-scaffolded target` | Electromagnetic transient benchmark is staged and benchmark-validated on the current scaffold. |
| `Hermes Live Rerun Matrix` | `live native vs live reference` | Same-machine native/Hermès rerun matrix across representative 1D and 2D lanes. |
| `Hermes Offender Register` | `triage artifact` | Ranked parity/runtime/memory offender register from the live rerun matrix and reduced geometry summary. |
| `Implicit Solver Profile Audit` | `numerical-performance audit` | Sparse finite-difference Jacobian plan and Newton phase diagnostics for the shared implicit backend. |
| `Open-Field Operator Campaign` | `operator-verified` | Parallel-gradient, force-balance, target-recycling, and autodiff checks are locked on a publication artifact. |
| `Neutral Mixed Term-Balance Campaign` | `operator-localization audit` | Native `NVh` term decomposition localizes the one-step Hermès mismatch. |

## Diffusion Short Window

![Diffusion short-window parity](images/diffusion_short_window_parity.png)

What this locks down:

- structured mesh reconstruction;
- metric normalization on the transport path;
- Neumann guard handling;
- repeated output scheduling over a short transient.

## Open-Field Operator Campaign

![Open-field operator campaign](data/open_field_operator_campaign_artifacts/images/open_field_operator_campaign.png)

What this locks down:

- second-order refinement of the centered parallel-gradient kernel;
- second-order refinement of the electron-force-balance operator with a
  nonzero momentum source;
- exact finite-volume target-recycling particle and energy source identities on
  the promoted open-field source formula;
- a differentiability check comparing `jax.grad` against a centered
  finite-difference sensitivity for the force-balance objective;
- a publication-ready operator-verification figure that can be reused in the
  JAXDRB paper before moving to longer Hermes reruns.

## Neutral Mixed Term-Balance Campaign

![Neutral mixed term-balance audit](data/neutral_mixed_term_balance_campaign_artifacts/images/neutral_mixed_term_balance_campaign.png)

What this locks down:

- named native `NVh` term decomposition for the neutral mixed momentum equation;
- backward-Euler residual-rate reconstruction on both native and Hermès-3 final states;
- a direct diagnostic separating native one-step residual closure from the remaining Hermès final-state mismatch;
- a publication-grade lineout/bar figure for the current neutral mixed offender.

## Electrostatic Vorticity Short Window

![Vorticity short-window parity](images/vorticity_short_window_parity.png)

What this locks down:

- discrete X-Z XPPM advection;
- Boussinesq potential inversion;
- repeated electrostatic output parity for both `Vort` and `phi`.

## Coupled Drift-Wave One Step

![Drift-wave one-step parity](images/drift_wave_one_step_parity.png)

What this locks down:

- coupled density, electron momentum, vorticity, and potential output;
- quasineutral electron closure;
- fixed-temperature electron pressure;
- trimmed active-cell comparisons for the first 2D density-vorticity benchmark.

## Drift-Wave Short-Window Parity

![Drift-wave short-window parity](images/drift_wave_short_window_parity.png)

What this locks down:

- the full 50-output reduced drift-wave transient on the committed benchmark grid;
- benchmark-level growth and frequency agreement on the same stored history used by the regression harness;
- field-error history for `Ni`, `Ne`, `NVe`, `Vort`, and `phi`, published from the same native/reference comparison artifact used in docs and review material;
- the current documented native/reference envelope: max `|Ni-Ne|` error about `1.47e-3`, max `|NVe|` error about `1.70e-4`, max `|Vort|` error about `2.14e-2`, and max `|phi|` error about `4.31e-4`.

## Drift-Wave Short-Window Benchmark

![Drift-wave short-window diagnostics](images/drift_wave_short_window_diagnostics.png)

What this locks down:

- benchmark postprocessing on the committed short-window array baseline;
- measured growth-rate and frequency extraction from the periodic density history;
- analytic finite-electron-mass dispersion evaluation from the same normalization and geometry scalars used by the run;
- documentation-ready reviewer figures backed by automated regression tests.

## Blob2d Short-Window Parity

![Blob2d short-window parity](images/blob2d_short_window_parity.png)

What this locks down:

- the full 50-output sheath-connected blob transient on the recalc-metric benchmark geometry;
- the optimized X-Z ExB transport kernel that made the native long-enough blob run practical without changing the discrete flux formulas;
- benchmark-level parity on summary blob diagnostics rather than only pointwise field maxima: peak density excess plus radial and binormal center-of-mass trajectories;
- the current documented native/reference envelope: peak-excess max error about `1.41e-2`, radial COM max error about `6.29e-1` active cells, and binormal COM max error about `7.32e-1` active cells.

## Blob2d Meeting Movie

![Blob2D meeting snapshots](images/blob2d_meeting_snapshots.png)

![Blob2D meeting poster](images/blob2d_meeting_movie_poster.png)

What this locks down:

- a fast saved-result visualization workflow using [examples/blob2d_meeting_demo.py](examples/blob2d_meeting_demo.py);
- a real 2D movie artifact from [blob2d_one_step.npz](references/baselines/reference_arrays/blob2d_one_step.npz), written to [docs/movies/blob2d_meeting_2d.mp4](docs/movies/blob2d_meeting_2d.mp4);
- a matching 3D surface movie and poster for presentation use;
- an explicit `--skip-parity` mode for saved payloads whose output timeline does not match the short-window parity metrics.

## Diverted Tokamak Geometry Movie

![Diverted tokamak poster](data/diverted_tokamak_turbulence_artifacts/images/diverted_tokamak_turbulence_poster.png)

![Diverted tokamak movie](data/diverted_tokamak_turbulence_artifacts/movies/diverted_tokamak_turbulence.gif)

What this locks down:

- a stitched full-domain visualization path for the exact `tokamak_turbulence_short_window` benchmark lane;
- direct use of the tokamak mesh geometry via `Rxy`, `Zxy`, and `psixy`;
- LCFS overlay from `psixy = 0`, plus explicit wall and divertor target curves;
- a summary 2D diverted tokamak figure package generated by [examples/diverted_tokamak_movie_demo.py](../examples/diverted_tokamak_movie_demo.py).

## TCV-X21 Tokamak Scaffold

![TCV-X21 scaffold poster](data/tokamak_tcv_x21_scaffold_artifacts/images/tokamak_tcv_x21_scaffold_poster.png)

![TCV-X21 scaffold profiles](data/tokamak_tcv_x21_scaffold_artifacts/images/tokamak_tcv_x21_scaffold_profiles.png)

What this documents:

- the first 3D tokamak kickoff package in the tree;
- manifest resolution for `tokamak_tcv_x21_escalation`;
- a structured deck/input report plus a benchmark-data report alongside the public figure bundle;
- an explicit benchmark validation contract covering FHRP, HFS-LP, and LFS-LP observable families;
- a shared observable report that describes those profile families on the generic 3D adapter schema;
- a staged profile report and compact NPZ bundle for those observable families, now populated from the public benchmark observable record;
- a summary profile summary plot derived from the same staged bundle;
- a public benchmark-data mode that can be regenerated without a heavy local 3D solve;
- LCFS, wall, and divertor overlays in the same diverted-geometry style as the existing tokamak visualizations, now driven by the public sample geometry/snapshot files.
- a reviewer-friendly bridge between the case manifest and the future selected 3D execution lane.
- the reduced selected-field parity package now exists as the next explicit gate after the scaffold bundle, on compact `Ne`/`Pe`/`phi` histories from either two 3D workdirs or the public TCV-X21 benchmark-data root plus a reproducible derived candidate.

## TCV-X21 Toroidal Movie

![TCV-X21 toroidal movie](data/tokamak_tcv_x21_toroidal_movie_artifacts/movies/tokamak_tcv_x21_toroidal.gif)

What this documents:

- a true toroidal 3D visualization path built from the committed scaffold arrays;
- outer-shell fluctuation coloring plus orthogonal poloidal cuts carrying the staged time dynamics;
- a clearer device-scale view for the README and docs surface than the flat scaffold slice GIF.

## Native Tokamak Selected-Field Rung

What this documents:

- the first reduced native 3D execution rung in the tree;
- a promoted native tokamak one-step case on the compact `Ne`/`Pe`/`phi` surface;
- parity JSON/NPZ artifacts on the same compact surface used by the benchmark-backed gate;
- a direct native-vs-reference selected-field history comparison bundle on the same committed artifact path;
- a shared observable report plus a runtime/provenance report for the native run.

## Traced-Field-Line Geometry Scaffold

![Traced-field-line scaffold metrics](data/traced_field_line_scaffold_artifacts/images/traced_field_line_scaffold_metrics.png)

![Traced-field-line scaffold lineouts](data/traced_field_line_scaffold_artifacts/images/traced_field_line_scaffold_lineouts.png)

![Traced-field-line scaffold slice summary](data/traced_field_line_scaffold_artifacts/images/traced_field_line_scaffold_slice_summary.png)

![Traced-field-line scaffold slice movie](data/traced_field_line_scaffold_artifacts/images/traced_field_line_scaffold_slice_movie.gif)

What this documents:

- the second 3D geometry-adapter scaffold in the tree;
- a non-diverted, traced-field-line geometry family riding on the same general 3D infrastructure;
- a reusable metric summary JSON and compact NPZ metric bundle;
- a shared observable report covering the line-diagnostic and selected-plane families on the same adapter schema;
- reusable radial, toroidal, and poloidal line diagnostics on the same artifact model;
- reusable selected-plane summaries and a first geometry-family GIF on the same artifact model;
- a summary metric summary plot that can be regenerated from either a synthetic preview or an explicit mesh/metric JSON specification;
- an explicit geometry-adapter validation contract before any native non-tokamak 3D claim is made.

## Traced-Field-Line Selected-Field Parity

![Traced-field-line selected-field parity](data/traced_field_line_selected_field_artifacts/images/traced_field_line_selected_field_parity.png)

What this documents:

- the first reduced parity gate on the second 3D geometry family;
- a compact selected-field compare surface on traced-field-line metric fields;
- shared observable-report output on the same geometry-adapter schema used by the scaffold packages;
- a non-tokamak counterpart to the tokamak selected-field parity lane;
- an explicit external-pair source report when the public external FCI sample is available locally.

## Traced-Field-Line Native Reduced Selected-Field

![Traced-field-line native selected-field parity](data/traced_field_line_native_selected_field_artifacts/images/traced_field_line_native_selected_field.png)

![Traced-field-line native selected-field comparison](data/traced_field_line_native_selected_field_artifacts/images/traced_field_line_native_selected_field_comparison.png)

What this documents:

- the first native reduced rung on a non-tokamak 3D geometry family;
- a JAX-native radial-profile reduction on explicit traced-field-line metric pairs;
- the same public parity, comparison, observable, and runtime artifact surfaces used by the native tokamak reduced rung;
- an honest bridge between external-pair validation-only geometry adapters and future broader native non-tokamak execution work.

## Stellarator VMEC Scaffold

![Stellarator VMEC profiles](data/stellarator_vmec_scaffold_artifacts/images/stellarator_vmec_scaffold_profiles.png)

![Stellarator VMEC surface summary](data/stellarator_vmec_scaffold_artifacts/images/stellarator_vmec_scaffold_surface_summary.png)

![Stellarator VMEC surface movie](data/stellarator_vmec_scaffold_artifacts/images/stellarator_vmec_scaffold_surface_movie.gif)

What this documents:

- the third 3D geometry adapter in the tree;
- a VMEC-style stellarator equilibrium bundle on the same manifest and observable schema as the tokamak and traced-field-line adapters;
- reusable profile diagnostics for `iota`, `pressure`, and `toroidal_flux`;
- sampled `R`/`Z` flux-surface cross-sections across toroidal angle, with a summary summary figure and a compact movie path;
- a pressure test for the general 3D artifact model on a geometry family that is neither diverted tokamak nor FCI-grid-only.

## Stellarator VMEC Selected-Field Parity

![Stellarator VMEC selected-field parity](data/stellarator_vmec_selected_field_artifacts/images/stellarator_vmec_selected_field_parity.png)

What this documents:

- the first reduced selected-field parity gate on the VMEC-style stellarator family;
- a compact compare surface on `iota`, `pressure`, and `toroidal_flux`;
- the same shared observable-report and source-report contract used by the other 3D parity packages;
- a real explicit external VMEC-pair regeneration path in addition to the synthetic preview mode.

## Stellarator VMEC Native Reduced Selected-Field

![Stellarator VMEC native selected-field parity](data/stellarator_vmec_native_selected_field_artifacts/images/stellarator_vmec_native_selected_field.png)

![Stellarator VMEC native selected-field comparison](data/stellarator_vmec_native_selected_field_artifacts/images/stellarator_vmec_native_selected_field_comparison.png)

What this documents:

- the second native reduced rung on a non-tokamak 3D geometry family;
- a JAX-native profile reduction on `iota`, `pressure`, and `toroidal_flux`;
- the same parity, comparison, observable, and runtime artifact surface used by the traced-field-line native rung;
- a stronger general-geometry 3D story than a single non-tokamak native proof point.

## Native 3D Runtime Campaign

![Native 3D runtime campaign](data/native_3d_runtime_campaign_artifacts/images/native_3d_runtime_campaign.png)

What this documents:

- one explicit runtime/scaling summary across the promoted native reduced 3D matrix;
- committed runtime reports from the native tokamak, traced-field-line, and stellarator reduced rungs;
- compact synthetic scaling sweeps for the non-tokamak native reduction kernels;
- the current engineering blocker after geometry diversity is no longer the main missing surface.

## Native 3D Convergence Campaign

![Native 3D convergence campaign](data/native_3d_convergence_campaign_artifacts/images/native_3d_convergence_campaign.png)

What this documents:

- an explicit operator-level convergence gate on the promoted traced-field-line native reduction path;
- observed refinement order against an analytic radial-average target rather than only parity and runtime numbers;
- the first committed convergence bundle specific to the reduced native 3D matrix itself.

## Fluid 1D MMS Convergence

![Fluid 1D MMS convergence](data/fluid_1d_mms_convergence_artifacts/images/fluid_1d_mms_convergence.png)

What this documents:

- an explicit manufactured-solution refinement bundle for the promoted 1D fluid density, pressure, and momentum operators;
- per-resolution L2 errors and observed orders on the same native lane used for the compact verification tests;
- a literature-anchored verification figure that can be reused in docs and future paper surfaces instead of leaving the result trapped in a script output.

## JAX Native Profile Audit

![JAX native profile audit](data/jax_native_profile_audit_artifacts/images/jax_native_profile_audit.png)

What this documents:

- compile, first-execute, and warm-execute timings on the promoted traced-field-line and stellarator reduced native kernels;
- the first committed Perfetto-compatible trace bundle for the reduced native JAX surfaces;
- the concrete engineering conclusion from that profiling pass: batch same-shape selected fields before entering jitted reductions, and warm kernels once before timing summary runs.

This is an engineering-support figure, not a primary validation figure. It is
useful because differentiable/JAX papers routinely report compile-versus-execute
cost, but it should stay secondary to the physics and verification surfaces.

## Implicit Solver Profile Audit

![Implicit solver profile audit](data/implicit_solver_profile_audit_artifacts/images/implicit_solver_profile_audit.png)

What this documents:

- a controlled sparse finite-difference Jacobian assembly audit before the full
  recycling physics stack is involved;
- algebraic agreement between the original colored finite-difference path and
  the precomputed CSC/color extraction-plan path;
- agreement between serial and batched JAX sparse-JVP Jacobian construction,
  with the JAX path checked against the finite-difference reference;
- sparse Newton phase diagnostics for residual evaluation, Jacobian assembly,
  linear solve, and line search;
- the numerical-methods support figure needed before making stronger runtime
  claims on `recycling_dthe_one_step`.

## Local CPU Scaling Campaign

![Local CPU scaling campaign](data/local_cpu_scaling_campaign_artifacts/images/local_cpu_scaling_campaign.png)

What this documents:

- a reviewer-facing local CPU scaling result on a real promoted heavy recycling solve rather than a tiny synthetic kernel;
- a heavier fixed-work ensemble with `16` repeated heavy solves, so launch and warmup overhead are better amortized;
- the stronger local throughput story users actually care about for UQ, optimization, and parameter scans: repeated heavy solves scale well across local worker processes after warmup;
- on the committed artifact the steady-state speedup is about `1.88x`, `3.67x`, and `4.94x` at `2`, `4`, and `8` workers;
- the retained `16`-solve ensemble was chosen deliberately because heavier local sweeps did not improve the curve on this MacBook;
- the operational recommendation for laptop users: keep one Jacobian thread per worker for batched heavy solves and use multiple local workers when the workload is naturally parallel.

This is the current research-grade local performance figure because it matches
the actual workload pattern used in UQ, optimization, and repeated parameter
scans better than a tiny-kernel strong-scaling plot.

## Hermes Comparison Summary

![Hermes comparison summary](data/hermes_comparison_summary_artifacts/images/hermes_comparison_summary.png)

What this documents:

- one benchmark-facing summary plot across the committed native-vs-reference reduced comparison bundles;
- direct comparison of the native tokamak, traced-field-line, and stellarator reduced rungs on the same visual surface;
- a simpler summary entry point than asking readers to inspect each lane-specific comparison artifact separately.

This is intentionally a supporting summary figure. It should remain in the docs
as an index into the lane-specific artifacts, but the future paper should rely
primarily on the lane-specific comparison figures and the benchmark/closure
campaigns rather than on this rollup alone.

## Controller Feedback Campaign

![Controller feedback campaign](data/controller_feedback_campaign_artifacts/images/controller_feedback_campaign.png)

What this documents:

- the first reference-backed controller-history gate on a promoted native feedback path;
- dense-history comparison of controller multiplier, proportional term, integral term, reconstructed controller integral, and target recycling source;
- a controller-oriented closeout surface that is honest about stopping short of a full detachment-controller claim.

## Detachment Controller Campaign

![Detachment controller campaign](data/detachment_controller_campaign_artifacts/images/detachment_controller_campaign.png)

What this documents:

- the first genuinely bounded Hermes-backed `detachment_controller` lane on the local non-PETSc reference build;
- a reduced `cvode` deck that strips the incompatible `beuler`-only solver options and forces `settling_time = 0` so the bounded window actually exercises the controller;
- exact saved-diagnostic checks on the proportional term, controller balance, and `source_multiplier * source_shape`, plus a nontrivial response-span gate.

## Reactions And Collisions Campaign

![Reactions and collisions campaign](data/reactions_collisions_campaign_artifacts/images/reactions_collisions_campaign.png)

What this documents:

- a dedicated engineering and physics gate for reactions, collisions, and atomic-data breadth;
- explicit native checks for charge exchange, isotope coupling, CX multipliers, ionisation, ion-viscosity collisionality closure, and neon OpenADAS loading;
- profile-level lineouts for ionisation, isotope-resolved charge exchange, and collisionality closure, so the same validated surface can support future manuscript figures;
- a machine-readable JSON/NPZ/plot package instead of leaving this breadth implied only by low-level unit tests.

## Atomic Rate Differentiability Campaign

![Atomic rate differentiability campaign](data/atomic_rate_differentiability_campaign_artifacts/images/atomic_rate_differentiability_campaign.png)

What this documents:

- a derivative-validation figure for the AMJUEL, OpenADAS, and hydrogen
  charge-exchange rate surfaces used by the reaction-source path;
- autodiff slopes with respect to log temperature compared directly against
  centered finite differences;
- a concrete gate for the newly JAX-preserving atomic-rate helpers before
  those rates are used inside a full JAX-native recycling residual;
- a paper-ready differentiability figure that complements the existing
  reaction/collision physics campaign.

## Neutral Parallel Diffusion Campaign

![Neutral parallel diffusion campaign](data/neutral_parallel_diffusion_campaign_artifacts/images/neutral_parallel_diffusion_campaign.png)

What this documents:

- a dedicated closure study for the extracted neutral parallel-diffusion family on a prepared multispecies D/T/He recycling state;
- direct comparison between the `AFN` and `multispecies` collision modes on the same state, following the neutral-model distinction documented by Hermes-3;
- species-level summaries for the effective neutral diffusivity and the collision-budget decomposition that explains the AFN-versus-multispecies split;
- a machine-readable JSON/NPZ/plot package so this closure evidence is not trapped only inside operator tests.

## Collision Closure Campaign

![Collision closure campaign](data/collision_closure_campaign_artifacts/images/collision_closure_campaign.png)

What this documents:

- a dedicated closure study for the extracted Braginskii-style collision, viscosity, and conduction family on a prepared multispecies D/T/He recycling state;
- species-resolved ion-viscosity activity, representative collisional-friction activity, and conduction collision times on the same state;
- direct action-reaction checks for selected friction pairs, so the public artifact is tied back to a basic physical consistency condition rather than only to nonzero activity;
- a machine-readable JSON/NPZ/plot package so these closure diagnostics can feed the future paper directly instead of remaining trapped inside low-level tests.

## Tokamak Anomalous Diffusion Campaign

![Tokamak anomalous diffusion campaign](data/tokamak_anomalous_diffusion_campaign_artifacts/images/tokamak_anomalous_diffusion_campaign.png)

What this documents:

- a dedicated tokamak operator study for the extracted anomalous-diffusion family on the evolved D/T/He recycling state used by the direct tokamak validation ladder;
- direct comparison between orthogonal and non-orthogonal tokamak metrics on the same evolved state, so the geometry effect is visible instead of being implied only by low-level assertions;
- species-resolved anomalous coefficient summaries together with representative `d+` and `t+` anomalous-energy lineouts on the active direct-tokamak state;
- a machine-readable JSON/NPZ/plot package so this non-orthogonal transport evidence can feed the docs and future paper directly.

## Target Recycling Campaign

![Target recycling campaign](data/target_recycling_campaign_artifacts/images/target_recycling_campaign.png)

What this documents:

- a prepared-state audit for the extracted target-recycling and current-free electron-velocity support layer on the multispecies `1D-recycling-dthe` lane;
- target recycling density-source lineouts and integrated source totals for the neutral `d`, `t`, and `he` channels;
- the boundary-conditioned electron energy sink and current-free electron-velocity reconstruction on the same prepared state;
- a machine-readable JSON/NPZ/plot package so this boundary/recycling evidence can feed the docs and future paper directly.

## Hermes Live Rerun Matrix

![Hermes live rerun campaign](data/hermes_live_rerun_campaign_artifacts/images/hermes_live_rerun_campaign.png)

What this documents:

- a same-machine native-versus-live-reference rerun matrix across representative
  curated 1D and 2D lanes, instead of relying only on committed reference
  arrays;
- a code-to-code validation figure that follows the literature pattern of
  showing fidelity and runtime together rather than publishing an isolated
  dashboard;
- four exact-match lanes on the current compare surface:
  `tokamak_isothermal_one_step`, `tokamak_turbulence_one_step`,
  `tokamak_diffusion_transport_short_window`, and `annulus_he_emag_one_step`;
- the current most difficult live one-step lane in the selected matrix:
  `neutral_mixed_one_step`, with worst RMS error normalized by reference
  amplitude about `9.17e-1` and native/reference wall-time ratio about `2.93`;
- heavy 1D recycling lanes that are closer in fidelity but still slower than
  Hermès-3 on this machine:
  `recycling_1d_one_step` and `recycling_dthe_one_step`, with worst normalized
  RMS errors about `4.62e-3` and `4.92e-3`, and runtime ratios about `3.65`
  and `7.82`;
- integrated and direct tokamak recycling one-step lanes that are already close
  to wall-time parity or faster on this machine, but their current relative
  mismatch is dominated by near-zero `NVd` on the guarded compare surface; the
  corresponding worst absolute max-errors stay small at about `7.48e-12` and
  `3.09e-7`.
- process-tree peak RSS is now sampled during each native and Hermès run; the
  largest native peak is about `722 MiB` on the integrated 2D recycling lane,
  while the largest native/Hermès peak-RSS ratio is about `0.95` on
  `recycling_dthe_one_step`.

This is the current main live code-to-code validation figure for the docs. It
also shows the honest remaining gap: full live 3D Hermès reruns are still not
part of this matrix, so the 3D evidence remains the selected-field
reference-backed packages.

## Hermes Offender Register

![Hermes offender register](data/hermes_offender_register_artifacts/images/hermes_offender_register.png)

What this documents:

- a ranked triage artifact that turns the live rerun matrix into concrete next
  debugging targets;
- the current top parity offender: `neutral_mixed_one_step` on `NVh`, pointing
  to neutral mixed boundary and parallel momentum closure;
- the current top runtime offender: `recycling_dthe_one_step`, pointing to
  sparse Jacobian, residual, pack/unpack, and target-recycling closure
  profiling;
- memory is now ranked from measured process-tree peak RSS; the current top
  ratio is `recycling_dthe_one_step` at about `0.95`, so the next memory step
  is phase-resolved profiling rather than broad peak-RSS triage;
- near-zero normalized `NVd`/`NVt` mismatches are explicitly flagged so
  absolute error is inspected before changing equations.

## Neutral Mixed Boundary Audit

![Neutral mixed boundary campaign](data/neutral_mixed_boundary_campaign_artifacts/images/neutral_mixed_boundary_campaign.png)

What this documents:

- a literature-style parallel-profile follow-up to the live rerun matrix rather
  than another scalar dashboard;
- Hermès-3 versus JAX-DRB lineouts at the `x,z` locations where `Nh`, `Ph`,
  and `NVh` attain their worst one-step absolute error;
- a `max_{x,z} |Δ|(y)` panel that makes the parallel localization of the
  remaining mismatch visible across the whole compare surface;
- the current one-step worst-field numbers on this focused surface:
  `Nh max |Δ| ≈ 1.07e-2`, `Ph max |Δ| ≈ 8.65e-4`, and
  `NVh max |Δ| ≈ 3.37e-3`, with a native/reference runtime ratio about `3.12x`.

This is the right follow-up figure for the current neutral lane because it
turns the remaining live rerun mismatch into a physical profile question that
can be tied back to closures and boundary treatment.

## Temperature Feedback Campaign

![Temperature feedback campaign](data/temperature_feedback_campaign_artifacts/images/temperature_feedback_campaign.png)

What this documents:

- a bounded reduced Hermes-backed temperature-control lane on `1D-recycling-with-Tt-control`;
- exact saved-diagnostic balance on the controller multiplier, proportional term, and `source_multiplier * source_shape`;
- bounded output-time integral reconstruction and visible target-temperature error reduction on the reduced `cvode` window;
- an auto-patched clean reference-worktree path for local Hermes trees that still carry the known `temperature_feedback.hxx` permission bug.

## Autodiff Diffusion Uncertainty

![Autodiff diffusion uncertainty](data/autodiff_diffusion_uncertainty_artifacts/images/autodiff_diffusion_uncertainty.png)

What this documents:

- a standard uncertainty-propagation example on the same compact native differentiable diffusion lane used for sensitivity and inverse design;
- a scalar QoI based on the final active-domain density variance plus a field QoI based on the final radial profile;
- agreement between first-order autodiff covariance pushforward and a vectorized Monte Carlo estimate on the same native solve path.

This is currently the strongest differentiable-science figure in the public
surface because it presents a standard uncertainty-propagation comparison rather
than only raw gradients.

## Impurity And Radiation Campaign

![Impurity and radiation campaign](data/impurity_radiation_campaign_artifacts/images/impurity_radiation_campaign.png)

What this documents:

- the first explicit impurity/radiation validation bundle rather than only an audit placeholder;
- neon OpenADAS ionisation/recombination loading plus finite radiation-loss evaluation on the native path;
- exact direct tokamak `D/T/He/Ne` RHS closure on the committed `Nne+`, `Pne+`, and `Pe` compare surface;
- an honest claim boundary that still leaves controller-oriented temperature/detachment workflows open.

## Neutral Mixed Short-Window Benchmark Target

![Neutral mixed short-window diagnostics](images/neutral_mixed_short_window_diagnostics.png)

What this locks down:

- a compact reference-side transient target for the staged neutral branch before the native stiff solver is promoted through the public runner;
- center-probe histories for `Nh`, `Ph`, and `NVh` at the committed benchmark location `(x=5, y=3, z=5)`;
- the derived center temperature `Ph / Nh`, which stays close to the expected `0.1` throughout the short window;
- summary compact metrics rather than large raw arrays: final total `Nh` about `7.86197875e+02`, final total `Ph` about `7.86184063e+01`, and final momentum RMS about `5.56121767e-08`.

## Alfven-Wave Short-Window Benchmark

![Alfven-wave short-window diagnostics](images/alfven_wave_short_window_diagnostics.png)

What this locks down:

- the first multi-output electromagnetic transient rung on the finite-electron-mass slab benchmark;
- benchmark-quality phase-speed extraction from the committed `nout=20` history;
- analytic-vs-measured validation on the same stored arrays used by the regression harness;
- the current documented benchmark numbers:
  - analytic phase speed about `9.48585409e+05 m/s`;
  - measured phase speed about `9.42218662e+05 m/s`;
  - relative phase-speed error about `6.71e-03`;
- native/reference parity on that same short-window history, published from the exact committed comparison artifact.

The same short-window rung now also drives the meeting-ready visual package in [docs/alfven_wave_meeting_demo.md](docs/alfven_wave_meeting_demo.md), which adds 2D and 3D movies plus a snapshot panel from a live native run.

## Regeneration

These figures are generated from the committed baseline arrays plus native case runs. The current gallery uses:

- `diffusion_short_window`
- `vorticity_short_window`
- `drift_wave_one_step`
- `drift_wave_short_window`
- `blob2d_short_window`
- `blob2d_one_step` saved-payload visualization
- `alfven_wave_short_window`
- `tokamak_turbulence_short_window` stitched full-domain geometry visualization
- live Hermès reruns across the representative 1D and 2D curated matrix
- atomic-rate differentiability audit for AMJUEL, OpenADAS, and hydrogen
  charge exchange

The next gallery pass should add:

- periodic 1D fluid short-window figures;
- native neutral transient parity figures once the stiff `neutral_mixed` path is benchmark-clean;
- benchmark and validation plots for additional integrated 2D production and broader EM cases as those stages land.
