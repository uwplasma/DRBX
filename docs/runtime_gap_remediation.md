# Runtime And Fidelity Gap Remediation

!!! note "Plan authority"
    This page is a subordinate technical appendix for runtime and parity
    offender details. The active execution plan is
    [research_grade_execution_plan.md](research_grade_execution_plan.md). If
    this page conflicts with that plan, follow the execution plan and update
    this appendix afterward.

This document records the current worst runtime and native-versus-Hermès
mismatch cases that feed the consolidated execution plan.

It is based on the refreshed same-machine live rerun matrix in:

- [hermes_live_rerun_campaign.md](hermes_live_rerun_campaign.md)

and on the current profiling workflow in:

- [profiling_runtime.md](profiling_runtime.md)

## Current Highest-Priority Cases

The refreshed live matrix identifies four distinct categories.

### 1. Real open-field neutral mismatch

- case: `neutral_mixed_one_step`
- current worst normalized RMS error: about `8.90e-4`
- current relative-L2 error on the dominant field: about `2.14e-3`
- current runtime ratio: about `4.19x`
- dominant field: `NVh`

This is now a small absolute-error regression surface rather than the clearest
current fidelity gap. The focused follow-up figure is now in
[neutral_mixed_boundary_campaign.md](neutral_mixed_boundary_campaign.md), which
shows the worst-error `Nh`, `Ph`, and `NVh` lineouts plus the
`max_{x,z} |Δ|(y)` profile on the same live rerun surface. The next diagnostic
layer is now in
[neutral_mixed_term_balance_campaign.md](neutral_mixed_term_balance_campaign.md):
it inserts both the native and Hermès-3 final states into the native
neutral-mixed momentum operator and decomposes the `NVh` residual-rate into
parallel inertia, pressure gradient, perpendicular diffusion, parallel
viscosity, and perpendicular viscosity.
The latest term-level reconstruction shows that the native pressure-gradient
and viscosity formulas close against the written source diagnostics to near
roundoff; the remaining mismatch is now target-adjacent state and boundary
evolution, with current maximum final-field errors about `2.19e-4` on `Nh`,
`2.11e-5` on `Ph`, and `4.47e-6` on `NVh`.

### 2. Heavy 1D recycling runtime bottleneck

- case: `recycling_1d_one_step`
- current worst normalized RMS error: about `4.62e-3`
- current runtime ratio: about `3.95x`
- dominant normalized field: `Pd+`

This lane is already tight in fidelity and still one of the main runtime
offenders.

### 3. Heavy multispecies 1D recycling bottleneck

- case: `recycling_dthe_one_step`
- current worst normalized RMS error: about `4.92e-3`
- current runtime ratio: about `7.17x`
- dominant field: `NVd`

This is the current worst runtime ratio in the live matrix and the main
production-path runtime target. The latest target-boundary geometry caching
pass reduced the local timed run from about `54.1 s` to about `52.76 s`
without changing the fidelity band.
After the metric-selector regression fix and fixed-layout bridge promotion, the
fresh profiling bundle measured `74.33 s` under cProfile and `49.22 s` on the
separate RSS run, with peak process-tree RSS about `231.4 MiB`. The cProfile
wall time is inflated by profiling overhead on this Python-heavy path; the
useful result is the split. Sparse finite-difference Jacobian assembly still
consumes about `51.2 s` cumulative, while the packed RHS is evaluated `11838`
times and remains the dominant repeated host-side workload.

### 4. Near-zero normalized tokamak recycling mismatch

- cases:
  - `integrated_2d_recycling_one_step`
  - `tokamak_recycling_one_step`
- current normalized RMS errors: about `1.78e-1` and `1.62e-1`
- current runtime ratios: about `1.46x` and `0.47x`
- dominant field in both cases: `NVd`
- current worst absolute max-errors:
  - integrated: about `7.48e-12`
  - tokamak: about `3.09e-7`

These are important, but they are not the same class of issue as the neutral
mixed mismatch. The current public figure now marks them as
normalization-sensitive because the relative metric is dominated by a near-zero
reference field.

## Current Measured Runtime Improvements

The current optimization pass already changed the runtime picture materially.

### Neutral mixed

- earlier local timed mean: about `1.15 s`
- current local timed mean: about `0.63 s`
- current live runtime: about `1.52 s`
- remaining issue: target-adjacent `Nh`/`Ph` state-history drift, with `NVh`
  now small in absolute error after the term-level closure pass

### 1D recycling

- current live runtime: about `14.77 s`
- current runtime ratio to live reference: about `3.95x`
- remaining issue: sparse FD Jacobian plus repeated heavy RHS evaluation

## Current Root-Cause Hypotheses

The current bottlenecks split into two classes.

### Runtime bottlenecks

- sparse finite-difference Jacobian assembly in the implicit backbone
- repeated host-side RHS/source assembly in `recycling_1d.py`
- remaining host/SciPy structure in the heavy transient path
- repeated per-run compilation or warmup cost on GPU if cache is not enabled

### Fidelity bottlenecks

- neutral mixed closure/operator mismatch concentrated on `NVh`
- near-zero-field normalization on `NVd` in the integrated/direct tokamak
  one-step compare surfaces
- possible closure or initial-state differences between native and Hermès on
  the heavy open-field neutral/recycling ladders

## Near-Term Remediation Order

### Priority 1: neutral mixed fidelity

1. profile `neutral_mixed_one_step` with the public profiling script
2. isolate the `NVh` residual terms and compare them term-by-term against the
   Hermès lane
3. keep the new live rerun boundary-audit package and extend it term-by-term
   for the offending neutral terms
4. lock the fix with a direct regression test plus the same paper-grade figure

The term-balance package now completes item 2 on the native side. It shows that
the native final state nearly satisfies the native backward-Euler residual, but
the Hermès-3 final state does not. The next neutral fidelity fix should
therefore inspect Hermès-3's corresponding pressure-gradient and viscosity
operator outputs, not another aggregate field-error plot.

### Priority 2: multispecies recycling runtime

1. profile `recycling_dthe_one_step` with:
   - cProfile
   - JAX trace
   - XLA dump
2. use the sparse Newton phase diagnostics to split the runtime into residual
   evaluation, finite-difference Jacobian refresh, linear solve, line search,
   and fallback time
3. identify the current Jacobian/RHS hot splits after the recent Horner,
   vectorization, cached-geometry, and precomputed color-plan improvements
4. cut repeated source/closure recomputation where possible
5. keep the current fidelity band unchanged while reducing runtime

The latest local cProfile/RSS pass sharpens this target. After AMJUEL log-input
reuse and BDF Jacobian-plan reuse, a one-run cProfile measurement took about
`67.00 s`, while the separate RSS run took about `53.37 s` and peaked at about
`229.4 MiB`. The cProfile split is still dominated by SciPy BDF stepping,
packed RHS calls, and sparse finite-difference Jacobian construction, with
reaction-source/AMJUEL evaluation, neutral parallel advection, collision
closure, state preparation, and target recycling as the next source-level
offenders. This argues for fewer host residual calls, stronger source/closure
caching, and a JAX-native residual/JVP lane before spending more effort on
generic local-thread tuning.

The latest source-kernel cleanup confirms that ordering. With the packed D/T/He
residual routed through the fixed-layout reaction-source kernel and D/T AMJUEL
fits reused inside both source and neutral-ionisation rate assembly, a refreshed
one-run cProfile measured `64.45 s` and the separate RSS run measured `50.00 s`
with peak RSS near `232.7 MiB`. Fixed-layout reaction sources fell to about
`9.64 s`, neutral-ionisation collision-rate assembly to about `2.72 s`, and
AMJUEL fit evaluations to `117380` calls. Sparse finite-difference Jacobian
assembly still consumed about `43.3 s`, so the remaining high-value work is
residual-call reduction and JAX-linearized Jacobian products rather than more
localized reaction-source tuning.

The latest backend-selector correction keeps that production path on NumPy
unless a dynamic state or rate array is actually a JAX value. This matters
because `StructuredMetrics` currently stores concrete JAX arrays, and those
static metrics had briefly forced eager JAX execution inside hot open-field
operators. The measured D/T/He packed RHS returned from about `8e-2 s` per warm
call to about `4e-3 s`, and the bounded current-code
`recycling_dthe_one_step` timing completed in `44.60 s` with the promoted
parity gate passing in `44.66 s`. This is now the metric-selection rule for all
future residual ports: dynamic arrays select the backend; static metrics are
converted inside the selected branch.

The fresh full profile after that fix keeps the remediation order unchanged but
quantifies it more sharply: BDF still requests `86` Jacobian callbacks, which
expand to `8428` colored perturbation residuals, and the source-level split is
now collision closure, fixed-layout reactions, open-field state preparation,
parallel advection, AMJUEL evaluation, ion/electron/neutral RHS assembly,
target recycling, and neutral parallel diffusion. The work should therefore
continue by moving those residual pieces into fixed-layout JAX kernels before
promoting matrix-free or sparse-JVP solves as the default heavy backend.
For each preconditioner or JAX-linearized residual candidate, the gate should
record both work and achieved linear-update quality. The compare/profile
scripts now expose maximum absolute and relative `J v + r` update-residual
ceilings, so a candidate can be rejected if it preserves the final nonlinear
residual only by spending the same Krylov/operator budget or by accepting a
poor linear update. These gates are advisory for current opt-in solvers and
required evidence for any future default-promotion claim.

The next call-count cleanup is now in-tree: the SciPy BDF callback caches the
most recent exact RHS evaluation and reuses it as the base state for
`jac(t, y)` when SciPy requests the Jacobian at that same state. The history
result exposes BDF RHS evaluation, cache-hit, Jacobian-callback, and
Jacobian-worker counters. The BDF callback now also sends finite-difference
perturbation residuals around the mutable RHS cache and honors
`JAX_DRB_FD_JACOBIAN_THREADS=<N>` for explicit local CPU threading.
This closes an avoidable duplicate-base-RHS path, but it should be treated as
instrumentation and cleanup rather than a solved performance milestone; a
post-change one-run `recycling_dthe_one_step` timing measured `61.38 s`, which
does not establish an end-to-end speedup against the previous noisy local run.
The follow-up timing-only check reinforces that point: serial, two-thread, and
four-thread BDF runs measured about `50.00 s`, `49.81 s`, and `54.57 s`,
respectively, so one-solve threading is not the reviewer-grade scaling result
for this host-backed residual.
The latest phase-resolved RSS replay again measured about `48.82 s`, with
`46.41 s` in fixed-layout RHS object evaluation, `33.60 s` in Jacobian
callbacks, and only about `2e-3 s` in RHS NumPy conversion. This rules out
host-array conversion as the primary target for the next pass.
The stronger next step remains a JAX-transformable residual plus grouped
JVP-based Jacobian products for the dominant recycling kernels.

That residual path now has a concrete first production gate. A real hydrogen
`1D-recycling` backward-Euler step with the fixed-layout residual reaches the
JAX-linearized solver without crossing the previous species, sheath,
charge-exchange, feedback, or RHS-pack host barriers. The gate is intentionally
small-step so it proves transformability rather than a physical transient. The
current cProfile/RSS/JAX-trace artifact for this gate lives under
`docs/data/runtime_profile_artifacts/recycling_1d_jax_linearized_gate/` and
records residual `2.49e-12`, one JAX linearization refresh, one residual
evaluation, and no fallback. The remaining runtime lane is to repeat the same
barrier removal for the full adaptive BDF callback and for the D/T/He branch,
then refresh the heavy cProfile/RSS/JAX-trace bundle before promoting grouped
JVP or matrix-free solves as a default backend.

The solver package now has the first production-tested piece of that JAX path:
the grouped sparse-JVP Jacobian builder batches colored tangent pushes with
`jax.vmap` after a single `jax.linearize` call. This is the correct derivative
algorithm for a JAX residual because it removes finite-difference step-size
choice and perturbation residual calls. It does not by itself fix the heavy
recycling lane because the current residual is still dominated by NumPy/SciPy
assembly. The next runtime fixes should therefore be ordered as residual-kernel
ports first, solver-backend promotion second.

The BDF compatibility callback now exposes this algorithm through
`JAX_DRB_RECYCLING_BDF_JACOBIAN_MODE=jvp`, with
`JAX_DRB_RECYCLING_JVP_BATCH_SIZE` controlling tangent batching. This closes
the API lane needed to test JVP Jacobian construction on a full history solve,
but the default remains finite difference. The reason is empirical and should
stay visible in the paper plan: bounded full-output adaptive JAX-linearized
D/T/He trials are not yet stable or fast enough to replace the validated BDF
path, while the fixed-layout backward-Euler D/T/He gate is stable and
transformable. The next promotion criterion is therefore a full output-window
run that passes the live-reference compare surface and shows lower residual
call count or memory than the finite-difference BDF callback.

The current named gate for that comparison is
`runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp`. It preserves
the SciPy BDF timestepper but changes the callback seam to a fixed-full-field
RHS and grouped-JVP sparse Jacobian, with diagnostics reporting
`bdf_rhs_backend`, `bdf_rhs_object_evaluation_seconds`,
`bdf_rhs_numpy_conversion_seconds`, and `bdf_jacobian_mode`. This is the route
to profile before attempting a matrix-free or Lineax replacement for the full
output-window solve.

The compact real gate now passes as a parity check but fails as a promotion
candidate. On `recycling_1d_one_step`, the command
`compare_recycling_transient_modes.py --mode bdf --mode
bdf_fixed_full_field_jvp --field Pe --require-fixed-jvp-diagnostics
--require-bdf-pairwise-max 1e-5` reports an active-mesh `Pe` pairwise delta of
`6.28e-6`, so the fixed-layout callback is following the same physics to the
requested tolerance. The runtime split is the blocker: the fixed-JVP path takes
about `59.9 s` versus about `8.2 s` for default BDF, with about `57.1 s` spent
inside JVP Jacobian callbacks. The subphase split is now measured: about
`36.8 s` is repeated `jax.linearize`, about `20.0 s` is the batched tangent
push, and sparse assembly plus tangent construction are below `0.02 s`. The
next optimization is therefore to avoid repeated full JAX linearization/JVP
materialization inside SciPy BDF, or to move the output-window solve to a native
JAX-linearized/matrix-free nonlinear solver after parity is preserved.

The June 2026 self-contained promotion gate confirms the same conclusion on
both the hydrogen and D/T/He fixture decks. The hydrogen gate matched default
`bdf` within `7.20e-6` but took about `62.7 s` versus `9.07 s`; the D/T/He gate
matched within `1.02e-6` but took about `195.7 s` versus `62.6 s`. In the
D/T/He fixed-JVP run, repeated JAX linearization consumed about `113.0 s` and
tangent pushes about `63.4 s`, while sparse assembly and host transfer were
negligible. This is strong parity evidence for the fixed-layout callback seam,
but it is negative performance evidence for making `bdf_fixed_full_field_jvp`
the default output-window solver.

The sparse Newton interface now exposes that derivative algorithm directly as
`jacobian_mode="jvp"`, and the implicit-solver profile audit compares it
against the finite-difference sparse Newton path on a transformable residual.

The next low-risk runtime patch is intentionally smaller than a solver-default
change: `solve_jax_linearized_newton_system` now evaluates the initial residual
before calling `jax.linearize` and returns immediately when the predictor
already satisfies the nonlinear tolerance. This removes wasted linearization on
accepted predictor states while preserving the existing JAX-linearized Newton
path for genuinely nonlinear updates. It does not change the production BDF
default or make the slower fixed-JVP output-window path default.
The same solver now also exposes an opt-in `jit_residual=True` path. That path
wraps the residual in `jax.jit` before the initial residual check and
`jax.linearize`, then records `residual_jitted=True` in the returned
`ImplicitStepInfo`. This is a seam for fixed-layout recycling profiling and
future matrix-free promotion; it remains opt-in until full output-window parity
and runtime gates justify changing a production route.
The recycling BE/BDF2 wrappers expose the same seam through
`runtime:recycling_jax_linear_jit_residual=true` or
`JAX_DRB_RECYCLING_JAX_LINEAR_JIT_RESIDUAL=1`, so profiling decks can pin the
pre-JIT behavior without changing the finite-difference BDF default.
They also expose opt-in JAX-GMRES row-scaling and JVP-derived probes through
`runtime:recycling_jax_linear_preconditioner=state_scale`, `field_scale`,
`linearized_diag`, `field_diag`, `local_block_diag`, `neutral_line`,
`momentum_line`, or `sheath_line` and the matching
`JAX_DRB_RECYCLING_JAX_LINEAR_PRECONDITIONER`
environment variable. The
`linearized_diag` path samples the full packed-state diagonal and is therefore
bounded separately from cheaper field-only probes. The
`field_diag` path samples only active field-block diagonal entries from the
JAX-linearized residual and leaves feedback scalars unscaled, so it is a
cheaper diagnostic than dense same-cell block inversion. The
`field_block_sample`/`field_split` path samples one representative
field-by-equation block and applies that small inverse at every active cell,
making it a cheaper field-split/Schur probe than full same-cell block
inversion. Solver-level tests show it can cut operator calls from `10` to `5`
when a repeated local field block is the dominant stiff operator, but the first
hydrogen fixed-BDF2 recycling sweep kept the same `115` operator calls and
only added `20` sampled-block builds. The feedback-aware extension,
`field_block_feedback_diag`/`field_split_feedback`, adds diagonal JVP scaling
for packed feedback-integral variables and passes a reduced-budget
stiff-feedback fixture, but the same hydrogen fixed-BDF2 sweep again kept
`115` operator calls and added `20` feedback-aware block builds. The
`local_block_diag` path is the current physics/block preconditioner probe: it
uses the JAX-linearized residual to assemble same-cell field-coupling blocks
with JVPs, solves those small blocks on device, and treats off-cell transport
coupling through the outer Krylov iteration. The optional
`neutral_line`, `momentum_line`, and `sheath_line` paths use the same
JVP-derived line-block builder but restrict the approximate inverse to neutral
fields, `NV*` parallel-momentum fields, or target/sheath-coupled plasma fields,
respectively. The neutral and momentum selected-field line paths now
have bounded solver-level effectivity gates: on packed two-field stiff-line
fixtures they cut linear-operator calls from `10` to `5` while preserving
machine-precision convergence. This confirms the algorithmic seam, but does
not yet replace the missing heavy same-case recycling speedup; the sheath-line
route must be judged by the same heavy fixed-BDF2 update-residual gate before
promotion. A same-case
hydrogen fixed-BDF2 sweep with preconditioner refresh set to `100` kept the
same `115` linear iterations/operator calls for unpreconditioned,
`neutral_line`, `momentum_line`, `field_block_sample`, and
`field_block_feedback_diag` runs; all dynamic routes only added
preconditioner-build work and wall time. The matching `sheath_line` screen
preserved solver health and improved the achieved linear-update residual to
`6.94e-18` absolute (`4.75e-15` relative), but it also retained `115`
iterations/operator calls and took `64.227 s`/`59.113 s` on the fixed-full and
active-array routes. Future preconditioner work should
therefore target approximate target/sheath, Schur, field-line transport, or
neutral-plasma blocks that reduce the real recycling Krylov spectrum, not more
exact selected-line or sampled-local probes on this deck. The optional
`runtime:recycling_jax_linear_diagnose_update_residual=true` diagnostic now
adds an achieved `J v + r` residual check after each Krylov solve and records
absolute/relative update residuals in fixed/adaptive BDF summaries. Use it
when screening the next preconditioner so a candidate can be rejected if it
preserves final nonlinear residuals but does not improve update quality or
operator work under a constrained Krylov budget. The optional
`runtime:recycling_jax_linear_preconditioner_refresh` control reuses the
dynamic block preconditioner within one implicit solve. The matching
`runtime:recycling_jax_linear_preconditioner_floor`,
`runtime:recycling_jax_linear_preconditioner_max_linearized_unknowns`,
`runtime:recycling_jax_linear_preconditioner_max_field_unknowns`, and
`runtime:recycling_jax_linear_preconditioner_max_local_unknowns` controls make
bounded `linearized_diag`, `field_diag`, and `local_block_diag` campaigns
reproducible without editing source. These remain
diagnostics only. The June 15, 2026 hydrogen adaptive
BDF gate showed that `field_scale`, lower `maxiter=8`, and residual JIT did
not improve wall time or solver status, so the next runtime patch should not
repeat those knobs without a changed residual or preconditioner. The
`linearized_diag` diagnostic is also opt-in: it builds an exact JVP-derived
Jacobian diagonal after `jax.linearize`, but the first fixed-layout hydrogen
probe spent enough time building the diagonal that it did not improve the gate.
The June 18, 2026 `field_diag` probe verified the runtime surface and
preconditioner gate, but it was still slower than the matched unpreconditioned
bounded control (`7.37 s` versus `6.95 s`), so it remains diagnostic rather
than a promoted speedup path.
The follow-up BDF2 initial-guess cleanup is similarly bounded: JAX-linearized
adaptive-BDF modes now pass the embedded backward-Euler predictor into the
BDF2 corrector initial state. The hydrogen `recycling_1d_one_step`,
`timestep=1.0` gate completed cleanly in `106.3 s` with `42` fixed-layout
JAX-linearized trial solves and no failed substeps, but still spent `85.4 s`
in Krylov linear solves. This keeps the seam useful for convergence studies
while confirming that the next performance patch must reduce Krylov work or
residual/JVP kernel cost, not only alter initial guesses.
The inner-tolerance sweep now separates JAX Krylov tolerance from the outer
nonlinear/adaptive gate through
`runtime:recycling_jax_linear_tolerance_factor`. On the same hydrogen gate,
factor `10` completed cleanly in `103.9 s`, while factor `100` completed in
`105.3 s`. This is a small, bounded improvement and useful diagnostic, but the
run remains dominated by JAX-linearized Krylov solves.
The June 16, 2026 local-block probe closed the first physics/block
preconditioner pass without default promotion. The matched fixed-BDF2 hydrogen
gate took `13.15 s` without a preconditioner, `13.27 s` when rebuilding
same-cell JVP blocks every nonlinear update, and `13.02 s` with dynamic-block
reuse. The full adaptive-BDF hydrogen gate passed the same no-fallback and
accepted-error gates in `137.2 s` with rebuild-every-update local blocks and
`113.5 s` with local-block reuse, but this still trails the retained
unpreconditioned tolerance-factor gate. The sparse-JVP Jacobian builder now
also has an opt-in device-gather path that copies only structurally needed JVP
rows to the host sparse assembler; the small hydrogen gate did not benefit
(`1.596 s` with gather versus `1.551 s` without), so this remains a large-case
profiling knob rather than a default.
The next opt-in non-SciPy lane is
`runtime:recycling_transient_solver_mode=fixed_bdf2_jax_linearized` or the
`fixed_bdf2_jax_linearized_lineax` variant. It bypasses the SciPy `solve_ivp`
callback by taking a fixed-layout backward-Euler startup step followed by
fixed-layout BDF2 output steps, with controller integrals evolved inside the
packed residual state. The self-contained wrapper currently treats this as a
bounded-step diagnostic, not as a full-output production gate: the hydrogen
fixture passes at `timestep = 10`, while the full `timestep = 5000` output
window and the D/T/He bounded route still need nonlinear-convergence work before
default promotion.
The audit is deliberately small: it proves the solver backend boundary, not the
full recycling migration. The full migration still depends on moving the
remaining recycling residual kernels out of dictionary/NumPy assembly.

Near-term residual-kernel ports should target the measured hot path in this
order:

1. active packed-state layout as a static PyTree rather than repeated dict and
   full-field reconstruction
2. reaction/source and AMJUEL/OpenADAS table evaluation in JAX arrays
3. collision closure and neutral parallel diffusion in JAX arrays
4. target recycling and target-boundary source assembly in JAX arrays
5. backward-Euler/BDF residual assembly as a pure function with no `np.asarray`
   barrier inside the nonlinear solve

Each port should carry three gates before promotion: current NumPy parity on a
small deterministic state, JVP-versus-finite-difference derivative parity, and
the existing Hermès one-RHS/one-step compare surface.

Items 1, 3, 4, and 5 now have their first compact gates:

- `RecyclingFixedState` provides the active fixed-layout PyTree state and
  transformable backward-Euler/BDF2 residual builders;
- `build_fixed_array_rhs` now provides the pure active-array RHS adapter for
  staged source/closure/boundary ports, `build_fixed_full_field_array_rhs`
  stages guard-cell kernels such as target recycling through that same fixed
  PyTree interface, and `build_fixed_host_rhs_bridge` remains the parity oracle
  against the current full-field RHS;
- active-region and recycling active-state pack/unpack preserve JAX tracers;
- target recycling and the target-source kernel preserve JAX when dynamic state
  is a JAX array even if metrics are static NumPy arrays;
- neutral parallel diffusion and key collision/conduction closure helpers have
  compact JVP tests with precomputed rate surfaces;
- the open-field parallel advection and inertia operators used by the
  recycling/neutral-mixed RHS now have JAX branches with NumPy parity and
  JVP-versus-finite-difference gates;
- ion and electron RHS-term assembly now preserves JAX arrays through the
  transport, pressure-gradient, source-addition, and soft-floor pieces;
- the electron-force-balance parallel pressure-gradient stencil is now
  vectorized and backend-preserving, with a JVP finite-difference gate;
- the recycling RHS now calls the backend-preserving ion/electron assemblers
  without immediately coercing their inputs through `np.asarray`;
- neutral density, pressure, and momentum RHS assembly now uses the same
  backend-preserving term object as ions/electrons, including the Hermès branch
  where a neutral pressure source override is already a total source;
- recycling species-state preparation helpers now preserve JAX arrays through
  soft floors, safe temperatures, raw velocities, neutral target density
  guards, no-flow guards, and target-guard merges;
- density, pressure/energy, and momentum source accumulation now initializes,
  adds, and overrides per-species sources through one backend-preserving helper
  instead of forcing the accumulator dictionaries through NumPy at the start of
  each RHS call;
- electron parallel force balance and the resulting ion electric-force source
  updates now use a backend-preserving RHS helper with a direct
  JVP-versus-finite-difference gate, rather than adding those momentum sources
  through a NumPy-only block in the full RHS;
- electron sheath state preparation now applies no-flow scalar/flow guards,
  derives safe temperature and momentum, and reconstructs the full
  zero-current ion-sum/sheath-potential boundary formula through
  backend-preserving helpers with NumPy/JAX parity and JVP gates;
- collision friction/heat exchange, neutral parallel diffusion, and target
  recycling now have fixed-layout full-field adapter gates: each guard-cell or
  closure kernel can be called from a `RecyclingFixedState` RHS, defaults
  non-participating fields to zero, and differentiates through the active
  source response;
- the production backward-Euler/BDF2 recycling steppers now construct their
  residuals through the fixed-layout bridge and expose `sparse_jvp` and
  `jax_linearized` solver modes for transformable residual gates;
- the open-field state-preparation wrapper now preserves JAX arrays through
  electron-density reconstruction and the boundary-free electron/ion state
  path, giving the fixed-layout residual a transformable no-sheath control
  surface before the full sheath formulas are ported;
- the simple ion Bohm-sheath guard and energy-source formula is now isolated in
  a backend-preserving helper and covered by NumPy/JAX parity plus
  JVP-versus-finite-difference tests;
- the full electron sheath response after zero-current potential and the full
  ion sheath Bohm/energy response are now isolated in backend-preserving
  helpers, wired back into the existing full-sheath branches, and covered by
  NumPy/JAX parity plus JVP-versus-finite-difference tests;
- BE/BDF2 residual algebra no longer forces NumPy on JAX inputs.

Those gates are intentionally not promoted to the full heavy solve yet. The
first full-deck bridge is now tested on the local Hermès
`1D-recycling-dthe` input: `RecyclingFixedState` round-trips the active
DTHE recycling state, reconstructs full guard-cell fields, and drives the
current packed RHS oracle through an explicit `build_fixed_host_rhs_bridge`.
The bridge matches the current packed RHS and backward-Euler residual value.
It is deliberately named as a host bridge because the wrapped RHS still crosses
the NumPy/SciPy boundary. The next production step is narrower and safer:
replace the bridge internals one term at a time with fixed-layout JAX kernels,
then compare JVP actions against finite differences before replacing the
finite-difference Jacobian in the heavy BDF path.

The atomic-rate part of item 2 is now complete at helper level: AMJUEL paired
rate/radiation evaluation, OpenADAS bilinear interpolation, and the hydrogen
charge-exchange fit are backend-preserving and have direct `jit`/`grad`
coverage. The remaining item-2 work is to lift the surrounding reaction-source
accumulation out of mutable dictionaries and into a fixed array/PyTree layout
so those differentiable helpers can be used inside the full recycling residual.
The current dictionary-oriented reaction-source layer now also preserves JAX
arrays through the single-isotope ionisation, recombination, and
charge-exchange formulas, so the fixed-layout accumulator can reuse validated
formula code rather than reimplementing the physics.
The first such accumulator now exists for the hydrogenic same-isotope reaction
block and is parity-tested against the dictionary path. The next reaction
source step is to generalize that pattern to multispecies D/T/He and then
OpenADAS-enabled impurity states.
The D/T/He generalization is now in-tree as `fixed_layout_dthe_reaction_sources`:
it returns stacked neutral/ion/electron source arrays, includes D-D, T-T, D-T,
and T-D charge exchange, matches the existing dictionary path on the
`1D-recycling-dthe` deck, and supports `jit`/`grad`. The remaining reaction
source runtime step is partly complete: the packed recycling RHS now requests
reaction sources without diagnostics, and the exact D/T/He Hermès reaction
block is dispatched through the fixed-layout array kernel on that hot path. The
full dictionary implementation remains the reporting path when diagnostics are
requested. The next source-runtime step is the equivalent OpenADAS impurity
source block, followed by moving collision, diffusion, target-recycling, and
BDF residual assembly into a pure-JAX residual.

### Priority 3: tokamak recycling observables

1. keep the current one-step compare metrics
2. add observables closer to the benchmark literature:
   - target profiles
   - source/ionization lineouts
   - target flux summaries
3. move the paper-facing story from raw `NVd` relative mismatch to physically
   interpretable observables

The first version of this observable layer is now in-tree as
[tokamak_recycling_observable_campaign.md](tokamak_recycling_observable_campaign.md).
It uses the direct-tokamak D/T/He recycling one-step lane to report charged
target-density profiles, `|NV_s+|` target momentum-flux proxies, neutral
parallel-density buildup, and target electron-temperature proxy errors. The
next expansion should add source-diagnostic fields from live or cached Hermes
dumps so the same package can show explicit ionisation/recombination source
lineouts rather than neutral-density proxies only.

## Literature Standard

The expected validation and runtime storytelling standard is set by:

- [Roy 2005](https://www.sciencedirect.com/science/article/pii/S0021999104004747)
- [Bufetov et al. 2022, TCV-X21 benchmark](https://arxiv.org/abs/2109.01618)
- [Wang et al. 2023, SOLPS-ITER versus TCV-X21](https://arxiv.org/abs/2310.17390)
- [Dudson et al. 2024, Hermès-3 code paper](https://www.sciencedirect.com/science/article/pii/S0010465523003363)
- [Dudson et al. 2025, Hermès-3 versus TCV-X21](https://arxiv.org/abs/2506.12180)
- [official JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html)
- [official JAX persistent compilation cache documentation](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)

That means the next fixes should not stop at "bar chart looks better". They
need:

- operator-level diagnosis
- physically interpretable observables
- a reproducible profiler workflow
- explicit runtime/fidelity tradeoff discussion

## GPU And Memory Plan

The reachable `office` machine is the current GPU testbed. It already has two
visible CUDA devices under JAX, so the GPU plan is now operational rather than
speculative.

The clean repo-local GPU environment is now in place with
`jax[cuda12]==0.6.2`. The first useful GPU evidence is therefore already
available on the compact native JAX lanes:

- traced-field-line reduced lane:
  compile `4.41e-2 s`, first execute `1.23e-3 s`, warm execute `3.30e-4 s`
- stellarator VMEC reduced lane:
  compile `7.36e-3 s`, first execute `3.98e-4 s`, warm execute `1.14e-4 s`

That means the next GPU work should be:

1. enable persistent compilation cache
2. keep GPU runtime work on compact and fixed-layout JAX residual lanes first
3. compare CPU and GPU runtime splits on:
   - traced-field-line reduced kernels
   - stellarator VMEC reduced kernels
   - D/T/He fixed-layout recycling residual gates
   - other selected-field compact lanes before promoted transient ladders
4. keep the heavy recycling and neutral-mixed lanes on the CPU remediation
   path until the host/SciPy structure is reduced enough for accelerator work
   to be meaningful
5. decide whether the next parallelization gain should come from:
   - single-case acceleration
   - batched independent solves
   - multi-device sharding on accelerator hardware

The first D/T/He fixed-layout GPU gate has been run. It reaches the same
residual norm as the CPU gate and samples much lower process-tree RSS on the
small active-state problem, but the warm GPU wall time is still slower than
CPU. That result is a useful architectural gate, not a performance closeout:
future GPU evidence needs a larger transformed residual or a batched ensemble
before the paper can claim accelerator speedup.

The new batched D/T/He residual/JVP CPU gate is the first version of that
batched-ensemble evidence. It confirms that the fixed-layout residual is
transformable and that vectorized residual/JVP products amortize dispatch and
kernel work on CPU. The remote GPU attempt on the same heavy residual still has
unacceptable compile latency, so it remains an open solver-backbone target
rather than a release claim. By contrast, the atomic-rate source kernel already
shows a measured GPU throughput win because it is a dense, batched, fully
JAX-native calculation with no host/SciPy barrier: the retained largest batch
is about `2.5x` faster for the rate surface and about `2.0x` faster for its
autodiff derivative on the office GPU, with scalar sensitivity parity at about
`1e-10` relative error.

## Required New Evidence

Before the next paper-facing performance pass, the codebase should have:

- a refreshed same-machine live Hermès rerun figure
- one dedicated neutral-mixed boundary campaign and one term-level `NVh`
  balance campaign
- one refreshed heavy recycling runtime profile bundle
- one committed CPU/GPU profiling bundle for the D/T/He fixed-layout residual
- one committed CPU/GPU source-kernel throughput bundle for batched reaction
  rates and derivatives
- one tokamak-observable comparison package closer to the TCV-X21/Hermès style
