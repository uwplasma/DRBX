# Research Campaigns

This page defines the reproducible campaign layer used to keep the codebase
research-grade without turning every pull request into a multi-hour run.

The public entry point is:

```bash
python scripts/run_research_campaign_bundle.py --campaign scheduled-fast-research
```

The scheduled GitHub Actions workflow runs the same bounded public research
slice weekly. It does not require external reference checkouts and therefore
stays suitable for hosted CI. Longer live-reference and heavy profiling runs
are exposed through the same wrapper, but they are intended for local or
self-hosted machines where the reference checkout and heavier runtime budget
are available.

## Campaign Bundles

Use the CI-safe bundle for scheduled hosted checks:

```bash
python scripts/run_research_campaign_bundle.py --campaign all-ci
```

Use the local live-reference bundle when the external reference checkout is
available:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign live-reference \
  --reference-root /path/to/reference/root
```

Use the heavy recycling runtime bundle after solver changes:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign heavy-recycling-profile \
  --reference-root /path/to/reference/root
```

Use the fixed-layout D/T/He JAX-linearized residual gate when changing the
JAX-native recycling residual seam:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign dthe-jax-linearized-gate \
  --reference-root /path/to/reference/root
```

Use the promoted active-source D/T/He profile gate after changing reaction,
collision, neutral-diffusion, target-recycling, feedback, or source-composition
kernels in the promoted recycling residual. This gate records cProfile and RSS
evidence for the opt-in `promoted_active_sources` backend without changing the
stable default solver. It is intentionally nontrivial: the profile uses
`dt=1e-4`, obtains the initial residual from the first linearization, and
requires at least one nonlinear iteration, one JAX-GMRES solve, one
matrix-free operator call, and residual closure below `1e-6`.

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign dthe-promoted-active-sources-profile-gate \
  --reference-root /path/to/reference/root
```

Use the bounded D/T/He fixed-BDF2 active-array output-window gate when changing
the matrix-free recycling history route. This is the current local replacement
for the timeout-bound SciPy-BDF sparse-JVP output-window profile:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign dthe-fixed-bdf2-active-array-gate \
  --reference-root /path/to/reference/root
```

Use the longer D/T/He fixed-BDF2 active-array output-window gate before
promoting matrix-free recycling changes beyond the compact two-step gate:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign dthe-fixed-bdf2-active-array-long-window-gate \
  --reference-root /path/to/reference/root
```

Use the D/T/He fixed-BDF2 physical-output parity gate when a matrix-free
recycling change must prove that evolved fields still match the stable BDF
route on the active mesh:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign dthe-fixed-bdf2-active-array-physical-parity-gate \
  --reference-root /path/to/reference/root
```

Use the larger-timestep parity ramp after the physical-output gate passes and
before attempting production-window fixed-BDF2 claims:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign dthe-fixed-bdf2-active-array-parity-ramp-gate \
  --reference-root /path/to/reference/root
```

Use the scalar observable screen when testing the next timestep decade. This
gate intentionally compares only density and pressure fields (`Nd+`, `Pd+`,
`Nd`, `Pd`, `Pe`) because near-zero momentum inventories make relative
momentum observables non-diagnostic:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign dthe-fixed-bdf2-active-array-scalar-observable-gate \
  --reference-root /path/to/reference/root
```

Use the substepped full-field screen when the full momentum state must be
checked at the same `dt=1e-2` output window. This gate is intentionally heavy:
it caps the internal fixed-BDF2 timestep at `2.5e-3`, giving four implicit
substeps per output window.

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign dthe-fixed-bdf2-active-array-substepped-full-field-gate \
  --reference-root /path/to/reference/root
```

Use the adaptive-BDF JAX-versus-Lineax controller-health gate after changing the
adaptive residual route or linear-action solver:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign adaptive-bdf-jax-lineax-gate \
  --reference-root /path/to/reference/root
```

Use the bounded fixed-BDF2 direct-counting output-window gate after changing
the active-array JAX-linearized recycling solve. This is the practical local
campaign for proving that the direct-counting path executes real JAX GMRES
solves without relying on Python operator-call callbacks:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign fixed-bdf2-direct-counting-gate \
  --reference-root /path/to/reference/root
```

Use the strict fixed-BDF2 linear-update residual gate after changing
JAX-linearized Newton/GMRES internals, line-search settings, or output-window
preconditioner plumbing:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign fixed-bdf2-linear-update-residual-gate \
  --reference-root /path/to/reference/root
```

Use the lightweight active-array matrix-free Newton-update gate after changing
fixed-layout residual linearization, GMRES controls, or preconditioner plumbing:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign dthe-active-array-linearized-update-gate \
  --reference-root /path/to/reference/root
```

After the strict gate above has passed, use the paired throughput probe when
measuring production-style update cost without the extra post-GMRES linearized
residual diagnostic:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign dthe-active-array-linearized-update-throughput-probe \
  --reference-root /path/to/reference/root
```

Use the companion diagonal-preconditioner screen only when changing
preconditioner builders or deciding whether a packed diagonal is worth carrying
into heavier output-window runs:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign dthe-active-array-linearized-update-jvp-diag-gate \
  --reference-root /path/to/reference/root
```

Use the GPU bundle on a self-hosted machine with CUDA-visible devices when
collecting same-fidelity D/T/He residual evidence, larger fixed-layout
residual traces, full output-window profiles, memory snapshots, and pmap
evidence:

```bash
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 \
python scripts/run_research_campaign_bundle.py \
  --campaign all-gpu \
  --reference-root /path/to/reference/root
```

The `all-local` bundle runs the fast public slice, local CPU scaling, the
D/T/He JAX-linearized residual gate, the fixed-BDF2 direct-counting
output-window gate, the full heavy recycling cProfile/RSS profile, and the
live-reference matrix. It should only be used on machines where multi-hour runs
are acceptable.

## Current Evidence

The live-reference matrix remains the primary code-to-code fidelity dashboard.
The refreshed matrix no longer treats neutral mixed as the dominant
absolute-error outlier: `neutral_mixed_one_step` is now
`native_operational`, with the remaining mismatch useful mainly as a
source-term and boundary-history regression. The heavy 1D recycling ladders
remain the main runtime gap, while the integrated and direct tokamak recycling
one-step lanes are now better interpreted as normalization-sensitive because
their dominant compare field is near-zero `NVd`. Those results are documented
in [hermes_live_rerun_campaign.md](hermes_live_rerun_campaign.md) and
[runtime_gap_remediation.md](runtime_gap_remediation.md).

The local CPU scaling evidence is the heavy fixed-work ensemble in
[local_cpu_scaling_campaign.md](local_cpu_scaling_campaign.md). It uses
repeated direct tokamak recycling solves rather than a synthetic microkernel,
and the committed artifact reaches about `4.79x` steady-state speedup from
`1 -> 8` worker processes on the retained `16`-solve ensemble.

The GPU bundle contains the current `dt=1.0` D/T/He fixed-layout residual gate,
a separate larger active-array residual trace, fixed-BDF2 output-window,
active-array output-window, full-field compatibility output-window, and
batched-JVP lanes. The `gpu-dthe-current-jax-linearized-gate` command mirrors
the CPU `dthe-jax-linearized-gate` timestep, GMRES budget, residual ceiling,
line-search trial budget, and residual-evaluation budget, then adds JAX trace,
device-memory, and persistent-compilation-cache outputs. The older
`gpu-dthe-jax-linearized-gate` remains a larger active-array readiness probe and
requires the active-array RHS backend, so trace and memory evidence does not
accidentally fall back to the slower full-field compatibility residual. The
bounded non-SciPy fixed-BDF2 path still reports
both fixed-full-field and active-array diagnostics before any default solver
promotion. The `fixed-bdf2-direct-counting-gate` campaign is the bounded local
counterpart for that output-window path. On June 19, 2026 it ran
`recycling_1d_one_step` for `2` output steps at `dt=10`, executed `20`
active-array RHS steps, `20` jitted JAX-linearized fixed-BDF2 steps, and `23`
JAX GMRES solve attempts, and completed with zero failed or unconverged
substeps, maximum residual `2.90e-6`, `46` residual evaluations, and `43.76 s`
mode elapsed time. The campaign now gates that residual-evaluation budget with
`--require-fixed-bdf2-max-residual-evaluations=46`, so line-search or residual
rebuild changes cannot silently regress this bounded output-window solve.
Because it uses `runtime:recycling_jax_linear_operator_counting=direct`, the
correct low-overhead health metric is the reported solve count rather than
Python operator-call callbacks. The compact checked-in summary also reports
mean per-call timings: `0.242 s` per residual evaluation and `1.345 s` per
JAX-GMRES solve attempt:
[profile_summary.json](data/runtime_profile_artifacts/recycling_1d_fixed_bdf2_active_array_direct_counting_cpu/profile_summary.json).
The D/T/He companion `dthe-fixed-bdf2-active-array-gate` runs
`recycling_dthe_one_step` for two output steps at `dt=1e-4` through the
active-array fixed-BDF2/JAX-GMRES path. Its retained artifact reports one
startup backward-Euler step, one BDF2 corrector, two active-array RHS steps,
two jitted JAX-linearized actions, two JAX-GMRES solves, four residual
evaluations, zero failed or unconverged implicit steps, maximum residual
`4.10e-14`, and `11.27 s` mode elapsed time:
[profile_summary.json](data/runtime_profile_artifacts/recycling_dthe_fixed_bdf2_active_array_direct_counting_cpu/profile_summary.json).
This is the current local output-window D/T/He matrix-free promotion gate.
The longer `dthe-fixed-bdf2-active-array-long-window-gate` runs the same case
for eight output steps at `dt=1e-4`. Its retained wrapper-generated artifact
reports one startup step, seven BDF2 correctors, eight active-array RHS steps,
eight jitted JAX-linearized actions, eight JAX-GMRES solves, sixteen residual
evaluations, zero failed or unconverged implicit steps, maximum residual
`4.10e-14`, `32.40 s` total linear-solve time, `10.86 s` total residual
evaluation time, and `44.92 s` mode elapsed time:
[profile_summary.json](data/runtime_profile_artifacts/recycling_dthe_fixed_bdf2_active_array_long_window_cpu/profile_summary.json).
This is the current long-window D/T/He matrix-free output-window gate. It is
kept explicit rather than folded into `all-local` so local all-lane runs do not
silently acquire an additional minute-class solve.
The companion `dthe-fixed-bdf2-active-array-physical-parity-gate` runs the same
eight-step D/T/He window but also runs the stable `bdf` route and gates the
active-mesh field delta between `bdf` and
`fixed_bdf2_active_array_jax_linearized`. Its retained artifact reports a worst
field delta of `1.745e-7` on `NVd+`, below the `2.5e-7` gate threshold, with
the same fixed-BDF2 health metrics: eight JAX-GMRES solves, sixteen residual
evaluations, zero failed or unconverged implicit steps, and maximum residual
`4.10e-14`:
[profile_summary.json](data/runtime_profile_artifacts/recycling_dthe_fixed_bdf2_active_array_physical_parity_cpu/profile_summary.json).
This is the current bounded physical-output parity gate for the D/T/He
matrix-free route.
The `dthe-fixed-bdf2-active-array-parity-ramp-gate` increases the same
eight-step comparison from `dt=1e-4` to `dt=1e-3`. Its retained artifact
reports worst active-mesh field delta `1.761e-5` on `NVd+`, below the
`2.5e-5` threshold, with eight JAX-GMRES solves, sixteen residual evaluations,
zero failed or unconverged implicit steps, maximum residual `4.05e-11`, and
`44.93 s` fixed-BDF2 mode elapsed time:
[profile_summary.json](data/runtime_profile_artifacts/recycling_dthe_fixed_bdf2_active_array_parity_ramp_cpu/profile_summary.json).
A separate exploratory `dt=1e-2`, two-step probe also converged but was not
promoted because the worst `NVd+` field delta increased to `1.55e-3`. Treat
that as the next accuracy target, not as production-window promotion evidence.
The narrower `dthe-fixed-bdf2-active-array-scalar-observable-gate` retains the
same `dt=1e-2`, two-step window but gates scalar active-profile observables
instead of all-field max norms. It passes with worst scalar relative L2 error
`3.97e-5` on `Pd+`, worst scalar active-inventory relative error `4.72e-6`
on `Pd+`, fixed-BDF2 maximum residual `3.51e-8`, two JAX-GMRES solves, four
residual evaluations, zero failed or unconverged implicit steps, and
`10.74 s` fixed-BDF2 elapsed time:
[profile_summary.json](data/runtime_profile_artifacts/recycling_dthe_fixed_bdf2_active_array_scalar_observable_cpu/profile_summary.json).
This is scalar density/pressure observable evidence only. It does not close
momentum parity at `dt=1e-2`, where the full-field probe still shows the
largest `NVd+` pointwise delta and the `NVd` relative inventory is dominated by
an absolute near-zero denominator.
The `dthe-fixed-bdf2-active-array-substepped-full-field-gate` closes that
pointwise full-field gap for the same `dt=1e-2`, two-output-window comparison
by forcing `runtime:recycling_fixed_bdf2_max_internal_timestep=2.5e-3`. Its
retained artifact passes the `1.25e-4` full-field max-delta gate with worst
`NVd+` delta `1.099e-4`; the next pointwise deltas are `Pd+ = 3.03e-5` and
`Pe = 7.36e-7`. It also reports eight internal substeps, four internal
substeps per output window, eight JAX-GMRES solves, sixteen residual
evaluations, maximum residual `6.17e-10`, zero failed or unconverged implicit
steps, `32.10 s` in linear solves, `10.70 s` in residual evaluations, and
`44.43 s` fixed-BDF2 elapsed time:
[profile_summary.json](data/runtime_profile_artifacts/recycling_dthe_fixed_bdf2_active_array_substepped_full_field_cpu/profile_summary.json).
This promotes an explicit full-field correctness gate at `dt=1e-2`; it does
not make the route a default solver or a speedup claim.
A same-fidelity preconditioner follow-up keeps that decision unchanged. On
the same substepped D/T/He full-field gate, `momentum_line` preserved the
`NVd+ = 1.099e-4` worst field delta and `6.17e-10` residual but slowed the run
to `140.69 s`, with `33.49 s` spent building eight line-block preconditioners
and `96.09 s` in JAX-GMRES solves. The static `field_scale` preconditioner
also preserved the same parity metrics but slowed to `86.37 s`, with no
dynamic build cost and `74.56 s` in JAX-GMRES solves. These are negative
promotion results for the current D/T/He route; the next solver campaign
should reduce residual/JVP cost or change the block/transport approximation,
not rerun the same scaling or selected-line preconditioners. Residual JIT is
also negative on this same gate: `runtime:recycling_jax_linear_jit_residual=true`
preserved parity but slowed to `152.32 s`, with residual-evaluation time
increasing from `10.70 s` to `61.52 s`.
The matching `gpu-fixed-bdf2-direct-counting-gate` is intentionally guarded by
a process-level timeout in addition to the inner mode timeout. The first
office-GPU attempt on one RTX A4000 entered the solve but remained host-side
bound: after more than `12 min` it was still using about `137%` CPU, about
`3.8 GiB` RSS, roughly `250-300 MiB` on the GPU, and `0-1%` GPU utilization,
so it was terminated before writing a JSON summary. Treat that as negative
promotion evidence for this small fixed-BDF2 output-window case; the next GPU
claim should come from reduced host-side solver overhead or heavier batched
same-shape kernels, not from rerunning the same bounded gate unchanged.
The `gpu-dthe-active-array-output-jvp-profile` lane runs the full
`recycling_dthe_one_step` output window through
`runtime:recycling_transient_solver_mode=bdf_active_array_jvp`, requires
`bdf_rhs_backend=active_array`, `bdf_jvp_jacobian_gather_on_device=True`, and
at least one sparse-JVP Jacobian batch. It is the primary output-window GPU
profile for the active-array migration path. The
local `dthe-active-array-output-jvp-profile` lane exercises the same
active-array sparse-JVP output-window route on CPU and now has a command-level
timeout so it cannot silently consume a long local campaign. The June 19, 2026
bounded run timed out before producing an artifact, even after removing
redundant float64 casts from the fixed-layout adapter. Treat that as evidence
that this path is still dominated by repeated SciPy-BDF sparse-JVP Jacobian
materialization; exploratory longer runs should call
`scripts/profile_curated_case.py` directly with a chosen timeout.
The `gpu-dthe-full-output-jvp-profile` lane runs the same case through
`runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp`, so it is the
compatibility output-window profile to keep while comparing against the newer
active-array path; it carries the same sparse-JVP device-gather diagnostic gate.
The fixed-full-field `gpu-dthe-batched-jvp-gate` and active-array
`gpu-dthe-active-array-batched-jvp-gate` lanes measure ensemble throughput on
the same D/T/He residual family. The local active-array counterpart is
`dthe-active-array-batched-jvp-gate`; its retained `ny=100` CPU artifact
reaches about `2.55x` residual and `2.02x` JVP same-kernel speedup through
batch `64`, with JVP/finite-difference relative error about `5.97e-9` and
reusable linearized-action agreement with direct JVPs at about `3.47e-18`.
The companion `dthe-active-array-linearized-update-gate` runs a smaller
`ny=16` active-array D/T/He residual and solves one jitted matrix-free Newton
update. The retained CPU artifact reports GMRES solver status `0`, successful
solve metadata, linear-update relative residual `3.26e-16`, post-update
nonlinear residual `2.11e-11`, and update-check time `4.33 s`. The profile
gate reuses the existing diagnostic linearization, so the artifact records
`linearization_reused=true` and zero solve-only Python action callbacks under
the jitted linear operator. This makes the
preconditioner lane auditable before the update solve is considered for any
default BDF promotion. For throughput-only follow-up sweeps after this strict
gate has passed, `profile_recycling_batched_jvp_gate.py` also accepts
`--skip-linearized-update-residual-diagnostic`, which removes the extra
post-GMRES linearized residual action while keeping the nonlinear candidate
residual and solver status in the profile artifact.
The retained throughput-probe artifact reports solver status `0`,
`linear_update_residual_checked=false`, null linearized residual norms by
construction, the same `2.11e-11` candidate nonlinear residual, and update-check
time about `3.89 s`.

The `dthe-active-array-linearized-update-jvp-diag-gate` variant applies a
JVP-derived packed diagonal preconditioner. It is correct but not useful on
this fixture: the diagonal is nearly identity, with absolute entries from
`1.000000007` to `1.000043761`, the build cost is `0.61 s`, and update time
increases to `4.99 s` while preserving the same `2.11e-11` nonlinear
residual. It remains a negative performance screen rather than a promoted
preconditioner.
The active-array GPU batched campaign is deliberately single-device for now
(`--disable-pmap`) and uses residual/JVP batch partitions of `16` because
larger `ny=100` pmap and single-device office-GPU attempts were host/compiler
or memory bound and wrote no JSON summary. A tiny `ny=16` single-device CUDA
readiness probe did finish with JVP/finite-difference relative error
`3.95e-10`; this proves GPU executability of the reduced active-array
residual, not release-level GPU speedup. A second retained `ny=16` partitioned
probe writes the same error level with residual/JVP partition counts of `2`,
which proves the batch-partition code path on CUDA but remains neutral for
speedup.

The committed `ny=100`, `dt=1e-4` CPU comparison now records explicit Krylov
status metadata for both JAX GMRES and Lineax GMRES. Both runs reach
`residual_inf_norm=1.74e-12`; the JAX backend reports status `0` and a timed
run of about `8.15 s`, while the Lineax backend reports `RESULTS<>`, success,
`2` reported iterations, and a timed run of about `7.54 s`. These artifacts are
small JSON summaries, not trace bundles, so they are suitable for git and give
the heavier GPU/output-window campaigns a stable backend-health baseline.

The strict output-window fixed-BDF2 diagnostic gate now has a retained hydrogen
fixture artifact at
`docs/data/runtime_profile_artifacts/recycling_1d_fixed_bdf2_active_array_linear_update_residual_cpu/profile_summary.json`.
It uses the active-array JAX-linearized fixed-BDF2 route with the post-GMRES
linearized residual diagnostic enabled. The retained run reports zero failed
linear solves, `fixed_bdf2_max_residual_inf_norm=2.90e-6`,
`fixed_bdf2_max_linear_update_residual_inf_norm=1.58e-8`,
`fixed_bdf2_max_linear_update_relative_residual=1.02e-5`, and `23`
linear-update residual checks costing about `4.03 s`. This is the strict
health gate paired with the cheaper direct-counting gate; it is not a speedup
claim.

The adaptive-BDF promotion lane now also writes a lightweight JSON artifact at
`docs/data/runtime_profile_artifacts/recycling_1d_adaptive_bdf_jax_lineax_gate/profile_summary.json`.
For the `recycling_1d_one_step`, `timestep=1.0` diagnostics-only gate, both
JAX GMRES and Lineax GMRES follow the same controller trajectory: `21`
accepted substeps, `6` rejected trials, `61` fixed-full-field JAX-linearized
implicit solves, zero minimum-dt fallbacks, zero unconverged substeps, and
`adaptive_bdf_max_accepted_error_ratio=0.9315`. On this local run the Lineax
backend took about `151.9 s`, compared with about `174.2 s` for the in-tree JAX
GMRES backend. This remains promotion evidence for the opt-in adaptive-BDF
route, not a default-solver change.

The generic multi-ion one-step selector still defaults to the stable BDF route.
A local `recycling_dthe_one_step`, `timestep=1.0`,
`adaptive_bdf_jax_linearized` diagnostics-only probe exceeded the 720-second
mode bound, so multispecies adaptive-BDF promotion remains a runtime blocker
rather than a default-solver change. The next promotion attempt should first
reduce the D/T/He adaptive controller cost or use a heavier fixed-layout
matrix-free preconditioner, then rerun the gate with no fallbacks and no
unconverged substeps. For controller experiments, set
`runtime:recycling_adaptive_bdf_initial_dt=<dt>` to avoid spending expensive
JAX-linearized trial solves on rejected startup timesteps; this is an opt-in
runtime knob and does not alter the stable default BDF route. A follow-up local
probe with `runtime:recycling_adaptive_bdf_initial_dt=0.03125` still exceeded a
360-second bound on the same D/T/He diagnostics-only gate, so the current
blocker is per-step solve cost and/or accepted-step count rather than only
startup rejection. For timeout-bound probes, set
`JAX_DRB_RECYCLING_ADAPTIVE_BDF_TRACE_JSONL=/tmp/dthe_adaptive_bdf_trace.jsonl`
to get flushed JSONL records before and after each startup, backward-Euler
predictor, BDF2 corrector, and embedded-error estimate; the trace carries
residual-evaluation, Jacobian/linearization, Krylov-solve, line-search, route,
and convergence metadata from each implicit substep. The first committed trace
summary for this lane is
`docs/data/runtime_profile_artifacts/recycling_dthe_adaptive_bdf_trace_probe/profile_summary.json`:
with a 180-second bound, eight completed startup implicit trials consumed
`179.9 s`, of which `138.0 s` were Krylov solves and `37.3 s` were
Jacobian/linearization work. The run did not reach BDF2 before timeout because
the startup embedded-error ratios stayed very large (`2.0e6`, then `3.2e5`).
The first
bounded sweeps should vary
`runtime:recycling_jax_linear_restart=<n>` and
`runtime:recycling_jax_linear_maxiter=<m>` (or
`JAX_DRB_RECYCLING_JAX_LINEAR_RESTART` and
`JAX_DRB_RECYCLING_JAX_LINEAR_MAXITER`) while keeping the JSONL trace enabled,
then compare elapsed time, `converged`, residual norm, and embedded-error
ratio against the fixed default `20 x 20` Krylov budget. A first cold local
`10 x 10` probe did not improve the blocker: six completed startup trials still
used `150.6 s` total, with `127.5 s` in linear solves and the same startup
rejection pattern. That negative result argues for preconditioning or a
different startup estimator rather than simply lowering the JAX GMRES budget.
A matching cold Lineax probe also failed to improve the multi-ion lane: six
completed startup trials used `153.3 s`, only one completed substep reported
linear-solver success, and the remaining Lineax statuses reported iterative
breakdown or non-finite output. Future promotion gates must therefore check
`adaptive_bdf_linear_solver_failed_steps == 0`, not only nonlinear convergence
and embedded-error ratios. A later hydrogen adaptive-BDF control confirmed the
same conclusion on the cheaper full-controller gate: opt-in `field_scale`
preconditioning completed but slowed to `111.8 s` with `9` unknown inner linear
statuses, a lower `JAX_DRB_RECYCLING_JAX_LINEAR_MAXITER=8` budget slowed to
`116.6 s`, and `runtime:recycling_jax_linear_jit_residual=true` timed out at a
`220 s` guard. The exact JVP-derived `linearized_diag` diagnostic also failed
to improve the cheap fixed-layout probe: it ran in `3.66 s`, with `0.36 s` in
diagonal construction and the same full GMRES update budget. The subsequent
backward-Euler-predictor initial-guess cleanup completed the cheaper hydrogen
`adaptive_bdf_jax_linearized`, `timestep=1.0` gate cleanly in `106.3 s`, but
the run still spent `85.4 s` in Krylov linear solves and used `16,800` inner
linear iterations. A follow-up inner-tolerance sweep added
`runtime:recycling_jax_linear_tolerance_factor`; factor `10` completed the same
gate cleanly in `103.9 s`, while factor `100` slowed to `105.3 s`. This is a
small controlled improvement and useful diagnostic, not enough to justify a
heavy D/T/He allocation by itself. Do not spend the next D/T/He campaign
allocation on these specific knobs unless the residual, preconditioner, or
Krylov algorithm itself changes.
The June 16, 2026 physics/block-preconditioner probe changed the residual
seam, but still did not justify default promotion. The new
`local_block_diag`/`block_jacobi` preconditioner builds same-cell field
Jacobian blocks from JVPs after `jax.linearize`; this is the closest local JAX
analogue of a block-Jacobi field preconditioner while preserving the true
matrix-free residual in GMRES. It is unit-tested and passes the hydrogen
fixed-BDF2 and adaptive-BDF gates. The measured fixed-BDF2 gate was effectively
neutral (`13.15 s` unpreconditioned, `13.27 s` with rebuild-every-update local
blocks, `13.02 s` with block reuse), and the adaptive hydrogen gate remained
slower than the retained unpreconditioned tolerance-factor run (`137.2 s`
rebuild, `113.5 s` reuse, versus `103.9 s`). The sparse-JVP device-gather
probe is also implemented and tested, but the small gate was negative
promotion evidence (`1.596 s` gather, `1.551 s` default full transfer). Future
campaigns should therefore treat both knobs as profiling controls for larger
systems or GPUs, not as default solver improvements.
The strongest measured path is now
`adaptive_bdf_sparse_jvp`: in the same 180-second trace window it completed
`96` implicit trials, reached BDF2, and reduced average completed-trial cost to
about `1.8 s`, with sparse linear solves near milliseconds. That route still
fails promotion because the embedded-error ratios remained hundreds by the end
of the bounded trace, and `166.3 s` of the `170.3 s` completed-trial time was
spent assembling grouped-JVP Jacobians. The trace can now also write
field-level error contributors on `error_estimate` records. A 75-second
contributor probe showed that the remaining adaptive-BDF error is dominated by
ion parallel-momentum fields (`NVd+`, `NVt+`, and `NVhe+`), while feedback
integral errors are below `1e-6` in the same norm. The contributor trace now
also includes raw field differences and tolerance scales. Those diagnostics
showed that the momentum ratios were caused by small absolute momentum changes
being divided by `~1e-12` to `1e-9` scales. An opt-in probe with
`JAX_DRB_RECYCLING_ADAPTIVE_BDF_MOMENTUM_ATOL_FLOOR=1e-2` reduced the startup
error ratios by about two orders of magnitude, but did not close the gate; the
dominant offender shifted to low-density/pressure fields such as `Nd`, `Pd`,
and `Phe+`. The next implementation target is therefore sparse-JVP
linearize/push reduction plus a component-wise adaptive-norm audit, not a
lower JAX GMRES budget or looser feedback tolerances.
A follow-up opt-in component-wise norm gate combined
`runtime:recycling_adaptive_bdf_initial_dt=0.0625` with
`JAX_DRB_RECYCLING_ADAPTIVE_BDF_DENSITY_ATOL_FLOOR=1e-6`,
`JAX_DRB_RECYCLING_ADAPTIVE_BDF_PRESSURE_ATOL_FLOOR=1e-3`, and
`JAX_DRB_RECYCLING_ADAPTIVE_BDF_MOMENTUM_ATOL_FLOOR=1e-2`. On the local
D/T/He diagnostics-only `timestep=1.0` gate, this route completed in `28.6 s`,
accepted eight adaptive steps, accepted seven BDF2 steps, reported no
minimum-dt fallback, no unconverged substeps, and kept the maximum accepted
embedded-error ratio at `0.715`. This is the first passing D/T/He sparse-JVP
adaptive-BDF promotion-style gate, but it remains opt-in until longer
output-window reference-parity and GPU/CPU scaling campaigns pass with the
same component-wise norm.
After sparse-JVP workspace threading, the same bounded gate still accepts the
same eight steps and reports `adaptive_bdf_sparse_jvp_workspace_reuses=17`,
one reuse for each trial solve. With synchronized sparse-JVP timing enabled,
the local wall time is `29.1 s`; of `27.9 s` spent in JVP Jacobian construction,
`17.8 s` is JAX linearization, `10.1 s` is grouped-push device execution,
`1.6e-4 s` is host transfer, and `5.0e-3 s` is sparse assembly. Static sparse
plan/direction allocation has therefore been removed from the adaptive loop,
but the runtime blocker remains residual linearization and grouped-JVP push
execution.

The D/T/He fixed-layout JAX-linearized residual gate writes CPU profile
summaries under `docs/data/runtime_profile_artifacts/`. The current
`dt=1.0`, `950`-active-variable CPU gate passes with residual `7.315`, one
line-search trial, two residual evaluations, five jitted matrix-free operator
calls, `8.92 s` profiled runtime, and about `2.86 GiB` sampled peak
process-tree RSS. The same-fidelity
`gpu-dthe-current-jax-linearized-gate` now passes on one RTX A4000 with the
same residual, line-search, residual-evaluation, and operator-call counts, but
it takes `109.49 s` and samples peak process-tree RSS near `12.34 GiB`. The
retained older office-GPU summaries remain useful tiny-step readiness probes,
but the current-gate result is negative speedup evidence for the full-field
fixed-layout residual. The next GPU claim should use active-array residuals,
smaller compiled kernels, or heavier same-shape batching that amortizes compile
and launch overhead.

## Promotion Policy

A validation or performance campaign is promoted into the README or paper plan
only when it satisfies four conditions:

- it is tied to a named physics, numerical, or differentiability claim;
- it has a deterministic script entry point;
- it writes JSON evidence plus a publication-ready figure or profile bundle;
- it states the limitation of the result instead of extrapolating beyond the
  measured case.

For the remaining recycling solver work, the order is therefore fixed:

- keep the stable production BDF path as the default while it is the only
  fully validated output-window path;
- use `JAX_DRB_RECYCLING_BDF_JACOBIAN_MODE=jvp` only as an opt-in derivative
  experiment on transformable residual callbacks;
- use `runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp` as
  the named output-window BDF gate for the fixed-full-field RHS plus grouped
  JVP Jacobian seam;
- run the self-contained fixed-JVP parity gate before any heavy promotion
  attempt: `PYTHONPATH=src python scripts/run_recycling_jvp_promotion_gate.py`;
  this uses the committed lightweight fixture decks by default and accepts
  `--reference-root /path/to/reference/root` when a live reference checkout is
  available; use `--output-dir docs/data/runtime_profile_artifacts/<new-run>`
  when the run should leave per-case JSON reports and an aggregate
  `summary.json` for release review; the wrapper now writes separate
  `bdf_jvp` and `fixed_bdf2` phase reports, so full-output BDF/JVP parity
  remains distinct from bounded-step fixed-BDF2 residual diagnostics;
- keep fixed-BDF2 promotion honest: the default bounded fixed-BDF2 phase runs
  on the hydrogen fixture at `timestep = 10` and the D/T/He fixture at
  `timestep = 1` with
  `runtime:recycling_fixed_bdf2_max_internal_timestep=0.5`. Both
  JAX-linearized and active-array variants pass the residual/status gate there
  with `fixed_bdf2_max_residual_inf_norm = 3.77e-9`, four internal substeps,
  zero unconverged steps, and zero failed inner linear solves. The full-output
  lane still needs a production substep policy and a lower-cost Krylov or
  preconditioned solve before it can be treated as speedup evidence. A local
  active-array control run with `runtime:recycling_jax_linear_restart=10` and
  `runtime:recycling_jax_linear_maxiter=20` remained correct but slowed to
  about `136.8 s`, so future sweeps should prioritize preconditioning,
  nonlinear damping/startup policy, or residual/JVP kernel cost rather than
  simple GMRES restart reduction;
- run the adaptive-BDF JAX-linearized promotion gate only as an explicit
  rejection test until it clears without fallback:
  `PYTHONPATH=src python scripts/compare_recycling_transient_modes.py --case
  recycling_1d_one_step --reference-root /path/to/reference/root --mode
  adaptive_bdf_jax_linearized --field Pe --diagnostics-only --timestep 1.0
  --max-nonlinear-iterations 3
  --require-adaptive-bdf-no-fallback
  --require-adaptive-bdf-no-unconverged-substeps
  --require-adaptive-bdf-max-accepted-error-ratio 0.95
  --mode-timeout-seconds 480`;
  the current local result for this gate is a pass at `timestep=1.0` with zero
  fallback, zero unconverged substeps, and
  `adaptive_bdf_max_accepted_error_ratio=9.315e-1`; the variable-step BDF2
  controller used `21` accepted steps, `3` rejected trials, `50` implicit trial
  solves, `22` BDF2 trial solves, `20` accepted BDF2 correctors, and reused
  valid BDF history after `2` rejected trials. Increase the timeout on laptops
  because the measured wall time was about `108 s`, with the linear solve still
  accounting for about `86 s`. The same gate now also requires route
  provenance:
  `adaptive_bdf_fixed_full_field_rhs_solver_steps` must be positive for every
  opt-in JAX/JVP adaptive mode, `adaptive_bdf_sparse_jvp_jacobian_solver_steps`
  must be positive for `adaptive_bdf_sparse_jvp`, and
  `adaptive_bdf_jax_linearized_action_solver_steps` must be positive for
  `adaptive_bdf_jax_linearized` and `adaptive_bdf_jax_linearized_lineax`;
- the active-array adaptive-BDF route now has the same controlled gate surface:
  `adaptive_bdf_active_array_jax_linearized` completed the local
  `recycling_1d_one_step`, `timestep=1.0` diagnostics run in about `103 s`,
  used `49` active-array RHS/JAX-linearized trial solves, and reported zero
  fallback accepts, zero unconverged implicit substeps, and zero failed linear
  solves. Its maximum accepted-step error ratio was `9.315e-1`, while rejected
  trials raised the all-trial maximum error ratio to about `2.99`. The improved
  run is the result of damping the default first internal step for
  JAX-linearized adaptive modes to at most one sixteenth of the output window
  unless an explicit `recycling_adaptive_bdf_initial_dt` override is supplied.
  Treat this as route and controller-health evidence, not a full promotion
  pass, until the full output window and D/T/He heavy cases clear the same
  runtime/parity gates;
- keep the Lineax seam as an opt-in backend comparison. The current
  `timestep=1.0` adaptive-BDF gate has identical controller diagnostics for
  `adaptive_bdf_jax_linearized` and `adaptive_bdf_jax_linearized_lineax`
  (`21` accepted substeps, `3` rejected trials, `50` implicit trial solves,
  zero fallback, zero unconverged substeps,
  `adaptive_bdf_max_accepted_error_ratio=9.315e-1`). On the local CPU, Lineax
  GMRES ran in about `91 s` versus about `108 s` for the in-tree JAX GMRES
  path, but it reported `41` failed inner linear solves. Treat that as negative
  promotion evidence until the backend is made convergence-clean;
- keep the native JAX BiCGSTAB seam as a diagnostic backend only. With
  `JAX_DRB_RECYCLING_JAX_LINEAR_SOLVER=bicgstab`, the same hydrogen
  `timestep=1.0` gate ran in about `108 s`, effectively matching JAX GMRES,
  and local JAX does not report an inner success flag for BiCGSTAB. Diagnostics
  therefore expose `adaptive_bdf_bicgstab_action_solver_steps` and
  `adaptive_bdf_unknown_linear_solver_steps`, but this backend is not a current
  promotion speedup;
- do not spend more full D/T/He adaptive-JAX wall time before reducing
  per-trial cost. A post-history-reuse `1D-recycling-dthe`, `timestep=1.0`
  adaptive JAX-linearized gate still exceeded a `360 s` guard before writing a
  completed mode report;
- report `bdf_jvp_jacobian_linearize_seconds`,
  `bdf_jvp_jacobian_push_seconds`, and
  `bdf_jvp_jacobian_total_seconds` for any fixed-JVP run, and also check
  `bdf_jvp_direction_batch_count` plus
  `bdf_jvp_jacobian_tangent_build_seconds`; the current fixed-JVP bridge
  prebuilds sparse tangent batches once per solve, so remaining regressions
  should be attributed to linearization and tangent pushes, not repeated
  tangent allocation;
- require JAX-linearized profiles to include `linear_solver_backend`,
  `linear_solver_status`, and `linear_solver_success` in their diagnostics, so
  backend comparisons separate residual linearization time from failed or
  marginal Krylov solves;
- run `gpu-dthe-full-output-jvp-profile` on a self-hosted GPU runner after any
  output-window recycling solver change, because this is the first campaign in
  the bundle that combines the production curated case, the fixed-full-field
  RHS, grouped JVP Jacobian assembly, JAX trace, device-memory profile, and RSS
  sampling;
- continue moving source, closure, boundary, and target-recycling kernels into
  fixed-layout JAX functions with parity and JVP gates;
- promote matrix-free/JVP nonlinear solves only after the full heavy residual
  is transformable and has passed live-reference and runtime campaigns.
