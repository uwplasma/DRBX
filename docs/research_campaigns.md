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

Use the adaptive-BDF JAX-versus-Lineax controller-health gate after changing the
adaptive residual route or linear-action solver:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign adaptive-bdf-jax-lineax-gate \
  --reference-root /path/to/reference/root
```

Use the GPU bundle on a self-hosted machine with CUDA-visible devices when
collecting larger fixed-layout residual, full output-window, trace, memory, and
pmap evidence:

```bash
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 \
python scripts/run_research_campaign_bundle.py \
  --campaign all-gpu \
  --reference-root /path/to/reference/root
```

The `all-local` bundle runs the fast public slice, local CPU scaling, the
D/T/He JAX-linearized residual gate, the full heavy recycling cProfile/RSS
profile, and the live-reference matrix. It should only be used on machines
where multi-hour runs are acceptable.

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

The GPU bundle contains three distinct lanes. The fixed-layout JAX-linearized
gate measures the residual/JVP seam without the full output-window driver. The
`gpu-dthe-full-output-jvp-profile` lane runs `recycling_dthe_one_step` through
`runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp`, so it is the
production-output profile to use before claiming heavy recycling GPU speedup or
promoting that solver path. The batched-JVP lane measures ensemble and
multi-device throughput on the same D/T/He residual family.

The committed `ny=100`, `dt=1e-4` CPU comparison now records explicit Krylov
status metadata for both JAX GMRES and Lineax GMRES. Both runs reach
`residual_inf_norm=1.74e-12`; the JAX backend reports status `0` and a timed
run of about `8.15 s`, while the Lineax backend reports `RESULTS<>`, success,
`2` reported iterations, and a timed run of about `7.54 s`. These artifacts are
small JSON summaries, not trace bundles, so they are suitable for git and give
the heavier GPU/output-window campaigns a stable backend-health baseline.

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
and embedded-error ratios. The strongest measured path is now
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

The D/T/He fixed-layout JAX-linearized residual gate now has both CPU and GPU
profile summaries under `docs/data/runtime_profile_artifacts/`. On the current
small `950`-active-variable gate, the CPU run completes in about `4.74 s` with
about `5.0 GiB` sampled peak process-tree RSS. The office GPU run reaches the
same residual norm and cuts sampled peak RSS to about `1.4 GiB`, but the warm
wall time remains about `6.66 s`. That is useful evidence that the seam is
accelerator-executable and lower-memory; it is not yet a speedup claim because
this problem size is too small and still dominated by compile/launch overhead.

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
  available;
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
  controller used `21` accepted steps, `6` rejected trials, `61` implicit trial
  solves, `20` BDF2 trial solves, and `19` accepted BDF2 trial solves, so
  increase the timeout on laptops because the measured wall time was about
  `174 s`. The same gate now also requires route provenance:
  `adaptive_bdf_fixed_full_field_rhs_solver_steps` must be positive for every
  opt-in JAX/JVP adaptive mode, `adaptive_bdf_sparse_jvp_jacobian_solver_steps`
  must be positive for `adaptive_bdf_sparse_jvp`, and
  `adaptive_bdf_jax_linearized_action_solver_steps` must be positive for
  `adaptive_bdf_jax_linearized` and `adaptive_bdf_jax_linearized_lineax`;
- keep the Lineax seam as an opt-in backend comparison. The current
  `timestep=1.0` adaptive-BDF gate has identical controller diagnostics for
  `adaptive_bdf_jax_linearized` and `adaptive_bdf_jax_linearized_lineax`
  (`21` accepted substeps, `6` rejected trials, `61` implicit trial solves,
  zero fallback, zero unconverged substeps,
  `adaptive_bdf_max_accepted_error_ratio=9.315e-1`). On the local CPU, Lineax
  GMRES ran in about `152 s` versus about `174 s` for the in-tree JAX GMRES
  path, which is useful evidence but not a default-solver promotion;
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
