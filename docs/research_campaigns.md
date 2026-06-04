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
It identifies the neutral mixed `NVh` operator mismatch as the main fidelity
offender and the heavy D/T/He recycling one-step path as the main runtime
offender. Those results are documented in
[hermes_live_rerun_campaign.md](hermes_live_rerun_campaign.md) and
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
  `162 s`. The same gate now also requires route provenance:
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
  GMRES ran in about `132 s` versus about `162 s` for the in-tree JAX GMRES
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
