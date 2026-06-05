# Release Notes: 1.0.2

`jax_drb 1.0.2` is a validation and solver-hardening release focused on
neutral-mixed parity closure, auditable recycling JVP paths, and honest
CPU/GPU evidence for the current JAX-linearized recycling lane.

## Highlights

- Neutral mixed one-step and short-window parity runs now use internal BDF
  substeps by default. The refreshed connected-y boundary reconstruction and
  one-step substep audit reduce the promoted `NVh` one-step active-domain
  metric from the older `3.37e-3` boundary-local mismatch through the
  intermediate `5.81e-4` result to about `4.47e-6` in the tracked JSON report.
  The short-window path remains at four internal substeps because its remaining
  error is a total-history sequencing issue rather than the one-step
  target-band momentum closure.
- Direct neutral `NVh` source diagnostics close the pressure-gradient,
  parallel-viscosity, and perpendicular-viscosity implementation question:
  the written reference diagnostics and JAXDRB reconstructions agree to
  roundoff after active-domain scaling.
- A new Hermès-free neutral substep/hybrid diagnostic sweeps
  `runtime:neutral_mixed_internal_substeps`, records failed high-substep
  attempts explicitly, and swaps `Nh`, `Ph`, and `NVh` reference final fields
  into the native final state to rank target-band state/history sequencing
  drivers before changing production boundary logic.
- The production SciPy BDF recycling compatibility path now records the
  resolved `bdf_jacobian_mode` and `bdf_jvp_batch_size` alongside RHS,
  cache-hit, RHS phase-timing, and Jacobian callback counters.
- The transient-mode comparison helper now includes the opt-in
  `bdf_fixed_full_field_jvp` lane, prints a direct `bdf` versus
  full-field-JVP delta table when both modes are run, and can now fail as a
  real promotion gate with `--require-fixed-jvp-diagnostics` plus
  `--require-bdf-pairwise-max`.
- A new self-contained wrapper,
  `scripts/run_recycling_jvp_promotion_gate.py`, runs that gate on the
  committed hydrogen and D/T/He lightweight fixture decks. The current local
  results pass with worst BDF-vs-fixed-JVP active-mesh deltas of `7.59e-6`
  and `2.20e-7`, respectively. The BDF bridge now prebuilds sparse-JVP
  tangent batches once per solve, and the refreshed gate reports
  `bdf_jvp_jacobian_tangent_build_seconds=0` with
  `bdf_jvp_direction_batch_count=1` for both fixture decks. The fixed-JVP
  route is still opt-in because the same runs are slower than default BDF
  (`72.9 s` versus `10.1 s` for hydrogen, `189.3 s` versus `54.2 s` for
  D/T/He) while repeated `jax.linearize` and tangent pushes remain inside the
  SciPy BDF callback.
- The D/T/He JAX-linearized GMRES profiling script now supports repeated
  BOUT.inp overrides and warmup runs, so heavier real-kernel CPU/GPU gates can
  be reproduced without committing large input decks.
- Larger matched D/T/He GMRES profile artifacts compare CPU and office-GPU
  runs at `ny=100` and `ny=200`. They close to the same residuals, but the
  current GPU path is slower despite lower sampled RSS, so GPU speedup is not
  promoted as a release claim.
- A new batched D/T/He residual/JVP gate exercises the real fixed-layout
  recycling backward-Euler residual under `jit`, `vmap`, `jvp`, and `grad`.
  The retained local CPU batch sweep reaches about `2.8x` residual throughput
  speedup and about `2.2x` JVP throughput speedup, with
  JVP/finite-difference error about `6e-9`.
- A new batched atomic-rate throughput gate gives the release a measured GPU
  speedup on a fully JAX-native source kernel: at `4,194,304` temperature
  points, the office GPU is about `2.5x` faster for the rate surface and about
  `2.0x` faster for the autodiff derivative than the local CPU reference. The
  same gate checks a scalar sensitivity objective against finite differences
  at about `1e-10` relative error.
- Public validation/profile summaries now sanitize local reference and repo
  paths before writing committed JSON artifacts.

## Validation

The current release candidate passed the bounded closeout gate at `97%`
coverage and the promoted native-solver gate at `95%` coverage on the local
developer machine. On a developer machine with the external reference checkout
available, the promoted gate runs the live operational-band comparisons. In a
no-reference CI environment the same gate uses committed lightweight BOUT input
fixtures for the recycling solver-unit coverage, skips only the
external-reference-only operational-band checks, and still passes the promoted
surface at `95%` coverage (`450` passed, `14` skipped, `7` deselected, and `1`
expected xfail in the local CI-like simulation). Release publication is
therefore gated by technical promotion decisions rather than CI availability.

Live-reference and large `all-gpu` campaigns remain manual self-hosted runs:
they require a valid reference checkout and CUDA-visible devices. Their
commands are now exposed in the research-campaign workflow dispatch and tested
against the bundle script, but the retained release evidence should still be
read as committed-profile evidence rather than a blanket full-output-window
GPU speedup claim.

## Current Boundary

The full output-window recycling BDF default remains the stable
finite-difference compatibility path. JVP and JAX-linearized GMRES modes are
audited opt-in lanes for transformable residual surfaces; they should not yet
be described as a blanket end-to-end differentiable heavy recycling backend.
