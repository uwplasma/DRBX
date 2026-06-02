# Release Notes: 1.0.2

`jax_drb 1.0.2` is a validation and solver-hardening release focused on
neutral-mixed parity closure, auditable recycling JVP paths, and honest
CPU/GPU evidence for the current JAX-linearized recycling lane.

## Highlights

- Neutral mixed one-step and short-window parity runs now use internal BDF
  substeps by default. The refreshed campaign reduces the `NVh` history error
  from about `3.37e-3` to about `5.81e-4`.
- Direct neutral `NVh` source diagnostics close the pressure-gradient,
  parallel-viscosity, and perpendicular-viscosity implementation question:
  the written reference diagnostics and JAXDRB reconstructions agree to
  roundoff after active-domain scaling.
- The production SciPy BDF recycling compatibility path now records the
  resolved `bdf_jacobian_mode` and `bdf_jvp_batch_size` alongside RHS,
  cache-hit, and Jacobian callback counters.
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
  `2.1x` faster for the autodiff derivative than the local CPU reference. The
  same gate checks a scalar sensitivity objective against finite differences
  at about `1e-10` relative error.
- Public validation/profile summaries now sanitize local reference and repo
  paths before writing committed JSON artifacts.

## Validation

The release candidate passed `pytest -q`, `scripts/run_closeout_coverage.py`,
`scripts/run_promoted_solver_coverage.py`, and the local all-campaign research
bundle with the live reference checkout. GitHub Actions are now green on the
documentation, test, and coverage workflows, so release publication is gated by
the remaining technical promotion checks rather than CI availability.

## Current Boundary

The full output-window recycling BDF default remains the stable
finite-difference compatibility path. JVP and JAX-linearized GMRES modes are
audited opt-in lanes for transformable residual surfaces; they should not yet
be described as a blanket end-to-end differentiable heavy recycling backend.
