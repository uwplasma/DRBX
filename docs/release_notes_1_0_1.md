# Release Notes: 1.0.1

`jax_drb 1.0.1` is a research-hardening release focused on the recycling
solver backbone, Hermès diagnostic parity, and refreshed performance evidence.

## Highlights

- Production backward-Euler, BDF2, and legacy BDF recycling paths now route
  residual/RHS assembly through the fixed-layout recycling state bridge.
- Recycling implicit steps expose opt-in `sparse_jvp` and `jax_linearized`
  solver modes for residuals that have been proven JAX-transformable.
- The sparse Jacobian path can select grouped JVP assembly with
  `JAX_DRB_RECYCLING_JACOBIAN_MODE=jvp` and can bound JVP color batching with
  `JAX_DRB_RECYCLING_JVP_BATCH_SIZE`.
- The neutral mixed `NVh` offender campaign now ingests a direct patched
  Hermès `SNVh_pressure_gradient` diagnostic and keeps the matched
  reconstruction used for normalized JAXDRB-side operator comparison.
- Fresh runtime artifacts document the remaining heavy recycling bottleneck:
  sparse finite-difference Jacobian construction and repeated host-side RHS
  assembly still dominate the legacy SciPy BDF compatibility path.
- The local CPU scaling artifact was refreshed on the heavy fixed-work
  tokamak-recycling ensemble, and the reduced native JAX profile audit was
  rerun on the two-GPU `office` host.
- The PyPI workflow now publishes on GitHub release publication or manual
  dispatch, avoiding duplicate publishes from a tag push plus release event.

## Validation

The release was checked with the targeted recycling, solver, neutral-mixed,
packaging, release-surface, and closeout-coverage slices documented in the
repository. The repo-wide monolithic coverage run remains too slow to be the
local release gate; the maintained ship gate is the bounded closeout coverage
script described in [closeout_coverage.md](closeout_coverage.md).

## Current Boundary

The fixed-layout residual bridge is now part of the production recycling
steppers, but the heaviest SciPy BDF history path still depends on host-side
RHS assembly and finite-difference Jacobian callbacks. JVP and matrix-free
linearized solves should therefore be treated as promoted opt-in modes for
transformable residual surfaces, not as a blanket claim that every recycling
workflow is already end-to-end differentiable.
