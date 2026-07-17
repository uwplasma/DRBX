# Changelog

All notable changes to `jax_drb` are recorded here. The current development
series is summarized at the top; the historical `1.x` release notes are
condensed into this file (their original pages remain in git history).

## 2.0.0.dev0 (unreleased)

Development series for the v2 research-grade program defined in
`plan_jax_drb.md`. See
[docs/release_notes_2_0_0_dev0.md](docs/release_notes_2_0_0_dev0.md) for the
running, phase-by-phase detail of what has landed in this series.

Recent highlights (2026-07-17):

- `PerpLaplacianInverseSolver` fast path (`check_residual=False`, no
  diagnostics → phi-only jitted solve, no host syncs), honored
  `phi_inversion_tol`, and a prefactored LU coarse solve in the multigrid
  hierarchy; the 4-field RK4 step now compiles as one jit program
  (1.200 s → 0.623 s per step at `(24, 32, 8)` on one CPU).
- New `jax_drb.native.stellarator_turbulence` (whole-step turbulence driver,
  moved out of `tests/`) and `jax_drb.native.fci_differentiable_case`
  (reusable differentiable-FCI case API, moved out of the example).
- `jax_drb.config.boutinp.rewrite_input_precision()` (moved from the
  precision-benchmark example).
- vmec_jax geometry adapter (`jax_drb.geometry.vmec_jax_import`) with
  closed/open field-line examples under `examples/geometry-3D/vmec-jax/`.
- Examples rewritten as flat PARAMETERS-block pedagogical scripts (no CLI
  flags); docs gained math rendering (`pymdownx.arithmatex`), a Tutorials
  section, and the Models/Solvers reference pages; docs media is now
  committed compressed under `docs/media/`.

## 1.0.3

- Aligned the README, examples, packaging docs, and execution plan on one
  solver boundary: compact native solvers and differentiable examples are
  promoted where evidence supports them, while full output-window recycling
  stays on the compatibility BDF path and JAX-linearized/JVP lanes remain
  opt-in.
- Refreshed the private docs-media release bundle and verified
  `scripts/fetch_example_artifacts.py` restores all 174 manifest media files.
- Extended the artifact downloader with shared cache-directory, timeout, and
  retry controls.
- Added `CITATION.cff` and the `scripts/audit_release_readiness.py` pre-tag
  audit (version, release notes, citation, artifact counts, workflow wiring,
  and repository footprint).
- Closed out the near-term stellarator vacuum-geometry scope as
  machine-readable workflow boundaries and made the anomalous-diffusion
  guard-cell path safe for JAX as well as NumPy arrays.

## 1.0.2

- Switched neutral-mixed one-step/short-window parity to internal BDF substeps,
  cutting the `NVh` history error from about 3.37e-3 to about 5.81e-4.
- Closed the neutral `NVh` pressure-gradient, parallel-viscosity, and
  perpendicular-viscosity source diagnostics to roundoff after active-domain
  scaling.
- Recorded the resolved BDF Jacobian mode and JVP batch size alongside RHS,
  cache-hit, and Jacobian-callback counters on the SciPy BDF compatibility
  path.
- Added batched D/T/He residual/JVP and atomic-rate throughput gates (measured
  CPU speedups, plus a real GPU win on the atomic-rate kernel) and sanitized
  local paths in committed artifacts.

## 1.0.1

- Routed the backward-Euler, BDF2, and legacy BDF recycling paths through the
  fixed-layout recycling state bridge.
- Exposed opt-in `sparse_jvp` and `jax_linearized` recycling solver modes with
  `JAX_DRB_RECYCLING_JACOBIAN_MODE` and `JAX_DRB_RECYCLING_JVP_BATCH_SIZE`
  controls.
- Ingested a patched Hermes `SNVh_pressure_gradient` diagnostic for the
  neutral-mixed `NVh` campaign and refreshed the CPU scaling and native
  profile artifacts.
- Fixed the PyPI workflow to publish once, on release publication or manual
  dispatch, instead of on both a tag push and the release event.

## 1.0.0

- First packaged distribution: `pyproject.toml`, PyPI Trusted Publishing, and a
  Python 3.10-3.12 test workflow.
- Promoted native validation lanes in 1D, 2D, and reduced 3D with structured
  runtime, comparison, convergence, and profiling artifact bundles.
- TOML-driven native runs plus a Python API, verbose run logs, restart bundles,
  and portable JSON/NPZ outputs.
- Bounded controller, recycling, neutral, impurity, and geometry-adapter
  validation surfaces behind a bounded closeout coverage gate.
