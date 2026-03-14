# Parity Matrix

This document tracks the parity buildout against the private reference implementation. `legacy/` is archival only and is not part of the active implementation plan. For visual snapshots of the validated slices, see [docs/validation_gallery.md](/Users/rogerio/local/jax_drb/docs/validation_gallery.md).

## Stage 1: Configuration And Runtime Skeleton

Goal: reproduce reference input semantics, component scheduling semantics, and normalization bookkeeping before any PDE kernels are ported.

Deliverables:

- `BOUT.inp` parser with section/key order preservation.
- Scalar expression resolver for reference-style numeric expressions such as `AA = 1/1836`, `dy = Ly / ny`, `Bnorm = mesh:Bxy`, and `bxcvz = 1./Rxy^2`.
- Component expansion from `[model] components = ...` and per-species `type = ...`.
- Normalization model reproducing `Nnorm`, `Tnorm`, `Bnorm`, `Cs0`, `Omega_ci`, `rho_s0`, and output-unit metadata.
- Scheduler contract that executes all `transform()` hooks before any finalization hook, matching `ComponentScheduler::transform()` semantics in reference.
- Immutable state container for fields, diagnostics, and metadata.
- CLI skeleton for `inspect` and `run --dry-run`.

Tests:

- parser ordering and comment stripping;
- multiline component lists;
- numeric expression resolution;
- normalization formulas;
- scheduler ordering;
- live reference output summary extraction from `BOUT.dmp.0.nc`.

## Stage 2: Parity Harness

Goal: support `nout = 0` one-RHS checks, one fixed step, and short-time windows against selected reference inputs.

Deliverables:

- manifest of reference cases;
- live reference runner that stages isolated work directories and applies parity-mode overrides such as `nout=0` and `nout=1`;
- NetCDF summary extraction for selected compare variables and scalar metadata;
- NetCDF full-array extraction for small curated cases and compressed baseline artifacts;
- reference/JAX comparison harness;
- reference dump metadata schema;
- first regression baselines;
- native JAX `one_rhs` execution for `evolve_density_rhs`, including structured-mesh coordinates, array-expression evaluation, boundary reconstruction, portable summary emission, and baseline regression tests;
- native JAX `one_step` execution for `diffusion_one_step`, including strict `H(...)` support, structured metric normalization, Neumann guard reconstruction, and an exact one-step radial transport advance;
- native JAX `short_window` execution for `diffusion_short_window`, including repeated transport-history output and field-level array comparisons against committed baselines.
- native JAX `one_rhs` execution for `fluid_1d_mms_rhs`, including case-specific diagnostic overrides on the reference side, trimmed interior RHS comparisons, periodic-Y guards, and MC-limited parallel flux operators;
- native JAX `one_step` execution for `fluid_1d_mms_one_step`, including coupled density/pressure/momentum RK4 subcycling on the periodic 1D MMS benchmark;
- native JAX `short_window` execution for `fluid_1d_mms`, including 50-output regression coverage against committed full-array baselines.
- native JAX `one_rhs` execution for `vorticity_rhs`, including the exact discrete X-Z XPPM advection operator and diagnostic `ddt(Vort)` parity;
- native JAX `one_step` execution for `vorticity_one_step`, including Fourier-in-`z` / tridiagonal-in-`x` potential inversion and `phi` output parity;
- native JAX `short_window` execution for `vorticity_short_window`, including adaptive JAX ODE integration across the full 10-output electrostatic benchmark window.
- native JAX `one_rhs` execution for `blob2d_rhs`, including curvature-driven `ddt(Vort)` parity on the first sheath-connected blob benchmark.
- native JAX `one_step` execution for `blob2d_one_step`, including orthogonal `recalculate_metric` parity, electrostatic inversion on the blob geometry, ExB density advection, and sheath-current closure on the first transient output.
- native JAX `short_window` execution for `blob2d_short_window`, including the full 50-output transient on the blob benchmark plus reviewer-facing peak-excess and center-of-mass parity metrics.
- native benchmark-specific `one_rhs` execution for `drift_wave_rhs`, including quasineutral density closure, fixed-temperature electron pressure, electron-ion drag, spectral potential inversion, and trimmed interior-cell parity against the committed reference baseline;
- native benchmark-specific `one_step` execution for `drift_wave_one_step`, including coupled density, electron momentum, vorticity, and potential output parity on the first 2D density-vorticity benchmark;
- native benchmark-specific `short_window` execution for `drift_wave_short_window`, using the validated reduced adaptive branch over the full 50-output benchmark window;
- native `one_rhs` execution for `neutral_mixed_rhs`, including the reference neutral diffusivity formula, mirror-style communicated scalar `y` guards at RHS time, traced covariant `g_22` metric usage in the parallel FV operators, exact local `Div_par_mod` / `Div_par_fvv` flux formulas, and active-domain parity against trimmed neutral baselines with documented field-level tolerances;
- compact diagnosed-reference regression for `neutral_mixed_rhs`, including the live reference centerline state, isolated parallel density term, parallel advective flows, neutral sound speed, and `g22` / `g_22` metric values from [neutral_mixed_rhs_diagnostics.json](/Users/rogerio/local/jax_drb/references/baselines/reference_metrics/neutral_mixed_rhs_diagnostics.json);
- active-domain implicit neutral stepping substrate, including pack/unpack of the solved domain, backward-Euler residual assembly, and matrix-free Newton-Krylov convergence tests, staged for the upcoming neutral `one_step` / `short_window` parity pass but not yet wired into the public runner.
- active-domain neutral Jacobian sparsity construction, matching the local `x/y/z` stencil and cross-field coupling needed by the traced `cvode`/BDF/GMRES reference path, staged for the next sparse implicit transient pass.
- neutral benchmark postprocessing on the committed `neutral_mixed_short_window` arrays, including center-history extraction, derived temperature tracking, total neutral mass/pressure histories, momentum-RMS decay, CLI reporting, JSON export, and a documentation figure;
- drift-wave operator-scale regressions locked against the committed `drift_wave_one_step` arrays so the parallel transport and scalar damping terms can be tuned without breaking the validated first-output milestone.
- drift-wave benchmark postprocessing on the committed `drift_wave_short_window` arrays, including measured growth/frequency extraction, analytic dispersion evaluation, CLI reporting, JSON export, and a documentation figure.
- drift-wave short-window parity reporting on the committed `drift_wave_short_window` arrays plus current native output, including benchmark deltas, per-field error histories, JSON export, and a documentation figure.
- evolved-state drift-wave diagnostics locked against a committed reference `one_step` baseline with `ddt(Ni)`, `ddt(NVe)`, and `ddt(Vort)`, so the first post-step density operator mismatch is regression-tested directly.

## Stage 3+: Physics Buildout

The remaining stages stay as defined in [PLAN.md](/Users/rogerio/local/jax_drb/PLAN.md):

- mesh and metric parity;
- finite-volume operators and MMS parity beyond the periodic 1D fluid branch;
- 1D open-field fluid core;
- sheath, recycling, and control terms;
- 2D electrostatic drifts and density-vorticity coupling beyond the current drift-wave `one_step` branch;
- 3D electromagnetic capabilities;
- neutrals, reactions, and impurities;
- performance, packaging, validation, and documentation.
