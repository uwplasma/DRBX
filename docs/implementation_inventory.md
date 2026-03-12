# Implementation Inventory

These notes capture the external-reference facts already driving the first implementation slices.

## Solver And Scheduler

- the external reference uses both adaptive transient integration (`cvode`) and steady-state/backward-Euler style solves (`beuler`) in the documented workflow.
- `nout = 0` is the shortest parity loop because the reference executes one RHS evaluation and exits.
- `ComponentScheduler::transform()` runs every component's `transform()` first and only then runs each component's `finally()` hook. JAX-DRB mirrors that contract in its initial scheduler abstraction.

## Normalization

From the main source driver, the reference defines:

- `Cs0 = sqrt(qe * Tnorm / Mp)`
- `Omega_ci = qe * Bnorm / Mp`
- `rho_s0 = Cs0 / Omega_ci`
- output-unit metadata: `inv_meters_cubed -> Nnorm`, `eV -> Tnorm`, `Tesla -> Bnorm`, `seconds -> 1 / Omega_ci`, `meters -> rho_s0`

The initial normalization module reproduces those exact derived quantities and tracks both `normalise_metric` and `recalculate_metric`.

## Root And Mesh Scalars

Reference inputs routinely define reusable scalar parameters before `[mesh]` and `[model]`, for example `tnorm_setting`, `core_ne`, and `initial_pi`. Mesh sections then reference local and root scalars (`dy = Ly / ny`, `dz = 2 * pi / nz`, `Bnorm = mesh:Bxy`). JAX-DRB now resolves these into a structured run configuration rather than reparsing them inside later kernels.

## Live Output Facts

Direct runs against the local reference build confirmed:

- `nout=0` writes a `BOUT.dmp.0.nc` file with `t_array = [0.0]`;
- `nout=1` writes initial plus one evolved output time slice;
- scalar normalization metadata is present directly in the dump file;
- for the structured identity-metric transport cases, the dumped metric fields follow the normalized forms `dx / (rho_s0^2 * Bnorm)`, `J / rho_s0`, `g11 / rho_s0^2`, and `Bxy / Bnorm`;
- the first portable reference baselines are stored in [references/baselines/reference](/Users/rogerio/local/jax_drb/references/baselines/reference).
- the first native JAX execution path now matches the committed `evolve_density_rhs` portable baseline exactly, including dimensions, scalar metadata, and variable summary statistics.
- the native one-step transport path now reproduces the committed `diffusion_one_step` summary statistics within regression tolerance, using structured metrics, strict Heaviside support, Neumann guard reconstruction, and an exact matrix-exponential radial advance.
- the transport parity harness now also stores full comparison arrays in [references/baselines/reference_arrays](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays), so regressions can be checked against complete fields instead of summaries only.
- the same transport slice now covers a short-window benchmark, `diffusion_short_window`, using the configured output cadence from the input file.
- the first coupled fluid MMS slice is now in place:
  - `fluid_1d_mms_rhs` compares trimmed interior `ddt(Ni)`, `ddt(Pi)`, and `ddt(NVi)` against a diagnostic reference run;
  - `fluid_1d_mms_one_step` and `fluid_1d_mms` compare full state histories for `Ni`, `Pi`, and `NVi`;
  - the native path uses periodic-Y guard wrapping, MC-limited parallel finite-volume fluxes, centered `Grad_par`, and fixed-step RK4 subcycling.
- the first electrostatic vorticity slice is now in place:
  - `vorticity_rhs` matches the diagnostic `ddt(Vort)` field to machine precision;
  - `vorticity_one_step` and `vorticity_short_window` compare both `Vort` and `phi`;
  - the native path uses the same discrete X-Z XPPM advection stencil as the reference, a Fourier-in-`z` / tridiagonal-in-`x` Boussinesq potential inversion, and an adaptive JAX ODE solve over the 60 evolved interior cells.
- the first coupled 2D drift-wave slice is now in place:
  - `drift_wave_rhs` compares trimmed active-cell `Ni`, `Ne`, `Pe`, `ddt(Ni)`, `ddt(NVe)`, and `ddt(Vort)` outputs;
  - `drift_wave_one_step` compares trimmed active-cell `Ni`, `Ne`, `NVe`, `Vort`, and `phi`;
  - the native path uses a benchmark-specific reduced operator set: ion ExB advection, quasineutral electron density, fixed-temperature electron pressure, electron-ion Braginskii drag, parallel current closure, and Fourier-in-`z` electrostatic inversion with slab `Bxy` recovered from `mesh:B`.
- the committed `drift_wave_short_window` array baseline now also feeds a public benchmark-analysis path:
  - `jax-drb analyze-drift-wave` reports `omega_*`, `sigma_parallel / omega_*`, measured growth/frequency, and the analytic finite-electron-mass dispersion target;
  - the same command can emit JSON plus a documentation figure, so reviewer-facing validation plots are generated from the same stored arrays used by the regression suite.
- the next drift-wave transient slice is under active investigation:
  - native finite-volume parallel electron transport and `phi` dissipation stencils have been reconstructed for the benchmark-specific branch;
  - their normalized strength depends on the same `rho_s0` scaling already used by `Grad_par`, which is now captured in the native implementation;
  - the one-step operator calibration now includes a regression against the committed drift-wave array baseline, locking the small but nonzero scale of parallel momentum flux, drag, and `phi` damping terms on the first evolved state;
  - an adaptive native RK23 history integrator is now available for this branch, so longer transient probes can be run without introducing a new solver dependency;
  - these operators are staged behind the transient path so the validated `one_rhs` and `one_step` baselines remain locked while the longer-window branch is tuned.
- the drift-wave parity harness now trims both X and Y guards for the committed benchmark baselines, because the first implementation target is the physically evolved interior cell rather than reference-specific guard bookkeeping.
- structured metric handling now respects `normalise_metric = false`, which is required for the 1D MMS fluid case and future benchmark inputs that specify already-physical mesh coefficients.
- structured metric handling now also reproduces the default periodic-binormal spacing and normalized `g33` needed by the electrostatic vorticity benchmark.

## Input Syntax Observations

Representative reference inputs require support for:

- inline comments after assignments;
- quoted strings, booleans, integers, and floats;
- symbolic expressions that must stay unevaluated unless scalar resolution is requested;
- top-level comma-separated lists such as `type = evolve_density, evolve_pressure`;
- multiline parenthesized component lists;
- Unicode `π`, section references like `mesh:Bxy`, and power syntax using `^`.

## Selected Reference Cases

The first parity ladder is recorded in [references/reference_case_ladder.toml](/Users/rogerio/local/jax_drb/references/reference_case_ladder.toml). It starts with one-RHS and one-step cases from integrated tests, then grows into blobs, recycling, turbulence, and the TCV X-point example.

Current native execution coverage:

- `evolve_density_rhs`: implemented and regression-tested;
- `diffusion_one_step`: implemented and regression-tested as the first genuine time-advance benchmark;
- `diffusion_short_window`: implemented and regression-tested at both summary and full-array level;
- `fluid_1d_mms_rhs`: implemented and regression-tested on trimmed interior RHS outputs;
- `fluid_1d_mms_one_step`: implemented and regression-tested for the first coupled fluid advance;
- `fluid_1d_mms`: implemented and regression-tested for a 50-output short window;
- `vorticity_rhs`: implemented and regression-tested at summary and full-array level;
- `vorticity_one_step`: implemented and regression-tested for the first electrostatic output interval;
- `vorticity_short_window`: implemented and regression-tested for the full 10-output benchmark window;
- `drift_wave_rhs`: implemented and regression-tested on trimmed active-cell outputs;
- `drift_wave_one_step`: implemented and regression-tested on trimmed active-cell outputs;
- `drift_wave_short_window`: reference baselines committed; native transient support is the next target;
- the current diffusion history path has JIT and `grad` smoke coverage, so the first transport slice is exercised as an actual differentiable JAX computation rather than only an eager NumPy-style check;
- next target: extend the new drift-wave branch from one output interval to the committed short-window transient, then move to sheath-connected blob physics.
