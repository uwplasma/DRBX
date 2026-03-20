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
- the drift-wave short-window slice is now in place:
  - `drift_wave_short_window` runs through the native runner with an adaptive reduced branch that keeps the validated density, momentum, vorticity, and potential history on the committed 50-output benchmark window;
  - the current transient milestone is locked by benchmark scalars rather than a single global array tolerance: `gamma / omega_*` and `omega / omega_*` match the committed reference analysis to within the documented test tolerances;
  - the density boundary reconstruction now uses the same `gradient * dx` guard update implied by the benchmark input and confirmed by the reference dump;
  - a committed `drift_wave_one_step_diagnostics` array baseline now locks the evolved-state `ddt(Ni)`, `ddt(NVe)`, and `ddt(Vort)` comparison, so the first post-step operator drift is regression-tested directly.
  - the validation layer now also emits a source-neutral short-window parity report with benchmark-scalar deltas and per-field max/RMS error histories, so the published docs figures are derived from the exact same comparison artifact used for review.
- the next drift-wave transient slice is still under active investigation:
  - native finite-volume parallel electron transport and `phi` dissipation stencils have been reconstructed for the benchmark-specific branch;
  - their normalized strength depends on the same `rho_s0` scaling already used by `Grad_par`, which is now captured in the native implementation;
  - these extra transient-only operators currently move the benchmark away from parity, so they remain staged behind the validated reduced branch until their long-window effect is matched.
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

The next queued staged baselines are now committed as well:

- `neutral_mixed_rhs`, `neutral_mixed_one_step`, and `neutral_mixed_short_window`, with corrected `h`-species compare variables and an explicit `output_ddt` RHS baseline;
- `blob2d_rhs`, `blob2d_one_step`, and `blob2d_short_window`, so the upcoming sheath-connected blob work starts from stored low-iteration targets instead of ad hoc runs.
- `jax-drb validate-reference-baselines`, which re-runs committed reference cases and compares the live summaries to the stored baseline JSON files as a smoke-validation step.

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
- `blob2d_rhs`: implemented and regression-tested against the committed curvature-driven blob baseline;
- `blob2d_one_step`: implemented and regression-tested against the committed single-output blob baseline, using the reference-style orthogonal `recalculate_metric` geometry path;
- the blob one-step electrostatic inversion now uses a direct Fourier/tridiagonal solve on NumPy arrays rather than repeated dense solves, which keeps the validated sheath-connected first-output benchmark practical in the default regression suite;
- `blob2d_short_window`: implemented and regression-tested against the committed summary baseline, with benchmark-level parity locked on peak-density and center-of-mass histories from the committed full-array baseline;
- the shared X-Z ExB transport kernel is now vectorized over whole active planes and regression-checked against a scalar reference implementation, which is what made the long blob transient practical without changing limiter or flux semantics;
- `drift_wave_rhs`: implemented and regression-tested on trimmed active-cell outputs;
- `drift_wave_one_step`: implemented and regression-tested on trimmed active-cell outputs;
- `drift_wave_short_window`: implemented and regression-tested against benchmark scalars plus documented field-difference tolerances on the committed array baseline;
- `neutral_mixed_rhs`: implemented and regression-tested on the trimmed active `y` domain, now using the traced covariant `g_22` metric in the parallel FV operators, mirror-style communicated scalar `y` guards at RHS time, exact local `Div_par_mod` / `Div_par_fvv` flux formulas, and documented full-array tolerances against the committed reference RHS baseline;
- the neutral RHS slice now also has a compact diagnosed-reference artifact in [neutral_mixed_rhs_diagnostics.json](/Users/rogerio/local/jax_drb/references/baselines/reference_metrics/neutral_mixed_rhs_diagnostics.json), locking the live reference centerline state, the isolated parallel neutral density term, the advective parallel flows, the neutral sound-speed value, and the `g22` / `g_22` metric semantics that caused the earlier mismatch;
- the neutral source-parity pass now also includes the reference soft-floor formula inside [neutral_mixed.py](/Users/rogerio/local/jax_drb/src/jax_drb/native/neutral_mixed.py), with direct unit tests so later transient work does not silently regress that low-level rule;
- the remaining neutral transient blocker is now narrowed: evaluating the native neutral RHS on the reference one-step state shows the dominant mismatch at the target-adjacent active `y` cells in the parallel viscosity/conduction neighborhood, so transient parity should be treated as a boundary-operator problem rather than a generic Newton/BDF infrastructure problem;
- the active-domain implicit substrate has now been extracted into the shared [solver](/Users/rogerio/local/jax_drb/src/jax_drb/solver) package, so pack/unpack, backward-Euler/BDF2 residual forms, sparse locality/color grouping, grouped difference-quotient Jacobians, and matrix-free/sparse Newton paths are no longer trapped inside the neutral model;
- the neutral implicit branch now consumes that shared solver backbone, which is the first concrete Step 1 freeze of common stepping/Jacobian infrastructure rather than another case-local implementation;
- the shared sparse path now includes backtracking globalization before the Krylov fallback, which is what made the `solver_mode="sparse"` neutral backward-Euler regression stable enough to keep in the suite;
- the electrostatic inversion path is now shared as well through [elliptic.py](/Users/rogerio/local/jax_drb/src/jax_drb/solver/elliptic.py): blob and vorticity now use the same JAX Fourier-Helmholtz / tridiagonal backend rather than separate dense-mode and custom Thomas implementations;
- the new inversion backend now has direct JIT/`grad` coverage in [test_solver_elliptic.py](/Users/rogerio/local/jax_drb/tests/test_solver_elliptic.py), and the vorticity/blob branches now have end-to-end differentiability smoke tests in [test_native_vorticity.py](/Users/rogerio/local/jax_drb/tests/test_native_vorticity.py) and [test_native_blob2d.py](/Users/rogerio/local/jax_drb/tests/test_native_blob2d.py);
- on the current machine, warm compiled kernel timings for the new electrostatic backbone are already in the sub-millisecond range on the small Step 1 fixtures, while one-shot CLI timings remain dominated by Python startup and case-staging overhead; that separation now matters for all future performance judgments;
- the native CLI/runtime entrypoint now enables a persistent JAX compilation cache through [performance.py](/Users/rogerio/local/jax_drb/src/jax_drb/runtime/performance.py), so repeated process launches can reuse compiled executables instead of recompiling the same kernels every time;
- on the current machine, that persistent cache reduces representative repeated CLI runs from `8.968s` to `3.428s` for `vorticity_one_step` and from `3.541s` to `1.575s` for `blob2d_one_step`, so the warm second-run path is now about `2.3x-2.6x` faster without changing numerics;
- the current diffusion history path has JIT and `grad` smoke coverage, so the first transport slice is exercised as an actual differentiable JAX computation rather than only an eager NumPy-style check;
- neutral transient RK scaffolding now exists in the native neutral module, but it is not yet promoted through the runner because the stiff one-step and short-window solves still need a benchmark-clean integrator strategy;
- `jax-drb analyze-neutral-mixed` now postprocesses the committed `neutral_mixed_short_window` array baseline into compact center-history, temperature, total-mass/pressure, and momentum-RMS metrics, plus a documentation figure for the staged neutral branch;
- live reference runs now confirm that the staged neutral transient case uses `cvode` with `BDF` and `gmres` (`rtol = 1e-5`, `atol = 1e-12`, `mxstep = 1000`), so the next neutral transient implementation needs to follow that implicit path rather than tune the explicit RK scaffolding;
- direct active-domain probes now show that a single backward-Euler solve converges robustly but is too diffusive, while simple BDF2 substepping reduces momentum error but still misses the reference density history, so the next neutral transient iteration needs closer reference-style multistep/adaptive behavior rather than more first-order substep tuning;
- direct low-level SciPy BDF probing with the new sparsity pattern is still too slow to be reviewer-safe on the staged neutral one-step case, so the next transient iteration should target a more direct sparse implicit path rather than simply wrapping `solve_ivp`;
- the sparse direct path is now in-tree and routed through the shared solver backbone, but it is not the default validated stepper yet; the public implicit helpers still default to the matrix-free nonlinear solve while transient parity is tightened;
- next target: finish the neutral one-step and short-window transient path on top of the hardened RHS slice by reproducing the reference `cvode`/BDF role more faithfully, while keeping the trimmed active-domain reference baselines and the blob benchmark report in sync with future solver changes.
