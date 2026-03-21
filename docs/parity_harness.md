# Parity Harness

The first executable parity harness is centered on the curated case ladder in [references/reference_case_ladder.toml](/Users/rogerio/local/jax_drb/references/reference_case_ladder.toml).

For a figure-first view of the currently locked cases, see [docs/validation_gallery.md](/Users/rogerio/local/jax_drb/docs/validation_gallery.md).

## Live Reference Protocol

`jax-drb run-reference-case <case>` performs the following steps:

1. resolve the case input under a reference checkout;
2. stage the case input directory into an isolated workdir using symlinks, without modifying the reference source tree;
3. apply parity-mode overrides:
   - `one_rhs -> nout=0`
   - `one_step -> nout=1`
   - case-specific overrides from the manifest are merged after parity-mode defaults, so diagnostic cases can request extra outputs without duplicating `nout`
4. run the reference binary;
   - the harness now launches the binary with `cwd` set to the staged workdir, which is required for curated cases that resolve relative mesh files from the case directory;
   - multi-rank geometry cases can now also request a manifest `process_count`, which prefixes the launch with `mpirun -np <N>` while keeping the same staged-workdir flow;
5. verify `BOUT.settings`, `BOUT.log.0`, `BOUT.dmp.0.nc`, and `BOUT.restart.0.nc`;
6. summarize selected comparison variables and normalization scalars from `BOUT.dmp.0.nc`.
7. compare future JAX portable summaries against the committed reference baselines with `jax-drb compare-summary`.

If requested, the same command can also emit full comparison arrays to compressed NPZ files. Those artifacts are intended for the smallest curated cases where full-field regression is practical.
For diagnostic RHS cases and the current drift-wave benchmark, the harness can trim X and/or Y guard cells before writing summary and array baselines so comparisons focus on physically meaningful interior outputs.

## Native Protocol

`jax-drb run-case <case>` resolves the same curated input and runs the supported native JAX path.

Current support is intentionally narrow:

- `evolve_density_rhs` is implemented end to end;
- `diffusion_one_step` is implemented for the current structured, axisymmetric, `nz = 1` transport benchmark;
- `diffusion_short_window` is implemented on the same transport path, using the configured output cadence from the curated input;
- `fluid_1d_mms_rhs` is implemented for trimmed interior RHS parity on the periodic 1D manufactured-solution benchmark;
- `fluid_1d_mms_one_step` is implemented for the first coupled density/pressure/momentum advance;
- `fluid_1d_mms` is implemented for a full 50-output short window on the same benchmark, using RK4 subcycling;
- `vorticity_rhs` is implemented for diagnostic electrostatic RHS parity on the standalone vorticity benchmark;
- `vorticity_one_step` is implemented for the first electrostatic output interval, comparing both `Vort` and `phi`;
- `vorticity_short_window` is implemented for the full 10-output electrostatic benchmark window using an adaptive JAX ODE solve;
- `blob2d_rhs` is implemented for the first sheath-connected blob milestone, matching `Ne`, `Pe`, zero `phi`, zero `ddt(Ne)`, and the curvature-driven `ddt(Vort)` source on the committed reference baseline;
- `blob2d_one_step` is implemented for the first sheath-connected blob transient, using the reference-style orthogonal `recalculate_metric` path together with reduced RK4 evolution for `Ne`, `Vort`, and `phi`, and a direct Fourier/tridiagonal electrostatic solve to keep the benchmark practical in the regression suite;
- `blob2d_short_window` is now implemented for the full 50-output sheath-connected blob benchmark, with the long-window parity locked through the committed summary baseline plus reviewer-facing peak-excess and center-of-mass metrics on a compact committed benchmark artifact;
- `drift_wave_rhs` is implemented for the first coupled 2D density-vorticity benchmark, comparing trimmed active-cell state and RHS outputs;
- `drift_wave_one_step` is implemented for the same benchmark at the first output time;
- `drift_wave_short_window` is now implemented with the validated reduced adaptive branch over the full 50-output benchmark window;
- `neutral_mixed_rhs` is now implemented for the first neutral-fluid milestone, comparing the active `y` domain against trimmed reference baselines while the live diagnosed reference state now also locks the exact RHS-time scalar guard reconstruction and the covariant `g_22` metric usage in the parallel FV operators;
- the neutral RHS slice now also has a committed full-array regression with explicit max/RMS tolerances on `ddt(Nh)` and `ddt(Ph)`, plus a compact diagnosed-reference artifact in [neutral_mixed_rhs_diagnostics.json](/Users/rogerio/local/jax_drb/references/baselines/reference_metrics/neutral_mixed_rhs_diagnostics.json) for the isolated parallel density term and advective-flow centerlines;
- the neutral model now also carries the traced reference soft-floor formula, locked by unit tests before transient parity is reopened;
- the active-domain implicit machinery is now centralized in the shared [solver](/Users/rogerio/local/jax_drb/src/jax_drb/solver) package rather than living only inside the neutral model, so future fluid/recycling/EM branches can reuse the same pack/unpack, Jacobian, and Newton infrastructure;
- the electrostatic inversion path is now centralized there as well through the shared Fourier-Helmholtz backend, which both blob and vorticity now use;
- the neutral branch now regression-tests both the validated matrix-free backward-Euler path and a stable sparse-backend backward-Euler solve on a small active-domain case;
- the public runner still keeps `neutral_mixed_one_step` and `neutral_mixed_short_window` disabled until the shared backbone is driven by a reference-faithful adaptive multistep transient, but the low-level stepper/Jacobian substrate is now frozen into common code rather than staged as a private prototype;
- the current neutral transient mismatch is localized to the target-adjacent active `y` cells in the neutral momentum RHS, with the dominant error sitting in the parallel viscosity/conduction neighborhood rather than the core interior transport operators;
- staged reference-only baselines are now also committed for `neutral_mixed_rhs`, `neutral_mixed_one_step`, `neutral_mixed_short_window`, `blob2d_rhs`, `blob2d_one_step`, and `blob2d_short_window`, so the next transport/sheath implementation passes start from stored low-iteration targets rather than fresh local runs;
- the reference ladder now also includes committed `one_rhs` and `one_step` baselines for the single-species and multi-species 1D recycling workflows, so the sheath/recycling branch can lock target-recycling sources and `ddt(...)` fields before the first output-step state comparison;
- the same ladder now also stages `tokamak_recycling_one_step` and `tokamak_recycling_dthe_one_step` as the first named 2D recycling geometry targets; their committed baselines still need a stable curated processor split, but no longer need ad hoc launch scripting;
- shared open-field operator utilities are now available in [open_field.py](/Users/rogerio/local/jax_drb/src/jax_drb/native/open_field.py), covering no-flow guard fills, limited free extrapolation, electron force balance, parallel electric-force deposition, and target-recycling source assembly before these terms are wired into the coupled native recycling runner;
- the drift-wave branch now also has a locked operator-scale regression on the committed `drift_wave_one_step` arrays, covering the small parallel momentum-flux, drag, and `phi`-damping terms that matter for longer transients;
- the same branch now has a committed `one_step` diagnostics baseline with evolved-state `ddt(Ni)`, `ddt(NVe)`, and `ddt(Vort)` outputs, so density-operator regressions can be caught one step after the initial condition;
- `jax-drb analyze-drift-wave <input> <arrays.npz>` now postprocesses the committed drift-wave short-window arrays into measured growth/frequency scalars, the analytic dispersion target, a JSON report, and a benchmark figure for the docs;
- `jax-drb compare-drift-wave <input> <expected.npz> <actual.npz>` now emits the current short-window drift-wave parity report, including benchmark-scalar deltas plus per-field max/RMS error histories and a documentation figure;
- `jax-drb compare-blob2d <expected.npz> <actual.npz>` now emits the current blob short-window parity report, including peak-density and center-of-mass history deltas plus a documentation figure;
- `jax-drb analyze-neutral-mixed <arrays.npz>` now postprocesses the committed neutral short-window arrays into compact center-history, derived-temperature, total-mass/pressure, and momentum-RMS metrics, so the staged neutral transient has a reviewer-facing target before the native stiff solver is exposed;
- `jax-drb validate-reference-baselines` now re-runs committed reference cases and checks their live summaries against the stored baseline JSON files, so baseline drift can be caught as an explicit smoke-validation step;
- the native runner builds the structured mesh, evaluates the configured initial profile on the JAX grid, reconstructs the current X/Y guards, builds the normalized structured metrics, and emits the portable summary schema;
- the same native run can emit compressed full-array parity artifacts, so small cases can be checked at field level with `jax-drb compare-arrays`;
- the resulting JSON can be compared directly against the committed baseline with `jax-drb compare-summary`.

For performance checks, Step 1 now distinguishes between:

- end-to-end CLI timings, which include Python startup, config parsing, case staging, and summary generation;
- warm compiled kernel timings, which isolate the actual numerical backbone after JAX compilation.

Both matter, but only the second number should be used to judge whether a shared JAX operator backend is fundamentally too slow relative to the private reference.

The CLI entrypoint now also enables a persistent JAX compilation cache by default, so repeated `jax-drb run-case ...` invocations on the same machine can reuse previously compiled kernels. That is part of Step 1 performance hardening and should be kept enabled for parity campaigns unless explicitly debugging compilation behavior. On the current machine, representative repeated runs improved from `8.968s` to `3.428s` for `vorticity_one_step` and from `3.541s` to `1.575s` for `blob2d_one_step` when reusing the same cache directory across processes.

For the current one-step diffusion milestone, summary comparison should use a modest scalar tolerance, for example:

```bash
PYTHONPATH=src python -m jax_drb compare-summary \
  references/baselines/reference/diffusion_one_step.json \
  /tmp/jax_drb_diffusion_one_step_native.json \
  --scalar-rtol 1e-3 \
  --scalar-atol 2e-6
```

That tolerance is only for the first transport milestone. The intent is to tighten it as more of the operator and time-integration stack becomes native and shared across cases.

## Confirmed Reference Behavior

Live runs against `local reference build` established:

- `nout=0` still writes `BOUT.dmp.0.nc`, `BOUT.restart.0.nc`, `BOUT.settings`, and `BOUT.log.0`;
- for `nout=0`, `t_array` contains a single time point `(0.0,)`;
- for `nout=1`, `t_array` contains two time points, the initial state and one output step;
- `BOUT.dmp.0.nc` includes scalar normalization metadata `Nnorm`, `Tnorm`, `Bnorm`, `Cs0`, `Omega_ci`, and `rho_s0`.

These behaviors are the basis of the low-iteration parity workflow in [PLAN.md](/Users/rogerio/local/jax_drb/PLAN.md).

## Committed Reference Baselines

The first portable baseline summaries generated from live reference runs are:

- [evolve_density_rhs.json](/Users/rogerio/local/jax_drb/references/baselines/reference/evolve_density_rhs.json)
- [diffusion_one_step.json](/Users/rogerio/local/jax_drb/references/baselines/reference/diffusion_one_step.json)
- [diffusion_short_window.json](/Users/rogerio/local/jax_drb/references/baselines/reference/diffusion_short_window.json)
- [fluid_1d_mms_rhs.json](/Users/rogerio/local/jax_drb/references/baselines/reference/fluid_1d_mms_rhs.json)
- [fluid_1d_mms_one_step.json](/Users/rogerio/local/jax_drb/references/baselines/reference/fluid_1d_mms_one_step.json)
- [fluid_1d_mms.json](/Users/rogerio/local/jax_drb/references/baselines/reference/fluid_1d_mms.json)
- [vorticity_rhs.json](/Users/rogerio/local/jax_drb/references/baselines/reference/vorticity_rhs.json)
- [vorticity_one_step.json](/Users/rogerio/local/jax_drb/references/baselines/reference/vorticity_one_step.json)
- [vorticity_short_window.json](/Users/rogerio/local/jax_drb/references/baselines/reference/vorticity_short_window.json)
- [neutral_mixed_rhs.json](/Users/rogerio/local/jax_drb/references/baselines/reference/neutral_mixed_rhs.json)
- [neutral_mixed_one_step.json](/Users/rogerio/local/jax_drb/references/baselines/reference/neutral_mixed_one_step.json)
- [neutral_mixed_short_window.json](/Users/rogerio/local/jax_drb/references/baselines/reference/neutral_mixed_short_window.json)
- [drift_wave_rhs.json](/Users/rogerio/local/jax_drb/references/baselines/reference/drift_wave_rhs.json)
- [drift_wave_one_step.json](/Users/rogerio/local/jax_drb/references/baselines/reference/drift_wave_one_step.json)
- [drift_wave_short_window.json](/Users/rogerio/local/jax_drb/references/baselines/reference/drift_wave_short_window.json)
- [blob2d_rhs.json](/Users/rogerio/local/jax_drb/references/baselines/reference/blob2d_rhs.json)
- [blob2d_one_step.json](/Users/rogerio/local/jax_drb/references/baselines/reference/blob2d_one_step.json)
- [blob2d_short_window.json](/Users/rogerio/local/jax_drb/references/baselines/reference/blob2d_short_window.json)

The compact benchmark-metric references are:

- [blob2d_short_window_metrics.json](/Users/rogerio/local/jax_drb/references/baselines/reference_metrics/blob2d_short_window_metrics.json)
- [neutral_mixed_rhs_diagnostics.json](/Users/rogerio/local/jax_drb/references/baselines/reference_metrics/neutral_mixed_rhs_diagnostics.json)
- [neutral_mixed_short_window_metrics.json](/Users/rogerio/local/jax_drb/references/baselines/reference_metrics/neutral_mixed_short_window_metrics.json)

These files are not full field dumps. They intentionally store:

- parity mode and applied overrides;
- required output artifacts;
- output dimensions and time points;
- normalization scalars from `BOUT.dmp.0.nc`;
- selected comparison-variable statistics and first-to-last deltas.

For the staged neutral branch, the committed reference summaries and array baselines now trim `y` guards so the parity checks focus on the active domain while the guard-fill rules remain an explicit follow-up target.

Future JAX runs should emit the same portable schema through the generic summary helpers in [portable.py](/Users/rogerio/local/jax_drb/src/jax_drb/parity/portable.py), so that `jax-drb compare-summary` can be used unchanged for reference vs. JAX comparisons.

The first committed full-array baselines are:

- [diffusion_one_step.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/diffusion_one_step.npz)
- [diffusion_short_window.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/diffusion_short_window.npz)
- [fluid_1d_mms_rhs.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/fluid_1d_mms_rhs.npz)
- [fluid_1d_mms_one_step.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/fluid_1d_mms_one_step.npz)
- [fluid_1d_mms.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/fluid_1d_mms.npz)
- [vorticity_rhs.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/vorticity_rhs.npz)
- [vorticity_one_step.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/vorticity_one_step.npz)
- [vorticity_short_window.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/vorticity_short_window.npz)
- [neutral_mixed_rhs.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/neutral_mixed_rhs.npz)
- [neutral_mixed_one_step.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/neutral_mixed_one_step.npz)
- [neutral_mixed_short_window.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/neutral_mixed_short_window.npz)
- [drift_wave_rhs.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/drift_wave_rhs.npz)
- [drift_wave_one_step.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/drift_wave_one_step.npz)
- [drift_wave_short_window.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/drift_wave_short_window.npz)
- [blob2d_rhs.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/blob2d_rhs.npz)
- [blob2d_one_step.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/blob2d_one_step.npz)

These are written and read through [arrays.py](/Users/rogerio/local/jax_drb/src/jax_drb/parity/arrays.py). For the current diffusion milestone, the intended comparison command is:

```bash
PYTHONPATH=src python -m jax_drb compare-arrays \
  references/baselines/reference_arrays/diffusion_short_window.npz \
  /tmp/jax_drb_diffusion_short_window_native.npz \
  --array-rtol 2e-4 \
  --array-atol 2e-6
```

For the current electrostatic vorticity milestone, the intended comparison command is:

```bash
PYTHONPATH=src python -m jax_drb compare-arrays \
  references/baselines/reference_arrays/vorticity_short_window.npz \
  /tmp/jax_drb_vorticity_short_window_native.npz \
  --array-rtol 2e-3 \
  --array-atol 1e-5
```

For the current drift-wave `one_step` milestone, the intended comparison command is:

```bash
PYTHONPATH=src python -m jax_drb compare-arrays \
  references/baselines/reference_arrays/drift_wave_one_step.npz \
  /tmp/jax_drb_drift_wave_one_step_native.npz \
  --array-rtol 5e-2 \
  --array-atol 5e-6
```

For the current drift-wave `short_window` milestone, the reviewer-facing comparison command is:

```bash
PYTHONPATH=src python -m jax_drb compare-drift-wave \
  /path/to/curated/drift_wave/BOUT.inp \
  references/baselines/reference_arrays/drift_wave_short_window.npz \
  /tmp/jax_drb_drift_wave_short_window_native.npz \
  --json-out docs/data/drift_wave_short_window_parity.json \
  --plot-out docs/images/drift_wave_short_window_parity.png
```

For the current blob `short_window` milestone, the reviewer-facing comparison command is:

```bash
PYTHONPATH=src python -m jax_drb compare-blob2d \
  references/baselines/reference_metrics/blob2d_short_window_metrics.json \
  /tmp/jax_drb_blob2d_short_window_native.npz \
  --json-out docs/data/blob2d_short_window_parity.json \
  --plot-out docs/images/blob2d_short_window_parity.png
```

For the staged neutral `short_window` benchmark target, the current compact-analysis command is:

```bash
PYTHONPATH=src python -m jax_drb analyze-neutral-mixed \
  references/baselines/reference_arrays/neutral_mixed_short_window.npz \
  --x-index 5 \
  --y-index 3 \
  --z-index 5 \
  --json-out references/baselines/reference_metrics/neutral_mixed_short_window_metrics.json \
  --plot-out docs/images/neutral_mixed_short_window_diagnostics.png
```
