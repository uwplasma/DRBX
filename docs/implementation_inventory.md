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

### Step 2/3 Status Markers

| Case | Status | Note |
| --- | --- | --- |
| `neutral_mixed_rhs` | `native-validated` | Active-domain RHS parity is locked. |
| `neutral_mixed_one_step` | `reference-only target` | Baseline exists; native transient is not runner-promoted. |
| `neutral_mixed_short_window` | `reference-only target` | Baseline exists; native transient is not runner-promoted. |
| `recycling_1d_rhs` | `native-validated` | Open-field recycling RHS is locked. |
| `recycling_dthe_rhs` | `native-validated` | Multispecies recycling RHS is locked. |
| `recycling_1d_one_step` | `blocked` | Native first-step transient is not parity-clean yet. |
| `recycling_dthe_one_step` | `blocked` | Native first-step transient is not parity-clean yet. |
| `integrated_2d_recycling_rhs` | `native-scaffolded` | Staged local-dump RHS path now runs natively with slab-local target routing, dump-state preservation, dump-backed `SNd`/`SNd+` density-source replacement, dump-backed `SPd`/`SPd+` ion-pressure source replacement, restored staged target-recycling diagnostics, and a source-faithful `sheath_boundary_simple` electron boundary path; parity is not locked, but the current near-term work should focus on dump-backed source completion plus staged-path performance/differentiability instrumentation rather than a larger solver rewrite. |
| `integrated_2d_recycling_one_step` | `native-scaffolded` | First native transient scaffold now starts from the staged one-RHS dump state, marches one native recycling step on the shared implicit backbone, preserves the dump-backed target state during transient RHS evaluations and on the final diagnostic pass, and uses the same dump-backed `SNd`/`SNd+` and `SPd`/`SPd+` staging during the step that tightened the staged RHS path. |
| `integrated_2d_recycling_short_window` | `native-scaffolded` | The same integrated 2D recycling workflow now has a committed short-window reference baseline and a native multi-output transient scaffold over the full configured `nout=5` window, reusing the dump-backed source staging and target-state preservation from the one-step path. The remaining live mismatch is concentrated in `Ed_target_recycle`, `Pe`, and tiny neutral-side residuals rather than a broad geometry or source failure. |
| `integrated_2d_recycling_medium_window` | `native-scaffolded` | The same integrated 2D recycling workflow now also has a committed medium-window reference baseline and a native multi-output transient scaffold over the staged `nout=20` window, again reusing the dump-backed source staging and target-state preservation from the shorter transient paths. After correcting the staged recycling-energy path to use the configured sheath `gamma_i`, the remaining live mismatch is led by `Sd_target_recycle`, then `Pe`, with `Ed_target_recycle` reduced into the small residual band. |
| `blob2d_short_window` | `native-validated` | Blob benchmark history is locked. |
| `drift_wave_short_window` | `native-validated` | Drift-wave benchmark history is locked. |

The next queued staged baselines are now committed as well:

- `neutral_mixed_rhs`, `neutral_mixed_one_step`, and `neutral_mixed_short_window`, with corrected `h`-species compare variables and an explicit `output_ddt` RHS baseline;
- staged RHS baselines for [recycling_1d_rhs.json](/Users/rogerio/local/jax_drb/references/baselines/reference/recycling_1d_rhs.json) and [recycling_dthe_rhs.json](/Users/rogerio/local/jax_drb/references/baselines/reference/recycling_dthe_rhs.json), so the divertor branch now has explicit target-recycling source and `ddt(...)` parity checkpoints before the first output step;
- staged one-step open-field recycling baselines for [recycling_1d_one_step.json](/Users/rogerio/local/jax_drb/references/baselines/reference/recycling_1d_one_step.json) and [recycling_dthe_one_step.json](/Users/rogerio/local/jax_drb/references/baselines/reference/recycling_dthe_one_step.json), so Step 2 now has low-iteration sheath/recycling targets for both the single-species and multi-species divertor workflows before any native recycling runner is exposed;
- staged evolved-state RHS regressions against the live reference dumps for `recycling_1d_one_step` and `recycling_dthe_one_step`, so the open-field operator fixes are now locked before the transient runner is promoted;
- staged manifest entries for `integrated_2d_recycling_rhs`, `integrated_2d_recycling_one_step`, `integrated_2d_recycling_short_window`, and `integrated_2d_recycling_medium_window`, including external artifact staging and `process_count = 10`, so the 2D recycling branch now has stable integrated geometry targets from one-RHS through a longer staged transient window rather than the broken tokamak example;
- the native runner now has a first integrated 2D recycling RHS scaffold: it stages the reference workdir, loads the local `BOUT.dmp.0.nc` mesh/metric/state slab through a NetCDF-backed snapshot loader, and reuses the existing open-field recycling RHS stack instead of failing immediately on missing `mesh:nx/ny/nz`;
- that Step 3 scaffold is intentionally not counted as a parity win yet: the recent slab-local target, dump-state preservation, dump-backed `SNd`/`SNd+` and `SPd`/`SPd+` staging, restored staged target-recycling diagnostics, and source-faithful `sheath_boundary_simple` electron boundary fixes moved the integrated `2D-recycling` live compare down to 9 strict summary issues, all in `ddt(Pd)`, `ddt(Pd+)`, and `ddt(Pe)` summary statistics rather than missing recycling channels;
- the same staged workflow now extends through the full configured `nout=5` short window, so Step 3 transient work can move on the stable integrated geometry before the blocked tokamak examples are revived.
- that same native transient path now also honors the staged `nout=20` medium-window override, so Step 3 can measure longer transient drift on the stable integrated geometry before the broader 2D production path is promoted.
- the staged integrated transient path now also consumes dump-backed ion momentum-source fields (`SNVd+`, `SNVd`) in addition to the staged density and ion-pressure source fields. A direct attempt to stage `SPe` was measured and rejected because it worsened the medium-window `Pe` drift, which is a useful constraint for the next Step 3 slice.
- the same dump-backed geometry lane now also has a committed `integrated_2d_production_rhs` reference baseline, widening Step 3 beyond the recycling-only benchmark while preserving the same stable artifact-staged workflow and 10-rank reference launch. The production RHS path now also uses the same ion-only target-preservation split as the transient lane, which removes the old target-band `ddt(NVd+)` miss entirely and leaves only small `ddt(Pe)` / `ddt(Pd)` residuals.
- the same lane now also has a committed `integrated_2d_production_one_step` reference baseline and native entry path. It is not yet a parity win, but the blocker is much narrower now: with ion-only target preservation active on both the RHS and transient paths, the current one-step target-band residuals are down to about `1.5e-1` in `Pe`, `7.8e-2` in `Sd_target_recycle`, and `5.8e-2` in `NVd+`.
- recent production-specific probes narrowed that one-step blocker further: naively staging dump-backed `SPe` or disabling dump-target preservation both make the overall production one-step surface worse, so the remaining work is in the target-band recycling/source update itself rather than another broad source-override pass.
- the staged production harness now also loads per-time dumped ion velocity diagnostics (`Vd+`, `Vd`) when rebuilding `Sd_target_recycle` / `Ed_target_recycle` on the integrated production ladder. That is still a staged-harness move rather than a native-kernel claim, but it materially tightens the remaining production recycling residual:
  - `integrated_2d_production_one_step` `Sd_target_recycle` max abs diff dropped from about `1.65e-1` to about `7.78e-2`;
  - `integrated_2d_production_short_window` `Sd_target_recycle` max abs diff dropped from about `7.81e-1` to about `3.75e-1`.
- that production one-step path now explicitly selects the stiffer `bdf` transient backend instead of the default single-ion continuation path. The main practical effect is that `Pd` tightens sharply and `Pe` improves, leaving `Pe` as the dominant remaining production one-step residual rather than a broad state drift.
- the same integrated production lane now also has a committed `integrated_2d_production_short_window` reference baseline and native scaffold. It is not parity-clean yet, but it gives Step 3 the next broader 2D transient rung beyond production one-step and confirms the current residual ordering after the ion-only preservation fix on both paths: `Pe` is now the leading short-window target-band field (about `1.43` max abs diff), followed by `NVd+` (about `9.8e-1`) and `Sd_target_recycle` (about `3.75e-1`), while `Pd+` and `Nd+` are much tighter.
- the staged integrated production transient entry now also applies the committed optional-history `Vd+`/`Vd` overrides to its initial state before the implicit march. That keeps the transient lane internally consistent with the already-corrected production RHS lane and is now locked by the cache-based regression tests, even though it does not materially change the current production residual ranking on its own.
- that same broader production lane now also has a committed `integrated_2d_production_medium_window` reference baseline and native scaffold. It extends the production transient ladder to `nout=20` and shows the same broader pattern over a longer interval: `Pe` is still the dominant production residual, followed by `Sd_target_recycle` and the neutral/momentum side, while `Pd` remains comparatively tighter.
- a broader staged production experiment that replayed per-interval dumped total source histories inside the transient march was measured and rejected. It did not materially improve `Sd_target_recycle`, and it made the target-band `Pe` / `NVd+` state errors worse, so the integrated production lane deliberately stays on the narrower staged source surface while the remaining target-band update is diagnosed directly.
- the latest narrow Step 3 correction is lower-level than the staged source surface: `preserve_dump_ion_target_state_only` now keeps the sheath-generated ion guard cells and ion sheath energy sinks while preserving the dump-backed ion target cell itself. That is the correct local state split for the integrated production target band, and the new behavior is locked by focused integrated-2D tests.
- the integrated production diagnostics are now cheaper to use between code changes as well: [diagnose_integrated_2d_production_parity.py](/Users/rogerio/local/jax_drb/scripts/diagnose_integrated_2d_production_parity.py) can compare against the committed baselines instead of rerunning Hermes, and it can filter to target-band-only worst cells. That should be the default Step 3 inner loop before paying for a fresh live-reference run.
- that same diagnostic path now also has a reusable time-trace mode for the worst target-band cells. The current committed-baseline traces make the remaining production blocker concrete: on `integrated_2d_production_one_step`, `Pe` starts diverging on the first accepted step at one target-row cell while `NVd+` and `Sd_target_recycle` diverge together at the adjacent deeper target-band cell, and on `integrated_2d_production_short_window` those same residuals then grow monotonically at every stored output. That means the next Step 3 fix should be evaluated against first-step target-band history, not only final-window maxima.
- the staged integrated production runner now also supports compact committed snapshot caches, with the shared [integrated_2d_production_rhs_snapshot.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_snapshots/integrated_2d_production_rhs_snapshot.npz) plus staged [integrated_2d_production_one_step_optional_history.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_snapshots/integrated_2d_production_one_step_optional_history.npz) and [integrated_2d_production_short_window_optional_history.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_snapshots/integrated_2d_production_short_window_optional_history.npz) already committed. That removes the repeated live-reference bootstrap for the production transient ladder and eliminates the repeated staged diagnostic-history fetch for both the one-step and short-window rungs, giving Step 3 a second, lower-latency harness improvement in addition to the committed-baseline target-band diagnostics. The same path can now be extended incrementally to the medium-window rung without more runner changes.
- Step 4 is now formally staged with the smallest electromagnetic benchmark in the source test suite:
  - `alfven_wave_rhs` locks `Apar`, `Ajpar`, `phi`, `Vort`, `NVe`, `ddt(NVe)`, and `ddt(Vort)` from an `nout=0` diagnostic run on `tests/integrated/alfven-wave`;
  - `alfven_wave_one_step` locks the first evolved EM state on the same benchmark through `Apar`, `Ajpar`, `phi`, `Vort`, and `NVe`;
  - `alfven_wave_short_window` now locks the first multi-output EM transient rung on the same benchmark at `nout=20`, which is the smallest saved history that yields a stable benchmark-quality phase-speed estimate while staying well under the repository artifact size cap;
  - `alfven_wave_medium_window` now locks the longer default-history EM transient rung on the same benchmark at `nout=50`, which still fits under the repository artifact size cap and gives Step 4 a stronger transient parity surface;
  - `annulus_he_emag_rhs` now locks a broader electromagnetic RHS rung on `examples/other/linear/annulus-isothermal-he-emag`, using a slim EM-only compare surface (`Apar`, `alpha_em`, `ddt(Ne)`, `ddt(NVe)`, `ddt(Vort)`) so the committed array artifact stays under the repository size cap while widening Step 4 beyond the slab benchmark;
  - `annulus_he_emag_one_step` now locks a curated small-step transient rung on the same annulus benchmark with `timestep=10` and compare variables `Apar`, `Ne`, `NVe`, `phi`, and `Vort`; the committed array artifact is about `1.7 MB`, so it widens the EM transient surface without violating the repository size rule;
  - `annulus_he_emag_short_window` now locks a curated multi-output annulus EM transient rung with `timestep=10`, `nout=5`, and a slim transient compare surface (`Apar`, `Ne`, `phi`); the committed array artifact is about `4.6 MB`, so it remains under the repository cap while giving Step 4 a second non-slab short-window history;
  - both rungs now also run through a dump-backed native scaffold in [runner.py](/Users/rogerio/local/jax_drb/src/jax_drb/native/runner.py), and the live `run-case` outputs compare exactly to the committed summary and array baselines;
  - the first real EM operator slices are now in-tree: `Ajpar` is computed natively from the charged-species momentum sum, `alpha_em` is computed natively from charged densities, `Apar` is solved natively on the single-cell slab/Neumann Alfvén benchmark from the source-faithful Helmholtz coefficients (`alpha_em` from charged densities, `beta_em` from normalization), the one-step physical/inner-radial `NVe` planes are reconstructed by inverting that same slab solve, the `nout=0` physical/inner-radial `ddt(NVe)` core is ported as the benchmark’s periodic central-difference closure on `Vort` in `y` and `z`, and the `nout=0` inner-radial shoulder `ddt(Vort)` planes on `x=1,3` are ported as the benchmark’s exact inner-radial `DDY/ DDZ` closure. The tiny central-plane `x=2` `ddt(Vort)` signal and the outermost saved radial and parallel guard planes remain staged where the reference dump is not current-consistent. The live `one_step` / `one_rhs` summary and array compares remain exact after those replacements;
  - the low-level EM unit surface now also mirrors the source unit test more directly through [test_native_electromagnetic.py](/Users/rogerio/local/jax_drb/tests/test_native_electromagnetic.py): zero-flutter on constant `Apar`, explicit DC-subtraction behavior for `compute_apar_flutter(...)`, and the existing `Apar`/current/`alpha_em` operator checks;
  - the new validation layer now also postprocesses that short-window array baseline into an analytic-vs-measured Alfvén-wave benchmark report and a native/reference parity report through [validation/alfven_wave.py](/Users/rogerio/local/jax_drb/src/jax_drb/validation/alfven_wave.py), with CLI entrypoints in [cli.py](/Users/rogerio/local/jax_drb/src/jax_drb/cli.py);
  - current benchmark numbers on the committed short-window reference history are:
    - analytic phase speed about `9.48585409e+05 m/s`;
    - measured phase speed about `9.42218662e+05 m/s`;
    - relative phase-speed error about `6.71e-03`;
  - the longer medium-window history stays in the same band:
    - measured phase speed about `9.42628846e+05 m/s`;
    - relative phase-speed error about `6.28e-03`;
  - native/reference parity on the committed short-window arrays is exact on the current scaffold.
  - that gives the EM/`Apar` branch the same staged `one_rhs -> one_step` ladder used earlier for electrostatic and recycling paths before wider short-window dispersion checks are added.
- the staged integrated `2D-recycling` path now also has explicit performance and differentiability tracking:
  - on this machine, the full curated case run is about `3.89 s`, while loading the local dump is about `35 ms`, the first direct dump-backed RHS evaluation is about `4.2 ms`, and repeated direct RHS evaluations average about `4.5 ms`;
  - the staged path is therefore currently harness-bound rather than kernel-bound for Step 3 iteration;
  - the current dump-backed RHS surface is not yet `grad`-traceable, with the first hard barrier at `np.asarray(..., copy=True)` in `_initialize_species()`; that limitation is now tracked by an explicit xfailed regression rather than only ad hoc probing;
  - reproducible local timing and residual-classification helpers now exist at [benchmark_integrated_2d_recycling_rhs.py](/Users/rogerio/local/jax_drb/scripts/benchmark_integrated_2d_recycling_rhs.py), [diagnose_integrated_2d_recycling_parity.py](/Users/rogerio/local/jax_drb/scripts/diagnose_integrated_2d_recycling_parity.py), and [diagnose_integrated_2d_production_parity.py](/Users/rogerio/local/jax_drb/scripts/diagnose_integrated_2d_production_parity.py), so future Step 3 performance and parity changes can be measured against the same staged surfaces.
- `blob2d_rhs`, `blob2d_one_step`, and `blob2d_short_window`, so the upcoming sheath-connected blob work starts from stored low-iteration targets instead of ad hoc runs.
- `jax-drb validate-reference-baselines`, which re-runs committed reference cases and compares the live summaries to the stored baseline JSON files as a smoke-validation step.
- `recycling_1d_rhs`: implemented and regression-tested against committed summary and full-array baselines, including target-recycling diagnostics, sheath boundary fluxes, AMJUEL-based ionization/recombination, hydrogenic charge exchange, literal section-reference resolution for source expressions, and the open-field electron-force-balance source path;
- the active recycling package now vendors the compact atomic-rate JSON files inside [src/jax_drb/data/atomic_rates](/Users/rogerio/local/jax_drb/src/jax_drb/data/atomic_rates), so the open-field recycling branch no longer depends on out-of-tree rate data for the currently staged hydrogen and helium reactions;
- `recycling_dthe_rhs`: now implemented and regression-tested against the committed summary baseline, with full-array parity passing at the documented `5e-2` tolerance; the traced fixes were the exact multispecies ion-neutral collision-table ordering and the previously missing cross-isotope D-T charge-exchange channels;
- the latest localized multispecies fix was the missing ion-ion thermal-force exchange for the D-T pair when `override_ion_mass_restrictions = true`, which brings the staged `SNVd+` / `SNVt+` diagnostics back inside the locked tolerances and further narrows the remaining Step 2 defect to the transient solver path;
- the native recycling RHS now also includes the upstream density-feedback controller source path and its stored integral state semantics at the operator level:
  - single-species initial feedback diagnostics are now regression-tested as zero when the upstream density starts on target;
  - the multi-species helium controller now reproduces the expected nonzero initial proportional multiplier while still depositing zero density source when the configured source shape is zero;
- the recycling transient infrastructure now reuses a cached runtime model rather than rebuilding species/controller/source metadata on every packed RHS call, and the shared sparse Newton backend now supports direct sparse linear solves for small active systems; that combination reduced the packed `1D-recycling` RHS evaluation cost from the old O(1e-1 s) range to about `2.9e-2 s` per call on this machine while preserving all current RHS parity locks;
- the recycling runner now uses a continuation-based sparse implicit ladder rather than the old generic adaptive BDF wrapper, and that ladder has a dedicated finite-step regression in the suite; the full `recycling_1d_one_step` and `recycling_dthe_one_step` cases are still not promoted as parity-complete because the output-interval runs remain too slow for the current Step 2 budget;
- a new experimental `adaptive_be` recycling transient mode is now in-tree and regression-tested at the short-step level; it reuses the shared sparse backward-Euler solve with accepted/rejected step doubling, and on the `timestep = 25` single-species probe it reduces the dominant ion/electron state errors from O(1e-1) to O(1e-3 .. 1e-2), which is the strongest current evidence that Step 2 can be closed by a more accurate accepted-step march rather than another RHS/source rewrite;
- a follow-on experimental `adaptive_bdf` recycling transient mode is now in-tree as the intended replacement for that diagnostic-only probe. It uses backward-Euler startup, BDF2 continuation, WRMS error control, and the recycling-aware initial `dt` heuristic. It is not routed through the public runner yet because the full `recycling_1d_one_step` output interval is still too expensive to treat as a completed Step 2 milestone;
- the open-field transient path now follows the reference controller/state ordering more closely:
  - upstream density-feedback integrals are updated with an accepted-step trapezoid rule rather than being solved as extra implicit unknowns;
  - the sheath preparation order now applies the electron boundary state before the ion boundary state, and the ion sheath uses the electron boundary density/pressure fields instead of the pre-sheath quasineutral sum;
- the transient blocker is now localized with a dedicated short-step probe in [diagnose_recycling_transient_step.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_transient_step.py):
  - on a `timestep = 25` single-species reference run, the native backward-Euler step still misses the evolved state by about `5.96e-2` on `Nd+` and about `1.54e-1` on `NVd+`;
  - on that same reference-evolved state, the native RHS stays tight, with max diffs around `1.03e-6` for `ddt(Nd+)`, `9.06e-4` for `ddt(NVd+)`, and `1.53e-4` for `ddt(Pe)`;
  - that is the clearest current evidence that the remaining Step 2 mismatch is in the transient integrator path rather than in the localized recycling/open-field RHS operators;
  - a follow-on `100 x dt = 25` backward-Euler march over the same first-output interval still runs to `NaN` after about `34.5 s` on this machine, so smaller backward-Euler accepted steps alone are not enough to finish Step 2;
- recent transient performance work materially reduced the current first-output cost:
  - NumPy fast paths in [open_field.py](/Users/rogerio/local/jax_drb/src/jax_drb/native/open_field.py) remove the old `jax.numpy` scatter/device-put overhead when the recycling solver is working with plain NumPy state;
  - the neutral parallel operator in [neutral_mixed.py](/Users/rogerio/local/jax_drb/src/jax_drb/native/neutral_mixed.py) is now vectorized;
  - the current probe runtimes are about `40.6 s` for `recycling_1d_one_step` and about `76.9 s` for `recycling_dthe_one_step` on this machine;
- the latest source-level Step 2 fix came from the target-band diagnostics rather than another solver rewrite: neutrals were inheriting the charged-species default `temperature_floor = 0.1`, which forced `Pd` too high near the target and then biased `Sd_Dpar` / `Ed_Dpar`; neutrals now default to zero temperature floor unless the input file explicitly overrides it, and the dense `dt = 1`, total-time `25` single-species recycling probe now lands in the low-`1e-3` / `1e-2` range across the tracked fields;
- those first-output probes are still not parity-clean:
  - `recycling_1d_one_step` is now down to about `1.00e-1` in the trimmed active domain, worst field `Nd+` at the penultimate target-adjacent cell;
  - the remaining single-species first-output mismatch is concentrated in `Nd+`, `Pe`, `Nd`, and `NVd` in the top two active `y` cells, and the new [diagnose_recycling_timeline.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_timeline.py) report is now the fastest way to track those fields against the staged reference dump;
  - the new dense-history diagnostics, [diagnose_recycling_target_cell_history.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_target_cell_history.py) and [diagnose_recycling_controller_history.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_controller_history.py), now separate the target-cell drift from the controller/source drift: the earliest visible single-species state error appears first in `NVd+` at `t = 25`, while the controller multiplier and target-recycling source drift remain noticeably smaller;
  - a direct continuation sweep over `suggested_dt = 500, 100, 50, 25, 10` leaves the single-species first-output max error essentially unchanged, so the current blocker is not simple continuation substep resolution;
  - the current accepted-step controller history still underestimates the staged reference restart integral (`~5.05` native versus `~6.81` reference after the first output interval for `d+`), but the resulting feedback source difference is too small to explain the whole target-band state error;
  - a more reference-like mutable-controller `bdf` callback path now exists in-tree and is materially faster on the single-species one-step probe, but it still lands at essentially the same first-output state error, so the remaining Step 2 defect survives both the continuation ladder and the faster BDF callback variant;
  - evaluating the native one-step result through the same sheath-preparation path confirms that the remaining active-cell error is not just a raw-versus-prepared output mismatch: the target-adjacent active `Nd+` and `Pe` values remain off even though the prepared upper guard cell moves much closer to the staged reference dump;
  - `recycling_dthe_one_step` is still blocked on the default continuation ladder, but the native `bdf` path now reaches the first output interval cleanly and is the current candidate for the multispecies Step 2 transient route;
- the latest transient diagnosis tightened the single-species picture further:
  - the SciPy `bdf` path is now mathematically cleaner because the feedback integrals are part of the ODE state and the RHS is pure rather than side-effecting controller memory;
  - that cleanup does not materially change the long `recycling_1d_one_step` result by itself, and shrinking the BDF `max_step` from `25` to `10` or `5` also leaves the final error essentially unchanged, so the current blocker is not just coarse internal BDF stepping;
  - on the other hand, a short `timestep = 25` reference comparison shows the charged channels are already close on that BDF path (`Nd+`, `Pd+`, `Pe` all in the `1e-4` relative range), while the visible remaining short-step mismatch is concentrated in `NVd+`, `Nd`, `Pd`, and especially near-zero `NVd`;
  - a medium `t = 250` probe keeps the same pattern: charged channels are down to roughly `1e-2`, while neutral-side channels still dominate the relative error. The next Step 2 work should therefore target the neutral / neutral-momentum transient terms directly rather than another generic recycling timestepper rewrite.
- the new magnitude-aware short-window neutral diagnostic in [diagnose_recycling_neutral_transient.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_neutral_transient.py) confirms that the remaining short-step mismatch is not only a small-denominator artifact:
  - `NVd` is mostly denominator-driven because the reference stays near zero;
  - `Nd`, `Pd`, `NVd+`, `ddt(Nd)`, and `ddt(NVd+)` still show real `O(5e-2 .. 1e-1)` significant relative error on cells above a `1e-2 * max(|ref|)` magnitude floor;
  - `SNVd+` and the currently dumped ionization / charge-exchange force diagnostics are already tight on that same probe, so the next fix should focus on the neutral diffusion / pressure / momentum transient evolution rather than the charged-source bookkeeping.
- the latest localized Step 2 RHS fix removed most of that short-window neutral-side error:
  - the recycling neutrals now assemble `ddt(Nd)`, `ddt(Pd)`, and `ddt(NVd)` with the same final transport/compression pattern used by the evolving density/pressure/momentum components instead of treating those channels as pure source equations;
  - [neutral_mixed.py](/Users/rogerio/local/jax_drb/src/jax_drb/native/neutral_mixed.py) now follows the reference `DDY / sqrt(g_22)` centered metric form for `Grad_par` instead of the older `1 / (J * Δy)` approximation, which materially reduces the target-adjacent ion-momentum remainder in the recycling probe;
  - on a fresh `timestep = 25` single-species recycling reference run, `Nd`, `Pd`, `Nd+`, `Pd+`, and `Pe` are now all below `1e-3` significant relative error, `NVd` remains dominated by near-zero denominators, and the only still-visible short-window mismatch is the target-band `NVd+` channel at roughly `5e-2` significant relative error.
- the next open-field blocker is no longer the 1D source stack; it is the transient ladder above it, starting with `recycling_1d_one_step` and `recycling_dthe_one_step`, then moving to the short-window and long-run divertor cases on the shared implicit solver;

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
- the live reference harness now launches binaries with `cwd` set to the staged case directory, which is required for any curated case that uses relative mesh paths rather than only self-contained structured inputs;
- the same harness can now also launch curated multi-rank reference cases through a manifest `process_count`, so future tokamak geometry baselines can be validated without ad hoc shell wrappers;
- the first shared open-field utilities now exist in [open_field.py](/Users/rogerio/local/jax_drb/src/jax_drb/native/open_field.py), covering the traced no-flow guard semantics, limited free extrapolation, electron force balance, parallel electric-force deposition, and target-recycling source assembly, all with direct regression and differentiability tests;
- the current diffusion history path has JIT and `grad` smoke coverage, so the first transport slice is exercised as an actual differentiable JAX computation rather than only an eager NumPy-style check;
- neutral transient RK scaffolding now exists in the native neutral module, but it is not yet promoted through the runner because the stiff one-step and short-window solves still need a benchmark-clean integrator strategy;
- `jax-drb analyze-neutral-mixed` now postprocesses the committed `neutral_mixed_short_window` array baseline into compact center-history, temperature, total-mass/pressure, and momentum-RMS metrics, plus a documentation figure for the staged neutral branch;
- live reference runs now confirm that the staged neutral transient case uses `cvode` with `BDF` and `gmres` (`rtol = 1e-5`, `atol = 1e-12`, `mxstep = 1000`), so the next neutral transient implementation needs to follow that implicit path rather than tune the explicit RK scaffolding;
- direct active-domain probes now show that a single backward-Euler solve converges robustly but is too diffusive, while simple BDF2 substepping reduces momentum error but still misses the reference density history, so the next neutral transient iteration needs closer reference-style multistep/adaptive behavior rather than more first-order substep tuning;
- direct low-level SciPy BDF probing with the new sparsity pattern is still too slow to be reviewer-safe on the staged neutral one-step case, so the next transient iteration should target a more direct sparse implicit path rather than simply wrapping `solve_ivp`;
- the sparse direct path is now in-tree and routed through the shared solver backbone, but it is not the default validated stepper yet; the public implicit helpers still default to the matrix-free nonlinear solve while transient parity is tightened;
- the recycling branch now has the same kind of transient substrate in-tree:
  - packed active-domain field state plus controller-integral auxiliary state;
  - backward-Euler residual wiring on the shared implicit backbone;
  - adaptive BDF probing with grouped sparse finite-difference Jacobians;
  - this is not yet promoted as a validated parity path because the current one-step recycling solves are still too slow and not benchmark-clean enough for the committed first-output baselines.
- next targets:
  - finish the first validated recycling transient milestone, starting with `recycling_1d_one_step` and then `recycling_dthe_one_step`;
  - in parallel, replace the broken staged tokamak Step 3 geometry target with the integrated 2D recycling workflow plus explicit artifact staging in the reference harness.
