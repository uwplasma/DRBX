# Parity Harness

The first executable parity harness is centered on the curated case ladder in [references/reference_case_ladder.toml](/Users/rogerio/local/jax_drb/references/reference_case_ladder.toml).

For a figure-first view of the currently locked cases, see [docs/validation_gallery.md](/Users/rogerio/local/jax_drb/docs/validation_gallery.md).

## Step Status

| Case | Status | Meaning |
| --- | --- | --- |
| `evolve_density_rhs` | `native-validated` | Smallest one-RHS parity target is locked. |
| `diffusion_one_step` | `native-validated` | First transport transient is locked. |
| `diffusion_short_window` | `native-validated` | Repeated transport history is locked. |
| `fluid_1d_mms_rhs` | `native-validated` | Trimmed interior MMS RHS is locked. |
| `fluid_1d_mms_one_step` | `native-validated` | First coupled 1D MMS advance is locked. |
| `fluid_1d_mms` | `native-validated` | Short-window MMS history is locked. |
| `vorticity_rhs` | `native-validated` | Electrostatic RHS parity is locked. |
| `vorticity_one_step` | `native-validated` | First electrostatic output interval is locked. |
| `vorticity_short_window` | `native-validated` | Electrostatic short-window history is locked. |
| `blob2d_rhs` | `native-validated` | Blob RHS parity is locked. |
| `blob2d_one_step` | `native-validated` | First blob transient is locked. |
| `blob2d_short_window` | `native-validated` | Blob short-window benchmark is locked. |
| `drift_wave_rhs` | `native-validated` | Drift-wave RHS parity is locked. |
| `drift_wave_one_step` | `native-validated` | First drift-wave output interval is locked. |
| `drift_wave_short_window` | `native-validated` | Drift-wave benchmark history is locked. |
| `neutral_mixed_rhs` | `native-validated` | Neutral active-domain RHS is locked. |
| `neutral_mixed_one_step` | `reference-only target` | Baseline exists; native transient is not runner-promoted. |
| `neutral_mixed_short_window` | `reference-only target` | Baseline exists; native transient is not runner-promoted. |
| `recycling_1d_rhs` | `native-validated` | Open-field recycling RHS is locked. |
| `recycling_dthe_rhs` | `native-validated` | Multispecies recycling RHS is locked. |
| `recycling_1d_one_step` | `blocked` | Native transient solve is not parity-clean yet. |
| `recycling_dthe_one_step` | `blocked` | Native transient solve is not parity-clean yet. |
| `recycling_1d_long` | `blocked` | Depends on the transient ladder. |
| `integrated_2d_recycling_rhs` | `native-scaffolded target` | Stable integrated 2D recycling geometry target with staged grid artifact, 10-rank launch, a native local-dump RHS scaffold, slab-local target routing, dump-state sheath preservation, dump-backed `SNd`/`SNd+` and `SPd`/`SPd+` staging for the integrated case, restored staged target-recycling diagnostics, and a source-faithful `sheath_boundary_simple` electron boundary path. |
| `integrated_2d_recycling_one_step` | `native-scaffolded target` | Stable integrated 2D recycling first-output target with staged grid artifact, 10-rank launch, and a first native transient scaffold that starts from the staged one-RHS dump state, marches one native recycling step, preserves the dump-backed target state during transient RHS evaluations and when rebuilding the final staged recycling diagnostics, and uses the same dump-backed `SNd`/`SNd+` and `SPd`/`SPd+` staging during the step that already tightened the staged RHS path. |
| `integrated_2d_recycling_short_window` | `native-scaffolded target` | Stable integrated 2D recycling short-window target with staged grid artifact, 10-rank launch, committed reference baselines, and a native multi-output transient scaffold over the full configured `nout=5` window using the same dump-backed source staging and target-state preservation as the one-step path. |
| `integrated_2d_recycling_medium_window` | `native-scaffolded target` | Stable integrated 2D recycling medium-window target with staged grid artifact, 10-rank launch, committed reference baselines, and a native multi-output transient scaffold over the staged `nout=20` window, again using the same dump-backed source staging and target-state preservation as the shorter transient paths. The remaining meaningful residuals are now led by `Sd_target_recycle` and `Pe`, after the staged recycling-energy path was corrected to use the configured sheath `gamma_i`. |
| `alfven_wave_rhs` | `native-scaffolded target` | First Stage 4 electromagnetic rung on the finite-electron-mass Alfvén-wave benchmark, routed through a partially native scaffold that now reconstructs both `Ajpar` and `Apar` natively while preserving exact committed parity for the staged diagnostic RHS fields. |
| `alfven_wave_one_step` | `native-scaffolded target` | First-output electromagnetic state rung on the same Alfvén-wave benchmark, routed through the same partially native scaffold, with `Ajpar` reconstructed from charged momentum and `Apar` solved natively from the EM Helmholtz equation on the slab/Neumann geometry while the remaining EM state fields stay staged. |
| `tokamak_recycling_one_step` | `blocked` | Reference-side geometry staging is not stable yet. |
| `tokamak_recycling_dthe_one_step` | `blocked` | Reference-side geometry staging is not stable yet. |

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
For recycling debugging, `jax-drb compare-recycling <expected> <actual>` auto-detects summary JSON or portable array NPZ inputs and prints the worst variable plus the worst cell/index location for array payloads without requiring any large artifact format.

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
- `recycling_1d_rhs` is now implemented natively and locked against both the committed portable summary baseline and the committed full-array NPZ baseline;
- the native recycling branch currently vendors its active hydrogen/helium AMJUEL fits inside the package, so the staged open-field parity path does not depend on a separate external rate-data checkout;
- `recycling_dthe_rhs` is now also implemented natively and live-reference clean at the committed summary tolerances; the field-level array comparison passes against the committed NPZ baseline at `5e-2` relative tolerance, and the last staged multispecies edge residual was closed by adding the missing D-T ion-ion thermal-force exchange when `override_ion_mass_restrictions = true`;
- the native recycling branch now also carries the traced upstream density-feedback controller physics:
  - controller source-shape evaluation from `N<species>:source_shape`,
  - proportional/integral multiplier diagnostics,
  - stored controller-integral auxiliary state for transient stepping;
- focused regressions now lock the initial controller behavior for both the single-species and multi-species 1D recycling cases, so that source term cannot silently disappear again while the transient ladder is being finished;
- a compact recycling parity scaffold now exists in [parity/diff.py](/Users/rogerio/local/jax_drb/src/jax_drb/parity/diff.py) and [parity/recycling.py](/Users/rogerio/local/jax_drb/src/jax_drb/parity/recycling.py), giving per-variable max-absolute-difference and argmax-location reports plus staged reference controller snapshot extraction for the one-step recycling cases;
- the `recycling_1d_one_step` and `recycling_dthe_one_step` native parity tests are intentionally `xfail`-guarded by default until the transient solver matches the reference behavior; set `JAX_DRB_RUN_RECYCLING_ONE_STEP_PARITY=1` to opt into an explicit native probe when debugging the transient path;
- [diagnose_recycling_timeline.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_timeline.py) now gives a compact target-band timeline report for `recycling_1d_one_step` and `recycling_dthe_one_step`, so the first-output mismatch can be tracked field-by-field without writing large artifacts;
- [diagnose_recycling_dense_history.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_dense_history.py) now stages a dense-output reference run (for example `nout=40`, `timestep=125`) and compares that trajectory directly to the native transient history, which is the fastest way to see whether drift is already present at each accepted step or only in the coarse first-output comparison;
- the same ladder now stages `integrated_2d_recycling_rhs` and `integrated_2d_recycling_one_step` off the stable integrated `2D-recycling` workflow, with manifest-driven `process_count = 10` and explicit artifact staging for `grid_test2.nc` from the published reference bundle;
- the integrated `2D-recycling` ladder now also includes `integrated_2d_recycling_short_window`, so Step 3 has a committed multi-output transient reference target on the stable integrated workflow rather than only `one_rhs` and `one_step`;
- the same ladder now also includes `integrated_2d_recycling_medium_window`, so Step 3 can exercise a longer staged transient on the stable integrated workflow before the broader 2D production path is exposed;
- the native runner can now enter `integrated_2d_recycling_rhs` through a staged local-dump path, which loads the local `BOUT.dmp.0.nc` mesh/metric/state slab instead of failing on missing `mesh:nx/ny/nz`, honors whether the slab owns a physical lower or upper target, preserves dump-backed target states without reapplying sheath closures, keeps the sheath-generated guard cells needed by the transport stencil, injects dump-backed `SNd`/`SNd+` density sources and `SPd`/`SPd+` ion-pressure sources for the staged integrated case, restores staged `Sd_target_recycle` and `Ed_target_recycle` directly from the dump, and follows the reference `sheath_boundary_simple` electron closure closely enough to remove the large target-row `ddt(Pe)` deficit; this is still a Step 3 scaffolding milestone rather than a locked parity result;
- the native runner can now enter `integrated_2d_recycling_short_window` through the same staged dump-backed workflow and march the full configured `nout=5` window; current native/reference differences are still visible in `Ed_target_recycle`, `Pe`, and tiny neutral-side residuals, so this is the main Step 3 transient parity target rather than a locked parity milestone;
- the native runner can now also enter `integrated_2d_recycling_medium_window` through the same staged workflow while honoring the manifest `nout=20` override; after correcting the staged recycling-energy path to use the configured sheath `gamma_i`, the remaining native/reference differences are led by `Sd_target_recycle`, then `Pe`, with `Ed_target_recycle` reduced into the same small residual band as the other staged diagnostics;
- that staged transient path now consumes dump-backed `SNVd+` / `SNVd` momentum-source fields alongside `SNd` / `SNd+` and `SPd` / `SPd+`. A direct trial of staging `SPe` as a total electron-pressure source was rejected because it increased the medium-window `Pe` drift, so the remaining Step 3 work should focus on broader 2D recycling channels rather than forcing `Pe` from the dump.
- the same dump-backed geometry lane now also supports `integrated_2d_production_rhs`, giving Step 3 a broader integrated 2D RHS target beyond the recycling-specific benchmark. Its first live compare is already limited to the same small `ddt(Pe)` / `ddt(Pd)` pressure-stat deltas seen on the integrated recycling RHS path.
- that same lane now also supports `integrated_2d_production_one_step`, which widens Step 3 from RHS-only production checks into the first output interval on the broader integrated case. The first live compare is not parity-clean yet; the dominant one-step residuals are in `Pe`, then `Pd`, then `Sd_target_recycle`, so that is the next focused production-side transient target.
- `integrated_2d_production_one_step` now forces the `bdf` transient backend instead of inheriting the single-ion continuation default. That materially tightens `Pd` and improves `Pe`; the remaining field-level miss is led by `Pe`, while `Sd_target_recycle`, `Ed_target_recycle`, `NVd+`, and `Nd` are now all in the same small residual band.
- the same broader production lane now also has `integrated_2d_production_short_window`, widening Step 3 from a single production output into a multi-output production transient. It is still a scaffold target rather than a parity win; the current residuals are led by `Pe`, then `Sd_target_recycle`, then `NVd+` / `Nd`, while `Pd` is already much tighter. A direct switch to `bdf` for that longer window improved `Pd` but worsened `NVd+`, so the next improvement should come from physics/source closure rather than another backend flip alone.
- the same broader production lane now also has `integrated_2d_production_medium_window`, extending the wider production transient ladder through `nout=20`. It is likewise a scaffold target rather than a parity win, and it confirms the same residual ordering over a longer interval: `Pe` remains the dominant gap, followed by `Sd_target_recycle` and the neutral/momentum side (`NVd+`, `Nd`), while `Pd` stays relatively tighter.
- Step 4 is now formally staged in the same harness with `alfven_wave_rhs` and `alfven_wave_one_step`, both driven by the smallest finite-electron-mass electromagnetic benchmark in the source test suite. Both now have a dump-backed native scaffold: the first rung reproduces the committed `Apar`, `Ajpar`, `phi`, `Vort`, `NVe`, `ddt(NVe)`, and `ddt(Vort)` `nout=0` baseline exactly, and the second does the same for the first evolved EM state before wider dispersion-history comparisons are added.
- the first real electromagnetic operator slices are now ported inside that scaffold: `Ajpar` is no longer read from the dump, but reconstructed natively from the charged-species momentum sum, and `Apar` is solved natively from the EM Helmholtz equation using charged-density `alpha_em`, normalization-derived `beta_em`, and the periodic-Y/Neumann-X slab guard conventions observed in the saved reference field. The live `alfven_wave_rhs` and `alfven_wave_one_step` summary and array comparisons remain exact after those replacements.
- the older `tokamak_recycling_one_step` and `tokamak_recycling_dthe_one_step` manifest entries should still be treated as blocked reference-side geometry targets until their initialization conflicts are fixed upstream;
- shared open-field operator utilities are now available in [open_field.py](/Users/rogerio/local/jax_drb/src/jax_drb/native/open_field.py), covering no-flow guard fills, limited free extrapolation, electron force balance, parallel electric-force deposition, and target-recycling source assembly before these terms are wired into the coupled native recycling runner;
- the recycling transient branch now uses a continuation-based sparse implicit ladder on top of the shared backward-Euler stepper rather than the older generic adaptive BDF wrapper; the packed RHS still reuses the cached runtime model and the sparse Newton path still uses a direct sparse linear solve on these small active systems, but the full first-output recycling cases remain too slow to promote as parity-complete yet;
- an experimental `adaptive_be` transient mode is now available for localized Step 2 diagnostics; it performs accepted/rejected backward-Euler step doubling on top of the shared sparse implicit solve, and the short `timestep = 25` recycling probe now uses it to verify that the dominant remaining one-step mismatch can be reduced sharply without changing the locked RHS operators;
- an experimental `adaptive_bdf` transient mode is now available as the next Step 2 candidate. It combines backward-Euler startup with BDF2 continuation, uses the same active-domain WRMS error norm as the localized diagnostics, and keeps the recycling-aware initial `dt` heuristic. It remains internal-only until the full first-output recycling interval is both parity-clean and fast enough to replace the current staged transient routes;
- the public recycling runner now chooses the transient backend per one-step case:
  - `recycling_1d_one_step` still uses the continuation ladder, because the current BDF wrapper does not materially improve the single-species first-output parity;
  - `recycling_dthe_one_step` now routes through the existing BDF path, because that path completes the multispecies first-output interval while the continuation ladder is still vulnerable to zero-Jacobian failures on that case;
- the active transient branch now matches two additional reference-side details:
  - controller integrals are advanced on accepted steps with a trapezoid rule instead of being solved as extra implicit unknowns;
  - the sheath preparation order applies the electron boundary state before the ion boundary state so the ion sheath sees the electron boundary density/pressure fields;
- the current transient blocker is now localized by the short-step reference probe in [diagnose_recycling_transient_step.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_transient_step.py):
  - on a `timestep = 25` single-species recycling run, the native backward-Euler step still misses the evolved state (`Nd+` about `5.96e-2`, `NVd+` about `1.54e-1`);
  - when the native RHS is evaluated on that same reference-evolved state, the localized operator differences remain small (`ddt(Nd+)` about `1.03e-6`, `ddt(NVd+)` about `9.06e-4`, `ddt(Pe)` about `1.53e-4`);
  - that short-step split is the current proof that the remaining Step 2 defect is the transient integrator path itself rather than another missing recycling/open-field source term;
  - a fixed `100 x dt = 25` backward-Euler march over the same first-output interval still runs to `NaN` after about `34.5 s` on this machine, which rules out “just use smaller backward-Euler steps” as the Step 2 completion path;
- performance of the transient debug loop improved materially after the latest open-field cleanup:
  - the shared no-flow / limited-free / target-recycling helpers now have NumPy fast paths, so they no longer spend most of their time in `jax.numpy` scatter/device-put when called from the recycling solver;
  - the current native probe runtimes are about `40.6 s` for `recycling_1d_one_step` and about `76.9 s` for `recycling_dthe_one_step` on this machine;
- parity for those first-output cases is still blocked, but the remaining defect is now tightly localized:
  - `recycling_1d_one_step`: max active-domain absolute error is now down to about `1.00e-1`, worst field `Nd+` at the penultimate target-adjacent active cell;
  - the dominant remaining single-species first-output errors are still concentrated in `Nd+`, `Pe`, `Nd`, and `NVd` in the top two active `y` cells, which is why the timeline report is now part of the default Step 2 debugging workflow;
  - the newest dense-history diagnostics now separate that target-band drift into field history and controller/source history:
    - [diagnose_recycling_target_cell_history.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_target_cell_history.py) shows the earliest visible single-species drift at `t = 25` in `NVd+`, followed by `Nd`, then `Nd+` / `Pe`;
    - [diagnose_recycling_controller_history.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_controller_history.py) shows that the controller multiplier and target-recycling source do drift, but by much less than the target-cell state itself, so the remaining Step 2 blocker is not just the PI-source history in isolation;
  - a real neutral-state bug was fixed during that target-band diagnosis: neutrals were inheriting the charged-species default temperature floor (`0.1`) in the recycler sanitizer, which inflated `Pd` near the target and drove `Sd_Dpar` / `Ed_Dpar` positive; neutrals now default to zero temperature floor unless explicitly configured otherwise, and the dense `dt = 1`, total-time `25` single-species probe dropped into the low-`1e-3` / `1e-2` range on all tracked fields after that fix;
  - a direct continuation-step sweep (`suggested_dt = 500, 100, 50, 25, 10`) does not materially improve the single-species first-output mismatch; the active-domain max error stays at about `1.00e-1`, so Step 2 will not be finished by simply shrinking the continuation substep;
  - the accepted-step controller path is still under the staged reference restart integral after the first output interval (`~5.05` native versus `~6.81` reference for `d+`), but the resulting feedback-source delta is too small to explain the remaining target-band state error by itself;
  - a more reference-like mutable-controller `bdf` callback path now runs much faster than the older generic BDF wrapper, but it still lands at essentially the same `recycling_1d_one_step` error level, so the remaining defect is now known to survive both the continuation ladder and the reference-like BDF callback variant;
  - the current `bdf` path has now been cleaned up so the feedback integrals are part of the ODE state and the RHS is pure during SciPy stepping; that is the right formulation to keep, but it does not materially improve the long single-species first-output parity on its own;
  - direct short-window checks on that cleaned `bdf` path show a sharper picture of the remaining blocker:
    - over `t = 25`, `Nd+`, `Pd+`, and `Pe` are already at about `1e-4` relative error;
    - the visible short-step miss is concentrated in `NVd+`, `Nd`, `Pd`, and especially near-zero `NVd`;
    - by `t = 250`, the charged channels are down to roughly `1e-2`, while the neutral-side channels still dominate the relative error;
  - shrinking the internal SciPy BDF `max_step` from `25` to `10` or `5` leaves the long single-species one-step error essentially unchanged, so the remaining Step 2 blocker is not just coarse BDF internal stepping; the next work needs to tighten the neutral-side transient evolution itself;
  - the new [diagnose_recycling_neutral_transient.py](/Users/rogerio/local/jax_drb/scripts/diagnose_recycling_neutral_transient.py) report now makes the denominator issue explicit:
    - on the short `t = 25` probe, only `NVd` is mostly a near-zero-reference artifact;
    - `Nd`, `Pd`, `NVd+`, `ddt(Nd)`, and `ddt(NVd+)` still carry real `O(5e-2 .. 1e-1)` significant relative error after applying a `1e-2 * max(|ref|)` magnitude floor;
    - `SNVd+` and the dumped ionization / charge-exchange force terms are already tight there, which is why the next pass should stay focused on neutral diffusion / pressure / momentum evolution rather than revisiting charged-source bookkeeping;
  - the latest localized recycling RHS fix removes most of that short-window neutral-side gap:
    - recycling neutrals now assemble `ddt(Nd)`, `ddt(Pd)`, and `ddt(NVd)` with the same final transport/compression pattern used by the evolving density/pressure/momentum components instead of treating those channels as source-only;
    - [neutral_mixed.py](/Users/rogerio/local/jax_drb/src/jax_drb/native/neutral_mixed.py) now follows the reference `DDY / sqrt(g_22)` centered metric form for `Grad_par` instead of the older `1 / (J * Δy)` approximation, which materially reduces the target-adjacent ion-momentum remainder;
    - on the fresh `timestep = 25` single-species probe, `Nd`, `Pd`, `Nd+`, `Pd+`, and `Pe` are now all below `1e-3` significant relative error, `NVd` is still denominator-sensitive, and the only clearly visible short-window blocker is the target-band `NVd+` channel at roughly `5e-2` significant relative error;
  - `recycling_dthe_one_step`: the continuation path is still blocked by the sparse Jacobian inversion failure, but the existing native `bdf` path now reaches the first output interval successfully and is the current candidate route for the multi-species Step 2 milestone;
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
- the current staged `integrated_2d_recycling_rhs` runtime split is now measured explicitly:
  - on this machine, the full curated case path takes about `3.89 s`;
  - loading the already-written local dump takes about `35 ms`;
  - the first direct dump-backed RHS evaluation takes about `4.2 ms`;
  - repeated direct dump-backed RHS evaluations average about `4.5 ms`;
  - that means near-term performance work should focus on avoiding unnecessary reference reruns during iteration, not on micro-optimizing the current staged RHS kernel first.
- the current staged `integrated_2d_recycling_rhs` surface is also not end-to-end differentiable yet:
  - the present `grad` barrier is a `TracerArrayConversionError` triggered by `np.asarray(..., copy=True)` in `_initialize_species()` inside [recycling_1d.py](/Users/rogerio/local/jax_drb/src/jax_drb/native/recycling_1d.py);
  - an xfailed regression now tracks that limitation directly in [test_native_integrated_2d_recycling.py](/Users/rogerio/local/jax_drb/tests/test_native_integrated_2d_recycling.py);
  - the fastest near-term differentiability improvement is therefore to keep the staged harness as-is while gradually replacing the early NumPy materialization points in the dump-backed RHS path, rather than attempting a larger Step 3 solver rewrite.

For a reproducible timing report on the staged integrated 2D RHS path, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_integrated_2d_recycling_rhs.py
```

For a reproducible residual-classification report on the staged integrated 2D cases, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_integrated_2d_recycling_parity.py --reference-root /Users/rogerio/local/hermes-3
```

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
