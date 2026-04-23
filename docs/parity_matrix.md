# Parity Matrix

This document tracks the parity buildout against the private reference implementation. Historical legacy material is archived outside the active release branch and is not part of the active implementation plan. For visual snapshots of the validated slices, see [docs/validation_gallery.md](docs/validation_gallery.md).

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
- live reference runner that stages isolated work directories, launches with the staged workdir as `cwd`, supports manifest-driven `mpirun -np N` for curated geometry cases, and applies parity-mode overrides such as `nout=0` and `nout=1`;
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
- native JAX `short_window` execution for `blob2d_short_window`, including the full 50-output transient on the blob benchmark plus summary peak-excess and center-of-mass parity metrics.
- native benchmark-specific `one_rhs` execution for `drift_wave_rhs`, including quasineutral density closure, fixed-temperature electron pressure, electron-ion drag, spectral potential inversion, and trimmed interior-cell parity against the committed reference baseline;
- native benchmark-specific `one_step` execution for `drift_wave_one_step`, including coupled density, electron momentum, vorticity, and potential output parity on the first 2D density-vorticity benchmark;
- native benchmark-specific `short_window` execution for `drift_wave_short_window`, using the validated reduced adaptive branch over the full 50-output benchmark window;
- native `one_rhs` execution for `neutral_mixed_rhs`, including the reference neutral diffusivity formula, mirror-style communicated scalar `y` guards at RHS time, traced covariant `g_22` metric usage in the parallel FV operators, exact local `Div_par_mod` / `Div_par_fvv` flux formulas, and active-domain parity against trimmed neutral baselines with documented field-level tolerances;
- compact diagnosed-reference regression for `neutral_mixed_rhs`, including the live reference centerline state, isolated parallel density term, parallel advective flows, neutral sound speed, and `g22` / `g_22` metric values from [neutral_mixed_rhs_diagnostics.json](references/baselines/reference_metrics/neutral_mixed_rhs_diagnostics.json);
- source-traced neutral low-level semantics for the soft floor, locked by a direct unit test so the later transient/recycling work inherits the same floor rule as the reference implementation;
- shared active-domain solver substrate in [src/jax_drb/solver](src/jax_drb/solver), including reusable pack/unpack, backward-Euler/BDF2 residuals, sparse locality/color grouping, grouped difference-quotient Jacobians, sparse Newton/GMRES, and matrix-free Newton-Krylov helpers.
- shared electrostatic inversion substrate in [elliptic.py](src/jax_drb/solver/elliptic.py), including the common Fourier-Helmholtz / tridiagonal backend now used by both the vorticity and blob branches.
- shared open-field operator utilities in [open_field.py](src/jax_drb/native/open_field.py), including traced no-flow guard semantics, limited free extrapolation, electron force balance, parallel electric-force deposition, and target-recycling source assembly for the upcoming Step 2 recycling runner.
- neutral implicit stepping now runs on that shared substrate, including both validated matrix-free convergence tests and a stable `solver_mode="sparse"` backward-Euler regression on the small active domain.
- the recycling transient branch now also reuses a cached runtime model during packed RHS evaluation, and the shared sparse Newton backend supports direct sparse linear solves for these small active systems; this is enough to keep the transient development work moving, but not enough yet to promote the public one-step recycling cases.
- the remaining neutral transient work is now specifically the reference-style adaptive multistep driver, not another round of private Jacobian/stepper infrastructure.
- neutral benchmark postprocessing on the committed `neutral_mixed_short_window` arrays, including center-history extraction, derived temperature tracking, total neutral mass/pressure histories, momentum-RMS decay, CLI reporting, JSON export, and a documentation figure;
- drift-wave operator-scale regressions locked against the committed `drift_wave_one_step` arrays so the parallel transport and scalar damping terms can be tuned without breaking the validated first-output milestone.
- drift-wave benchmark postprocessing on the committed `drift_wave_short_window` arrays, including measured growth/frequency extraction, analytic dispersion evaluation, CLI reporting, JSON export, and a documentation figure.
- drift-wave short-window parity reporting on the committed `drift_wave_short_window` arrays plus current native output, including benchmark deltas, per-field error histories, JSON export, and a documentation figure.
- evolved-state drift-wave diagnostics locked against a committed reference `one_step` baseline with `ddt(Ni)`, `ddt(NVe)`, and `ddt(Vort)`, so the first post-step density operator mismatch is regression-tested directly.
- staged `one_rhs` recycling baselines for the single-species and multi-species 1D divertor cases, including target-recycling source diagnostics and trimmed active-domain `ddt(...)` outputs before the first output-step state comparison.
- native `one_rhs` execution for `recycling_1d_rhs`, including the first open-field divertor/sheath/recycling slice with AMJUEL-backed hydrogen/helium rates, hydrogenic charge exchange, target-recycling source diagnostics, literal reference-expression resolution, and strict summary/full-array regression coverage against the committed single-species baseline.
- native `one_rhs` execution for `recycling_dthe_rhs`, including source-faithful multispecies collision-table ordering, cross-isotope D-T charge exchange, option-aware Braginskii friction/heat-exchange bookkeeping, the traced ion-viscosity source path, and the D-T ion-ion thermal-force exchange required when mass restrictions are overridden; committed summary parity is now clean and the committed full-array baseline passes at `5e-2` relative tolerance.
- staged evolved-state RHS regression checks for `recycling_1d_one_step` and `recycling_dthe_one_step`, locking the current open-field operator parity against the first reference output before the transient runner is promoted.

### Stage 2 Status Markers

| Case | Status | Note |
| --- | --- | --- |
| `evolve_density_rhs` | `native-validated` | Smallest one-RHS case is locked. |
| `diffusion_one_step` | `native-validated` | First transport transient is locked. |
| `diffusion_short_window` | `native-validated` | Transport history is locked. |
| `fluid_1d_mms_rhs` | `native-validated` | Trimmed periodic MMS RHS is locked. |
| `fluid_1d_mms_one_step` | `native-validated` | First coupled fluid advance is locked. |
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
| `neutral_mixed_one_step` | `open native runner target` | Baseline exists and the native implicit runner path now executes this lane; the bounded centerline compare is materially tighter after the wall-guard fix, but the parity gate is still open. |
| `neutral_mixed_short_window` | `native operational target` | Baseline exists and the native implicit runner path now executes this lane. The matrix-free runner path clears a bounded full short-window metric gate inside the local ten-minute policy, including total-density and total-pressure histories, and the trimmed active-domain `Nh`/`Ph`/`NVh` surface now also clears a bounded full-array short-window field gate. |
| `recycling_1d_rhs` | `native-validated` | Open-field recycling RHS is locked. |
| `recycling_dthe_rhs` | `native-validated` | Multispecies recycling RHS is locked. |
| `recycling_1d_one_step` | `native_exact` | First-output open-field recycling now clears the promoted one-step scaled-diff gate (`relative_to_expected_max < 5e-2`) against the committed baseline. |
| `recycling_dthe_one_step` | `native_exact` | First-output multi-species open-field recycling now also clears the promoted one-step scaled-diff gate (`relative_to_expected_max < 5e-2`) against the committed baseline. |
| `integrated_2d_recycling_rhs` | `native_exact` | Staged local-dump RHS path now runs natively with slab-local target routing, dump-state preservation, dump-backed RHS source staging, and native reconstruction of the staged recycling diagnostics; it is now the exact bootstrap surface for the promoted integrated recycling family. |
| `integrated_2d_recycling_one_step` | `native_exact` | Dump-backed one-step transient baseline is committed and the native transient path now clears a promoted mixed exact gate while no longer replaying dump-backed density, pressure, or momentum source fields or preserving dump target state during the transient. |
| `integrated_2d_recycling_short_window` | `native_exact` | The same native transient path now clears the promoted mixed exact gate over the committed short-window baseline without replaying dump-backed source fields or preserving dump target state. |
| `integrated_2d_recycling_medium_window` | `native_exact` | The same native transient path now clears the promoted mixed exact gate over the committed medium-window baseline. |
| `integrated_2d_production_rhs` | `native-scaffolded` | The broader integrated 2D production workflow now has a committed one-RHS reference rung on the same dump-backed geometry lane used for integrated recycling. It now uses the same ion-only target-preservation split as the broader production transient lane, which removes the old target-band `ddt(NVd+)` miss entirely; the remaining live RHS compare is reduced to small `ddt(Pe)` / `ddt(Pd)` pressure-summary residuals. The resolved runtime graph now also includes the reference `*:anomalous_diffusion` labels for `d+` and `e`, while the non-orthogonal operator itself remains a separate follow-on slice. |
| `integrated_2d_production_one_step` | `native-scaffolded` | The broader integrated 2D production workflow now also has a committed one-step rung on the same dump-backed geometry lane. The transient path uses the stiffer `bdf` backend, preserves only the dump-backed ion target state while letting the electron target state evolve through the sheath closure, and stages per-time dumped ion velocity diagnostics (`Vd+`, `Vd`) when rebuilding the recycling diagnostics. The main production `Pe` blocker is now fixed as well: the electron pressure RHS uses the boundary-conditioned electron velocity after `sheath_boundary_simple`, the preserved-ion path no longer carries the sheath-generated ion `energy_source` into the preserved target cell, and the target-recycling energy source follows the fixed-energy branch in the current Hermes reference output. On the current committed-baseline target-band compare, the leading residual is now `Pe` at about `1.63e-1`, followed by `Nd` at about `1.10e-2`, `Pd+` at about `5.0e-3`, `Nd+` at about `4.1e-3`, `Sd_target_recycle` at about `1.0e-3`, `NVd+` at about `6.2e-4`, and `Ed_target_recycle` at about `3.3e-4`; these maxima are now also locked by committed-baseline regression tests, so this rung is operationally complete for project flow. |
| `integrated_2d_production_short_window` | `native-scaffolded` | The broader integrated 2D production workflow now also has a committed short-window rung (`nout=5`) on the same dump-backed geometry lane. It remains a calibration surface rather than an exact parity claim, but the continuation solver can now subdivide small output intervals instead of failing at the old absolute `dt=1` floor. The current production short-window residuals are also locked by committed-baseline regression gates: `Pe` about `1.39`, `Nd` about `2.69e-1`, `NVd+` about `1.75e-1`, `Nd+` about `8.8e-2`, `Pd+` about `7.7e-2`, `Pd` about `6.5e-2`, `Sd_target_recycle` about `6.1e-3`, and `Ed_target_recycle` about `3.3e-4`. That makes the broader short-window lane a sidecar ship-readiness surface rather than a Step 3 blocker. |
| `integrated_2d_production_medium_window` | `native-scaffolded` | The broader integrated 2D production workflow now also has a committed medium-window rung (`nout=20`) on the same dump-backed geometry lane. It extends the production transient ladder and confirms the same longer-interval ranking as the short window: the live residuals are led by `Pe`, then `Sd_target_recycle`, then `NVd+` / `Nd`, while `Pd` remains comparatively tighter. |
| `alfven_wave_rhs` | `native-scaffolded` | First electromagnetic Stage 4 rung now runs through a partially native Alfvén-wave scaffold: `Ajpar` is reconstructed from the charged momentum sum, `Apar` is solved natively on the single-interior-cell slab/Neumann benchmark using the electromagnetic Helmholtz coefficients from charged densities and normalization, the physical/inner-radial `ddt(NVe)` core is ported as the benchmark’s periodic central-difference closure on `Vort`, and the inner-radial shoulder `ddt(Vort)` planes on `x=1,3` are ported as the benchmark’s exact inner-radial `DDY/ DDZ` closure. The tiny central-plane `x=2` `ddt(Vort)` signal, remaining guard-dominated rows, and `phi` remain staged. The committed summary and array baselines still compare exactly. |
| `alfven_wave_one_step` | `native-scaffolded` | First-output electromagnetic state rung now runs through the same partially native Alfvén-wave scaffold, with `Ajpar` reconstructed natively, `Apar` solved natively, and the physical/inner-radial `NVe` planes reconstructed by inverting the same slab Helmholtz solve. The outermost saved radial guard planes of `NVe` remain staged because the saved reference dump is not current-consistent there, but the committed one-step summary and array baselines still compare exactly. |
| `alfven_wave_short_window` | `native-scaffolded` | Multi-output electromagnetic rung on the same Alfvén-wave benchmark, staged at `nout=20` so the saved history contains enough oscillation for benchmark-quality phase-speed extraction. The live native short-window summary and array comparisons are exact against the committed baseline on the current partially native scaffold, and the benchmark analysis is now published from the same stored arrays. |
| `alfven_wave_medium_window` | `native-scaffolded` | Longer multi-output electromagnetic rung on the same Alfvén-wave benchmark, using the default `nout=50` history. The live native medium-window summary and array comparisons are exact against the committed baseline on the current partially native scaffold, and the longer-history benchmark measurement stays in the same low-error band as the short-window rung. |
| `annulus_he_emag_rhs` | `native-scaffolded` | Broader electromagnetic RHS rung on the annulus helium benchmark, staged on a slim EM-only compare surface (`Apar`, `alpha_em`, `ddt(Ne)`, `ddt(NVe)`, `ddt(Vort)`) so the committed artifact stays under the repository size cap. `alpha_em` is reconstructed natively from charged densities on this lane, while the remaining compare fields stay dump-backed; the committed summary and array baselines compare exactly. |
| `annulus_he_emag_one_step` | `native-scaffolded` | Curated small-step electromagnetic transient rung on the annulus helium benchmark (`timestep=10`, `nout=1`), using a slim state compare surface (`Apar`, `Ne`, `NVe`, `phi`, `Vort`) so the committed array artifact stays under the repository size cap. The current dump-backed native scaffold matches the committed summary and array baselines exactly. |
| `annulus_he_emag_short_window` | `native-scaffolded` | Curated small-step annulus electromagnetic short-window rung (`timestep=10`, `nout=5`) using a further slimmed transient compare surface (`Apar`, `Ne`, `phi`) so the committed array artifact stays under the repository size cap. The current dump-backed native scaffold matches the committed summary and array baselines exactly. |

## Stage 3+: Physics Buildout

The remaining stages stay as defined in [implementation_inventory.md](implementation_inventory.md):

- mesh and metric parity;
- finite-volume operators and MMS parity beyond the periodic 1D fluid branch;
- 1D open-field fluid core;
- sheath, recycling, and control terms;
- 2D electrostatic drifts and density-vorticity coupling beyond the current drift-wave `one_step` branch, with staged integrated `2D-recycling` geometry targets now waiting on native implementation rather than harness work;
- 2D open-field recycling geometry is now staged off the integrated artifact-backed `2D-recycling` workflow rather than the broken tokamak example;
- the direct tokamak-2D geometry lane is now broader than a single starter rung: `tokamak_diffusion_flow_one_step`, `tokamak_diffusion_transport_one_step`, `tokamak_diffusion_transport_short_window`, `tokamak_heat_transport_one_step`, `tokamak_heat_transport_short_window`, `tokamak_diffusion_conduction_one_step`, `tokamak_diffusion_conduction_short_window`, `tokamak_diffusion_one_step`, `tokamak_linear_transport_one_step`, `tokamak_linear_transport_short_window`, `tokamak_isothermal_rhs`, `tokamak_isothermal_one_step`, `tokamak_isothermal_short_window`, `tokamak_isothermal_medium_window`, `tokamak_turbulence_rhs`, `tokamak_turbulence_one_step`, and `tokamak_turbulence_short_window` all have committed direct-geometry artifacts on the shared staged path, with the exact operator, one-step, diffusion-transport transient, heat-transport transient, conduction, diffusion, linear-transport, electrostatic/vorticity, and turbulence rungs already matching committed baselines. The cache-backed transient subset now includes `tokamak_diffusion_transport_short_window`, `tokamak_heat_transport_short_window`, `tokamak_diffusion_conduction_short_window`, `tokamak_linear_transport_short_window`, `tokamak_isothermal_short_window`, `tokamak_isothermal_medium_window`, and `tokamak_turbulence_short_window`, while the exact isothermal and turbulence operator rungs are cache-backed too, so exact native rechecks on those cases no longer require a fresh Hermes launch. The recycling lane is widened beyond the single-species case too: `tokamak_recycling_rhs` launches live through the shared `process_count = 6` harness path, has committed baselines plus a committed snapshot cache, and already matches its direct-tokamak RHS baselines natively; `tokamak_recycling_one_step` remains the committed curated small-step transient rung (`timestep=1`) with optional-history cache support and a native `bdf` path that stays within a tight operational band without relying on dump target preservation; `tokamak_recycling_dthe_rhs` is now also committed and exact on the same direct tokamak lane, with a committed snapshot cache and a narrow local Hermes permission fix required to make the multispecies reference runnable; `tokamak_recycling_dthe_drifts_rhs` is now also committed and exact on the same direct tokamak lane, with a committed snapshot cache and deterministic `sound_speed` plus `solver:type=cvode` curation in the manifest, but the current native compare surface is intentionally limited to ion density/pressure/momentum plus `Pe`; `tokamak_recycling_dthe_one_step` is now also curated at `timestep=0.1` with a committed optional-history cache and a promoted mixed exact gate, so its non-negligible ion/electron fields stay inside a `5e-2` scaled band while the near-zero neutral channels stay inside a small absolute band on the first-output compare surface; the first richer direct-window check is now bounded too, with a live Hermes-backed `nout=2` probe on the same D/T/He surface staying inside a mixed operational band under the current ten-minute policy; `tokamak_recycling_dthe_drifts_one_step` is now also curated at `timestep=0.1` with a committed optional-history cache and a native one-step path that stays within a small operational band on the compact drift-enabled D/T/He recycling surface, again without claiming `phi`/`Vort` yet; `tokamak_recycling_dthene_rhs` is now also committed and exact on the same direct tokamak lane, with a committed snapshot cache plus automatic staging of the shared `json_database/` directory for Hermes `OpenADAS` runs; `tokamak_recycling_dthene_one_step` is now also curated at `timestep=0.1` with a committed optional-history cache and a native one-step path that stays within a tighter operational band again on the D/T/He/Ne surface.
- 3D electromagnetic capabilities;
- neutrals, reactions, and impurities;
- performance, packaging, validation, and documentation.
