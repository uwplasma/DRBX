# Validation Gallery

!!! note "Plan authority"
    This page is a validation-gallery/status appendix. The active execution
    plan is [Research-Grade Execution Plan](research_grade_execution_plan.md).
    If this page conflicts with that plan, follow the execution plan and update
    this page afterward.

This page collects the current public validation figures from the active
validation ladder. Each figure is generated from the same committed baseline
artifacts used by the regression harness, so the visuals and the automated
checks stay in sync.

The figure classes are chosen to match the main literature patterns used in
verification and edge/SOL validation papers: convergence curves and observed
orders in the style of Roy 2005 and the GBS parallel-gradient work, profile and
target comparisons in the style of TCV-X21 and SOLPS-ITER, and
differentiable-science summaries closer to JAX-Fluids and related
JAX-based solver papers.

## Figure Status

| Figure | Status | Meaning |
| --- | --- | --- |
| `Diffusion Short Window` | `native-validated` | Transport baseline is locked. |
| `Electrostatic Vorticity Short Window` | `native-validated` | Electrostatic benchmark is locked. |
| `Coupled Drift-Wave One Step` | `native-validated` | First coupled 2D transient is locked. |
| `Drift-Wave Short-Window Parity` | `native-validated` | Drift-wave benchmark history is locked. |
| `Drift-Wave Short-Window Benchmark` | `native-validated` | Benchmark analysis is locked. |
| `Blob2d Short-Window Parity` | `native-validated` | Blob benchmark history is locked. |
| `Diverted Tokamak Geometry Movie` | `benchmark-backed visualization` | Full-domain stitched tokamak turbulence movie with LCFS, wall, and divertor overlays. || `Neutral Mixed Short-Window Benchmark Target` | `reference-backed plus one-step native gate` | Short-window artifact remains a reference target; one-step and substep/hybrid diagnostics now localize native `NVh` parity. |
| `Alfven-Wave Short-Window Benchmark` | `native-scaffolded target` | Electromagnetic transient benchmark is staged and benchmark-validated on the current scaffold. || `Neutral Mixed Term-Balance Campaign` | `operator-localization audit` | Native `NVh` term decomposition localizes the one-step Hermès mismatch. |
| `Stellarator FCI Validation` | `native non-axisymmetric gate` | Full-metric, field-line-map, conservative-operator, sheath/recycling, neutral, vorticity, and reduced 3D SOL dynamics campaign. |
| `VMEC-Extender Edge Field Import` | `self-contained synthetic imported-field gate` | Physical-phi field-grid import, FCI map construction, and compact scalar SOL smoke coupling are locked on synthetic NetCDF fixtures. |
| `ESSOS Field-Line Import` | `external geometry import gate` | ESSOS-owned field evaluation, adaptive field-line tracing, Poincare extraction, and portable trajectory/field-sample artifacts for later FCI use. |
| `ESSOS Imported QA DRB Movie` | `movie-grade reduced transient` | Imported Landreman-Paul QA coil, VMEC-coordinate, and hybrid FCI maps feeding a fixed-layout DRB transient with sheath/recycling/neutrals where open endpoints are present. |
| `Autodiff Diffusion Sensitivity` | `differentiable validation` | `jax.grad` sensitivities agree with finite differences on a compact native diffusion objective. |
| `Autodiff Diffusion Uncertainty` | `differentiable validation` | First-order autodiff covariance propagation is compared with vectorized Monte Carlo. |
| `Autodiff Diffusion Inverse Design` | `differentiable validation` | Gradient-based parameter recovery closes a compact inverse-design loop. |
| `Strong Scaling Diffusion` | `supporting performance audit` | Fixed-work differentiable diffusion scaling checks CPU process groups, host-device CPU `pmap`, and optional GPU `pmap`. |

## Diffusion Short Window

![Diffusion short-window parity](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__diffusion_short_window_parity.png)

What this locks down:

- structured mesh reconstruction;
- metric normalization on the transport path;
- Neumann guard handling;
- repeated output scheduling over a short transient.

## Restartable Diffusion Demo

![Restartable diffusion density snapshots](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__restartable_diffusion_demo_artifacts__images__restartable_diffusion_density_snapshots.png)

![Restartable diffusion density surface](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__restartable_diffusion_demo_artifacts__images__restartable_diffusion_density_surface.png)

![Restartable diffusion restart consistency](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__restartable_diffusion_demo_artifacts__images__restartable_diffusion_restart_consistency.png)

What this documents:

- the restartable native TOML workflow used by the public quick-start path;
- density snapshots and a surface view from the same compact diffusion run;
- restart-versus-continuous consistency as a user-facing artifact rather than
  only a JSON assertion.

## Neutral Mixed Term-Balance Campaign

![Neutral mixed term-balance audit](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_term_balance_campaign_artifacts__images__neutral_mixed_term_balance_campaign.png)

What this locks down:

- named native `NVh` term decomposition for the neutral mixed momentum equation;
- backward-Euler residual-rate reconstruction on both native and Hermès-3 final states;
- a direct diagnostic separating native one-step residual closure from the remaining Hermès final-state mismatch;
- a Hermès-free substep/hybrid-state diagnostic that ranks target-band `Nh`,
  `Ph`, and `NVh` sequencing errors before changing production neutral
  boundary or history updates;
- optional ingestion of direct Hermès diagnostic NetCDF fields from a
  one-step `output_ddt=true`, `diagnose=true` rerun, including `ddt(NVh)` and
  neutral momentum-flow diagnostics;
- direct ingestion of the patched Hermès `SNVh_pressure_gradient` diagnostic,
  plus the matched postprocessed `-Grad_par(Pn)` reconstruction on the Hermès
  final state for normalized operator comparison;
- a publication-grade lineout/bar figure for the current neutral mixed offender.

## Electrostatic Vorticity Short Window

![Vorticity short-window parity](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__vorticity_short_window_parity.png)

What this locks down:

- discrete X-Z XPPM advection;
- Boussinesq potential inversion;
- repeated electrostatic output parity for both `Vort` and `phi`.

## Coupled Drift-Wave One Step

![Drift-wave one-step parity](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__drift_wave_one_step_parity.png)

What this locks down:

- coupled density, electron momentum, vorticity, and potential output;
- quasineutral electron closure;
- fixed-temperature electron pressure;
- trimmed active-cell comparisons for the first 2D density-vorticity benchmark.

## Drift-Wave Short-Window Parity

![Drift-wave short-window parity](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__drift_wave_short_window_parity.png)

What this locks down:

- the full 50-output reduced drift-wave transient on the committed benchmark grid;
- benchmark-level growth and frequency agreement on the same stored history used by the regression harness;
- field-error history for `Ni`, `Ne`, `NVe`, `Vort`, and `phi`, published from the same native/reference comparison artifact used in docs and review material;
- the current documented native/reference envelope: max `|Ni-Ne|` error about `1.47e-3`, max `|NVe|` error about `1.70e-4`, max `|Vort|` error about `2.14e-2`, and max `|phi|` error about `4.31e-4`.

## Drift-Wave Short-Window Benchmark

![Drift-wave short-window diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__drift_wave_short_window_diagnostics.png)

What this locks down:

- benchmark postprocessing on the committed short-window array baseline;
- measured growth-rate and frequency extraction from the periodic density history;
- analytic finite-electron-mass dispersion evaluation from the same normalization and geometry scalars used by the run;
- documentation-ready reviewer figures backed by automated regression tests.

## Blob2d Short-Window Parity

![Blob2d short-window parity](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__blob2d_short_window_parity.png)

What this locks down:

- the full 50-output sheath-connected blob transient on the recalc-metric benchmark geometry;
- the optimized X-Z ExB transport kernel that made the native long-enough blob run practical without changing the discrete flux formulas;
- benchmark-level parity on summary blob diagnostics rather than only pointwise field maxima: peak density excess plus radial and binormal center-of-mass trajectories;
- the current documented native/reference envelope: peak-excess max error about `1.41e-2`, radial COM max error about `6.29e-1` active cells, and binormal COM max error about `7.32e-1` active cells.

## Diverted Tokamak Geometry Movie

![Diverted tokamak poster](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__images__diverted_tokamak_turbulence_poster.png)

![Diverted tokamak snapshots](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__images__diverted_tokamak_turbulence_snapshots.png)

![Diverted tokamak movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__movies__diverted_tokamak_turbulence.gif)

What this locks down:

- a stitched full-domain visualization path for the exact `tokamak_turbulence_short_window` benchmark lane;
- direct use of the tokamak mesh geometry via `Rxy`, `Zxy`, and `psixy`;
- LCFS overlay from `psixy = 0`, plus explicit wall and divertor target curves;
- a summary 2D diverted tokamak figure package generated by [examples/diverted_tokamak_movie_demo.py](../examples/diverted_tokamak_movie_demo.py).

## Stellarator FCI Validation

![Stellarator FCI geometry validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__geometry__images__stellarator_fci_geometry_campaign.png)

![Stellarator FCI multi-configuration suite](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__suite__images__stellarator_fci_suite_campaign.png)

![Stellarator FCI operator validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__operators__images__stellarator_fci_operator_campaign.png)

![Stellarator full metric MMS validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__metric_mms__images__stellarator_metric_mms_campaign.png)

![Stellarator sheath/recycling validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__sheath_recycling__images__stellarator_sheath_recycling_campaign.png)

![Stellarator neutral physics validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__neutral_physics__images__stellarator_neutral_physics_campaign.png)

![Stellarator vorticity validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__vorticity__images__stellarator_vorticity_campaign.png)

![Stellarator PyTree/JVP/scaling validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__pytree_drb__images__stellarator_drb_pytree_campaign.png)

![ESSOS Landreman-Paul QA field-line import](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_fieldline_import_artifacts__images__essos_landreman_paul_qa_fieldline_import.png)

![ESSOS field-line/VMEC surface registration](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_vmec_fieldline_surface_artifacts__images__essos_vmec_fieldline_surface_campaign.png)

![ESSOS imported FCI validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_artifacts__images__essos_imported_fci_campaign.png)

![ESSOS imported FCI VMEC-coordinate validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_vmec_artifacts__images__essos_imported_fci_vmec_campaign.png)

![ESSOS imported FCI hybrid validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_hybrid_artifacts__images__essos_imported_fci_hybrid_campaign.png)

![ESSOS imported PyTree/JVP validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_pytree_artifacts__images__essos_imported_pytree_campaign.png)

![ESSOS imported QA-coil DRB diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__images__essos_imported_drb_movie_campaign_diagnostics.png)

![ESSOS imported QA-coil DRB movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__movies__essos_imported_drb_movie_campaign.gif)

![ESSOS imported QA-hybrid DRB diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_hybrid_artifacts__images__essos_imported_drb_movie_hybrid_campaign_diagnostics.png)

![ESSOS imported QA-hybrid DRB movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_hybrid_artifacts__movies__essos_imported_drb_movie_hybrid_campaign.gif)

![Stellarator SOL snapshots](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__showcase__images__stellarator_sol_showcase_snapshots.png)

![Stellarator SOL diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__showcase__images__stellarator_sol_showcase_diagnostics.png)

![Stellarator SOL 3D movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__showcase__movies__stellarator_sol_showcase.gif)

What this documents:

- the first native non-axisymmetric field-line-map geometry lane;
- full covariant/contravariant metric checks with inverse residual about `1.44e-14`;
- three analytic 3D non-axisymmetric geometry variants passing the same metric/map gate;
- full \(J^{-1}\partial_i(JK g^{ij}\partial_j f)\) manufactured-solution
  convergence with observed order about `1.90`;
- interpolation and traced parallel-gradient convergence with observed orders about `1.96` and `1.54`;
- monotone parallel-diffusion energy decay on both compact and
  metric-weighted conservative operator probes;
- non-axisymmetric traced-endpoint sheath/recycling balance with particle
  recycling and zero-current residuals closed to roundoff;
- neutral ionisation/recombination/charge-exchange reaction balances with
  particle and momentum residuals closed to roundoff;
- a fixed-layout PyTree RHS gate where the combined 3D state is compiled,
  differentiated with JVP, checked against finite differences, matched under
  `vmap`, and profiled for local CPU and multi-device GPU execution;
- a movie-grade ESSOS-imported QA transient where the same fixed-layout DRB
  state is advanced on coil, VMEC-coordinate, or hybrid maps, with
  sheath/recycling/neutrals active where imported endpoint masks are present;
- an independent field-line/VMEC registration diagnostic that overlays
  long-trace coil-field Poincare points on the scaled Landreman-Paul QA VMEC
  surfaces and reports the strict closed-surface match flag separately from
  the finite-trace diagnostic pass;
- ESSOS-owned field evaluation, adaptive field-line tracing, and Poincare
  extraction exported into portable `jax_drb` JSON/NPZ artifacts without
  maintaining a duplicate coil-field or field-line tracer in this repository;
- metric-weighted vorticity inversion with relative potential error about `1.30e-3`;
- a reduced 3D SOL dynamics benchmark with R-Z panel snapshots at four toroidal angles;
- RMS, skewness, radial-flux proxy, time-trace, and toroidal-poloidal spectrum diagnostics;
- a README-ready opened traced-surface movie with radial cuts, field-line overlays, colorbar, and time annotation.

## VMEC-Extender Edge Field Import

This gate keeps the imported-field interface runnable without an external
equilibrium checkout. The example builds small synthetic NetCDF fixtures with
the same `(R, phi, Z)` field-grid contract expected from VMEC-extender output,
then exercises physical-toroidal interpolation, field-period wrapping,
field-line RHS evaluation, FCI map construction, and a compact scalar SOL smoke
case with field-aligned diffusion and open R-Z losses.

To regenerate the synthetic validation plots locally:

```bash
PYTHONPATH=src MPLBACKEND=Agg \
  python examples/geometry-3D/vmec-extender/imported_field_demo.py
```

The detailed interface contract, diagnostics, and plot paths are documented in
[VMEC-Extender Edge Fields](vmec_extender_edge_fields.md). Release-hosted images
will be added to this gallery once the next validation-artifact bundle is
published.

## Fluid 1D MMS Convergence

![Fluid 1D MMS convergence](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__fluid_1d_mms_convergence_artifacts__images__fluid_1d_mms_convergence.png)

What this documents:

- an explicit manufactured-solution refinement bundle for the promoted 1D fluid density, pressure, and momentum operators;
- per-resolution L2 errors and observed orders on the same native lane used for the compact verification tests;
- a literature-anchored verification figure that can be reused in docs and future paper surfaces instead of leaving the result trapped in a script output.

## Autodiff Diffusion Uncertainty

![Autodiff diffusion uncertainty](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_uncertainty_artifacts__images__autodiff_diffusion_uncertainty.png)

What this documents:

- a standard uncertainty-propagation example on the same compact native differentiable diffusion lane used for sensitivity and inverse design;
- a scalar QoI based on the final active-domain density variance plus a field QoI based on the final radial profile;
- agreement between first-order autodiff covariance pushforward and a vectorized Monte Carlo estimate on the same native solve path.

This is currently the strongest differentiable-science figure in the public
surface because it presents a standard uncertainty-propagation comparison rather
than only raw gradients.

## Autodiff Diffusion Sensitivity

![Autodiff diffusion sensitivity](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_sensitivity_artifacts__images__autodiff_diffusion_sensitivity.png)

What this documents:

- a compact native objective differentiated directly with `jax.grad`;
- centered finite-difference checks for all promoted design parameters;
- a local sweep that makes the leading tangent direction visible rather than
  treating the gradient as a black-box number;
- the first differentiable-science panel in the standard sensitivity, UQ, and
  inverse-design progression.

## Autodiff Diffusion Inverse Design

![Autodiff diffusion inverse design](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_inverse_design_artifacts__images__autodiff_diffusion_inverse_design.png)

What this documents:

- a gradient-based parameter recovery loop on the same compact native diffusion
  lane used by the sensitivity and UQ examples;
- objective reduction and final-profile agreement against a known target;
- a publication-ready demonstration that the differentiable path supports
  optimization, not only derivative inspection.

## Strong Scaling Diffusion

![Strong scaling diffusion](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__strong_scaling_diffusion_artifacts__images__strong_scaling_diffusion.png)

What this documents:

- fixed-work scaling on a differentiable native objective, with the total batch
  held fixed as device or worker count changes;
- local CPU process-group and host-device `pmap` modes on the same workload;
- the optional remote-GPU execution contract used by the docs and profiling
  plan.

## Neutral Mixed Short-Window Benchmark Target

![Neutral mixed short-window diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__neutral_mixed_short_window_diagnostics.png)

What this locks down:

- a compact reference-side transient target for the staged neutral branch before the native stiff solver is promoted through the public runner;
- center-probe histories for `Nh`, `Ph`, and `NVh` at the committed benchmark location `(x=5, y=3, z=5)`;
- the derived center temperature `Ph / Nh`, which stays close to the expected `0.1` throughout the short window;
- summary compact metrics rather than large raw arrays: final total `Nh` about `7.86197875e+02`, final total `Ph` about `7.86184063e+01`, and final momentum RMS about `5.56121767e-08`.

## Alfven-Wave Short-Window Benchmark

![Alfven-wave short-window diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__alfven_wave_short_window_diagnostics.png)

What this locks down:

- the first multi-output electromagnetic transient rung on the finite-electron-mass slab benchmark;
- benchmark-quality phase-speed extraction from the committed `nout=20` history;
- analytic-vs-measured validation on the same stored arrays used by the regression harness;
- the current documented benchmark numbers:
  - analytic phase speed about `9.48585409e+05 m/s`;
  - measured phase speed about `9.42218662e+05 m/s`;
  - relative phase-speed error about `6.71e-03`;
- native/reference parity on that same short-window history, published from the exact committed comparison artifact.

## Regeneration

These figures are generated from the committed baseline arrays plus native case runs. The current gallery uses:

- `diffusion_short_window`
- `vorticity_short_window`
- `drift_wave_one_step`
- `drift_wave_short_window`
- `blob2d_short_window`
- `blob2d_one_step` saved-payload visualization
- `alfven_wave_short_window`
- `tokamak_turbulence_short_window` stitched full-domain geometry visualization
- live Hermès reruns across the representative 1D and 2D curated matrix
- atomic-rate differentiability audit for AMJUEL, OpenADAS, and hydrogen
  charge exchange
- tokamak recycling target/neutral observable profiles on the direct D/T/He
  validation lane

The next gallery pass should add:

- periodic 1D fluid short-window figures;
- native neutral transient parity figures once the stiff `neutral_mixed` path is benchmark-clean;
- benchmark and validation plots for additional integrated 2D production and broader EM cases as those stages land.
