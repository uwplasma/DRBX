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
| `Diverted Tokamak Geometry Movie` | `benchmark-backed visualization` | Full-domain stitched tokamak turbulence movie with LCFS, wall, and divertor overlays. |
| `Stellarator FCI Validation` | `native non-axisymmetric gate` | Full-metric, field-line-map, conservative-operator, sheath/recycling, neutral, vorticity, and reduced 3D SOL dynamics campaign. |
| `Rotating-Ellipse FCI` | `genuinely non-axisymmetric gate` | Rotating-ellipse (`l = 2`) metric by autodiff; direct and traced-field-line parallel gradient converge at order 2; shape-differentiable. |
| `VMEC-Extender Edge Field Import` | `self-contained synthetic imported-field gate` | Physical-phi field-grid import, FCI map construction, and compact scalar SOL smoke coupling are locked on synthetic NetCDF fixtures. |
| `ESSOS Field-Line Import` | `external geometry import gate` | ESSOS-owned field evaluation, adaptive field-line tracing, Poincare extraction, and portable trajectory/field-sample artifacts for later FCI use. |
| `ESSOS Imported QA DRB Movie` | `movie-grade reduced transient` | Imported Landreman-Paul QA coil, VMEC-coordinate, and hybrid FCI maps feeding a fixed-layout DRB transient with sheath/recycling/neutrals where open endpoints are present. |
| `Autodiff Diffusion Sensitivity` | `differentiable validation` | `jax.grad` sensitivities agree with finite differences on a compact native diffusion objective. |
| `Autodiff Diffusion Uncertainty` | `differentiable validation` | First-order autodiff covariance propagation is compared with vectorized Monte Carlo. |
| `Autodiff Diffusion Inverse Design` | `differentiable validation` | Gradient-based parameter recovery closes a compact inverse-design loop. |
| `Strong Scaling Diffusion` | `supporting performance audit` | Fixed-work differentiable diffusion scaling checks CPU process groups, host-device CPU `pmap`, and optional GPU `pmap`. |

## Diffusion Short Window

![Diffusion short-window validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__diffusion_short_window_parity.png)

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

## Electrostatic Vorticity Short Window

![Vorticity short-window validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__vorticity_short_window_parity.png)

What this locks down:

- discrete X-Z XPPM advection;
- Boussinesq potential inversion;
- repeated electrostatic output agreement for both `Vort` and `phi`.

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

## Rotating-Ellipse FCI

![Rotating-ellipse FCI](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/rotating_ellipse_fci.png)

The classical rotating-ellipse (`l = 2`) stellarator — a torus whose elliptical
cross-section rotates with the toroidal angle — is the canonical minimal
non-axisymmetric field, with a metric that depends on all three logical
coordinates. Its metric is built by automatic differentiation of the analytic
embedding (exact, and differentiable with respect to the shape), and the FCI
parallel gradient converges at second order on it for both the direct
`b^i d_i f` operator and the traced-field-line operator that follows field lines
between toroidal planes. The gate lives in
[`tests/test_rotating_ellipse_fci.py`](../tests/test_rotating_ellipse_fci.py);
the full description is in [Rotating-Ellipse FCI](rotating_ellipse_fci.md).

To regenerate the figure:

```bash
PYTHONPATH=src python examples/stellarator/rotating_ellipse_fci_demo.py
```

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

## Regeneration

These figures are generated from the committed baseline arrays plus native case runs. The current gallery uses:

- `diffusion_short_window`
- `vorticity_short_window`
- `tokamak_turbulence_short_window` stitched full-domain geometry visualization
- the stellarator FCI, ESSOS/VMEC imported-geometry, and autodiff-diffusion
  validation campaigns

The next gallery pass should add:

- periodic 1D fluid short-window figures;
- release-hosted VMEC-extender import figures;
- benchmark and validation plots for additional integrated 2D and broader FCI
  cases as those stages land.
