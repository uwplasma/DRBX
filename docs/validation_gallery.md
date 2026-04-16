# Validation Gallery

This page collects the first curated parity figures from the active validation ladder. Each figure is generated from the same committed baseline artifacts used by the regression harness, so the visuals and the automated checks stay in sync.

## Figure Status

| Figure | Status | Meaning |
| --- | --- | --- |
| `Diffusion Short Window` | `native-validated` | Transport baseline is locked. |
| `Electrostatic Vorticity Short Window` | `native-validated` | Electrostatic benchmark is locked. |
| `Coupled Drift-Wave One Step` | `native-validated` | First coupled 2D transient is locked. |
| `Drift-Wave Short-Window Parity` | `native-validated` | Drift-wave benchmark history is locked. |
| `Drift-Wave Short-Window Benchmark` | `native-validated` | Benchmark analysis is locked. |
| `Blob2d Short-Window Parity` | `native-validated` | Blob benchmark history is locked. |
| `Blob2d Meeting Movie` | `saved-payload visualization` | Fast one-step Blob2D visualization from a committed `.npz` payload. |
| `Diverted Tokamak Geometry Movie` | `benchmark-backed visualization` | Full-domain stitched tokamak turbulence movie with LCFS, wall, and divertor overlays. |
| `TCV-X21 Tokamak Scaffold` | `scaffolded_reference_backed` | First 3D tokamak kickoff package with a manifest-resolved preview path. |
| `Neutral Mixed Short-Window Benchmark Target` | `reference-only target` | Review artifact is staged; native transient is not yet promoted. |
| `Alfven-Wave Short-Window Benchmark` | `native-scaffolded target` | Electromagnetic transient benchmark is staged and benchmark-validated on the current scaffold. |

## Diffusion Short Window

![Diffusion short-window parity](images/diffusion_short_window_parity.png)

What this locks down:

- structured mesh reconstruction;
- metric normalization on the transport path;
- Neumann guard handling;
- repeated output scheduling over a short transient.

## Electrostatic Vorticity Short Window

![Vorticity short-window parity](images/vorticity_short_window_parity.png)

What this locks down:

- discrete X-Z XPPM advection;
- Boussinesq potential inversion;
- repeated electrostatic output parity for both `Vort` and `phi`.

## Coupled Drift-Wave One Step

![Drift-wave one-step parity](images/drift_wave_one_step_parity.png)

What this locks down:

- coupled density, electron momentum, vorticity, and potential output;
- quasineutral electron closure;
- fixed-temperature electron pressure;
- trimmed active-cell comparisons for the first 2D density-vorticity benchmark.

## Drift-Wave Short-Window Parity

![Drift-wave short-window parity](images/drift_wave_short_window_parity.png)

What this locks down:

- the full 50-output reduced drift-wave transient on the committed benchmark grid;
- benchmark-level growth and frequency agreement on the same stored history used by the regression harness;
- field-error history for `Ni`, `Ne`, `NVe`, `Vort`, and `phi`, published from the same native/reference comparison artifact used in docs and review material;
- the current documented native/reference envelope: max `|Ni-Ne|` error about `1.47e-3`, max `|NVe|` error about `1.70e-4`, max `|Vort|` error about `2.14e-2`, and max `|phi|` error about `4.31e-4`.

## Drift-Wave Short-Window Benchmark

![Drift-wave short-window diagnostics](images/drift_wave_short_window_diagnostics.png)

What this locks down:

- benchmark postprocessing on the committed short-window array baseline;
- measured growth-rate and frequency extraction from the periodic density history;
- analytic finite-electron-mass dispersion evaluation from the same normalization and geometry scalars used by the run;
- documentation-ready reviewer figures backed by automated regression tests.

## Blob2d Short-Window Parity

![Blob2d short-window parity](images/blob2d_short_window_parity.png)

What this locks down:

- the full 50-output sheath-connected blob transient on the recalc-metric benchmark geometry;
- the optimized X-Z ExB transport kernel that made the native long-enough blob run practical without changing the discrete flux formulas;
- benchmark-level parity on reviewer-facing blob diagnostics rather than only pointwise field maxima: peak density excess plus radial and binormal center-of-mass trajectories;
- the current documented native/reference envelope: peak-excess max error about `1.41e-2`, radial COM max error about `6.29e-1` active cells, and binormal COM max error about `7.32e-1` active cells.

## Blob2d Meeting Movie

![Blob2D meeting snapshots](images/blob2d_meeting_snapshots.png)

![Blob2D meeting poster](images/blob2d_meeting_movie_poster.png)

What this locks down:

- a fast saved-result visualization workflow using [examples/blob2d_meeting_demo.py](examples/blob2d_meeting_demo.py);
- a real 2D movie artifact from [blob2d_one_step.npz](references/baselines/reference_arrays/blob2d_one_step.npz), written to [docs/movies/blob2d_meeting_2d.mp4](docs/movies/blob2d_meeting_2d.mp4);
- a matching 3D surface movie and poster for presentation use;
- an explicit `--skip-parity` mode for saved payloads whose output timeline does not match the short-window parity metrics.

## Diverted Tokamak Geometry Movie

![Diverted tokamak poster](data/diverted_tokamak_turbulence_artifacts/images/diverted_tokamak_turbulence_poster.png)

![Diverted tokamak movie](data/diverted_tokamak_turbulence_artifacts/movies/diverted_tokamak_turbulence.gif)

What this locks down:

- a stitched full-domain visualization path for the exact `tokamak_turbulence_short_window` benchmark lane;
- direct use of the tokamak mesh geometry via `Rxy`, `Zxy`, and `psixy`;
- LCFS overlay from `psixy = 0`, plus explicit wall and divertor target curves;
- a reviewer-facing 2D diverted tokamak figure package generated by [examples/diverted_tokamak_movie_demo.py](../examples/diverted_tokamak_movie_demo.py).

## TCV-X21 Tokamak Scaffold

![TCV-X21 scaffold poster](data/tokamak_tcv_x21_scaffold_artifacts/images/tokamak_tcv_x21_scaffold_poster.png)

![TCV-X21 scaffold movie](data/tokamak_tcv_x21_scaffold_artifacts/movies/tokamak_tcv_x21_scaffold.gif)

![TCV-X21 scaffold profiles](data/tokamak_tcv_x21_scaffold_artifacts/images/tokamak_tcv_x21_scaffold_profiles.png)

What this documents:

- the first 3D tokamak kickoff package in the tree;
- manifest resolution for `tokamak_tcv_x21_escalation`;
- a structured deck/input report plus a benchmark-data report alongside the public figure bundle;
- an explicit benchmark validation contract covering FHRP, HFS-LP, and LFS-LP observable families;
- a shared observable report that describes those profile families on the generic 3D adapter schema;
- a staged profile report and compact NPZ bundle for those observable families, now populated from the public benchmark observable record;
- a publication-style profile summary plot derived from the same staged bundle;
- a public benchmark-data mode that can be regenerated without a heavy local 3D solve;
- LCFS, wall, and divertor overlays in the same diverted-geometry style as the existing tokamak visualizations, now driven by the public sample geometry/snapshot files.
- a reviewer-friendly bridge between the case manifest and the future selected 3D execution lane.
- the reduced selected-field parity package now exists as the next explicit gate after the scaffold bundle, on compact `Ne`/`Pe`/`phi` histories from either two 3D workdirs or the public TCV-X21 benchmark-data root plus a reproducible derived candidate.

## Native Tokamak Selected-Field Rung

What this documents:

- the first reduced native 3D execution rung in the tree;
- a promoted native tokamak one-step case on the compact `Ne`/`Pe`/`phi` surface;
- parity JSON/NPZ artifacts on the same compact surface used by the benchmark-backed gate;
- a direct native-vs-reference selected-field history comparison bundle on the same committed artifact path;
- a shared observable report plus a runtime/provenance report for the native run.

## Traced-Field-Line Geometry Scaffold

![Traced-field-line scaffold metrics](data/traced_field_line_scaffold_artifacts/images/traced_field_line_scaffold_metrics.png)

![Traced-field-line scaffold lineouts](data/traced_field_line_scaffold_artifacts/images/traced_field_line_scaffold_lineouts.png)

![Traced-field-line scaffold slice summary](data/traced_field_line_scaffold_artifacts/images/traced_field_line_scaffold_slice_summary.png)

![Traced-field-line scaffold slice movie](data/traced_field_line_scaffold_artifacts/images/traced_field_line_scaffold_slice_movie.gif)

What this documents:

- the second 3D geometry-adapter scaffold in the tree;
- a non-diverted, traced-field-line geometry family riding on the same general 3D infrastructure;
- a reusable metric summary JSON and compact NPZ metric bundle;
- a shared observable report covering the line-diagnostic and selected-plane families on the same adapter schema;
- reusable radial, toroidal, and poloidal line diagnostics on the same artifact model;
- reusable selected-plane summaries and a first geometry-family GIF on the same artifact model;
- a publication-style metric summary plot that can be regenerated from either a synthetic preview or an explicit mesh/metric JSON specification;
- an explicit geometry-adapter validation contract before any native non-tokamak 3D claim is made.

## Traced-Field-Line Selected-Field Parity

![Traced-field-line selected-field parity](data/traced_field_line_selected_field_artifacts/images/traced_field_line_selected_field_parity.png)

What this documents:

- the first reduced parity gate on the second 3D geometry family;
- a compact selected-field compare surface on traced-field-line metric fields;
- shared observable-report publication on the same geometry-adapter schema used by the scaffold packages;
- a non-tokamak counterpart to the tokamak selected-field parity lane;
- an explicit external-pair source report when the public external FCI sample is available locally.

## Traced-Field-Line Native Reduced Selected-Field

![Traced-field-line native selected-field parity](data/traced_field_line_native_selected_field_artifacts/images/traced_field_line_native_selected_field.png)

![Traced-field-line native selected-field comparison](data/traced_field_line_native_selected_field_artifacts/images/traced_field_line_native_selected_field_comparison.png)

What this documents:

- the first native reduced rung on a non-tokamak 3D geometry family;
- a JAX-native radial-profile reduction on explicit traced-field-line metric pairs;
- the same public parity, comparison, observable, and runtime artifact surfaces used by the native tokamak reduced rung;
- an honest bridge between external-pair validation-only geometry adapters and future broader native non-tokamak execution work.

## Stellarator VMEC Scaffold

![Stellarator VMEC profiles](data/stellarator_vmec_scaffold_artifacts/images/stellarator_vmec_scaffold_profiles.png)

![Stellarator VMEC surface summary](data/stellarator_vmec_scaffold_artifacts/images/stellarator_vmec_scaffold_surface_summary.png)

![Stellarator VMEC surface movie](data/stellarator_vmec_scaffold_artifacts/images/stellarator_vmec_scaffold_surface_movie.gif)

What this documents:

- the third 3D geometry adapter in the tree;
- a VMEC-style stellarator equilibrium bundle on the same manifest and observable schema as the tokamak and traced-field-line adapters;
- reusable profile diagnostics for `iota`, `pressure`, and `toroidal_flux`;
- sampled `R`/`Z` flux-surface cross-sections across toroidal angle, with a publication-style summary figure and a compact movie path;
- a pressure test for the general 3D artifact model on a geometry family that is neither diverted tokamak nor FCI-grid-only.

## Stellarator VMEC Selected-Field Parity

![Stellarator VMEC selected-field parity](data/stellarator_vmec_selected_field_artifacts/images/stellarator_vmec_selected_field_parity.png)

What this documents:

- the first reduced selected-field parity gate on the VMEC-style stellarator family;
- a compact compare surface on `iota`, `pressure`, and `toroidal_flux`;
- the same shared observable-report and source-report contract used by the other 3D parity packages;
- a real explicit external VMEC-pair regeneration path in addition to the synthetic preview mode.

## Publication-Ready 3D Campaign

![Publication-ready 3D campaign](data/publication_ready_3d_artifacts/images/publication_ready_3d_campaign.png)

What this documents:

- one reviewer-facing summary package assembled from the already committed 3D reduced-lane artifacts;
- the promoted native tokamak one-step and short-window reduced rungs on the same figure as the traced-field-line native reduced rung and the external-pair gates;
- the current manufactured-solution convergence floor from the committed `fluid_1d_mms_convergence` report;
- the remaining blockers before a full publication-ready 3D claim, stated explicitly rather than implied.

## Reactions And Collisions Campaign

![Reactions and collisions campaign](data/reactions_collisions_campaign_artifacts/images/reactions_collisions_campaign.png)

What this documents:

- a dedicated engineering and physics gate for reactions, collisions, and atomic-data breadth;
- explicit native checks for charge exchange, isotope coupling, CX multipliers, ionisation, ion-viscosity collisionality closure, and neon OpenADAS loading;
- a machine-readable JSON/NPZ/plot package instead of leaving this breadth implied only by low-level unit tests.

## Neutral Mixed Short-Window Benchmark Target

![Neutral mixed short-window diagnostics](images/neutral_mixed_short_window_diagnostics.png)

What this locks down:

- a compact reference-side transient target for the staged neutral branch before the native stiff solver is promoted through the public runner;
- center-probe histories for `Nh`, `Ph`, and `NVh` at the committed benchmark location `(x=5, y=3, z=5)`;
- the derived center temperature `Ph / Nh`, which stays close to the expected `0.1` throughout the short window;
- reviewer-facing compact metrics rather than large raw arrays: final total `Nh` about `7.86197875e+02`, final total `Ph` about `7.86184063e+01`, and final momentum RMS about `5.56121767e-08`.

## Alfven-Wave Short-Window Benchmark

![Alfven-wave short-window diagnostics](images/alfven_wave_short_window_diagnostics.png)

What this locks down:

- the first multi-output electromagnetic transient rung on the finite-electron-mass slab benchmark;
- benchmark-quality phase-speed extraction from the committed `nout=20` history;
- analytic-vs-measured validation on the same stored arrays used by the regression harness;
- the current documented benchmark numbers:
  - analytic phase speed about `9.48585409e+05 m/s`;
  - measured phase speed about `9.42218662e+05 m/s`;
  - relative phase-speed error about `6.71e-03`;
- native/reference parity on that same short-window history, published from the exact committed comparison artifact.

The same short-window rung now also drives the meeting-ready visual package in [docs/alfven_wave_meeting_demo.md](docs/alfven_wave_meeting_demo.md), which adds 2D and 3D movies plus a snapshot panel from a live native run.

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

The next gallery pass should add:

- periodic 1D fluid short-window figures;
- native neutral transient parity figures once the stiff `neutral_mixed` path is benchmark-clean;
- benchmark and validation plots for additional integrated 2D production and broader EM cases as those stages land.
