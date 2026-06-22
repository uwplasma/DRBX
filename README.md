# jax_drb

[![Tests](https://github.com/uwplasma/jax_drb/actions/workflows/test.yml/badge.svg)](https://github.com/uwplasma/jax_drb/actions/workflows/test.yml)
[![Docs](https://github.com/uwplasma/jax_drb/actions/workflows/docs.yml/badge.svg)](https://github.com/uwplasma/jax_drb/actions/workflows/docs.yml)
[![Closeout Coverage](https://github.com/uwplasma/jax_drb/actions/workflows/coverage.yml/badge.svg)](https://github.com/uwplasma/jax_drb/actions/workflows/coverage.yml)
[![Research Campaigns](https://github.com/uwplasma/jax_drb/actions/workflows/research-campaigns.yml/badge.svg)](https://github.com/uwplasma/jax_drb/actions/workflows/research-campaigns.yml)
[![PyPI publish](https://github.com/uwplasma/jax_drb/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/uwplasma/jax_drb/actions/workflows/publish-pypi.yml)
[![PyPI](https://img.shields.io/pypi/v/jax-drb.svg)](https://pypi.org/project/jax-drb/)
[![Python](https://img.shields.io/pypi/pyversions/jax-drb.svg)](https://pypi.org/project/jax-drb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![JAX](https://img.shields.io/badge/JAX-enabled-0a9396.svg)](https://jax.readthedocs.io/)
[![Read the Docs](https://readthedocs.org/projects/jax-drb/badge/?version=latest)](https://jax-drb.readthedocs.io/)

`jax_drb` is a JAX-first edge and scrape-off-layer plasma code for drift-reduced Braginskii models, electrostatic turbulence, neutral transport, curated tokamak workflows, and reusable 3D geometry diagnostics.

Documentation is available at [jax-drb.readthedocs.io](https://jax-drb.readthedocs.io/).

The codebase is organized around:

- a standalone CLI and Python API,
- restartable native runs with structured terminal progress,
- portable analysis and visualization artifacts,
- explicit capability tiers for curated benchmark lanes,
- reusable 3D geometry, movie, and selected-field comparison tools,
- gridded VMEC-extender edge-field import with physical-phi interpolation, FCI map construction, and a compact SOL verification gate,
- differentiable driver paths for sensitivity analysis, uncertainty propagation, and inverse design.

The stable release boundary is explicit: compact native solvers, selected
operator gates, fixed-layout residual seams, and differentiable examples are
promoted where their tests and artifacts say so; full output-window recycling
still defaults to the validated compatibility BDF path, with JAX-linearized and
JVP variants kept as opt-in research gates until same-fidelity parity and
runtime evidence are strong enough to promote them.

![Diverted tokamak dynamics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__movies__diverted_tokamak_turbulence.gif)

![3D tokamak toroidal dynamics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_tcv_x21_toroidal_movie_artifacts__movies__tokamak_tcv_x21_toroidal.gif)

![3D imported QA stellarator dynamics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__movies__movie_compact.gif)

## Install

PyPI:

```bash
pip install jax-drb
```

From source:

```bash
git clone https://github.com/uwplasma/jax_drb
cd jax_drb
pip install -e .
```

The default installation already pulls in the runtime and analysis dependencies used by the main CLI, geometry tooling, and diagnostics, including `jax`, `diffrax`, `scipy`, `equinox`, `matplotlib`, and `netCDF4`.

## Example Artifacts And Movies

The git checkout is intentionally lightweight. Large generated arrays, figures,
and GIFs used by the README, docs, tests, tokamak examples, and stellarator
examples live in a GitHub release instead of git history.

For this private repository, authenticate first and then restore the artifacts:

```bash
gh auth login --hostname github.com
python scripts/fetch_example_artifacts.py
```

If you do not use the GitHub CLI, set `GH_TOKEN` or `GITHUB_TOKEN` to a token
with access to `uwplasma/jax_drb`, then run the same fetch command. The command
restores release-backed docs media under `docs/data/` and heavy validation
baselines under `references/baselines/`.

Set `JAX_DRB_ARTIFACT_CACHE_DIR=/path/to/cache` to share downloaded release
archives across clean checkouts. The older `JAX_DRB_ARTIFACT_CACHE` name is
also accepted.

After that, users can run the documented user-facing examples and inspect the
generated or restored PNG/GIF/NPZ outputs. Users do not need to install or
download any external plasma code to run those examples or the README/docs
movies. Live reference-code reruns are developer validation tasks only; the
user-facing examples use JAXDRB code plus release-backed artifacts.

## Quick Start

Run a TOML deck:

```bash
jax_drb path/to/input.toml
```

Inspect a deck without running it:

```bash
jax_drb inspect path/to/input.toml
```

Learn which model family, dimension, fluid closure, and boundary family to
choose before building a deck:

```bash
PYTHONPATH=src python examples/model_selection_guide.py
```

The guide is a lightweight dry run by default. It explains diffusion/reduced
transport, drift-reduced Braginskii open-field models, one-fluid versus
two-fluid choices, 1D/2D/3D tradeoffs, and diffusion/sheath/recycling/neutral
boundary families while writing parse-checked starter decks under
`output/model_selection_guide/`.

Resume from a restart bundle:

```bash
jax_drb run path/to/input.toml \
  --output-dir output/restarted_case \
  --restart-in output/base_case/my_case_restart.npz \
  --resume-steps 2
```

Use detailed runtime progress:

```bash
jax_drb run path/to/input.toml --verbose
```

## Python API

```python
from jax_drb.cli import main
from jax_drb.native import run_input_case

main(["run", "examples/inputs/restartable_diffusion.toml", "--quiet"])

driver_result = run_input_case(
    "examples/inputs/restartable_diffusion.toml",
    case_name="diffusion_driver",
    parity_mode="run",
    verbose=True,
)
print(driver_result.time_points[-1])
```

## Input Model

`jax_drb` uses structured TOML decks. Common top-level sections are:

- `[time]`
- `[runtime]`
- `[runtime.logging]`
- `[mesh]`
- `[solver]`
- `[model]`
- `[output]`
- `[restart]`
- `[species.<name>]`
- `[fields.<name>]`

Example:

```toml
[time]
nout = 2
timestep = 0.1

[runtime]
precision = "float64"

[runtime.logging]
verbosity = "detailed"
verbose = true
quiet = false

[mesh]
nx = 32
ny = 1
nz = 1
dx = 0.03125

[solver]
type = "native"

[model]
components = ["h"]

[species.h]
type = ["evolve_density", "evolve_pressure", "anomalous_diffusion"]

[fields.Nh]
function = { expr = "1 + 0.2 * exp(-((x-0.5)^2)/0.01)" }

[fields.Ph]
function = { expr = "0.1" }

[output]
directory = "output/my_case"
write_summary = true
write_arrays = true
write_restart = true
write_log = true
```

## Output Artifacts

Promoted native runs can write:

- summary JSON,
- arrays NPZ,
- restart NPZ,
- structured run-log JSON.

The run log records:

- capability tier,
- runtime precision and backend,
- mesh, solver, and time configuration,
- ordered runtime events,
- artifact locations,
- restart provenance,
- variable summaries.

Detailed terminal mode is designed to keep long runs from looking hung. The CLI reports:

- deck loading,
- restart loading,
- run launch and completion,
- transient interval progress on recycling lanes, including interval count,
  accepted timestep, simulated time, and estimated remaining wall time on the
  live native implicit paths,
- artifact writes.

## Capability Tiers

Curated validation cases are labeled explicitly:

- `native_exact`: fully native on its promoted compare surface and strong enough for the main public benchmark surface,
- `native_operational`: native and useful, but still carrying bounded residuals,
- `scaffolded_reference_backed`: useful for diagnostics or geometry staging, but not counted as native closure.

The current promoted matrix includes:

- exact compact 2D blob, drift-wave, and tokamak lanes,
- exact and operational recycling lanes with external-reference gates,
- native 3D reduced tokamak, traced-field-line, and stellarator selected-field bundles,
- non-axisymmetric FCI vorticity gates that compare Boussinesq and
  non-Boussinesq perpendicular polarization and verify their constant-\(n/B^2\)
  limit,
- control, reaction, impurity, neutral, and profiling campaign packages.

The detailed status surface lives in:

- [docs/implementation_inventory.md](docs/implementation_inventory.md)
- [docs/parity_harness.md](docs/parity_harness.md)
- [docs/parity_matrix.md](docs/parity_matrix.md)
- [docs/research_grade_validation_matrix.md](docs/research_grade_validation_matrix.md)
- [docs/validation_gallery.md](docs/validation_gallery.md)

## Validation Figures

The README shows only a few representative artifacts. The full figure index,
including operator convergence, reactions/collisions, neutrals, recycling,
reference parity, CPU scaling, and differentiability plots, lives in
[docs/validation_gallery.md](docs/validation_gallery.md). The validation matrix
that separates primary scientific evidence from supporting engineering gates is
[docs/research_grade_validation_matrix.md](docs/research_grade_validation_matrix.md).

The neutral-mixed validation docs now separate pointwise target-cell drift from
legacy zone max/rms summaries. The current accepted-step trace matches
`309/309` max-order-2 reference-grid points, writes the `Dnnh` preparation
ladder plus `Vh`/`eta_h`, records reference solver order, and uses the
covariant `Grad(logPnlim)` metric norm with the carried metric terms. The
latest component-enabled rerun explicitly replays the reference startup order
sequence and removes the only solver-order mismatch. It still leaves
`Dnnh_flux_max` as a `5.13e-3` target-cell cap drift driven by `logPnlimh`
and scalar `grad_logPnlimh` preparation, while optional
`grad_logPnlim*_x/y/z` fields are treated as diagnostic components rather than
the scalar cap input. The remaining neutral-mixed blocker is therefore
CVODE-style accepted-step state/history preparation feeding the neutral
pressure/log-pressure limiter, not a missing pressure-gradient, viscosity, or
raw diffusion source formula.

![Fluid 1D MMS convergence](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__fluid_1d_mms_convergence_artifacts__images__fluid_1d_mms_convergence.png)

![Imported QA stellarator SOL poster](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__images__poster_compact.png)

![Autodiff diffusion uncertainty](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_uncertainty_artifacts__images__autodiff_diffusion_uncertainty.png)

### Release-Closed Example Gallery

These are the main `1.0.3` release-scope artifacts. They are intentionally
release-hosted instead of tracked in git. Restore the local copies first:

```bash
python scripts/fetch_example_artifacts.py --skip-baselines --force
```

Regenerate the diverted tokamak movie and profile analysis from the restored
arrays:

```bash
PYTHONPATH=src python examples/diverted_tokamak_movie_demo.py
PYTHONPATH=src python examples/diverted_tokamak_profile_analysis_demo.py
```

![Diverted tokamak profile analysis](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__images__diverted_tokamak_turbulence_profiles.png)

Run the core self-contained operator, neutral, and recycling validation
campaigns:

```bash
PYTHONPATH=src python examples/engineering/fluid_1d_mms_convergence_demo.py
PYTHONPATH=src python examples/engineering/open_field_operator_campaign_demo.py
PYTHONPATH=src python examples/engineering/target_recycling_campaign_demo.py
PYTHONPATH=src python examples/engineering/neutral_parallel_diffusion_campaign_demo.py
PYTHONPATH=src python examples/engineering/neutral_mixed_term_balance_campaign_demo.py
```

![Open-field operator validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__open_field_operator_campaign_artifacts__images__open_field_operator_campaign.png)

![Neutral mixed NVh term-balance audit](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_term_balance_campaign_artifacts__images__neutral_mixed_term_balance_campaign.png)

Run the compact differentiability examples:

```bash
PYTHONPATH=src python examples/autodiff_diffusion_sensitivity_demo.py
PYTHONPATH=src python examples/autodiff_diffusion_uncertainty_demo.py
PYTHONPATH=src python examples/autodiff_diffusion_inverse_design_demo.py
```

![Autodiff sensitivity](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_sensitivity_artifacts__images__autodiff_diffusion_sensitivity.png)

![Autodiff inverse design](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_inverse_design_artifacts__images__autodiff_diffusion_inverse_design.png)

Run the local CPU scaling example:

```bash
PYTHONPATH=src python examples/strong_scaling_diffusion_demo.py \
  --skip-gpu \
  --cpu-device-counts 1,2,4 \
  --total-batch 32 \
  --nx 512 \
  --ny 64 \
  --steps 12 \
  --repeats 2
```

![Local CPU scaling](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__local_cpu_scaling_campaign_artifacts__images__local_cpu_scaling_campaign.png)

## 3D Geometry And Movies

`jax_drb` includes reusable 3D geometry tooling for:

- tokamak sample-data scaffolds,
- traced-field-line metric and selected-plane workflows,
- analytic stellarator and imported-equilibrium scaffolds,
- native reduced selected-field comparisons,
- toroidal and slice-based movie generation.

The stellarator examples are plain Python driver scripts in the same spirit as
SIMSOPT examples: edit the constants at the top, run the file, and inspect the
JSON/NPZ/PNG/GIF artifacts it writes. The README figures can be regenerated or
closely reproduced from:

- [examples/geometry-3D/stellarator-fci/geometry_plotting_demo.py](examples/geometry-3D/stellarator-fci/geometry_plotting_demo.py)
- [examples/geometry-3D/stellarator-fci/linear_mode_demo.py](examples/geometry-3D/stellarator-fci/linear_mode_demo.py)
- [examples/geometry-3D/stellarator-fci/vorticity_bracket_demo.py](examples/geometry-3D/stellarator-fci/vorticity_bracket_demo.py)
- [examples/geometry-3D/stellarator-fci/nonlinear_turbulence_demo.py](examples/geometry-3D/stellarator-fci/nonlinear_turbulence_demo.py)
- [examples/geometry-3D/stellarator-fci/turbulent_profile_analysis_demo.py](examples/geometry-3D/stellarator-fci/turbulent_profile_analysis_demo.py)
- [examples/geometry-3D/stellarator-fci/validation_campaign_demo.py](examples/geometry-3D/stellarator-fci/validation_campaign_demo.py)
- [examples/geometry-3D/essos-field-lines/direct_coil_open_sol_demo.py](examples/geometry-3D/essos-field-lines/direct_coil_open_sol_demo.py)
- [examples/geometry-3D/essos-field-lines/imported_connection_length_refinement_demo.py](examples/geometry-3D/essos-field-lines/imported_connection_length_refinement_demo.py)
- [examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py](examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py)
- [examples/geometry-3D/essos-field-lines/imported_drb_movie_stationarity_campaign.py](examples/geometry-3D/essos-field-lines/imported_drb_movie_stationarity_campaign.py)
- [examples/diverted_tokamak_movie_demo.py](examples/diverted_tokamak_movie_demo.py)
- [examples/diverted_tokamak_profile_analysis_demo.py](examples/diverted_tokamak_profile_analysis_demo.py)
- [examples/autodiff_diffusion_uncertainty_demo.py](examples/autodiff_diffusion_uncertainty_demo.py)

For nonlinear stellarator physics, start with
[examples/geometry-3D/stellarator-fci/vorticity_bracket_demo.py](examples/geometry-3D/stellarator-fci/vorticity_bracket_demo.py):
it shows the potential/vorticity solve and tested logical \(E\times B\)
bracket explicitly before the faster movie-oriented reduced benchmark.

### ESSOS Coil, VMEC, And Hybrid Stellarator Maps

The ESSOS lane separates field generation from plasma evolution. ESSOS owns
the coil representation, Biot-Savart field evaluation, adaptive tracing, and
Poincare extraction. JAXDRB imports those arrays, builds fixed-shape FCI maps,
and runs JAX-native geometry, sheath/recycling, neutral, PyTree/JVP, and
reduced DRB validation gates on the imported maps.

The imported stellarator scripts support three map semantics:

- `coil`: ESSOS Biot-Savart coil-traced adjacent-plane endpoints with
  open-field endpoint masks. This is the direct open-field coil-geometry lane.
- `vmec`: VMEC-coordinate closed-field map from \(d\theta/d\phi =
  B^\theta/B^\phi\), preserving closed flux surfaces and disabling target
  endpoint masks. This is the closed-field control.
- `hybrid`: VMEC-coordinate map positions with coil-derived endpoint masks,
  connection-length proxy, and \(|B|\) modulation. This is the current
  open-field SOL bridge used by the strongest release-hosted imported QA
  movie.

Current status: open-field `coil` and `hybrid` maps are implemented and carry
endpoint/sheath/recycling diagnostics, but pure-coil long traces do not remain
on a single scaled VMEC seed surface over the long trace. Therefore the
release promotes compact imported-field validation and the high-resolution
hybrid movie, not a device-scale predictive coil-field turbulence claim. The
closed-field path should use the `vmec` map source until a coil-only closed
surface construction passes its own Poincare and refinement gates.

Restore the release-hosted ESSOS/VMEC/hybrid figures and movie:

```bash
python scripts/fetch_example_artifacts.py --skip-baselines --force
```

Run the self-contained connection-length refinement gate:

```bash
PYTHONPATH=src python \
  examples/geometry-3D/essos-field-lines/imported_connection_length_refinement_demo.py
```

Run the direct-coil open-SOL workflow contract. The default command is
self-contained and writes a dry-run promotion ledger under `artifacts/`; edit
the live flags at the top of the script to regenerate the ESSOS coil FCI,
connection-length, endpoint/source, stationarity, and diagnostic media gates
from local geometry. The live FCI stage now feeds a separate source/profile
gate JSON and PNG that check target labels, sheath heat load, neutral
ionisation, target particle-loss flux, radial profiles, and source-balance
residuals from the same consumed endpoint masks. The endpoint-label refinement
gate also requires a nonzero endpoint population, so open-field promotion
cannot pass on mostly interior cells with no target contact. The
`RUN_LIVE_MEDIA_GATE`
flag writes GIF/PNG/NPZ media from the direct coil field, but the workflow keeps
that media out of promotion unless the
geometry, source-accounting, refinement, and visual-QA gates also pass. The
summary JSON lists `promotion_rejection_reasons`, `promotion_blocking_stages`,
and `next_actions`, so a default dry run explains that no live promotion gates
have run rather than looking like a silent failure:

```bash
PYTHONPATH=src python \
  examples/geometry-3D/essos-field-lines/direct_coil_open_sol_demo.py
```

Run the hybrid VMEC/coil open-SOL workflow contract. This is the planned
promotion path when pure direct-coil endpoint maps remain too rough: VMEC
provides smooth map coordinates, while coil traces provide endpoint masks and
magnetic-field modulation. The default command is self-contained and writes a
dry-run promotion ledger; live mode adds FCI/source-profile,
parallel-step-refinement, stationarity, grid/time-refinement, and optional
media gates before any hybrid movie can be promoted. The
`STATIONARITY_PRESET = "quick"` setting is a bounded workflow smoke test and
is deliberately not promotion evidence; use `"promotion"` plus the documented
grid/time and visual-QA gates before using a hybrid movie as README or paper
evidence:

```bash
PYTHONPATH=src python \
  examples/geometry-3D/essos-field-lines/hybrid_open_sol_demo.py
```

Run the direct-coil closed/near-closed control. The default command is
self-contained and classifies manufactured non-axisymmetric traces. It now
writes both a base return-map/Poincare report and a refinement report that
checks whether closed/near-closed classification, same-section return distance,
and Poincare sampling remain stable as seed and trace samples increase. It
also writes a reduced closed-trace transient plot and GIF. Set
`RUN_LIVE_ESSOS = True` in the script to classify live Landreman-Paul QA
direct-coil traces from ESSOS, or reuse a restored live trace bundle through
the validation API. This remains a closed-field diagnostic and does not apply
target, sheath, recycling, or neutral semantics:

```bash
PYTHONPATH=src python \
  examples/geometry-3D/essos-field-lines/direct_coil_closed_field_demo.py
```

Run the VMEC closed-field control. The default command writes self-contained
live-run contracts for the periodic FCI/operator gate and the reduced
closed-field transient. Set `RUN_LIVE_VMEC = True` to regenerate the periodic
VMEC FCI map and set `RUN_LIVE_VMEC_TRANSIENT = True` to generate the
profile/spectrum plot and GIF. This is the smooth closed-field tutorial path
and does not apply target, sheath, recycling, or neutral-loss semantics:

```bash
PYTHONPATH=src python \
  examples/geometry-3D/essos-field-lines/vmec_closed_field_demo.py
```

Run the imported FCI campaign. By default this is a safe dry run for `coil`;
edit the constants at the top of the script to set
`MAP_SOURCES_TO_RUN = ("coil", "vmec", "hybrid")`, set `DRY_RUN = False`, and
provide `JAX_DRB_ESSOS_ROOT`, `COIL_JSON_PATH`, or `VMEC_WOUT_PATH` when
regenerating live imported geometry:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src python \
  examples/geometry-3D/essos-field-lines/imported_fci_campaign.py
```

Run the current high-resolution hybrid stationarity gate:

```bash
PYTHONPATH=src python \
  examples/geometry-3D/essos-field-lines/imported_drb_movie_stationarity_campaign.py
```

Regenerate a movie package from the external geometry by editing
`MAP_SOURCE = "coil"`, `"vmec"`, or `"hybrid"` near the top of
`imported_drb_movie_campaign.py`, then running:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src python \
  examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py
```

![ESSOS imported QA-hybrid diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__images__diagnostics.png)

![ESSOS imported QA-hybrid snapshots](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__images__snapshots.png)

![ESSOS imported QA-hybrid movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__movies__movie_compact.gif)

Detailed guides:

- [docs/stellarator_examples.md](docs/stellarator_examples.md)
- [docs/tokamak_tcv_x21_scaffold_demo.md](docs/tokamak_tcv_x21_scaffold_demo.md)
- [docs/tokamak_tcv_x21_toroidal_movie_demo.md](docs/tokamak_tcv_x21_toroidal_movie_demo.md)
- [docs/tokamak_tcv_x21_selected_field_demo.md](docs/tokamak_tcv_x21_selected_field_demo.md)
- [docs/tokamak_native_selected_field_demo.md](docs/tokamak_native_selected_field_demo.md)
- [docs/traced_field_line_scaffold_demo.md](docs/traced_field_line_scaffold_demo.md)
- [docs/traced_field_line_selected_field_demo.md](docs/traced_field_line_selected_field_demo.md)
- [docs/traced_field_line_native_selected_field_demo.md](docs/traced_field_line_native_selected_field_demo.md)
- [docs/stellarator_fci_validation.md](docs/stellarator_fci_validation.md)
- [docs/vmec_extender_edge_fields.md](docs/vmec_extender_edge_fields.md)
- [docs/essos_imported_fci_validation.md](docs/essos_imported_fci_validation.md)
- [docs/essos_direct_coil_closed_control.md](docs/essos_direct_coil_closed_control.md)
- [docs/essos_vmec_closed_field.md](docs/essos_vmec_closed_field.md)
- [docs/essos_vmec_fieldline_surface.md](docs/essos_vmec_fieldline_surface.md)
- [docs/non_axisymmetric_stellarator_sol_plan.md](docs/non_axisymmetric_stellarator_sol_plan.md)
- [docs/dynamics_gallery.md](docs/dynamics_gallery.md)
- [docs/validation_gallery.md](docs/validation_gallery.md)

## Differentiable Driver Lanes

The differentiable examples currently include:

- sensitivity analysis,
- uncertainty propagation,
- inverse design,
- fixed-workload scaling.

Entry points:

- [examples/autodiff_diffusion_sensitivity_demo.py](examples/autodiff_diffusion_sensitivity_demo.py)
- [examples/autodiff_diffusion_uncertainty_demo.py](examples/autodiff_diffusion_uncertainty_demo.py)
- [examples/autodiff_diffusion_inverse_design_demo.py](examples/autodiff_diffusion_inverse_design_demo.py)
- [examples/strong_scaling_diffusion_demo.py](examples/strong_scaling_diffusion_demo.py)
- [docs/autodiff_and_scaling_examples.md](docs/autodiff_and_scaling_examples.md)
- [docs/autodiff_diffusion_uncertainty_demo.md](docs/autodiff_diffusion_uncertainty_demo.md)

## Physics, Algorithms, And Performance

The governing equations, closures, numerical operators, runtime design, and differentiability boundary are documented here:

- [docs/physics_models.md](docs/physics_models.md)
- [docs/code_structure.md](docs/code_structure.md)
- [docs/performance_and_differentiability.md](docs/performance_and_differentiability.md)
- [docs/profiling_runtime.md](docs/profiling_runtime.md)
- [docs/runtime_gap_remediation.md](docs/runtime_gap_remediation.md)
- [docs/native_runtime_cli.md](docs/native_runtime_cli.md)
- [docs/geometry_roadmap.md](docs/geometry_roadmap.md)
- [docs/research_directions.md](docs/research_directions.md)
- [docs/refactoring_plan.md](docs/refactoring_plan.md)
- [docs/research_grade_execution_plan.md](docs/research_grade_execution_plan.md)

The runtime/performance audit tools include:

- [docs/native_3d_runtime_campaign.md](docs/native_3d_runtime_campaign.md)
- [docs/native_3d_convergence_campaign.md](docs/native_3d_convergence_campaign.md)
- [docs/fluid_1d_mms_convergence.md](docs/fluid_1d_mms_convergence.md)
- [docs/jax_native_profile_audit.md](docs/jax_native_profile_audit.md)
- [docs/local_cpu_scaling_campaign.md](docs/local_cpu_scaling_campaign.md)
- [docs/research_campaigns.md](docs/research_campaigns.md)
- [docs/repo_size_audit.md](docs/repo_size_audit.md)
- [scripts/profile_curated_case.py](scripts/profile_curated_case.py)
- [scripts/run_research_campaign_bundle.py](scripts/run_research_campaign_bundle.py)
- [scripts/profile_stellarator_drb_pytree.py](scripts/profile_stellarator_drb_pytree.py)
- [scripts/profile_recycling_batched_jvp_gate.py](scripts/profile_recycling_batched_jvp_gate.py)
- [scripts/profile_atomic_rate_throughput_gate.py](scripts/profile_atomic_rate_throughput_gate.py)

The strongest current same-machine native-versus-reference evidence is the
public live-rerun matrix in the validation docs. It shows exact compact 2D
lanes on guarded compare surfaces, bounded but normalization-sensitive
one-step mismatch on the integrated and direct-tokamak recycling ladders, and
the remaining main runtime/fidelity gaps on the heavy 1D neutral/recycling
paths.

For local MacBook-class CPU use, the strongest current scaling result is the
heavy fixed-work ensemble on repeated neon-enabled direct tokamak recycling
solves rather than extra threads on one warmed single solve. The committed
local artifact reaches about `4.79x` steady-state speedup from `1 -> 8`
workers on a `16`-solve heavy ensemble, with intermediate speedups of about
`1.94x` and `3.32x` at `2` and `4` workers.

For the new non-axisymmetric 3D lane, the current performance gate is the
fixed-layout PyTree RHS campaign. It verifies JVP derivatives against finite
differences, checks `vmap` against serial objective evaluation, records
single-device batched throughput, and can run `pmap` when multiple local
devices are visible, pass an identity-map runtime sanity check, and then pass
the real-kernel parity check.

The heavier D/T/He fixed-layout recycling residual also has CPU and GPU
profile evidence. The current same-fidelity GPU gate reaches the same residual
norm as the CPU gate, but it is much slower and uses more sampled process-tree
RSS on the full-field residual. That is useful negative evidence: the full
production BDF recycling lane remains the active target for active-array
residuals, smaller JAX residual/JVP kernels, and batched GPU promotion.

The new batched recycling residual/JVP gate verifies the D/T/He fixed-layout
residual under `jit`, `vmap`, `jvp`, and `grad` on the real recycling state.
It uses the fixed full-field RHS by default, exposes the active-array RHS as
the migration seam, and keeps the older host bridge only as an explicit
diagnostic comparison backend. On the local CPU run, the retained batch sweep
through `64` states gives about `2.28x` residual throughput speedup and about
`1.96x` JVP throughput speedup over serial same-kernel calls, while the
JVP/finite-difference error is about `5.97e-9`. The same artifact now also
checks a reusable `jax.linearize` action against direct JVPs, with agreement at
about `3.47e-18`.

The adaptive-BDF recycling solver also has a bounded JAX-linearized promotion
gate. It is intentionally opt-in rather than the production default: the stable
full output-window path still uses the validated compatibility backend, while
the JAX-linearized gate checks fixed-layout residuals, JVP/Jacobian-action
solves, internal timestep control, and implicit substep convergence. The
current local single-species gate passes at `timestep=1.0` with zero fallback,
zero unconverged substeps, max accepted embedded error ratio about `0.93`, and
variable-step BDF2 history reuse after rejected-step timestep changes. On this
gate, the in-tree JAX GMRES path used `50` implicit trial solves in about
`108 s` with zero failed inner linear solves. The optional Lineax GMRES seam
ran the same controller history in about `91 s`, but it reported failed inner
linear solves and remains a diagnostic backend rather than a promoted default.
The exact commands, caveats, and latest numbers are in
[docs/performance_and_differentiability.md](docs/performance_and_differentiability.md)
and [docs/research_campaigns.md](docs/research_campaigns.md).

The multi-ion D/T/He adaptive-BDF route is also still opt-in. The current
passing diagnostics-only promotion-style result uses the sparse-JVP
adaptive-BDF route plus component-wise absolute-tolerance floors; it has no
fallbacks or unconverged substeps on the bounded local gate, but it is not a
default solver until longer output-window reference-parity and runtime
campaigns pass on the same route.

For accelerator evidence, the source-term throughput gate now shows a real
GPU win on the office machine for batched atomic-rate kernels: at `4,194,304`
temperature points the GPU is about `2.5x` faster for the rate surface and
about `2.0x` faster for its autodiff derivative. The same gate checks a
log-temperature sensitivity objective against finite differences at about
`1e-10` relative error. Heavy output-window recycling
GPU speedup is still not claimed until that path exits the host/SciPy barrier.
The optional `pmap` branch in that profiler is guarded by a device-level
identity check before any multi-device timing is reported, so a broken
self-hosted runtime is recorded as unavailable rather than as a speedup claim.

## Validation And Control Packages

Focused engineering and benchmark packages:

- [docs/reactions_collisions_campaign.md](docs/reactions_collisions_campaign.md)
- [docs/atomic_rate_differentiability_campaign.md](docs/atomic_rate_differentiability_campaign.md)
- [docs/neutral_parallel_diffusion_campaign.md](docs/neutral_parallel_diffusion_campaign.md)
- [docs/collision_closure_campaign.md](docs/collision_closure_campaign.md)
- [docs/tokamak_anomalous_diffusion_campaign.md](docs/tokamak_anomalous_diffusion_campaign.md)
- [docs/tokamak_recycling_observable_campaign.md](docs/tokamak_recycling_observable_campaign.md)
- [docs/target_recycling_campaign.md](docs/target_recycling_campaign.md)
- [docs/neutral_mixed_term_balance_campaign.md](docs/neutral_mixed_term_balance_campaign.md)
- [docs/hermes_neutral_mixed_accepted_step_trace_monitor.md](docs/hermes_neutral_mixed_accepted_step_trace_monitor.md)
- [docs/impurity_radiation_campaign.md](docs/impurity_radiation_campaign.md)
- [docs/controller_feedback_campaign.md](docs/controller_feedback_campaign.md)
- [docs/temperature_feedback_campaign.md](docs/temperature_feedback_campaign.md)
- [docs/detachment_controller_campaign.md](docs/detachment_controller_campaign.md)
- [docs/stellarator_fci_validation.md](docs/stellarator_fci_validation.md)

## Testing

Run the fast bounded research slice:

```bash
python scripts/run_fast_research_checks.py
```

Run the full suite:

```bash
pytest -q
```

Run the bounded closeout coverage gate:

```bash
python scripts/run_closeout_coverage.py
```

Run the promoted native-solver and public-surface coverage gate:

```bash
python scripts/run_promoted_solver_coverage.py
```

Run the scheduled/manual research campaign wrapper:

```bash
python scripts/run_research_campaign_bundle.py --campaign scheduled-fast-research
```

The shipping CI matrix runs on Python 3.10, 3.11, and 3.12.

The GitHub coverage workflow enforces the two release-readiness coverage lanes:
bounded closeout coverage and promoted solver/public-surface coverage. The
separate research-campaign workflow runs the hosted `scheduled-fast-research`
slice weekly and exposes heavier live-reference, local, GPU, adaptive-BDF, and
profiling bundles as explicit manual lanes.

Testing policy and refactor coverage goals are documented in:

- [docs/testing_strategy.md](docs/testing_strategy.md)

## Packaging And Release

Build locally:

```bash
python scripts/audit_release_readiness.py
python -m build
```

Release/package documentation:

- [docs/release_packaging.md](docs/release_packaging.md)
- [docs/release_notes_1_0_3.md](docs/release_notes_1_0_3.md)
- [docs/release_notes_1_0_2.md](docs/release_notes_1_0_2.md)
- [docs/release_notes_1_0_1.md](docs/release_notes_1_0_1.md)
- [docs/release_notes_1_0_0.md](docs/release_notes_1_0_0.md)
