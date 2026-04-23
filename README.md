# jax_drb

`jax_drb` is a JAX-native edge and scrape-off-layer plasma code for drift-reduced Braginskii models, electrostatic turbulence, neutral transport, curated tokamak workflows, and reusable 3D geometry diagnostics.

The codebase is organized around:

- a standalone CLI and Python API,
- restartable native runs with structured terminal progress,
- portable analysis and visualization artifacts,
- explicit capability tiers for curated benchmark lanes,
- reusable 3D geometry, movie, and selected-field comparison tools,
- differentiable driver paths for sensitivity analysis, uncertainty propagation, and inverse design.

![Diverted tokamak dynamics](docs/data/diverted_tokamak_turbulence_artifacts/movies/diverted_tokamak_turbulence.gif)

![3D tokamak toroidal dynamics](docs/data/tokamak_tcv_x21_toroidal_movie_artifacts/movies/tokamak_tcv_x21_toroidal.gif)

## Install

PyPI:

```bash
pip install jax-drb
```

From source:

```bash
git clone https://github.com/rogeriojorge/uw_plasma
cd jax_drb
pip install -e .
```

The default installation already pulls in the runtime and analysis dependencies used by the main CLI, geometry tooling, and diagnostics, including `jax`, `diffrax`, `scipy`, `equinox`, `matplotlib`, and `netCDF4`.

## Quick Start

Run a TOML deck:

```bash
jax_drb path/to/input.toml
```

Inspect a deck without running it:

```bash
jax_drb inspect path/to/input.toml
```

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
from pathlib import Path

from jax_drb.cli import main
from jax_drb.native import run_curated_case, run_input_case

main(["run", "examples/inputs/restartable_diffusion.toml", "--quiet"])

result = run_curated_case("tokamak_isothermal_one_step", reference_root=Path("/path/to/reference-suite"))
print(result.payload["capability_tier"])
print(sorted(result.variables))

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
- transient interval progress on recycling lanes,
- artifact writes.

## Capability Tiers

Curated validation cases are labeled explicitly:

- `native_exact`: fully native and strong enough for the main public benchmark surface,
- `native_operational`: native and useful, but still carrying bounded residuals,
- `scaffolded_reference_backed`: useful for diagnostics or geometry staging, but not counted as native closure.

The current promoted matrix includes:

- exact compact 2D blob, drift-wave, and tokamak lanes,
- exact and operational recycling lanes with live Hermes-backed gates,
- native 3D reduced tokamak, traced-field-line, and stellarator selected-field bundles,
- control, reaction, impurity, neutral, and profiling campaign packages.

The detailed status surface lives in:

- [docs/implementation_inventory.md](docs/implementation_inventory.md)
- [docs/hermes_capability_audit.md](docs/hermes_capability_audit.md)
- [docs/parity_harness.md](docs/parity_harness.md)
- [docs/parity_matrix.md](docs/parity_matrix.md)

## 3D Geometry And Movies

`jax_drb` includes reusable 3D geometry tooling for:

- tokamak sample-data scaffolds,
- traced-field-line metric and selected-plane workflows,
- stellarator/VMEC equilibrium scaffolds,
- native reduced selected-field comparisons,
- toroidal and slice-based movie generation.

Useful entry points:

- [docs/tokamak_tcv_x21_scaffold_demo.md](docs/tokamak_tcv_x21_scaffold_demo.md)
- [docs/tokamak_tcv_x21_toroidal_movie_demo.md](docs/tokamak_tcv_x21_toroidal_movie_demo.md)
- [docs/tokamak_tcv_x21_selected_field_demo.md](docs/tokamak_tcv_x21_selected_field_demo.md)
- [docs/tokamak_native_selected_field_demo.md](docs/tokamak_native_selected_field_demo.md)
- [docs/traced_field_line_scaffold_demo.md](docs/traced_field_line_scaffold_demo.md)
- [docs/traced_field_line_selected_field_demo.md](docs/traced_field_line_selected_field_demo.md)
- [docs/traced_field_line_native_selected_field_demo.md](docs/traced_field_line_native_selected_field_demo.md)
- [docs/stellarator_vmec_scaffold_demo.md](docs/stellarator_vmec_scaffold_demo.md)
- [docs/stellarator_vmec_selected_field_demo.md](docs/stellarator_vmec_selected_field_demo.md)
- [docs/stellarator_vmec_native_selected_field_demo.md](docs/stellarator_vmec_native_selected_field_demo.md)
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
- [docs/native_runtime_cli.md](docs/native_runtime_cli.md)
- [docs/geometry_roadmap.md](docs/geometry_roadmap.md)
- [docs/research_directions.md](docs/research_directions.md)
- [docs/refactoring_plan.md](docs/refactoring_plan.md)

The runtime/performance audit tools include:

- [docs/native_3d_runtime_campaign.md](docs/native_3d_runtime_campaign.md)
- [docs/native_3d_convergence_campaign.md](docs/native_3d_convergence_campaign.md)
- [docs/fluid_1d_mms_convergence.md](docs/fluid_1d_mms_convergence.md)
- [docs/jax_native_profile_audit.md](docs/jax_native_profile_audit.md)
- [docs/local_cpu_scaling_campaign.md](docs/local_cpu_scaling_campaign.md)

For local MacBook-class CPU use, the strongest current scaling result is the
heavy fixed-work ensemble on repeated neon-enabled direct tokamak recycling
solves rather than extra threads on one warmed single solve. The committed
local artifact reaches about `4.94x` steady-state speedup from `1 -> 8`
workers on a `16`-solve heavy ensemble, with intermediate speedups of about
`1.88x` and `3.67x` at `2` and `4` workers.

## Validation And Control Packages

Focused engineering and benchmark packages:

- [docs/reactions_collisions_campaign.md](docs/reactions_collisions_campaign.md)
- [docs/neutral_parallel_diffusion_campaign.md](docs/neutral_parallel_diffusion_campaign.md)
- [docs/collision_closure_campaign.md](docs/collision_closure_campaign.md)
- [docs/impurity_radiation_campaign.md](docs/impurity_radiation_campaign.md)
- [docs/controller_feedback_campaign.md](docs/controller_feedback_campaign.md)
- [docs/temperature_feedback_campaign.md](docs/temperature_feedback_campaign.md)
- [docs/detachment_controller_campaign.md](docs/detachment_controller_campaign.md)
- [docs/hermes_comparison_gallery.md](docs/hermes_comparison_gallery.md)

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

The shipping CI matrix runs on Python 3.10, 3.11, and 3.12.

Testing policy and refactor coverage goals are documented in:

- [docs/testing_strategy.md](docs/testing_strategy.md)

## Packaging And Release

Build locally:

```bash
python -m build
```

Release/package documentation:

- [docs/release_packaging.md](docs/release_packaging.md)
- [docs/release_notes_1_0_0.md](docs/release_notes_1_0_0.md)
