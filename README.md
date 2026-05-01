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

`jax_drb` is a JAX-native edge and scrape-off-layer plasma code for drift-reduced Braginskii models, electrostatic turbulence, neutral transport, curated tokamak workflows, and reusable 3D geometry diagnostics.

Documentation is available at [jax-drb.readthedocs.io](https://jax-drb.readthedocs.io/).

The codebase is organized around:

- a standalone CLI and Python API,
- restartable native runs with structured terminal progress,
- portable analysis and visualization artifacts,
- explicit capability tiers for curated benchmark lanes,
- reusable 3D geometry, movie, and selected-field comparison tools,
- differentiable driver paths for sensitivity analysis, uncertainty propagation, and inverse design.

![Diverted tokamak dynamics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__movies__diverted_tokamak_turbulence.gif)

![3D tokamak toroidal dynamics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_tcv_x21_toroidal_movie_artifacts__movies__tokamak_tcv_x21_toroidal.gif)

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
- transient interval progress on recycling lanes, including interval count,
  accepted timestep, simulated time, and estimated remaining wall time on the
  live native implicit paths,
- artifact writes.

## Capability Tiers

Curated validation cases are labeled explicitly:

- `native_exact`: fully native and strong enough for the main public benchmark surface,
- `native_operational`: native and useful, but still carrying bounded residuals,
- `scaffolded_reference_backed`: useful for diagnostics or geometry staging, but not counted as native closure.

The current promoted matrix includes:

- exact compact 2D blob, drift-wave, and tokamak lanes,
- exact and operational recycling lanes with external-reference gates,
- native 3D reduced tokamak, traced-field-line, and stellarator selected-field bundles,
- control, reaction, impurity, neutral, and profiling campaign packages.

The detailed status surface lives in:

- [docs/implementation_inventory.md](docs/implementation_inventory.md)
- [docs/parity_harness.md](docs/parity_harness.md)
- [docs/parity_matrix.md](docs/parity_matrix.md)
- [docs/research_grade_validation_matrix.md](docs/research_grade_validation_matrix.md)
- [docs/validation_gallery.md](docs/validation_gallery.md)

## Publication-Ready Validation Artifacts

The current public figures are generated from committed regression or campaign
artifacts, so the docs, tests, and manuscript planning all point to the same
evidence. The full figure index is in
[docs/validation_gallery.md](docs/validation_gallery.md); the matrix that
separates primary scientific figures from supporting engineering figures is in
[docs/research_grade_validation_matrix.md](docs/research_grade_validation_matrix.md).

Verification and operator accuracy:

![Fluid 1D MMS convergence](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__fluid_1d_mms_convergence_artifacts__images__fluid_1d_mms_convergence.png)

![Open-field operator campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__open_field_operator_campaign_artifacts__images__open_field_operator_campaign.png)

Physics closures and tokamak/recycling observables:

![Reactions and collisions campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__reactions_collisions_campaign_artifacts__images__reactions_collisions_campaign.png)

![Neutral parallel diffusion campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_parallel_diffusion_campaign_artifacts__images__neutral_parallel_diffusion_campaign.png)

![Collision closure campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__collision_closure_campaign_artifacts__images__collision_closure_campaign.png)

![Target recycling campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__target_recycling_campaign_artifacts__images__target_recycling_campaign.png)

![Tokamak recycling observable campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_recycling_observable_campaign_artifacts__images__tokamak_recycling_observable_campaign.png)

Reference parity and offender localization:

![Neutral mixed boundary campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_boundary_campaign_artifacts__images__neutral_mixed_boundary_campaign.png)

![Neutral mixed term-balance campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_term_balance_campaign_artifacts__images__neutral_mixed_term_balance_campaign.png)

Non-axisymmetric 3D geometry and reduced SOL dynamics:

The imported QA geometry lane supports three map sources for development and
validation: `coil` for open Biot-Savart coil traces, `vmec` for a closed
surface-preserving VMEC-coordinate control, and `hybrid` for VMEC-coordinate
interpolation with coil-derived sheath/recycling endpoint masks.

![Stellarator FCI geometry validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__geometry__images__stellarator_fci_geometry_campaign.png)

![Stellarator FCI multi-configuration suite](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__suite__images__stellarator_fci_suite_campaign.png)

![Stellarator FCI operator validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__operators__images__stellarator_fci_operator_campaign.png)

![Stellarator full metric MMS validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__metric_mms__images__stellarator_metric_mms_campaign.png)

![Stellarator sheath/recycling validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__sheath_recycling__images__stellarator_sheath_recycling_campaign.png)

![Stellarator neutral physics validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__neutral_physics__images__stellarator_neutral_physics_campaign.png)

![Stellarator vorticity validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__vorticity__images__stellarator_vorticity_campaign.png)

![Stellarator PyTree/JVP/scaling validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__pytree_drb__images__stellarator_drb_pytree_campaign.png)

![ESSOS field-line import](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_fieldline_import_artifacts__images__essos_landreman_paul_qa_fieldline_import.png)

![ESSOS field-line/VMEC surface registration](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_vmec_fieldline_surface_artifacts__images__essos_vmec_fieldline_surface_campaign.png)

![ESSOS VMEC equilibrium surface-preservation gate](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_vmec_equilibrium_fieldline_surface_artifacts__images__essos_vmec_equilibrium_fieldline_surface_campaign.png)

![ESSOS imported FCI validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_artifacts__images__essos_imported_fci_campaign.png)

![ESSOS imported FCI VMEC-coordinate validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_vmec_artifacts__images__essos_imported_fci_vmec_campaign.png)

![ESSOS imported FCI hybrid validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_hybrid_artifacts__images__essos_imported_fci_hybrid_campaign.png)

![ESSOS imported PyTree/JVP validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_pytree_artifacts__images__essos_imported_pytree_campaign.png)

![ESSOS imported QA-coil DRB movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__movies__essos_imported_drb_movie_campaign.gif)

![ESSOS imported QA-hybrid DRB movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_hybrid_artifacts__movies__essos_imported_drb_movie_hybrid_campaign.gif)

![Stellarator SOL diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__showcase__images__stellarator_sol_showcase_diagnostics.png)

![Stellarator SOL 3D movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__showcase__movies__stellarator_sol_showcase.gif)

Differentiability, uncertainty propagation, and local performance:

![Autodiff diffusion sensitivity](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_sensitivity_artifacts__images__autodiff_diffusion_sensitivity.png)

![Atomic rate differentiability campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__atomic_rate_differentiability_campaign_artifacts__images__atomic_rate_differentiability_campaign.png)

![Autodiff diffusion uncertainty](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_uncertainty_artifacts__images__autodiff_diffusion_uncertainty.png)

![Autodiff diffusion inverse design](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_inverse_design_artifacts__images__autodiff_diffusion_inverse_design.png)

![Strong scaling diffusion](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__strong_scaling_diffusion_artifacts__images__strong_scaling_diffusion.png)

![Local CPU scaling campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__local_cpu_scaling_campaign_artifacts__images__local_cpu_scaling_campaign.png)

![Implicit solver profile audit](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__implicit_solver_profile_audit_artifacts__images__implicit_solver_profile_audit.png)

## 3D Geometry And Movies

`jax_drb` includes reusable 3D geometry tooling for:

- tokamak sample-data scaffolds,
- traced-field-line metric and selected-plane workflows,
- analytic stellarator and imported-equilibrium scaffolds,
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
- [docs/stellarator_fci_validation.md](docs/stellarator_fci_validation.md)
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
local artifact reaches about `4.94x` steady-state speedup from `1 -> 8`
workers on a `16`-solve heavy ensemble, with intermediate speedups of about
`1.88x` and `3.67x` at `2` and `4` workers.

For the new non-axisymmetric 3D lane, the current performance gate is the
fixed-layout PyTree RHS campaign. It verifies JVP derivatives against finite
differences, checks `vmap` against serial objective evaluation, records
single-device batched throughput, and can run `pmap` when multiple local
devices are visible and pass parity checks.

The heavier D/T/He fixed-layout recycling residual also has CPU and GPU
profile evidence. The current GPU gate reaches the same residual norm as the
CPU gate and lowers peak RSS on the small fixed-layout problem, but it is not
yet claimed as a GPU speedup because this problem size is still launch- and
compile-overhead limited. The full production BDF recycling lane remains the
active target for JAX-native residual and Jacobian-action promotion.

The new batched recycling residual/JVP gate verifies the D/T/He fixed-layout
residual under `jit`, `vmap`, `jvp`, and `grad` on the real recycling state.
On the local CPU run with `mesh:ny=100`, the retained batch sweep through 256
states gives about `2.8x` residual throughput speedup and about `2.2x` JVP
throughput speedup over serial same-kernel calls, while the
JVP/finite-difference error is about `6e-9`.
For accelerator evidence, the source-term throughput gate now shows a real
GPU win on the office machine for batched atomic-rate kernels: at `4,194,304`
temperature points the GPU is about `2.5x` faster for the rate surface and
about `2.1x` faster for its autodiff derivative. The same gate checks a
log-temperature sensitivity objective against finite differences at about
`1e-10` relative error. Heavy output-window recycling
GPU speedup is still not claimed until that path exits the host/SciPy barrier.

## Validation And Control Packages

Focused engineering and benchmark packages:

- [docs/reactions_collisions_campaign.md](docs/reactions_collisions_campaign.md)
- [docs/atomic_rate_differentiability_campaign.md](docs/atomic_rate_differentiability_campaign.md)
- [docs/neutral_parallel_diffusion_campaign.md](docs/neutral_parallel_diffusion_campaign.md)
- [docs/collision_closure_campaign.md](docs/collision_closure_campaign.md)
- [docs/tokamak_anomalous_diffusion_campaign.md](docs/tokamak_anomalous_diffusion_campaign.md)
- [docs/tokamak_recycling_observable_campaign.md](docs/tokamak_recycling_observable_campaign.md)
- [docs/target_recycling_campaign.md](docs/target_recycling_campaign.md)
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

Testing policy and refactor coverage goals are documented in:

- [docs/testing_strategy.md](docs/testing_strategy.md)

## Packaging And Release

Build locally:

```bash
python -m build
```

Release/package documentation:

- [docs/release_packaging.md](docs/release_packaging.md)
- [docs/release_notes_1_0_2.md](docs/release_notes_1_0_2.md)
- [docs/release_notes_1_0_1.md](docs/release_notes_1_0_1.md)
- [docs/release_notes_1_0_0.md](docs/release_notes_1_0_0.md)
