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
- gridded VMEC-extender edge-field import with physical-phi interpolation, FCI map construction, and a compact SOL smoke gate,
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

## Validation Figures

The README shows only a few representative artifacts. The full figure index,
including operator convergence, reactions/collisions, neutrals, recycling,
reference parity, CPU scaling, and differentiability plots, lives in
[docs/validation_gallery.md](docs/validation_gallery.md). The validation matrix
that separates primary scientific evidence from supporting engineering gates is
[docs/research_grade_validation_matrix.md](docs/research_grade_validation_matrix.md).

The neutral-mixed validation docs now separate pointwise target-cell drift from
legacy zone max/rms summaries. The current accepted-step trace matches
`148/148` reference-grid points, writes the `Dnnh` preparation ladder plus
`Vh`/`eta_h`, records reference solver order, and uses the covariant
`Grad(logPnlim)` metric norm with the carried metric terms. The remaining
neutral-mixed blocker is variable-order accepted-step history replay feeding
the near-target flux-limit cap, not a missing pressure-gradient or viscosity
source formula.

![Fluid 1D MMS convergence](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__fluid_1d_mms_convergence_artifacts__images__fluid_1d_mms_convergence.png)

![Stellarator SOL diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__showcase__images__stellarator_sol_showcase_diagnostics.png)

![Autodiff diffusion uncertainty](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_uncertainty_artifacts__images__autodiff_diffusion_uncertainty.png)

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
- [examples/geometry-3D/essos-field-lines/imported_connection_length_refinement_demo.py](examples/geometry-3D/essos-field-lines/imported_connection_length_refinement_demo.py)
- [examples/diverted_tokamak_movie_demo.py](examples/diverted_tokamak_movie_demo.py)
- [examples/diverted_tokamak_profile_analysis_demo.py](examples/diverted_tokamak_profile_analysis_demo.py)
- [examples/autodiff_diffusion_uncertainty_demo.py](examples/autodiff_diffusion_uncertainty_demo.py)

For nonlinear stellarator physics, start with
[examples/geometry-3D/stellarator-fci/vorticity_bracket_demo.py](examples/geometry-3D/stellarator-fci/vorticity_bracket_demo.py):
it shows the potential/vorticity solve and tested logical \(E\times B\)
bracket explicitly before the faster movie-oriented reduced benchmark.

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

## Parallelization (Sharded FCI Execution)

The flux-coordinate-independent (FCI) stack runs on multiple devices through
JAX `shard_map` domain decomposition. The 3D grid is partitioned over a device
mesh with partition spec `("x", "y", "z")`; each shard advances its subdomain
and exchanges one-cell face halos with its neighbours (`lax.ppermute`) once per
RK4 stage. The sharded step is bit-for-bit equivalent to the single-device step
(verified in `tests/test_fci_sharded_2field.py`: single-device agreement to
2e-16, and a forced-4-device subprocess to <1e-12).

Public API (`from jax_drb.native import ...`):

- `make_shard_mesh(shard_counts)` — build the device mesh,
- `build_local_fci_geometries(geometry, shard_counts)` — per-shard geometry,
- `make_sharded_2field_step(geometry, shard_counts, parameters, bcs, dt=...)` —
  a jitted, sharded RK4 step for the reduced two-field model.

Reproduce the strong-scaling study with the self-contained example (it
re-invokes itself once per device count so the XLA device count is fixed before
JAX imports):

```bash
# CPU: bind one core per shard so the baseline is not already multi-threaded
JAX_DRB_SCALING_GRID=256x128x32 JAX_DRB_SCALING_DEVICES=1,2,4,8,16,32 \
PYTHONPATH=src python examples/fci_sharded_strong_scaling_demo.py

# Real accelerators (one worker per GPU):
JAX_DRB_SCALING_PLATFORM=cuda JAX_DRB_SCALING_DEVICES=1,2 \
PYTHONPATH=src python examples/fci_sharded_strong_scaling_demo.py
```

It writes `output/fci_sharded_strong_scaling/scaling_<platform>.{json,png}` and
asserts the final-state checksum is invariant across device counts.

Measured CPU strong scaling (36-core host, grid `256x128x32`, one core per
shard, warmup step excluded):

| Shards | s / RK4 step | Speedup | Efficiency |
|-------:|-------------:|--------:|-----------:|
| 1  | 1.245 | 1.00x | 100% |
| 2  | 0.711 | 1.75x |  88% |
| 4  | 0.386 | 3.22x |  81% |
| 8  | 0.286 | 4.35x |  54% |
| 16 | 0.264 | 4.72x |  30% |
| 32 | 0.216 | 5.77x |  18% |

Scaling is near-linear to 4 shards and continues to gain through 8; beyond that
the per-shard subdomain (`256x128x32 / 32 ≈ 32k` cells) is too small for halo
exchange to amortize, which is the expected strong-scaling saturation point for
a fixed problem size. Larger grids push the near-linear region further right.

The example writes the scaling curve to
`output/fci_sharded_strong_scaling/scaling_cpu.png` (this repository keeps
generated figures out of git). On a single GPU the sharded step is correct but
per-step time is dispatch-bound at this problem size; multi-GPU wins require
larger subdomains or batched ensembles rather than a single small step.

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
profile evidence. The current GPU gate reaches the same residual norm as the
CPU gate and lowers peak RSS on the small fixed-layout problem, but it is not
yet claimed as a GPU speedup because this problem size is still launch- and
compile-overhead limited. The full production BDF recycling lane remains the
active target for JAX-native residual and Jacobian-action promotion.

The new batched recycling residual/JVP gate verifies the D/T/He fixed-layout
residual under `jit`, `vmap`, `jvp`, and `grad` on the real recycling state.
It uses the fixed full-field active-array RHS by default and keeps the older
host bridge only as an explicit diagnostic comparison backend.
On the local CPU run, the retained batch sweep through 256 states gives about
`4.94x` residual throughput speedup and about `3.11x` JVP
throughput speedup over serial same-kernel calls, while the
JVP/finite-difference error is about `2.19e-9`.

The adaptive-BDF recycling solver also has a bounded JAX-linearized promotion
gate. It is intentionally opt-in rather than the production default: the stable
full output-window path still uses the validated compatibility backend, while
the JAX-linearized gate checks fixed-layout residuals, JVP/Jacobian-action
solves, internal timestep control, and implicit substep convergence. The
current local single-species gate passes at `timestep=1.0` with zero fallback,
zero unconverged substeps, max accepted embedded error ratio about `0.93`, and
variable-step BDF2 history reuse after rejected-step timestep changes. On this
gate, the in-tree JAX GMRES path used `61` implicit trial solves in about
`174 s`, and the optional Lineax GMRES seam used the same controller history in
about `152 s`. The exact commands, caveats, and latest numbers are in
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
python -m build
```

Release/package documentation:

- [docs/release_packaging.md](docs/release_packaging.md)
- [docs/release_notes_1_0_2.md](docs/release_notes_1_0_2.md)
- [docs/release_notes_1_0_1.md](docs/release_notes_1_0_1.md)
- [docs/release_notes_1_0_0.md](docs/release_notes_1_0_0.md)
