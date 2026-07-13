# jax_drb

[![Tests](https://github.com/uwplasma/jax_drb/actions/workflows/test.yml/badge.svg)](https://github.com/uwplasma/jax_drb/actions/workflows/test.yml)
[![Docs](https://github.com/uwplasma/jax_drb/actions/workflows/docs.yml/badge.svg)](https://github.com/uwplasma/jax_drb/actions/workflows/docs.yml)
[![Coverage](https://github.com/uwplasma/jax_drb/actions/workflows/coverage.yml/badge.svg)](https://github.com/uwplasma/jax_drb/actions/workflows/coverage.yml)
[![PyPI](https://img.shields.io/pypi/v/jax-drb.svg)](https://pypi.org/project/jax-drb/)
[![Python](https://img.shields.io/pypi/pyversions/jax-drb.svg)](https://pypi.org/project/jax-drb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![JAX](https://img.shields.io/badge/JAX-enabled-0a9396.svg)](https://jax.readthedocs.io/)
[![Read the Docs](https://readthedocs.org/projects/jax-drb/badge/?version=latest)](https://jax-drb.readthedocs.io/)

**`jax_drb` is a JAX-based, end-to-end differentiable drift-reduced Braginskii
(DRB) code for edge and scrape-off-layer (SOL) plasma turbulence** — on both
closed and open field lines, in axisymmetric (tokamak) and non-axisymmetric
(stellarator) geometry via the flux-coordinate-independent (FCI) approach.

Because the whole model is written in JAX, every simulation is `jit`-compiled,
runs on CPU or GPU unchanged, and is differentiable: you can take gradients of
any output (a target heat flux, a detachment-front position) with respect to
any input (an anomalous diffusivity, an impurity fraction) through the solver.
To our knowledge no other published DRB SOL turbulence code is differentiable,
and none combines differentiability with FCI stellarator geometry.

Documentation: [jax-drb.readthedocs.io](https://jax-drb.readthedocs.io/).

## What it does

- Drift-reduced Braginskii models for edge/SOL turbulence: density, parallel
  momentum, and pressure evolution with Braginskii closures, vorticity/potential,
  sheath boundary conditions, and selected electromagnetic terms.
- Neutrals and recycling: an advanced-fluid-neutral model, parallel neutral
  diffusion, AMJUEL/ADAS ionization/recombination/charge-exchange rates,
  target recycling, and impurity radiation — enough for 1D detachment studies.
- Closed and open field lines in tokamak and stellarator geometry through the
  FCI map, including imported ESSOS coil, VMEC, and hybrid equilibria.
- Differentiable driver lanes: sensitivity analysis, uncertainty propagation,
  and inverse design, with autodiff derivatives checked against finite
  differences.
- A TOML-deck CLI and a small Python API, restartable runs, and portable
  JSON/NPZ output artifacts.

## Install

```bash
pip install jax-drb          # from PyPI
# or, from source:
git clone https://github.com/uwplasma/jax_drb && cd jax_drb && pip install -e .
```

Runtime dependencies are `jax`, `scipy`, `matplotlib`, `netCDF4`, `rich`, and
`pillow`. Python 3.10–3.12.

## Quick start

Run a simulation from a TOML deck, or inspect one without running it:

```bash
jax_drb inspect examples/inputs/restartable_diffusion.toml   # resolve and print the plan
jax_drb run     examples/inputs/restartable_diffusion.toml   # run and write artifacts
```

A deck declares the mesh, the model components, and the initial fields:

```toml
[time]
nout = 2
timestep = 0.1

[mesh]
nx = 32
ny = 1
nz = 1
dx = 0.03125

[model]
components = ["h"]

[species.h]
type = ["evolve_density", "evolve_pressure", "anomalous_diffusion"]

[fields.Nh]
function = { expr = "1 + 0.2 * exp(-((x-0.5)^2)/0.01)" }
```

From Python:

```python
from jax_drb.native import run_input_case

result = run_input_case("examples/inputs/restartable_diffusion.toml", case_name="demo")
print(result.time_points[-1])
```

See [docs/input_output_reference.md](docs/input_output_reference.md) for the
full deck schema and outputs, and
[docs/native_runtime_cli.md](docs/native_runtime_cli.md) for the CLI.

## Validation

`jax_drb` is validated against a ladder of literature-anchored benchmarks.
Each rung has a test (or a documented gate) and an example that regenerates
its figure.

| Case | Anchor | What is checked |
|------|--------|-----------------|
| Method of manufactured solutions | Riva et al., *Phys. Plasmas* 21, 062301 (2014); Dudson et al. 23, 062303 (2016) | operator/1D convergence order → 2 |
| Resistive drift-wave dispersion | Dudson et al., *Comput. Phys. Commun.* 180, 1467 (2009) | growth rate and frequency vs analytic dispersion |
| Shear-Alfvén wave dispersion | Stegmeir et al., *Phys. Plasmas* 26, 052517 (2019) | phase velocity vs analytic (with electron inertia) |
| Seeded blob / filament | Riva et al., *PPCF* 58, 044005 (2016) | radial velocity and velocity-vs-size scaling |
| hermes-3 component parity | Dudson et al., *CPC* 296, 108991 (2024) | per-term agreement on a documented case ladder |
| 1D detachment | Dudson et al., *PPCF* 61, 065008 (2019, SD1D); Body et al., *NME* 41, 101819 (2024) | target-flux rollover and detachment-onset scaling |
| FCI in non-axisymmetric geometry | Shanahan et al., *PPCF* 61, 025007 (2019, BSTING) | parallel-operator convergence; filament propagation |
| TCV-X21 diverted L-mode | Oliveira, Body et al., *Nucl. Fusion* 62, 096001 (2022) | agreement metric over the public dataset |

The ladder and its current status are tracked in
[`plan_jax_drb.md`](plan_jax_drb.md); benchmark reports live under
[docs/](docs/alfven_wave_benchmark.md) (Alfvén, drift-wave, MMS, neutral-mixed)
and [docs/validation_gallery.md](docs/validation_gallery.md).

## Examples

Flagship simulations span closed and open field lines in both geometries:

| | Closed field lines | Open field lines |
|---|---|---|
| **Tokamak** | [drift-wave turbulence](examples/tokamak/drift_wave_turbulence_demo.py) (Hasegawa-Wakatani; linear phase B2-verified, differentiable) | 1D SOL with sheath + recycling + reactions (detachment); [blob2d](examples/); diverted transport |
| **Stellarator** | VMEC closed-field turbulence; rotating-ellipse control | island-divertor open SOL; hybrid VMEC/coil imports |

Differentiable and geometry examples:

- Autodiff: [inverse design through turbulence](examples/tokamak/drift_wave_inverse_design_demo.py)
  (recover a parameter by gradient descent through a nonlinear drift-wave run),
  plus [sensitivity](examples/autodiff_diffusion_sensitivity_demo.py),
  [uncertainty](examples/autodiff_diffusion_uncertainty_demo.py), and reduced
  [inverse design](examples/autodiff_diffusion_inverse_design_demo.py).
- Stellarator FCI and imported geometry:
  [examples/geometry-3D/](examples/geometry-3D/).
- Start with [examples/model_selection_guide.py](examples/model_selection_guide.py)
  to choose a model family, dimension, and boundary conditions.

The user-facing examples are self-contained. Users do not need to install or
download any external plasma code to run those examples or the README/docs
movies. Live reference-code reruns are developer validation tasks only. Large
figures and movies are hosted in GitHub releases (the checkout stays small);
`python scripts/fetch_example_artifacts.py` restores them.

## Geometry and parallelization

The FCI operator and domain-decomposition stack (`FciGeometry3D`,
`fci_operators`, halo exchange) was contributed by **Aiken Xie** in
[PR #3](https://github.com/uwplasma/jax_drb/pull/3) and is incorporated here.
Multi-device `shard_map` execution built on it is completed in
[PR #5](https://github.com/uwplasma/jax_drb/pull/5), which adds a sharded
two-field step (bit-exact vs single-device) and a strong-scaling example.
See [docs/stellarator_examples.md](docs/stellarator_examples.md) and
[docs/non_axisymmetric_stellarator_sol_plan.md](docs/non_axisymmetric_stellarator_sol_plan.md).

## Documentation

- Physics and numerics: [physics_models.md](docs/physics_models.md),
  [equation_to_code_map.md](docs/equation_to_code_map.md),
  [code_structure.md](docs/code_structure.md).
- Performance and differentiability:
  [performance_and_differentiability.md](docs/performance_and_differentiability.md),
  [profiling_runtime.md](docs/profiling_runtime.md).
- Validation and parity: [validation_gallery.md](docs/validation_gallery.md),
  [parity_harness.md](docs/parity_harness.md).
- Testing policy: [testing_strategy.md](docs/testing_strategy.md).

## Testing

```bash
pytest -q -m "not slow"                                   # full fast suite
pytest -q -m "not slow" --cov=jax_drb --cov-branch        # with coverage
```

CI runs the full fast suite on Python 3.10–3.12. Reference-parity tests that
need an external hermes-3 checkout skip automatically when it is absent.

## Releases

Changes are recorded in [CHANGELOG.md](CHANGELOG.md); the current development
series is [docs/release_notes_2_0_0_dev0.md](docs/release_notes_2_0_0_dev0.md).

## Citing

If you use `jax_drb`, please cite it via [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
