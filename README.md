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
any output (a saturated fluctuation energy, a transport level) with respect to
any input (a density gradient, an adiabaticity, a diffusivity) through the
solver. To our knowledge no other published DRB SOL turbulence code is
differentiable, and none combines differentiability with FCI stellarator
geometry.

Documentation: [jax-drb.readthedocs.io](https://jax-drb.readthedocs.io/).

## The stellarator, in 3D

Turbulence evolving on a **rotating-ellipse stellarator** — a torus whose
elliptical cross-section rotates as you follow it around, so the magnetic
geometry is genuinely three-dimensional. The cutaway shows the turbulent
density fluctuations on a flux surface and through the interior, every frame a
`jit`-compiled, differentiable JAX step:

![Stellarator turbulence in 3D](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/stellarator_3d_turbulence.gif)

The same geometry carries **closed and open field lines at once**: core field
lines wind around the torus forever, while beyond a toroidal limiter the
scrape-off-layer field lines are open — they end on the limiter, where a Bohm
sheath drains the plasma:

![Closed and open field lines in 3D](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/stellarator_3d_field_lines.png)

Watching the turbulence in the rotating cross-sections makes the difference
concrete — with the same multi-mode seed, the closed configuration conserves
its particles while the open one drains 45x more through the limiter sheath:

| Closed field lines | Open field lines (limiter SOL) |
|---|---|
| ![Stellarator turbulence, closed](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/stellarator_turbulence_closed.gif) | ![Stellarator SOL turbulence, open](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/stellarator_turbulence_open.gif) |

The non-axisymmetric machinery underneath is verified, not just rendered: the
metric comes from *automatic differentiation* of the analytic embedding, the
FCI parallel gradient converges at second order on it (direct **and**
traced-field-line operators), and a seeded filament develops the interchange
vorticity dipole in the rotating cross-sections:

| Rotating-ellipse FCI verification | Seeded filament (four-field interchange) |
|---|---|
| ![Rotating-ellipse FCI](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/rotating_ellipse_fci.png) | ![Rotating-ellipse filament](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/rotating_ellipse_filament.png) |

## The scrape-off layer: sheaths, neutrals, detachment

Open field lines end on material targets. The open SOL flux tube relaxes to the
textbook two-point steady state — the flow accelerates from stagnation to
**Mach 1 at the sheath**, the target density is half the upstream density, and
the Bohm particle balance closes exactly. Coupling the **hermes-3 neutral
model** (packaged AMJUEL ionization/recombination/charge-exchange rates,
target recycling) builds the neutral cushion at the target:

| Open SOL flux tube (two-point) | Recycling SOL (neutrals) |
|---|---|
| ![Open SOL flux tube](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/open_sol_flux_tube.png) | ![Recycling SOL](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/recycling_sol.png) |

With an *evolved* temperature (implicit Spitzer conduction, self-limiting
radiation) the SOL **detaches**: scanning the upstream density, the target
cools through 1 eV into the recombining regime and the target ion flux rolls
over — the SD1D benchmark, in a differentiable solve:

![B6 detachment rollover](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/b6_detachment.png)

## Fast, parallel, and differentiable end-to-end

Turbulence throughput is grid-size-independent per cell; one reverse-mode
gradient through a full 200-step rollout costs ~2.7x a forward run, and for a
handful of parameters forward-mode is even cheaper (~2.0x) — same gradient to
machine precision, pick the efficient method
([guide](docs/performance_and_differentiability.md)):

| Turbulence performance | Choosing a differentiation method |
|---|---|
| ![Performance](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/performance.png) | ![Differentiation methods](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/differentiation_methods.png) |

The FCI stack also runs across devices with `shard_map` — the sharded step is
bit-exact against single-device (~1e-16) and scales 1.75x/3.22x/4.35x at
2/4/8 core-bound shards on a 36-core host
([demo](examples/benchmarks/fci_sharded_strong_scaling_demo.py)). The
differentiable FCI rollout on helical geometry matches finite differences to
6e-11:

![Differentiable FCI on helical geometry](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/fci_differentiable.png)

## The classic benchmark, to close

Where it all starts: Hasegawa-Wakatani drift-wave turbulence on a periodic flux
tube, grown from noise through the linear instability (verified against the
analytic dispersion relation to ~1e-14) into nonlinear E×B transport — and
optimized *through*: gradient descent through the turbulent rollout recovers
the transport-drive parameter:

![Drift-wave turbulence](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/drift_wave_turbulence.gif)

| Linear dispersion benchmarks (B2, B3) | Inverse design *through* turbulence |
|---|---|
| ![Dispersion](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/linear_dispersion.png) | ![Inverse design](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/drift_wave_inverse_design.png) |

All figures and movies are release-hosted (not in git, so the checkout stays
small) and regenerated by the example scripts below.

## What it does

| Capability | What ships |
|---|---|
| **Turbulence models** | Hasegawa-Wakatani drift-wave (pseudo-spectral, differentiable); FCI 2-field, 4-field interchange (density/vorticity/parallel flows), and electromagnetic drift-reduced stacks with curvature and vorticity/potential closures |
| **Geometry** | Rotating-ellipse stellarator (closed core + optional limiter SOL), shifted-torus helical flux tube, open slab SOL, imported ESSOS coil / VMEC / hybrid equilibria — metrics by autodiff of analytic embeddings where analytic |
| **Field-line topology** | Closed and open field lines; FCI traced field-line maps with endpoint masks; Bohm sheath + target recycling closure on open endpoints |
| **Neutrals (hermes-3 model)** | Packaged AMJUEL ionization/recombination + charge-exchange rates (no external database); Galilean-invariant plasma↔neutral coupling; recycling SOL and a self-consistent detaching SOL (implicit Spitzer conduction, self-limiting radiation, SD1D rollover) |
| **Linear solver** | `jax_drb.linear` linearizes any model about an equilibrium → eigenmode growth rates/frequencies; drift-wave, shear-Alfvén, and interchange dispersion to machine precision |
| **Differentiability** | `jit`/`grad`/`vmap` through every model — sensitivity, uncertainty propagation, inverse design *through turbulence*; forward/reverse/checkpointed methods measured and gated to agree |
| **Parallelism** | Multi-device `shard_map` FCI stepping (bit-exact vs single device) with halo exchange; CPU strong scaling demonstrated, GPU-ready |
| **Solvers** | Structured solves via [`solvax`](https://github.com/uwplasma/SOLVAX) (spectral Fourier-Helmholtz elliptic, tridiagonal, Krylov, preconditioners) |
| **Runtime** | TOML-deck CLI (`jax_drb inspect` / `run`) and a small Python API; restartable runs; portable JSON/NPZ artifacts |

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

Verified today (each with a passing test):

| Case | Anchor | What is checked |
|------|--------|-----------------|
| Method of manufactured solutions | Riva et al., *Phys. Plasmas* 21, 062301 (2014); Dudson et al. 23, 062303 (2016) | operator / 1D-fluid / FCI convergence order → 2 |
| Resistive drift-wave dispersion | Dudson et al., *Comput. Phys. Commun.* 180, 1467 (2009) | growth rate and frequency vs analytic dispersion |
| Shear-Alfvén wave dispersion | Stegmeir et al., *Phys. Plasmas* 26, 052517 (2019) | phase velocity vs analytic (with electron inertia) |
| Interchange / Rayleigh-Taylor | curvature-driven flute dispersion | growth rate vs `√(gκ)·k_y/k` analytic |
| FCI on non-axisymmetric geometry | Shanahan et al., *PPCF* 61, 025007 (2019, BSTING) | parallel-operator MMS; differentiable rollout (grad vs FD 6e-11) |
| Rotating-ellipse (`l = 2`) FCI | Stegmeir et al., *Comput. Phys. Commun.* 198, 139 (2016, GRILLIX) | direct & traced-field-line parallel gradient converge at order 2 on a genuinely non-axisymmetric metric; shape-differentiable; a seeded four-field filament generates interchange vorticity on the rotating surfaces |
| Island-divertor field (B8) | Shanahan et al., *J. Plasma Phys.* 90 (2024, BSTING); GBS island-divertor studies | sheared-iota island chains + stochastic edge; closed core and finite-connection-length open SOL emerge from multi-transit tracing; turbulence drains through the emergent divertor masks |
| Open-field-line SOL flux tube | two-point / Bohm-sheath SOL theory (Stangeby, *The Plasma Boundary of Magnetic Fusion Devices*, 2000) | parallel flow reaches Mach 1 at the targets; target density = half upstream; exact Bohm particle balance and sheath-recycling accounting |
| Neutrals and recycling (hermes-3 model) | hermes-3: Dudson et al., *Comput. Phys. Commun.* 296, 108991 (2024); AMJUEL atomic rates | physically-correct ionization/recombination/CX rates; exact plasma↔neutral particle & momentum conservation; neutrals conserve on the 3D closed rotating ellipse and recycle on the open slab |
| SD1D detachment rollover (B6) | SD1D: Dudson et al., *Plasma Phys. Control. Fusion* 61, 065008 (2019) | self-consistent SOL (evolved temperature, implicit Spitzer conduction, self-limiting radiation): the target cools through 1 eV into the recombining regime and the target ion flux rolls over as upstream density rises; differentiable |
| Differentiable inverse design | — | gradient descent through turbulence recovers a drive parameter |

Planned rungs (seeded-blob inertial scaling and others) are
tracked in [`plan_jax_drb.md`](plan_jax_drb.md); benchmark reports live under
[docs/](docs/linear_dispersion_benchmark.md) and
[docs/validation_gallery.md](docs/validation_gallery.md).

## Examples

Flagship simulations, by geometry:

| | Turbulence flagship | Geometry |
|---|---|---|
| **Tokamak** | [drift-wave turbulence](examples/tokamak/drift_wave_turbulence_demo.py) (Hasegawa-Wakatani; linear phase B2-verified, differentiable) + [inverse design](examples/tokamak/drift_wave_inverse_design_demo.py) | periodic flux tube |
| **Stellarator** | [turbulence on closed + open field lines](examples/stellarator/stellarator_turbulence_demo.py) (four-field, limiter SOL, movies) + [3D renders](examples/stellarator/stellarator_3d_render_demo.py) (cutaway turbulence movie, field-line topology) + [island divertor](examples/stellarator/island_divertor_demo.py) (B8: Poincare, connection lengths, emergent open SOL) + [rotating-ellipse FCI](examples/stellarator/rotating_ellipse_fci_demo.py) (parallel-operator convergence) + [seeded filament](examples/stellarator/rotating_ellipse_filament_demo.py) + [differentiable FCI drift-reduced model](examples/stellarator/fci_differentiable_demo.py) | rotating ellipse (closed core + limiter SOL) + shifted-torus helical + imported [ESSOS/VMEC](examples/geometry-3D/) |
| **SOL (open)** | [open SOL flux tube](examples/sol/open_sol_flux_tube_demo.py) (parallel transport to Bohm-sheath targets; two-point steady state) + [recycling SOL](examples/sol/recycling_sol_demo.py) (neutrals, ionization/recombination, detachment onset) | open slab flux tube |

Open-field-line SOL:
[open slab flux tube](examples/sol/open_sol_flux_tube_demo.py) — parallel
transport to Bohm-sheath-bounded targets, relaxing to the classic two-point
steady state (Mach 1 at the targets, target density half the upstream density),
with the FCI sheath/recycling closure on the target plates.

Benchmarks, differentiable, and geometry examples:

- Linear dispersion (B2/B3):
  [examples/benchmarks/linear_dispersion_demo.py](examples/benchmarks/linear_dispersion_demo.py)
  reproduces the drift-wave and shear-Alfvén dispersion relations from the
  linear solver.
- Autodiff: [inverse design through turbulence](examples/tokamak/drift_wave_inverse_design_demo.py)
  (recover a parameter by gradient descent through a nonlinear drift-wave run),
  [choosing the most efficient differentiation method](examples/autodiff/differentiation_methods_demo.py)
  (forward vs reverse vs checkpointed reverse — same gradient, different cost),
  plus [sensitivity](examples/autodiff_diffusion_sensitivity_demo.py),
  [uncertainty](examples/autodiff_diffusion_uncertainty_demo.py), and reduced
  [inverse design](examples/autodiff_diffusion_inverse_design_demo.py).
- Stellarator FCI and imported geometry:
  [examples/geometry-3D/](examples/geometry-3D/).
- Start with [examples/model_selection_guide.py](examples/model_selection_guide.py)
  to choose a model family, dimension, and boundary conditions.

The examples are self-contained — no external plasma code is needed to run
them. Large figures and movies are hosted in GitHub releases so the checkout
stays small.

## Geometry and parallelization

The FCI operator and domain-decomposition stack (`FciGeometry3D`,
`fci_operators`, halo exchange) was contributed by **Aiken Xie** in
[PR #3](https://github.com/uwplasma/jax_drb/pull/3) and is incorporated here.
Built on it, the drift-reduced two-field step runs across multiple devices with
`shard_map`: the domain is decomposed into halo-exchanged shards and the sharded
RK4 step is **bit-exact** against the single-device step (checked to ~1e-16 for
single-device and forced-four-device runs in
[`tests/test_fci_sharded_2field.py`](tests/test_fci_sharded_2field.py)). On a
36-core Linux host with one core bound per shard, a `256x128x32` step scales
**1.75x at 2 shards, 3.22x at 4, 4.35x at 8**
([strong-scaling demo](examples/benchmarks/fci_sharded_strong_scaling_demo.py),
[docs](docs/performance_and_differentiability.md)).

## Documentation

- Physics and numerics: [physics_models.md](docs/physics_models.md),
  [equation_to_code_map.md](docs/equation_to_code_map.md),
  [code_structure.md](docs/code_structure.md).
- Performance and differentiability:
  [performance_and_differentiability.md](docs/performance_and_differentiability.md),
  [profiling_runtime.md](docs/profiling_runtime.md).
- Validation: [validation_gallery.md](docs/validation_gallery.md).
- Testing policy: [testing_strategy.md](docs/testing_strategy.md).

## Testing

```bash
pytest -q -m "not slow"                                   # full fast suite
pytest -q -m "not slow" --cov=jax_drb --cov-branch        # with coverage
```

CI runs the full fast suite on Python 3.10–3.12.

## Releases

Changes are recorded in [CHANGELOG.md](CHANGELOG.md); the current development
series is [docs/release_notes_2_0_0_dev0.md](docs/release_notes_2_0_0_dev0.md).

## Citing

If you use `jax_drb`, please cite it via [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
