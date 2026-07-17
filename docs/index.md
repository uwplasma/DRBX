# JAXDRB Documentation

JAXDRB is a JAX-first edge and scrape-off-layer plasma toolkit for
drift-reduced Braginskii modeling, Hasegawa-Wakatani drift-wave turbulence,
linear stability analysis, differentiable reduced studies, open-field-line SOL
and neutral/detachment physics, and reusable 3D stellarator geometry.

The documentation is written for two audiences. New users should be able to
install the package, run the examples, and understand what each output file
means. Developers and reviewers should be able to trace every advertised
feature to source code, validation tests, algorithm notes, and capability
boundaries.

![Drift-wave turbulence](media/drift_wave_turbulence.gif)

![Stellarator SOL turbulence (open field lines)](media/stellarator_turbulence_open.gif)

![Stellarator 3D turbulence](media/stellarator_3d_turbulence.gif)

## Start Here

Install from PyPI:

```bash
pip install jax-drb
```

Or install from a clone:

```bash
git clone https://github.com/uwplasma/jax_drb
cd jax_drb
pip install -e .
```

Run the smallest native deck:

```bash
jax_drb inspect examples/inputs/restartable_diffusion.toml
jax_drb run examples/inputs/restartable_diffusion.toml --verbose
```

Run the flagship examples (each is a flat, self-documenting script:
imports, a PARAMETERS block, explicit setup, a run loop with progress prints,
then plotting):

```bash
PYTHONPATH=src python examples/tokamak/drift_wave_turbulence.py
PYTHONPATH=src python examples/sol/open_sol_flux_tube.py
PYTHONPATH=src python examples/benchmarks/b6_detachment_rollover.py
PYTHONPATH=src python examples/stellarator/stellarator_turbulence.py
```

Or start with the narrative walkthroughs in the **Tutorials** section:
[turbulence from zero](tutorial_hasegawa_wakatani.md),
[open SOL / neutrals / detachment](tutorial_open_sol.md), and
[stellarator FCI turbulence](tutorial_stellarator_fci.md).

## Documentation Map

| Need | Read |
| --- | --- |
| Decide what JAXDRB can run today, with links to examples, source, tests, and outputs | [Feature Reference](feature_reference.md) |
| Learn by doing, with every parameter explained | [Tutorials](tutorial_hasegawa_wakatani.md) |
| Install, run the CLI, resume runs, and inspect outputs | [Installation](installation.md), [Native Runtime CLI](native_runtime_cli.md), [Input And Output Reference](input_output_reference.md) |
| The governing equations of every shipped model | [Models And Equations](models_and_equations.md) |
| The solvers and design decisions behind the code | [Solvers And Design](solvers_and_design.md), [Physics Models](physics_models.md), [Equation To Code Map](equation_to_code_map.md) |
| Browse the example scripts | [Examples And Artifacts](examples.md) |
| Understand validation status and figures | [Validation Gallery](validation_gallery.md), [Fluid 1D MMS Convergence](fluid_1d_mms_convergence.md) |
| Work with stellarator, VMEC, imported field-line, and FCI geometry | [Stellarator Examples](stellarator_examples.md), [Connection Length](connection_length.md), [VMEC Extender Edge Fields](vmec_extender_edge_fields.md), [ESSOS Field-Line Import](essos_fieldline_import.md) |
| Understand differentiability, JAX transforms, performance, and profiling | [Performance And Differentiability](performance_and_differentiability.md), [Autodiff And Scaling Examples](autodiff_and_scaling_examples.md), [Profiling Runtime](profiling_runtime.md) |
| Understand repository size, releases, PyPI, and artifact storage | [Release Packaging](release_packaging.md) |

## What Ships

The current release includes:

| Feature family | User-facing surface |
| --- | --- |
| Native runtime | TOML decks, CLI runs, restart artifacts, structured progress logs, and Python drivers |
| 1D and compact reduced-fluid models | restartable diffusion, manufactured-solution convergence, electrostatic vorticity |
| Drift-wave turbulence | JAX-native Hasegawa-Wakatani flagship with differentiable inverse design |
| Linear stability | drift-wave, shear-Alfven, and interchange dispersion solver plus the general Jacobian engine |
| Open-field-line SOL | open slab flux tube with Bohm sheath targets, two-point steady state, sheath/recycling closure |
| Neutrals and detachment | hermes-3 AMJUEL atomic rates (packaged), recycling SOL, self-consistent detaching SOL with the SD1D rollover, gradient-based detachment control |
| 3D stellarator workflows | rotating-ellipse and island-divertor geometry, FCI 2-field/4-field/DRB models, closed vs limiter-open turbulence, imported ESSOS coil / VMEC / vmec_jax geometry |
| Differentiability | `jax.grad` through every model — sensitivity, uncertainty, inverse design, detachment control |
| Parallelism | multi-device `shard_map` FCI stepping with halo exchange, bit-exact vs single device |
| Validation | manufactured-solution convergence, geometry and operator campaigns, and publication-ready plots |

## Claim Boundaries

The documentation uses explicit capability labels so examples do not overstate
their maturity.

| Label | Meaning |
| --- | --- |
| `native_exact` | Native JAX model with exact or roundoff-level agreement on the stated validation target. |
| `native_operational` | Native and useful, with bounded documented residuals or reduced fidelity. |
| `self-contained` | Runnable from a clean clone after installation. |
| `developer/geometry-input` | Regenerates source data from heavier local geometry inputs (e.g. an ESSOS or vmec_jax checkout). |
| `opt-in research gate` | Tested enough for development and evidence collection, but not a stable default claim. |

## Literature Anchors

The model family follows standard edge/SOL reduced-fluid practice: Braginskii
collisional closures, drift-reduced ordering, parallel sheath losses, neutral
reaction-diffusion sources, and field-line-following geometry operators. Start
with [Models And Equations](models_and_equations.md) and
[Physics Models](physics_models.md) for citations, then use
[Equation To Code Map](equation_to_code_map.md) to inspect the exact source
modules and tests.

| Topic | Representative links |
| --- | --- |
| Collisional transport and reduced Braginskii models | [Braginskii 1965](https://link.springer.com/book/10.1007/978-1-4615-2808-1), [GBS code paper](https://www.sciencedirect.com/science/article/pii/S0021999116001923) |
| Flux-coordinate-independent field-line operators | [Hariri and Ottaviani 2013](https://cir.nii.ac.jp/crid/1360299150620318336), [Hariri et al. 2014](https://doi.org/10.1063/1.4892405) |
| SOL detachment (SD1D) and neutral closures | [Dudson et al. 2019](https://doi.org/10.1088/1361-6587/ab1321), [Dudson et al. 2024 (hermes-3)](https://doi.org/10.1016/j.cpc.2023.108991) |
| JAX and differentiable scientific computing | [JAX documentation](https://jax.readthedocs.io/) |
| Stellarator geometry and optimization scripting style | [SIMSOPT overview](https://simsopt.readthedocs.io/v0.18.0/overview.html), [VMEC documentation](https://princetonuniversity.github.io/STELLOPT/VMEC) |

## Repository And Media Policy

The git repository stays lightweight. Source code, examples, documentation,
tests, and compressed docs figures/movies (under `docs/media/`, sized for the
rendered docs) are tracked. Heavyweight NPZ payloads and full-resolution
legacy campaign media live in GitHub Releases; because this repository is
private, release-hosted images do not render inline for readers, so every
image embedded in these docs is a committed compressed copy. Release assets
can be restored locally with `scripts/fetch_example_artifacts.py`
(authenticate with `gh auth login` or set `GH_TOKEN`/`GITHUB_TOKEN`).
