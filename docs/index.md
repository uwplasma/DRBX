# JAXDRB Documentation

JAXDRB is a JAX-first edge and scrape-off-layer plasma toolkit for
drift-reduced Braginskii modeling, electrostatic turbulence, neutral
transport, differentiable reduced studies, curated tokamak workflows, and
reusable 3D geometry diagnostics.

The documentation is written for two audiences. New users should be able to
install the package, restore release-backed example media, run the examples,
and understand what each output file means. Developers and reviewers should be
able to trace every advertised feature to source code, validation tests,
algorithm notes, and capability boundaries.

Large figures, movies, and reference arrays are stored in GitHub Releases, not
in git history. This keeps clone size small while still letting the docs render
engaging visual examples.

![Diverted tokamak dynamics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__movies__diverted_tokamak_turbulence.gif)

![TCV-X21 toroidal dynamics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_tcv_x21_toroidal_movie_artifacts__movies__tokamak_tcv_x21_toroidal.gif)

![Imported QA stellarator dynamics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__movies__movie_compact.gif)

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

Restore documentation media and self-contained example payloads:

```bash
python scripts/fetch_example_artifacts.py --skip-baselines
```

Use the full restore when running cached reference and parity checks:

```bash
python scripts/fetch_example_artifacts.py
```

Run the smallest native deck:

```bash
jax_drb inspect examples/inputs/restartable_diffusion.toml
jax_drb run examples/inputs/restartable_diffusion.toml --verbose
```

Run the main self-contained tutorial and movie examples:

```bash
PYTHONPATH=src python examples/restartable_diffusion_tutorial.py --quiet
PYTHONPATH=src python examples/diverted_tokamak_movie_demo.py
PYTHONPATH=src python examples/diverted_tokamak_profile_analysis_demo.py
PYTHONPATH=src python examples/geometry-3D/stellarator-fci/nonlinear_turbulence_demo.py
PYTHONPATH=src python examples/geometry-3D/stellarator-fci/turbulent_profile_analysis_demo.py
```

## Documentation Map

| Need | Read |
| --- | --- |
| Decide what JAXDRB can run today, with links to examples, source, tests, and outputs | [Feature Reference](feature_reference.md) |
| Install, run the CLI, resume runs, and inspect outputs | [Installation](installation.md), [Native Runtime CLI](native_runtime_cli.md), [Input And Output Reference](input_output_reference.md) |
| Learn the governing equations and where each term is implemented | [Physics Models](physics_models.md), [Equation To Code Map](equation_to_code_map.md) |
| Run self-contained tutorials and movie examples | [Examples And Artifacts](examples.md), [Example Status Matrix](example_status_matrix.md), [Dynamics Gallery](dynamics_gallery.md) |
| Understand validation status and publication-ready figures | [Validation Gallery](validation_gallery.md), [Research-Grade Validation Matrix](research_grade_validation_matrix.md), [Parity Matrix](parity_matrix.md) |
| Work with tokamak, stellarator, VMEC, imported field-line, and FCI geometry | [Stellarator Examples](stellarator_examples.md), [Connection Length](connection_length.md), [VMEC Extender Edge Fields](vmec_extender_edge_fields.md), [ESSOS Field-Line Import](essos_fieldline_import.md) |
| Understand differentiability, JAX transforms, performance, and profiling | [Performance And Differentiability](performance_and_differentiability.md), [Autodiff And Scaling Examples](autodiff_and_scaling_examples.md), [Profiling Runtime](profiling_runtime.md) |
| Understand repository size, releases, PyPI, and artifact storage | [Release Packaging](release_packaging.md), [Repo Size Audit](repo_size_audit.md) |

## What Ships

The current release includes:

| Feature family | User-facing surface |
| --- | --- |
| Native runtime | TOML decks, CLI runs, restart artifacts, structured progress logs, and Python drivers |
| 1D and compact reduced-fluid tests | restartable diffusion, manufactured-solution convergence, open-field operators, reaction/collision/neutral gates |
| Tokamak workflows | diverted tokamak movie/profile analysis, TCV-X21 selected-field and toroidal movie packages |
| 3D stellarator workflows | synthetic FCI geometry, reduced SOL turbulence, vorticity/bracket examples, imported coil/VMEC/hybrid geometry diagnostics |
| Differentiability | `jax.grad`, covariance pushforward, inverse design, JVP residual hooks, and opt-in fixed-layout solver seams |
| Validation and parity | cached reference comparisons, live-reference developer hooks, research-grade validation matrix, and publication-ready plots |
| Performance evidence | local CPU scaling, profiling bundles, atomic-rate throughput gates, and JAX-native residual profiling |

## Claim Boundaries

The documentation uses explicit capability labels so examples do not overstate
their maturity.

| Label | Meaning |
| --- | --- |
| `native_exact` | Promoted native compare surface with exact or roundoff-level agreement on the stated validation target. |
| `native_operational` | Native and useful, with bounded documented residuals or reduced fidelity. |
| `scaffolded_reference_backed` | Diagnostic or staging artifact backed by reference data, not a promoted native closure. |
| `self-contained` | Runnable from a clean clone after installation and, when needed, release-artifact restoration. |
| `developer/live-reference` | Regenerates source data from heavier local reference or geometry inputs. |
| `opt-in research gate` | Tested enough for development and evidence collection, but not a stable default claim. |

The stable full-output recycling path still defaults to the validated
compatibility BDF route. JAX-linearized and JVP variants are documented as
opt-in research gates until same-fidelity parity and runtime evidence are
strong enough to promote them.

## Literature Anchors

The model family follows standard edge/SOL reduced-fluid practice: Braginskii
collisional closures, drift-reduced ordering, parallel sheath losses, neutral
transport and recycling, field-line-following geometry operators, and
operator-level comparison against reference-style campaigns. Start with
[Physics Models](physics_models.md) for citations and derivations, then use
[Equation To Code Map](equation_to_code_map.md) to inspect the exact source
modules and tests.

Important external references linked throughout the docs include:

| Topic | Representative links |
| --- | --- |
| Collisional transport and reduced Braginskii models | [Braginskii 1965](https://link.springer.com/book/10.1007/978-1-4615-2808-1), [GBS code paper](https://www.sciencedirect.com/science/article/pii/S0021999116001923), [Hermes-3 paper](https://www.sciencedirect.com/science/article/pii/S0010465523003363) |
| Flux-coordinate-independent field-line operators | [Hariri and Ottaviani 2013](https://cir.nii.ac.jp/crid/1360299150620318336), [Hariri et al. 2014](https://doi.org/10.1063/1.4892405) |
| JAX and differentiable scientific computing | [JAX documentation](https://jax.readthedocs.io/), [Diffrax documentation](https://docs.kidger.site/diffrax/), [Equinox documentation](https://docs.kidger.site/equinox/) |
| Stellarator geometry and optimization scripting style | [SIMSOPT overview](https://simsopt.readthedocs.io/v0.18.0/overview.html), [VMEC documentation](https://princetonuniversity.github.io/STELLOPT/VMEC) |

## Repository And Media Policy

The git repository should remain lightweight. Source code, TOML examples,
documentation, tests, small JSON reports, and tiny fixtures are tracked. Large
simulation payloads, GIFs, release figures, and reference baselines are restored
from the release manifest with `scripts/fetch_example_artifacts.py`.

For private-repository users, authenticate with `gh auth login --hostname
github.com` or set `GH_TOKEN`/`GITHUB_TOKEN` before fetching artifacts.
