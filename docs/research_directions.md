# Research Directions

This page connects the current `jax_drb` roadmap to active edge and scrape-off-layer research programs. The goal is to make it easy to map a code contribution or validation campaign to a real scientific target.

## Near-Term Validation Targets

### Seeded Blob And Filament Dynamics

Priority use case:

- TORPEX-style seeded blob and filament validation

Why it matters:

- it is one of the clearest bridges between reduced plasma turbulence models, controlled experiments, and cross-code comparison
- it gives a natural home for reviewer-facing blob movies, center-of-mass diagnostics, and convergence studies

Current code surface:

- [examples/blob2d_meeting_demo.py](../examples/blob2d_meeting_demo.py)
- [src/jax_drb/native/blob2d.py](../src/jax_drb/native/blob2d.py)
- [src/jax_drb/validation/blob2d.py](../src/jax_drb/validation/blob2d.py)

### Diverted Tokamak Validation

Priority use case:

- TCV-X21-style diverted L-mode validation

Why it matters:

- it provides a credible external validation target beyond curated internal benchmark ladders
- it exercises turbulence, transport, divertor physics, and geometry handling in a reviewer-relevant setting

Current code surface:

- [src/jax_drb/native/runner.py](../src/jax_drb/native/runner.py)
- [src/jax_drb/native/recycling_1d.py](../src/jax_drb/native/recycling_1d.py)
- [docs/research_grade_validation_matrix.md](research_grade_validation_matrix.md)

### Detachment And Divertor Scaling

Priority use case:

- scripted 1D and open-field detachment-scaling campaigns

Why it matters:

- detachment and divertor closure are central edge/SOL review topics
- they force the code to handle sources, sinks, boundary conditions, and restartable parameter scans cleanly

Current code surface:

- [src/jax_drb/native/recycling_1d.py](../src/jax_drb/native/recycling_1d.py)
- [examples/restartable_diffusion_tutorial.py](../examples/restartable_diffusion_tutorial.py)

## Medium-Term Research Lanes

- impurity and radiation workflows on curated open-field and tokamak geometries
- X-point and divertor-instability studies
- electrostatic and electromagnetic transport closure studies
- differentiable inverse and design loops on compact native-exact lanes

## Architecture Priorities That Support Research

To stay maintainable for researchers and graduate students, the highest-value architectural work is:

- finish one fully native open-field recycling transient backbone
- reduce NumPy/SciPy-only barriers on promoted paths
- keep runtime/output/restart behavior uniform across promoted cases
- keep capability tiers explicit in docs, logs, and validation reports
- make plotting/movie scripts part of the normal workflow rather than post-hoc notebooks

## External Reading And Active Context

These links are useful context for the current roadmap:

- diverted L-mode validation and benchmark culture:
  - [TCV-X21 FAIR dataset on Zenodo](https://zenodo.org/records/5776286)
  - [Validation of SOLPS-ITER simulations against the TCV-X21 reference case](https://arxiv.org/abs/2310.17390)
- detachment physics background:
  - [Physics of ultimate detachment of a tokamak divertor plasma](https://www.cambridge.org/core/product/B1A927D0F8DD3BB9C19A436C25C6FF31/core-reader)
- blob and filament validation context:
  - [Blob dynamics in the TORPEX experiment: a multi-code validation](https://graphsearch.epfl.ch/fr/publication/e67955ef-d60f-4897-8271-fde5dfb50c2e)
  - [Blob dynamics in TORPEX poloidal null configurations](https://arxiv.org/abs/1605.00963)

## What To Contribute Next

If you want to contribute to the current release program, the most useful next tasks are:

- operator-focused recycling and viscosity tests
- direct tokamak convergence campaigns
- publication-ready blob and divertor figures
- performance and memory benchmarks for promoted native lanes
- differentiable solver-path cleanup on the strongest native-exact workflows
