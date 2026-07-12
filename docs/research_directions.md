# Research Directions

!!! note "Plan authority"
    This page is a subordinate research-context appendix. The active execution
    plan is [research_grade_execution_plan.md](research_grade_execution_plan.md).
    If this page conflicts with that plan, follow the execution plan and update
    this appendix afterward.

This page connects the current `jax_drb` roadmap to active edge and
scrape-off-layer research programs. The goal is to make it easy to map a code
contribution or validation campaign to a real scientific target.

## Near-Term Validation Targets

The near-term research mapping mirrors the execution order in the consolidated
plan:

1. finish the native open-field recycling transient backbone
2. reuse that backbone for integrated and direct-tokamak recycling lanes
3. run the external benchmark campaigns on the already-promoted native lanes
4. widen the selected 3D and electromagnetic validation matrix only after the recycling backbone is stable

### Seeded Blob And Filament Dynamics

Priority use case:

- TORPEX-style seeded blob and filament validation

Why it matters:

- it is one of the clearest bridges between reduced plasma turbulence models, controlled experiments, and cross-code comparison
- it gives a natural home for summary blob movies, center-of-mass diagnostics, and convergence studies

Current code surface:

- [src/jax_drb/native/blob2d.py](../src/jax_drb/native/blob2d.py)
- [src/jax_drb/validation/blob2d.py](../src/jax_drb/validation/blob2d.py)

### General 3D Geometry Infrastructure

Priority use cases:

- benchmark adapters for diverted tokamaks
- traced-field-line and stellarator-style meshes
- reusable 3D diagnostics, parity, and movie pipelines across geometry families

Why it matters:

- the 3D architecture should not be defined by a single benchmark geometry
- new research programs will need mesh and metric ingestion beyond one diverted tokamak case
- geometry portability is part of the maintainability story for researchers and graduate students

Current code surface:

- [src/jax_drb/validation/diverted_tokamak_movie.py](../src/jax_drb/validation/diverted_tokamak_movie.py)
- [src/jax_drb/validation/tokamak_tcv_x21_scaffold.py](../src/jax_drb/validation/tokamak_tcv_x21_scaffold.py)
- [src/jax_drb/validation/tokamak_tcv_x21_selected_field.py](../src/jax_drb/validation/tokamak_tcv_x21_selected_field.py)
- [docs/geometry_roadmap.md](geometry_roadmap.md)

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
- factor benchmark-specific geometry logic behind reusable 3D mesh, metric, and diagnostics layers

## External Reading And Active Context

These links are useful context for the current roadmap:

- diverted L-mode validation and benchmark culture:
  - [TCV-X21 FAIR dataset on Zenodo](https://zenodo.org/records/5776286)
  - [Validation of SOLPS-ITER simulations against the TCV-X21 reference case](https://arxiv.org/abs/2310.17390)
- detachment physics background:
  - [Physics of ultimate detachment of a tokamak divertor plasma](https://www.cambridge.org/core/product/B1A927D0F8DD3BB9C19A436C25C6FF31/core-reader)
  - [Detachment scalings derived from 1D scrape-off-layer simulations](https://arxiv.org/abs/2406.16375)
  - [SPLEND1D: a reduced one-dimensional model to investigate plasma detachment](https://arxiv.org/abs/2402.04656)
- blob and filament validation context:
  - [Blob dynamics in TORPEX poloidal null configurations](https://arxiv.org/abs/1605.00963)
- related code and benchmark context:
  - [UEDGE](https://github.com/LLNL/UEDGE)
  - [BSTING mesh/script bundle search](https://github.com/search?q=bsting_files&type=repositories)
  - [Zoidberg traced-field-line metrics branch](https://github.com/boutproject/zoidberg/tree/better-metric)
  - [Zoidberg metric pull request discussion](https://github.com/boutproject/zoidberg/pull/62)

## What The Literature Implies For `jax_drb`

The current roadmap should be interpreted conservatively:

- TORPEX-style seeded blob work is the right external benchmark for the already-strong compact electrostatic lane
- TCV-X21 is the right summary diverted benchmark after the native recycling/tokamak transient backbone is stable
- TCV-X21 should remain a benchmark adapter, not the definition of the whole 3D architecture
- traced-field-line and stellarator-style meshes should be treated as the second pressure test for the 3D infrastructure, because they force the code to separate geometry ingestion from benchmark-specific diagnostics
- 1D detachment scaling should be treated as a required summary package, not optional polish, because it stresses sources, sinks, reactions, sheath closure, restart, and scan workflows simultaneously
- 3D and broader EM claims should stay selected and benchmark-first until the native recycling/tokamak transient backbone no longer depends on replayed or scaffolded state

## What To Contribute Next

If you want to contribute to the current release program, the most useful next tasks are:

- operator-focused recycling and viscosity tests
- direct tokamak convergence campaigns
- detailed blob and divertor figures
- geometry-agnostic 3D diagnostics and metric-validation tools
- a second 3D geometry adapter beyond the current diverted tokamak benchmark
- performance and memory benchmarks for promoted native lanes
- differentiable solver-path cleanup on the strongest native-exact workflows
