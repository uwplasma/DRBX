# Code Structure

!!! note "Plan authority"
    This page is a developer map and structural context appendix. The active
    execution plan is
    [Research-Grade Execution Plan](research_grade_execution_plan.md). If this
    page conflicts with that plan, follow the execution plan and update this
    page afterward.

This page is the developer-facing map of the `jax_drb` source tree. The goal is
to make the package understandable before reading the solver files or the
validation campaigns in detail.

The architecture follows standard edge-code practice: separate the governing
operators from the orchestration layer, separate verification from benchmark
validation, and keep geometry, numerics, and plotting reusable.

The validation package carries a shared publication-plot helper in
[src/jax_drb/validation/publication_plotting.py](../src/jax_drb/validation/publication_plotting.py).
That helper is part of the research-grade validation surface: the figure
standard lives next to the tested campaigns, not only in downstream paper
scripts.

## Package Map

The current top-level layout is:

- `src/jax_drb/native`
  native solvers and problem-family implementations (Hasegawa-Wakatani,
  the FCI operator stack, 1-D fluid/diffusion/vorticity/electromagnetic, and the
  deck runner)
- `src/jax_drb/linear`
  the linear stability / dispersion solver
- `src/jax_drb/geometry`
  structured, analytic-stellarator, FCI, imported field-line, and VMEC-extender
  geometry
- `src/jax_drb/validation`
  benchmark campaigns, geometry diagnostics, plots, and publication-oriented
  artifacts
- `src/jax_drb/config`
  structured input-deck parsing and numeric option resolution
- `src/jax_drb/runtime`
  runtime configuration, precision, profiling, artifacts, and execution helpers

The command-line entry points are `src/jax_drb/cli.py` and
`src/jax_drb/__main__.py`.

## Current Responsibilities

The native solver families are:

- `hasegawa_wakatani.py`
  the JAX-native 2-D Hasegawa-Wakatani drift-wave turbulence flagship, with
  differentiable inverse design
- the FCI stack:
  `fci_operators.py` (parallel/perpendicular gradient and Laplacian stencils on
  the field-line maps), `fci_boundaries.py`, `fci_halo.py`, `fci_2_field_rhs.py`
  and `fci_4_field_rhs.py` (reduced models), `fci_drb_EB_rhs.py` and
  `fci_drb_rhs.py` (drift-reduced Braginskii right-hand sides),
  `fci_vorticity.py` (perpendicular vorticity inversion),
  `fci_sheath_recycling.py` (3-D FCI Bohm-sheath target closure),
  `fci_neutral.py` (neutral reaction-diffusion), and
  `fci_time_integrator.py` (RK4)
- `fluid_1d.py`
  compact manufactured-solution and differentiable verification lane
- `transport.py`, `vorticity.py`, `electromagnetic.py`
  the anomalous-diffusion, electrostatic-vorticity, and reduced electromagnetic
  families
- `deck_runner.py`
  deck resolution, native run execution, restart orchestration, and portable
  summary/array artifact writing for the `jax_drb run` command

The `linear/` package holds the general Jacobian/eigenmode engine (`eigen.py`)
and the three reduced dispersion operators (`dispersion.py`).

The validation layer contains four kinds of modules, although they are not yet
split cleanly on disk:

- campaign builders (FCI operator/geometry/suite, ESSOS- and VMEC-imported
  geometry, stellarator SOL, autodiff diffusion)
- geometry adapters and diagnostics (`geometry_lineouts.py`,
  `geometry_slices.py`)
- plotting/report helpers
- publication-facing summary packages (`publication_plotting.py`)

## Structure And Direction

The active plan (`plan_jax_drb.md`, repository root) keeps the code focused on
the accuracy-tested core: the compact native deck models, the Hasegawa-Wakatani
flagship, the FCI operator stack on tokamak and non-axisymmetric geometry, the
linear dispersion solver, and the imported-geometry adapters. New physics is
added by reusing the shared operator, mesh/metric, and geometry layers rather
than by adding standalone solver paths, and each new branch is expected to land
with operator/boundary unit tests and at least one physics-facing diagnostic
before it is treated as accuracy-tested.

## JAX Boundary

The architecture keeps the JAX boundary explicit:

- the compact verification, reduced-operator, Hasegawa-Wakatani, and FCI
  drift-reduced lanes are JAX-native and are appropriate for `jit`, `vmap`,
  `grad`, `jvp`, and `vjp`;
- the CLI, deck parsing, file I/O, and output serialization are ordinary
  NumPy/SciPy boundary code and are documented and tested as such rather than
  marketed as end-to-end differentiable.

This distinction matches the current boundary between purely differentiable
JAX-native workflows and the surrounding host-side orchestration.
