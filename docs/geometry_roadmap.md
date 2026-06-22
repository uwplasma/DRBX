# Geometry Roadmap

!!! note "Plan authority"
    This page is a subordinate technical appendix. The active execution plan is
    [Research-Grade Execution Plan](research_grade_execution_plan.md). If this
    roadmap conflicts with that plan, follow the execution plan and update this
    appendix afterward.

This page records how the `jax_drb` 3D program should stay geometry-agnostic while still shipping benchmark-specific validation packages.

For the detailed implementation plan for native non-axisymmetric stellarator SOL turbulence, including FCI metrics, equations, validation gates, and README movie targets, see [Non-Axisymmetric Stellarator SOL Implementation Plan](non_axisymmetric_stellarator_sol_plan.md).

The active execution sequence is not defined here. It is defined in
[research_grade_execution_plan.md](research_grade_execution_plan.md#current-authoritative-open-lane-implementation-plan).
As of the June 22, 2026 plan consolidation, the active 3D sequence is:
direct ESSOS-coil open-field diagnostics and direct-coil closed controls on
`main`, VMEC closed-field controls, hybrid VMEC/coil open-SOL promotion, then
VMEC-extender finite-beta frozen-artifact intake. The adapter architecture on
this page supports that sequence but does not reorder it.

## Design Rule

No single benchmark geometry defines the 3D architecture.

The reusable infrastructure should own:

- mesh and metric ingestion
- field-history assembly across ranks and toroidal planes
- geometry-aware probe, target, and surface extraction
- compact selected-field parity bundles
- summary movie and figure generation
- runtime provenance, restart, and validation artifacts

Benchmark-specific packages should only add:

- benchmark observables
- benchmark compare surfaces
- benchmark averaging windows
- benchmark methods notes and figure layouts

## Current Layers

### Reusable 3D Infrastructure

Current reusable pieces already in tree:

- [src/jax_drb/validation/diverted_tokamak_movie.py](../src/jax_drb/validation/diverted_tokamak_movie.py)
- [src/jax_drb/validation/tokamak_tcv_x21_selected_field.py](../src/jax_drb/validation/tokamak_tcv_x21_selected_field.py)
- [src/jax_drb/runtime/output.py](../src/jax_drb/runtime/output.py)
- [src/jax_drb/cli.py](../src/jax_drb/cli.py)

These pieces should be treated as the seed of a general 3D diagnostics layer, even if some names still reflect the first benchmark package built on top of them.

The current shared layer now includes:

- a reusable profile-report builder
- a reusable diagnostic-profile NPZ writer
- a reusable summary profile plotting path
- a reusable line-diagnostic builder and lineout plotting/export path

The current TCV-X21 scaffold consumes those shared pieces instead of owning a private benchmark-specific implementation.

### Current Benchmark Adapter

The current 3D benchmark adapter is the TCV-X21 scaffold package:

- [src/jax_drb/validation/tokamak_tcv_x21_scaffold.py](../src/jax_drb/validation/tokamak_tcv_x21_scaffold.py)
- [docs/tokamak_tcv_x21_scaffold_demo.md](tokamak_tcv_x21_scaffold_demo.md)
- [docs/tokamak_tcv_x21_selected_field_demo.md](tokamak_tcv_x21_selected_field_demo.md)

That package is useful and should stay, but it is an adapter, not the architecture.

The first second-adapter scaffold is now also in tree:

- [src/jax_drb/validation/traced_field_line_scaffold.py](../src/jax_drb/validation/traced_field_line_scaffold.py)
- [docs/traced_field_line_scaffold_demo.md](traced_field_line_scaffold_demo.md)

It is intentionally lighter than the TCV package. Its purpose is to pressure-test
the general geometry and diagnostics layer on a non-diverted geometry family
before a real external traced-field-line mesh is wired in.

An older Fourier-equilibrium scaffold also remains in tree. It pressure-tests
the same public 3D artifact model on a stellarator source, so the general layer
is forced to support sampled flux-surface figures and movies as well as
profile bundles.

The first native non-axisymmetric field-line-map validation lane is now also
in tree:

- [src/jax_drb/geometry/stellarator.py](../src/jax_drb/geometry/stellarator.py)
- [src/jax_drb/native/fci.py](../src/jax_drb/native/fci.py)
- [src/jax_drb/native/fci_sheath_recycling.py](../src/jax_drb/native/fci_sheath_recycling.py)
- [src/jax_drb/native/fci_neutral.py](../src/jax_drb/native/fci_neutral.py)
- [src/jax_drb/native/fci_vorticity.py](../src/jax_drb/native/fci_vorticity.py)
- [src/jax_drb/native/fci_drb_rhs.py](../src/jax_drb/native/fci_drb_rhs.py)
- [src/jax_drb/validation/stellarator_fci_geometry_campaign.py](../src/jax_drb/validation/stellarator_fci_geometry_campaign.py)
- [src/jax_drb/validation/stellarator_fci_operator_campaign.py](../src/jax_drb/validation/stellarator_fci_operator_campaign.py)
- [src/jax_drb/validation/stellarator_sheath_recycling_campaign.py](../src/jax_drb/validation/stellarator_sheath_recycling_campaign.py)
- [src/jax_drb/validation/stellarator_neutral_physics_campaign.py](../src/jax_drb/validation/stellarator_neutral_physics_campaign.py)
- [src/jax_drb/validation/stellarator_vorticity_campaign.py](../src/jax_drb/validation/stellarator_vorticity_campaign.py)
- [src/jax_drb/validation/stellarator_sol_showcase.py](../src/jax_drb/validation/stellarator_sol_showcase.py)
- [docs/stellarator_fci_validation.md](stellarator_fci_validation.md)

That lane is the first actual native non-axisymmetric execution path in this
roadmap. It validates full metric tensors, traced field-line maps,
conservative operators, traced-endpoint sheath/recycling balance, neutral
reaction/diffusion conservation, vorticity inversion, and a reduced 3D SOL
dynamics artifact before any device-specific claim is made.

## Required Next Geometry Families

### Diverted Tokamak Benchmark Adapters

These remain the first summary 3D benchmark family because they connect directly to the external validation program and existing 2D closure work.

Requirements:

- reduced selected-field parity
- observable extraction for benchmark probe and target families
- detailed profile and movie products
- explicit methods notes and compare-surface metadata

### Traced-Field-Line / Stellarator-Style Mesh Adapters

This is the next pressure test for whether the 3D infrastructure is really general.

Requirements:

- metric and mesh ingestion that does not assume a single tokamak benchmark layout
- explicit validation of metric fields and coordinate conventions
- field and diagnostic extraction on traced-field-line meshes
- reusable plotting and parity bundles on that geometry family

The current scaffold now already supports both:

- synthetic JSON mesh specifications for lightweight preview artifacts
- NetCDF FCI grids for real external metric bundles

The external implementation pattern to match is metric-first: every imported
map bundle should carry enough covariant/contravariant metric data, map masks,
connection-length metadata, and boundary distances to reproduce the same gates
as the analytic native lane.

### Stellarator Equilibrium Adapters

This is the next pressure test for whether the 3D infrastructure can support
equilibrium-driven geometry families that are not naturally framed as tokamak
benchmark workdirs or FCI metric grids.

Requirements:

- equilibrium/profile ingestion from Fourier-equilibrium NetCDF data
- sampled flux-surface cross-sections and geometry movies on the shared artifact model
- explicit validation of finite profiles and finite sampled surfaces
- observable and provenance reporting on the same adapter schema used elsewhere

## Validation Gates

Every new geometry family should pass the same staged gate sequence:

1. geometry scaffold gate
2. external-workdir artifact gate
3. reduced selected-field parity gate
4. benchmark-observable gate
5. native execution promotion gate

No geometry family should skip directly to benchmark figures without first producing structured metadata, compact compare surfaces, and parity artifacts.

## Implication For The Current 3D Program

The current TCV-X21 work should continue, but only as the first adapter package on top of a general layer.

The next architectural refactor should therefore:

- separate reusable 3D diagnostics from benchmark-specific observable definitions
- add a geometry adapter interface for mesh/metric/probe extraction
- add a second and third geometry family after TCV-X21 so the abstractions get pressure-tested
- keep release claims tied to the supported geometry matrix, not to aspirational generality
