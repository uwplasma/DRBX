# Geometry Roadmap

!!! note "Plan authority"
    This page is a subordinate technical appendix. The active execution plan is
    [Research-Grade Execution Plan](research_grade_execution_plan.md). If this
    roadmap conflicts with that plan, follow the execution plan and update this
    appendix afterward.

This page records how the `drbx` 3D program should stay geometry-agnostic while still shipping benchmark-specific validation packages.

For the detailed implementation plan for native non-axisymmetric stellarator SOL turbulence, including FCI metrics, equations, validation gates, and README movie targets, see [Non-Axisymmetric Stellarator SOL Implementation Plan](non_axisymmetric_stellarator_sol_plan.md).

The active execution sequence is not defined here. It is defined in
[research_grade_execution_plan.md](research_grade_execution_plan.md).
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
- summary movie and figure generation
- runtime provenance, restart, and validation artifacts

Benchmark-specific packages should only add:

- benchmark observables
- benchmark averaging windows
- benchmark methods notes and figure layouts

## Current Layers

### Reusable 3D Infrastructure

Current reusable pieces already in tree:

- [src/drbx/validation/diverted_tokamak_movie.py](../src/drbx/validation/diverted_tokamak_movie.py)
- [src/drbx/runtime/output.py](../src/drbx/runtime/output.py)
- [src/drbx/cli.py](../src/drbx/cli.py)

These pieces should be treated as the seed of a general 3D diagnostics layer, even if some names still reflect the first benchmark package built on top of them.

The current shared layer now includes:

- a reusable profile-report builder
- a reusable diagnostic-profile NPZ writer
- a reusable summary profile plotting path
- a reusable line-diagnostic builder and lineout plotting/export path

The current TCV-X21 scaffold consumes those shared pieces instead of owning a private benchmark-specific implementation.

### Current Benchmark Adapter

The first native non-axisymmetric field-line-map validation lane is now also
in tree:

- [src/drbx/geometry/stellarator.py](../src/drbx/geometry/stellarator.py)
- [src/drbx/native/fci.py](../src/drbx/native/fci.py)
- [src/drbx/native/fci_sheath_recycling.py](../src/drbx/native/fci_sheath_recycling.py)
- [src/drbx/native/fci_neutral.py](../src/drbx/native/fci_neutral.py)
- [src/drbx/native/fci_vorticity.py](../src/drbx/native/fci_vorticity.py)
- [src/drbx/native/fci_drb_rhs.py](../src/drbx/native/fci_drb_rhs.py)
- [src/drbx/validation/stellarator_fci_geometry_campaign.py](../src/drbx/validation/stellarator_fci_geometry_campaign.py)
- [src/drbx/validation/stellarator_fci_operator_campaign.py](../src/drbx/validation/stellarator_fci_operator_campaign.py)
- [src/drbx/validation/stellarator_sheath_recycling_campaign.py](../src/drbx/validation/stellarator_sheath_recycling_campaign.py)
- [src/drbx/validation/stellarator_neutral_physics_campaign.py](../src/drbx/validation/stellarator_neutral_physics_campaign.py)
- [src/drbx/validation/stellarator_vorticity_campaign.py](../src/drbx/validation/stellarator_vorticity_campaign.py)
- [src/drbx/validation/stellarator_sol_showcase.py](../src/drbx/validation/stellarator_sol_showcase.py)
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

- observable extraction for benchmark probe and target families
- detailed profile and movie products
- explicit methods notes

### Traced-Field-Line / Stellarator-Style Mesh Adapters

This is the next pressure test for whether the 3D infrastructure is really general.

Requirements:

- metric and mesh ingestion that does not assume a single tokamak benchmark layout
- explicit validation of metric fields and coordinate conventions
- field and diagnostic extraction on traced-field-line meshes
- reusable plotting on that geometry family

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
3. benchmark-observable gate
4. native execution promotion gate

No geometry family should skip directly to benchmark figures without first producing structured metadata and validation artifacts.

## Implication For The Current 3D Program

The current TCV-X21 work should continue, but only as the first adapter package on top of a general layer.

The next architectural refactor should therefore:

- separate reusable 3D diagnostics from benchmark-specific observable definitions
- add a geometry adapter interface for mesh/metric/probe extraction
- add a second and third geometry family after TCV-X21 so the abstractions get pressure-tested
- keep release claims tied to the supported geometry matrix, not to aspirational generality
