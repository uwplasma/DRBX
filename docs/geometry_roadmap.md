# Geometry Roadmap

This page records how the `jax_drb` 3D program should stay geometry-agnostic while still shipping benchmark-specific validation packages.

## Design Rule

No single benchmark geometry defines the 3D architecture.

The reusable infrastructure should own:

- mesh and metric ingestion
- field-history assembly across ranks and toroidal planes
- geometry-aware probe, target, and surface extraction
- compact selected-field parity bundles
- publication-style movie and figure generation
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
- a reusable publication-style profile plotting path
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

## Required Next Geometry Families

### Diverted Tokamak Benchmark Adapters

These remain the first reviewer-facing 3D benchmark family because they connect directly to the external validation program and existing 2D closure work.

Requirements:

- reduced selected-field parity
- observable extraction for benchmark probe and target families
- publication-ready profile and movie products
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

Relevant external references:

- [BSTING mesh/script bundle search](https://github.com/search?q=bsting_files&type=repositories)
- [Zoidberg traced-field-line metrics branch](https://github.com/boutproject/zoidberg/tree/better-metric)
- [Zoidberg metric pull request discussion](https://github.com/boutproject/zoidberg/pull/62)

These references are useful because they show the kind of metric bookkeeping and traced-field-line mesh handling that a general 3D plasma-edge code needs to accommodate.

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
- add a second geometry family after TCV-X21 so the abstractions get pressure-tested
- keep publication claims tied to the supported geometry matrix, not to aspirational generality
