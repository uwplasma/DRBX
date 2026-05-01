# JAX-DRB handoff plan for VMEC-extender edge fields

Status: deferred handoff scaffold.

This file is intentionally not a full implementation plan yet. Populate it once
the coordinated VMEC-extender work in `vmec_jax`, `virtual_casing_jax`, and
ESSOS is 100% complete against its physics, differentiability, documentation,
CLI, coverage, and benchmark acceptance gates.

## Handoff trigger

Do not start the jax_drb implementation work until the upstream extender has:

1. A validated differentiable VMEC exterior-field object using
   `B_total_out = B_coils + B_internal^VC`.
2. ESSOS `trace`, `grid`, and `validate` workflows with documented sign and
   coordinate conventions.
3. Coil-coupled LCFS normal-field cancellation tests.
4. Grid export metadata that records `nfp`, stellarator symmetry, `phi` versus
   `zeta`, units, source commits, virtual-casing branch, source resolution, and
   schedule levels.
5. External-code comparisons against SIMSOPT virtual casing and at least one
   EXTENDER/BMW-style reference where available.
6. BMW/vector-potential prototype comparisons to VCP outside the LCFS.
7. Documentation and examples stable enough for downstream users.

## Target capability

Build a new jax_drb scrape-off-layer and edge-turbulence simulation workflow
that can load and use VMEC-extender edge fields. The target imported field
source is the gridded/exported exterior field from the VMEC-extender workflow,
with support for direct callback use only after the callback API is stable.

The first downstream scenario should be a non-axisymmetric stellarator edge
case that uses:

- magnetic geometry from the VMEC-extender field export;
- open-field-line connection information from ESSOS where available;
- jax_drb edge/SOL transport operators;
- deterministic test fixtures suitable for CPU CI;
- diagnostics for field-aligned advection, parallel losses, cross-field
  diffusion, source/sink balance, and turbulence-relevant observables.

## Plan sections to populate after upstream completion

1. Upstream artifact contract
   - Required files, metadata, coordinate conventions, and unit conventions.
   - Accepted NetCDF variables and dimensions.
   - Required validation JSON fields and tolerances.

2. jax_drb imported-field API
   - Loader signatures for VMEC-extender grids.
   - PyTree/static-shape requirements.
   - Interpolation and field-period handling.
   - Differentiability expectations and non-differentiable diagnostics.

3. SOL/edge model definition
   - State variables.
   - Boundary conditions.
   - Source, sink, sheath, recycling, and turbulence-closure choices.
   - Geometry coupling and field-line metric usage.

4. Numerical implementation
   - Field-aligned operators.
   - Parallel advection/loss terms.
   - Cross-field diffusion and turbulence surrogate terms.
   - Time integration and stability constraints.

5. Validation matrix
   - Axisymmetric limit.
   - Imported-field interpolation parity.
   - Field-period symmetry.
   - Connection-length consistency with ESSOS.
   - Manufactured-solution tests for new operators.
   - Conservation and source/sink balance.

6. Examples and docs
   - Minimal VMEC-extender imported-field example.
   - Stellarator SOL turbulence demo.
   - Performance and differentiability notes.
   - Handoff instructions for generating upstream field artifacts.

7. CI and release gates
   - CPU-only smoke tests.
   - Optional larger physics cases.
   - Artifact size policy.
   - Coverage and runtime thresholds.

## Current upstream dependency status

As of this scaffold, the VMEC-extender work is not complete enough to freeze the
jax_drb interface. Keep this file as the handoff anchor and update it only when
the upstream API and exported-field contract are stable.
