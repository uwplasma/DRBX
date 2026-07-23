# FCI Main-Integration Compatibility Status

This branch treats the migrated `3D_fci` implementation as the authoritative
FCI API.  It does not preserve adapters for superseded `main` FCI call paths.
The current cut-wall, aggregate-control-volume, local-gradient, and GMRES
implementation is therefore the validation target.

## Migrated And In Scope

- `drbx.geometry.fci_geometry` and its local control-volume geometry types.
- FCI boundaries, halos, operators, two-field and four-field RHS modules.
- `Rk4Stepper`, local GMRES, cut-wall slab checks, shifted-torus cut-wall MMS
  harness, and the associated developer documentation.
- `fci_sharding.make_sharded_2field_step`, migrated to construct and invoke
  `Rk4Stepper` directly.

## Deliberately Deferred Main Paths

The following inherited `main` paths still import the retired functional
`rk4_step` entry point or assume the pre-control-volume stencil contract.
They are intentionally not compatibility-adapted during this port and should
be migrated separately before being used as gates:

- `src/drbx/native/fci_differentiable_case.py`
- `tests/fci_sharded_2field_case.py`
- `tests/test_mms_shifted_torus_2_field.py`
- `tests/test_mms_shifted_torus_4_field.py`
- `tests/test_mms_shifted_torus_EB.py`
- `tests/test_shifted_torus_EB_blob.py`

The coincident and oblique slab cut-wall scripts also still call
domain-decomposition helpers through `test_mms_slab_2_field` that the module
does not export. They must be rewired to the shared helper module before those
seven tests can be restored as gates:

- `tests/test_fci_cutwall_slab_2field_physical_coincident.py`
- `tests/test_fci_cutwall_slab_2field_oblique.py`

## Current Cut-Wall Validation Boundary

The authoritative focused contracts currently cover:

- canonical global agglomeration and translated moments;
- unique compact physical faces and periodic seams;
- direct cubic functional reproduction and diagnostics;
- owned, halo, and boundary runtime gathers;
- reverse face-halo residual accumulation;
- required use of valid direct closures by conservative compact operators.

The full repository test collection is not a release gate while the deferred
callers above still import removed APIs. Do not restore compatibility aliases
only to make collection green; migrate each caller to the authoritative API.

The cut-wall implementation is ready for the forward operator convergence
sweep described in
`cutwall_numerical_problem_report.md`. It is not yet ready for the phi-solve,
full RK/MMS, or final legacy-code-removal gates.

## Migration Rule

New or repaired FCI code must use `Rk4Stepper(rhs_fn)(state, time=...,\
timestep=..., carry=...)` and the control-volume-aware local geometry/
boundary preparation path.  Do not reintroduce `rk4_step` merely to make a
legacy test import.  Migrate that test or script to the current API, then
validate its numerical assumptions against the cut-wall representation.
