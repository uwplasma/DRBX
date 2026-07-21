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

## Migration Rule

New or repaired FCI code must use `Rk4Stepper(rhs_fn)(state, time=...,\
timestep=..., carry=...)` and the control-volume-aware local geometry/
boundary preparation path.  Do not reintroduce `rk4_step` merely to make a
legacy test import.  Migrate that test or script to the current API, then
validate its numerical assumptions against the cut-wall representation.
