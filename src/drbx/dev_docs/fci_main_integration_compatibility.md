# FCI Main-Integration Compatibility Status

This branch treats the migrated `3D_fci` implementation as the authoritative
FCI API.  It does not preserve adapters for superseded `main` FCI call paths.
The current cut-wall, aggregate-control-volume, local-gradient, and GMRES
implementation is therefore the validation target.

## Migrated And In Scope

- `drbx.geometry.fci_geometry` and its local control-volume geometry types.
- FCI boundaries, halos, operators, the ported sharded two-field RHS, and the
  shard-local full-EB RHS.
- `Rk4Stepper`, local GMRES, cut-wall slab checks, shifted-torus cut-wall MMS
  harness, and the associated developer documentation.
- `fci_sharding.make_sharded_2field_step`, migrated to construct and invoke
  `Rk4Stepper` directly.

## Retired Global Paths

The global/reference RHS, operator, boundary-payload, differentiable-case, and
stellarator-turbulence compatibility paths have been removed. The shifted-torus
two-field MMS and equivalence case now run through local geometry and
`jax.shard_map`. Model-level slab cut-wall scripts that depended on missing
helper exports were retired; the lower-level local cut-wall operator, halo,
agglomeration, and domain-decomposition tests remain the supported gates.

## Current Cut-Wall Validation Boundary

The authoritative focused contracts currently cover:

- canonical global agglomeration and translated moments;
- unique compact physical faces and periodic seams;
- direct cubic functional reproduction and diagnostics;
- owned, halo, and boundary runtime gathers;
- reverse face-halo residual accumulation;
- required use of valid direct closures by conservative compact operators.

Full repository collection is a release gate again. The seven-field EB model
is covered by `tests/test_mms_shifted_torus_EB_sharded.py`, which lowers
retained host-side `FciGeometry3D` data to `LocalFciGeometry3D` and runs the RHS
and time integration through `shard_map`. The supported reduced-model gates
are `tests/test_fci_sharded_2field.py`,
`tests/test_mms_shifted_torus_2_field.py`, and
`examples/benchmarks/fci_sharded_strong_scaling.py`. Do not restore global
compatibility aliases; new callers must use the local/sharded API.

The experimental cut-wall implementation remains available through its local
control-volume APIs and focused tests. It is independent of the production
no-cut-wall full-EB RHS.

## Migration Rule

New or repaired FCI code must use `Rk4Stepper(rhs_fn)(state, time=...,\
timestep=..., carry=...)` and the local geometry/boundary preparation path.
Cut-wall callers additionally use the local control-volume APIs. Do not
reintroduce global RHS or operator aliases merely to make a legacy caller
import; migrate the caller to the current API.
