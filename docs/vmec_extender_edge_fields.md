# VMEC-Extender Edge Field Import

JAXDRB can import a gridded VMEC-extender magnetic field for scrape-off-layer
geometry work without depending on VMEC, virtual-casing, or field-line tracing
packages at runtime. The import path consumes a NetCDF artifact on an
`(R, phi, Z)` grid, where `phi` is the physical toroidal angle in radians, and
loads cylindrical field components `BR`, `Bphi`, `BZ`, and `absB`.

This is a prescribed-field geometry layer, not a self-consistent SOL plasma
equilibrium. The first supported artifact is expected to describe the exterior
field region outside the VMEC last closed flux surface. In upstream
virtual-casing workflows the plasma-current contribution should use the
internal branch, because the plasma currents are inside the LCFS and the
JAXDRB import layer only receives the exported field values.

## Import Contract

The NetCDF file must define dimensions `nR`, `nphi`, and `nZ`; coordinate
variables `R`, `phi`, and `Z`; and field variables `BR`, `Bphi`, `BZ`, and
`absB`, each field shaped `(nR, nphi, nZ)`. Strict loading rejects missing
coordinate-convention metadata, coordinate conventions that mention VMEC
`zeta`, non-positive `nfp`, nonmonotone axes, inconsistent field shapes, and
large `absB` mismatches relative to `sqrt(BR**2 + Bphi**2 + BZ**2)`.

The importer wraps physical `phi` by the field period. By default the period is
`2*pi/nfp`; a positive `phi_period` attribute or explicit full-torus metadata
can override that when the artifact is exported over `2*pi`.

## Numerical Surface

After loading, the following operations are JAX-transformable:

```python
from jax_drb.geometry import (
    load_vmec_extender_grid_netcdf,
    interpolate_vmec_extender_B_cyl,
    vmec_extender_absB,
    vmec_extender_fieldline_rhs_RZ_phi,
    build_vmec_extender_fci_maps,
)

grid = load_vmec_extender_grid_netcdf("vmec_extender_field.nc")
B = interpolate_vmec_extender_B_cyl(grid, points_R_phi_Z)
absB = vmec_extender_absB(grid, points_R_phi_Z)
rhs = vmec_extender_fieldline_rhs_RZ_phi(grid, points_R_phi_Z)
maps = build_vmec_extender_fci_maps(grid)
```

The field-line RHS uses

`dR/dphi = R BR / Bphi`, `dZ/dphi = R BZ / Bphi`.

Small `Bphi` values are bounded by a sign-preserving denominator so the RHS can
remain inside compiled JAX code. File I/O and NetCDF metadata validation are
not differentiable. Interpolation is differentiable with respect to field
values and target coordinates inside a fixed interpolation cell.

## Validation Campaign

`create_vmec_extender_edge_field_campaign_package` writes a JSON summary, NPZ
arrays, and a PNG figure. The campaign checks node interpolation, midpoint
trilinear interpolation, physical-field-period wrapping, `absB` consistency,
and the field-line RHS definition. The default tests build a synthetic NetCDF
fixture and do not require external VMEC, virtual-casing, or tracing checkouts.

`create_vmec_extender_sol_smoke_package` adds the first downstream SOL coupling
gate. It loads an imported field grid, builds one-plane FCI maps, and advances
a compact scalar model with conservative field-aligned diffusion, open R-Z
perpendicular diffusion, a localized source, and edge or endpoint losses. The
campaign also runs an analytic toroidal-field diffusion decay check: for a pure
toroidal imported field, the FCI parallel operator must reproduce the exact
discrete decay rate of a single toroidal Fourier mode. This separates the
geometry/map/operator check from any claim of self-consistent edge turbulence.

Hard Poincare, wall-hit, and connection-length diagnostics remain downstream
non-smooth validation outputs rather than differentiable objectives. Once the
upstream VMEC-extender exporters are merged and their artifact contract is
stable, this layer can be used with real exterior-field grids while retaining
the same JAXDRB import and validation path.
