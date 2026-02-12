# 3D examples (FCI / toroidal-plane geometry)

This folder contains **3D** examples that use field-line maps between **toroidal planes**
to evolve turbulence and validate target/sheath closures.

The key point of these examples is that the domain is genuinely **3D in toroidal space**:
each perpendicular plane is in cylindrical coordinates $(R, Z)$ at fixed toroidal angle
$\\phi$, and the model evolves a stack of planes (FCI) with a target-aware parallel
derivative.

## Examples

- `toroidal_fci_drb3d_min_movie.py`:
  minimal 3D DRB-like model $(n, \\Omega)$ on an **analytic tokamak-like toroidal field**,
  rendered as a true 3D movie (toroidal geometry) using a 3D point cloud.

## Notes

- These examples are intended to be short and reproducible (tens of seconds).
- For ESSOS-backed geometries (VMEC/coils/near-axis), see `examples/09_fci/`.

