# jax_drb (Unified DRB)

This is a **fresh rewrite** of `jax_drb` centered on a **single unified drift-reduced Braginskii system**.
All physics variants (ES/EM, hot/cold ions, sheath/no-sheath, Boussinesq/non-Boussinesq, 1D/2D/3D)
are controlled **only by toggles and geometry adapters**. There are no separate model branches.

## Quick Start

### Run via TOML
```bash
jaxdrb path/to/input.toml
```

### Example TOML
```toml
[geometry]
kind = "plane"         # plane | line | fci
nx = 64
ny = 64
Lx = 6.283185
Ly = 6.283185

[physics]
em_on = false
hot_ion_on = false
nonlinear_on = true
boussinesq = true

[numerics]
bracket = "arakawa"
poisson = "spectral"

[closures]
sheath_on = false
```

## Normalization (Physical Inputs)
You can supply physical parameters and let `jaxdrb` normalize them for you. See
`/Users/rogerio/local/jax_drb/docs/normalization.md` for details.

```toml
[normalization]
enabled = true
mode = "physics"
Te0_eV = 50.0
Ti0_eV = 50.0
n0 = 1e19
B0 = 2.0
m_i_amu = 2.0
Z_i = 1
length_unit = "rho_s"

[geometry_physical]
Lx = 0.1
Ly = 0.1
Lz = 6.283185
R0 = 2.0
r0 = 0.2
B0 = 2.0

[physics_physical]
omega_n = 20.0
```

### CLI Example (Normalization Enabled)
```bash
jaxdrb /path/to/salpha_physical.toml
```

```toml
[normalization]
enabled = true
mode = "physics"
Te0_eV = 40.0
Ti0_eV = 40.0
n0 = 2e19
B0 = 2.0
m_i_amu = 2.0
Z_i = 1
length_unit = "rho_s"

[geometry]
kind = "axisymmetric_analytic"
model = "salpha"
nx = 32
ny = 32
nz = 32

[geometry_physical]
Lx = 0.12
Ly = 0.12
Lz = 6.283185
R0 = 2.0
r0 = 0.2
B0 = 2.0

[physics_physical]
omega_n = 15.0

[transport_physical]
Dn = 0.3
```

## Status
- Core system + geometry adapters are in `src/jaxdrb/core`.
- A new CLI lives in `src/jaxdrb/cli/main.py`.
- Legacy code moved to `legacy/`.

## Geometry Comparisons
Use the helper scripts in `tools/` to compare analytic geometry against external grids:
`/Users/rogerio/local/jax_drb/docs/geometry_compare.md`.

## Geometry Models
Analytic geometry models (s-alpha, Miller, X-point) and curvature definitions are documented here:
`/Users/rogerio/local/jax_drb/docs/geometry_models.md`.

## Next Steps
- Full config schema & validation
- Diffrax integration driver
- Unified diagnostics + plotting helpers
- New tests and benchmarks from scratch
