# ESSOS Imported FCI Validation

This page documents the first downstream use of externally traced
Landreman-Paul QA field lines inside `jax_drb` FCI operators. ESSOS supplies
the coil-field evaluation and adaptive trajectories. `jax_drb` converts those
trajectories into fixed-shape plane-to-plane maps, builds a lightweight
VMEC-shaped metric for the imported logical grid, and then evaluates JAX-native
sheath/recycling and neutral reaction-diffusion closures on those maps.

The published FCI validation figures and arrays are restored by
`python scripts/fetch_example_artifacts.py --skip-baselines`. Regenerating the
import from the external coil geometry is a developer workflow and requires the
geometry source checkout:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_fci_campaign.py \
  --map-source hybrid
```

Use `--all-map-sources` to regenerate the published `coil`, `vmec`, and
`hybrid` artifact directories in one run:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_fci_campaign.py \
  --all-map-sources
```

Use `--dry-run` to confirm the resolved artifact paths and grid settings
without importing ESSOS or writing files. Pass `--coil-json-path`,
`--vmec-wout-path`, or `--essos-root` when the external checkout is not located
at the default `~/local/ESSOS`. Set `--map-source` to one of three imported-map
semantics:

- `coil` traces the external Biot-Savart coil field to adjacent toroidal
  planes and keeps the resulting open-field endpoint masks.
- `vmec` evaluates a VMEC-coordinate field-line map from
  \(d\theta/d\phi=B^\theta/B^\phi\), preserving closed flux surfaces and
  disabling target endpoint masks.
- `hybrid` uses the VMEC-coordinate map locations but keeps the coil-derived
  endpoint masks, connection-length proxy, and \(|B|\) modulation. This is the
  intended bridge for open-field SOL closure tests while the VMEC map supplies
  smooth non-axisymmetric interpolation coordinates.

The command chooses source-specific defaults, so `--map-source vmec` writes
`docs/data/essos_imported_fci_vmec_artifacts/` and `--map-source hybrid` writes
`docs/data/essos_imported_fci_hybrid_artifacts/` unless `--output-root` or
`--case-label` is supplied for a single-source run.

## Geometry Import

The imported grid is a scaled VMEC Landreman-Paul QA flux-surface shell
centered on the magnetic axis reported by the external Biot-Savart field
object. The VMEC Fourier boundary is read from
`wout_LandremanPaul2021_QA_reactorScale_lowres.nc`, then rescaled and
translated onto the ESSOS coil-field coordinate system so that the rendered
surface has the QA non-axisymmetric cross-section while the traced field lines
remain in the coordinate system used by the coil JSON. The stellarator-symmetric
surface evaluation uses

\[
R(s,\theta,\phi)=\sum_{mn} R_{mn}(s)\cos(m\theta-n\phi),\qquad
Z(s,\theta,\phi)=\sum_{mn} Z_{mn}(s)\sin(m\theta-n\phi),
\]

Forward and backward coil trajectories are traced from every seed. For each
seed, the adapter interpolates the external trajectory to the adjacent toroidal
planes \(\phi\pm\Delta\phi\), projects the endpoint onto the nearest structured
VMEC-shaped target plane, and marks a boundary if the endpoint leaves the
resolved shell or lands on a radial edge. For VMEC-coordinate maps the adapter
instead integrates

\[
\frac{d\theta}{d\phi} = \frac{B^\theta(s,\theta,\phi)}{B^\phi(s,\theta,\phi)}
\]

with a fixed-step RK4 rule over one toroidal-plane spacing and stores the
resulting poloidal interpolation coordinate at fixed \(s\). Boundary map
indices are stored as finite placeholders and the boundary mask carries the
physics meaning; this keeps the JAX interpolation kernels shape-stable and
safe under `jit`, `vmap`, `jvp`, and future implicit residual promotion.

The metric is computed from the Cartesian embedding
\(\mathbf{x}(\rho,\phi,\theta)\). The covariant basis vectors are finite
differences of the scaled VMEC surface coordinates, \(g_{ij} =
\partial_i\mathbf{x}\cdot\partial_j\mathbf{x}\), \(J=\sqrt{\det g_{ij}}\), and
the contravariant metric is the matrix inverse of \(g_{ij}\). This keeps the
closure accounting on the same non-axisymmetric surface used for visualization.

## Physics Gates

The sheath/recycling gate applies a normalized Bohm target flux to every
forward or backward field-line endpoint,

\[
\Gamma_i = N_i\sqrt{(T_e+T_i)/m_i},
\]

reconstructs the electron particle flux from zero-current balance, and checks
that recycled particle and neutral-energy sources exactly close their global
accounting identities. The neutral gate then evaluates FCI parallel diffusion,
perpendicular metric diffusion, ionisation, recombination, and charge exchange
on the same imported maps. The report records endpoint fractions, magnetic
field modulation, connection-length statistics, target heat-load contrast,
particle balance residuals, current residuals, and neutral momentum balance.

## Current Artifacts

![ESSOS imported FCI coil validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_artifacts__images__essos_imported_fci_campaign.png)

![ESSOS imported FCI VMEC-coordinate validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_vmec_artifacts__images__essos_imported_fci_vmec_campaign.png)

![ESSOS imported FCI hybrid validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_hybrid_artifacts__images__essos_imported_fci_hybrid_campaign.png)

The first figure shows the default `coil` artifact: imported VMEC-shaped QA
cross-section, endpoint map structure, connection-length proxy, sheath
heat-load response, neutral ionisation response, and radial diagnostics. The
`vmec` artifact is the closed-field surface-preservation control; it has zero
target endpoint fraction and zero target heat load while still exercising
metric diffusion and neutral source accounting on the VMEC-coordinate map. The
`hybrid` artifact uses the VMEC-coordinate map positions but keeps the
coil-derived endpoint masks, connection-length proxy, and \(|B|\), making it
the preferred open-field SOL bridge. All three routes pass and feed the same
JAX-native closure kernels used by the synthetic non-axisymmetric validation
suite.

The next imported-map gate is documented in
[ESSOS imported PyTree/JVP validation](essos_imported_pytree_validation.md).
It drives the fixed-layout drift-reduced Braginskii PyTree RHS, `jax.jvp`, and
`jax.vmap` checks from the same external field-line map construction.

## Artifact Files

- `docs/data/essos_imported_fci_artifacts/data/essos_imported_fci_campaign.json`
- `docs/data/essos_imported_fci_artifacts/data/essos_imported_fci_campaign.npz`
- `docs/data/essos_imported_fci_artifacts/images/essos_imported_fci_campaign.png`
- `docs/data/essos_imported_fci_vmec_artifacts/data/essos_imported_fci_vmec_campaign.json`
- `docs/data/essos_imported_fci_vmec_artifacts/data/essos_imported_fci_vmec_campaign.npz`
- `docs/data/essos_imported_fci_vmec_artifacts/images/essos_imported_fci_vmec_campaign.png`
- `docs/data/essos_imported_fci_hybrid_artifacts/data/essos_imported_fci_hybrid_campaign.json`
- `docs/data/essos_imported_fci_hybrid_artifacts/data/essos_imported_fci_hybrid_campaign.npz`
- `docs/data/essos_imported_fci_hybrid_artifacts/images/essos_imported_fci_hybrid_campaign.png`
