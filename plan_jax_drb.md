# JAX-DRB handoff plan for VMEC-extender edge fields

Status: implemented first imported-field layer on `feature/vmec-extender-edge-fields`.

Last updated: 2026-05-05.

Implementation update:

- Added `src/jax_drb/geometry/vmec_extender_import.py` with strict NetCDF
  metadata validation, physical-phi periodic interpolation, `absB`,
  field-line RHS, and one-plane FCI map construction.
- Added synthetic numerical tests in `tests/test_vmec_extender_import.py`.
- Added `src/jax_drb/validation/vmec_extender_edge_field_campaign.py` plus
  synthetic campaign tests and public JSON/NPZ/PNG artifact generation.
- Added `docs/vmec_extender_edge_fields.md` and
  `examples/geometry-3D/vmec-extender/imported_field_demo.py`.
- Registered the public docs/example/modules in release-surface tests and the
  focused CI test workflow.
- Verified the implementation with the focused VMEC-extender/release-surface
  checks and the full local test suite (`956 passed, 30 skipped, 1 xfailed`).
- Added the next local SOL coupling gate in
  `src/jax_drb/validation/vmec_extender_sol_smoke_campaign.py`: imported FCI
  maps drive a compact scalar model with conservative field-aligned diffusion,
  open R-Z perpendicular diffusion, localized source, edge/endpoint loss, and
  an analytic toroidal-field Fourier-mode decay check.

This document is the handoff for the `jax_drb` agent that will start the
scrape-off-layer and edge-turbulence work using VMEC-extender fields. The
upstream VMEC-extender PRs are green but not merged yet, so the first
`jax_drb` implementation must target the exported grid artifact contract, not a
direct Python callback into the upstream field object.

## Current branch and PR state

Use this local checkout and branch:

```bash
cd /Users/rogerio/local/jax_drb
git checkout feature/vmec-extender-edge-fields
```

Draft PR:

- `uwplasma/jax_drb#2`
- URL: `https://github.com/uwplasma/jax_drb/pull/2`
- current head when this handoff was written: `9c6e33e`
- state: draft, mergeable/clean, hosted checks green

Companion upstream PRs:

- `uwplasma/virtual_casing_jax#2`
  - branch: `/Users/rogerio/local/virtual_casing_jax`, `feature/jax-vmec-extender`
  - current green head at handoff time: `f7a4b85`
  - owns VMEC-extender field object, grid export, objectives, benchmarks
- `uwplasma/ESSOS#31`
  - branch: `/Users/rogerio/local/ESSOS-vmec-extender`, `feature/jax-vmec-extender`
  - checks green; blocked by review requirement
  - owns tracing, Poincare/connection diagnostics, CLI workflows
- `uwplasma/vmec_jax#12`
  - branch: `/Users/rogerio/local/vmec_jax`, `feature/jax-vmec-extender`
  - checks green
  - owns any small VMEC helper surface/field APIs

## Handoff prompt for the jax_drb agent

Use this prompt verbatim or near-verbatim for the next agent:

```text
You are working in /Users/rogerio/local/jax_drb on branch
feature/vmec-extender-edge-fields.

Goal: implement the first jax_drb integration layer for VMEC-extender edge
fields. Start with a gridded NetCDF import path from virtual_casing_jax, not a
direct vmec_jax/ESSOS callback. The output should let jax_drb load an
R-phi-Z exterior magnetic-field grid, interpolate B in cylindrical components,
build field-line RHS data for FCI/open-field-line operators, and run small
CPU-friendly validation tests and a campaign artifact.

Do not reimplement VMEC or virtual casing in jax_drb. Treat the upstream field
artifact as the source of truth. Treat phi-vs-zeta and field-period wrapping as
high-risk. Add real numerical tests, not scaffold-only tests.

Primary files to add:
- src/jax_drb/geometry/vmec_extender_import.py
- tests/test_vmec_extender_import.py
- src/jax_drb/validation/vmec_extender_edge_field_campaign.py
- tests/test_validation_vmec_extender_edge_field_campaign.py
- examples/geometry-3D/vmec-extender/imported_field_demo.py
- docs/vmec_extender_edge_fields.md

Update:
- src/jax_drb/geometry/__init__.py
- src/jax_drb/validation/__init__.py
- tests/test_release_surface.py
- README.md or docs index if this repo requires release-surface registration

Acceptance:
- Synthetic NetCDF import and interpolation tests pass.
- Physical-phi periodicity is tested.
- Missing/wrong metadata fails loudly.
- Field-line RHS dR/dphi = R*BR/Bphi and dZ/dphi = R*BZ/Bphi is tested.
- A validation campaign writes JSON, NPZ, and PNG artifacts.
- The work is CPU-only in normal CI and does not require upstream repos at test time.
```

## Upstream artifact contract

The first `jax_drb` implementation should load the lightweight NetCDF grid
written by `virtual_casing_jax.grid_export.write_extended_field_netcdf` or
`write_mgrid_like`.

Required dimensions:

- `nR`
- `nphi`
- `nZ`

Required coordinate variables:

- `R`, shape `(nR,)`, units meters
- `phi`, shape `(nphi,)`, physical toroidal angle in radians
- `Z`, shape `(nZ,)`, units meters

Required field variables:

- `BR`, shape `(nR, nphi, nZ)`, cylindrical radial magnetic field
- `Bphi`, shape `(nR, nphi, nZ)`, cylindrical toroidal magnetic field
- `BZ`, shape `(nR, nphi, nZ)`, vertical magnetic field
- `absB`, shape `(nR, nphi, nZ)`, magnetic-field magnitude

Required or strongly recommended metadata:

- `format`, ideally `mgrid_like` or `extended_field`
- `coordinate_convention`, must indicate `(R, phi, Z)` and physical `phi`
- `field_components`, must indicate `BR,Bphi,BZ`
- `nfp`
- `stellsym`
- `source`
- `src_nphi`
- `src_ntheta`
- `digits`
- `branch`, expected `internal` for plasma-current contribution outside LCFS
- `units`
- `virtual_casing_jax_commit`
- `vmec_jax_commit`
- `essos_commit`, if the artifact came through ESSOS
- `coil_source`, if a coil field was included

Metadata validation rules for `jax_drb`:

- Reject files with no coordinate convention unless an explicit unsafe override
  is passed.
- Reject files whose convention mentions `zeta` instead of physical `phi`.
- Reject nonpositive `nfp`.
- Reject nonmonotone coordinate axes.
- Warn or fail if `absB` is inconsistent with `sqrt(BR**2 + Bphi**2 + BZ**2)`
  beyond a small tolerance.

The grid represents physical cylindrical coordinates. If it covers one field
period, `phi` should be wrapped by `2*pi/nfp`. If the metadata explicitly says
the file covers `2*pi`, wrapping can use `2*pi`. Do not assume VMEC `zeta`.

## Initial jax_drb API

Create `src/jax_drb/geometry/vmec_extender_import.py`.

Recommended public surface:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class VmecExtenderGrid:
    R: jax.Array
    phi: jax.Array
    Z: jax.Array
    BR: jax.Array
    Bphi: jax.Array
    BZ: jax.Array
    absB: jax.Array
    nfp: int
    phi_period: float
    metadata: Mapping[str, Any]


def load_vmec_extender_grid_netcdf(
    path: str | Path,
    *,
    strict_metadata: bool = True,
) -> VmecExtenderGrid:
    ...


def interpolate_vmec_extender_B_cyl(
    grid: VmecExtenderGrid,
    R_phi_Z: jax.Array,
) -> jax.Array:
    """Return BR, Bphi, BZ at target points with periodic physical-phi handling."""
    ...


def vmec_extender_absB(
    grid: VmecExtenderGrid,
    R_phi_Z: jax.Array,
) -> jax.Array:
    ...


def vmec_extender_fieldline_rhs_RZ_phi(
    grid: VmecExtenderGrid,
    R_phi_Z: jax.Array,
    *,
    min_abs_Bphi: float = 1.0e-12,
) -> jax.Array:
    """Return dR/dphi and dZ/dphi using R*BR/Bphi and R*BZ/Bphi."""
    ...
```

Implementation guidance:

- Use `scipy.io.netcdf_file` or `netCDF4` for loading. Convert to JAX arrays
  only after validating shapes and metadata with NumPy.
- Use trilinear interpolation in `(R, phi, Z)` with periodic wrapping in
  physical `phi`.
- Make interpolation JIT-friendly after loading. File I/O itself is not
  differentiable and does not need to be JIT-compatible.
- Support batched target points with shape `(..., 3)`.
- Return `(..., 3)` for `B_cyl` and `(..., 2)` for the field-line RHS.
- Keep `VmecExtenderGrid` a PyTree if it will be passed into `jax.jit`.
  Register it explicitly if needed, or keep metadata static outside the jitted
  numerical kernels.
- Do not add direct dependencies on `vmec_jax`, `virtual_casing_jax`, or ESSOS
  for normal `jax_drb` tests. Use small synthetic NetCDF fixtures.

## Interpolation requirements

Tests must cover:

1. Node parity
   - Build a synthetic grid where
     `BR = R + 2*phi + 3*Z`,
     `Bphi = 2 + R`,
     `BZ = R - phi + Z`.
   - Interpolation at grid nodes must match exactly or to roundoff.

2. Midpoint parity
   - For linear fields, midpoint interpolation must match the analytic value.

3. Periodic physical-phi wrapping
   - If `nfp = 5`, then `phi_period = 2*pi/5`.
   - Verify `B(R, phi, Z) == B(R, phi + phi_period, Z)`.

4. Shape handling
   - Single point `(3,)`, batch `(n, 3)`, and higher-rank batch `(..., 3)`.

5. Metadata guards
   - Missing `coordinate_convention` fails in strict mode.
   - Convention containing `zeta` fails.
   - Nonmonotone axes fail.
   - Inconsistent field shapes fail.

6. Field-line RHS
   - With a synthetic field where `BR/Bphi` and `BZ/Bphi` are known, verify
     `dR/dphi = R*BR/Bphi` and `dZ/dphi = R*BZ/Bphi`.
   - Verify tiny `Bphi` raises a useful error or returns a documented bounded
     value depending on chosen API.

## FCI and edge/SOL coupling

The first imported-field layer should not try to solve a full turbulence case.
It should provide enough geometry to connect to existing FCI/open-field-line
operators.

Near-term geometry bridge:

- Use the imported field to compute field-line RHS in `phi`.
- Integrate short forward/backward maps from each grid point to neighboring
  toroidal planes.
- Store or return map arrays compatible with the existing `FciMaps` patterns
  in `src/jax_drb/geometry/fci_maps.py`.
- Compare simple maps against analytic synthetic fields before using any
  VMEC-extender artifact.

Initial SOL model smoke should be conservative and small:

- scalar density or temperature-like passive field
- field-aligned advection or diffusion using imported maps
- cross-field diffusion in `R`/`Z`
- simple source and parallel-loss sink
- fixed time step or existing time integrator path

Implementation status: this is now covered by
`create_vmec_extender_sol_smoke_package`. The campaign remains synthetic and
CPU-only in normal tests; it validates imported-field map/operator coupling
without claiming real VMEC-extender SOL turbulence before upstream exporters
and real artifacts are stable.

Do not claim a research-grade turbulence result from the first PR. The first
PR should establish the imported-field geometry contract, interpolation, maps,
and validation campaign.

## Validation campaign

Create `src/jax_drb/validation/vmec_extender_edge_field_campaign.py`.

Recommended campaign outputs:

- summary JSON
- arrays NPZ
- PNG figure

Recommended report fields:

- `case`
- `source`
- `grid_shape`
- `nfp`
- `phi_period`
- `metadata_passed`
- `node_interpolation_max_abs_error`
- `midpoint_interpolation_max_abs_error`
- `field_period_relative_l2`
- `fieldline_rhs_max_abs_error`
- `absB_consistency_max_abs_error`
- `passed`

Campaign tests should run entirely from a synthetic fixture created in the
test temporary directory. Do not require upstream artifacts in CI.

Later, after upstream PRs merge, add an optional larger campaign path that
loads a small committed VMEC-extender NetCDF artifact and compares:

- imported-grid point samples against upstream `B_cyl` samples
- field-period symmetry
- ESSOS connection lengths or Poincare sections where available

## Documentation and examples

Add:

- `docs/vmec_extender_edge_fields.md`
- `examples/geometry-3D/vmec-extender/imported_field_demo.py`

The docs should state:

- this imports a gridded VMEC-extender field, not a self-consistent SOL plasma
  equilibrium
- `phi` is physical toroidal angle
- the first field artifact is expected to be outside the VMEC LCFS
- the plasma contribution in upstream virtual casing uses the internal branch
  because plasma currents are inside the LCFS
- hard Poincare/wall-hit diagnostics are not differentiable objectives
- interpolation can be differentiated with respect to field values and target
  coordinates, but not through file I/O

Update release-surface tests if this repository requires every public doc,
example, or validation module to be registered.

## CI and dependency constraints

Normal CI must stay CPU-only and deterministic.

Required tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_vmec_extender_import.py \
  tests/test_validation_vmec_extender_edge_field_campaign.py
```

Before opening or updating the PR, also run the release-surface and relevant
geometry/FCI tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_release_surface.py \
  tests/test_validation_stellarator_fci_operator_campaign.py \
  tests/test_validation_essos_imported_fci_campaign.py
```

If a full run is feasible:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Avoid adding heavyweight optional dependencies unless already present in
`pyproject.toml`. This repository already uses `jax`, `scipy`, `matplotlib`,
and `netCDF4` in its broader validation stack.

## Acceptance checklist for the first jax_drb PR

- [x] `VmecExtenderGrid` loader validates NetCDF shape and metadata.
- [x] JAX interpolation returns `BR`, `Bphi`, `BZ` for batched target points.
- [x] Physical-phi periodicity is implemented and tested.
- [x] `absB` consistency is checked.
- [x] Field-line RHS in `phi` is implemented and tested.
- [x] A small validation campaign writes JSON, NPZ, and PNG artifacts.
- [x] Docs and example are registered in release-surface tests if required.
- [x] Tests are numerical/physics tests, not smoke-only tests.
- [x] Normal CI does not require `vmec_jax`, `virtual_casing_jax`, ESSOS, or
      external STELLOPT artifacts.

## Work that should wait for upstream merge

Defer these until `virtual_casing_jax`, ESSOS, and `vmec_jax` Phase 1 PRs are
merged and their exported-field contract is stable:

- direct callback integration with `VirtualCasingExteriorField`
- automated generation of VMEC-extender field grids from inside `jax_drb`
- full ESSOS connection-length parity using a VMEC-extender field
- a committed real VMEC/coils NetCDF fixture unless artifact size and
  provenance are approved
- SOL turbulence claims based on real stellarator exterior fields

## Upstream artifact generation command shape

After upstream merge, the expected upstream workflow should look like this
conceptually:

```bash
essos-vmec-extender grid \
  --wout wout.nc \
  --coils coils.json \
  --R 0.8:2.0:64 \
  --phi 0:1.25663706144:32 \
  --Z -0.8:0.8:64 \
  --src-nphi 64 \
  --src-ntheta 64 \
  --digits 8 \
  --out vmec_extender_field.nc

essos-vmec-extender validate \
  --wout wout.nc \
  --coils coils.json \
  --src-nphi 64 \
  --src-ntheta 64 \
  --digits 8 \
  --out vmec_extender_validation.json
```

The exact CLI flags are owned by ESSOS and may change before merge. The
`jax_drb` importer should not depend on the CLI; it should depend on the
NetCDF variables and metadata listed above.

## Risks to keep visible

- `phi` versus `zeta` mistakes will silently corrupt stellarator field-period
  behavior. Test periodicity and document physical `phi`.
- `Bphi` can be small in field-line RHS formulas. Add a guard and test it.
- NetCDF metadata may arrive as bytes from `scipy.io.netcdf_file`; normalize
  bytes to strings before validation.
- Interpolation near boundaries must have documented behavior. Prefer in-bounds
  strict checks for the first PR, with optional clipping only if explicitly
  requested.
- Imported gridded fields are not self-consistent SOL plasma equilibria.
  `jax_drb` transport simulations remain downstream models using a prescribed
  magnetic field.

## Best next implementation sequence for the jax_drb agent

1. Add synthetic NetCDF fixture helper in tests.
2. Implement NetCDF loader and metadata validation.
3. Implement JAX trilinear interpolation with periodic `phi`.
4. Implement `absB` and field-line RHS helpers.
5. Add validation campaign and artifact writer.
6. Add docs/example and release-surface registration.
7. Run focused tests, then full CI-equivalent checks.
8. Keep the PR draft until upstream VMEC-extender PRs are merged or the
   gridded artifact contract is explicitly approved.
