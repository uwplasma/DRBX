# Cut-Wall Test Suite Plan

## Summary

Add a cut-wall-focused test suite under `jax_drb/tests`, separate from
`test_fci_operators_domain_decomp.py`.

Use four test files:

- `test_fci_cutwall_slab_operators.py`
- `test_fci_cutwall_slab_2field_physical_coincident.py`
- `test_fci_cutwall_slab_2field_oblique.py`
- `test_fci_cutwall_shifted_torus_4field.py`

## Test 1: Slab Oblique Wall Operator Convergence

File: `jax_drb/tests/test_fci_cutwall_slab_operators.py`

- Build a Cartesian slab geometry with periodic tangential directions and an
  embedded oblique planar wall, for example `x + alpha*y = c`.
- Construct non-empty `CutWallGeometry3D` / `LocalCutWallGeometry3D` with
  correct owner cells, normals, distances, area covectors, signs, active masks,
  and stencil-side metadata.
- Use a manufactured scalar field with exact wall values and exact derivatives.
- Test one representative local operator, such as `local_grad_perp_op_direct`.
- Test one representative conservative operator, such as
  `local_perp_laplacian_conservative_op`.
- Run at two or three resolutions and assert error decreases with refinement.

## Test 2: Slab Cut Walls Coincident With Physical Walls, 2-Field MMS

File: `jax_drb/tests/test_fci_cutwall_slab_2field_physical_coincident.py`

- Reuse the slab 2-field MMS setup.
- Replace coordinate physical-wall handling with cut-wall payloads exactly
  coincident with the physical wall faces.
- Build cut-wall BCs from exact MMS boundary values.
- Compare the cut-wall result against the existing physical-boundary MMS path.

## Test 3: Slab Oblique Planar Wall, 2-Field MMS

File: `jax_drb/tests/test_fci_cutwall_slab_2field_oblique.py`

- Use a true oblique embedded wall that is not coordinate-aligned.
- Choose smooth MMS fields on the remaining computational domain.
- Build exact `LocalCutWallBC3D` values for all fields touched by the 2-field
  RHS.
- Exercise both direct stencil patching and conservative flux-wall contribution
  through the normal 2-field RHS path.

## Test 4: Shifted Torus 4-Field With A Closed Embedded Cut-Wall Box

File: `jax_drb/tests/test_fci_cutwall_shifted_torus_4field.py`

- Reuse the shifted-torus 4-field MMS harness.
- Add six finite cut-wall faces that close a logical box strictly inside the
  computational domain.
- Use lower/upper faces in radial, poloidal, and toroidal logical coordinates,
  with the toroidal extent chosen so the box covers several zeta shards but
  leaves at least one shard outside the closed volume.
- Use exact MMS field values to populate cut-wall BCs for every 4-field model
  field that touches the wall.

### Implementation Plan

- Define a closed box with finite ranges in `x`, logical poloidal angle, and
  `zeta`; keep every face away from the physical radial boundaries.
- Build coordinate-stencil replacements on both sides of each crossed logical
  face so both the inside and storage-outside cells see Dirichlet wall values
  instead of reading across the closed surface.
- Close the corresponding regular conservative faces in `x`, `y`, and `z`
  using `LocalRegularFaceGeometry3D` open masks.
- Build shard-local padded `LocalCutWallGeometry3D`,
  `LocalCoordinateStencilDependencyMap3D`, and exact Dirichlet
  `LocalCutWallBC3D` values for `phi`, `density`, `omega`,
  `v_ion_parallel`, and `v_electron_parallel`.
- Add a test-local 4-field RHS wrapper that passes the cut-wall geometry and
  exact `phi` BC into `LocalPerpLaplacianInverseSolver`, then builds all local
  stencils with cut-wall dependency maps and field-specific exact wall values.
- Restrict convergence error norms to cell centers outside the closed box,
  treating the box interior as solid-obstacle storage rather than fluid.
- Add geometry, RHS isolation, phi-solve, and MMS convergence tests plus a CLI
  matching the shifted-torus 4-field MMS convergence harness.
