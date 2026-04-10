# Performance And Differentiability

This page records the current fast paths, the current differentiable paths, and the known blockers on heavier edge/SOL workflows.

## Current Fast Native Lanes

The strongest current native paths are the compact native-exact ladders that stay inside JAX-native field updates and lightweight analysis/output:

- diffusion
- vorticity
- drift-wave
- blob2d
- selected direct tokamak operator and short-window ladders

These are the best lanes for:

- performance measurements
- precision studies
- restart demonstrations
- future differentiable optimization loops

## Current End-To-End Differentiable Target

The intended end-to-end differentiable lane is:

- TOML deck
- native JAX field evolution
- portable array payload
- JAX-side objective or analysis functional

The compact diffusion, vorticity, and drift-wave-style native paths are the best starting points for this today because they avoid the heaviest SciPy-only transient machinery used by the recycling backbone.

## Current Performance And Differentiability Blockers

The main blockers are concentrated in the promoted recycling/tokamak transient backbone:

- SciPy implicit stepping in [src/jax_drb/native/recycling_1d.py](../src/jax_drb/native/recycling_1d.py)
- finite-difference Jacobian construction and sparse linear algebra in [src/jax_drb/solver/implicit.py](../src/jax_drb/solver/implicit.py)
- repeated `np.asarray(...)` coercions and host-side copies through the recycling RHS path
- repeated pack/unpack of large transient state dictionaries in the implicit solve path

These are the highest-value refactor targets for the next release cycle because they limit:

- accelerator performance
- memory efficiency
- automatic differentiation
- maintainability of the promoted recycling/tokamak transient lane

## Guidance For Users

If you need:

- the cleanest standalone runtime workflow:
  - start from [restartable_diffusion_tutorial.md](restartable_diffusion_tutorial.md)
- compact high-quality figures and movies:
  - use [alfven_wave_meeting_demo.md](alfven_wave_meeting_demo.md) and [blob2d_meeting_demo.md](blob2d_meeting_demo.md)
- the best current base for differentiable research code:
  - start from the compact native-exact electrostatic lanes rather than the heavier recycling transient backbone

## Recommended Next Refactors

- replace finite-difference Jacobians with JAX linearization or JVP-driven solves on promoted lanes
- reduce or remove per-term `np.asarray(...)` barriers on native transient kernels
- move the strongest recycling transient lane to a backend-stable residual and state layout
- keep plotting, output writing, and CLI serialization as boundary code rather than inside hot kernels
