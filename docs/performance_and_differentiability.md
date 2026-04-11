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

The diffusion lane now also has committed publication-oriented differentiable examples:

- sensitivity analysis: [examples/autodiff_diffusion_sensitivity_demo.py](../examples/autodiff_diffusion_sensitivity_demo.py)
- inverse design: [examples/autodiff_diffusion_inverse_design_demo.py](../examples/autodiff_diffusion_inverse_design_demo.py)
- fixed-workload CPU/GPU scaling: [examples/strong_scaling_diffusion_demo.py](../examples/strong_scaling_diffusion_demo.py)

The current artifact bundle is documented in [autodiff_and_scaling_examples.md](autodiff_and_scaling_examples.md).

## Current Differentiable Example Results

On the committed diffusion examples:

- autodiff and finite-difference gradients match closely on the compact four-parameter sensitivity study
- the inverse-design example reduces the objective from about `2.95e-3` to about `5.52e-5`
- the current fixed-workload scaling artifact shows:
  - local CPU process-parallel reference: about `1.13x` speedup from `1 -> 8`
  - remote GPU device-parallel reference: about `2.19x` speedup from `1 -> 2`

Those scaling numbers are intentionally framed narrowly:

- the GPU curve is the meaningful accelerator result on the current artifact
- the CPU curve is a local single-node reference, not the main performance claim
- both are measured on a differentiable objective, not only on a forward solve

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
