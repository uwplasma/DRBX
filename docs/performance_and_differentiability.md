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

The diffusion lane now also has committed focused differentiable examples:

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

## What The Current Profiling Already Says

The committed profiling and runtime bundles already answer the first practical
performance questions:

- avoid tiny per-field JIT dispatches on the reduced 3D kernels;
- batch same-shape selected fields before entering the jitted kernel;
- batch the reference/candidate pair through the same reduced kernel when the
  compare surface is shape-aligned;
- warm once before timing;
- keep solver/case metadata out of static JIT arguments;
- keep file I/O, plotting, and JSON serialization outside hot kernels.

They now also answer the first CPU parallelism question on this MacBook:

- the default JAX CPU runtime still appears as one CPU device and relies on
  XLA's internal CPU threading;
- explicit host-device CPU parallelism is possible by setting
  `JAX_DRB_HOST_DEVICE_COUNT=N` before importing `jax_drb` or `jax`;
- on the current heavier committed differentiable diffusion scaling surface,
  the local process-group mode is slightly stronger than the host-device
  `pmap` mode on this MacBook, but both are modest;
- that means CPU parallelization is real and usable here, but it should be
  treated as a bounded strong-scaling tool, not as an automatic replacement for
  accelerator execution.

That guidance is not speculative; it is the measured result of the committed
Perfetto-backed reduced-kernel audits in:

- [jax_native_profile_audit.md](jax_native_profile_audit.md)
- [native_3d_runtime_campaign.md](native_3d_runtime_campaign.md)

## Where More JAX Can Still Help

There are still real opportunities for more JAX-native execution, but they are
not all equally safe on the current parity surface.

### Parallelization Model

Today there are three distinct execution modes worth separating:

- default CPU execution:
  - one JAX CPU device with XLA-managed internal threading
- explicit host-device CPU execution:
  - multiple CPU devices exposed with `JAX_DRB_HOST_DEVICE_COUNT=N`
  - then mapped with `pmap` or equivalent device-parallel transforms
- process-group CPU execution:
  - multiple Python workers with one JAX CPU device each

The committed diffusion scaling artifact now measures the last two explicitly.
On this MacBook, the current fixed-workload result is:

- local process-group reference:
  - about `1.08x` from `1 -> 2`
  - about `1.10x` from `1 -> 4`
- local host-device `pmap`:
  - about `1.07x` from `1 -> 2`
  - about `1.08x` from `1 -> 4`

That is useful, but it also sets the right expectation:

- explicit CPU-device parallelism is available and now supported by the runtime;
- the stronger current laptop CPU result is still the process-group mode, not
  by a large margin;
- the scaling ceiling on this differentiable lane is still modest;
- the highest-value long-term acceleration target is still the heavier transient
  backbone and genuine accelerator hardware, not only more CPU-device splitting.
- additional heavier fixed-workload CPU probes did not materially change that
  conclusion on this MacBook, so the CPU strong-scaling story should stay
  narrow and reviewer-safe.

## Current Solver-Side Optimization Pass

The latest implicit-solver pass tightened the heaviest host/SciPy path without
changing the validated physics surface:

- the sparse Newton path now reuses CSC structure where possible instead of
  rebuilding CSC conversions repeatedly inside the linear solve loop;
- the recycling implicit step now carries a packed-state layout explicitly so
  active slices, active shape, field size, and field templates are not rebuilt
  on every residual/unpack call;
- the packed residual path now avoids repeated full-field copies between unpack,
  packed-RHS staging, and species override;
- the hottest neutral/tokamak transport operators now use vectorized NumPy
  kernels instead of per-cell Python loops on the production residual path;
- on the profiled `tokamak_recycling_dthene_one_step` case, those changes drop
  the end-to-end wall time from about `11.84 s` to about `3.16 s` on this
  MacBook;
- the live neon direct-tokamak recycling parity slice still passes after those
  changes, which means the refactor removed overhead without changing the
  compare surface.

That is a real improvement, but it is not the final optimization story. The
dominant remaining blocker is still the finite-difference Jacobian and the
host/SciPy residual structure itself.

### Highest-Value Near-Term Opportunities

- replace finite-difference Jacobian construction on the heavier transient lanes
  with JAX linearization or JVP-based products;
- reduce repeated host/device boundary crossings on the recycling transient
  backbone;
- keep state packing layouts stable enough that larger sections of the transient
  solve can stay inside one compiled function;
- widen the already-batched selected-field native kernels to more fields and
  broader reduced 3D workflows.

The source tree now also includes a JAX-linearized Newton-GMRES path for
residuals that are already JAX-transformable:

- [solve_jax_linearized_newton_system](../src/jax_drb/solver/implicit.py)

That path is appropriate for compact pure-JAX residuals and future reduced
native kernels. It is not yet the default on the promoted recycling/tokamak
backbone because that residual still crosses the host/SciPy boundary too often
to make a JVP-driven solve the right production choice today.

### Lower-Risk Structural JAX Improvements

- fuse small same-shape analysis reductions where they currently enter JAX one
  field at a time;
- use more `vmap`-style batching where case structure is already homogeneous;
- keep scalar diagnostics and compare surfaces on array-native code paths rather
  than repeated Python loops where practical.

## Where Extra JAX Ecosystem Pieces Might Help

The current code already benefits most from plain `jax`, structured JIT
boundaries, and explicit kernel batching. Additional ecosystem tools are most
likely to help in specific places:

- `equinox`: useful if larger native kernels are restructured into clearer
  pure-function model objects or if filtered transforms simplify mixed static
  metadata and array state;
- `lineax`: potentially useful if future native linear solves move further away
  from the current SciPy/sparse boundary and toward JAX-native linear-operator
  interfaces;
- `diffrax`: useful for clean differentiable time integration on compact native
  lanes, but not a drop-in replacement for the currently validated recycling
  backbone without new parity work.

For the current release, that distinction is now explicit in the source tree:

- the promoted native kernels do not currently depend on `equinox`, `lineax`,
  or `diffrax` in their active shipping paths;
- those libraries remain packaged as optional future-tooling hooks and legacy
  lineage, not as active explanations for the current reduced-kernel speedups;
- the measured bottlenecks are still more about solver structure and host
  barriers than about the absence of one extra library.

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
- only widen `equinox`/`lineax` usage where it removes a measured bottleneck or
  simplifies a parity-critical kernel, not as a cosmetic dependency expansion
