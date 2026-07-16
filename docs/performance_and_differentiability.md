# Performance And Differentiability

!!! note "Plan authority"
    This page explains current performance and differentiability evidence. The
    active execution plan is [`plan_jax_drb.md`](../plan_jax_drb.md) at the
    repository root. If this page conflicts with that plan, follow the plan and
    update this page afterward.

This page records the current fast paths, the current differentiable paths, and
the reproducible profiling workflow.

## Measured Turbulence Performance

The core closed-field-line drift-wave turbulence model (Hasegawa-Wakatani,
FFT-spectral RK4) is `jit`-compiled and differentiable end-to-end. Two measured
numbers quantify the "fast and differentiable" claim:

![Performance and differentiability](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/performance.png)

- **Throughput** is roughly grid-size-independent at about **2 million
  cell-updates per second** on a single CPU in float64 (each step is a dealiased
  spectral Poisson bracket); it rises substantially on GPU and in float32.
- **Differentiating *through* the turbulence is cheap.** One reverse-mode
  gradient of a diagnostic of the evolved state with respect to the
  transport-drive parameter — taken through the *entire* multi-step rollout —
  costs only about **2.7x a single forward evaluation** (here, 200 steps at
  `n = 64`), the expected small constant factor of reverse-mode autodiff rather
  than a cost that grows with the number of steps.

Regenerate with

```bash
PYTHONPATH=src python examples/benchmarks/performance_benchmark.py
```

Absolute timings depend on the host; the scalings do not.

## Choosing a Differentiation Method

The same gradient can be computed several ways, and the choice changes cost,
never the answer (gated to machine agreement in
`tests/test_autodiff_methods.py`). Measured on 200 turbulence steps at `n = 64`
(one scalar parameter, CPU f64):

![Differentiation methods](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/differentiation_methods.png)

- **Forward mode** (`jax.jacfwd`) is the most efficient for a *few* parameters:
  one tangent rides along the rollout — no reverse sweep, no stored trajectory —
  costing about **2x a forward evaluation** (vs ~3.1x for reverse mode here).
- **Reverse mode** (`jax.grad`) wins when differentiating with respect to
  *many* parameters at once (fields, geometry): one backward sweep covers them
  all, at the cost of storing the trajectory.
- **Checkpointed reverse** (`jax.grad` + `jax.checkpoint` on the step) trades a
  modest recompute for bounded memory — the fix when a long reverse rollout is
  memory-bound.

Rule of thumb: `jacfwd` for O(1-10) scalars, `grad` for parameter fields,
add `jax.checkpoint` when reverse mode runs out of memory. Reproduce with

```bash
PYTHONPATH=src python examples/autodiff/differentiation_methods_demo.py
```

## Multi-Device Strong Scaling

The FCI drift-reduced two-field step runs across multiple devices with
`shard_map`. The domain is decomposed into shards, each device owns a block of
cells plus a halo, and the halo is exchanged every step. The sharded RK4 step is
**bit-exact** against the single-device step — `tests/test_fci_sharded_2field.py`
checks a single-device sharded run and a forced four-device run
(`XLA_FLAGS=--xla_force_host_platform_device_count=4`) both reproduce the direct
step to ~1e-16 — so sharding changes only *where* the work runs, not the result.

The strong-scaling driver is
[`examples/benchmarks/fci_sharded_strong_scaling_demo.py`](../examples/benchmarks/fci_sharded_strong_scaling_demo.py).
It sweeps device counts by re-invoking itself once per count (the XLA host device
count must be set before JAX imports) and, on Linux, binds one physical core per
shard with `taskset` — the crucial detail: without core-binding a single-device
CPU program already spreads across all cores via XLA intra-op threading, so the
domain decomposition looks like it does nothing. On a 36-core Linux host with
core-binding, a `256 x 128 x 32` two-field step scales as **1.75x at 2 shards,
3.22x at 4 (about 81% efficiency), and 4.35x at 8**.

```bash
PYTHONPATH=src python examples/benchmarks/fci_sharded_strong_scaling_demo.py
```

!!! note "Host requirement"
    Meaningful strong-scaling numbers need a Linux host with `taskset` and at
    least as many physical cores as the maximum shard count. On macOS (no
    `taskset`) the demo still runs and verifies the cross-shard checksums, but
    the wall times are threading-limited, not a true scaling curve.

## Current Fast Native Lanes

The strongest current native paths are the compact JAX-native field updates that
stay inside `jax.numpy` kernels with lightweight analysis/output:

- anomalous diffusion (matrix-exponential propagator);
- electrostatic vorticity;
- Hasegawa-Wakatani drift-wave turbulence (pseudo-spectral);
- the reduced FCI operator and selected-field 3-D geometry kernels.

These are the best lanes for:

- performance measurements;
- precision studies;
- restart demonstrations;
- differentiable optimization loops.

## Current End-To-End Differentiable Lanes

The intended end-to-end differentiable lane is:

- TOML deck or Python driver;
- native JAX field evolution;
- portable array payload;
- JAX-side objective or analysis functional.

The compact diffusion and vorticity kernels, the Hasegawa-Wakatani flagship, and
the differentiable FCI drift-reduced RHS (`native/fci_drb_rhs.py`) are the best
starting points today because they stay fully inside JAX.

The diffusion lane has committed focused differentiable examples:

- sensitivity analysis: [examples/autodiff_diffusion_sensitivity_demo.py](../examples/autodiff_diffusion_sensitivity_demo.py)
- inverse design: [examples/autodiff_diffusion_inverse_design_demo.py](../examples/autodiff_diffusion_inverse_design_demo.py)
- fixed-workload CPU/GPU scaling: [examples/strong_scaling_diffusion_demo.py](../examples/strong_scaling_diffusion_demo.py)

The current artifact bundle is documented in
[autodiff_and_scaling_examples.md](autodiff_and_scaling_examples.md).

The Hasegawa-Wakatani flagship is differentiable end-to-end, enabling
gradient-based inverse design through turbulence; see
[Drift-Wave Turbulence](drift_wave_turbulence.md). The FCI drift-reduced RHS is
a PyTree that can be passed through `jax.jvp`, checked against finite
differences, and matched under `vmap`, as documented in
[Stellarator FCI Validation](stellarator_fci_validation.md).

## Current Differentiable Example Results

On the committed diffusion examples:

- autodiff and finite-difference gradients match closely on the compact
  four-parameter sensitivity study;
- first-order autodiff uncertainty propagation agrees with the vectorized
  Monte Carlo comparison on the compact field and scalar quantities of interest;
- the inverse-design example reduces the objective from about `2.95e-3` to about
  `5.52e-5`;
- the compact differentiable fixed-workload scaling artifact shows modest local
  CPU scaling on a MacBook: about `1.08x` from `1 -> 2` and `1.10x` from
  `1 -> 4` in process-group mode, and about `1.07x` and `1.08x` in host-device
  CPU `pmap` mode.

Those scaling numbers are intentionally framed narrowly: the compact diffusion
curve is a differentiability and execution-mode check, not a headline
performance claim, and it is measured on a differentiable objective rather than
only on a forward solve.

## What The Current Profiling Already Says

The committed profiling and runtime bundles already answer the first practical
performance questions:

- avoid tiny per-field JIT dispatches on the reduced 3-D kernels;
- batch same-shape selected fields before entering the jitted kernel;
- warm once before timing;
- keep solver/case metadata out of static JIT arguments;
- keep file I/O, plotting, and JSON serialization outside hot kernels.

They also answer the first CPU-parallelism question:

- the default JAX CPU runtime appears as one CPU device and relies on XLA's
  internal CPU threading;
- explicit host-device CPU parallelism is possible by setting
  `JAX_DRB_HOST_DEVICE_COUNT=N` before importing `jax_drb` or `jax`;
- on the committed differentiable diffusion scaling surface the local
  process-group mode is slightly stronger than the host-device `pmap` mode on
  this MacBook, but both are modest;
- CPU parallelization is real and usable here, but it should be treated as a
  bounded strong-scaling tool, not as an automatic replacement for accelerator
  execution.

### Parallelization Model

There are three distinct execution modes worth separating:

- default CPU execution: one JAX CPU device with XLA-managed internal threading;
- explicit host-device CPU execution: multiple CPU devices exposed with
  `JAX_DRB_HOST_DEVICE_COUNT=N`, then mapped with `pmap` or equivalent
  device-parallel transforms;
- process-group CPU execution: multiple Python workers with one JAX CPU device
  each.

The committed diffusion scaling artifact measures the last two explicitly, with
the modest results quoted above.

## Current GPU Status

The reachable `office` machine exposes two CUDA-visible JAX devices
(`RTX A4000`, `cuda:0` and `cuda:1`) with `jax[cuda12]`. The first meaningful
GPU measurements on the compact reduced lanes are:

- traced-field-line reduced lane:
  compile `4.41e-2 s`, first execute `1.23e-3 s`, warm execute `3.30e-4 s`;
- stellarator VMEC reduced lane:
  compile `7.36e-3 s`, first execute `3.98e-4 s`, warm execute `1.14e-4 s`.

Those are the right GPU benchmark surfaces for the current codebase. These
kernels are small, so the honest next GPU step is not to claim whole-code
acceleration from them, but to keep more physics on the same array-native
contract and rerun the profiling script with JAX traces, device-memory
snapshots, and a persistent compilation cache on the GPU host.

## Reproducible Profiling Workflow

The supported profiling entry point for the reduced FCI/geometry lanes is
[scripts/profile_stellarator_drb_pytree.py](../scripts/profile_stellarator_drb_pytree.py),
which can collect `cProfile` output, JAX TensorBoard / Perfetto traces,
device-memory profiles, persistent compilation-cache runs, and XLA dump trees.
The workflow and recommended cases are documented in
[profiling_runtime.md](profiling_runtime.md).

## Where Extra JAX Ecosystem Pieces Might Help

The current code already benefits most from plain `jax`, structured JIT
boundaries, and explicit kernel batching. Additional ecosystem tools are most
likely to help in specific places:

- `equinox`: useful if larger native kernels are restructured into clearer
  pure-function model objects or if filtered transforms simplify mixed static
  metadata and array state;
- `lineax`: potentially useful if future native linear solves move toward
  JAX-native linear-operator interfaces;
- `diffrax`: useful for clean differentiable time integration on compact native
  lanes.

For the current release the promoted native kernels do not depend on `equinox`,
`lineax`, or `diffrax`; those libraries remain packaged as optional
future-tooling hooks rather than active explanations for the current
reduced-kernel speedups.

## Guidance For Users

If you need:

- the cleanest standalone runtime workflow:
  start from [restartable_diffusion_tutorial.md](restartable_diffusion_tutorial.md);
- compact high-quality figures and movies:
  use [validation_gallery.md](validation_gallery.md);
- the best current base for differentiable research code:
  start from the compact native-exact diffusion and vorticity lanes, the
  Hasegawa-Wakatani flagship, or the differentiable FCI drift-reduced RHS.

## Recommended Next Refactors

- keep expressing new physics directly on fixed-layout JAX-native arrays so the
  linearized residual does not need to reconstruct full guard-cell fields for
  each transform;
- fuse small same-shape analysis reductions where they currently enter JAX one
  field at a time;
- use more `vmap`-style batching where case structure is already homogeneous;
- keep plotting, output writing, and CLI serialization as boundary code rather
  than inside hot kernels;
- only widen `equinox`/`lineax` usage where it removes a measured bottleneck.
