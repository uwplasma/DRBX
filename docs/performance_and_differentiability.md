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

The refreshed same-machine live Hermès rerun matrix now sharpens those generic
blockers into specific case priorities:

- `neutral_mixed_one_step`
  - worst normalized RMS mismatch about `9.17e-1`
  - native/reference wall-time ratio now about `2.93x`
  - dominant field: `NVh`
- `recycling_1d_one_step`
  - worst normalized RMS mismatch about `4.62e-3`
  - native/reference wall-time ratio about `3.65x`
  - dominant normalized field: `Pd+`
- `recycling_dthe_one_step`
  - worst normalized RMS mismatch about `4.92e-3`
  - native/reference wall-time ratio now about `7.82x`
  - dominant field: `NVd`

The current integrated and direct tokamak recycling one-step ladders still show
visible relative mismatch, but the updated live report now marks them as
normalization-sensitive because the dominant compare field is near-zero `NVd`
while the absolute max-error stays tiny.

The detailed remediation order is now documented in:

- [runtime_gap_remediation.md](runtime_gap_remediation.md)
- [profiling_runtime.md](profiling_runtime.md)

The implicit solver now exposes phase-resolved diagnostics on every sparse
Newton step used by the neutral and recycling native paths. Those diagnostics
record residual calls/time, Jacobian refreshes/time, linear-solve time,
line-search time, and fallback use. This is the instrumentation needed for the
next `recycling_dthe_one_step` runtime campaign: profiler plots can now be tied
to the solver phases that reviewers care about rather than only to Python
function names. The sparse finite-difference Jacobian builder also accepts a
precomputed color-group extraction plan, reducing repeated host-side indexing
work during each Newton refresh while keeping the numerical finite-difference
surface unchanged. The current SciPy BDF recycling path now reuses that plan in
its Jacobian callback as well, and the AMJUEL source path reuses shared
log-temperature/log-density inputs across paired rate/radiation fits. The BDF
callback also caches the most recent exact `(t, y)` RHS evaluation, so a
`jac(t, y)` request immediately following `rhs(t, y)` reuses the base RHS
instead of recomputing the full recycling source/closure assembly. The
recycling history object records `bdf_rhs_evaluation_count`,
`bdf_rhs_cache_hit_count`, and `bdf_jacobian_callback_count` for future runtime
audits.

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

The local heavy-solve scaling package now documents the stronger local result on
a real promoted production solve rather than on a compact differentiable
kernel:

- [local_cpu_scaling_campaign.md](local_cpu_scaling_campaign.md)

That package now focuses only on the fixed-work steady-state ensemble result on
`tokamak_recycling_dthene_one_step`, because the warmed single-solve thread
curve stayed essentially flat on this MacBook and was not the right local
scaling figure.

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
- reaction/source assembly on the heavy recycling lane now reuses top-level
  accumulators instead of allocating full per-reaction source dictionaries on
  every RHS call;
- the recycling RHS now reuses already-computed collision, ionisation, and
  charge-exchange rate surfaces across collision closure and neutral parallel
  diffusion instead of recomputing them independently;
- on the profiled `tokamak_recycling_dthene_one_step` case, the cumulative
  implicit-solver optimization pass now drops end-to-end wall time from about
  `11.84 s` to about `2.08 s` on this MacBook;
- on the profiled `recycling_dthe_one_step` case, the current allocation and
  source-assembly pass drops the local timed mean from about `75.3 s` to about
  `54.1 s`;
- the live neon direct-tokamak recycling parity slice still passes after those
  changes, which means the refactor removed overhead without changing the
  compare surface.

That is a real improvement, but it is not the final optimization story. The
dominant remaining blocker is still the finite-difference Jacobian and the
host/SciPy residual structure itself. The refreshed cProfile on the heavy
multispecies recycling lane now points most clearly at:

- finite-difference Jacobian assembly
- neutral parallel diffusion
- collision closure
- target recycling / target boundary-source assembly
- open-field prepared-state and boundary setup

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
- [build_sparse_jvp_jacobian](../src/jax_drb/solver/implicit.py)

The grouped sparse-JVP builder uses the same coloring contract as the sparse
finite-difference builder, but obtains each color group from a JAX linearized
push rather than from perturbed residual calls. This is the intended bridge for
pure-JAX residuals and future jaxified recycling kernels. It is deliberately
not forced onto the current promoted recycling BDF path because that RHS is
still host/NumPy/SciPy based; using JVPs there first requires moving the
dominant source, closure, boundary, and pack/unpack kernels into a
JAX-transformable residual.

That bridge now follows the documented JAX autodiff pattern more closely:
`jax.linearize` evaluates the primal residual once and returns a reusable
linear map, while `jax.vmap` batches the colored tangent pushes. The default
path pushes all color groups in one vectorized batch; `batch_size=1` gives the
memory-bounded serial form, and intermediate batch sizes provide a knob between
temporary tangent memory and dispatch overhead. This removes finite-difference
step-size sensitivity for JAX-transformable residuals and gives a direct
operator-level check against the older sparse finite-difference builder.

The existing `solve_jax_linearized_newton_system` path is already the stronger
matrix-free option for residuals that can remain inside JAX: it passes the
linearized Jacobian action directly to JAX GMRES instead of materializing a
Jacobian. The next production migration should therefore not wrap the current
host residual in more solver adapters. It should first make the recycling
residual kernels JAX-transformable, then select between:

- materialized sparse JVP Jacobian assembly when a sparse direct or SciPy
  compatibility solve is still needed;
- matrix-free JVP actions when Krylov solves and differentiable objectives are
  the target;
- VJP/implicit-function sensitivity when the output is a scalar objective or a
  steady-state quantity of interest.

The first residual-kernel step is now also in place. The packaged AMJUEL,
OpenADAS, and hydrogen charge-exchange rate helpers preserve the existing
NumPy production path when called with NumPy arrays, but stay in JAX when
called with JAX arrays. Focused tests now check `jit` and `grad` through the
paired AMJUEL rate/radiation path, the OpenADAS interpolation path, and the
hydrogen charge-exchange fit. This does not yet make the full recycling
transient differentiable, but it removes one of the source-term barriers that
blocked the JAX residual/JVP backend.

The surrounding single-isotope reaction-source formulas have also been made
backend-preserving for JAX array inputs. Ionisation, recombination, and
charge-exchange source accumulation still use the existing dictionary-oriented
public API for compatibility, but the temperature floors, velocity
reconstruction, charge-exchange effective temperature, and kinetic-energy
exchange pieces no longer force NumPy when the caller supplies JAX arrays.
Focused tests now differentiate through a compact hydrogen
ionisation/recombination/charge-exchange source objective.

## Current GPU-Native Audit

The office GPU environment is now usable for the compact native JAX lanes with
`jax[cuda12]==0.6.2` and two visible CUDA devices. The first meaningful GPU
measurements are:

- traced-field-line reduced lane:
  compile `4.41e-2 s`, first execute `1.23e-3 s`, warm execute `3.30e-4 s`
- stellarator VMEC reduced lane:
  compile `7.36e-3 s`, first execute `3.98e-4 s`, warm execute `1.14e-4 s`

Those are the right GPU benchmark surfaces for the current codebase. The heavy
recycling lanes remain primarily CPU/runtime-architecture problems until more
of the transient backbone is moved out of the host/SciPy path.

That path is appropriate for compact pure-JAX residuals and future reduced
native kernels. It is not yet the default on the promoted recycling/tokamak
backbone because that residual still crosses the host/SciPy boundary too often
to make a JVP-driven solve the right production choice today.

The sparse finite-difference Jacobian path now also has a practical CPU
parallelization policy:

- color-group residual evaluations can run in parallel threads;
- by default that threading now turns on automatically for heavy sparse solves;
- users can still override it explicitly with
  `JAX_DRB_FD_JACOBIAN_THREADS=<N>`;
- on the profiled neon tokamak one-step case, that gives a small but real
  additional local speedup on top of the larger residual/Jacobian cleanup.

The new committed local CPU scaling artifact now sharpens that conclusion with
real numbers on the heavier D/T/He/Ne tokamak recycling lane:

- the committed figure uses `16` repeated heavy solves on
  `tokamak_recycling_dthene_one_step`;
- steady-state fixed-work ensemble speedup is about:
  - `1.88x` from `1 -> 2` workers
  - `3.67x` from `1 -> 4` workers
  - `4.94x` from `1 -> 8` workers
- that is the right local-CPU scaling story for users running parameter scans,
  UQ, optimization, or repeated solver evaluations on a laptop: spread
  independent heavy solves across workers instead of expecting one warmed solve
  to approach ideal thread-level strong scaling.
- that `16`-solve artifact is also the right committed benchmark size on this
  MacBook: heavier local ensembles were checked, but they started to lose the
  cleaner scaling curve because thermal/scheduling effects outweighed the extra
  fixed work.

The last cProfile pass on the same promoted heavy solve also clarifies the
remaining bottleneck split after the recent optimization work:

- sparse finite-difference Jacobian assembly is still a dominant cost;
- the next dominant production-path cost is now the recycling RHS/source
  assembly path in `recycling_1d.py`, not the older per-cell transport loops;
- the vectorized `neutral_mixed.py` parallel-gradient kernel is no longer a
  top-level hotspot after the latest array-kernel rewrite;
- that means the next real performance step is a deeper restructuring of the
  implicit/recycling residual and Jacobian path rather than another cosmetic
  CPU-thread sweep.

The same pass also changed the live runtime picture in a way that matters for
the paper and for users:

- `neutral_mixed_one_step` dropped from roughly `6.44 s` live runtime to about
  `1.43 s`, while keeping the same dominant fidelity gap on `NVh`
- `recycling_1d_one_step` now runs at about `15.98 s` live runtime with
  fidelity preserved, which is materially better than the earlier recycling
  baseline but still slower than Hermès on the same machine

That means the next work should be divided clearly:

- fix the neutral mixed `NVh` mismatch as a fidelity problem
- reduce `recycling_dthe_one_step` and `recycling_1d_one_step` as runtime
  problems without widening their already-tight fidelity band

For the current paper and release, the parallelization claim should also stay
operationally concrete:

- the best current local acceleration comes from making each solve cheaper, not
  from splitting one heavy solve across CPU devices on this laptop;
- explicit CPU multi-device execution is implemented and usable, but its local
  scaling curve is still bounded;
- the most promising wider parallelization model for general workflows is
  batching independent solves, objectives, or parameter studies over JAX maps
  and accelerator devices, rather than overselling single-case laptop CPU
  strong scaling.

## Reproducible Profiling Workflow

The supported profiling entry point is now:

- [scripts/profile_curated_case.py](../scripts/profile_curated_case.py)

That script can collect:

- `cProfile` output
- JAX TensorBoard / Perfetto traces
- JAX device-memory profiles
- persistent compilation cache runs
- XLA dump trees

The workflow and recommended worst-offender cases are documented in:

- [profiling_runtime.md](profiling_runtime.md)

This is now the preferred path for runtime work because it keeps profiling tied
to the same curated cases that define the public Hermès comparison surface.

## Current GPU Status

The reachable `office` machine already exposes two CUDA-visible JAX devices
(`RTX A4000`). GPU performance work is now limited by environment consistency
rather than missing hardware.

The current blocker there is a package mismatch:

- JAX sees both GPUs
- `numpy`, `scipy`, `matplotlib`, and `netCDF4` import cleanly
- `diffrax` and `equinox` currently fail in the system environment because of a
  `jaxlib` extension mismatch

So the next GPU step is operationally clear:

1. create a clean repo-local environment on `office`
2. install a matching JAX/JAXLIB pair plus `jax_drb`
3. rerun the current worst-offender profiling cases with:
   - JAX trace
   - memory profile
   - persistent compilation cache
4. compare CPU and GPU runtime splits before making broader accelerator claims

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
