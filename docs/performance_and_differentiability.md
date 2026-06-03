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

The recycling lane now has a separate fixed-layout residual differentiability
gate. The new `scripts/profile_recycling_batched_jvp_gate.py` command builds
the real D/T/He recycling backward-Euler residual, keeps the active state in a
static packed layout, and checks the same residual under `jit`, `vmap`, `jvp`,
and a scalar-objective `grad`. This is the strongest current heavy-residual
differentiability evidence because it exercises the multispecies recycling
state rather than a synthetic diffusion objective. On the committed local CPU
run with `mesh:ny=100`, the residual JVP agrees with a centered finite
difference to about `6e-9`, the objective directional derivative agrees to
about `1.3e-7`, and the retained batch sweep through 256 states reaches about
`2.8x` residual throughput speedup and `2.2x` JVP throughput speedup over
serial same-kernel calls.

The source-term lane now also has a dedicated accelerator-throughput gate:
`scripts/profile_atomic_rate_throughput_gate.py`. That gate evaluates a
batched AMJUEL/CX reaction-source surface, its reverse-mode derivative, and a
scalar log-temperature sensitivity objective. On the office GPU run, the
largest committed batch (`4,194,304` points) is about `2.5x` faster than the
local CPU run for the rate surface and about `2.1x` faster for the autodiff
derivative. The scalar sensitivity agrees with centered finite differences at
about `1e-10` relative error on both CPU and GPU. This is an accelerator
speedup claim for a source kernel, not for the full output-window recycling
solve.

## Current Differentiable Example Results

On the committed diffusion examples:

- autodiff and finite-difference gradients match closely on the compact four-parameter sensitivity study
- first-order autodiff uncertainty propagation agrees with the vectorized
  Monte Carlo comparison on the compact field and scalar quantities of interest
- the inverse-design example reduces the objective from about `2.95e-3` to about `5.52e-5`
- the compact differentiable fixed-workload scaling artifact shows modest
  local CPU scaling on this MacBook: about `1.08x` from `1 -> 2` and `1.10x`
  from `1 -> 4` in process-group mode, and about `1.07x` and `1.08x` in
  host-device CPU `pmap` mode

Those scaling numbers are intentionally framed narrowly:

- the compact diffusion curve is a differentiability and execution-mode check,
  not the main performance claim
- the stronger local CPU result is the separate heavy-solve ensemble campaign
  on repeated recycling solves
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
  - worst normalized RMS mismatch about `1.44e-1`
  - dominant-field relative-L2 mismatch about `3.46e-1`
  - native/reference wall-time ratio now about `3.18x`
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
`bdf_rhs_cache_hit_count`, `bdf_rhs_callback_seconds`,
`bdf_rhs_evaluation_seconds`, `bdf_rhs_object_evaluation_seconds`,
`bdf_rhs_numpy_conversion_seconds`, `bdf_jacobian_callback_count`,
`bdf_jacobian_mode`, and `bdf_jvp_batch_size` for future runtime audits. These
phase counters separate SciPy callback overhead, fixed-layout RHS object
construction, and host-array conversion before the next JAX-native residual
promotion. The packed recycling RHS now also disables reaction diagnostics during
implicit residual/Jacobian evaluations and routes the exact D/T/He Hermès
reaction block through the fixed-layout array kernel, so the solver hot path no
longer pays for dictionary diagnostics that are only needed for reporting.

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
- the diagnostics-free packed D/T/He residual path now uses the fixed-layout
  reaction-source kernel, and a refreshed cProfile shows that D/T AMJUEL reuse
  reduces the fixed-layout reaction-source split to about `9.64 s` and AMJUEL
  polynomial evaluations to `117380` calls, although the full BDF solve remains
  dominated by sparse finite-difference Jacobian work;
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
- [solve_sparse_newton_system](../src/jax_drb/solver/implicit.py) with
  `jacobian_mode="jvp"`

The grouped sparse-JVP builder uses the same coloring contract as the sparse
finite-difference builder, but obtains each color group from a JAX linearized
push rather than from perturbed residual calls. This is the intended bridge for
pure-JAX residuals and future jaxified recycling kernels. It is deliberately
not forced onto the current promoted recycling BDF path because that RHS is
still host/NumPy/SciPy based; using JVPs there first requires moving the
dominant source, closure, boundary, and pack/unpack kernels into a
JAX-transformable residual.

The sparse Newton bridge now has two tested modes: the legacy materialized
finite-difference Jacobian and a materialized sparse-JVP Jacobian for
transformable residuals. The implicit solver audit records both modes on a
diagonal nonlinear solve and verifies that both recover the same root to
machine precision. The JVP mode intentionally remains opt-in because a JVP
Jacobian is only valid when the residual keeps dynamic state inside JAX.

The legacy SciPy BDF recycling history path now has the same opt-in derivative
knob for its Jacobian callback:
`JAX_DRB_RECYCLING_BDF_JACOBIAN_MODE=jvp`. The default remains finite
difference because the full output-window BDF residual still contains
host-oriented source, closure, boundary, and nonlinear-driver pieces. Trial
full-output adaptive JAX-linearized D/T/He runs are therefore not promoted as
the default path yet: the bounded direct run either failed the nonlinear update
guard or exceeded the local runtime budget before producing a stable output
window. The safe production policy is to expose the JVP callback for
transformable residual experiments while continuing to keep the validated BDF
path as the default.

A narrower BDF-compatible opt-in now exercises the same migration seam without
changing the output-window timestepper:
`runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp`. This route
keeps SciPy BDF, routes the RHS through the fixed-layout full-field active-array
adapter, and forces the sparse Jacobian callback through grouped JVPs. The
history diagnostics record `bdf_rhs_backend="fixed_full_field_array"` and
`bdf_jacobian_mode="jvp"` so profile artifacts identify the route explicitly.
It remains a parity/runtime gate rather than the default until the full
output-window campaigns show equal reference agreement with lower call count
and memory use. The current hydrogen one-step gate is now explicit:
`compare_recycling_transient_modes.py --case recycling_1d_one_step --mode bdf
--mode bdf_fixed_full_field_jvp --field Pe --require-fixed-jvp-diagnostics
--require-bdf-pairwise-max 1e-5` passes with a BDF-vs-fixed-JVP active-mesh
`Pe` delta of `6.28e-6`, but the same run takes about `59.9 s` through the
fixed-JVP route versus about `8.2 s` through the default BDF route. The new
subphase diagnostics show why: out of `56.86 s` in JVP Jacobian construction,
about `36.82 s` is repeated `jax.linearize` work and about `20.02 s` is the
batched tangent push, while tangent construction and sparse assembly are
negligible. This proves the fixed-layout/JVP seam is numerically aligned on the
compact gate, while the remaining promotion blocker is repeated JAX
linearization/JVP Jacobian construction inside the SciPy BDF callback.

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

The first fixed-layout source kernel is now in-tree as well:
`fixed_layout_hydrogen_reaction_sources` returns array-only ionisation,
recombination, and same-isotope charge-exchange sources for the hydrogenic
reaction block. It is parity-tested against the existing dictionary-oriented
reaction path and has direct `jit`/`grad` coverage. This is the template for
the next residual ports: add fixed layouts with parity gates first, then wire
them into the full recycling solve.

That template has now been widened to the multispecies D/T/He block used by the
heavy recycling runtime lane. `fixed_layout_dthe_reaction_sources` returns
stacked neutral, ion, and electron source arrays for D, T, He ionisation and
recombination plus D-D, T-T, D-T, and T-D charge exchange. It is parity-tested
against the existing dictionary path on the local Hermès `1D-recycling-dthe`
deck and has direct `jit`/`grad` coverage. This is the concrete bridge needed
for the packed recycling residual: diagnostics-free D/T/He residual calls now
use this fixed-layout kernel, while the full dictionary path remains available
for reported reaction diagnostics. Focused tests lock dictionary parity,
charge-exchange multiplier handling, species-specific floor handling, and the
packed-RHS diagnostics flag. The remaining production step is broader than
reaction sources: move the surrounding collision, diffusion, target-recycling,
and BDF residual assembly into the same JAX-transformable PyTree style.

The first fixed-layout residual container is now also in-tree as
`RecyclingFixedState` in
[recycling_fixed_residual.py](../src/jax_drb/native/recycling_fixed_residual.py).
It stores active-domain field blocks and controller scalars as a JAX PyTree and
provides transformable backward-Euler and BDF2 residual builders. Focused tests
check active recycling pack/unpack under `jax.jvp`, the fixed-layout residual
Jacobian action, the host-oracle bridge against the D/T/He deck, and the new
active-array RHS adapter that lets source/closure/boundary terms enter without
full guard-cell dictionary reconstruction. This is the state-layout bridge for
the heavy residual migration; it is not yet a claim that the full
Hermès-compatible recycling history is end-to-end differentiable.
The latest bridge test runs this fixed state on the actual Hermès
`1D-recycling-dthe` deck: it reconstructs full guard-cell fields, calls the
current packed RHS oracle through `build_fixed_host_rhs_bridge`, and verifies
the packed RHS and backward-Euler residual value against the legacy packed
path. That establishes an implementation-level parity seam for the next
term-by-term ports without hiding the remaining host barrier.

The newest core-transport slice moves another residual dependency into the
JAX-native lane. The open-field parallel advection operator
`div_par_mod_open` and parallel inertia operator `div_par_fvv_open` now keep
JAX inputs on the JAX backend, and the ion/electron RHS-term assemblers preserve
JAX arrays through transport, pressure-gradient, source, and soft-floor terms.
The gates are not smoke tests: they compare JAX and NumPy operator values and
compare `jax.jvp` tangents with centered finite differences. The full heavy
recycling transient still calls those assemblers through host-oriented
dictionary plumbing, but the mathematical kernels needed by the fixed-layout
residual are no longer NumPy-only.
The electron-force-balance pressure-gradient stencil used to build `Epar` has
also been moved from a Python loop to vectorized backend-preserving code and is
covered by the same JVP-versus-finite-difference standard.
The recycling RHS assembly boundary now passes ion/electron source and
transport arrays into these backend-preserving assemblers directly instead of
coercing them through `np.asarray` first; the remaining full-solve barriers are
therefore increasingly localized to species preparation, neutral RHS assembly,
and the host-backed nonlinear solve.
Neutral RHS assembly has now been moved into the same term-object pattern as
the ion and electron paths. Density transport, pressure advection/divergence,
neutral pressure-source override semantics, momentum inertia, pressure
gradient, and momentum-error addition all have NumPy parity and JVP
finite-difference gates. This narrows the remaining heavy recycling residual
work to the full-field species-preparation layer, source/closure accumulation,
and the host/SciPy nonlinear driver.
The species-preparation layer has now also started moving across that boundary.
`prepare_species_state`, `safe_temperature`, `raw_species_velocity`, target
guard merging, and the neutral target/no-flow guard path preserve JAX inputs
and have focused JVP coverage. The remaining full residual barrier is therefore
less about individual field algebra and more about the surrounding
dictionary-oriented closure orchestration and nonlinear solve driver. The
source-accumulation dictionaries are now less of a backend barrier: source
zeros, additive updates, and source overrides use a shared backend-preserving
helper with JVP coverage, so future fixed-layout residual ports can reuse the
current source composition order without an initial NumPy conversion. The
boundary-free open-field state wrapper now also keeps electron-density
reconstruction and electron/ion boundary-state construction on the JAX backend,
which provides a transformable control surface before the more parity-sensitive
sheath formulas are ported. Electron parallel force balance and the
corresponding ion electric-force source additions have likewise moved into a
backend-preserving RHS helper with a JVP gate, removing another NumPy-only
block between accumulated sources and the final ion momentum RHS.

The corresponding paper/docs artifact is:

- [atomic_rate_differentiability_campaign.md](atomic_rate_differentiability_campaign.md)

The latest boundary-kernel steps also isolate the simple ion Bohm-sheath guard
and energy-source formula, the full electron sheath response after the
zero-current potential is known, and the full ion sheath Bohm/energy response
in backend-preserving helpers. The existing simple and full sheath branches
call those helpers, and the open-field tests compare NumPy and JAX values plus
JVPs against centered finite differences. The remaining host-oriented sheath
barrier is now the surrounding orchestration: no-flow guard application,
zero-current ion-sum reconstruction, full-field dictionary plumbing, and the
Hermès parity gates that decide when those pieces can move into the fixed
active-array residual.

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

The heavier D/T/He fixed-layout recycling residual has now also been profiled
on the same GPU host. The CPU gate has `950` active variables, reaches residual
`2.41e-11`, and the current all-local profiling bundle completes the
skip-cProfile run in about `1.51 s`. The first retained GPU run reaches the
same residual with two visible CUDA devices, completes in about `13.21 s`, and
samples peak process-tree RSS near `1.49 GiB`; a second run with the same
persistent compilation cache completes in about `6.66 s` with
sampled peak RSS near `1.43 GiB`. This is useful accelerator evidence for the
fixed-layout residual seam, but it is deliberately not described as GPU
speedup: the current gate is still too small and launch/compile dominated.
The next GPU claim must come from a larger transformed residual or a batched
ensemble of independent residual solves.

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
- the SciPy BDF recycling callback now honors the same environment variable
  while keeping its default serial, which avoids oversubscription on one-off
  runs but lets MacBook users opt into multiple cores for heavy local runs;
- on the heavy `recycling_dthe_one_step` BDF path, local timing checks were
  effectively flat from serial to two Jacobian threads and slower at four
  threads, so this is not yet a strong-scaling path for one solve;
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

One concrete backend-policy fix is now part of that restructuring. Static
metric arrays are stored as JAX arrays, but they are not dynamic solve state.
The hot open-field operators therefore select JAX only from dynamic field,
source, closure, or rate arrays, and convert metrics inside the chosen branch.
That avoids eager-JAX execution on the current NumPy/SciPy BDF path while still
preserving JAX transforms when a future fixed-layout residual supplies JAX
state. On the D/T/He recycling lane, this restored warm packed-RHS calls to
about `4e-3 s` and the current bounded one-step timing to `44.60 s`.

A fresh full cProfile/RSS bundle after adding RHS phase counters to the
production BDF callback measured the `recycling_dthe_one_step` path at
`68.64 s` under cProfile and `48.82 s` on the separate unprofiled RSS replay,
with peak process-tree RSS about `228.9 MiB`. The new solver diagnostics from
the RSS replay show the physical BDF solve itself at `48.77 s`, with
`46.41 s` in fixed-layout RHS object evaluation, `33.60 s` in Jacobian
callbacks, and only about `2e-3 s` in RHS NumPy conversion. The top split is
therefore still decisive: sparse finite-difference Jacobian construction and
repeated host-side recycling RHS/source assembly dominate, while host
conversion is not the current limiter. This keeps the next optimization lane
focused on replacing the production SciPy BDF finite-difference Jacobian with
the fixed-layout JAX-linearized/JVP seam rather than per-solve CPU threading.
The same pass also validates a smaller hot-path cleanup: simplifying the JAX
backend selector reduces selector/type-detection overhead from a large cProfile
hotspot to about `5.15 s` cumulative, without changing the numerical solve or
the residual/Jacobian call counts.

The fixed-layout residual migration now has concrete boundary gates. The
electron sheath path uses backend-preserving no-flow state preparation plus the
current-free ion-sum/sheath-potential reconstruction, with NumPy/JAX parity
and JVP checks. Collision friction/heat exchange, neutral parallel diffusion,
and target recycling can also be staged through
`build_fixed_full_field_array_rhs`, which reconstructs guard-cell fields for
the kernel while keeping the public residual a static `RecyclingFixedState`.
Those gates now feed the same fixed-state bridge used by the production
backward-Euler, BDF2, and legacy BDF RHS paths, but the individual source and
boundary kernels still need to be ported term by term before the heavy SciPy
BDF path can be replaced by a fully JAX-linearized solve.

The latest residual-seam pass moves both the hydrogen and D/T/He recycling
one-step paths further down that migration. The species override,
collision-rate helpers, full electron and ion sheath orchestration,
target-adjacent feedback source, charge-exchange rate helpers, and packed-RHS
output now preserve JAX arrays when the fixed residual is traced. Real
`1D-recycling` and `1D-recycling-dthe` reference decks at small backward-Euler
steps now reach `solver_mode="jax_linearized"` without a host-array conversion
barrier, and the regression tests check that the linearized residual is
assembled through JAX. This is still a gate, not the default heavy transient
backend: the long production output-window solve remains on the SciPy BDF
compatibility path until the same residual is promoted to a production
JAX-linearized or matrix-free timestepper.

The hydrogen gate has a concrete profile bundle in
`docs/data/runtime_profile_artifacts/recycling_1d_jax_linearized_gate/`. The
local cProfile/JAX-trace run completed with residual `2.49e-12`, one JAX
linearization refresh, one residual evaluation, no fallback, and a separate RSS
timing of about `0.83 s`. This is now the reference evidence for the
transformable hydrogen BE residual seam.

The D/T/He gate has a separate profile bundle in
`docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate/`.
That run exercised the real 19-field multispecies deck with residual
`2.41e-11`, one JAX linearization refresh, one residual evaluation, no
fallback, and an RSS-sampled replay taking about `1.32 s` with an incremental
RSS increase of about `4.3 MiB` in the current skip-cProfile all-local bundle.
This is the current proof that the
multispecies fixed-layout residual seam is transformable. The heavier
`recycling_dthe_one_step` profile remains the production offender baseline.

The heavier real-kernel GMRES scaling pass uses the same D/T/He residual seam
with `mesh:ny=100` and `mesh:ny=200`, `timestep=1e-4`, and two nonlinear
iterations allowed. This forces a nontrivial JAX-linearized update and records
400 GMRES-equivalent iterations. Local CPU runs closed to `1.74e-12` and
`7.47e-11` in about `7.28 s` and `7.32 s`; office-GPU runs closed to the same
residuals in about `30.19 s` and `30.76 s`, with large shape-specific compile
warmups. The GPU memory delta is lower, but the current JAX GMRES heavy
recycling path is not a production speedup. This is why the release keeps the
full output-window BDF default on the stable finite-difference compatibility
path while retaining JAX-linearized/JVP modes as explicit development gates.

The production backward-Euler/BDF2 recycling steppers now build their nonlinear
residuals through the same fixed-layout state bridge. The default sparse path
still uses finite-difference Jacobians unless explicitly configured otherwise,
because several production RHS branches remain host-backed. Two promoted
JAX-native lanes are available for residuals that stay transformable:
`solver_mode="sparse_jvp"` materializes a sparse Jacobian from grouped JVPs,
and `solver_mode="jax_linearized"` sends JAX-linearized Jacobian actions
directly to GMRES. The adaptive BDF controller can route its trial BE/BDF2
steps through the same seam with
`solver_mode="adaptive_bdf_jax_linearized"` or
`solver_mode="adaptive_bdf_jax_linearized_lineax"`, while
`solver_mode="adaptive_bdf_sparse_jvp"` uses the sparse-JVP Jacobian path.
These variants are promoted as controlled solver gates, not yet as the default
production backend. The environment variable
`JAX_DRB_RECYCLING_JACOBIAN_MODE=jvp` can select the sparse-JVP Jacobian for
the standard sparse solver, and `JAX_DRB_RECYCLING_JVP_BATCH_SIZE` bounds the
color-group batch size. These modes should be used only on gates where the
residual has been proven JAX-transformable; the heavy SciPy BDF callback
remains a host compatibility path. For the long SciPy BDF callback itself,
`runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp` exercises
the fixed-full-field RHS plus grouped-JVP Jacobian seam while preserving the
same BDF timestepper.

The adaptive BDF history result now reports solver-health diagnostics for these
opt-in paths: accepted and rejected internal steps, minimum-`dt` fallbacks,
startup versus BDF2 trials, accepted-`dt` bounds, the last and maximum embedded
error ratios, and the step solver backend used by the controller. These fields
are intentionally exposed in `Recycling1DHistoryResult.diagnostics` so the
research campaigns can distinguish a genuinely stable output-window solve from
a run that only completed by falling back to the minimum internal timestep.
The matrix-free BE/BDF2 trial solvers also start from the same explicit
predictor used by the sparse and JAX-linearized paths, rather than from the
previous state. That keeps the native solver variants comparable and avoids an
unnecessary convergence penalty in the matrix-free lane.
The BE/BDF2 `jax_linearized` and `jax_linearized_lineax` modes now build their
residuals through the fixed full-field array adapter instead of the host
packing bridge; the sparse and default production modes intentionally remain on
the validated host-compatibility adapter until each heavier RHS term has passed
its transformability and parity gates.
Implicit step diagnostics now also include a `converged` flag when the solver
can determine whether its residual or step-tolerance criterion was satisfied;
validation campaigns should require this flag before promoting an opt-in solver
mode to a paper or release claim.
For adaptive BDF promotion attempts, the comparison script can now enforce the
same policy explicitly:
`compare_recycling_transient_modes.py --mode adaptive_bdf_jax_linearized
--require-adaptive-bdf-no-fallback --require-adaptive-bdf-max-error-ratio
0.95 --mode-timeout-seconds 120`. This gate is intentionally strict: a run
that completes only through minimum-`dt` fallback, exceeds the timeout, or
reports an embedded error ratio above the threshold is treated as evidence that
the output-window JAX-linearized path is not ready for default use.

There is also an optional Lineax evaluation seam for transformable gates:
`solver_mode="jax_linearized_lineax"` or
`JAX_DRB_RECYCLING_JAX_LINEAR_SOLVER=lineax` routes the JAX-linearized Newton
update through a Lineax GMRES `FunctionLinearOperator`. This is intentionally
not a required dependency and not the default; it gives a controlled way to
compare JAX-native Krylov backends once the residual itself is free of host
barriers.

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

The reachable `office` machine exposes two CUDA-visible JAX devices
(`RTX A4000`). The refreshed reduced-kernel audit was run there in a clean GPU
environment and records `backend="gpu"` with devices `cuda:0` and `cuda:1` in
the committed `jax_native_profile_audit` artifact. The two promoted reduced 3D
lanes remain tiny compared with the recycling transient backbone, but they now
provide a reproducible GPU trace bundle for the JAX-native geometry kernels.

The next GPU step is not to claim acceleration for the whole code from these
small kernels. It is to move heavier recycling residual pieces into the same
array-native contract, then rerun the profiling script with JAX traces, device
memory snapshots, and persistent compilation cache enabled on the GPU host.

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
