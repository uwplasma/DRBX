# Performance And Differentiability

!!! note "Plan authority"
    This page explains current performance and differentiability evidence. The
    active execution plan is
    [Research-Grade Execution Plan](research_grade_execution_plan.md). If this
    page conflicts with that plan, follow the execution plan and update this
    page afterward.

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
static packed layout, routes the RHS through `fixed_full_field_array` by
default, and checks the same residual under `jit`, `vmap`, `jvp`, and a
scalar-objective `grad`. This is the strongest current heavy-residual
differentiability evidence because it exercises the multispecies recycling
state rather than a synthetic diffusion objective. On the committed local CPU
run, the residual JVP agrees with a centered finite difference to about
`2.19e-9`, the objective directional derivative agrees to about `4.35e-8`,
and the retained batch sweep through 256 states reaches about `4.94x`
residual throughput speedup and `3.11x` JVP throughput speedup over
serial same-kernel calls.

The same profile gate now accepts `--rhs-backend active_array`. That backend is
not a new physics model; it is the opt-in migration seam that passes the
validated full-field recycling RHS through `build_fixed_array_rhs`, preserving
D/T/He backward-Euler and BDF2 residual parity while giving the next
term-by-term sheath, collision, neutral-diffusion, and target-recycling ports a
stable active-field surface.

The source-term lane now also has a dedicated accelerator-throughput gate:
`scripts/profile_atomic_rate_throughput_gate.py`. That gate evaluates a
batched AMJUEL/CX reaction-source surface, its reverse-mode derivative, and a
scalar log-temperature sensitivity objective. On the office GPU run, the
largest committed batch (`4,194,304` points) is about `2.5x` faster than the
local CPU run for the rate surface and about `2.0x` faster for the autodiff
derivative. The scalar sensitivity agrees with centered finite differences at
about `1e-10` relative error on both CPU and GPU. This is an accelerator
speedup claim for a source kernel, not for the full output-window recycling
solve. The optional multi-device branch first runs an identity-map `pmap`
sanity check and only reports pmap speedups after the real source-kernel parity
check passes, so broken self-hosted multi-device runtimes do not create
ambiguous performance claims.

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

The same-machine live reference matrix and follow-on term-balance reports now
sharpen those generic blockers into specific case priorities:

- `neutral_mixed_one_step`
  - the older live matrix exposed `NVh` as the visible offender, but the
    current term-balance campaign has since closed the direct pressure-gradient
    and viscosity source formulas against written reference diagnostics
  - the remaining issue is target-adjacent state/history reconstruction and
    release-media refresh, not a missing `NVh` operator formula
  - native/reference wall-time ratio now about `4.19x`
- `recycling_1d_one_step`
  - worst normalized RMS mismatch about `4.62e-3`
  - native/reference wall-time ratio about `3.95x`
  - dominant normalized field: `Pd+`
- `recycling_dthe_one_step`
  - worst normalized RMS mismatch about `4.92e-3`
  - native/reference wall-time ratio now about `7.17x`
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
adapter, and forces the sparse Jacobian callback through grouped JVPs. A second
diagnostic bridge,
`runtime:recycling_transient_solver_mode=bdf_active_array_jvp`, routes the same
SciPy-BDF/JVP callback through the newer `build_fixed_array_rhs` active-array
surface. The history diagnostics record
`bdf_rhs_backend="fixed_full_field_array"` or `bdf_rhs_backend="active_array"`
and `bdf_jacobian_mode="jvp"` so profile artifacts identify the route
explicitly. The active-array bridge is intentionally opt-in through
`scripts/run_recycling_jvp_promotion_gate.py --include-active-array-jvp`: the
current local fixture run exceeded the stable BDF baseline by more than an
order of magnitude before completing that mode. Both routes therefore remain
parity/runtime experiments rather than defaults until full output-window
campaigns show equal reference agreement with lower call count and memory use.
The current self-contained gate is:
`PYTHONPATH=src python scripts/run_recycling_jvp_promotion_gate.py`. It runs the
hydrogen and D/T/He one-step decks from the committed lightweight fixture root
unless `JAX_DRB_REFERENCE_ROOT` or `--reference-root` points to a live reference
checkout. Add `--output-dir docs/data/runtime_profile_artifacts/<new-run>` for
release or paper audits; the wrapper writes one JSON report per case and gate
phase plus an aggregate `summary.json` with commands, thresholds, return codes, timings,
diagnostics, and pairwise deltas. Use `--mode-timeout-seconds <seconds>` for
bounded experimental probes, especially when adding
`--include-active-array-jvp`, so a slow migration route does not consume a full
promotion run. The latest two-output-window hydrogen gate
passes with worst active-mesh BDF-vs-fixed-JVP delta `7.20e-6` under the
`1e-5` threshold; the refreshed two-output-window D/T/He gate passes with worst
delta `1.02e-6` under the `2e-5` threshold. The
default wrapper still runs the stable BDF/JVP compatibility bridge and requires
it to report the expected RHS backend, `bdf_jacobian_mode="jvp"`, zero
finite-difference base-RHS Jacobian calls, positive JVP RHS calls, and prebuilt
direction-batch reuse. The full-output parity phase intentionally stays separate
from the non-SciPy fixed-BDF2 phase because the raw fixed-BDF2 stepper is not
validated at the production output cadence. The experimental
`--include-active-array-jvp`
flag adds the active-array bridge to the same diagnostics, but current timeout
evidence keeps it out of the default gate.
For cases with a validated bounded step, the wrapper also runs the non-SciPy
`fixed_bdf2_jax_linearized` lane with an explicit `--timestep` override and
requires `fixed_bdf2_fixed_full_field_rhs_steps > 0`,
`fixed_bdf2_jax_linearized_action_steps > 0`,
`fixed_bdf2_evolve_feedback_integrals=true`, `fixed_bdf2_bdf2_steps > 0`, zero
failed or unknown solver-status counters, and a finite residual norm below the
configured threshold. The default bounded fixed-BDF2 diagnostic currently runs
on the hydrogen fixture at `timestep = 10` and on the D/T/He fixture at
`timestep = 1` with the explicit opt-in override
`runtime:recycling_fixed_bdf2_max_internal_timestep=0.5`. That override splits
each output interval into two internal implicit substeps while keeping the same
stored output cadence.

The June 5, 2026 two-output-window `recycling_1d_one_step` local gate passed
with worst active-mesh `bdf` versus `bdf_fixed_full_field_jvp` delta
`7.20e-6`, below the `1e-5` threshold. The JVP bridge reported one prebuilt
direction batch and reused it on all `106` Jacobian callbacks; tangent
construction time was zero, while JAX linearization and device execution still
accounted for most of the `62.7 s` fixed-JVP runtime versus `9.07 s` for the
default BDF route. The same run also proved that `fixed_bdf2_jax_linearized`
executes a real BDF2 corrector (`fixed_bdf2_bdf2_steps=1`), but its maximum
residual norm was about `1.93e29`. That lane is therefore useful for routing,
packing, and action-count diagnostics, but not yet for production promotion.
The JVP path removes finite-difference sensitivity and preserves parity, but
repeated `jax.linearize` and batched tangent pushes inside the SciPy BDF
callback are still more expensive than the current finite-difference sparse
Jacobian for these decks.

That bridge now follows the documented JAX autodiff pattern more closely:
`jax.linearize` evaluates the primal residual once and returns a reusable
linear map, while `jax.vmap` batches the colored tangent pushes. The default
path pushes all color groups in one vectorized batch; `batch_size=1` gives the
memory-bounded serial form, and intermediate batch sizes provide a knob between
temporary tangent memory and dispatch overhead. The BDF bridge now prebuilds
those colored tangent batches once per solve rather than on every SciPy
Jacobian callback. This removes finite-difference step-size sensitivity for
JAX-transformable residuals and gives a direct operator-level check against the
older sparse finite-difference builder, but it does not remove the larger
linearization and tangent-push cost.

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

The adaptive-BDF route follows the same promotion policy. The stable production
default remains the validated BDF compatibility path. The single-species
`recycling_1d_one_step`, `timestep=1.0` JAX-linearized adaptive-BDF gate passes
with zero fallback, zero unconverged substeps, and
`adaptive_bdf_max_accepted_error_ratio=9.315e-1`; the optional Lineax backend
uses the same controller history and is only a backend comparison. The first
passing D/T/He adaptive-BDF promotion-style result is narrower still: it uses
the opt-in sparse-JVP adaptive-BDF route with component-wise density, pressure,
and momentum absolute-tolerance floors. That diagnostics-only gate is useful
evidence for the migration seam, but it does not promote adaptive BDF as the
default until longer output-window reference-parity and performance campaigns
pass on the same route.

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
Jacobian action, batched linearized tangent pushes against dense and serial JVP
oracles, the host-oracle bridge against the D/T/He deck, and the new active-array
RHS adapter that lets source/closure/boundary terms enter without full
guard-cell dictionary reconstruction. This is the state-layout bridge for the
heavy residual migration; it is not yet a claim that the full Hermès-compatible
recycling history is end-to-end differentiable.
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
  - `1.94x` from `1 -> 2` workers
  - `3.32x` from `1 -> 4` workers
  - `4.79x` from `1 -> 8` workers
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

The production backward-Euler/BDF2 recycling steppers now expose a
fixed-layout state bridge for opt-in JAX-transformable nonlinear residuals. The
default sparse path still uses the host-compatible finite-difference Jacobian
route, because several production RHS branches remain host-backed. Three
promoted JAX-native lanes are available for residuals that stay transformable:
`solver_mode="sparse_jvp"` builds the BE/BDF2 residual through the fixed
full-field adapter and materializes a sparse Jacobian from grouped JVPs, while
`solver_mode="jax_linearized"` and `solver_mode="jax_linearized_lineax"` send
JAX-linearized Jacobian actions directly to GMRES. The adaptive BDF controller
can route its trial BE/BDF2 steps through the same seam with
`solver_mode="adaptive_bdf_sparse_jvp"`,
`solver_mode="adaptive_bdf_jax_linearized"` or
`solver_mode="adaptive_bdf_jax_linearized_lineax"`. The active-array
counterparts,
`solver_mode="adaptive_bdf_active_array_jax_linearized"` and
`solver_mode="adaptive_bdf_active_array_jax_linearized_lineax"`, use the same
adaptive controller but route the implicit trial residual through
`build_fixed_array_rhs` instead of the fixed-full-field compatibility adapter.
These variants are promoted as controlled solver gates, not yet as the default
production backend. The environment variable
`JAX_DRB_RECYCLING_JACOBIAN_MODE=jvp` can select the sparse-JVP Jacobian for
the standard sparse solver, and `JAX_DRB_RECYCLING_JVP_BATCH_SIZE` bounds the
color-group batch size. The JAX-linearized BE/BDF2 and adaptive-BDF trial
solves also expose opt-in Krylov controls through
`runtime:recycling_jax_linear_restart`,
`runtime:recycling_jax_linear_maxiter`, or the matching environment variables
`JAX_DRB_RECYCLING_JAX_LINEAR_RESTART` and
`JAX_DRB_RECYCLING_JAX_LINEAR_MAXITER`; these keep the default `20 x 20`
GMRES-equivalent budget unchanged but allow bounded speed/accuracy sweeps on
heavy multi-ion gates. Profiling decks can also set
`runtime:recycling_jax_linear_jit_residual=true` or
`JAX_DRB_RECYCLING_JAX_LINEAR_JIT_RESIDUAL=1` to JIT-wrap the fixed-layout
residual before the initial nonlinear check and `jax.linearize` call. These
modes should be used only on gates where the
residual has been proven JAX-transformable; the heavy SciPy BDF callback
remains a host compatibility path. For the long SciPy BDF callback itself,
`runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp` exercises
the fixed-full-field RHS plus grouped-JVP Jacobian seam while preserving the
same BDF timestepper. Use
`runtime:recycling_transient_solver_mode=bdf_active_array_jvp` for the matching
active-array migration seam; it is diagnostic evidence for the future PyTree RHS
default, not a faster production mode yet.
The next non-SciPy output-window promotion lane is
`runtime:recycling_transient_solver_mode=fixed_bdf2_jax_linearized`,
`fixed_bdf2_jax_linearized_lineax`, or
`fixed_bdf2_active_array_jax_linearized`, with matching `_lineax` variants.
These modes take a fixed-layout backward-Euler startup step, then fixed-layout
BDF2 output steps, and evolve controller integrals inside the packed residual
state. The active-array variants route the same output-window path through
`build_fixed_array_rhs`, avoiding the full-field reconstruction seam used by
the compatibility fixed-full-field variant. They are still opt-in research
gates; they exist to measure JAX-linearized full-output behavior without the
`solve_ivp` callback barrier before any default solver change. The promotion
gate now also rejects unconverged steps, unknown convergence status, failed
inner linear solves, and fixed-BDF2 residuals above the configured threshold, so
large finite residuals cannot pass as healthy diagnostics. On the local
`recycling_1d_one_step` diagnostics-only fixture at `timestep = 10`, the
`fixed_bdf2_active_array_jax_linearized` route passes this gate with
`fixed_bdf2_max_residual_inf_norm = 4.02e-6`, two active-array RHS steps, zero
unconverged steps, zero unknown-convergence steps, and zero failed linear
solves. The D/T/He fixture now also has a bounded fixed-BDF2 diagnostic at
`timestep = 1` using two internal `0.5` substeps per output interval. Both
fixed-full-field and active-array routes pass there with
`fixed_bdf2_max_residual_inf_norm = 3.77e-9`, four internal substeps, three
BDF2 corrector steps, zero unconverged steps, zero unknown-convergence steps,
and zero failed linear solves. The same diagnostic is intentionally not a
runtime win yet: the local fixed-full-field run took `98.9 s` and the
active-array run took `117.6 s`, both dominated by `5200` inner linear
iterations. A follow-up active-array GMRES-control probe with
`runtime:recycling_jax_linear_restart=10`,
`runtime:recycling_jax_linear_maxiter=20`, and the same internal substep
policy converged cleanly but slowed to `136.8 s`, with `100.1 s` in linear
solves and `36.5 s` in residual evaluations. This negative result rules out
simple restart reduction as the next promotion path; the next runtime work
needs a real preconditioner, a cheaper residual/JVP kernel, or a better
startup/nonlinear damping policy. On the hydrogen fixture at the full
`timestep = 5000`, both
fixed-full-field and active-array fixed-BDF2 routes currently expose the same
large nonlinear residual (`fixed_bdf2_max_residual_inf_norm` about `1.93e29`),
so the next promotion blocker is fixed-BDF2 nonlinear/linear solver efficiency
and full-output-window substepping policy rather than an active-array RHS
parity failure.

The adaptive BDF history result now reports solver-health diagnostics for these
opt-in paths: accepted and rejected internal steps, minimum-`dt` fallbacks,
startup versus BDF2 trials, accepted-`dt` bounds, the last and maximum embedded
error ratios, the step solver backend used by the controller, and route
provenance counters for `fixed_full_field_array` versus `host_bridge`
residuals, the active-array RHS path, sparse-JVP Jacobian steps,
finite-difference Jacobian steps, and JAX-linearized action steps. It also
aggregates wall-clock buckets for startup
trials, backward-Euler predictor solves, BDF2 corrector solves, embedded-error
estimation, residual evaluation, Jacobian/linearization, Krylov solves, and
line search, plus residual-evaluation, Jacobian-refresh, and linear-iteration
counts. `adaptive_bdf_linear_solver_failed_steps` is a promotion blocker for
JAX-linearized backends because it means the nonlinear step may have continued
after an inner Krylov solve reported breakdown or non-finite output. Set
`JAX_DRB_RECYCLING_ADAPTIVE_BDF_TRACE_JSONL=/path/to/trace.jsonl` to emit a
flushed start/end record around each expensive adaptive-BDF implicit trial;
this is the preferred diagnostic for timeout runs where final
`Recycling1DHistoryResult.diagnostics` would otherwise be lost. These fields
are intentionally exposed in
`Recycling1DHistoryResult.diagnostics` so the research campaigns can
distinguish a genuinely stable output-window solve from a run that only
completed by falling back to the minimum internal timestep or silently taking
the wrong residual/Jacobian route.
The adaptive BDF controller uses a conservative accepted-step threshold of
`0.95` for the embedded error ratio and a timestep safety factor of `0.85`.
Those values are deliberately aligned with the promotion gate rather than the
looser mathematical `error <= 1` acceptance boundary, because the JAX-linearized
path is still an opt-in research backend.
For JAX-linearized adaptive modes, including the active-array route, the
default first internal step is also damped to at most one sixteenth of the
requested output window. An explicit
`runtime:recycling_adaptive_bdf_initial_dt` or legacy
`jax_drb:recycling_adaptive_bdf_initial_dt` setting still takes precedence.
This change targets the startup rejected-trial cost observed in the active
array bridge without changing the validated sparse production default.
The matrix-free BE/BDF2 trial solvers also start from the same explicit
predictor used by the sparse and JAX-linearized paths, rather than from the
previous state. That keeps the native solver variants comparable and avoids an
unnecessary convergence penalty in the matrix-free lane.
For JAX-linearized adaptive-BDF modes, the BDF2 corrector now also reuses the
just-computed backward-Euler embedded predictor as its initial guess by
default. This is controlled by
`runtime:recycling_bdf2_use_be_initial_guess` or
`JAX_DRB_RECYCLING_BDF2_USE_BE_INITIAL_GUESS`. The June 15, 2026 hydrogen
`recycling_1d_one_step`, `timestep=1.0` gate completed with this path enabled
in `106.3 s`, used `42` fixed-layout JAX-linearized trial solves, took `19`
accepted internal steps, rejected one trial, and reported zero failed or
unconverged implicit substeps. The retained artifact is
`docs/data/runtime_profile_artifacts/recycling_1d_adaptive_bdf_be_initial_guess_gate.json`.
This is useful cleanup and clean convergence evidence, but it is not a
meaningful speedup: the same run still spent `85.4 s` in Krylov linear solves
and used `16,800` inner linear iterations.
The JAX-linearized solver now also records and exposes a separate inner Krylov
tolerance, controlled by `runtime:recycling_jax_linear_tolerance_factor` or
`JAX_DRB_RECYCLING_JAX_LINEAR_TOLERANCE_FACTOR`. This preserves the outer
nonlinear/adaptive-BDF gates while allowing bounded inner-solve sweeps. On the
same hydrogen `timestep=1.0` gate, factor `10` completed cleanly in `103.9 s`
with zero failed substeps and the same accepted-error bound, while factor `100`
completed in `105.3 s`. Both runs still used the same `42` JAX-linearized
trial solves and remained Krylov dominated, so factor `10` is the current best
bounded probe rather than a default-promotion result.
The current multi-ion traces show a clear split between solver routes:
JAX-linearized GMRES is dominated by Krylov linear solves, Lineax reports inner
linear-solver breakdown on most completed substeps, and sparse-JVP reaches BDF2
with millisecond sparse linear solves but spends almost all time assembling
grouped-JVP Jacobians. This makes sparse-JVP the near-term optimization lane for
multi-ion adaptive BDF, while full JAX-linearized matrix-free promotion remains
blocked on preconditioning. The same JSONL trace now records per-field and
feedback-integral contributors for embedded-error estimates when tracing is
enabled. On the D/T/He sparse-JVP probe, those contributors show that the
adaptive-BDF rejection is driven by ion parallel-momentum fields (`NVd+`,
`NVt+`, and `NVhe+`), not by the recycling feedback integrals. That result
initially narrowed the next promotion work to momentum-field startup/norm
scaling and Jacobian reuse/batching instead of generic controller loosening.
The follow-up raw-scale probe showed why the ratios were so large: the default
single absolute tolerance leaves near-zero momentum cells with scales as small
as `1e-12`. The opt-in
`runtime:recycling_adaptive_bdf_momentum_atol_floor` setting, also available as
`JAX_DRB_RECYCLING_ADAPTIVE_BDF_MOMENTUM_ATOL_FLOOR`, reduced the short
D/T/He startup ratios by about two orders of magnitude but shifted the dominant
error to low-density/pressure fields. The remaining default-promotion work is
therefore a component-wise adaptive norm, not a one-field tolerance override.
The first local component-wise opt-in gate used density, pressure, and momentum
floors of `1e-6`, `1e-3`, and `1e-2`, respectively, together with
`runtime:recycling_adaptive_bdf_initial_dt=0.0625`. It completed the D/T/He
`timestep=1.0` diagnostics-only sparse-JVP adaptive-BDF gate in `28.6 s` with
no rejected accepted-window steps, no minimum-dt fallback, no unconverged
substeps, and a maximum accepted embedded-error ratio of `0.715`. That is
promotion evidence for the opt-in route only; the stable default remains
unchanged until full output-window parity, reference campaigns, and larger
CPU/GPU profiling reproduce the same behavior.
The subsequent sparse-JVP workspace pass moves static sparse metadata and
direction-batch construction out of the adaptive BE/BDF2 trial loop. The same
bounded D/T/He gate reports `adaptive_bdf_sparse_jvp_workspace_reuses=17`,
matching its 17 sparse-JVP trial solves, while preserving the accepted-step and
embedded-error behavior. With `JAX_DRB_SPARSE_JVP_SYNC_TIMING=1`, its local
wall time is about `29.1 s`; the JVP Jacobian path spends `17.8 s` in JAX
linearization and `10.1 s` in grouped-push device execution, while host
transfer is only `1.6e-4 s` and sparse assembly is only `5.0e-3 s`. The next
performance target is therefore residual linearization and batched push
execution inside grouped-JVP Jacobian assembly, not sparse-plan allocation or
host/device transfer.
The sparse assembly loop now also writes gathered JVP rows directly into the
final COO data buffer rather than allocating a temporary per color group. This
keeps the opt-in sparse-JVP path memory-stable during larger color sweeps, but
it is intentionally not claimed as a solver-promotion result because the
measured blocker remains `jax.linearize` plus grouped tangent pushes.
The adaptive controller now keeps BDF2 history across timestep changes using
the variable-step BDF2 residual
`U^{n+1} - a_1 U^n + a_0 U^{n-1} - b Δt R(U^{n+1})`, with
`a_1=(r+1)^2/(2r+1)`, `a_0=r^2/(2r+1)`, `b=(r+1)/(2r+1)`, and
`r=Δt_n/Δt_{n-1}`. This reduces unnecessary backward-Euler restarts after
rejected trial steps while preserving the constant-step BDF2 formula at
`r=1`.
The BE/BDF2 `sparse_jvp`, `jax_linearized`, and `jax_linearized_lineax` modes
now build their residuals through the fixed full-field array adapter instead
of the host packing bridge; the default `sparse` production mode intentionally
remains on the validated host-compatibility adapter until each heavier RHS term
has passed its transformability and parity gates.
Implicit step diagnostics now also include a `converged` flag when the solver
can determine whether its residual or step-tolerance criterion was satisfied;
validation campaigns should require this flag before promoting an opt-in solver
mode to a paper or release claim.
For adaptive BDF promotion attempts, the comparison script can now enforce the
same policy explicitly:
`compare_recycling_transient_modes.py --mode adaptive_bdf_jax_linearized
--diagnostics-only --timestep 1.0 --max-nonlinear-iterations 3
--require-adaptive-bdf-no-fallback
--require-adaptive-bdf-no-unconverged-substeps
--require-adaptive-bdf-max-accepted-error-ratio 0.95 --mode-timeout-seconds
480`. The diagnostics-only flag is important for this bounded promotion check
because the committed one-step array baselines are generated at the full
reference output time. A shortened timestep is useful for solver-health
testing, but it is not a replacement for a full parity run. This gate is
intentionally strict: a run that completes only through minimum-`dt` fallback,
exceeds the timeout, reports any unconverged implicit BE/BDF2 substep, or
accepts an embedded error ratio above the threshold is treated as evidence that
the output-window JAX-linearized path is not ready for default use. Rejected
trial error ratios remain available as `adaptive_bdf_max_error_ratio` for
controller diagnostics, while `adaptive_bdf_max_accepted_error_ratio` is the
promotion gate.
With these checks, the local single-species gate has now been extended to a
`timestep=1.0` diagnostic output window on the reference recycling deck. The
current variable-step BDF2 controller completed the in-tree JAX GMRES run in
about `108 s`, took `21` accepted substeps and `3` rejected trials, reported
`50` implicit trial solves, reused valid BDF2 history after `2` rejected
trials, accepted `20` BDF2 correctors, and had zero fallback, zero unconverged
substeps, zero failed linear solves, and
`adaptive_bdf_max_accepted_error_ratio=9.315e-1`. Earlier retained artifacts
for the same gate needed `61` trial solves and about `174 s`, while the older
constant-step-history-reset controller needed `207` trial solves and about
`259 s`. The rejected-history reuse policy therefore removes restart overhead
without loosening the embedded-error acceptance policy. On the same
`timestep=1.0` gate, `adaptive_bdf_jax_linearized_lineax` now ran in about
`91 s` but reported `41` failed inner linear solves; it is faster but remains
negative promotion evidence until the backend reports clean linear convergence.
The alternative JAX BiCGSTAB backend is available through
`JAX_DRB_RECYCLING_JAX_LINEAR_SOLVER=bicgstab`; on the same hydrogen gate it
ran in about `108 s`, essentially matching JAX GMRES, and local JAX reports no
inner success flag for this solver, so diagnostics count those linear updates
as unknown-status rather than promotion-clean. A bounded `1D-recycling-dthe`
adaptive JAX-linearized run with the same controller policy still exceeded a
`360 s` guard before producing a completed mode report.
The JAX GMRES result is therefore the current bounded solver-health reference,
not a default-production solver claim, because the committed reference one-step
deck still uses the full `timestep=5000` output interval and the D/T/He heavy
case must pass the same parity/runtime gates.
The active-array version of the same bounded gate now uses the damped
JAX-linearized startup step by default and completed locally in about `103 s`,
with `21` accepted substeps, `2` rejected trials, `49` active-array
RHS/JAX-linearized trial solves, zero fallback, zero unconverged implicit
substeps, and `adaptive_bdf_max_accepted_error_ratio=9.315e-1`. This improves
on the earlier `161 s`, `6`-rejection, `61`-trial route-health run, but it
still remains opt-in until the full output-window and D/T/He heavy cases pass
the same parity/runtime gates.

There is also an optional Lineax evaluation seam for transformable gates:
`solver_mode="jax_linearized_lineax"` or
`JAX_DRB_RECYCLING_JAX_LINEAR_SOLVER=lineax` routes the JAX-linearized Newton
update through a Lineax GMRES `FunctionLinearOperator`. This is intentionally
not a required dependency and not the default; it gives a controlled way to
compare JAX-native Krylov backends once the residual itself is free of host
barriers. Current promotion gates must treat Lineax speedups as unusable when
`adaptive_bdf_linear_solver_failed_steps` is nonzero.
The native JAX BiCGSTAB backend is similarly opt-in; it is useful as an
algorithmic probe but not a current speedup lane on the retained hydrogen gate.

The JAX-GMRES path also has opt-in preconditioner hooks through
`runtime:recycling_jax_linear_preconditioner=<name>` or
`JAX_DRB_RECYCLING_JAX_LINEAR_PRECONDITIONER=<name>`. The current diagnostic
names are `state_scale`, which scales each packed residual row by the matching
initial-state magnitude, and `field_scale`, which scales each fixed-layout
field block by a conservative RMS field scale while leaving feedback rows
separate. The `field_diag`/`field_jacobi` option is the cheapest
JVP-derived field-active diagnostic: after `jax.linearize`, it samples only the
active field-block diagonal entries and leaves feedback scalars unscaled. The
`local_block_diag`/`block_jacobi` option is a stronger physics preconditioner
probe: after `jax.linearize`, it builds same-cell dense field-by-equation
Jacobian blocks with batched JVPs, inverts those small blocks on device, and
leaves transport/off-cell coupling to the outer JAX GMRES iteration. The
`parallel_line`/`transport_line` option is the next
transport-aware probe: it extracts JVP-derived dense blocks along the active
parallel line for all evolved fields, solves those line blocks on device, and
leaves feedback variables plus off-line coupling to the outer Krylov update.
The companion
`runtime:recycling_jax_linear_preconditioner_refresh=<n>` or
`JAX_DRB_RECYCLING_JAX_LINEAR_PRECONDITIONER_REFRESH=<n>` control lets a run
reuse this approximate dynamic preconditioner across Newton updates inside one
implicit solve. The same runtime surface exposes bounded-build controls:
`runtime:recycling_jax_linear_preconditioner_floor=<x>` sets the diagonal or
block regularisation floor, `runtime:recycling_jax_linear_preconditioner_max_field_unknowns=<n>`
caps the `field_diag` JVP build, and
`runtime:recycling_jax_linear_preconditioner_max_local_unknowns=<n>` caps the
`local_block_diag` JVP build. The matching environment variables use the
uppercase `JAX_DRB_RECYCLING_JAX_LINEAR_PRECONDITIONER_*` names. These are
diagnostic seams, not promoted accelerators. On the
June 15, 2026 local hydrogen fixed-layout gate, the unpreconditioned solve ran
in `2.96 s`, while `state_scale` ran in `7.25 s` with the same residual norm
and solver status. A follow-up `field_scale` fixed-layout probe at
`timestep=1.0` ran in `3.38 s`, reached the full `400` GMRES update budget,
and did not improve the residual. On the adaptive hydrogen
`adaptive_bdf_jax_linearized`, `timestep=1.0` gate, `field_scale` completed
cleanly but slowed to `111.8 s` and reported `9` unknown inner linear statuses,
whereas the retained unpreconditioned run is about `108 s` with clean solver
status. A lower `maxiter=8` JAX-GMRES budget also slowed to `116.6 s`, and
`runtime:recycling_jax_linear_jit_residual=true` exceeded a `220 s` guard on
the same adaptive-BDF gate. A later bounded backward-Euler hydrogen probe gives
the narrower positive case for keeping the seam alive: with one warmup solve,
the default three-run median was `4.64 s`, while the jitted-residual route ran
in `3.02 s` with identical residual norm and lower reported linear-solve time
(`1.73 s` versus `3.47 s`). This means residual JIT is a useful fixed-layout
profiling seam but still not a default production route until the
adaptive-BDF/output-window gates pass with the same solver-health checks. The
solver also exposes a bounded
`linearized_diag` diagnostic that builds an exact JVP-derived Jacobian diagonal
after `jax.linearize`; on the same fixed-layout `timestep=1.0` gate it ran in
`3.66 s`, spent `0.36 s` building the diagonal, and still reached the full
`400` GMRES update budget. A June 18, 2026 bounded hydrogen one-step probe
verified the `field_diag` runtime surface and required-preconditioner gate
(`linear_preconditioner=field_diag`, one build, `0.44 s` build time, two
residual evaluations, five linear-operator calls, solver status `0`), but it
did not improve that tiny same-case runtime (`7.37 s` versus `6.95 s`
unpreconditioned with the same linear budget). The local-block probe is correct
but not a current default speedup: the matched two-step fixed-BDF2 hydrogen
gate ran in `13.15 s` without preconditioning, `13.27 s` with local blocks
rebuilt on every nonlinear update, and `13.02 s` with block reuse. On the full
adaptive hydrogen
`timestep=1.0` gate, rebuilding local blocks completed in `137.2 s`, while
reusing blocks inside each implicit solve completed in `113.5 s`; both passed
the fallback/convergence/error gates, but both remain slower than the retained
unpreconditioned tolerance-factor gate at `103.9 s`. A June 18, 2026 bounded
hydrogen JAX-linearized probe also keeps `parallel_line` opt-in rather than
promoted: the unpreconditioned control took `5.02 s`, while
`runtime:recycling_jax_linear_preconditioner=parallel_line` took `5.62 s` with
the same residual, same `800` JAX GMRES update count, and `0.52 s` spent
building two line-block preconditioners. The line-block builder now batches
multiple field-line blocks per JVP launch, with the bounded runtime control
`runtime:recycling_jax_linear_preconditioner_max_batch_unknowns=<n>` or
`JAX_DRB_RECYCLING_JAX_LINEAR_PRECONDITIONER_MAX_BATCH_UNKNOWNS=<n>`. This
improves the implementation path for multi-line 2D/3D transport
preconditioning, but it does not change the 1D conclusion because the hydrogen
gate contains only one parallel line. A same-worktree rerun with residual JIT,
skipped initial residual check, and batched JAX GMRES gave `3.07 s`
unpreconditioned, `3.79 s` with `parallel_line`, and `3.76 s` with
`parallel_line` plus preconditioner reuse; all runs retained the same residual
band and the full `800` GMRES update budget. That closes simple row, field,
diagonal, field-active diagonal, same-cell block-Jacobi, and first line-block
preconditioning as default speedup lanes and points the next performance work
toward fewer
accepted trial solves, lower residual/JVP kernel cost, or a stronger
Schur/transport preconditioner with measurable iteration-count reduction before
spending more D/T/He wall time.

Promotion scripts can now require this diagnostic evidence explicitly. For
fixed-BDF2 campaigns, pass
`--fixed-bdf2-linear-preconditioner=<name>` to
`scripts/run_recycling_jvp_promotion_gate.py`; the wrapper forwards
`runtime:recycling_jax_linear_preconditioner=<name>` and also asks
`scripts/compare_recycling_transient_modes.py` to require
`fixed_bdf2_linear_preconditioner=<name>` plus a positive preconditioner-build
count. The lower-level compare script also exposes
`--require-fixed-bdf2-linear-preconditioner=<name>` and
`--require-adaptive-bdf-linear-preconditioner=<name>` for manual campaigns.
These gates are deliberately stricter than a runtime-only override: they fail
if a preconditioner request silently falls back to an unpreconditioned or
Lineax path. Use
`--fixed-bdf2-linear-preconditioner-refresh=<n>` on the wrapper to forward
`runtime:recycling_jax_linear_preconditioner_refresh=<n>` and test reuse inside
each implicit solve. A bounded local run on `recycling_1d_one_step` with
`--timestep=10`, `--steps=2`, and `local_block_diag` passed these new gates on
both fixed-BDF2 routes. The fixed-full-field route reported residual
`1.90e-6`, `9` preconditioner builds, zero failed linear solves, and
`22.7 s`; the active-array route reported the same residual and build count,
zero failed linear solves, and `30.3 s`. Both routes still used the full
`3600` JAX-GMRES update budget. With refresh set to `100`, the same gate
reduced builds from `9` to `2`, ran in `18.9 s` for the fixed-full-field route
and `23.4 s` for the active-array route, preserved zero failed linear solves,
and kept the residual below the `1e-5` gate. This is useful opt-in reuse
evidence, but it is not a default-promotion result because the Krylov budget
remains saturated.
For that reason, performance campaigns should add explicit budget gates rather
than relying on correctness diagnostics alone. The compare script provides
`--require-fixed-bdf2-max-linear-iterations=<n>` and
`--require-fixed-bdf2-max-preconditioner-builds=<n>`, and the wrapper forwards
the same checks as `--fixed-bdf2-max-linear-iterations=<n>` and
`--fixed-bdf2-max-preconditioner-builds=<n>`. A run that passes residual,
fallback, and preconditioner-name checks but exceeds these budgets is a valid
solver-correctness result and a failed performance-promotion result.
The bounded refresh-100 `recycling_1d_one_step` rerun passes these stricter
budget gates with `--require-fixed-bdf2-max-linear-iterations=3200` and
`--require-fixed-bdf2-max-preconditioner-builds=2`: the fixed-full-field route
ran in `18.9 s`, the active-array route ran in `22.5 s`, both reported
residual `3.76e-6`, and both kept failed linear solves at zero. This is the
current bounded fixture evidence for preconditioner reuse; heavier recycling
and D/T/He gates still need the same budgeted treatment before default
promotion.

The real-kernel JAX-linearized profiling script has the same control surface
for heavier runs. `scripts/profile_recycling_jax_linearized_gate.py` accepts
`--linear-restart=<n>`, `--linear-maxiter=<n>`, and
`--linear-tolerance-factor=<factor>` for reproducible Krylov-budget sweeps
without hand-written BOUT overrides. It also accepts
`--linear-preconditioner=<name>` and
`--linear-preconditioner-refresh=<n>` to forward the corresponding runtime
options into the profiled solve. It also accepts
`--require-linear-preconditioner=<name>`,
`--require-max-residual-inf-norm=<x>`,
`--require-min-nonlinear-iterations=<n>`,
`--require-min-linear-iterations=<n>`,
`--require-max-linear-iterations=<n>`, and
`--require-max-preconditioner-builds=<n>`. These gates are evaluated after the
profile is written, recorded in `profile_summary.json`, and return a nonzero
status if the reported diagnostics miss the requested preconditioner, exceed
the requested residual ceiling, or exceed the requested iteration floors or
budgets. The minimum-iteration gates are important for heavy decks because a
tiny timestep can converge at the predictor and otherwise produce a misleading
"passed" profile without exercising Newton, JVP, or Krylov work. Dynamic
JVP-derived preconditioners such as
`linearized_diag`, `local_block_diag`, and `parallel_line` must report at least
one finite preconditioner build when they are required. This lets local CPU and
office-GPU D/T/He campaigns distinguish a solver-correctness pass from a
performance-promotion pass without hiding failed budget evidence. A local
D/T/He run at `timestep=1.0` shows why these gates are necessary: the
unpreconditioned JAX-linearized gate completed in `7.59 s`, while
`local_block_diag` with refresh `100` completed in `29.84 s`. Both reached the
same residual (`7.315`) and the full `400` JAX-GMRES update budget, and the
local-block run spent `1.65 s` building one preconditioner. This is a
correctness pass for the heavy preconditioner diagnostics and a negative
performance-promotion result, so the next preconditioning work must reduce the
Krylov budget or residual/JVP cost instead of merely adding local dense blocks.
A follow-up sweep reached the same conclusion for cheaper controls. On the
same D/T/He gate, `field_scale` took `8.01 s`, `linearized_diag` took `8.31 s`,
and `state_scale` took `27.42 s`; all retained the full `400` update budget.
Reducing the unpreconditioned Krylov budget to `10 x 10` updates gave a similar
residual (`7.316` instead of `7.315`) but did not speed up the local CPU run
(`8.49 s`), while `5 x 10` updates both slowed down (`27.84 s`) and worsened
the residual (`8.09`). Incremental GMRES at `10 x 10` was also slower
(`26.46 s`). The retained local campaign therefore keeps the measured-fast
`20 x 20` batched JAX-GMRES control and adds
`--require-max-residual-inf-norm=7.4` so future budget sweeps cannot pass by
silently degrading the nonlinear residual.
Residual JIT remains opt-in on this D/T/He CPU gate. A single non-warmed
`--jit-residual` run passed the same residual gate but took `30.90 s`, with
`7.37 s` in residual evaluations and `23.37 s` in the linear solve. A warmed
run with one warmup and two timed solves reduced the timed median to `10.22 s`
but still did not beat the non-JIT baseline; the warmup solve itself took
`20.08 s`. This keeps residual JIT useful as a diagnostic seam and possible
accelerator/GPU probe, but not as the local CPU default.
The same profile exposed one cheap residual-evaluation reduction. The generic
JAX-linearized Newton solver now reports `line_search_trial_count`,
`line_search_last_step_scale`, and `line_search_initial_step_scale`, and the
recycling runtime can set
`runtime:recycling_jax_linear_line_search_initial_step_scale=<s>` (or the
matching `JAX_DRB_RECYCLING_JAX_LINEAR_LINE_SEARCH_INITIAL_STEP_SCALE`
environment variable). On the D/T/He `timestep=1.0` gate, the default line
search tried scales `1`, `0.5`, then `0.25`, accepted `0.25`, and spent four
total residual evaluations. Starting directly at `0.25` preserved the same
residual (`7.315`), the same `400` JAX-GMRES update budget, and clean solver
status, while reducing total residual evaluations from `4` to `2`,
line-search trials from `3` to `1`, and local wall time from `7.84 s` to
`7.34 s`. The local D/T/He research gate now requires this reduced residual
evaluation count with `--require-max-residual-evaluations=2` and
`--require-max-line-search-trials=1`.

The same gate now records and gates matrix-free linear-operator calls. The
generic JAX-linearized solver reports `linear_operator_call_count` and
`linear_operator_dispatch_seconds` for Python-visible calls to the linearized
operator during the Krylov solve. These diagnostics are separated from dynamic
preconditioner builds, residual evaluations, and line-search residuals. A
local wrapper run with the D/T/He gate passed with `5` operator calls,
`1.16 s` of operator-dispatch time, `5.24 s` total linear-solve time,
`1.30 s` JAX-linearization time, residual `7.315`, two residual evaluations,
and one line-search trial. The gate now requires
`--require-min-linear-operator-calls=1`, which prevents profiles from passing
without exercising the matrix-free JVP/Krylov path. The dispatch time is not a
substitute for `linear_solve_seconds`, because JAX device work can be
asynchronous, but it gives the preconditioner lane a stable call-count metric
for future reductions.

A fresh cProfile/RSS run with these diagnostics enabled keeps the same
conclusion. The cProfile-instrumented D/T/He gate took `12.67 s`, while the
separate RSS sample took `8.97 s`. Solver diagnostics reported `5` operator
calls, `2.47 s` operator-dispatch time, `9.54 s` total linear-solve time,
`2.64 s` JAX-linearization time, two residual evaluations, and one line-search
trial. The cumulative cProfile rows were dominated by JAX
`custom_linear_solve`/`gmres`, JAX tracing/cache-miss paths, and the
fixed-layout residual linearization; the residual body itself appeared through
`recycling_fixed_residual.residual` and `_compute_recycling_1d_rhs_from_species`.
That profile confirms that the next speedup lane is not more line-search
tuning. It is either a cheaper fixed-layout residual/JVP kernel, a solver path
that amortizes JAX tracing/compilation more effectively, or a preconditioner
that genuinely reduces matrix-free linear-operator work.

The first small residual-kernel cleanup removes a redundant pack of the
unchanged state vector inside the fixed-layout backward-Euler and BDF2
residual builders. The residual still unpacks the state to evaluate the RHS,
but the left-hand state term now reuses the incoming packed vector directly
instead of reconstructing it from field blocks. Focused BE/BDF2 residual tests
and the D/T/He gate preserve the same residual (`7.315`) and the same `5`
matrix-free operator calls. A warmed D/T/He check reported warmup `6.81 s`,
timed solves `6.59 s` and `6.55 s`, median `6.57 s`, `5.31 s` linear-solve
time, and `1.13 s` JAX-linearization time. The result should be treated as a
low-risk residual simplification and a better baseline for future JVP-kernel
work, not as sufficient evidence for a release-level speedup claim.

Repeating the preconditioner sweep after this damping does not yet justify a
new default. With the damped line search, `field_scale` remained neutral
(`8.65 s` versus `8.65 s` for the same-run unpreconditioned control), while
`linearized_diag` reduced linear-solve time (`6.60 s` versus `6.79 s`) but lost
more time in its one JVP-derived diagonal build (`0.73 s`). A first
`local_block_diag` run looked promising (`7.90 s` versus `8.65 s`, with
linear-solve time reduced from `6.79 s` to `5.64 s` after a `0.69 s` build),
but the repeat pair reversed the outcome (`8.40 s` for `local_block_diag`
versus `7.65 s` for the unpreconditioned control). The residual, line-search
budget, and `400` JAX-GMRES update budget were unchanged. This keeps
`local_block_diag` as useful instrumentation and a correctness gate, but not as
a robust performance promotion. The next preconditioner implementation should
approximate transport/neutral Schur structure or reduce residual/JVP kernel
cost directly, rather than adding another local dense block variant.

The explicit full parallel-line block is also not a local CPU promotion path
for this deck. Raising the line-block bounds to allow the single active
parallel line with 950 field unknowns produced a clean solve with the same
residual, but wall time increased to `27.46 s`, and linear-solve time increased
to `25.01 s` versus `5.99 s` for the repeat unpreconditioned control. This
shows that a mathematically larger block is not automatically a useful left
preconditioner for the current JAX-GMRES implementation; future transport
preconditioners should be cheaper approximate line/Schur solves rather than
dense inversion of the full active line.

The recycling wrappers also expose
`runtime:recycling_jax_linear_initial_residual_mode=linearize` or
`JAX_DRB_RECYCLING_JAX_LINEAR_INITIAL_RESIDUAL_MODE=linearize` for profiling
decks that are not expected to start converged. The generic solver keeps the
standalone initial residual check enabled by default because it avoids
unnecessary linearization when a predictor already satisfies the nonlinear
tolerance. The `linearize` mode keeps that convergence check but obtains the
first residual norm from the first JAX linearization, avoiding the duplicate
standalone residual call on non-converged heavy solves. The profiling script
exposes this as `--initial-residual-mode linearize`; the older
`--skip-initial-residual-check` switch remains available only when a campaign
intentionally wants to remove the initial check altogether. On earlier bounded
hydrogen `timestep=1.0` JAX-linearized gates, removing the standalone initial
residual call reduced the deterministic residual-evaluation count from `6` to
`5` with identical residual norm, so this remains a host/device-barrier
reduction seam rather than a default speedup claim. On the bounded D/T/He
JAX-linearized recycling gate, the safer `linearize` mode passed with
`check_initial_residual=true`, residual evaluations `2`, one line-search trial,
residual norm `7.315`, clean JAX-GMRES status `0`, median timed run `6.97 s`,
and sampled peak RSS delta `525 MiB`. That evidence supports the profiling
surface but does not promote the full output-window recycling solve to the
default path. The profile script can now also require this mode with
`--require-initial-residual-mode linearize`, and the local D/T/He research
bundle uses that gate so future profiles cannot silently fall back to the
standalone residual path.

The generic JAX-linearized solver now also reuses the already-known residual
norm for the final accepted state when a bounded solve exits because the
nonlinear iteration budget was exhausted. This removes a redundant final
residual evaluation and host/device synchronization without changing the
reported residual. On the same bounded hydrogen gate with residual JIT, skipped
initial residual check, and the default batched JAX GMRES solve method, the
residual-evaluation count dropped from `5` to `4` with identical residual norm
(`8.815424126680732e-05`) and clean GMRES status. The three-run median was
`3.08 s` versus the retained `2.98 s` comparison artifact, so this patch should
be treated as deterministic residual-count cleanup rather than a measured
runtime win.

The JAX GMRES wrapper now also forwards the optional JAX
`solve_method={"batched","incremental"}` control through
`runtime:recycling_jax_linear_gmres_solve_method=<method>`,
`JAX_DRB_RECYCLING_JAX_LINEAR_GMRES_SOLVE_METHOD`, or the profiling-script flag
`--gmres-solve-method`. JAX documents the incremental method as building a
Givens-rotation QR factorization with an intra-restart residual estimate, while
the batched method has lower accelerator overhead. On the warmed bounded
hydrogen gate with residual JIT and the initial residual check disabled, the
incremental method was correct but not a speedup: the three-run median was
`3.06 s` versus `2.98 s` for the default batched method, with the same solver
status and residual band. This makes the method switch useful for CPU/GPU
sweeps, but it is not a default promotion.

The sparse-JVP Jacobian builder also has an opt-in host-transfer reduction
probe through `JAX_DRB_SPARSE_JVP_GATHER_ON_DEVICE=1`. When enabled, each
color batch gathers only structurally nonzero pushed rows on device before
copying data to the host sparse assembler. The promoted small hydrogen gate
was not large enough to benefit from the first version of this path: the
device-gather run took `1.596 s`, while the default full-transfer run took
`1.551 s`. The path remains useful for larger CPU/GPU profiling, but it is off
by default because the measured local gate does not justify changing the
production behavior. A follow-up micro-kernel cleanup prebuilds the device-side
row and batch-index gather arrays in `SparseJvpDirectionBatch` so a reused
workspace does not recreate those static JAX arrays on every Jacobian build. On
a local 288-state, 4896-nnz sparse-JVP device-gather microbenchmark this
reduced the median build time from `0.0128 s` to `0.0114 s`. This is only a
small opt-in sparse-JVP kernel cleanup; larger recycling gates still need to
show a runtime win before changing defaults.

The sparse Newton compatibility path now also reports explicit SciPy linear
solver health in the same diagnostics payload used by adaptive-BDF recycling
gates. Direct sparse solves report `scipy_spsolve`; GMRES solves report
`scipy_gmres`; failed GMRES attempts that fall back to the direct sparse solve
report `scipy_gmres_spsolve_fallback`. Immediate-convergence records with zero
linear iterations are no longer counted as unknown linear-solver status. On the
bounded `recycling_1d_one_step` sparse-JVP adaptive-BDF gate with
`timestep=0.25`, `steps=1`, and `max_nonlinear_iterations=3`, this changes the
health summary from ambiguous sparse solves to `24` sparse-JVP solver steps,
`0` failed linear solves, and `0` unknown linear-solver steps. The elapsed time
remains about `16.45 s`, with the dominant cost still in JVP Jacobian assembly,
so this is a promotion-gate instrumentation fix rather than a runtime speedup.

The same pass also changed the live runtime picture in a way that matters for
the paper and for users:

- `neutral_mixed_one_step` dropped from roughly `6.44 s` live runtime to about
  `1.43 s`, and the later term-balance audit localizes the remaining
  state-history drift while preserving direct operator-source parity
- `recycling_1d_one_step` now runs at about `15.98 s` live runtime with
  fidelity preserved, which is materially better than the earlier recycling
  baseline but still slower than Hermès on the same machine

That means the next work should be divided clearly:

- keep the neutral mixed lane on state-history/media refresh rather than
  formula replacement
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
