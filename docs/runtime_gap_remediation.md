# Runtime And Fidelity Gap Remediation

This document is the execution plan for the current worst runtime and
native-versus-Hermès mismatch cases.

It is based on the refreshed same-machine live rerun matrix in:

- [hermes_live_rerun_campaign.md](hermes_live_rerun_campaign.md)

and on the current profiling workflow in:

- [profiling_runtime.md](profiling_runtime.md)

## Current Highest-Priority Cases

The refreshed live matrix identifies four distinct categories.

### 1. Real open-field neutral mismatch

- case: `neutral_mixed_one_step`
- current worst normalized RMS error: about `9.17e-1`
- current runtime ratio: about `2.93x`
- dominant field: `NVh`

This is still the clearest current fidelity gap. The runtime was reduced
substantially by vectorizing the perpendicular diffusion kernel and the local
profile mean dropped from about `1.15 s` to about `0.63 s`, but the main
physics mismatch remains. The focused follow-up figure is now in
[neutral_mixed_boundary_campaign.md](neutral_mixed_boundary_campaign.md), which
shows the worst-error `Nh`, `Ph`, and `NVh` lineouts plus the
`max_{x,z} |Δ|(y)` profile on the same live rerun surface. The next diagnostic
layer is now in
[neutral_mixed_term_balance_campaign.md](neutral_mixed_term_balance_campaign.md):
it inserts both the native and Hermès-3 final states into the native
neutral-mixed momentum operator and decomposes the `NVh` residual-rate into
parallel inertia, pressure gradient, perpendicular diffusion, parallel
viscosity, and perpendicular viscosity.

### 2. Heavy 1D recycling runtime bottleneck

- case: `recycling_1d_one_step`
- current worst normalized RMS error: about `4.62e-3`
- current runtime ratio: about `3.65x`
- dominant normalized field: `Pd+`

This lane is already tight in fidelity and still one of the main runtime
offenders.

### 3. Heavy multispecies 1D recycling bottleneck

- case: `recycling_dthe_one_step`
- current worst normalized RMS error: about `4.92e-3`
- current runtime ratio: about `7.82x`
- dominant field: `NVd`

This is the current worst runtime ratio in the live matrix and the main
production-path runtime target. The latest target-boundary geometry caching
pass reduced the local timed run from about `54.1 s` to about `52.76 s`
without changing the fidelity band.

### 4. Near-zero normalized tokamak recycling mismatch

- cases:
  - `integrated_2d_recycling_one_step`
  - `tokamak_recycling_one_step`
- current normalized RMS errors: about `1.79e-1` and `1.62e-1`
- current runtime ratios: about `0.85x` and `0.39x`
- dominant field in both cases: `NVd`
- current worst absolute max-errors:
  - integrated: about `7.48e-12`
  - tokamak: about `3.09e-7`

These are important, but they are not the same class of issue as the neutral
mixed mismatch. The current public figure now marks them as
normalization-sensitive because the relative metric is dominated by a near-zero
reference field.

## Current Measured Runtime Improvements

The current optimization pass already changed the runtime picture materially.

### Neutral mixed

- earlier local timed mean: about `1.15 s`
- current local timed mean: about `0.63 s`
- current live runtime: about `1.32 s`
- remaining issue: fidelity on `NVh`, not only runtime

### 1D recycling

- current live runtime: about `20.11 s`
- current runtime ratio to live Hermès: about `3.65x`
- remaining issue: sparse FD Jacobian plus repeated heavy RHS evaluation

## Current Root-Cause Hypotheses

The current bottlenecks split into two classes.

### Runtime bottlenecks

- sparse finite-difference Jacobian assembly in the implicit backbone
- repeated host-side RHS/source assembly in `recycling_1d.py`
- remaining host/SciPy structure in the heavy transient path
- repeated per-run compilation or warmup cost on GPU if cache is not enabled

### Fidelity bottlenecks

- neutral mixed closure/operator mismatch concentrated on `NVh`
- near-zero-field normalization on `NVd` in the integrated/direct tokamak
  one-step compare surfaces
- possible closure or initial-state differences between native and Hermès on
  the heavy open-field neutral/recycling ladders

## Near-Term Remediation Order

### Priority 1: neutral mixed fidelity

1. profile `neutral_mixed_one_step` with the public profiling script
2. isolate the `NVh` residual terms and compare them term-by-term against the
   Hermès lane
3. keep the new live rerun boundary-audit package and extend it term-by-term
   for the offending neutral terms
4. lock the fix with a direct regression test plus the same paper-grade figure

The term-balance package now completes item 2 on the native side. It shows that
the native final state nearly satisfies the native backward-Euler residual, but
the Hermès-3 final state does not. The next neutral fidelity fix should
therefore inspect Hermès-3's corresponding pressure-gradient and viscosity
operator outputs, not another aggregate field-error plot.

### Priority 2: multispecies recycling runtime

1. profile `recycling_dthe_one_step` with:
   - cProfile
   - JAX trace
   - XLA dump
2. use the sparse Newton phase diagnostics to split the runtime into residual
   evaluation, finite-difference Jacobian refresh, linear solve, line search,
   and fallback time
3. identify the current Jacobian/RHS hot splits after the recent Horner,
   vectorization, cached-geometry, and precomputed color-plan improvements
4. cut repeated source/closure recomputation where possible
5. keep the current fidelity band unchanged while reducing runtime

The latest local cProfile/RSS pass sharpens this target. After AMJUEL log-input
reuse and BDF Jacobian-plan reuse, a one-run cProfile measurement took about
`67.00 s`, while the separate RSS run took about `53.37 s` and peaked at about
`229.4 MiB`. The cProfile split is still dominated by SciPy BDF stepping,
packed RHS calls, and sparse finite-difference Jacobian construction, with
reaction-source/AMJUEL evaluation, neutral parallel advection, collision
closure, state preparation, and target recycling as the next source-level
offenders. This argues for fewer host residual calls, stronger source/closure
caching, and a JAX-native residual/JVP lane before spending more effort on
generic local-thread tuning.

The latest source-kernel cleanup confirms that ordering. With the packed D/T/He
residual routed through the fixed-layout reaction-source kernel and D/T AMJUEL
fits reused inside both source and neutral-ionisation rate assembly, a refreshed
one-run cProfile measured `64.45 s` and the separate RSS run measured `50.00 s`
with peak RSS near `232.7 MiB`. Fixed-layout reaction sources fell to about
`9.64 s`, neutral-ionisation collision-rate assembly to about `2.72 s`, and
AMJUEL fit evaluations to `117380` calls. Sparse finite-difference Jacobian
assembly still consumed about `43.3 s`, so the remaining high-value work is
residual-call reduction and JAX-linearized Jacobian products rather than more
localized reaction-source tuning.

The next call-count cleanup is now in-tree: the SciPy BDF callback caches the
most recent exact RHS evaluation and reuses it as the base state for
`jac(t, y)` when SciPy requests the Jacobian at that same state. The history
result exposes BDF RHS evaluation, cache-hit, Jacobian-callback, and
Jacobian-worker counters. The BDF callback now also sends finite-difference
perturbation residuals around the mutable RHS cache and honors
`JAX_DRB_FD_JACOBIAN_THREADS=<N>` for explicit local CPU threading.
This closes an avoidable duplicate-base-RHS path, but it should be treated as
instrumentation and cleanup rather than a solved performance milestone; a
post-change one-run `recycling_dthe_one_step` timing measured `61.38 s`, which
does not establish an end-to-end speedup against the previous noisy local run.
The follow-up timing-only check reinforces that point: serial, two-thread, and
four-thread BDF runs measured about `50.00 s`, `49.81 s`, and `54.57 s`,
respectively, so one-solve threading is not the reviewer-grade scaling result
for this host-backed residual.
The stronger next step remains a JAX-transformable residual plus grouped
JVP-based Jacobian products for the dominant recycling kernels.

The solver package now has the first production-tested piece of that JAX path:
the grouped sparse-JVP Jacobian builder batches colored tangent pushes with
`jax.vmap` after a single `jax.linearize` call. This is the correct derivative
algorithm for a JAX residual because it removes finite-difference step-size
choice and perturbation residual calls. It does not by itself fix the heavy
recycling lane because the current residual is still dominated by NumPy/SciPy
assembly. The next runtime fixes should therefore be ordered as residual-kernel
ports first, solver-backend promotion second.

The sparse Newton interface now exposes that derivative algorithm directly as
`jacobian_mode="jvp"`, and the implicit-solver profile audit compares it
against the finite-difference sparse Newton path on a transformable residual.
The audit is deliberately small: it proves the solver backend boundary, not the
full recycling migration. The full migration still depends on moving the
remaining recycling residual kernels out of dictionary/NumPy assembly.

Near-term residual-kernel ports should target the measured hot path in this
order:

1. active packed-state layout as a static PyTree rather than repeated dict and
   full-field reconstruction
2. reaction/source and AMJUEL/OpenADAS table evaluation in JAX arrays
3. collision closure and neutral parallel diffusion in JAX arrays
4. target recycling and target-boundary source assembly in JAX arrays
5. backward-Euler/BDF residual assembly as a pure function with no `np.asarray`
   barrier inside the nonlinear solve

Each port should carry three gates before promotion: current NumPy parity on a
small deterministic state, JVP-versus-finite-difference derivative parity, and
the existing Hermès one-RHS/one-step compare surface.

Items 1, 3, 4, and 5 now have their first compact gates:

- `RecyclingFixedState` provides the active fixed-layout PyTree state and
  transformable backward-Euler/BDF2 residual builders;
- active-region and recycling active-state pack/unpack preserve JAX tracers;
- target recycling and the target-source kernel preserve JAX when dynamic state
  is a JAX array even if metrics are static NumPy arrays;
- neutral parallel diffusion and key collision/conduction closure helpers have
  compact JVP tests with precomputed rate surfaces;
- BE/BDF2 residual algebra no longer forces NumPy on JAX inputs.

Those gates are intentionally not promoted to the full heavy solve yet. The
first full-deck bridge is now tested on the local Hermès
`1D-recycling-dthe` input: `RecyclingFixedState` round-trips the active
DTHE recycling state, reconstructs full guard-cell fields, and drives the
current packed RHS oracle through an explicit `build_fixed_host_rhs_bridge`.
The bridge matches the current packed RHS and backward-Euler residual value.
It is deliberately named as a host bridge because the wrapped RHS still crosses
the NumPy/SciPy boundary. The next production step is narrower and safer:
replace the bridge internals one term at a time with fixed-layout JAX kernels,
then compare JVP actions against finite differences before replacing the
finite-difference Jacobian in the heavy BDF path.

The atomic-rate part of item 2 is now complete at helper level: AMJUEL paired
rate/radiation evaluation, OpenADAS bilinear interpolation, and the hydrogen
charge-exchange fit are backend-preserving and have direct `jit`/`grad`
coverage. The remaining item-2 work is to lift the surrounding reaction-source
accumulation out of mutable dictionaries and into a fixed array/PyTree layout
so those differentiable helpers can be used inside the full recycling residual.
The current dictionary-oriented reaction-source layer now also preserves JAX
arrays through the single-isotope ionisation, recombination, and
charge-exchange formulas, so the fixed-layout accumulator can reuse validated
formula code rather than reimplementing the physics.
The first such accumulator now exists for the hydrogenic same-isotope reaction
block and is parity-tested against the dictionary path. The next reaction
source step is to generalize that pattern to multispecies D/T/He and then
OpenADAS-enabled impurity states.
The D/T/He generalization is now in-tree as `fixed_layout_dthe_reaction_sources`:
it returns stacked neutral/ion/electron source arrays, includes D-D, T-T, D-T,
and T-D charge exchange, matches the existing dictionary path on the
`1D-recycling-dthe` deck, and supports `jit`/`grad`. The remaining reaction
source runtime step is partly complete: the packed recycling RHS now requests
reaction sources without diagnostics, and the exact D/T/He Hermès reaction
block is dispatched through the fixed-layout array kernel on that hot path. The
full dictionary implementation remains the reporting path when diagnostics are
requested. The next source-runtime step is the equivalent OpenADAS impurity
source block, followed by moving collision, diffusion, target-recycling, and
BDF residual assembly into a pure-JAX residual.

### Priority 3: tokamak recycling observables

1. keep the current one-step compare metrics
2. add observables closer to the benchmark literature:
   - target profiles
   - source/ionization lineouts
   - target flux summaries
3. move the paper-facing story from raw `NVd` relative mismatch to physically
   interpretable observables

The first version of this observable layer is now in-tree as
[tokamak_recycling_observable_campaign.md](tokamak_recycling_observable_campaign.md).
It uses the direct-tokamak D/T/He recycling one-step lane to report charged
target-density profiles, `|NV_s+|` target momentum-flux proxies, neutral
parallel-density buildup, and target electron-temperature proxy errors. The
next expansion should add source-diagnostic fields from live or cached Hermes
dumps so the same package can show explicit ionisation/recombination source
lineouts rather than neutral-density proxies only.

## Literature Standard

The expected validation and runtime storytelling standard is set by:

- [Roy 2005](https://www.sciencedirect.com/science/article/pii/S0021999104004747)
- [Bufetov et al. 2022, TCV-X21 benchmark](https://arxiv.org/abs/2109.01618)
- [Wang et al. 2023, SOLPS-ITER versus TCV-X21](https://arxiv.org/abs/2310.17390)
- [Dudson et al. 2024, Hermès-3 code paper](https://www.sciencedirect.com/science/article/pii/S0010465523003363)
- [Dudson et al. 2025, Hermès-3 versus TCV-X21](https://arxiv.org/abs/2506.12180)
- [official JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html)
- [official JAX persistent compilation cache documentation](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)

That means the next fixes should not stop at "bar chart looks better". They
need:

- operator-level diagnosis
- physically interpretable observables
- a reproducible profiler workflow
- explicit runtime/fidelity tradeoff discussion

## GPU And Memory Plan

The reachable `office` machine is the current GPU testbed. It already has two
visible CUDA devices under JAX, so the GPU plan is now operational rather than
speculative.

The clean repo-local GPU environment is now in place with
`jax[cuda12]==0.6.2`. The first useful GPU evidence is therefore already
available on the compact native JAX lanes:

- traced-field-line reduced lane:
  compile `4.41e-2 s`, first execute `1.23e-3 s`, warm execute `3.30e-4 s`
- stellarator VMEC reduced lane:
  compile `7.36e-3 s`, first execute `3.98e-4 s`, warm execute `1.14e-4 s`

That means the next GPU work should be:

1. enable persistent compilation cache
2. keep GPU runtime work on the compact native JAX lanes first
3. compare CPU and GPU runtime splits on:
   - traced-field-line reduced kernels
   - stellarator VMEC reduced kernels
   - other selected-field compact lanes before promoted transient ladders
4. keep the heavy recycling and neutral-mixed lanes on the CPU remediation
   path until the host/SciPy structure is reduced enough for accelerator work
   to be meaningful
5. decide whether the next parallelization gain should come from:
   - single-case acceleration
   - batched independent solves
   - multi-device sharding on accelerator hardware

## Required New Evidence

Before the next paper-facing performance pass, the codebase should have:

- a refreshed same-machine live Hermès rerun figure
- one dedicated neutral-mixed boundary campaign and one term-level `NVh`
  balance campaign
- one refreshed heavy recycling runtime profile bundle
- one committed GPU profiling bundle from `office`
- one tokamak-observable comparison package closer to the TCV-X21/Hermès style
