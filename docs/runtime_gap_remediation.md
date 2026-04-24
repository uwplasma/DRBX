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
`max_{x,z} |Δ|(y)` profile on the same live rerun surface.

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

### Priority 3: tokamak recycling observables

1. keep the current one-step compare metrics
2. add observables closer to the benchmark literature:
   - target profiles
   - source/ionization lineouts
   - target flux summaries
3. move the paper-facing story from raw `NVd` relative mismatch to physically
   interpretable observables

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
- one dedicated neutral-mixed mismatch campaign
- one refreshed heavy recycling runtime profile bundle
- one committed GPU profiling bundle from `office`
- one tokamak-observable comparison package closer to the TCV-X21/Hermès style
