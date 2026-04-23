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
- current runtime ratio: about `4.27x`
- dominant field: `NVh`

This is still the clearest current fidelity gap. The runtime was reduced
substantially by vectorizing the perpendicular diffusion kernel, but the main
physics mismatch remains.

### 2. Heavy 1D recycling runtime bottleneck

- case: `recycling_1d_one_step`
- current worst normalized RMS error: about `4.62e-3`
- current runtime ratio: about `3.80x`
- dominant normalized field: `Pd+`

This lane is already tight in fidelity and still one of the main runtime
offenders.

### 3. Heavy multispecies 1D recycling bottleneck

- case: `recycling_dthe_one_step`
- current worst normalized RMS error: about `4.92e-3`
- current runtime ratio: about `8.45x`
- dominant field: `NVd`

This is the current worst runtime ratio in the live matrix and the main
production-path runtime target.

### 4. Near-zero normalized tokamak recycling mismatch

- cases:
  - `integrated_2d_recycling_one_step`
  - `tokamak_recycling_one_step`
- current normalized RMS errors: about `1.79e-1` and `1.62e-1`
- current runtime ratios: about `0.83x` and `0.35x`
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

- earlier live runtime: about `6.44 s`
- current live runtime: about `1.43 s`
- remaining issue: fidelity on `NVh`, not only runtime

### 1D recycling

- current live runtime: about `15.98 s`
- current runtime ratio to live Hermès: about `3.80x`
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
3. add a dedicated operator/closure campaign for the offending neutral terms
4. lock the fix with a direct regression test plus a paper-grade figure

### Priority 2: multispecies recycling runtime

1. profile `recycling_dthe_one_step` with:
   - cProfile
   - JAX trace
   - XLA dump
2. identify the current Jacobian/RHS hot splits after the recent Horner and
   vectorization improvements
3. cut repeated source/closure recomputation where possible
4. keep the current fidelity band unchanged while reducing runtime

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

The immediate blocker is a clean runtime environment. Once that is in place,
the next GPU work should be:

1. enable persistent compilation cache
2. collect JAX trace and device-memory profiles on the same worst-offender
   cases
3. compare CPU and GPU runtime splits on:
   - `neutral_mixed_one_step`
   - `recycling_1d_one_step`
   - `recycling_dthe_one_step`
   - `tokamak_recycling_one_step`
4. decide whether the next parallelization gain should come from:
   - single-case acceleration
   - batched independent solves
   - multi-device sharding on accelerator hardware

## Required New Evidence

Before the next paper-facing performance pass, the codebase should have:

- a refreshed same-machine live Hermès rerun figure
- one dedicated neutral-mixed mismatch campaign
- one refreshed heavy recycling runtime profile bundle
- one GPU profiling bundle from `office`
- one tokamak-observable comparison package closer to the TCV-X21/Hermès style
