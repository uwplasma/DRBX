# Profiling And Runtime Workflow

This page records the supported profiling workflow for `jax_drb` so runtime
work is reproducible instead of based on one-off local shell snippets.

The current public entry point is:

- [scripts/profile_curated_case.py](../scripts/profile_curated_case.py)

It is meant for the real worst offenders and live validation lanes, not only
for compact microbenchmarks.

The script now requires one of:

- `--reference-root /path/to/hermes-3`
- `JAX_DRB_REFERENCE_ROOT=/path/to/hermes-3`

## Recommended CPU Cases

The current highest-value local CPU cases are:

- `neutral_mixed_one_step`
- `recycling_1d_one_step`
- `recycling_dthe_one_step`
- `integrated_2d_recycling_one_step`
- `tokamak_recycling_one_step`

These are the cases where the current same-machine Hermès rerun matrix still
shows either the largest runtime ratios, the largest fidelity gaps, or both.

## Current Measured CPU Results

The latest local profiling pass gives the following reviewer-usable numbers on
this machine:

- `neutral_mixed_one_step`
  - timed local mean dropped from about `1.15 s` to about `0.63 s` after
    vectorizing `_gradient_magnitude`
  - fresh live Hermès rerun ratio is about `2.93x`
  - the dominant mismatch remains the boundary-localized `NVh` field
- `recycling_dthe_one_step`
  - timed local mean dropped from about `75.3 s` to about `54.1 s` after the
    reaction/source allocation cleanup, and then to about `52.76 s` after
    caching target-boundary geometry in the recycling runtime model
  - the latest one-run cProfile pass on this machine measured `67.00 s`; the
    separate RSS run measured `53.37 s` with peak process-tree RSS about
    `229.4 MiB`
  - fresh live Hermès rerun ratio is about `7.82x`
  - the fidelity band stayed essentially unchanged at about `4.9e-3` relative
    RMS on `NVd`
  - the isolated target-recycling operator now also shows a direct kernel-level
    improvement of about `1.17x` from the same cached-geometry path on the CPU
    NumPy RHS

The next heavy CPU optimization target is no longer generic reaction
allocation. The refreshed cProfile still shows the dominant remaining work in:

- the SciPy BDF history path itself
- finite-difference Jacobian assembly
- neutral parallel diffusion
- collision closure
- target recycling / target boundary-source assembly
- prepared-state and boundary setup on the open-field lane

On the latest `recycling_dthe_one_step` cProfile pass, after AMJUEL log-input
reuse and BDF Jacobian-plan reuse, the cumulative top costs were: SciPy
`solve_ivp`/BDF at about `66.4 s`, packed RHS calls at about `64.0 s`, species
RHS assembly at about `61.2 s`, sparse finite-difference Jacobian construction
at about `45.7 s`, reaction sources at about `13.7 s`, AMJUEL fit evaluation
at about `11.4 s`, neutral parallel advection at about `7.6 s`,
collision closure at about `6.3 s`, state preparation at about `6.1 s`, and
target recycling at about `6.1 s`. That split confirms that the next runtime
fix has to attack both the Jacobian/RHS call count and repeated source/closure
work; local threading alone is not the right primary fix for this path.

A follow-up `recycling_dthe_one_step` pass after wiring the diagnostics-free
packed RHS through `fixed_layout_dthe_reaction_sources` and reusing D/T AMJUEL
fits measured `64.45 s` under cProfile and `50.00 s` on the separate RSS run,
with a sampled peak process-tree RSS of about `232.7 MiB`. The source-level
split moved in the intended direction: fixed-layout D/T/He reaction sources
dropped to about `9.64 s`, neutral-ionisation collision-rate assembly dropped
to about `2.72 s`, and AMJUEL polynomial evaluations dropped to `117380` calls
with about `7.81 s` cumulative time. The full solve did not show a defensible
end-to-end speedup because the sparse finite-difference Jacobian still consumed
about `43.3 s` and the packed RHS was still called `11738` times. Treat this
as a validated source-kernel cleanup, not as the final performance result.

The current BDF callback now removes one avoidable source of call inflation:
when SciPy asks for `rhs(t, y)` and then for `jac(t, y)` at the same state, the
Jacobian callback reuses the cached base RHS for that state before applying the
colored sparse finite-difference perturbations. Perturbed Jacobian residuals now
bypass the mutable RHS cache directly, so setting
`JAX_DRB_FD_JACOBIAN_THREADS=<N>` can parallelize the BDF color groups without
thread races in that cache. This is a call-count and execution-policy cleanup,
not yet a full runtime solution. A post-change unprofiled
`recycling_dthe_one_step` timing on this MacBook measured `61.38 s`, which is
within local noise and not a defensible end-to-end speedup over the earlier
`~53 s` unprofiled RSS run. Future heavy reports should therefore include the
new BDF diagnostics counters and `bdf_jacobian_parallel_workers` in addition to
wall time.

The shared sparse Newton backend now records per-step diagnostics for:

- residual evaluation count and wall time
- sparse finite-difference Jacobian refresh count and wall time
- sparse/direct or Krylov linear-solve wall time
- line-search wall time
- fallback use

Those diagnostics are intentionally attached to the solver step info rather
than hidden inside one profiler script. They can now be surfaced by recycling,
neutral, and future tokamak campaign packages when the paper needs
phase-resolved runtime evidence. The sparse finite-difference Jacobian path
also precomputes the CSC row/column extraction plan once per solve, so each
Newton refresh no longer rebuilds the same color-group indexing metadata.

For residuals that are already JAX-transformable, the solver package also has a
grouped sparse-JVP Jacobian builder. It uses `jax.linearize` and one pushed
direction per color group, avoiding finite-difference residual perturbations
altogether. The heavy recycling RHS is not yet in that category, so the
remaining runtime work is to migrate the dominant recycling residual kernels to
JAX-native array code before making the JVP path a production backend.
The sparse-JVP builder now batches those color-group pushes with `jax.vmap`.
That makes the derivative path closer to the JAX autodiff cookbook model:
linearize once, then push a matrix of tangent directions through the same
linearized residual. The public `batch_size` parameter should be used during
profiling to separate memory pressure from dispatch overhead.

The related JAX Newton path is matrix-free: JAX GMRES receives the linearized
Jacobian action as a callable rather than a materialized sparse matrix. This is
the preferred algorithmic target for future differentiable recycling kernels,
but it is intentionally not claimed as a production speedup until the residual
itself stops crossing the host/SciPy boundary.

## Basic Usage

From the repo root:

```bash
PYTHONPATH=src python3 scripts/profile_curated_case.py neutral_mixed_one_step \
  --reference-root /path/to/hermes-3 \
  --output-dir tmp/profiles/neutral_mixed_one_step \
  --warm-runs 1 \
  --timed-runs 2 \
  --rss-profile
```

The script writes:

- `profile_summary.json`
- `cprofile_top.txt`
- process-tree peak RSS fields in the summary when `--rss-profile` is enabled

and, when requested:

- `jax_trace/` for TensorBoard / Perfetto-compatible traces
- `device_memory_profile.prof` for JAX device-memory snapshots

When `--rss-profile` and cProfile are both enabled, the script collects RSS on
a separate unprofiled run so the sampler thread does not contaminate the
cProfile table.

## JAX Trace And Perfetto

To capture a JAX trace that can be opened in TensorBoard/XProf or uploaded to
Perfetto:

```bash
PYTHONPATH=src python3 scripts/profile_curated_case.py recycling_1d_one_step \
  --reference-root /path/to/hermes-3 \
  --output-dir tmp/profiles/recycling_1d_one_step \
  --jax-trace
```

This uses the official JAX tracing path described in:

- [JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html)
- [`jax.profiler.start_trace`](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.start_trace.html)

## Device Memory Profiles

On GPU-capable systems, the same script can also snapshot device memory:

```bash
PYTHONPATH=src python3 scripts/profile_curated_case.py tokamak_turbulence_one_step \
  --reference-root /path/to/hermes-3 \
  --output-dir tmp/profiles/tokamak_turbulence_one_step \
  --jax-trace \
  --device-memory-profile
```

This follows the official JAX guidance in:

- [JAX device memory profiling](https://docs.jax.dev/en/latest/device_memory_profiling.html)
- [JAX GPU memory allocation notes](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)

## Compilation Cache And XLA Dumps

The script can also set up two useful runtime diagnostics before importing JAX:

- persistent compilation cache
- XLA dump directory

Example:

```bash
PYTHONPATH=src python3 scripts/profile_curated_case.py tokamak_recycling_one_step \
  --reference-root /path/to/hermes-3 \
  --output-dir tmp/profiles/tokamak_recycling_one_step \
  --compilation-cache-dir tmp/jax_cache \
  --xla-dump-dir tmp/xla_dump \
  --jax-trace
```

The supporting JAX references are:

- [persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
- [JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html)

## GPU Workflow On `office`

The currently reachable office machine has:

- two `NVIDIA RTX A4000` GPUs
- a clean repo-local `jax_drb` environment with
  `jax[cuda12]==0.6.2`
- CUDA-visible JAX devices in that repo-local environment

The remote environment is now operational rather than speculative:

- `jax.devices()` reports `CudaDevice(id=0)` and `CudaDevice(id=1)`
- `jax.default_backend()` reports `gpu`

The first GPU-native audit on `office` is the compact selected-field profile
bundle, not the heavy host/SciPy recycling lane. Current measured results on
that GPU environment are:

- traced-field-line reduced lane
  - compile `4.41e-2 s`
  - first execute `1.23e-3 s`
  - warm execute `3.30e-4 s`
- stellarator VMEC reduced lane
  - compile `7.36e-3 s`
  - first execute `3.98e-4 s`
  - warm execute `1.14e-4 s`

That is the correct runtime split for the current codebase:

- compact native JAX lanes are ready for CPU/GPU audit
- heavy recycling lanes are still primarily CPU/host-side optimization targets

## Current Interpretation Standard

Profiler output should not be read in isolation.

- `cProfile` tells us where Python and host-side time is spent.
- JAX traces tell us where time is spent in compiled dispatches, kernels, and
  host/device synchronization.
- memory profiles tell us whether runtime problems are actually memory-pressure
  problems.
- the live Hermès rerun matrix tells us whether a faster case is still solving
  the right problem.

For `jax_drb`, those have to be read together. A faster run that worsens the
Hermès compare surface is not an improvement.
