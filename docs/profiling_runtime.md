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
  - live Hermès rerun ratio improved from about `4.27x` to about `2.79x`
  - the dominant mismatch remains the boundary-localized `NVh` field
- `recycling_dthe_one_step`
  - timed local mean dropped from about `75.3 s` to about `54.1 s` after the
    reaction/source allocation cleanup
  - live Hermès rerun ratio improved from about `8.45x` to about `7.81x`
  - the fidelity band stayed essentially unchanged at about `4.9e-3` relative
    RMS on `NVd`

The next heavy CPU optimization target is no longer generic reaction
allocation. The refreshed cProfile still shows the dominant remaining work in:

- the SciPy BDF history path itself
- finite-difference Jacobian assembly
- neutral parallel diffusion
- collision closure
- target recycling / target boundary-source assembly
- prepared-state and boundary setup on the open-field lane

## Basic Usage

From the repo root:

```bash
PYTHONPATH=src python3 scripts/profile_curated_case.py neutral_mixed_one_step \
  --reference-root /path/to/hermes-3 \
  --output-dir tmp/profiles/neutral_mixed_one_step \
  --warm-runs 1 \
  --timed-runs 2
```

The script writes:

- `profile_summary.json`
- `cprofile_top.txt`

and, when requested:

- `jax_trace/` for TensorBoard / Perfetto-compatible traces
- `device_memory_profile.prof` for JAX device-memory snapshots

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
