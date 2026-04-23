# Profiling And Runtime Workflow

This page records the supported profiling workflow for `jax_drb` so runtime
work is reproducible instead of based on one-off local shell snippets.

The current public entry point is:

- [scripts/profile_curated_case.py](../scripts/profile_curated_case.py)

It is meant for the real worst offenders and live validation lanes, not only
for compact microbenchmarks.

## Recommended CPU Cases

The current highest-value local CPU cases are:

- `neutral_mixed_one_step`
- `recycling_1d_one_step`
- `recycling_dthe_one_step`
- `integrated_2d_recycling_one_step`
- `tokamak_recycling_one_step`

These are the cases where the current same-machine Hermès rerun matrix still
shows either the largest runtime ratios, the largest fidelity gaps, or both.

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
- CUDA-visible JAX devices under the system Python environment

The current blocker is not hardware visibility. It is environment consistency:
the repo still needs a clean `jax_drb` environment there before we can trust
full GPU parity and runtime campaigns. The current remote observation is:

- JAX sees both CUDA devices
- `numpy`, `scipy`, `matplotlib`, and `netCDF4` import cleanly
- `diffrax` and `equinox` fail in the current environment because of a
  `jaxlib` extension mismatch

That means the next GPU runtime step is:

1. create a clean repo-local environment on `office`
2. install `jax_drb` and the exact matching JAX/JAXLIB pair
3. rerun the curated profiling cases above
4. then collect TensorBoard/XProf traces, device-memory profiles, and
   same-machine native-versus-Hermès timings on GPU

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
