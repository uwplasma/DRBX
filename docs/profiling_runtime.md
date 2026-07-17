# Profiling And Runtime Workflow

!!! note "Plan authority"
    This page documents the reproducible profiling workflow. The active
    execution plan is
    [Research-Grade Execution Plan](research_grade_execution_plan.md). If this
    page conflicts with that plan, follow the execution plan and update this
    page afterward.

This page describes how to profile the compact JAX-native lanes and how to read
the resulting artifacts. The higher-level performance picture is in
[Performance And Differentiability](performance_and_differentiability.md).

## Recommended Cases

The highest-value profiling surface today is the non-axisymmetric 3-D PyTree
drift-reduced Braginskii FCI lane, because it exercises the field-line-map
operators, the sheath/neutral/vorticity closures, and the differentiable RHS in
one compiled kernel. The compact diffusion, electrostatic-vorticity, and
Hasegawa-Wakatani lanes are lighter surfaces for precision and execution-mode
studies.

## Basic Usage

From the repo root, profile the 3-D PyTree DRB lane with
[scripts/profile_stellarator_drb_pytree.py](../scripts/profile_stellarator_drb_pytree.py):

```bash
PYTHONPATH=src python3 scripts/profile_stellarator_drb_pytree.py \
  --nx 18 --ny 16 --nz 32 --steps 8 \
  --output-dir tmp/profiles/stellarator_drb_pytree \
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
- a JAX device-memory snapshot

When `--rss-profile` and cProfile are both enabled, the script collects RSS on a
separate unprofiled run so the sampler thread does not contaminate the cProfile
table. Use `--skip-cprofile` to disable cProfile collection.

## JAX Trace And Perfetto

To capture a JAX trace that can be opened in TensorBoard/XProf or uploaded to
Perfetto, add `--jax-trace`:

```bash
PYTHONPATH=src python3 scripts/profile_stellarator_drb_pytree.py \
  --output-dir tmp/profiles/stellarator_drb_pytree \
  --jax-trace
```

This uses the official JAX tracing path described in:

- [JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html)
- [`jax.profiler.start_trace`](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.start_trace.html)

## Device Memory Profiles

On GPU-capable systems, the same script can also snapshot device memory:

```bash
PYTHONPATH=src python3 scripts/profile_stellarator_drb_pytree.py \
  --output-dir tmp/profiles/stellarator_drb_pytree \
  --jax-trace \
  --device-memory-profile
```

This follows the official JAX guidance in:

- [JAX device memory profiling](https://docs.jax.dev/en/latest/device_memory_profiling.html)
- [JAX GPU memory allocation notes](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)

## Compilation Cache And XLA Dumps

The script can also set up two useful runtime diagnostics before importing JAX:

- a persistent compilation cache;
- an XLA dump directory.

Example:

```bash
PYTHONPATH=src python3 scripts/profile_stellarator_drb_pytree.py \
  --output-dir tmp/profiles/stellarator_drb_pytree \
  --compilation-cache-dir tmp/jax_cache \
  --xla-dump-dir tmp/xla_dump \
  --jax-trace
```

The supporting JAX references are:

- [persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
- [JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html)

## GPU Workflow On Self-Hosted Machines

The reachable `office` machine exposes two `NVIDIA RTX A4000` GPUs. Before
profiling on a self-hosted GPU node, confirm the backend in the same shell you
will run the profiler from:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
PYTHONPATH=src python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

The compact native JAX lanes are ready for CPU/GPU audit once the profiling
script is rerun on the intended backend with `--jax-trace`,
`--device-memory-profile`, and a persistent compilation cache enabled.

## Current Interpretation Standard

Profiler output should not be read in isolation.

- `cProfile` tells us where Python and host-side time is spent.
- JAX traces tell us where time is spent in compiled dispatches, kernels, and
  host/device synchronization.
- memory profiles tell us whether runtime problems are actually memory-pressure
  problems.

For `dkx`, those have to be read together, and any speedup should be checked
against the validation diagnostics so that a faster case is still solving the
right problem.
