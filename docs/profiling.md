# Profiling (JAX/XLA + Memory)

This repository includes a lightweight profiler that captures **XLA HLO**, **kernel traces**, and **device memory** for a short jax_drb run.
It stays fully differentiable and uses only JAX built‑ins.

## Quick Start

```
python /Users/rogerio/local/jax_drb/tools/profile_jaxdrb.py \
  --config /Users/rogerio/local/jax_drb/benchmarks/cases/jaxdrb/salpha_linear.toml \
  --steps 200 \
  --dt 1e-3 \
  --outdir /Users/rogerio/local/jax_drb/benchmarks/profiles/salpha_linear
```

## Outputs

The output directory contains:

- `jaxdrb_scan.hlo.txt` and/or `jaxdrb_scan.stablehlo.txt`  
  XLA HLO / StableHLO for the compiled scan.
- `compile_stats.json`  
  Backend/device info and executable size.
- `memory_profile.pb`  
  Device memory profile (if supported by your JAX build).
- `timing.txt`  
  Wall‑clock timing and time‑per‑step.
- `plugins/profile/...`  
  Trace events for TensorBoard’s profiler UI.

## Viewing the Trace

Run TensorBoard and open the “Profile” tab:

```
tensorboard --logdir /Users/rogerio/local/jax_drb/benchmarks/profiles/salpha_linear
```

For low‑level timeline inspection, you can also open the trace in Chrome:
`chrome://tracing` → Load trace from the `plugins/profile/` directory.

## Notes

- Use **small step counts** for profiling; the goal is kernel inspection, not long dynamics.
- If `memory_profile.pb` is missing, your JAX build may not support memory profiling.
- For GPU runs, ensure the correct backend is active before profiling.
- `--warm-start` enables Poisson warm‑start caching (default).
- Use `time.poisson_track_iters = true` in configs to record per‑saved‑frame CG
  iteration stats (mean/max over the RK4 steps since the last save).
- Kernel traces now include named scopes for `poisson_solve`, `bracket_terms`,
  `curvature`, and `parallel_*` blocks to simplify attribution.

## Comparing Warm‑Start vs Cold‑Start

For a quick A/B run, use the tiny preset config:

```
python tools/profile_jaxdrb.py \
  --config benchmarks/cases/jaxdrb/benchmark_preset_linear.toml \
  --steps 50 \
  --dt 1e-3 \
  --outdir benchmarks/profiles/preset_linear_warm \
  --warm-start

python tools/profile_jaxdrb.py \
  --config benchmarks/cases/jaxdrb/benchmark_preset_linear.toml \
  --steps 50 \
  --dt 1e-3 \
  --outdir benchmarks/profiles/preset_linear_cold \
  --no-warm-start
```

This will produce two `timing.txt` files plus HLO/trace artifacts to compare the
impact of Poisson warm‑start and the new preconditioner caches.

## Non‑periodic Poisson (Dirichlet/Neumann)

To stress CG + preconditioner caching, use the non‑periodic presets:

```
python tools/profile_jaxdrb.py \
  --config benchmarks/cases/jaxdrb/benchmark_preset_linear_dirichlet.toml \
  --steps 50 \
  --dt 1e-3 \
  --outdir benchmarks/profiles/preset_linear_dirichlet_warm \
  --warm-start

python tools/profile_jaxdrb.py \
  --config benchmarks/cases/jaxdrb/benchmark_preset_linear_neumann.toml \
  --steps 50 \
  --dt 1e-3 \
  --outdir benchmarks/profiles/preset_linear_neumann_warm \
  --warm-start
```

Compare with `--no-warm-start` runs to quantify CG warm‑start impact.

If you want to stress the new FFT‑based preconditioner for non‑periodic CG solves,
set:

```toml
[numerics]
poisson_preconditioner = "fd_fft"
```

(`auto` keeps `jacobi` for non‑periodic BCs; use `fd_fft` explicitly when it wins.)

## Non‑Boussinesq CG Benchmark (fd_fft vs jacobi)

Two configs are provided to quantify the fd‑FFT preconditioner win on the
variable‑coefficient polarization solve:

- `/Users/rogerio/local/jax_drb/benchmarks/cases/jaxdrb/benchmark_preset_linear_dirichlet_nonbouss_fdfft.toml`
- `/Users/rogerio/local/jax_drb/benchmarks/cases/jaxdrb/benchmark_preset_linear_dirichlet_nonbouss_jacobi.toml`
