# Benchmark Presets

This repo includes **term schedule presets** for quick performance checks and
baseline comparisons. These presets are intended to reduce the RHS cost while
preserving a physically meaningful subset of the DRB system.

## Presets

Available presets (from `jaxdrb.core.terms.registry.PRESET_TERM_SCHEDULES`):

- `benchmark_linear`: parallel + curvature + drive + diffusion (no nonlinear ExB).
- `benchmark_nonlinear`: adds ExB advection to `benchmark_linear`.
- `benchmark_min`: advection + parallel + curvature + diffusion (no drive).

You can activate a preset via:

```toml
[numerics]
term_schedule_preset = "benchmark_linear"
```

## Example Config

See:

`/Users/rogerio/local/jax_drb/benchmarks/cases/jaxdrb/benchmark_preset_linear.toml`

This config uses a tiny `s-alpha` grid with `term_schedule_preset = "benchmark_linear"`
and `diag_mode = "basic"` to keep runtime and memory minimal.

Additional preset configs:
- `/Users/rogerio/local/jax_drb/benchmarks/cases/jaxdrb/benchmark_preset_nonlinear.toml`
- `/Users/rogerio/local/jax_drb/benchmarks/cases/jaxdrb/benchmark_preset_linear_dirichlet.toml`
- `/Users/rogerio/local/jax_drb/benchmarks/cases/jaxdrb/benchmark_preset_linear_neumann.toml`
