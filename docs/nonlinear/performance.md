# Performance notes

`jaxdrb` is designed to exploit JAX compilation and vectorization.

## Nonlinear kernels

The nonlinear HW2D kernel is implemented in a way that is friendly to XLA:

- FFT-based operators use `jax.numpy.fft`.
- Fixed-step time stepping uses `jax.lax.scan` to avoid Python loops in the compiled region.
- Dealiasing uses a precomputed mask to avoid dynamic shapes.

## Benchmark

The repository includes a micro-benchmark:

```bash
python benchmarks/bench_hw2d_step.py
```

This compiles and runs a short RK4 integration and prints a rough throughput in steps/s.

## Recommended workflow for performance

- Use `jax_enable_x64=False` unless you need high-precision diagnostics.
- Keep `nx, ny` and the time-step fixed across repeated runs to maximize JIT reuse.
- Prefer `lax.scan` stepping for long runs; use Diffrax for reference/verification and adaptive stepping.

