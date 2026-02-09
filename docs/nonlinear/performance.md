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

## CI regression gate

`jaxdrb` also ships a CI-oriented performance gate:

```bash
python benchmarks/check_core_kernels.py
```

This benchmarks and enforces minimum throughput for:

- HW2D nonlinear RK4 stepping (`steps/s`),
- linear matrix-free matvec application (`matvec/s`).

Thresholds are configurable from the CLI and are intentionally conservative to avoid
flaky failures on shared CI runners.

## Recommended workflow for performance

- Use `jax_enable_x64=False` unless you need high-precision diagnostics.
- Keep `nx, ny` and the time-step fixed across repeated runs to maximize JIT reuse.
- Prefer `lax.scan` stepping for long runs; use Diffrax for reference/verification and adaptive stepping.
