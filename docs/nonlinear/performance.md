# Performance notes

`jaxdrb` is designed to exploit JAX compilation and vectorization.

## Nonlinear kernels

The nonlinear HW2D kernel is implemented in a way that is friendly to XLA:

- FFT-based operators use `jax.numpy.fft`.
- Fixed-step time stepping uses Diffrax with a constant step size controller.
- Dealiasing uses a precomputed mask to avoid dynamic shapes.

## Benchmark

The repository includes a micro-benchmark:

```bash
python benchmarks/bench_hw2d_step.py
```

This compiles and runs short fixed-step integrations and prints throughput in steps/s
for several Diffrax solvers.

## CI regression gate

`jaxdrb` also ships a CI-oriented performance gate:

```bash
python benchmarks/check_core_kernels.py
```

This benchmarks and enforces minimum throughput for:

- HW2D nonlinear fixed-step Diffrax stepping (`steps/s`),
- linear matrix-free matvec application (`matvec/s`).

Thresholds are configurable from the CLI and are intentionally conservative to avoid
flaky failures on shared CI runners.

## CI physics gate (conservative DRB)

For the field-line cold-ion DRB branch, CI also enforces a strict conservation benchmark:

```bash
python benchmarks/check_drb_conservative_gate.py
```

This gate checks:

- instantaneous operator residuals (`dE/dt`, mean-rate residuals),
- finite-time drifts (`(E-E0)/E0`, mean invariant drifts),

on the periodic conservative subset used for hard regression protection.

## Recommended workflow for performance

- Use `jax_enable_x64=False` unless you need high-precision diagnostics.
- Keep `nx, ny` and the time-step fixed across repeated runs to maximize JIT reuse.
- Prefer fixed-step Diffrax for long runs; use adaptive Diffrax solvers for verification and stiff cases.
