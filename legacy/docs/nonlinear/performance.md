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

## Solver comparison (Diffrax)

We include a lightweight solver comparison that uses a short DRB2D run and a
smaller-step Dopri8 reference to estimate solver accuracy vs runtime:

```bash
python examples/10_verification/drb2d_solver_comparison.py
```

![DRB2D solver comparison](../assets/images/drb2d_solver_comparison.png)

**Guidance**:
- Use `dopri5`/`tsit5` for fast fixed-step runs and regression gates.
- Use `dopri8` or adaptive stepping when validating stiff closures or when you need
  tighter accuracy.

## Non-Boussinesq Poisson preconditioning

The non-Boussinesq polarization solve uses a variable-coefficient SPD operator. We
ship a small benchmark comparing Jacobi vs FFT/circulant preconditioning:

```bash
python examples/10_verification/poisson_preconditioner_bench.py
```

![Poisson preconditioner benchmark](../assets/images/poisson_preconditioner_bench.png)

**Guidance**:
- For periodic grids, the spectral/circulant preconditioner often reduces residuals
  at the cost of extra FFTs. It tends to win once you amortize repeated solves.
- For non-periodic BCs, Jacobi is robust and cheap; start there, then switch to
  spectral if you see slow CG convergence.
