# Running Simulations

The production driver lives in `jaxdrb.driver.run_simulation` and is exposed via
the CLI `jaxdrb --run`. It supports a **JIT‑compiled fixed‑step scan** as well as
**Diffrax** solvers with adaptive stepping.

## `[time]` Configuration

Defaults: `method="diffrax"`, `solver="dopri8"`, `adaptive=true`, `progress=true`.

```toml
[time]
method = "diffrax"    # rk4_scan | diffrax
dt = 1e-3
nsteps = 1000
save_every = 10
t_end = 1.0           # optional; overrides nsteps*dt for diffrax
```

### Common Options
- `save_every`: save diagnostics every N steps.
- `remat`: `true` enables checkpointing for lower memory in long runs.
- `return_numpy`: `true` transfers diagnostics to host memory (needed when saving).
- `diag_mode`: `full` (default) or `basic` (skip Poisson and only compute RMS(n, Te, omega)).
- `diag_phi_every`: compute `phi` diagnostics only every N saved frames (default: 1).
- `poisson_warm_start`: reuse the previous `phi` as CG initial guess (RK4 scan only).
- `poisson_track_iters`: record mean/max CG iteration stats per saved frame
  (averaged over the RK4 steps since the last save; RK4 scan only).

## JIT Fixed‑Step (RK4 Scan)

```toml
[time]
method = "rk4_scan"
dt = 1e-3
nsteps = 2000
save_every = 20
remat = false
```

The RK4 scan path is **JIT‑compiled by default** and is optimized for throughput.
It is fully differentiable and can optionally use `remat = true` for memory savings.

## Diffrax (Adaptive / High‑Order)

```toml
[time]
method = "diffrax"
solver = "dopri8"     # dopri8 | dopri5 | tsit5 | euler
adaptive = true
rtol = 1e-5
atol = 1e-7
progress = true       # tqdm‑like progress meter
jit = false           # set true to JIT‑compile (disables progress meter)
```

Notes:
- When `jit = true`, the progress meter is disabled (Diffrax cannot emit progress
  callbacks from inside a JIT‑compiled solve).
- `remat = true` uses a checkpointing adjoint (`RecursiveCheckpointAdjoint`).
- If `bc_x/bc_y` are periodic, Poisson inversion defaults to spectral for speed
  (`numerics.poisson_force_spectral_when_periodic = true`).
- For non‑periodic BCs, Poisson inversion defaults to the FD‑FFT solver when
  `numerics.poisson_force_fd_fft_when_nonperiodic = true` (significantly faster
  than CG for Dirichlet/Neumann).

## CLI Example

```bash
jaxdrb /path/to/input.toml --run --output /tmp/jaxdrb_out.npz
```

When `--output` is provided, the CLI forces `return_numpy = true` so diagnostics
are transferred to the host before saving.

## Compilation Cache

The CLI enables JAX’s persistent compilation cache by default. You can override
the directory or disable it:

```bash
jaxdrb /path/to/input.toml --compile-cache ~/.cache/jaxdrb/compilation
jaxdrb /path/to/input.toml --compile-cache off
```
