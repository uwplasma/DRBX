# Implementation Inventory

These notes capture the external-reference facts already driving the first implementation slices.

## Solver And Scheduler

- the external reference uses both adaptive transient integration (`cvode`) and steady-state/backward-Euler style solves (`beuler`) in the documented workflow.
- `nout = 0` is the shortest parity loop because the reference executes one RHS evaluation and exits.
- `ComponentScheduler::transform()` runs every component's `transform()` first and only then runs each component's `finally()` hook. JAX-DRB mirrors that contract in its initial scheduler abstraction.

## Normalization

From the main source driver, the reference defines:

- `Cs0 = sqrt(qe * Tnorm / Mp)`
- `Omega_ci = qe * Bnorm / Mp`
- `rho_s0 = Cs0 / Omega_ci`
- output-unit metadata: `inv_meters_cubed -> Nnorm`, `eV -> Tnorm`, `Tesla -> Bnorm`, `seconds -> 1 / Omega_ci`, `meters -> rho_s0`

The initial normalization module reproduces those exact derived quantities and tracks both `normalise_metric` and `recalculate_metric`.

## Root And Mesh Scalars

Reference inputs routinely define reusable scalar parameters before `[mesh]` and `[model]`, for example `tnorm_setting`, `core_ne`, and `initial_pi`. Mesh sections then reference local and root scalars (`dy = Ly / ny`, `dz = 2 * pi / nz`, `Bnorm = mesh:Bxy`). JAX-DRB now resolves these into a structured run configuration rather than reparsing them inside later kernels.

## Live Output Facts

Direct runs against the local reference build confirmed:

- `nout=0` writes a `BOUT.dmp.0.nc` file with `t_array = [0.0]`;
- `nout=1` writes initial plus one evolved output time slice;
- scalar normalization metadata is present directly in the dump file;
- for the structured identity-metric transport cases, the dumped metric fields follow the normalized forms `dx / (rho_s0^2 * Bnorm)`, `J / rho_s0`, `g11 / rho_s0^2`, and `Bxy / Bnorm`;
- the first portable reference baselines are stored in [references/baselines/reference](/Users/rogerio/local/jax_drb/references/baselines/reference).
- the first native JAX execution path now matches the committed `evolve_density_rhs` portable baseline exactly, including dimensions, scalar metadata, and variable summary statistics.
- the native one-step transport path now reproduces the committed `diffusion_one_step` summary statistics within regression tolerance, using structured metrics, strict Heaviside support, Neumann guard reconstruction, and an exact matrix-exponential radial advance.
- the transport parity harness now also stores full comparison arrays in [references/baselines/reference_arrays](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays), so regressions can be checked against complete fields instead of summaries only.
- the same transport slice now covers a short-window benchmark, `diffusion_short_window`, using the configured output cadence from the input file.

## Input Syntax Observations

Representative reference inputs require support for:

- inline comments after assignments;
- quoted strings, booleans, integers, and floats;
- symbolic expressions that must stay unevaluated unless scalar resolution is requested;
- top-level comma-separated lists such as `type = evolve_density, evolve_pressure`;
- multiline parenthesized component lists;
- Unicode `π`, section references like `mesh:Bxy`, and power syntax using `^`.

## Selected Reference Cases

The first parity ladder is recorded in [references/reference_case_ladder.toml](/Users/rogerio/local/jax_drb/references/reference_case_ladder.toml). It starts with one-RHS and one-step cases from integrated tests, then grows into blobs, recycling, turbulence, and the TCV X-point example.

Current native execution coverage:

- `evolve_density_rhs`: implemented and regression-tested;
- `diffusion_one_step`: implemented and regression-tested as the first genuine time-advance benchmark;
- `diffusion_short_window`: implemented and regression-tested at both summary and full-array level;
- the current diffusion history path has JIT and `grad` smoke coverage, so the first transport slice is exercised as an actual differentiable JAX computation rather than only an eager NumPy-style check;
- next target: extend from this one-step transport slice into reusable finite-volume operator kernels and longer-window transport cases.
