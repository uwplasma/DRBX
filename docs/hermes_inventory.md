# Hermes-3 Inventory Notes

These notes capture the Hermes facts already driving the first implementation slices.

## Solver And Scheduler

- Hermes exposes both adaptive transient integration (`cvode`) and steady-state/backward-Euler style solves (`beuler`) in the documented workflow.
- `nout = 0` is the shortest parity loop because Hermes executes one RHS evaluation and exits.
- `ComponentScheduler::transform()` runs every component's `transform()` first and only then runs each component's `finally()` hook. JAX-DRB mirrors that contract in its initial scheduler abstraction.

## Normalization

From `hermes-3.cxx`, Hermes defines:

- `Cs0 = sqrt(qe * Tnorm / Mp)`
- `Omega_ci = qe * Bnorm / Mp`
- `rho_s0 = Cs0 / Omega_ci`
- output-unit metadata: `inv_meters_cubed -> Nnorm`, `eV -> Tnorm`, `Tesla -> Bnorm`, `seconds -> 1 / Omega_ci`, `meters -> rho_s0`

The initial normalization module reproduces those exact derived quantities and tracks both `normalise_metric` and `recalculate_metric`.

## Root And Mesh Scalars

Hermes inputs routinely define reusable scalar parameters before `[mesh]` and `[hermes]`, for example `tnorm_setting`, `core_ne`, and `initial_pi`. Mesh sections then reference local and root scalars (`dy = Ly / ny`, `dz = 2 * pi / nz`, `Bnorm = mesh:Bxy`). JAX-DRB now resolves these into a structured run configuration rather than reparsing them inside later kernels.

## Input Syntax Observations

Representative Hermes inputs require support for:

- inline comments after assignments;
- quoted strings, booleans, integers, and floats;
- symbolic expressions that must stay unevaluated unless scalar resolution is requested;
- top-level comma-separated lists such as `type = evolve_density, evolve_pressure`;
- multiline parenthesized component lists;
- Unicode `π`, section references like `mesh:Bxy`, and power syntax using `^`.

## Selected Reference Cases

The first parity ladder is recorded in [references/hermes_case_ladder.toml](/Users/rogerio/local/jax_drb/references/hermes_case_ladder.toml). It starts with one-RHS and one-step cases from integrated tests, then grows into blobs, recycling, turbulence, and the TCV X-point example.
