# Parity Harness

The first executable parity harness is centered on the curated case ladder in [references/hermes_case_ladder.toml](/Users/rogerio/local/jax_drb/references/hermes_case_ladder.toml).

## Live Hermes Protocol

`jax-drb run-reference-case <case>` performs the following steps:

1. resolve the case input under a Hermes checkout;
2. stage the case input directory into an isolated workdir using symlinks, without modifying the Hermes source tree;
3. apply parity-mode overrides:
   - `one_rhs -> nout=0`
   - `one_step -> nout=1`
4. run the Hermes binary;
5. verify `BOUT.settings`, `BOUT.log.0`, `BOUT.dmp.0.nc`, and `BOUT.restart.0.nc`;
6. summarize selected comparison variables and normalization scalars from `BOUT.dmp.0.nc`.
7. compare future JAX portable summaries against the committed Hermes baselines with `jax-drb compare-summary`.

## Confirmed Hermes Behavior

Live runs against `/Users/rogerio/local/hermes-3/build/hermes-3` established:

- `nout=0` still writes `BOUT.dmp.0.nc`, `BOUT.restart.0.nc`, `BOUT.settings`, and `BOUT.log.0`;
- for `nout=0`, `t_array` contains a single time point `(0.0,)`;
- for `nout=1`, `t_array` contains two time points, the initial state and one output step;
- `BOUT.dmp.0.nc` includes scalar normalization metadata `Nnorm`, `Tnorm`, `Bnorm`, `Cs0`, `Omega_ci`, and `rho_s0`.

These behaviors are the basis of the low-iteration parity workflow in [PLAN.md](/Users/rogerio/local/jax_drb/PLAN.md).

## Committed Hermes Baselines

The first portable baseline summaries generated from live Hermes runs are:

- [evolve_density_rhs.json](/Users/rogerio/local/jax_drb/references/baselines/hermes/evolve_density_rhs.json)
- [diffusion_one_step.json](/Users/rogerio/local/jax_drb/references/baselines/hermes/diffusion_one_step.json)

These files are not full field dumps. They intentionally store:

- parity mode and applied overrides;
- required output artifacts;
- output dimensions and time points;
- normalization scalars from `BOUT.dmp.0.nc`;
- selected comparison-variable statistics and first-to-last deltas.
