# Parity Harness

The first executable parity harness is centered on the curated case ladder in [references/reference_case_ladder.toml](/Users/rogerio/local/jax_drb/references/reference_case_ladder.toml).

## Live Reference Protocol

`jax-drb run-reference-case <case>` performs the following steps:

1. resolve the case input under a reference checkout;
2. stage the case input directory into an isolated workdir using symlinks, without modifying the reference source tree;
3. apply parity-mode overrides:
   - `one_rhs -> nout=0`
   - `one_step -> nout=1`
4. run the reference binary;
5. verify `BOUT.settings`, `BOUT.log.0`, `BOUT.dmp.0.nc`, and `BOUT.restart.0.nc`;
6. summarize selected comparison variables and normalization scalars from `BOUT.dmp.0.nc`.
7. compare future JAX portable summaries against the committed reference baselines with `jax-drb compare-summary`.

If requested, the same command can also emit full comparison arrays to compressed NPZ files. Those artifacts are intended for the smallest curated cases where full-field regression is practical.

## Native Protocol

`jax-drb run-case <case>` resolves the same curated input and runs the supported native JAX path.

Current support is intentionally narrow:

- `evolve_density_rhs` is implemented end to end;
- `diffusion_one_step` is implemented for the current structured, axisymmetric, `nz = 1` transport benchmark;
- `diffusion_short_window` is implemented on the same transport path, using the configured output cadence from the curated input;
- the native runner builds the structured mesh, evaluates the configured initial profile on the JAX grid, reconstructs the current X/Y guards, builds the normalized structured metrics, and emits the portable summary schema;
- the same native run can emit compressed full-array parity artifacts, so small cases can be checked at field level with `jax-drb compare-arrays`;
- the resulting JSON can be compared directly against the committed baseline with `jax-drb compare-summary`.

For the current one-step diffusion milestone, summary comparison should use a modest scalar tolerance, for example:

```bash
PYTHONPATH=src python -m jax_drb compare-summary \
  references/baselines/reference/diffusion_one_step.json \
  /tmp/jax_drb_diffusion_one_step_native.json \
  --scalar-rtol 1e-3 \
  --scalar-atol 2e-6
```

That tolerance is only for the first transport milestone. The intent is to tighten it as more of the operator and time-integration stack becomes native and shared across cases.

## Confirmed Reference Behavior

Live runs against `local reference build` established:

- `nout=0` still writes `BOUT.dmp.0.nc`, `BOUT.restart.0.nc`, `BOUT.settings`, and `BOUT.log.0`;
- for `nout=0`, `t_array` contains a single time point `(0.0,)`;
- for `nout=1`, `t_array` contains two time points, the initial state and one output step;
- `BOUT.dmp.0.nc` includes scalar normalization metadata `Nnorm`, `Tnorm`, `Bnorm`, `Cs0`, `Omega_ci`, and `rho_s0`.

These behaviors are the basis of the low-iteration parity workflow in [PLAN.md](/Users/rogerio/local/jax_drb/PLAN.md).

## Committed Reference Baselines

The first portable baseline summaries generated from live reference runs are:

- [evolve_density_rhs.json](/Users/rogerio/local/jax_drb/references/baselines/reference/evolve_density_rhs.json)
- [diffusion_one_step.json](/Users/rogerio/local/jax_drb/references/baselines/reference/diffusion_one_step.json)
- [diffusion_short_window.json](/Users/rogerio/local/jax_drb/references/baselines/reference/diffusion_short_window.json)

These files are not full field dumps. They intentionally store:

- parity mode and applied overrides;
- required output artifacts;
- output dimensions and time points;
- normalization scalars from `BOUT.dmp.0.nc`;
- selected comparison-variable statistics and first-to-last deltas.

Future JAX runs should emit the same portable schema through the generic summary helpers in [portable.py](/Users/rogerio/local/jax_drb/src/jax_drb/parity/portable.py), so that `jax-drb compare-summary` can be used unchanged for reference vs. JAX comparisons.

The first committed full-array baselines are:

- [diffusion_one_step.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/diffusion_one_step.npz)
- [diffusion_short_window.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/diffusion_short_window.npz)

These are written and read through [arrays.py](/Users/rogerio/local/jax_drb/src/jax_drb/parity/arrays.py). For the current diffusion milestone, the intended comparison command is:

```bash
PYTHONPATH=src python -m jax_drb compare-arrays \
  references/baselines/reference_arrays/diffusion_short_window.npz \
  /tmp/jax_drb_diffusion_short_window_native.npz \
  --array-rtol 2e-4 \
  --array-atol 2e-6
```
