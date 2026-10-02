# P08 step 6: transverse-wave check on the `compact_c3` re-freeze artifacts

Computation only (CPU), one node, **no artifact rebuild**. The P reconstruction switches from a coupled Cartesian quartic
to a ringwise fit at `u = 0.21` (`inner_support = fixed_radius`). Per-ring sampling showed a weak band `u` 0.12 - 0.21
(P07 N - O order about 1.4 - 1.8) and the catalogue fields used so far have little degree >= 4 content there. This campaign
measures, on the FULL grids N32 / N48 / N64 with the artifacts of the completed re-freeze campaign
(`scripts/p08_step5_compact_c3/`), the combined perpendicular RHS (prescribed-phi and solved-phi arms) and the Dirichlet
psi solve for **transverse** manufactured fields, globally and per region, exactly like step 5.3 does for the catalogue
fields: the 5.3 machinery (`p08_step5_combined`: `reference_chunk`, `jaxstage.run_variant`, `reduction`) is reused
unchanged; only the manufactured fields, the owner averages and the region list (new `u` bands) differ.

Per grid, 32 -> 48 -> 64, resumable, one writer (`runner.lock`, `.runner.lock`):

```
owner_values (pooled) -> references (pooled) -> jax (one process) -> reduce -> sharding (subprocess, 8 forced host devices)
```

Nothing is accepted or rejected: solver gates (discrete consistency, convergence), finiteness are recorded pass/fail,
the headline order criterion is informational, **the user decides**.

## Fields (`configuration.json`, `fields.py`)

Real, O(1), smooth (including at the axis) functions of `(u, theta, eta)`, `x = u cos(theta)`, `y = u sin(theta)`; `eta`
enters as `(2 pi / eta_period) eta` (`eta_period = 2 pi` for the HSX grids). Values, gradients and Hessians are
`jax.jacfwd` of the `jnp` definition in float64 (components `d/du, d/dtheta, d/deta`, the `P06NState` convention).

| field set `transverse` | definition |
|---|---|
| `n` | `1 + 0.5 Re exp(i(2 pi x_0 / 2 + eta))` (direction 0 deg, lambda 2) |
| `Te` | `1 + 0.5 Im exp(i(2 pi x_60 / 2 + eta))` (direction 60 deg, lambda 2) |
| `Ti` | `1 + 0.5 Re exp(i(2 pi x_120 / 4 + eta))` (direction 120 deg, lambda 4) |
| `omega` | `B(u^2) (x^4 - 6 x^2 y^2 + y^4) / u_s^4 cos(eta)`, `B(s) = exp(-((s - u_s^2)/w2)^2)`, `u_s = 0.21`, `w2 = 0.06` |
| `phi` | `B(u^2) 4 (x^3 y - x y^3) / u_s^4 cos(eta + 0.3) + 0.5 Re exp(i(2 pi x_30 / 2 + eta))` |

Field set `transverse_phi_wave`: the same `n, Te, Ti, omega`, `phi = Re exp(i(2 pi x_150 / 2 + eta))`. Variants
`transverse_dirichlet` and `transverse_phi_wave_dirichlet` (all five fields Dirichlet). `x_a = x cos(a) + y sin(a)`.

**Deviation from the brief (needs your decision):** `n`, `Te`, `Ti` carry the offset 1 and amplitude 0.5 instead of being
the zero-mean waves themselves. The production curvature matrix divides by `n` (`curvature_principal_matrix`: `4 Te^2 /
(3 n)`, `4 Ti Te / (3 n)`, `2 B^2 (Te + tau Ti) / n`, `n_safe = max(n, 1e-30)`), so a zero-mean wave density (which vanishes
on a surface) makes the Te, Ti and omega curvature references about 1e31 (seen in the first bounded preflight) and the
operator meaningless; the catalogue fields are `1 + small` as well. The transverse wave content (lambda 2 / 4 plane waves)
is unchanged, only its amplitude is 0.5 (relative errors do not depend on it for the linear parts). Changing it is a
configuration edit (`fields` in `configuration.json` and `PINNED_FIELDS` in `fields.py`).

## Owner averages

Computed exactly like the frozen P06N `owner_values` (`p06n_field_derived_global.core.observation_chunk` +
`reduce_observations` = `p05n_field_derived_global.operator.live_observations`): `sum_{raw cells of the owner}
raw_volume * f(raw midpoint) / owner_volume`, i.e. the raw-volume-weighted average of the field at the raw-cell midpoints
`t.pts`; an owner that is a single raw cell gets its midpoint value (this is what the frozen P06N routine does; the brief's
"exact owner average ... NOT midpoint values" is read as "the P06N routine", which the bounded preflight verifies by
reproducing the committed P06N N32 `owner_values.npz` with the same code, max difference 3e-16). The pooled
`owner_values` stage accumulates in the same order as the one-shot `np.add.at`, so a chunked run is bit-identical to it
(checked in the preflight and in the tests).

## Regions

The P06N region masks of 5.3 plus `u` bands by owner ring centre (owners are single-ring on all three grids, checked; a
multi-ring owner is refused): `uband_0.00-0.06`, `uband_0.06-0.12`, `uband_0.12-0.21`, `uband_0.21-0.27`,
`uband_0.27-0.40`, `uband_0.40-1.00` (`[lo, hi)`, the last one closed). They go into `N{n}/context.npz`, so the 5.3
reduction reduces them like any region (owners per band at N32: 160 / 384 / 1280 / 1024 / 3072 / 19456).

## Inputs

* `--refreeze-campaign DIR`: the completed `p08_step5_compact_c3` campaign folder (**read only**): `artifact/N{n}/` (full
  artifacts incl. rows, `geometry.npz`, `census.npz`), `provenance/inputs.json` (recorded `artifact_identity_sha256`),
  `validation.json`, `localized_sidecar.json`, `summary/step5_combined_summary.json` (the catalogue errors). The campaign
  refuses a folder that is not a complete re-freeze campaign, carries other operator options (`provenance/inputs.json`,
  `validation.json`, every artifact build identity; pinned `curvature=autodiff, face_quadrature=q2,
  inner_support=fixed_radius, bfield_toroidal=compact_c3`), or whose artifact identities (build identity on disk vs
  `provenance/inputs.json` vs `validation.json`) differ, and, for a real run, an artifact folder without `geometry.npz`,
  `census.npz` or any row chunk file of its manifest (existence and size).
* `--input-root`, `--oracle-root` as in `scripts/p08_step5_compact_c3/README.md` (the immutable 17-file input root; the
  P06N `owner_values` oracle files, hash-checked by `verify-inputs` for the requested grids and by `preflight` for N32 -
  the preflight reproduces the frozen file; a `run` does not read them).
* `--output`: a fresh folder; everything goes under it.
* `--metadata-only` (**local testing only**, `verify-inputs` and `preflight`): skip the rows / geometry existence check for
  a stripped copy of the re-freeze folder; refused for every other command, recorded in `provenance/inputs.json`, and
  `run-stage jax` refuses such provenance.

## Commands (from `DRBX/scripts`, never from inside a package directory)

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
A=(--refreeze-campaign "$REFREEZE" --input-root "$INPUT_ROOT" --oracle-root "$ORACLE_ROOT" --output "$OUT")

python -m p08_step6_global.campaign verify-inputs "${A[@]}"
python -m p08_step6_global.campaign preflight     "${A[@]}"                  # N32, bounded, no artifact rows; must pass before run
python -m p08_step6_global.campaign run           "${A[@]}" --resolutions 32 48 64 --workers 96
python -m p08_step6_global.campaign validate      "${A[@]}" --resolutions 32 48 64
python -m p08_step6_global.campaign analyze       "${A[@]}" --resolutions 32 48 64
# recovery, stage by stage (not preflight-gated)
python -m p08_step6_global.campaign run-stage     "${A[@]}" --stage owner_values --n 64 --workers 96
python -m p08_step6_global.campaign run-stage     "${A[@]}" --stage references   --n 64 --workers 96
python -m p08_step6_global.campaign run-stage     "${A[@]}" --stage jax          --n 64
python -m p08_step6_global.campaign run-stage     "${A[@]}" --stage reduce       --n 64
python -m p08_step6_global.campaign run-stage     "${A[@]}" --stage sharding     --n 64 [--sharding-max-shards 4]
```

`--workers` is the pool size of `owner_values` and `references` (the JAX stage is one process); `--memory-budget-gib B
--worker-memory-gib W --memory-reserve-gib R` cap the pool as in steps 1-5 (`min(workers, (B - R) // W)`);
`--max-tasks-per-worker` recycles workers.

* **`preflight`** (N32, bounded, about 25 s and 3.5 GiB peak RSS on a laptop; needs the environment, no artifact): the
  `compact_c3` evaluator and the eta period `2 pi` are verified; the owner averages (chunked vs one-shot bitwise, vs an
  exactly-rounded per-owner `math.fsum` on the 12 closure owners + 8 multi-cell owners, and the same routine on the P06N
  catalogue vs the committed `N32.owner_values.npz`); the 5.3 references of both field sets on the 12 `owner_closure`
  owners (finite, non-zero, `psi` operators non-zero, and `perpendicular_reference_rhs.production_formula_check` against
  the independent production expressions, tolerance 1e-8); gradients vs central differences of the values (<= 1e-7) and
  Hessians vs central differences of the gradients (<= 1e-6, symmetric) at random points, the closure support points and
  points in the switch band; the axis (finite derivatives and `O(u)` angular variation down to `u = 0`); the `u` bands
  partition the owners. `run` refuses without a matching passing preflight.
* **`run`**: per grid the five stages; it stops after the first grid whose `grid_pass` fails or whose sharding record is
  `fail` (data are checkpointed), then runs `validate` and `analyze` on the grids done, and exits nonzero on a failure
  (`last_exit.json` status `failed_gates`).
* **`validate`**: requires the preflight, the reduction of every requested grid (same identity, computed on the artifact
  recorded in the provenance); writes `validation.json` (`solver_gates_pass`, `all_finite`, `grids_pass`, the informational
  headline order criterion, per-grid solver numbers, the sharding record per grid and `sharding.status`).
* **`analyze`**: `summary/step6_report.md` and `summary/step6_summary.json`: the 5.3 report (gates, headline criterion,
  per-term errors and orders, potential error, psi diffusion), the **u-band table** (term `total`, relative L2 per band for
  both arms, both variants, all four fields, with orders), the psi `O - R_mid`, `N - O`, `N - R_mid` per band, and the
  **comparison with the re-freeze catalogue** (`main_phi_dirichlet` and `dirichlet_rich` global `total` relative L2 / L2
  and orders read from the re-freeze campaign's `summary/step5_combined_summary.json`, transverse / catalogue ratios, the
  potential error and psi N - O).

## Sharding stage

`sharding.py` (`sharding.run(*, n, args, identity, inputs, cfg) -> dict`, called by `run` after `reduce` of every grid and by
`run-stage --stage sharding --n N`; the record is stored in `N{n}/sharding.json`) checks the eta-sharded perpendicular
library (`fci_perpendicular_sharding`, `fci_perpendicular_phi_sharding`) against the single device on the SAME prepared
grid as the JAX stage (this package's `jaxstage.prepare`: the re-freeze artifact plan, the phi solver, boundary data of
5.3's `variant_boundary_data` / `psi_boundary_data`). Settings: the pinned `sharding` block of `configuration.json`.

* **Subprocess.** `sharding.run` starts `python sharding.py --worker spec.json --result result.json` with
  `XLA_FLAGS=--xla_force_host_platform_device_count=8` (plus `JAX_PLATFORMS=cpu`, x64, the thread settings), so the campaign
  process stays single-device. Log: `$OUT/logs/sharding_N{n}.log`; spec / result: `$OUT/work/sharding/N{n}/`. A crashed
  worker (OOM, lowering error) raises with the log tail and writes no record (rerun `run-stage --stage sharding`).
* **Shard counts** `(2, 4, 8)` filtered to those dividing `n` with a block of at least 3 planes (all three on N32 / N48 /
  N64). `--sharding-max-shards S` (or `sharding.max_shards`) drops the counts above `S` (for a memory-limited node): a capped
  run that passes is recorded as `pending` (`shard_counts_skipped`), is not skipped on resume and is redone without the cap.
* **Variants**: `transverse_dirichlet` and the catalogue `main_phi_dirichlet` (P06N owner values from `--oracle-root`
  exactly as 5.3; kinds checked against the pinned `catalogue_kinds`).
* **RHS check** (prescribed-phi arm, `phi = phi_bar`; same `perpendicular_rhs` call and parameters as the JAX stage):
  per (variant, Sz, field, term in `poisson_bracket`, `curvature`, `perpendicular_diffusion`, `total`) the metric
  `max|sharded - single| / max|single|` over all owners (back in the global owner order). Gate `<= 1e-12`; `bitwise` records
  exact equality. Non-finite values or a nonzero difference against an all-zero term fail.
* **phi check**: the psi Dirichlet solve of each variant, cold start, `phi_solve_rtol = 1e-11`: single-device `solve_phi` vs
  `sharded_solve_phi`. Gate: both converged, iteration counts differ by `<= 1`, `||psi_sharded - psi_single||_M <= 10 rtol
  ||psi_single||_M` (`M` = owner volume). The right-hand side is the reference stage's `O(psi)` for the transverse variant (the
  JAX stage's solved arm); the catalogue variant has no reference in this campaign, so it uses the discrete `A psi_bar + B g`
  (`rhs_source` in the record; single and sharded solve the same system).
* **Recorded per Sz and variant**: sharded-plan / solver lowering seconds, first-call (compile) and second-call wall seconds
  of the RHS and of the solve, the single-device first / second times, iterations, `dpsi_m`; peak RSS of the worker after the
  single-device part and after each stage of each Sz (`peak_rss_gib`; the sharded plan / solver is freed before the next Sz;
  the N64 single-device plan alone is tens of GiB, the lowering stacks per-shard plans on top, so use the cap if needed).
* **Status**: `pass` iff every gate of every configured shard count passed; `fail` (stops `run`, makes `validate` report
  `grids_pass = False`) if any failed or an entry is missing; `pending` when only the cap prevented completion. Resumable: a
  `pass` / `fail` record of the same identity is not recomputed.

The gates are applied in the parent by `sharding.assess` to the worker's raw metrics (`result.json`).

## Folder layout

```
$OUT/localized_sidecar.json            localized under --input-root; must equal the re-freeze one up to the machine paths
$OUT/provenance/inputs.json            identity, re-freeze record (identity, grids_pass, ...), per grid artifact_dir (in the
                                       re-freeze folder) and artifact_identity_sha256, metadata_only flag, sources, jax
$OUT/preflight.json, preflight/N32_transverse.json
$OUT/N{n}/owner_values.npz + owner_values.json     values (n_owners, 6: n, Te, Ti, omega, phi_switch, phi_wave), sha256
$OUT/N{n}/context.npz                  owner volumes, P06N region masks, u-band masks
$OUT/N{n}/references/{references.npz,manifest.json}, N{n}/jax_stage.json, N{n}/results.npz, N{n}/summary.json
$OUT/N{n}/sharding.json                the sharding record; $OUT/logs/sharding_N{n}.log, $OUT/work/sharding/N{n}/{spec,result}.json
$OUT/work/_chunks/N{n}/{owner_values,references,jax_<variant>}/   stage checkpoints (delete after reduce to save space)
$OUT/validation.json, $OUT/summary/{step6_report.md,step6_summary.json}
$OUT/invocations/, executions/, logs/, last_exit.json, .runner.lock
```

## Identity and resume rules

Identity = sha256 of the configuration (fields, variants, bands, solve settings, stages), the source hashes (this package,
the re-freeze package's sources which include the 5.3, step-4/2/1 and `compact_c3` evaluator sources, and the grid-context
sources), the 17-file input manifest, this machine's localized sidecar, the three P06N `owner_values` oracle entries, the
re-freeze campaign identity and its three artifact identities. A rerun refuses (new folder required) when the identity, the
oracle root or a recorded artifact identity changed. The package refuses a non-CPU backend or missing x64.

## Resources (CPU only; measure on the first run)

References stage: embarrassingly parallel over owner chunks of 1024 (each worker builds its own environment; two field sets
double the 5.3 per-chunk work). JAX stage: one process, the artifact plan plus the solver as in the re-freeze (about 5 /
15 / 36 GiB at N32 / N48 / N64 plus the per-variant arrays); two variants instead of four, so about half of the re-freeze's
JAX time. `owner_values` is cheap (seconds).

## Outputs to return

`summary/`, `validation.json`, `preflight*.json`, `preflight/`, `N*/results.npz`, `N*/summary.json`, `N*/owner_values.*`,
`N*/references/`, `N*/jax_stage.json`, `N*/sharding.json`, `provenance/inputs.json`, `invocations/`, `executions/`, `logs/`.
Leave `work/` out of the return archive.

## Local tests

`tests/test_p08_step6_global_campaign.py` (fast, about 8 s): contract and drift refusals, field definitions (closed forms,
finite differences incl. near the axis, smoothness of `B(u^2)`, positivity), owner averages (bitwise vs the frozen routine),
bands, the re-freeze reading (synthetic folders and the local stripped copy), identity / resume refusals, stage wiring, the
adapter feeding `run_variant`, the sharding stage (shard-count filter, subprocess wiring with a fake worker, gates, resume,
cap, crash, case builder), `validate`, `analyze`. `tests/test_p08_step6_global_campaign_real.py` (slow, skipped without the
local N32 inputs): the preflight at N32 and the sharding stage end to end (real 8-device subprocess) with the substitute worker
`tests/p08_step6_sharding_substitute_worker.py` (bounded N32 closure plan for the RHS part, the local N32 P07 export for
the phi part; about a minute).
