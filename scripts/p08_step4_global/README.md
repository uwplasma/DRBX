# P08 step 4: full-grid artifact build and smoke check of the final bundle operator

## What this computes

Roadmap: `src/drbx/dev_docs/perpendicular_second_order_roadmap.md`, P08 execution plan item 4.
Per grid (N32, N48, N64) this campaign builds the **final-operator row artifact** on the full grid and checks that
the full-grid JAX operators run on it and produce finite output. The artifacts **stay on the remote for step 5**.

The final operator options are pinned explicitly -- in `configuration.json` (`operator_options`, also checked against
the literals in `campaign.OPERATOR_OPTIONS`, hashed into the campaign identity) and passed to every build,
environment and closure call; no code default is relied on:

| option | value | meaning |
|---|---|---|
| `curvature` | `"autodiff"` | curvature `K` of the geometry arrays and references by `jax` autodiff |
| `face_quadrature` | `"q2"` | P05/P06 face rule at 2x2 Gauss (4 nodes per face); P07 stays q3 |
| `inner_support` | `"fixed_radius"` | C3 inner donor support (coupled quartic fit on a fixed radius) |

**There is no MMS acceptance gate.** The frozen step-1 oracles describe the old fd / q3 / profile7 operator, so the
comparison with them is *informational*: the Tier-B ratio is then the operator change relative to the archived
spatial error of the old operator, not an error. The gates are:

1. **Preflight** (per grid, minutes): `p_shared.jax_replay.run_jax_owner_closure_check` with the final options on the
   bounded 12-owner closure. Pass = every host-vs-JAX policy row passes (`all_diff_pass`, the 29 rows of design
   section 8), no `uniq_mismatches`, and the closure payload records the pinned options. The `compare_to_oracle`
   rows (`oracle_jax`) are recorded under `oracle_informational` (rows not matching the frozen oracles, max
   `ratio_to_oracle_NR`) and **do not gate**. The CPU backend and x64 are required. `run` refuses without a
   matching (same identity, every requested grid present) all-pass preflight.
2. **Smoke** (per grid, in the replay stage): every floating-point term array of the merged host-format output of the
   full-grid JAX replay -- the sparse owner numerators of all seven campaign keys and the per-face P05 jump values --
   is finite (no NaN / Inf) on every owner and face, and every campaign key produced output.

Per grid, `run` does:

1. **artifact** -- `p_shared.build_artifact.run_full_build` (unit-parallel, resumable, schema
   `drbx.p-row-artifact.v3` with tensor encoding) with the pinned options and step 1's chunk sizes. It refuses
   `P_SHARED_CSR_ONLY`; the build identity must record the options (`build_policy`: `curvature`,
   `quadrature = {raw: q1, face: q2, p07_face: q3}`, `inner_support`).
2. **replay / smoke** (`smoke.run_smoke_stage`) -- the step-2 JAX replay
   (`p08_step2_global.replay.run_replay_stage(..., operator_options=<pinned>, compare=False, return_out=True)`):
   plan streamed from the artifact, `jax.device_put`, the seven campaign keys sequentially in column blocks with
   per-campaign checkpoints (`replay/N{n}/_chunks`); then the finiteness check, then the **informational**
   `comparison.compare_operator_terms` against the frozen oracles with the final-options environment (an exception
   is recorded in `smoke.json`, not raised), then the statistics.

Outputs per grid, `replay/N{n}/`:

* `smoke.json` -- `smoke_pass`; `finite` (per campaign and term: array/element counts, NaN/Inf counts, max |value|);
  `artifact` (path, bytes per row kind, row bytes, `geometry.npz` bytes and tree total, row counts, point-row
  family counts including `coupled_quartic`, P07 integrated families, tensor-encoding counts and fallbacks, max
  residuals / Neumann condition, build wall / CPU seconds, peak worker RSS); `replay` (wall seconds, plan counts and
  `plan_bytes`, seconds and peak RSS per phase and per campaign, JAX facts); `change_vs_frozen_oracles`
  (`informational: true, gate: false`; per term the Tier-B ratio = operator change relative to the archived spatial
  error, its region, the max abs difference, the pointwise-cap violation count, and the nominal pass flag of the
  step-2 comparison, which is **not** a verdict here);
* `report.md` (human summary of the above), `change_vs_frozen_oracles.json` (the full comparison, when it ran),
  `plan_info.json`, `smoke_receipt.json` (identity + sha256 of `smoke.json`; a finished stage is skipped on resume).

`validate` requires the preflight pass and `smoke.json` (this identity, the pinned options, all seven campaigns) for
every requested grid, and writes `validation.json` (`operational_complete`, `smoke_pass` = preflight pass and all
finite, per grid `smoke_pass`, artifact bytes, family counts, the informational change summary) and
`summary/step4_summary.json` (per grid: absolute artifact path, total bytes, bytes per row kind, `coupled_quartic`
count, build / replay wall seconds and peak RSS, plan bytes).

## Inputs

Identical to steps 1 and 2 (nothing new to stage):

* `INPUT_ROOT` -- the immutable 17-file input root (`input_manifest.json` is step 1's), e.g. the remote
  `/pscratch/sd/y/yiqunx/hsx-midpoint-inputs-2458dbf6-8z2jrqq8`.
* `ORACLE_ROOT` -- the step-1 oracles, e.g. `/pscratch/sd/y/yiqunx/p08-step1-oracles-cb728443`. They supply the
  campaign owner values the operators are applied to (the saved `owner_values`) and the informational comparison.
  `verify-inputs` hash-checks every oracle file against step 1's committed `oracle_manifest.json` (reused; there is
  no `pack-oracles` command).
* The same conda environment and checkout as step 2 (`drbx` importable; JAX CPU, x64). The campaign identity includes
  the git commit, every source file in `campaign.SOURCE_FILES` (step 2's list, this package, and the option modules
  `p_shared/{curvature_reference,face_quadrature,inner_support}.py`, `drbx/geometry/curvature_autodiff.py`, the
  reconstruction / integrated-row modules, `stencils/{geometry_arrays,builder}.py`), the merged configuration
  (including `operator_options`) and both manifests; a folder written under another identity is refused.
  Use a fresh `--output` for this campaign (step 2 folders are not reusable).

## Commands (from `DRBX/scripts`, never from inside a package directory)

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
A=(--input-root "$INPUT_ROOT" --oracle-root "$ORACLE_ROOT" --output "$OUT")

python -m p08_step4_global.campaign verify-inputs "${A[@]}"
python -m p08_step4_global.campaign preflight     "${A[@]}" --resolutions 32 48 64
python -m p08_step4_global.campaign run           "${A[@]}" --resolutions 32 48 64 --workers 96
python -m p08_step4_global.campaign validate      "${A[@]}" --resolutions 32 48 64
```

`--workers` is the artifact-build pool size only (the replay is one process); step 2 measured with 96 build workers.
`--memory-budget-gib B --worker-memory-gib W --memory-reserve-gib R` cap the pool as in step 1
(`min(workers, (B - R) // W)`); `--max-tasks-per-worker` recycles workers. The default resolutions are `32 48 64`.

Every command is resumable and identity-checked: rerun the same command after an interruption. Completed build units,
the assembled artifact, valid per-campaign checkpoints and a finished smoke stage (`smoke_receipt.json`) are skipped;
a checkpoint or receipt of another identity, or a corrupt one, raises instead of being recomputed (use a new
`--output`). `run` always ends with `validate`. Recovery, stage by stage (`run-stage` is not preflight-gated):

```bash
python -m p08_step4_global.campaign run-stage "${A[@]}" --n 64 --stage artifact --workers 96
python -m p08_step4_global.campaign run-stage "${A[@]}" --n 64 --stage replay
```

Each invocation appends `invocations/<ns>_<command>.json` and writes `last_exit.json` (`status` complete / failed
with the error); progress lines with timestamps (one per boundary-data slice) go to stdout -- tee them into `logs/`.
Environment defaults set by the package: `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=true`, `CUDA_VISIBLE_DEVICES=""`,
thread variables `=1`; it refuses a non-CPU backend or missing x64.

## Resources (CPU only)

One node, node-local (a single process pool for the build, one process for the replay); CPU compute only. The
allocation type, worker count and memory caps are chosen by the remote setup from the CPUs and host memory actually
available (pass `--workers` and, where useful, `--memory-budget-gib/--worker-memory-gib/--memory-reserve-gib`).

Estimates from step 2's measured run (96 build workers; the final options change the numbers):

| | N32 | N48 | N64 (scaled ~2.4x from N48) |
|---|---|---|---|
| artifact build | 620 s | 1160 s | ~2800 s, plus the extra coupled-quartic (C3) row work |
| JAX replay | 332 s | 957 s | ~2300 s |
| replay peak RSS | 5.2 GiB | 14.9 GiB | ~36 GiB |
| artifact on disk | 2.3 GB | 5.8 GB | ~14 GB |

q2 builds 4-node P05/P06 faces but the rows shrink about 2.25x for the face rows relative to q3, so artifact sizes
should come in below the scaled values; the `fixed_radius` support adds coupled-quartic point rows (count in
`smoke.json` `artifact.coupled_quartic`). Whole `run` for the three grids: about 2.5-3.5 h (build ~4600 s + replay
~3600 s + preflights and per-grid environment / comparison); request a 4-5 h allocation. Memory is far below the
node's 512 GB. The preflight takes minutes per grid (the N32 local figures are in the task report).

Disk: the build's own guard needs `10 GiB * (n/32)^3` plus 8 GiB free (N32 18 GiB, N48 ~43 GiB, N64 88 GiB free); the
row units are moved, not copied, into `rows/`. The three artifacts together are ~25 GB; replay checkpoints are ~0.2
GB per grid at N32 (about 0.5 GB at N64). Free space is checked before the build, so run on `$PSCRATCH`.

## Outputs and the return archive

Under `$OUT`: `campaign_manifest.json`, `oracle_manifest.json`, `localized_sidecar.json`, `preflight.json` and
`preflight/N{n}_owner_closure.json`, `artifact/N{n}/{build_receipt.json,build_identity.json,census.npz,manifest.json,
plan.json,geometry.npz,rows/}`, `replay/N{n}/{smoke.json,report.md,change_vs_frozen_oracles.json,plan_info.json,
smoke_receipt.json,_chunks/}`, `validation.json`, `summary/step4_summary.json`, `invocations/`, `executions/`,
`logs/`, `last_exit.json`.

The artifacts stay on the remote for step 5: record the absolute paths printed in `summary/step4_summary.json`
(`grids.<n>.artifact_path`, i.e. `$OUT/artifact/N{n}`) and do not delete or move them. The return archive excludes
the artifact rows, geometry and `_chunks`:

```bash
tar -I "pigz -1 -p 8" -cf "$ARCHIVE" -C "$(dirname "$RUN")" \
  --exclude="$(basename "$RUN")/campaign/artifact/_chunks" \
  --exclude="$(basename "$RUN")/campaign/artifact/N*/rows" \
  --exclude="$(basename "$RUN")/campaign/artifact/N*/geometry.npz" \
  --exclude="$(basename "$RUN")/campaign/replay/N*/_chunks" \
  --exclude="$(basename "$RUN")/cache" --exclude="$(basename "$RUN")/scratch" \
  "$(basename "$RUN")"
```

(Keeping `replay/N*/_chunks` lets the informational comparison be redone locally without a replay.) The campaign
returns the smoke and change data unchanged; the decision on step 5 is the user's.

## Local tests

`tests/test_p08_step4_campaign.py` (fast): pinned options in the configuration and identity, CLI, preflight gating
(oracle rows never gate), build / replay wiring, smoke finiteness (NaN / Inf, missing campaign), the informational
comparison (exceptions recorded), the smoke stage outputs and resume, `validate`, and that step 2's
`run_replay_stage` default options are unchanged. `tests/test_p08_step4_campaign_real.py` (slow, ~4 min): the
campaign's `preflight_grid` and the replay-stage path on a real tensor artifact of the N32 owner closure with the final
options. `tests/test_p08_step2_campaign.py` and `tests/test_p08_step2_campaign_real.py` still pass unchanged.
