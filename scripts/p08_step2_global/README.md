# P08 step 2b, G3: full-grid JAX replay against the frozen oracles

## What this computes

The step-1 campaign (`scripts/p08_step1_global`) replays the six frozen perpendicular
campaigns (P05 bracket, P05N frozen/upwind, P06N, P06-legacy, P07, P07N; seven campaign
keys) with the **host** code, unit-parallel, from a row artifact. This campaign is gate
**G3** of the step-2b design (`work/p08_step2b_operator_assembly_design_20260929/design.md`
section 3): the same replay with the **JAX operators** of `drbx.native`
(`fci_perpendicular_p05_operator`, `..._p06_operator`, `..._p07_operator`), applied from the
step-2a plan (`drbx.stencils.operator_plan`) that is streamed from a **freshly built
schema-v3 (tensor-factored) artifact**, in one process on the full grid, and compared with
the same frozen oracles at N32 and N48 (N64 is allowed, not default).

Per grid, `run` does:

1. **artifact** -- the step-1 unit-parallel build (`p_shared.build_artifact.run_full_build`
   through `p08_step1_global.campaign._build_artifact`), which now writes
   `drbx.p-row-artifact.v3` with tensor encoding by default. The build receipt records the
   tensor/fallback source counts (`diagnostics.tensor_encoding`); the replay refuses an
   artifact with no tensor-encoded source (e.g. built with `P_SHARED_CSR_ONLY` set).
2. **JAX replay stage** (`replay.run_replay_stage`):
   * `lower_perpendicular_plan_from_artifact` streams the plan chunk file by chunk file (tensor
     sources are never expanded), then one `jax.device_put` of the plan (the plan is a jit
     *argument* everywhere; nothing is baked in as a constant);
   * the seven campaign keys are evaluated **sequentially**; for each, the campaign's boundary
     data (MMS Dirichlet trace / physical-normal derivative, the `p_shared.campaign_fields`
     adapters) is evaluated once at the plan's full point tables (`dirichlet_points`,
     `neumann_points`), in slices of `boundary_batch` = 32768 points (a multiple of the frozen
     metric evaluator's 4096 internal batch), and the operators run in column blocks
     (`column_block` = 8 fields per call; P06N: at most `p06n_variant_block` = 2 variants per
     call; P05/P05N: blocks of bracket pairs). Fields are contracted column by column, so blocked
     and unblocked results agree to roundoff (`tests/test_p08_step2_campaign.py`). Each campaign's
     owner numerators are checkpointed (identity + sha256 receipt, `runner.write_unit`) and
     its intermediates freed before the next;
   * `comparison.compare_operator_terms` applies the step-1 reduction: the same accumulators,
     divisions (`t.vol`; `max(evolution_volume, 1e-300)` for P06 terms; P06N corrections divided
     once), saved oracle arrays, region masks and `compare_owner_term` /
     `compare_pointwise_only` calls (Tier-B ratio, corrected pointwise cap with the P05
     `cap_reference`), including P05's `live_jump_vs_upwind` pointwise check (per-face jump values
     keyed by p07 id against the saved upwind chunks). `tests/test_p08_step2_campaign_real.py`
     checks this function against `replay_units.reduce_grid` term by term (42 terms, bitwise).
   * Boundary data is evaluated **live** (`wall_cache: false`). E6 measured that routing through
     `WallDataCache` does not reproduce the host any better (the only change in the N32 table was
     P06N `raw_total`, 2.74e-13 -> 2.77e-13 absolute).
3. `replay/N{n}/replay.json` + `report.md` in the step-1 schema/format (`write_report`; the
   report header states the scope), `receipts.json` (timings per phase and campaign, peak RSS
   per phase via `p_shared.runner.peak_rss_gib`, plan bytes and counts, artifact bytes per group,
   tensor-encoding counts/fallbacks, JAX backend/x64/versions), `stage_receipt.json`.

**Scope: operator terms only.** The MMS *reference* terms -- P05N `raw_R` (frozen, upwind), P06N
`raw_R_material/remainder/total`, P07N `global_O_q3` -- are host-only (they are not operators;
they need the analytic fields at every raw cell / face node) and were already gated in step 1;
they are omitted from G3 (`comparison.OMITTED_TERMS`, recorded in `replay.json` `scope`, the
report header and `validation.json`). Everything else step 1 compared is compared here: 3 P05
terms, 4 terms per P05N catalogue, 4 P06N terms (raw material/remainder/total + faces
correction, 14 variants each), 24 P06-legacy terms (4 fields x centered/U x 3), P07
`global_N`, P07N `global_N` and `global_D` -- 42 terms.

`preflight` (per grid, minutes) is the design-section-8 gate: `p_shared.jax_replay
.run_jax_owner_closure_check` on the bounded 12-owner closure (host + JAX assembly through the
same blocked code path, the uniform tolerance policy with measured one-ulp conditioning floors for
the cancellation terms, and `compare_to_oracle` with the JAX terms; 29 policy rows, 143 oracle
rows). It also records and requires the JAX CPU backend and x64. `run` refuses without a
matching (same identity, every requested grid present) **all-pass** preflight.

## Inputs

Identical to step 1 (nothing new to stage):

* `INPUT_ROOT` -- the immutable 17-file input root (`input_manifest.json` is step 1's, itself
  byte-identical to `p07n_field_derived_global/input_manifest.json`): e.g. the remote
  `/pscratch/sd/y/yiqunx/hsx-midpoint-inputs-2458dbf6-8z2jrqq8`.
* `ORACLE_ROOT` -- the frozen oracles already extracted for step 1 (e.g.
  `/pscratch/sd/y/yiqunx/p08-step1-oracles-cb728443`). `verify-inputs` hash-checks every oracle
  file against step 1's committed `oracle_manifest.json` (this package reuses that file; there is no
  `pack-oracles` command and no second manifest).
* The same conda environment and checkout as step 1 (`drbx` importable; JAX CPU, x64). The campaign
  identity includes the git commit, every source file it depends on (`campaign.SOURCE_FILES`), the
  merged configuration and both manifests; a folder written under another identity is refused.

## Commands (from `DRBX/scripts`, never from inside a package directory)

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
A=(--input-root "$INPUT_ROOT" --oracle-root "$ORACLE_ROOT" --output "$OUT")

python -m p08_step2_global.campaign verify-inputs "${A[@]}"
python -m p08_step2_global.campaign preflight     "${A[@]}" --resolutions 32 48
python -m p08_step2_global.campaign run           "${A[@]}" --resolutions 32 48 --workers 256
python -m p08_step2_global.campaign validate      "${A[@]}" --resolutions 32 48
```

`--workers` is the artifact-build pool size only (the replay is one process). The resource flags
`--memory-budget-gib B --worker-memory-gib W --memory-reserve-gib R` cap the build pool exactly as
in step 1 (`min(workers, (B - R) // W)`); step 1 ran `--workers 256` (all CPUs of a 512 GB node,
no cap; peak worker RSS 9.2 GiB at N48). `--max-units` is for local testing only (a bounded build;
`run` then stops before the replay).

Every command is resumable and identity-checked; rerun the same command after an interruption:
completed build units, the assembled artifact, valid per-campaign checkpoints and a finished replay
stage (`stage_receipt.json`) are skipped (a checkpoint or receipt of another identity, or a corrupt one,
raises instead of being recomputed -- use a new `--output`). `run` writes `validation.json` when
every requested grid is complete. Recovery, stage by stage:

```bash
python -m p08_step2_global.campaign run-stage "${A[@]}" --n 48 --stage artifact --workers 256
python -m p08_step2_global.campaign run-stage "${A[@]}" --n 48 --stage replay
```

(`run-stage` is not preflight-gated.) Each invocation appends `invocations/<ns>_<command>.json`
and writes `last_exit.json` (`status` complete / failed with the error); progress lines with
timestamps (and one line per boundary-data slice) go to stdout -- tee them into `logs/`.

## Resources (CPU only)

* One Perlmutter CPU node, exclusive, all cores (as step 1: `--constraint=cpu --nodes=1
  --exclusive --cpus-per-task=256`, a 4 h allocation is comfortable for N32 + N48).
* Environment: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=true` (the campaign sets them by default and refuses a
  non-CPU backend or missing x64). Thread variables as above (`OMP_NUM_THREADS=1` etc.: the
  build workers are single-threaded). The replay stage is one process: XLA's own CPU thread pool
  (all cores by default) serves the dots/elementwise work; the P06 q3 characteristic solve
  (batched LAPACK `geev`) and the host MMS callbacks are single-threaded. **No XLA flags are needed**;
  on a shared node cap the pool with `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false
  intra_op_parallelism_threads=N"` or `taskset`. Point `JAX_COMPILATION_CACHE_DIR` at scratch if you
  want compile results kept across restarts (compile time is seconds per operator shape).
* Disk: the v3 artifacts are far smaller than step 1's 9 / 30 GB of rows. From the step-2 factoring
  report (point rows 1.0 GB at N32 with tensor-factored families, Neumann 0.86 GB, P07 0.32 GB) and
  step 1's measured N48/N32 row-byte ratios: **~2.5 GB at N32 and ~8 GB at N48** including
  `geometry.npz` (0.16 / 0.55 GB); N64 would be ~20 GB. The real `build_receipt.json`
  (`bytes_per_row_kind`) gives the exact numbers. The build's disk guard is deliberately conservative
  (`10 GiB * (n/32)^3` plus 8 GiB must be free: **~42 GiB free at N48**); the row units are moved, not
  copied, into `rows/`, so there is no doubling. Replay checkpoints are ~0.2 GB per grid.
* Memory (estimate; the measured peak RSS per phase is in `receipts.json`): the plan is the
  artifact's size in host arrays (~8 GB at N48) plus a transient second copy during `device_put`, the
  geometry (0.55 GB) and environment (~1-2 GB); operator blocks add gathers `(sources, width, F<=8)` of
  the CSR (untensored) buckets (~1 GB at N48), the P05 face state `(Fc, 9, 3, 8)` (0.6 GB each array,
  Fc = 331,776) and, for P06 with 2 variants per call, the q3 eigen-solve arrays
  `(2, Fc, 9, 4, 4)` (0.8 GB each, complex 1.5 GB). Plan for **<= 40 GB at N48** (~15 GB at N32); it
  runs comfortably in the 512 GB node's memory.
* Time (estimates; the campaign logs its own): artifact build as step 1's measured wall on 256
  CPUs (**N32 109 s, N48 1208 s**; the v3 tensor verification adds work that was not measured at
  scale -- budget +50%); plan lowering minutes; the P06 q3 solve dominates the JAX side (measured
  ~225 us per face per variant on one core on a synthetic plan: 18 variant-evaluations = 14 P06N + 4
  legacy -> ~7 min at N32, ~22 min at N48); the host MMS callbacks cost ~1-3 us/point (Dirichlet) and
  ~50 us/point (normal) except **P05's Dirichlet trace at ~6 ms/point** (`boundary_trace` with
  `omega_gradient=True`), so P05's boundary data is `dirichlet_points * 6 ms` (a few 10^5 points ->
  tens of minutes; the log prints every 32768-point slice); the comparison is ~10-20 s (step 1's
  reduction: 7 s / 17 s). Expect **~30 min (N32) and ~1-1.5 h (N48) for the replay stage**, ~2.5 h
  for the whole `run`.

## Outputs

Under `$OUT`: `campaign_manifest.json`, `oracle_manifest.json`, `localized_sidecar.json`,
`preflight.json` and `preflight/N{n}_owner_closure.json` (the full closure report),
`artifact/N{n}/{build_receipt.json,build_identity.json,census.npz,manifest.json,plan.json,
geometry.npz,rows/}`, `replay/N{n}/{replay.json,report.md,receipts.json,stage_receipt.json,_chunks/}`,
`validation.json` (per-campaign `status`, `pass`; all grids requested present; scope), `invocations/`,
`executions/`, `logs/`, `last_exit.json`.

Return archive -- exclude the artifact rows, geometry, `_chunks` and caches (as step 1):

```bash
tar -I "pigz -1 -p 8" -cf "$ARCHIVE" -C "$(dirname "$RUN")" \
  --exclude="$(basename "$RUN")/campaign/artifact/_chunks" \
  --exclude="$(basename "$RUN")/campaign/artifact/N*/rows" \
  --exclude="$(basename "$RUN")/campaign/artifact/N*/geometry.npz" \
  --exclude="$(basename "$RUN")/campaign/replay/N*/_chunks" \
  --exclude="$(basename "$RUN")/cache" --exclude="$(basename "$RUN")/scratch" \
  "$(basename "$RUN")"
```

(Keeping `replay/N*/_chunks`, ~0.2 GB per grid, would let the comparison be redone locally without a
replay.) Report whichever terms fail with `replay/N{n}/report.md`; like step 1, the campaign returns
pass/fail data unchanged and the acceptance decision is the user's.

## Local tests

`tests/test_p08_step2_campaign.py` (fast: CLI, identity/resume/refusal, preflight gating, checkpointed
evaluation, report writing, blocked == unblocked operators on the synthetic world) and
`tests/test_p08_step2_campaign_real.py` (slow, ~2.5 min: the replay-stage code path on a real tensor
artifact of the N32 owner closure against the host terms, E6's JAX terms and `compare_to_oracle`; the
campaign's `preflight_grid`; the comparison function against `reduce_grid`).
