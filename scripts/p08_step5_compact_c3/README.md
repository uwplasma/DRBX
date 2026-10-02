# P08 step 5 re-freeze on the `compact_c3` magnetic evaluator

Computation only (CPU), one node. This is the **one-time re-freeze** of the step-4 artifacts and the step-5.3 references
on the `compact_c3` magnetic evaluator (Q's compact C3 toroidal interpolation of the B field,
`drbx.geometry.Bfield_evaluator`, `toroidal_method="compact_c3"`, `extrapolate=False`; user decision, 2 October 2026).
"C3" alone elsewhere (`inner_support=fixed_radius`, coupled-quartic rows) means the inner donor support; the evaluator is
always `compact_c3`.

ONE campaign, per grid 32 -> 48 -> 64, all inside one output folder `$OUT`:

1. **artifact** -- the full-grid P row artifact (`p_shared.build_artifact.run_full_build`, step 1's chunk sizes) built with
   the four operator options below; the build identity must record all four.
2. **references -> jax -> reduce** -- the step-5.3 static Dirichlet-phi combined MMS stages (`p08_step5_combined`) on that
   artifact. See `scripts/p08_step5_combined/README.md` for the mathematics (two arms per frozen P06N variant, solver
   gates, reduction, regions); nothing in the scientific contract changed except the magnetic evaluator.

Nothing is accepted or rejected by the campaign: solver gates (discrete consistency, convergence) and finiteness are
recorded pass/fail, the headline order criterion is informational, and **the user decides acceptance**.

## Scientific contract (`configuration.json`; also checked against literals in `campaign.py`, hashed into the identity)

| item | value |
|---|---|
| schema | `drbx.p08-step5-compact-c3-v1` |
| resolutions | 32, 48, 64 |
| operator options | `curvature=autodiff`, `face_quadrature=q2`, `inner_support=fixed_radius`, `bfield_toroidal=compact_c3` (a drifted configuration is refused; the module passes the four keys directly to the lower-level functions) |
| parameters | `rho_star = 0.05`, `tau = 1`, `D_f = 1e-2` for n, Te, Ti, omega |
| phi solve | Dirichlet, block-Jacobi eta-plane preconditioner, float32 factors, restart 50, max_restarts 40, **rtol 1e-11** (measurement setting; production default 1e-8) |
| catalogue | the four frozen P06N Dirichlet-phi variants: `main_phi_dirichlet`, `heldout_phi_dirichlet`, `dirichlet_rich`, `control_constant_dirichlet` |
| inherited (read from `p08_step4_global.campaign.config()`, hashed into the identity) | artifact chunk sizes (cell 4096, face 2048, p07 2048, geometry raw 4096 / face 4096), `campaigns`, closure-preflight blocking (`column_block` 8, `p06n_variant_block` 2, `boundary_batch` 32768, `wall_cache` false, floor seeds 0, 1) |

The scientific part of `configuration.json` is a copy of `scripts/p08_step5_combined/configuration.json` (a test compares
them key by key); the other differences are the schema, the fourth option, the `re_freeze` note and the
`inherited_from_step4` key list. `p08_step4_global`, `p08_step5_combined`, `p08_step5_export`, `p08_step1_global`,
`p08_step2_global`, `src/drbx` and `hsx_mms_continuum_reference.py` are reused unchanged; their configurations pin the
three spline-era options and still do.

## Folder layout

`$OUT` doubles as the "step-4 folder" of the 5.3 stage functions (`output = step4 = $OUT`):

```
$OUT/localized_sidecar.json            canonical continuum sidecar localized under --input-root
$OUT/artifact/N{n}/                    build output (build_identity.json, build_receipt.json, manifest.json, rows/, geometry.npz, ...)
$OUT/artifact/_chunks/                 build unit checkpoints (removed by the build once assembled)
$OUT/preflight.json                    closure preflight (N32, before the build)         -> preflight/N32_owner_closure.json
$OUT/preflight_artifact.json           N32 owner-subset gate on the built artifact       -> preflight/N32_artifact_subset.json
$OUT/N{n}/context.npz, N{n}/references/{references.npz,manifest.json}
$OUT/N{n}/jax_stage.json, N{n}/results.npz, N{n}/summary.json
$OUT/work/_chunks/N{n}/{references,jax_<variant>}/   stage checkpoints
$OUT/validation.json, $OUT/summary/{artifacts.json,step5_combined_report.md,step5_combined_summary.json}
$OUT/provenance/inputs.json, invocations/, executions/, logs/, last_exit.json, .runner.lock
```

## Identity and resume rules

Identity = sha256 of the configuration, the inherited step-4 settings, the source hashes (this package, the 5.3
`SOURCE_FILES`, the step-4/2/1 `SOURCE_FILES` that pin the artifact build and the option modules,
`p_shared/{build_artifact,bfield,provider}.py`, `drbx/geometry/{Bfield_evaluator,compact_toroidal,jax_bfield_evaluator}.py`,
the step-4/2/1 campaign and configuration files), the 17-file input manifest, the localized-sidecar sha256 and the three
P06N `owner_values` oracle entries. `provenance/inputs.json` holds, per built grid, `artifact_identity_sha256` (the digest
of `build_identity.json`, which `jaxstage` checks) and is refreshed after each build. A rerun refuses (new folder
required) when the identity, the oracle root, the localized sidecar or an already-recorded artifact identity changed; an
assembled artifact whose build identity lacks `bfield_toroidal == compact_c3` (or any other pinned option) is refused. One
writer per folder (`runner.lock`, `.runner.lock`). The package refuses a non-CPU backend or missing x64 and
`P_SHARED_CSR_ONLY`.

## Commands (from `DRBX/scripts`, never from inside a package directory)

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
A=(--input-root "$INPUT_ROOT" --oracle-root "$ORACLE_ROOT" --output "$OUT")

python -m p08_step5_compact_c3.campaign verify-inputs "${A[@]}"
python -m p08_step5_compact_c3.campaign preflight     "${A[@]}"                       # N32, bounded, no build; must pass before run
python -m p08_step5_compact_c3.campaign run           "${A[@]}" --resolutions 32 48 64 --workers 96
python -m p08_step5_compact_c3.campaign validate      "${A[@]}" --resolutions 32 48 64
python -m p08_step5_compact_c3.campaign analyze       "${A[@]}" --resolutions 32 48 64
# recovery, stage by stage (not preflight-gated; run-stage artifact --n 32 also runs the N32 gate)
python -m p08_step5_compact_c3.campaign run-stage     "${A[@]}" --stage artifact   --n 64 --workers 96
python -m p08_step5_compact_c3.campaign run-stage     "${A[@]}" --stage references --n 64 --workers 96
python -m p08_step5_compact_c3.campaign run-stage     "${A[@]}" --stage jax        --n 64
python -m p08_step5_compact_c3.campaign run-stage     "${A[@]}" --stage reduce     --n 64
```

`--workers` is the process-pool size of the artifact build and the reference stage (the JAX stage is one process);
`--memory-budget-gib B --worker-memory-gib W --memory-reserve-gib R` cap the pool as in steps 1-4
(`min(workers, (B - R) // W)`); `--max-tasks-per-worker` recycles workers. Defaults: `--resolutions 32 48 64`, `--workers 4`.

* **`preflight`** (N32, bounded, no full build; minutes): builds the environment with the four options and records that
  `env.ref.provenance["bfield_toroidal"]`, the reference evaluator's `toroidal_method` and `env.bfield_toroidal` are all
  `compact_c3`; then `p_shared.jax_replay.run_jax_owner_closure_check` with the four options on the 12 owners of
  `owner_closure.select_owners`. Pass = every host-vs-JAX policy row passes (29 rows), no `uniq_mismatches`, the closure
  payload records the four options, and the evaluator check holds; the `compare_to_oracle` rows are informational (the
  frozen oracles describe the old spline operator). It also hash-checks the N32 oracle files of all seven campaigns
  (committed step-1 `oracle_manifest.json`) under `--oracle-root`. `run` refuses without a matching passing preflight; a
  failed preflight is recomputed on the next invocation.
* **`run`**: per grid 32 -> 48 -> 64: artifact stage (skipped when assembled; `check_artifact_options` must show the four
  pinned options), **at N32 only** the 5.3-style bounded owner-subset gate on the freshly built artifact
  (`references.reference_chunk` on the 12 closure owners; the JAX stage on the FULL N32 plan of the built artifact for
  `main_phi_dirichlet`; verdict = `p08_step5_combined.campaign.preflight_checks`: finite, solver gates, N - O of psi
  <= 0.1, prescribed-arm sanity <= 0.5, arm difference <= 1e-5), then references -> jax -> reduce, each resumable. It stops
  after the first grid whose gate/`grid_pass` fails (data are checkpointed), runs `validate` and `analyze` on the grids
  completed and exits nonzero (`last_exit.json` status `failed_gates`).
* **`validate`**: requires the closure preflight, the N32 gate (when N32 is requested), the artifact receipts with the
  pinned options and the reduction of every requested grid (same identity and artifact); writes `validation.json`
  (`solver_gates_pass`, `all_finite`, `grids_pass`, the informational headline order criterion, per-grid artifact
  records: policy, row counts and bytes, `coupled_quartic`, build wall time and peak worker RSS) and `summary/artifacts.json`
  (absolute artifact paths for later steps).
* **`analyze`**: the 5.3 analysis (observed orders between grids, solver table, potential error, psi diffusion triple) into
  `summary/step5_combined_report.md` / `step5_combined_summary.json`.

## Inputs

* `--input-root DIR`: the immutable HSX input root holding the 17 files of `p08_step1_global/input_manifest.json` at
  their relative paths (geometry artifacts, the metric cache, the ~5.8 GB `mgrid_res2p5cm_180pln.nc` MAKEGRID file, the
  continuum sidecar). Existence, size and sha256 of every file are checked on every invocation (like step 4). Example:
  the remote `/pscratch/sd/y/yiqunx/hsx-midpoint-inputs-2458dbf6-8z2jrqq8`.
* `--oracle-root DIR` (default `--input-root`): where the files of the committed step-1 `oracle_manifest.json` live. `run`
  reads and hash-checks only the P06N `owner_values` files of the requested grids (the frozen owner averages the arms are
  fed with); `preflight` additionally hash-checks the N32 files of all seven campaigns. Example:
  `/pscratch/sd/y/yiqunx/p08-step1-oracles-cb728443`.
* No step-4 folder is needed: this campaign builds its own artifacts. Use a fresh `--output`.

## Resources (CPU only)

Numbers from the step-4 and step-5.3 documentation (spline evaluator; `compact_c3` changes the B-field evaluation cost,
not the memory layout); measure on the first run.

| | N32 | N48 | N64 |
|---|---|---|---|
| artifact build (96 workers) | 620 s | 1160 s | ~2800 s plus the coupled-quartic row work |
| artifact on disk | 2.3 GB | 5.8 GB | ~14 GB (q2 rows are smaller than the q3 estimate) |
| JAX stage peak (one process) | ~5 GiB | ~15 GiB | ~36 GiB from step 4's replay estimate, plus the sparse P07 operator, the plane-preconditioner factors and the per-variant arrays |

* The build's disk guard needs `10 GiB * (n/32)^3` plus 8 GiB free (N32 18 GiB, N48 ~43 GiB, N64 88 GiB); the artifacts
  of the three grids total ~25 GB; run on `$PSCRATCH`. Free space is checked before each build.
* References stage: embarrassingly parallel over owner chunks of 1024 (each worker builds its own environment, a few GiB at
  N32, more at N64); use `--memory-budget-gib`/`--worker-memory-gib` to cap the pool.
* JAX stage: one process; the phi solves (two per variant, up to 50 x 40 GMRES iterations at rtol 1e-11) dominate. JAX chunk
  checkpoints are ~100 MB per variant at N64: delete `work/_chunks` after `reduce` to save space, keep it to redo the
  reduction.
* Order of magnitude for the whole `run`: step 4's build total (~4600 s) plus the 5.3 stages; request a multi-hour
  allocation. Node-local, one writer per folder.

## Outputs to return

`summary/`, `validation.json`, `preflight*.json`, `preflight/`, `N*/results.npz`, `N*/summary.json`, `N*/references/`,
`N*/jax_stage.json`, `provenance/inputs.json`, `campaign` logs/invocations. **Keep** `artifact/N{n}` on the remote (absolute
paths are in `summary/artifacts.json`); leave `artifact/N*/rows`, `artifact/N*/geometry.npz` and `work/` out of the return
archive.

## Out of scope

* The step-4 smoke replay of the full-grid JAX operators against the frozen step-1 oracles
  (`p08_step2_global.replay`'s full replay rejects four-key options; those oracles describe the old operator anyway).
  Finiteness of the full-grid RHS is covered by the jax-stage gates (every array finite) and by the N32 gate.
* The Neumann-phi variants (deferred, recorded in `configuration.json`) and the step-6 transverse-wave controls.

## Local tests

`tests/test_p08_step5_compact_c3_campaign.py` (fast, ~4 s): pinned contract and drift refusal, the 5.3 contract copy, the
inherited settings, identity and resume refusals (identity, oracle root, sidecar, artifact identity, artifact options),
the four-key options reaching `run_full_build` / `build_environment` / the closure check / every stage call, both
preflights, the run flow, `validate`, `analyze`. `tests/test_p08_step5_compact_c3_campaign_real.py` (slow, skipped when the
local N32 inputs are missing): the closure preflight at N32 with the four options.
