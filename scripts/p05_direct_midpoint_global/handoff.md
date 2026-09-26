# Remote computation handoff

## Assignment

Perform the frozen, static P05 direct raw-midpoint bracket accuracy
qualification at N32, N48, and N64 using both candidates and all eight
configured case/pair entries. This is computation and operational validation
only. Do not interpret the scientific results, change the numerical method or
inputs, tune the gate, run another experiment, or promote production code.

Use the remote **run drbx on perlmutter** skill for Perlmutter environment
setup and allocation. Request an allocation charged to the available GPU
budget, but run all numerical work on the allocated node's CPUs. The campaign
requires the JAX CPU backend; the requested GPU resources remain unused. Do
not silently substitute a CPU allocation if GPU allocation/accounting is
unavailable. Have the remote skill choose the partition/constraint, account,
QoS, scheduler GPU request, CPU affinity, worker count, host-memory budget,
worker-memory estimate, reserve, and walltime from the actual GPU node. Do not
hard-code resource counts. Use one node and one writer; this runner is a
node-local spawn-based process pool, not MPI or a distributed pool.

## Pinned source

- Repository: `git@github.com:uwplasma/DRBX.git`
- Branch to fetch: `2D_fci`
- Exact code revision: `0289697312e8cae9754035250946f1bae1dba825`
- Campaign README: `scripts/p05_direct_midpoint_global/README.md`
- Runner: `scripts/p05_direct_midpoint_global/campaign.py`
- Configuration: `scripts/p05_direct_midpoint_global/configuration.json`
- Committed reusable arrays: `scripts/p05_direct_midpoint_global/reuse_bundle/reuse_inputs_v1.zip`

Use a new source directory and verify the detached checkout exactly:

```bash
set -euo pipefail
export SOURCE_ROOT="$(mktemp -d -p /pscratch/sd/y/yiqunx DRBX-p05-direct-midpoint-02896973_XXXXXXXX)"
git clone --no-checkout git@github.com:uwplasma/DRBX.git "$SOURCE_ROOT"
cd "$SOURCE_ROOT"
git fetch origin 2D_fci
git checkout --detach 0289697312e8cae9754035250946f1bae1dba825
test "$(git rev-parse HEAD)" = 0289697312e8cae9754035250946f1bae1dba825
```

Set the immutable P05 input root and create one unique output folder before
input verification. Keep this same output folder for preflight, the global
run, validation, and any checkpoint resume:

```bash
set -euo pipefail
export INPUT_ROOT=/pscratch/sd/y/yiqunx/hsx-midpoint-inputs-2458dbf6-8z2jrqq8
export OUTPUT="$(mktemp -d -p /pscratch/sd/y/yiqunx p05_direct_midpoint_global_02896973_XXXXXXXX)"
: "${OUTPUT:?mktemp failed to create campaign folder}"
mkdir -p "$OUTPUT/logs"
```

The existing input-root path is a location to check, not permission to accept
stale or mismatched files. The runner verifies every size and SHA256 before
execution and refuses any identity mismatch. The nine required input files
are:

| Input-root relative path | Bytes | SHA256 |
|---|---:|---|
| `DRBX/work/perpendicular_second_order_hsx_p01_p03/continuous_reference_sidecar.json` | 9,715 | `8f933512338e18a40e6410e80d8fe950d42462d78421df8e772a6b4d413f5ab5` |
| `geometry_artifacts/rlp_convergence_32_48_64_20260917/32x32x32/base_geometry.npz` | 4,209,909 | `43a5490a319e1028fd8a7c01e4d9478d540e5ff01f8427a447d6c2d93dc1e91c` |
| `geometry_artifacts/rlp_convergence_32_48_64_20260917/32x32x32/rlp_topology.npz` | 12,362,542 | `1ae442292262f711ce194263d9b3b70df7aca9d5404c0de40041cfec2d8f9970` |
| `geometry_artifacts/rlp_convergence_32_48_64_20260917/48x48x48/base_geometry.npz` | 14,401,825 | `fc8aa6f803ec09dd4e0f7d72726534452bae4b6e41d85e61b8ed76f1f2faf09d` |
| `geometry_artifacts/rlp_convergence_32_48_64_20260917/48x48x48/rlp_topology.npz` | 43,481,612 | `02c516b32ca451522eddaf5720b7cafa242efce48a562427d0f46c24948c8e52` |
| `geometry_artifacts/rlp_convergence_32_48_64_20260917/64x64x64/base_geometry.npz` | 33,840,687 | `9cf7c047b38698d0e9fe043674ce29d13305c24579f17a859b604fcce217658d` |
| `geometry_artifacts/rlp_convergence_32_48_64_20260917/64x64x64/rlp_topology.npz` | 104,668,941 | `468b10b36a1ebeaf912edc65ec64c7db6ef99e92931890707c4ce9f741e239ed` |
| `hsx_metric_d58d392545fd3917efeb83b6.npz` | 154,138,698 | `216dbcfb343fa23dd7ac9e60da6592f5757c7f5e45c8d96a5b642c6419dc10a1` |
| `mgrid_res2p5cm_180pln.nc` | 5,831,797,532 | `45dd643a4ae3930386b8f8bff7a789c796a0d5174b5209e1202b5bfd4ac87e44` |

Geometry may instead be resolved from the matching
`prototype_runs/geometry/hsx_fci_NxNxN/` paths under `INPUT_ROOT`; the runner
hashes the canonical files against the same identities. If any input is
missing or mismatched, stop and report which exact file failed and its
observed identity. The reuse archive is already committed with the code and
has SHA256
`17cdce667cfdeb9deebe2c4b7fe6a10958f193ff8992d952b11db89050cd8ea9`.
It contains the saved owner observations, references and volumes, previous
centered action, saved `U - A` jump, and validity-masked old exact-input O
diagnostics; no external P05 checkpoint is required.

## Input verification and preflight

Use the DRBX environment established by the remote skill. Input verification
hashes files and materializes the reusable arrays but does not launch a
numerical campaign. Do this before the compute allocation when the remote
skill permits it:

```bash
cd "$SOURCE_ROOT"
conda run -n drb python scripts/p05_direct_midpoint_global/campaign.py \
  verify-inputs --input-root "$INPUT_ROOT" --output "$OUTPUT" \
  2>&1 | tee "$OUTPUT/logs/verify-inputs.log"
```

Then use the remote skill to request one GPU allocation under the user's GPU
budget for CPU computation. Direct scheduler stdout/stderr to
`$OUTPUT/logs/` using that skill's allocation/launch procedure. Inside the
allocation, set the CPU-only numerical environment before Python starts and
ask the skill to export these four resource values from the measured node:
`DRBX_WORKERS`, `DRBX_MEMORY_BUDGET_GIB`, `DRBX_WORKER_MEMORY_GIB`, and
`DRBX_MEMORY_RESERVE_GIB`. They must be positive and the total worker budget
must leave the stated reserve. The runner further caps effective workers by
the supplied worker and memory limits. BLAS thread counts are fixed to one;
do not change them.

```bash
set -euo pipefail
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
: "${DRBX_WORKERS:?set from actual allocated GPU-node CPUs}"
: "${DRBX_MEMORY_BUDGET_GIB:?set from actual allocated host memory}"
: "${DRBX_WORKER_MEMORY_GIB:?measure/estimate on the allocated GPU node}"
: "${DRBX_MEMORY_RESERVE_GIB:?retain controller and system headroom}"
mkdir -p "$OUTPUT/provenance"
```

Record the allocation job ID, actual GPU partition/constraint, account/QoS,
requested GPU resources, host CPU affinity, host memory, and the four runner
resource values in `$OUTPUT/provenance/` without logging secrets. Verify the
allocation's accounting before proceeding. Do not use extra nodes or launch a
second writer. Preflight is mandatory at all three resolutions and samples
complete raw-cell memberships for its deterministic wall/adjacent owners:

```bash
conda run -n drb python scripts/p05_direct_midpoint_global/campaign.py \
  preflight --input-root "$INPUT_ROOT" --output "$OUTPUT" \
  --workers "$DRBX_WORKERS" \
  --memory-budget-gib "$DRBX_MEMORY_BUDGET_GIB" \
  --worker-memory-gib "$DRBX_WORKER_MEMORY_GIB" \
  --memory-reserve-gib "$DRBX_MEMORY_RESERVE_GIB" \
  2>&1 | tee "$OUTPUT/logs/preflight.log"
```

Proceed only if the command exits successfully, `preflight.complete.json`
exists, and `preflight_validation.json` records `cpu_backend: ["cpu"]`. A
failed preflight is an operational stop; report the exact error and do not
launch the global run.

## Global run, validation, and recovery

Only after preflight passes, execute the complete frozen campaign. The runner
processes N32, N48, and N64 in sequence, writes restartable raw-index chunks,
and uses one parent writer. It sets per-process JAX to CPU and hides CUDA
devices; confirm `JAX_PLATFORMS=cpu` is present in the allocation process
environment before launch.

```bash
conda run -n drb python scripts/p05_direct_midpoint_global/campaign.py \
  run --input-root "$INPUT_ROOT" --output "$OUTPUT" \
  --workers "$DRBX_WORKERS" \
  --memory-budget-gib "$DRBX_MEMORY_BUDGET_GIB" \
  --worker-memory-gib "$DRBX_WORKER_MEMORY_GIB" \
  --memory-reserve-gib "$DRBX_MEMORY_RESERVE_GIB" \
  2>&1 | tee "$OUTPUT/logs/run.log"
conda run -n drb python scripts/p05_direct_midpoint_global/campaign.py \
  validate --input-root "$INPUT_ROOT" --output "$OUTPUT" \
  2>&1 | tee "$OUTPUT/logs/validate.log"
```

Do not run `validate` until `run.complete.json` exists and the run command
exits successfully. If time limit or another operational interruption stops a
stage, preserve the same folder, revision, input root, configuration, and
resource caps. Resume by rerunning the failed prescribed stage command in that
folder; completed chunks with matching identity/SHA256 receipts are reused,
while missing or corrupt chunks are computed again. Do not remove identity
checks, relabel checkpoints, or change numerical parameters to force a resume.

The output folder must contain the physical reuse arrays; campaign/input
manifests; localized reference sidecar; resolved runtime input mapping;
preflight owner selections, chunks, receipts, and validation; raw run chunks
and receipts for all resolutions; per-resolution owner results; `summary.json`,
`result_manifest.json`, and `completion.json`; execution records; progress or
failure records; command and scheduler logs; and the allocation/provenance
record. Report the absolute output path as the single download target and
inventory anything missing. The built-in summaries and flags are outputs to
return unchanged, not an invitation to analyze a scientific gate failure.

Return a short operational receipt with the exact commit, campaign identity,
all input identities, job/allocation IDs, selected partition/constraint,
account/QoS, requested GPU resources, verified CPU backend, requested and
effective workers, host-memory budget/reserve, peak worker RSS, per-stage
elapsed times and exit statuses, prescribed validation status, absolute
campaign-folder path, artifact inventory/retrieval procedure, and any
operational failure with its exact command/error and resume point. Do not
include scientific interpretation.

Run the specified computation and return its artifacts and operational receipt. Do not analyze the scientific results; we will do that locally.
