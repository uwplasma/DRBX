# Portable parallel FCI intersection/eta follow-up

This is a computation-only, N32 CPU campaign for the bounded boundary-aware
intersection and fixed-cap eta-gradient follow-up. It preserves the frozen
base/refined numerical rules and splits each eta quadrature plane into an
independent, resumable unit. It does not modify a production operator, expand
the selected patch, add polynomial degree, or perform scientific selection.

The already validated fixed-cap eta-gradient result is bundled as an immutable
input. The fresh remote computation is the ten-plane base intersection and the
fourteen-plane refined intersection.

## Inputs and source identity

`frozen_source_bundle.tar.gz` contains the exact research sources, current DRBX
Python package sources, the audited targeted-operator checkpoint, small parent
intersection artifacts, the archived local preflight, N32 Q01 references, and
the completed eta-gradient result.
`source_manifest.json` verifies every bundled file.

The following large immutable inputs remain in the established
`hsx_matched_global_parallel_v2` input root and are verified by
`external_input_manifest.json`:

- `mgrid_res2p5cm_180pln.nc`;
- `hsx_metric_d58d392545fd3917efeb83b6.npz`;
- `geometry_artifacts/rlp_convergence_32_48_64_20260917/32x32x32/`;
- `geometry_artifacts/rlp_convergence_32_48_64_20260917/64x64x64/`.

The source tree is extracted beneath the campaign folder. Links from that tree
to these immutable external inputs are inputs, not campaign outputs; all
generated results, logs, caches, manifests, checkpoints, and receipts remain
beneath the single campaign folder.

## Parallel execution model

The runner uses a node-local `multiprocessing` spawn pool. Each persistent
worker loads one N32 geometry/reference context and processes independent eta
planes. There is one parent writer, an exclusive campaign lock, atomic plane
payloads and receipts, and deterministic plane-order assembly. It is not a
distributed or multi-node runner. Every BLAS/OpenMP worker is capped to one
thread; the remote setup skill chooses the CPU allocation, affinity, worker
count, memory budget, per-worker memory allowance, reserve, and walltime.

The measured local reference worker footprint is approximately 1.3 GiB, but
the remote setup skill must choose a conservative allowance for its platform.
`--memory-budget-gib`, `--worker-memory-gib`, and `--memory-reserve-gib` cap the
effective worker count and must be supplied together as appropriate.

## Command sequence

Run from the repository root in the DRBX environment. `HSX_INPUT_ROOT` is the
extracted immutable input root described above. `CAMPAIGN_DIR` is one new,
uniquely named absolute output folder. `CAMPAIGN_WORKERS` and the memory
variables are chosen for the active CPU allocation.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$CAMPAIGN_DIR/tmp"
export XDG_CACHE_HOME="$CAMPAIGN_DIR/cache/xdg"
export PYTHONPYCACHEPREFIX="$CAMPAIGN_DIR/cache/pycache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX" "$CAMPAIGN_DIR/logs"

MEMORY_ARGS=(
  --memory-budget-gib "$CAMPAIGN_MEMORY_GIB"
  --worker-memory-gib "$CAMPAIGN_WORKER_MEMORY_GIB"
  --memory-reserve-gib "$CAMPAIGN_MEMORY_RESERVE_GIB"
)

python scripts/parallel_fci_intersection_eta_remote/campaign.py verify \
  --input-root "$HSX_INPUT_ROOT" --output "$CAMPAIGN_DIR"
python scripts/parallel_fci_intersection_eta_remote/campaign.py preflight \
  --input-root "$HSX_INPUT_ROOT" --output "$CAMPAIGN_DIR" \
  --workers "$CAMPAIGN_WORKERS" "${MEMORY_ARGS[@]}"

python scripts/parallel_fci_intersection_eta_remote/campaign.py run \
  --output "$CAMPAIGN_DIR" --case base --workers "$CAMPAIGN_WORKERS" \
  "${MEMORY_ARGS[@]}"
python scripts/parallel_fci_intersection_eta_remote/campaign.py assemble \
  --output "$CAMPAIGN_DIR" --case base
python scripts/parallel_fci_intersection_eta_remote/campaign.py validate \
  --output "$CAMPAIGN_DIR" --case base

python scripts/parallel_fci_intersection_eta_remote/campaign.py run \
  --output "$CAMPAIGN_DIR" --case refined --workers "$CAMPAIGN_WORKERS" \
  "${MEMORY_ARGS[@]}"
python scripts/parallel_fci_intersection_eta_remote/campaign.py assemble \
  --output "$CAMPAIGN_DIR" --case refined
python scripts/parallel_fci_intersection_eta_remote/campaign.py validate \
  --output "$CAMPAIGN_DIR" --case refined

python scripts/parallel_fci_intersection_eta_remote/campaign.py finalize \
  --output "$CAMPAIGN_DIR"
python scripts/parallel_fci_intersection_eta_remote/campaign.py validate-final \
  --output "$CAMPAIGN_DIR"
python scripts/parallel_fci_intersection_eta_remote/campaign.py provenance \
  --output "$CAMPAIGN_DIR"
```

Capture stdout, stderr, scheduler records, job IDs, and command exit statuses
under `CAMPAIGN_DIR`. The remote worker should add `operational_receipt.md`
there after completion.

## Resume and failure handling

Re-run the same `run --case ...` command with the same revision, immutable input
root, numerical settings, and campaign folder. Valid plane payloads are checked
and skipped; missing planes are computed. Worker count may change because it is
not part of a plane's numerical identity. Do not delete receipts, bypass hashes,
or relabel incompatible files. A second writer is rejected by the lock.

Routine allocation, affinity, environment, path, and scheduler issues may be
repaired operationally. If numerical code, inputs, or prescribed settings must
change, stop and return the exact failure evidence for local follow-up.

## Returned folder

Return the single real `CAMPAIGN_DIR` folder. It contains the extracted frozen
source, input/campaign manifests, preflight comparison, plane checkpoints and
receipts, base/refined assemblies, final summary/arrays, progress, operations,
cache/scratch/logs, provenance, scheduler records, and operational receipt.
Scientific interpretation and any follow-up experiment remain local work.
