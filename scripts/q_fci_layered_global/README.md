# Frozen layered Q global D/N campaign

Research qualification, not production promotion. This is a new runner; do not
reuse the old shortspan campaign's fits, exact screens, traces or checkpoints.
The immutable eight canonical inputs are shared. `input_manifest.json` pins
all bytes/hashes; do not regenerate geometry or replace missing files.

## Numerical contract

N32/N48/N64; all complete owners. Volume-weighted raw-midpoint observations and
reference projection. Eighteen frozen fields (`fields.FIELDS`), both prescribed
Dirichlet and physical-normal Neumann. All cases use the same fields, geometry
and traces. N/O/R are retained separately per owner; complex waves retain sign
and phase. h/16 and h/32 use four RK4 legs per raw cell, one midpoint seed and
six cap/center evaluation slots (two centers coincide). **64 RK4 steps on GPU**.
There is no subface quadrature or nested inner scalar tracing in this method.

Inner: frozen repaired compact28 Cartesian quartic through the last aggregate;
outer: ringwise angular7/radial-cubic reconstruction. Common five-plane eta
quartic in every region. Last two wall layers: four interior radial layers and
a quartic wall polynomial. D retains the analytic prescribed-trace extension
and supplied trace tangential derivatives. N eliminates35 wall trace values
using all metric-normal derivative components; only prescribed normal data
enter the runtime loading. Its35x35 solve is setup only. Include nonzero divB
in the short-span outer action. See the pinned research evidence in `VALIDATION.md`.

The geometry evaluator snapshot in `vendor/` is deliberately frozen to the
validated research runtime, avoiding unrelated production edits. Its origin
and hashes are in `vendor_receipt.json`. No workspace or /tmp imports are used.
CPU references use the same NumPy/SciPy evaluator inside the wall and frozen
JAX evaluator for rare outside queries. The fourth-order coordinate-flux
reference uses steps1e-4 and5e-5; no volume/face reference integration is added.

## Execution model and outputs

Single-node. One campaign coordinator/writer. GPU tracing uses one thread per
selected GPU, immutable resident field data and fixed batches; it does not run
CPU reconstruction concurrently. CPU stages use spawned single-thread workers,
capped by requested workers, allocation CPU affinity, and floor(memory budget /
per-worker GiB). On Linux each worker is pinned to one allowed CPU. Every worker
loads geometry and batches all18 fields. Start with at least2.5GiB per worker;
inspect pilot measured memory and increase this operational cap if necessary.
Bounded caches clear at chunk boundaries. Do not allocate extra nodes for this
node-local runner. No GPU benchmark or scaling study is prescribed.

One new uniquely named campaign directory contains geometry preflight, trace,
numerical checkpoints, reference arrays, owner actions, summary, JAX caches,
provenance and receipts. Put Slurm scripts, stdout/stderr, launch records and
monitor logs in it too. Software environment and immutable inputs may stay
outside. All outputs are ordinary files, never external-output symlinks.

`--raw-chunk 512` freezes chunks with at most512 raw members (2048 simultaneous
RK4 legs); owners are never split. Only this execution batching changes from
local small research loops. Resume reissues the same commands against the same
folder. Source, plan and input identities are frozen. Hash-verified completed
chunks are skipped; incomplete output without its receipt is recomputed.
Corrupt or incompatible completed chunks are fatal. Never delete a receipt or
bypass an identity check to force reuse. An OS lock prevents duplicate writers
and releases automatically after a process ends.

## Commands (run from the repository root)

Use the remote `run drbx on perlmutter` skill for GPU allocation/accounting,
environment, CPU affinity, memory budget and walltime. Actual GPU tracing is
explicitly requested; older CPU-only handoff defaults do not apply. Do not hide
CUDA devices or globally force JAX_PLATFORMS=cpu in the GPU command. CPU commands
select their own CPU backend. GPU devices must be visible in the allocation.
Set absolute INPUT_ROOT and CAMPAIGN, WORKERS, MEMORY_GIB, PER_WORKER_GIB, and
GPU_DEVICE_IDS (a shell array of allocated local GPU indices). Local replay used
Python3.12, JAX/jaxlib0.9.2; choose a compatible CUDA-enabled JAX environment
with NumPy/SciPy/netCDF4. The mandatory CPU/GPU check verifies actual execution.
The tracer uses functools.partial(jax.jit,...) and avoids the former JAX0.6
`jit() missing fun` decorator failure.

```bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$CAMPAIGN/logs" "$CAMPAIGN/provenance" "$CAMPAIGN/cache"
python -m scripts.q_fci_layered_global.campaign init \
  --campaign "$CAMPAIGN" --input-root "$INPUT_ROOT" --raw-chunk 512
python -m scripts.q_fci_layered_global.campaign verify \
  --campaign "$CAMPAIGN" --input-root "$INPUT_ROOT"

# Exhaustive reconstruction/conditioning coverage, including every eta plane
# and every wall patch. Parallel, resumable and without tracing or MMS scoring.
for N in 32 48 64; do
  python -m scripts.q_fci_layered_global.campaign geometry \
    --campaign "$CAMPAIGN" --input-root "$INPUT_ROOT" --N "$N" \
    --workers "$WORKERS" --memory-gib "$MEMORY_GIB" \
    --per-worker-gib "$PER_WORKER_GIB"
done

# Mandatory representative full numerical checks, then a throughput pilot,
# then global. No new numerical choices based on pilot/scientific results.
for STAGE in preflight pilot global; do
  for N in 32 48 64; do
    python -m scripts.q_fci_layered_global.campaign trace \
      --campaign "$CAMPAIGN" --input-root "$INPUT_ROOT" --N "$N" \
      --stage "$STAGE" --backend gpu --devices "${GPU_DEVICE_IDS[@]}"
    if [ "$STAGE" = preflight ]; then
      python -m scripts.q_fci_layered_global.campaign check \
        --campaign "$CAMPAIGN" --input-root "$INPUT_ROOT" --N "$N"
    fi
    python -m scripts.q_fci_layered_global.campaign score \
      --campaign "$CAMPAIGN" --input-root "$INPUT_ROOT" --N "$N" \
      --stage "$STAGE" --workers "$WORKERS" --memory-gib "$MEMORY_GIB" \
      --per-worker-gib "$PER_WORKER_GIB"
    python -m scripts.q_fci_layered_global.campaign validate \
      --campaign "$CAMPAIGN" --input-root "$INPUT_ROOT" --N "$N" --stage "$STAGE"
  done
done
python -m scripts.q_fci_layered_global.campaign reduce \
  --campaign "$CAMPAIGN" --input-root "$INPUT_ROOT"
```

Capture all command and scheduler logs below CAMPAIGN. Preflight `check` compares
GPU64 against CPU64 and CPU256 on deterministic raw samples; tolerances are
1e-8 and1e-7 in cell-scaled logical positions, respectively. This is a bounded
trace-equivalence check, not a full CPU campaign. Global64 remains unchanged.
Do not use the local-only `--test-mode`/`--backend cpu-test` on remote.

CPU fitting/rank, Neumann system, constant action, physical geometry, trace
validity/reentry/reach, coverage, checksum and missing-output failures are
operational stops. Return evidence for local repair; do not tune the method.
A completed scientific order flag below the gate is not an execution failure.
The reducer reports global/regional volume-weighted norms, maxima, reference
step sensitivity and orders for N-O/O-R/N-R without scientific interpretation.

After interruption, fetch the same revision, verify inputs and rerun the same
sequence. Keep the same source, raw-chunk, fields, spans, inputs and RK4 count.
GPU device selection, CPU worker/memory caps and allocation can change within
available hardware without changing numerical identity. Inspect throughput and
peak worker memory after pilot; change only resource settings or walltime.

Completion requires geometry, preflight, CPU/GPU checks, pilot and global
receipts for all three resolutions, `results_N32.npz`/N48/N64, `summary.json`
and `completion.json`. Return the entire campaign folder, including incomplete
checkpoints/logs if an operational failure prevents completion. No scientific
analysis, candidate selection, numerical repair, extra experiments or production
promotion is assigned to the remote worker.

## Logged single-command launcher

`run_all` executes precisely the sequence above, recording every command, exit
status and elapsed time under the campaign directory. It verifies inputs before
numerical work, and resumes rather than replacing completed chunks:

```bash
python -m scripts.q_fci_layered_global.run_all \
  --campaign "$CAMPAIGN" --input-root "$INPUT_ROOT" \
  --workers "$WORKERS" --memory-gib "$MEMORY_GIB" \
  --per-worker-gib "$PER_WORKER_GIB" --devices "${GPU_DEVICE_IDS[@]}"
```

The launcher holds its own lock, while stage commands hold the coordinator
lock. Keep one launcher per campaign. Store its stdout/stderr beneath
CAMPAIGN/logs. After pilot, the existing run receipts supply throughput and
memory observations; the launcher proceeds to global when the prescribed
operational gates pass. Resource selection should budget this full run.
