# Q08 polynomial full-grid GPU continuation

This is a new GPU-stage namespace reusing the immutable completed CPU work
from `/pscratch/sd/y/yiqunx/q08-extraction-global-ff1bff86-20261002T233534Z`.
It does not restart preparation, trace field lines or change the numerical
scheme. The candidate explicitly selects `characteristic_method="polynomial"`;
the original CPU eig action remains the independent validation reference.
Candidate computation remains on actual GPUs, with no CPU characteristic
offload. The production default is unchanged.

The baseline source is `OLD/source/scripts/q08_extraction_global` with identity
`ff1bff86e4af5045b84856d16550f0f45edcbe03b83fc83cc5584557af4c8722` (see the
authoritative full identity in `manifest.json`; source verification is mandatory).
The reviewed five-module overlay is byte-identical to the successful bounded
A100 profile. All other runtime/preparation code loads from the frozen original
source. Do not freeze or edit source/manifests remotely.

## Scope and optimization

- N32/N48/N64, all 22 six-field states, four uniform/mixed D/N patterns,
  both diffusion spans h/16 and h/32, one/four A100 execution: 352 records/grid.
- Material inner h/32, outer h/16; saved C3 RK4-64 endpoints and support choices.
- Every leaf, finite value and validity flag compared against CPU eig reference
  with unchanged `1e-8 + 1e-11*abs(expected)`. Eight nonconstant smooth-state
  merge audits per grid still compare to literal checked chunk actions.
- Exact boundary caching retains only wall rows plus bitwise-uniform nonwall
  templates. All22 full-grid and smooth literal-chunk entries are bounded by
  an explicit cache budget included in host preflight. No new fitting or BC law.
  Only one dense state's boundary arrays are used at a time.
- The optimized producer evaluates only the requested state, in batches of
  at most 128 wall rows. It builds compact wall data directly; nonwall zeros
  and the original affine omega offset are filled when restoring dense input.
  All 22 states are checked bitwise against the original producer on up to
  eight distributed actual wall rows before GPU work, including normal data
  and nonwall padding. No field formula, wall law or replay tolerance changes.
- The eight literal CPU eig audits remain eager and independent of the merged
  candidate. Their host plans are lowered once per chunk/span and reused across
  the four BC combinations. Plans reference the checked NumPy bank arrays;
  this does not cache full-grid reference outputs or move candidate work to CPU.
- Candidate compiler targets exclude general eigensolvers and host callbacks.
  Compilation, transfer, CPU reference, boundary preparation, seven synchronized
  warm calls and host/device memory are recorded separately.
- Hardware/driver inventory runs once at preflight; allocator memory is still
  sampled per record. Each compiler proof is read, hashed and inspected once
  per validation invocation, with a fresh check on restart/completion.
- An exclusive wall-clock ledger includes merge, hashing/provenance, BCs,
  references, chunk audits, output comparison, compiler inspection, I/O and
  cleanup. Separate record totals count the shared CPU/BC work once for each
  one/four-GPU pair. Seven warm repeats and all 352 records remain required.

This verification optimization has its own frozen source identity. It does
not edit or resume the already-running `0107dd5d` campaign in place. Existing
records must retain their original identity; a new deployment uses a separate
RUN unless an explicit, separately validated migration is provided.

Read-only verification independently re-reduces every CPU chunk receipt and
coverage gate, checks all bank/data payload hashes and original canonical
inputs. Missing/incomplete N48/N64 banks are an input failure; do not rebuild
them or relabel identities under this assignment. Old partial GPU records are
retained but never counted toward the polynomial matrix.

## Execution

Use the remote `run drbx on perlmutter` skill to select accounting/QoS, CPU
affinity, host memory budget and walltime. This campaign explicitly requires
one node with four actual A100 GPUs, one controller and one numerical process;
it has no CPU worker pool and no multi-node support. Keep the working JAX/JAXLIB
0.9.2 CUDA environment from the successful profile, float64 and scheduler CUDA
visibility. CPU processes are limited to verification/reference/setup work.

Create one new RUN outside OLD. All outputs, code snapshots, logs, caches,
scheduler records and operational receipt stay beneath RUN. Existing large
immutable CPU banks and canonical inputs stay outside; their original content
receipts are returned in provenance. Do not copy the old multi-GB banks into
the download folder or replace output files with symlinks.

From this directory at the pinned commit:

```bash
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$RUN/logs"
"$PYTHON" run_all.py --run "$RUN" --baseline-run "$OLD" \
  --baseline-source "$OLD/source/scripts/q08_extraction_global" \
  --host-gib "$HOST_GIB" --through N32
"$PYTHON" run_all.py --run "$RUN" --baseline-run "$OLD" \
  --baseline-source "$OLD/source/scripts/q08_extraction_global" \
  --host-gib "$HOST_GIB" --through complete
```

The first invocation runs portable harness tests, read-only verification of
all three grids, then N32. Record its measured memory/timing and verify its
successful exit before the second invocation. The second command rechecks
completed N32 records and continues N48/N64, analysis and independent completion
validation. The original expensive GPU eig method is not rerun on the candidate
device path. CPU eig references remain required for the replay.

`campaign.py` also exposes `verify`, `gpu --n N --host-gib B`, `analyze`, and
`validate-completion`; all require `--run`, `--baseline-run`, `--baseline-source`.
Use `--help` for argument verification. Source/input, memory, replay or compiler
failure stops the run; return the evidence without changing gates or numerics.

## Recovery and return

One writer per RUN; controller/stage locks prevent duplicates. Preserve the
old RUN and use the same new RUN, revision and inputs on restart. Rerun the
same `run_all.py ... --through complete` command after an ordinary allocation
interruption and confirmed absence of the old process. Checked per-case records
resume without repeating completed timing samples. Invalid records are not
silently promoted; completion rejects incomplete coverage or missing compiler
proof. Do not restart the stopped old controller.

Keep at least 3 GiB free disk. Resource guards precede dense merge/device
placement; allocation values come from the actual node. An OOM or replay
failure is not authorization to relax a limit. Use the remote skill's detached
supervision for long stages, keeping launch logs and exact exit statuses.

Return RUN with `verification.json`, `gpu_N32/48/64.json`, `gpu/` records and
compiler text, `analysis.json`, `report.md`, `completion.json`, source snapshots,
provenance, stage logs and operational receipt. Large compilation caches may be
excluded from the transfer after successful completion, with their size recorded;
never exclude files covered by completion hashes. Completion is implementation
qualification only, not a new MMS/evolution/physical-wall gate or production
promotion. Remote work is computation and operational validation only.
