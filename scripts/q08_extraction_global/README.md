# Q08 extraction replay and A100 execution audit

This campaign checks that the extracted shared Q preparation/runtime reproduces
the frozen pre-extraction implementation on every N32/N48/N64 owner. It then
measures the full six-field RHS on **one and four actual A100 GPUs**. It is an
implementation and resource gate, not another manufactured-solution convergence
campaign. No tracing or continuum-reference integration is performed.

## Frozen numerical scope

- The returned compact-C3 RK4-64 endpoints, canonical topology/volumes and
  accepted geometry-only support choices are immutable inputs.
- All 22 accepted six-field manufactured states, four uniform/mixed
  Dirichlet/physical-normal Neumann patterns, prescribed phi, material h/32
  inner and h/16 outer samples, and both explicit diffusion spans are replayed.
- Omega uses the accepted six-field audit's affine combination
  `0.2 + .6*(n-1) - .4*(Te-1.1) + .3*(Vi-.13)`.
- All 31 prepared arrays must be bitwise identical between literal legacy and
  paired preparation **on the same CPU**. Cross-platform action replay uses
  `1e-8 + 1e-11*abs(expected)` per element, with all output leaves checked;
  booleans must agree exactly and indicate validity. This is a floating-point
  replay budget, not a spatial-convergence criterion.
- The C3 geometry audit uses the accepted center-b absolute replay tolerance
  `1e-10`, zero relative tolerance. No gate may be relaxed remotely.
- The dense runtime remains selected. No new donors, fit degree, boundary law,
  magnetic evaluator, span or production selector is introduced.

## Inputs and source

`inputs.tar.gz` (about 24 MB) contains the complete saved traces, choices, plan,
C3 geometry helper source and independent bounded actions. `input_manifest.json`
records every member and eight large canonical files. Canonical geometry,
metric cache and MAKEGRID remain external immutable inputs below `Q08_INPUT_ROOT`.
Do not regenerate missing files. The prior canonical archive is
`q07-balanced-canonical.tar.gz`, SHA256
`d72adf0ceb83c192c84f2ef425144206e5207df9e9d9df6d187db566f518d634`,
5,758,269,059 bytes; extraction may be reused only when all manifest files verify.

`legacy/` is the literal pre-extraction Q source, with provenance in
`legacy_provenance.json`. `runtime/` is a byte-for-byte snapshot of the extracted
Q modules listed in `design.json`, with empty package initializers to avoid
unrelated production imports. Preparation and application do not import a past
campaign driver. Never run `freeze` or `freeze_inputs.py` on remote.

## Execution

Use one unique campaign folder `Q08_RUN` for results, input extraction, caches,
logs, scheduler records and provenance. Keep one writer; the controller takes
an exclusive lock. Commands below run from this directory at the pinned commit.
Python must provide NumPy, SciPy, netCDF4 and float64-enabled JAX/JAXLIB, with a
working CUDA backend for the GPU stage. No editable install or Solvax import is
required by this minimal runtime.

The remote setup skill selects CPU `WORKERS`, `WORKER_GIB` and `HOST_GIB` from
the actual GPU allocation; `WORKERS*WORKER_GIB + 4 <= HOST_GIB` is enforced.
Start with measured per-worker requirements from preflight/pilot (6 GiB is a
conservative initial limit, not a worker-count prescription). Workers use
spawn, single-threaded numerical libraries and bounded complete-owner chunks;
concurrency is node-local. CPU stages hide GPUs. A100 stages use actual GPU
arrays, contrary to the usual CPU-only GPU-allocation campaign default.

```bash
export Q08_RUN=/absolute/unique/run
export Q08_INPUT_ROOT=/absolute/immutable/canonical/root
python campaign.py verify
python verification/test_campaign.py
python campaign.py preflight --worker-gib "$WORKER_GIB"
for n in 32 48 64; do
  python campaign.py data --n "$n"
  python campaign.py pilot --n "$n" --workers "$WORKERS" --worker-gib "$WORKER_GIB" --host-gib "$HOST_GIB"
  python campaign.py cpu --n "$n" --workers "$WORKERS" --worker-gib "$WORKER_GIB" --host-gib "$HOST_GIB"
  python campaign.py validate-cpu --n "$n"
done
# Fresh processes: do not carry CUDA_VISIBLE_DEVICES="" or JAX_PLATFORMS=cpu
# from CPU launch wrappers into this stage.
for n in 32 48 64; do
  env JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python campaign.py gpu --n "$n" --host-gib "$HOST_GIB"
done
python campaign.py analyze
python campaign.py validate-completion
```

Capture each exit status; stop at a nonzero exit. The examples are command
contracts, not a shell supervisor. Run only one node and one controller per
RUN. GPU stage requires four visible A100 devices and separately executes
one-device and four-device paths. CPU topology emulation is only a local unit
test; it cannot satisfy the remote completion gate.

Completed CPU banks and per-case GPU receipts are content-checked on resume.
Rerun the same interrupted command at the same revision/input identity.
Corrupt or mismatched chunks stop validation rather than being silently reused.
The full-grid bank merge is an engineering view retaining each checked chunk's
identity, not a forged single-chunk artifact. One manufactured state is staged
at a time; all-case action arrays are not accumulated on disk. No full-grid
legacy directional rows are retained after replay.

## Outputs and interpretation

Return the whole RUN, including compact banks, geometry coefficients, owner
state data, CPU/GPU receipts, timings/memory, `analysis.json`, `report.md`,
`completion.json`, source/input provenance, logs and operational receipt.
Keep at least 3 GiB free disk throughout. Use the pilot/merge memory guard to
confirm capacity; compilation time and memory are measured, not assumed from
logical array bytes. A GPU out-of-memory or numerical mismatch is a failed gate
for local follow-up, not permission to tune the implementation.

All-owner replay, actual one/four-GPU equivalence and synchronized warm RHS
measurements are distinct from physical timestep stability, live sheath/SAT,
exterior crossings, C3 diffusion-transfer qualification and span selection.
Those roadmap gates remain open. The C3 evaluator stays opt-in.
