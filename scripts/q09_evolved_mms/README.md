# Q09 N32 timestep-refinement pilot

Run a short **full-domain N32 GPU** prescribed-phi evolution, using the accepted
saved Q08 rows and independent continuum reference. This is an engineering and
temporal-refinement pilot; completion is not a stability or evolved spatial
convergence qualification. No production solver is changed.

The frozen cases are diffusion-only and complete six-field parallel RHS, each
with `DDDDDD/phiD`, `NNNNNN/phiN`, `DNDNDN/phiN`, and `NDNDND/phiD`. Each case
uses `dt=1e-6,5e-7,2.5e-7` to exactly `t=1e-5` (10,20,40 accepted RK4 steps).
There are eight cases, 24 level runs, 560 accepted steps if all finish. At this
short interval temporal self differences may reach roundoff; no order threshold
is imposed or tuned. Solution errors and temporal differences remain separate.

Use actual A100 GPU/x64 for the evolving RHS; preparation and portable tests
use CPU. The runner preserves scheduler CUDA visibility and uses its first
visible GPU, running cases sequentially. Extra allocated devices are idle.
There is no CPU numerical fallback or multi-device RK4 claim. Single-threaded
BLAS and caches are configured before imports. Host memory budget is an explicit
admission bound chosen from the actual allocation. Device admission reuses the
Q08 conservative bound plus additional RK4 state/BC storage; measured use and
compile cost remain results to collect.

The runtime copies current Q source and committed geometry type definitions into
`runtime/`, avoiding unrelated production-package initialization and optional
solver imports. Only the campaign bootstrap uses this snapshot. `manifest.json`
hashes runtime, harness, tests, and the consumed shared reference/input authority.
Do not refreeze on remote or modify numerical/source/validation parameters.

Run from a pinned checkout inside a new unique `$RUN/source`. Use the remote
setup skill for GPU allocation and environment; `$PY` is its verified Python,
and `$HOST_GIB` its host budget. These commands are node-local; start one writer.

```bash
CAMPAIGN="$RUN/source/scripts/q09_evolved_mms/campaign.py"
"$PY" "$CAMPAIGN" verify --run "$RUN" \
  --baseline-run /pscratch/sd/y/yiqunx/q08-extraction-global-ff1bff86-20261002T233534Z \
  --science-run /pscratch/sd/y/yiqunx/q08-rhs-mms-bd3410af-U4rB3I \
  --canonical-root /pscratch/sd/y/yiqunx/uw_summer --host-gib "$HOST_GIB"
"$PY" "$CAMPAIGN" tests --run "$RUN"
"$PY" "$CAMPAIGN" preflight --run "$RUN"
"$PY" "$CAMPAIGN" run --run "$RUN"
"$PY" "$CAMPAIGN" validate --run "$RUN"
```

Verify checks all consumed bank/reference receipts and canonical geometry hashes.
The old scientific run does not need `completion.json`. Portable tests cover
shared RK4/source stages, bounded actual-HSX Q replay, input corruption, restart,
and independent reductions. GPU preflight compares every actual bounded Q action
component with CPU for all four BC patterns. Full bank/reference preparation is
checkpointed under `prepared_reference/`; it reuses saved continuum diffusion,
reevaluates only raw-center material reference geometry, and never retraces or
rebuilds rows. The factory revalidates full owner/raw/donor closure and t0 R replay.

`run` stages a single dynamic GPU payload once, then compiles once per mode/BC
case and reuses it for all three timestep levels. Every stage validates finite
state/source/RHS, positivity, and applicable characteristic/reconstruction
admissibility. Checkpoints after accepted steps contain state, stage history,
integrated RHS, run binding and content digests. Interrupted allocation: resume
the same RUN with the same commands/configuration. Accepted levels and cases
are validated and reused. Scientific/admissibility failures are evidence to
return; do not change dt, time, source, rows, spans, BCs or tolerances remotely.

Return the whole RUN with configuration, preparation cache/receipts, observations,
24 final level checkpoints, per-case reports/timings/receipts, summary/completion,
source, environment/scheduler records, operational logs and receipt. A final
validator independently reduces saved states, errors, time histories and balance
diagnostics and hashes the consumed artifacts. Keep timing reports across
interruptions: warm step times include host validation but exclude checkpoint
I/O; first executed step can include compilation. Overall case and preparation
timings include their full host work. No full-stage RHS arrays are retained.

See `src/drbx/dev_docs/q09_evolved_mms_contract.md` for the MMS/BC formulation.
