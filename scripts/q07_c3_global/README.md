# Compact C3 h/32 Q07 material global campaign

This new campaign tests the compact C3 magnetic evaluator at **h/32 total inner
cap separation**, h = one eta-plane spacing. Inner points are +/-h/64; outer
characteristic samples are +/-h/32. Both use fresh RK4-64 traces. The symmetric
five-point material action and geometry-consistent tube divergence
`D(F) = D_old(F) + [div(b) - D_old(1)] F_center` are unchanged.

All complete owners at N32/N48/N64, 22 states (including four held-out waves),
four D/physical-normal N/mixed BC combinations, nine regions and all five
material fields are included. Fifteen outputs preserve centered, correction
and combined actions. N-O, O-R and N-R use consistent compact C3 geometry and
continuum references. No old-evaluator totals are an equality target: changing
B intentionally changes both actions and references. Independent previously
completed C3 bounded actions at 46 owners are the preflight replay target.

The accepted donor choices, degrees, eta support, wall conditioning and owner
volume projection remain fixed. No cut-leg/endpoint-local experiment is
adopted. No exterior endpoint, wall crossing or re-entry is accepted. An explicit
**full-domain trace-only gate at all three resolutions precedes global scoring**;
failures save offending seeds/legs and halt without masking owners. Static
qualification does not establish exterior BC behavior, evolution, conservation,
current–phi/SAT compatibility or production adoption. Diffusion/vorticity are
outside this material campaign. h/128 and long-leg comparisons are deferred.

## Portable inputs

Extract the checked-in `source_bundle.tar.gz` and `input_bundle.tar.gz` into this
folder after verifying `bundles.sha256`. The source is isolated from installed
DRBX. It freezes the three compact-evaluator modules in both source namespaces
and explicitly selects compact_c3 in the canonical context loader. Runtime
sources and verification scripts are byte-pinned by `design.json`.

The small input bundle contains the unchanged complete-owner plan, accepted
choices and C3 bounded evidence. Old campaign trace/result archives and the
h128 supplement are not needed. `Q07_INPUT_ROOT` must contain the eight exact
canonical files in `inputs_manifest.json`. The existing
`q07-balanced-canonical.tar.gz` can supply them if necessary (SHA256
`d72adf0ceb83c192c84f2ef425144206e5207df9e9d9df6d187db566f518d634`,
5,758,269,059 bytes). These are immutable topology/metric/MAKEGRID inputs;
regenerated geometry must not substitute for missing files.

Fresh C3 endpoints and magnetic factors are mandatory. No old numerical chunks,
old spline traces, preflight receipts or compilation caches may be relabelled.
All runtime outputs, traces, logs, caches and source exports belong under one
new unique RUN directory; canonical immutable inputs may remain external.

## Environment and execution

Use Python 3.12, JAX/JAXLIB 0.9.2, CPU x64, NumPy, SciPy, netCDF4, h5py,
pydantic and Solvax. This campaign uses a GPU allocation charged to the GPU
budget but computes on its CPUs; no GPU kernels are required. The remote
**run drbx on perlmutter** skill chooses allocation, accounting, CPU affinity,
workers and memory from actual resources. The runner is node-local, with spawn
processes and single-threaded numerical libraries; extra nodes are not used.

Set absolute Q07_INPUT_ROOT and Q07_OUTPUT. Set Q07_WORKER_GIB (default 4 GiB),
Q07_TOTAL_MEMORY_GIB and Q07_WORKERS from actual resources. Enforce
workers*worker_guard + 2 GiB parent reserve <= usable host-memory budget.
Set JAX_PLATFORMS=cpu, JAX_ENABLE_X64=true and CUDA_VISIBLE_DEVICES="" before
imports; BLAS/OpenMP threads are one. Place JAX_COMPILATION_CACHE_DIR,
DRBX_CACHE_DIR and TMPDIR under RUN. At least 3 GiB free disk is required.

```sh
sha256sum -c bundles.sha256
tar -xzf source_bundle.tar.gz
tar -xzf input_bundle.tar.gz
python campaign.py verify
python verification/test_campaign.py
python campaign.py preflight --workers "$Q07_WORKERS"
python campaign.py pilot --workers "$Q07_WORKERS"
python campaign.py trace --n 32 --workers "$Q07_WORKERS"
python campaign.py trace --n 48 --workers "$Q07_WORKERS"
python campaign.py trace --n 64 --workers "$Q07_WORKERS"
python campaign.py run --n 32 --workers "$Q07_WORKERS"
python campaign.py run --n 48 --workers "$Q07_WORKERS"
python campaign.py run --n 64 --workers "$Q07_WORKERS"
python campaign.py analyze
python campaign.py validate-completion
```

Never run freeze remotely. Preflight and pilot use the same parallel worker
pool as scoring. Trace-only workers skip MMS state preparation. Fixed-size
trace/material kernels reuse compilations; workers recycle after 24 chunks.
All 793 complete-owner chunks retain sums, maxima with locations, signed
integrals and reference norms rather than every action array. Trace and score
checkpoints have byte hashes, coverage and identity checks; an output lock
prevents concurrent writers. Interruptions resume the same command and output.
Do not delete valid payloads or weaken numerical gates to resume.

## Gates and failure evidence

Hashes read bytes rather than timestamp-cached identities, including identical-
stat rewrite tests. Bounded centered/correction replay is 1e-8 each; combined
replay is 2e-8, with actual and expected constituent-sum residual separately
bounded by 128*eps64*(1+abs(centered)+abs(correction)+abs(combined)). Constants
use N-O <=1e-7; center-b replay <=1e-10. Complete owners, finite values, positive
thermodynamic states and admissible characteristic splits are mandatory.
These engineering/roundoff gates are distinct from scientific convergence.
There are no field-, location- or resolution-specific exceptions.

This carries the known timestamp-hash and constituent-budget fixes, but the
latest old-campaign failure report was not available during preparation; it
is not being declared diagnosed or fixed. If this campaign fails a gate,
return its exact error and evidence. Do not tune numerics or tolerances.
Scientific regional regressions in a completed run are results to retain.

Return chunks and trace payloads/receipts, trace_validation_N*.json,
validation_N*.json, verify/preflight/pilot receipts, totals.npz, analysis.json,
orders.csv, report.md, analysis_receipt.json, independently validated
completion.json, logs/source/provenance and an operational receipt, all inside
RUN. Built-in postprocessing is required; remote scientific interpretation,
extra experiments, production promotion and optimization are excluded.

## Local readiness — 1 October 2026

Frozen identity:
`ab607bf62cda21dd67935334da9fdea35c2ad0a0c529ff555006aa3a8ed83bc3`.
Fifteen focused tests passed in both the source workspace and a separately
extracted portable package. All 46 bounded complete-owner preflights passed;
worst N/O/R constituent/combined replay difference was 9.611e-12. Four N64
regional pilot groups passed; measured worker peak was 1.122 GiB. Pilot
projection is 148–228 minutes on two local workers, not a remote timing
promise. Statistical payload projection is 0.16 GiB, excluding traces, source,
caches and logs. Remote resource choice uses its own pilot.

Two full N64 bulk/wall chunks (1,024 owners) passed actual computation and
checksum-validated exact checkpoint reuse. Maximum constant N-O was 4.905e-8,
below the unchanged 1e-7 gate. No local full-grid campaign was launched; all
three full-domain trace gates remain mandatory remotely. Readiness receipts
are under `verification/`; they are evidence, not remote execution receipts.
