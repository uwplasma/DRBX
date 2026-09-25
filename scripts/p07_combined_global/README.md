# Combined P07 static diffusion qualification

This tracked, computation-only campaign prepares and qualifies the frozen combined structured diffusion candidate on canonical HSX N32/N48/N64. It is a static four-field operator MMS test. It does not run time evolution, a fine-cell H intermediary, an elliptic solve, an energy gate, or a production driver.

The candidate uses physical-raw-volume-weighted owner midpoint observations, one distinct shared flux per canonical owner face, direct singleton and ringwise recovery, full Cartesian total-degree quartic C at coupled axis/aggregate supports, and the frozen wall/near-wall state, BC-value, and BC-tangential rules. The ringwise/coupled donor policy, relative SVD cutoff, residual tolerance, and expansion limits are in `configuration.json`. Candidate face integration remains q3; the primary continuum reference now uses raw-cell midpoint projection (v2). The prior analytic-gradient review established quartic reproduction on coupled C faces and a leading degree-six regular-axis remainder; the observed axis-core/first-ring rebound is retained as a diagnostic.

`topology.py` stores 91,904 / 309,456 / 730,624 distinct canonical face rows. **These totals include** 1,024 / 2,304 / 4,096 collapsed r=0 rows, which must be exactly zero. The noncollapsed physical wall is retained. Every shared face has one row and two opposite recipient incidences.

## Immutable inputs

`input_manifest.json` contains SHA-256 and size for 17 files relative to `INPUT_ROOT`: the six N32/N48/N64 base/topology arrays, original continuous-reference sidecar, metric cache, MAKEGRID, and all eight components named by the sidecar's 64³ continuum artifact. `INPUT_ROOT` may be outside the campaign output folder. The original sidecar is not modified; `verify-inputs` writes a localized copy under `OUTPUT` with the metric, MAKEGRID, and artifact paths rebased to `INPUT_ROOT`. The historical comparison-cache path is metadata, not a builder input. Source hashes, Git revision, numerical configuration, input identities, and localized-sidecar hash are recorded in `OUTPUT/campaign_manifest.json`.

The numerical kernels in `kernels.py` and `candidate.py` are curated from the bounded P07 direct, axis-quartic, and boundary-quadrature studies. They import the tracked `scripts/p07_diffusion_global/numerics.py` continuum/geometry routines and tracked DRBX source; they do not import loose `work/p07_*` modules or archived smoke answers. `local_validation.md` records frozen row overlaps and the isolated tracked-source import check.

## Stages and gates

1. `verify-inputs` hashes all immutable inputs and source files and records an immutable campaign identity. `topology` exhaustively checks owner incidence, periodic seams, collapsed axis, wall, family coverage, and donor-index bounds. `plan` fixes contiguous chunk IDs.
2. `support` numerically validates every ringwise/coupled face using actual and uniform owner moments. Full retained rank certifies the target space; rank-deficient systems must pass the actual q3 target residual. It also stores the certified sparse row for reuse. Unsupported support stops dependent stages.
3. `observations` evaluates four fields at every raw-cell midpoint in independent chunks and reduces using physical raw volumes to one value per canonical owner.
4. `assembly` compiles every q3 face row. It reuses certified ringwise/coupled rows, builds ordinary and boundary rows, serializes separate state/BC-value/BC-tangential channels, applies the four observed fields and prescribed wall traces, and saves exact-gradient q3 oracle fluxes. The reducer verifies exact face IDs and endpoints, finite coefficients, BC channel shapes and B=-W, zero collapsed rows, and complete coverage.
5. `reference` evaluates the analytic pointwise diffusion expression once per raw-cell midpoint and projects it with stored physical raw volumes. `control` repeats midpoint evaluation with half the derivative step on geometry-selected complete owners. No integrated volume-reference or extra high-quadrature face-reference stage is scheduled.
6. `summarize` applies shared-face incidence to the saved fluxes and normalizes by stored physical owner volume; the target uses stored physical owner volume and the summed midpoint analytic numerator. It writes four-field global/region errors, maxima, q3 reconstruction-versus-exact-face diagnostics and bounded midpoint derivative-step controls, and N32→N48/N48→N64 orders. `validate` rechecks all chunks, reductions, identities, hashes, and summary artifacts.

The pre-existing roadmap's primary gate is physical-owner-volume-weighted global operator L2 order **at least 1.8 on both intervals for each field**. Bounded midpoint derivative half-step sensitivity must remain within 10% of the corresponding selected-owner spatial error. Midpoint-to-integrated and additional high-quadrature reference controls are deferred at the user's request. Regional and maximum orders are diagnostics, not additional global gates. A completed scientific gate failure remains an output and does not authorize tuning. Axis core, first ring, coupled/ringwise join, aggregate interfaces, ordinary interior, physical wall, and periodic seams remain visible in arrays and summary. No topology-dependent patch order is reported as a global order.

Small absolute tolerances in the reducer (for example 1e-12 for B=-W and 1e-8 for constant-row cancellation) check serialization/algebraic integrity. They are not substitute accuracy or convergence gates; any failure preserves the affected chunks for local review.

## Commands

From the DRBX repository root, create one new unique output directory and set `INPUT_ROOT` to the immutable dataset root. The remote **run drbx on perlmutter** skill chooses `CPU_WORKERS`, `MEMORY_BUDGET_GIB`, `WORKER_MEMORY_GIB`, `MEMORY_RESERVE_GIB`, process affinity, and allocation. These are resource controls, not numerical parameters. Set scratch/cache/bytecode paths inside `OUTPUT` before launch.

```bash
python scripts/p07_combined_global/campaign.py verify-inputs --input-root "$INPUT_ROOT" --output "$OUTPUT"
python scripts/p07_combined_global/campaign.py smoke --input-root "$INPUT_ROOT" --output "$OUTPUT"
python scripts/p07_combined_global/campaign.py run --input-root "$INPUT_ROOT" --output "$OUTPUT" \
  --workers "$CPU_WORKERS" --memory-budget-gib "$MEMORY_BUDGET_GIB" \
  --worker-memory-gib "$WORKER_MEMORY_GIB" --memory-reserve-gib "$MEMORY_RESERVE_GIB"
python scripts/p07_combined_global/campaign.py validate --input-root "$INPUT_ROOT" --output "$OUTPUT"
```

`run` automatically proceeds through all five numerical stages only after each predecessor reduction passes. It uses node-local spawned CPU processes, one BLAS/JAX thread per process, bounded pending work, worker recycling after 128 tasks by default, and one controller/writer per output folder. Independent chunks have atomic NPZ results and JSON hash/identity receipts; processes reuse initialized geometry, metric reference, owner observations, and a bounded cache of certified support rows. The memory flags cap *effective process concurrency* from the requested worker count; they are estimates rather than an operating-system memory limit. The remote skill should choose values from its allocation and observed smoke memory. The controller records requested/effective concurrency and peak worker RSS. It does not use extra nodes implicitly.

Resume with the same `run` command and output folder. Completed matching chunks are reused; interrupted chunks without receipts are recomputed. Mismatched or corrupt receipts, an altered sidecar/source/configuration, incomplete coverage, unsupported supports, or invalid BC serialization fail explicitly and preserve evidence. `plan`, `run-stage --stage NAME`, `reduce-stage --stage NAME`, `summarize`, and `validate` expose the same gated stages for operational recovery. Do not edit inputs or thresholds to force reuse. Capture scheduler stdout/stderr, launch records, logs, scratch, caches, and the operational receipt beneath the single `OUTPUT` directory; existing immutable inputs may remain under `INPUT_ROOT` and are identified in the campaign manifest.

## Midpoint identity and reuse

The numerical diffusion remains the same conservative shared-face q3 action;
there is no numerical cell-volume expression to replace by midpoint evaluation.
Only the primary MMS reference changes. Existing integrated-reference manifests
and summaries cannot resume under the new v2 configuration. A validated importer
could later reuse unchanged observations/support/assembly, but none is implemented
here. Do not rename old q3 reference arrays as midpoint data. New primary artifacts
use `reference_midpoint`/`q1` names, with explicit bounded step controls. Historical
accepted results remain attached to their original reference convention.
