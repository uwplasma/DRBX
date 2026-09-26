# P07N static physical-normal Neumann campaign

This tracked computation-only runner prepares a frozen N32/N48/N64 global static diffusion qualification. It does not evolve a field, solve a new elliptic problem, promote a package default, or change P05N/P06N. The four nonconstant axis-regular fields and constant, C-infinity radial cutoff, physical-normal data in field units per metre, q3 candidate, canonical face incidence, stored owner volumes and raw-midpoint continuum target are in `configuration.json`, `fields.py`, and `core.py`. The Neumann action receives only owner observations and prescribed physical-normal samples. Exact wall trace and tangent data enter only the matched Dirichlet diagnostic; analytic gradients enter only the oracle and reference.

`core.py` reuses the tracked P07 canonical topology and integrated-face rows. Physical-wall and adjacent boundary families use the structured 28-cardinal-node trace elimination: quartic radial for families 1/2 and cubic radial for transverse family 4. Its geometry-only 28x28 inverse/response is cached by the actual ordered angular patch and radial degree, with a bounded per-worker cache. Other faces use the unchanged integrated q3 rows. D is computed from the same common nonboundary face actions and adds diagnostic boundary loadings only at boundary faces; there is no second global Dirichlet run. Every complete owner includes every raw member, every incident face, and both signs of shared-face incidence.

## Immutable input contract

`input_manifest.json` lists SHA-256/byte identities for all 17 immutable files relative to `INPUT_ROOT`: six canonical N32/N48/N64 geometry/topology arrays, the original continuum sidecar, metric cache, MAKEGRID, and eight backing 64³ geometry-artifact files. `verify-inputs` hashes all 17, makes a localized sidecar under `OUTPUT`, verifies a CPU JAX backend and records source hashes, configuration, commit and input identities. Source identity is strict even when the checkout has unrelated dirty files. The original sidecar and immutable inputs remain unchanged. A missing remote input stops verification; no geometry regeneration is part of this campaign.

The portable code imports only tracked `scripts/p07_combined_global`, `scripts/p07_diffusion_global`, `hsx_mms_continuum_reference.py`, and tracked `src/drbx` modules. It has no import from local `work/` archives or hardcoded workstation paths. The runtime input root is an explicit argument.

## Frozen stages

1. `topology` runs each grid in bounded node-local CPU worker processes and checks canonical owner/face coverage, collapsed axis, wall and family census. It freezes complete preflight owners in core, aggregate, seam, wall, ordinary and correction-transition regions; every incident face is included. A smaller geometry-stratified q5 **face oracle** subset includes core/aggregate and transition regions.
2. `preflight` executes parallel independent face, midpoint-reference and q5-oracle chunks with the same worker/memory controls as the main computation. It checks complete member reductions, records signed N/D/O_q3/R at selected owners, q3-q5 face differences, and nominal/half-step numerical-reference uncertainty. This is bounded; there is no integrated-volume target or full q5/q7 campaign.
3. `run` requires completed preflight, then parallelizes all raw observations, every canonical face, and all raw midpoint references. Independent chunks have atomic NPZ and JSON SHA-256/identity receipts. The controller enforces one writer per output folder and bounded pending work. All successful matching chunks resume in place; mismatched identity, corruption or incomplete coverage stop. `--workers`, `--worker-memory-gib`, `--memory-budget-gib` and `--memory-reserve-gib` are required for every computational stage, including preflight and topology. The memory settings bound process concurrency and each worker checks its peak RSS against its declared cap. The remote setup skill chooses values from its allocation; this node-local runner does not span nodes.
4. Reduction writes global per-owner N, D, O_q3, R, N-O, O-R, N-R and N-D, regional L2/maxima, physical-volume-weighted global L2 errors, constant action and shared-face balance. Wall diagnostics include signed bias, physical-area-weighted RMS/max absolute residual, and normalization by physical `|grad_x f|=sqrt(g_contra^ij f_i f_j)`. The pointwise normalized maximum uses a reported `1e-12` field/metre denominator floor. The earlier coordinate-dependent `|a||grad_q f|` ratio is not used as an acceptance gate.
5. `validate` requires all chunk IDs, identities, hashes, full owner and face arrays, preflight and global summaries. Scientific flags remain outputs, never instructions to retune or rerun altered numerics.

## Predeclared scientific checks

For **each** of the four nonconstant fields, the stored-physical-owner-volume global L2 order of `N-R` must be at least **1.8 on both** N32→N48 and N48→N64 intervals. The held-out B2 field has no exemption. The constant-field maximum `|N-R|` must be at most `1e-8`. Regional norms/maxima and D/O splits are reported rather than separately gated. The selected-owner nominal versus half-step numerical-reference difference must be no more than 10% of the selected spatial `N-R` L2 plus an `1e-8` numerical floor. `O-R` is a legitimate face/midpoint consistency difference, not reference uncertainty, and is not required to be subdominant. For wall consistency, the global physical-area-weighted normal-residual RMS must be nonincreasing on both intervals for each nonconstant field; no arbitrary hard cutoff is imposed on the earlier held-out B2 relative residual. These policies were frozen before any global outputs.

## Commands and recovery

From the DRBX repository root, with a **new unique** `OUTPUT` folder and an immutable `INPUT_ROOT` containing the 17 mapped files:

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$OUTPUT/cache" DRBX_CACHE_DIR="$OUTPUT/cache/jax" TMPDIR="$OUTPUT/scratch"
python scripts/p07_neumann_global/campaign.py verify-inputs --input-root "$INPUT_ROOT" --output "$OUTPUT"
python scripts/p07_neumann_global/campaign.py preflight --input-root "$INPUT_ROOT" --output "$OUTPUT" --workers "$CPU_WORKERS" --memory-budget-gib "$MEMORY_BUDGET_GIB" --worker-memory-gib "$WORKER_MEMORY_GIB" --memory-reserve-gib "$MEMORY_RESERVE_GIB"
python scripts/p07_neumann_global/campaign.py run --input-root "$INPUT_ROOT" --output "$OUTPUT" --workers "$CPU_WORKERS" --memory-budget-gib "$MEMORY_BUDGET_GIB" --worker-memory-gib "$WORKER_MEMORY_GIB" --memory-reserve-gib "$MEMORY_RESERVE_GIB"
python scripts/p07_neumann_global/campaign.py validate --input-root "$INPUT_ROOT" --output "$OUTPUT"
```

Repeat `preflight` or `run` with the same arguments and output folder after an interruption. `run-stage --stage STAGE` permits operational replay of one stage. `--max-units` with `run-stage` is for a bounded local recovery test, not a scientific run configuration. A failed or incomplete scientific gate remains an immutable completed result. Keep logs, caches, scratch, chunks, manifests, scheduler stdout/stderr and receipt inside the single `OUTPUT` folder. The remote handoff is a draft until the exact changed sources are published at a verified pinned revision; do not substitute the current dirty local HEAD.
