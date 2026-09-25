# Complete HSX curvature with the common structured reconstruction

This is a research-only, static N32/N48/N64 qualification of the **complete**
P06 curvature operator. It uses the tracked
`scripts/perpendicular_structured/reconstruction.py` geometry-only service for
actual-owner values and all three logical gradients at q3 cell/face points.
There is no copy of the fit and no selection-v3 fallback. The service's
pointwise singleton/ringwise/coupled quartic rows and Dirichlet trace lift are
required to pass; unsupported support stops preparation. The centered action
uses the common rows. The `U` action uses the service's adjacent-cell anchored
side states in the original P06 characteristic matrix, absolute-action and
physical-wall characteristic solve. This is a **new** structured side-state
correction, not replay of P06's biased WLS jump.

## Fixed scientific contract

The frozen states are `corrected_frozen_mms`, `regular_chart_heldout`,
`homogeneous_dirichlet`, and `variable_dirichlet`. The first two preserve the
historical P06 analytic fields and are BC-consistency controls when evaluated
through the shared **Dirichlet** trace lift; they do not qualify a separate
Neumann implementation. The additional states keep positive thermodynamic
fields and prescribe phi traces of zero and a nonconstant Cartesian-regular
function, respectively. Both new phi fields have nonzero wall-normal
derivatives. The exact value and tangential trace derivatives enter at
application, while the normal derivative is reconstructed from interior owner
observations. Vorticity has no independent physical-wall condition in the
curvature characteristic algebra.

For each of density, Te, Ti and vorticity, retain material `M`, potential
remainder `R`, and total `M+R` separately, for centered and `U` actions,
including all three logical-direction splits. The unchanged pointwise algebra
is `M = A(U,B) (K·∇U)/B`, with the original P06 principal matrix `A`; `R`
contains the original coefficients multiplying `K·∇(phi+tau Ti)`. The
vorticity remainder is analytically zero and gets no fabricated order. `U` is
primary for density/Te/Ti and centered for vorticity. Every nonzero primary
component in every state must attain physical-owner-volume-weighted global L2
order at least 1.8 on **both** N32→N48 and N48→N64. Other actions, regions and
maxima remain visible diagnostics. A failed accuracy gate is saved unchanged.

Raw midpoint observations use immutable physical raw volumes and are projected
to complete owners. Numerical cell and face integrations use q3. The primary
continuous target integrates the exact fields and metric/curvature geometry
with **q5 physical-volume quadrature**, independently of reconstruction rows;
q3 target values from candidate cell chunks remain a diagnostic. Parallel
complete-owner preflight saves q3/q5/q7 exact-field volume controls and
matching sampled RMS/max discrepancies for each state, term and equation. The
10% q5-to-q7 sampled RMS screen is a bounded sensitivity check, **not** a
global uncertainty bound; global observed orders and reference-control flags
are reported separately. The known N64 MAKEGRID/interpolation hotspot can make
these controls unsettled. Do not silently substitute a favorable quadrature
or treat a reference-screen failure as a candidate failure.
The preflight selection is fixed before the new global run: every canonical
region and periodic seam, plus the archived N32/N48/N64 curvature/diffusion
reference-hotspot owners listed in `configuration.json`. All raw members of
each selected owner are integrated; the hotspot IDs are controls, not tuning
targets.

An independent surface/IBP formula was considered, but a surface-only flux is
not the curvature reference: the state-dependent principal coefficients,
`B`, `K` and the potential remainder generate legitimate interior divergence
terms under integration by parts. An independent IBP reference would need to
resolve those terms through the piecewise MAKEGRID derivative structure. This
campaign retains the direct exact-field volume target plus bounded quadrature
checks and does not claim full reference qualification from an incomplete IBP
identity. No evolved run, elliptic solve, energy gate or production promotion
is included.

## Inputs and execution

`input_manifest.json` hashes 47 immutable inputs: all canonical geometry and
topology, the historical continuous baseline used for the frozen omega
observation, the original continuum sidecar, metric cache, MAKEGRID, and all
eight sidecar artifact components. Existing immutable inputs can be outside the
campaign output folder. `verify-inputs` checks every byte and captures dynamic
hashes of the tracked campaign, shared service, curvature algebra and
reference sources. The sidecar is localized into the output folder, changing
only its paths to verified immutable inputs.

All generated observations, plans, chunks, receipts, reductions, cache,
scratch, logs, case arrays and summary belong under one output folder. A single
controller and output lock coordinate the campaign. Spawned CPU workers return
bounded chunks to the controller, which alone writes atomic checkpoints.
Preparation, complete-owner preflight and main face/cell/q5-reference work are
all CPU-parallel, chunked and resumable. Every checkpoint carries
source/input/configuration/prepare identity, exact index coverage and an array
hash; mismatched or corrupt data are rejected. Requested worker count is capped
by the supplied allocation memory budget, measured worker allowance and
reserve. BLAS/JAX workers use one thread; process recycling bounds retained
allocator memory. This is node-local parallelism.

From a clean checkout at the pinned handoff commit:

```bash
export INPUT_ROOT=/absolute/path/to/immutable/HSX-input-root
export CAMPAIGN=/absolute/path/to/one/new/downloadable/p06-structured-campaign
export CPU_WORKERS=...            # chosen by run drbx on perlmutter
export MEMORY_BUDGET_GIB=...     # available allocation memory
export WORKER_MEMORY_GIB=...     # measured worker RSS allowance
export MEMORY_RESERVE_GIB=...    # controller/system reserve
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export TMPDIR="$CAMPAIGN/scratch" XDG_CACHE_HOME="$CAMPAIGN/cache" PYTHONPYCACHEPREFIX="$CAMPAIGN/cache/pycache" DRBX_CACHE_DIR="$CAMPAIGN/cache/jax" JAX_COMPILATION_CACHE_DIR="$CAMPAIGN/cache/jax"
mkdir -p "$CAMPAIGN/scratch" "$CAMPAIGN/cache/jax" "$CAMPAIGN/logs"
python scripts/p06_structured_global/campaign.py verify-inputs --input-root "$INPUT_ROOT" --output "$CAMPAIGN/results"
python scripts/p06_structured_global/campaign.py preflight --input-root "$INPUT_ROOT" --output "$CAMPAIGN/results" --workers "$CPU_WORKERS" --memory-budget-gib "$MEMORY_BUDGET_GIB" --worker-memory-gib "$WORKER_MEMORY_GIB" --memory-reserve-gib "$MEMORY_RESERVE_GIB"
python scripts/p06_structured_global/campaign.py run --input-root "$INPUT_ROOT" --output "$CAMPAIGN/results" --workers "$CPU_WORKERS" --memory-budget-gib "$MEMORY_BUDGET_GIB" --worker-memory-gib "$WORKER_MEMORY_GIB" --memory-reserve-gib "$MEMORY_RESERVE_GIB"
python scripts/p06_structured_global/campaign.py validate --input-root "$INPUT_ROOT" --output "$CAMPAIGN/results"
```

On interruption, rerun the identical `preflight` or `run` command against the
same output and immutable sources. The controller validates existing chunks
before reusing them. Never mix revisions, change numerical settings to force
reuse, or bypass checks. `validate` checks exact full raw-cell, face and q5
reference coverage and numerical invariants. The final `summary.json` reports
`global_order_pass`, `bounded_reference_controls_passed`, and
`complete_qualification_passed` separately. Machine-generated numerical
results are for later local scientific review.

## Optimized execution and existing campaigns

Curvature-only geometry preparation computes J/B/K with the original K
derivative stencil, omitting unused diffusion, parallel-divergence and omega
work. Identical metric queries are reused within a batch; singleton weights
use bounded exact-coordinate caches; face states skip gradient contractions.
Quadrature, fields, support policies and acceptance thresholds are unchanged.

The tested source chain from `551ebbd7919ad3639499f35941118cfe5244bd1f`
(whose P06 sources are identical at `2ad730c8`) supports an explicit
`adopt-optimization` command with the original `--input-root` and `--output`.
Stop the old controller/workers first and update the same checkout location.
The original manifest and checkpoint identities remain the numerical lineage.
`optimization_execution.json` separately pins the new execution sources/commit;
new chunks include that certificate's provenance. The tracked shared release
policy requires the exact tested baseline/optimized sources and unchanged
configuration and inputs. Existing hash, coverage and input checks remain in
force. Run `preflight`, `run`, then `validate` after adoption; valid work is
reused. Never manually rewrite source identities or checkpoint receipts.
