# Geometry-aware traced FCI return: global static qualification

Run the frozen candidate on **all owners** at N32, N48, and N64. This is a
research computation, not production integration. The primary result is the
positive divergence of one shared, coordinate-positive flux per canonical
owner face. Each nonconstant field must attain global midpoint-physical-volume
weighted L2 order >=1.8 on both refinement intervals, with the empirical
reference difference below 10% of the numerical error. A completed experiment
can fail these scientific gates; return that result without tuning it.

## Frozen method and boundary scope

`configuration.json` freezes the four scalar fields and the numerical policy.
The observations are reconstructed endpoint secants on m2 transverse footprints,
with magnetic-flux weights and actual traced endpoints. The tracer now uses
compiled CPU RK4 with 64 fixed substeps, integrating regular x/y positions and
connection length together at fourth order. Four seeds remain at quarter and
three-quarter coordinates in each transverse cell; they are subcell midpoints,
not vertices. A cubic potential in
regular x,y,unwrapped eta generates 19 nonconstant b.grad(p) basis functions.
Its observation matrix uses endpoint differences of potential moments. The q5
face functional is applied through a weighted SVD map. Numerical and exact
endpoint secants remain separate channels. Exact analytical q5, q9, and q11
face integrals are controls, never numerical interior observations.

The face-owned catalogue initially uses radial/angular halo 2, three source
planes, both directions, the nearest 96 rows plus at least three rows per
surrounding interval. Stable ties use global row IDs. Geometry-only expansion
uses halos 3,4,6,8 if the target is unresolved; field errors never select support.
This is a new canonical policy, distinct from historical patch-owned catalogues.
Historical replay and bounded comparisons are documented in `local_validation.md`;
`rk4_validation.md` records the integrator change and its bounded checks.

The canonical topology omits collapsed-axis faces and same-owner internal
faces. A positive-coordinate flux contributes positively to the lower owner
and negatively to the upper owner. All physical-wall owners remain present.
Wall loads are dynamic integrated flux data; the frozen manufactured fields
have exactly zero gradient at u=1, so their prescribed boundary flux is zero.
This does not establish that arbitrary scalar normal-Neumann data imply zero
parallel diffusive flux, or qualify the complete evolved physical-wall model.

A footprint is excluded from the **interior observation catalogue** if any seed
has an RK4 stage or endpoint outside 0<u<1. Each seed is masked independently;
valid trajectories are never retraced because another seed exits. Inactive lanes
use an interior dummy query and cannot contribute observations. Geometry and
scalar fields are never evaluated beyond the wall; no recipient owner is dropped. This conservative domain admissibility
policy does not attempt to return a clipped-leg boundary observation.

The independent reference is analytical face q11 divergence divided by summed
continuous q7 volumes. q9 faces/q5 volumes give an empirical reference difference,
not a rigorous bound. Numerical actions retain midpoint owner volumes. Preflight
also compares independent strong-volume q9 integrals on ordinary and wall owners.
Static accuracy does not certify structure, evolution, or production readiness.

## Source and immutable inputs

Use the pinned repository revision. `geometry_source.tar.gz` freezes the exact
127 Python/data source files from the local geometry dependency snapshot;
`geometry_source_manifest.json` verifies extraction. The snapshot intentionally
isolates this research computation from concurrent uncommitted package work.
The runner imports this extracted package before the installed DRBX package.
The scalar catalogue comes from tracked `scripts/q03_direct_campaign/frozen_mms.py`.
Do **not** run `build_bundle.py` remotely: it is a maintainer preparation tool.

`input_manifest.json` verifies eight immutable files, approximately 6.20 GB,
under the established `hsx_matched_global_parallel_v2` input root:

- `hsx_metric_d58d392545fd3917efeb83b6.npz`
- `mgrid_res2p5cm_180pln.nc`
- `geometry_artifacts/rlp_convergence_32_48_64_20260917/{N}x{N}x{N}/base_geometry.npz`
- `geometry_artifacts/rlp_convergence_32_48_64_20260917/{N}x{N}x{N}/rlp_topology.npz`

Here N is 32,48,64. Set `INPUT_ROOT` to the absolute existing extraction root;
locate it through the established remote data setup. Do not regenerate inputs.
The software environment and these immutable inputs may stay outside the result
folder. `verify` records their absolute location and verifies bytes and hashes.

## CPU execution and commands

Run from the repository root. Use a single node-local multiprocessing spawn
pool and one parent writer. Workers preload a resolution context once per
stage; geometry/maps are shared across all four fields. BLAS/OpenMP threads are
one. On Linux, each child binds to one CPU from the scheduler-provided affinity
mask before JAX initializes, preventing a full-node JAX thread pool per child.
The setup skill must verify the effective worker count and allowed CPU mask.
Compiled tracing uses the existing frozen JAX metric/MAKEGRID representations.
One compilation is needed for each batch shape/substep count; final short
batches are padded, and the on-disk JAX cache lives in the campaign folder. This is not a distributed runner. The remote setup skill chooses allocation,
affinity, workers, memory allowances, and walltime; GPUs are unnecessary.

Both `preflight` and `run` use the requested worker pool for tracing plus endpoint
reconstruction, shared-face maps and analytical face references, and continuous
volume integrals. Preflight uses finer units (4 trace rows, 8 faces or 8 raw
cells) so its bounded work can occupy remote CPUs. Global units are 24 rows,
64 faces and 32 raw cells. The two independent strong-volume preflight controls
also execute in worker processes. Resolutions and stage dependencies proceed
in order, using the allocation within each stage; do not launch a separate
writer per resolution. Final deterministic owner reduction, norms, manifest
checks and small bookkeeping remain in the parent process.

Create one uniquely named absolute `CAMPAIGN` folder. Keep logs, scheduler
stdout/stderr, temporary files, caches, provenance, checkpoints and the final
receipt beneath it. Save real files, not symlinks to external generated outputs.

```bash
mkdir -p "$CAMPAIGN"/logs "$CAMPAIGN"/tmp "$CAMPAIGN"/cache "$CAMPAIGN"/provenance
export TMPDIR="$CAMPAIGN/tmp" XDG_CACHE_HOME="$CAMPAIGN/cache/xdg"
export PYTHONPYCACHEPREFIX="$CAMPAIGN/cache/pycache"
export DRBX_CACHE_DIR="$CAMPAIGN/cache/jax" JAX_COMPILATION_CACHE_DIR="$CAMPAIGN/cache/jax"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_ENABLE_X64=true JAX_PLATFORMS=cpu
RUNNER=scripts/q_fci_return_campaign/campaign.py
COMMON=(--input-root "$INPUT_ROOT" --output "$CAMPAIGN")
CPU=(--workers "$WORKERS" --memory-budget-gib "$MEMORY_GIB"
     --worker-memory-gib "$WORKER_MEMORY_GIB" --memory-reserve-gib "$RESERVE_GIB")
python "$RUNNER" verify "${COMMON[@]}"
python "$RUNNER" preflight "${COMMON[@]}" "${CPU[@]}"
python "$RUNNER" run "${COMMON[@]}" "${CPU[@]}"
python "$RUNNER" validate "${COMMON[@]}"
python "$RUNNER" status "${COMMON[@]}"
```

Set all resource variables from the CPU allocation. The memory options cap the
effective worker count; they are planning allowances, not an operating-system
memory limiter. Record requested/effective workers and actual observed memory.
Capture each command's stdout, stderr, elapsed time and exit code under `logs`.
The built-in `run` performs all stages and validates them; there is no separate
analysis command to invent. `validate` computes the prescribed norms/orders and
flags. Return them without interpretation or additional experiments.

## Resume and validation

After interruption, repeat the interrupted `preflight` or `run` command with
**the same campaign folder, pinned code, configuration and immutable inputs**.
Completed batch payloads and receipts are SHA256 checked and skipped. Missing
units run afresh. Unreceipted partial payloads are overwritten atomically.
Worker count and memory settings may change. The exclusive writer lock prevents
a duplicate parent. Never delete receipts, edit identities, or bypass a hash
failure to force reuse. Source/configuration changes require a new campaign.

Preflight covers seven complete geometry-selected owners at each resolution,
including the original three tracks, axis, wall, inward neighbor and seam.
It must complete before `run` starts global work. Small bounded errors/orders
are diagnostics, not a new scientific stop gate. Operational validation checks
source/input identity, exact stage coverage, payload hashes, finite outputs,
constant response, incidence balance, saved action replay and norm consistency.
Numerically unresolved face targets stop the computation with failure evidence.

**RK4 outputs require a new campaign folder.** Do not resume, relabel, or mix
DOP853 trace, face, or action checkpoints with this revision. Preserve the old
folder as historical evidence.

The entire fresh global catalogue is prepared; earlier patch or failed-preflight
outputs are not reusable campaign checkpoints. Every trace has a stable global
ID, every face a canonical key, and every field uses the same map. Face chunks
save selected row IDs, weights, singular values, interval counts, support extent,
and target defects. Checkpoints use field/config/source identities, not mutable
roadmap text. Any numerical source change invalidates checkpoints conservatively.

## Returned artifacts

Return the absolute path of the single campaign folder, containing:

- `campaign.json`, `input_locations.json`, `environment.json`, `status.json`;
- `software/` extracted and verified geometry sources;
- `preflight/N{32,48,64}/` and `preflight_validation.json`;
- `global/N{32,48,64}/`: selections, plans, trace and face/volume chunks with
  receipts, memory-mapped row catalogue, progress, actions, summaries, completion;
- `validation.json`: prescribed global errors, orders, reference fractions and flags;
- all logs, scheduler/job records, launch provenance and `operational_receipt.md`.

Actions include numerical/exact-secant/oracle paths, signed axis contributions,
midpoint and continuous volumes, and both reference levels. Regional error
shares are diagnostics. No analysis plots or production promotion are requested.
Report failed/incomplete stages and exact errors rather than silently returning
partial data as complete.

## RK4 optimizations and limits

Tracing is compiled as a fixed-step loop over batches. The NumPy geometry path
for flux/reference work now computes only position, Jacobian, field projection,
field magnitude and determinant, omitting unused metric tensors/inverses and
inverse diagnostics. Unused trajectory dense output and the extra inverse-B
integral are gone. Endpoint fitting retains the same cubic support/score policy
but skips its unused quadratic diagnostic fit during campaign execution.
Face endpoint-moment matrices are evaluated in batches rather than one row at a
time. Observations and maps remain shared across fields.

The prescribed quadrature, four seeds, both directions, fields, support count,
basis, wall loads and scientific acceptance gates are unchanged. Older one-seed
N32 results are encouraging but do not establish global second-order convergence;
this revision does not reduce the seed count. Large elapsed-time speedups are
expected especially for former wall-retry batches, but throughput and compilation
cost must be measured on the allocated CPU system. Historical DOP853 timing
projections in `local_validation.md` no longer predict this runner's runtime.
