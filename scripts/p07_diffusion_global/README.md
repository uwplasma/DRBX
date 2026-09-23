# P07 perpendicular diffusion/polarization: portable CPU qualification

Run the frozen cubic shared-face candidate on every owner of the actual HSX
32³/48³/64³ artifacts. This is a static operator campaign. It does not certify an
elliptic solve, time integration, or production integration.

The candidate, field catalogue, source identities and immutable input hashes are
in `configuration.json`, `source_manifest.json`, and `input_manifest.json`.
The four fields are `phi_mms`, `Ti_mms`, `regular_neumann`, and
`mixed_eta_neumann`. Each is tested independently. The primary global norm uses
stored physical owner volumes; the target is a continuous physical-volume
average, matching the preceding bounded study. The runner also saves references
normalized by stored volumes for diagnosis. Reference finite differences retain
the frozen 2e-4 step and evaluator conventions.

## Numerical and execution contract

- Total-degree cubic regular-chart reconstruction, selection-v3 and existing
  owner midpoint observations; no field-dependent support selection.
- One shared face flux; raw-cell incidence is accumulated into owners. Collapsed
  radial-axis faces have zero flux without singular metric queries. Stored
  periodic endpoint faces contribute once to their respective endpoint cells.
- Every wall-reaching support uses a single wall-center constraint: exact phi
  value or exact physical-normal derivative for the other fields. Continuous
  geometry is queried at actual face quadrature nodes. This is the manufactured
  boundary-data baseline, not a sheath-law qualification.
- q3 face integration; independent q3 volume reference. Before the global run,
  seven complete-owner selections per resolution cover axis, wall, transition,
  agglomerated bulk, ordinary interior, theta seam and eta seam. References are
  compared at q3/q5/q7. Selections depend only on geometry/owner identity.
- Global physical-volume-weighted L2 order >=1.8 on both 32->48 and 48->64
  intervals for every field. Regional and maximum errors remain diagnostic.
  Bounded reference discrepancy and global order flags are returned unchanged;
  scientific failure does not cause automatic tuning or abort the computation.
- No global donor-by-quadrature tensor. Prepare/apply 128-face chunks and
  reference 32-cell chunks. Geometry needed by the operator is evaluated once
  per batch and shared across all four fields. No unrelated curvature/reference
  work, and no unused left/right fits.
- Node-local spawned CPU processes with one BLAS thread per worker, bounded
  pending work, one writer per output, and deterministic ordered assembly.
  Worker recycling is supported; default 128 chunks per process. Input hashes
  are checked by the controller before launch, rather than rereading the large
  MAKEGRID file for each process.

## Inputs and commands

Use a checkout of the pinned revision. `INPUT_ROOT` is an absolute data root
containing the nine relative files in `input_manifest.json`. The files are the
six geometry/topology arrays, the continuous-reference sidecar, metric cache,
and MAKEGRID file. No local prototype scripts, archived candidate answers, or
local reconstruction caches are required. The runner localizes only the
sidecar paths into its output folder; the original input remains unchanged.

Use **run drbx on perlmutter** to select the CPU allocation/environment,
`CPU_WORKERS`, `MEMORY_BUDGET_GIB`, `WORKER_MEMORY_GIB`, and
`MEMORY_RESERVE_GIB`. The last quantity covers the controller and other
allocation overhead. The budget determines effective process concurrency; it
is not an operating-system RSS limit. Bounded preflight records observed worker
memory. Local measurements and portability checks are in `local_validation.md`.

From the DRBX repository root, with one new unique `CAMPAIGN` directory:

```bash
python scripts/p07_diffusion_global/campaign.py verify-inputs \
  --input-root "$INPUT_ROOT" --output "$CAMPAIGN"
python scripts/p07_diffusion_global/campaign.py preflight \
  --input-root "$INPUT_ROOT" --output "$CAMPAIGN" --resolutions 32 48 64 \
  --workers "$CPU_WORKERS" --memory-budget-gib "$MEMORY_BUDGET_GIB" \
  --worker-memory-gib "$WORKER_MEMORY_GIB" --memory-reserve-gib "$MEMORY_RESERVE_GIB"
python scripts/p07_diffusion_global/campaign.py run \
  --input-root "$INPUT_ROOT" --output "$CAMPAIGN" --resolutions 32 48 64 \
  --workers "$CPU_WORKERS" --memory-budget-gib "$MEMORY_BUDGET_GIB" \
  --worker-memory-gib "$WORKER_MEMORY_GIB" --memory-reserve-gib "$MEMORY_RESERVE_GIB"
python scripts/p07_diffusion_global/campaign.py validate \
  --input-root "$INPUT_ROOT" --output "$CAMPAIGN" --resolutions 32 48 64
```

Capture stdout/stderr, allocation/job records and scheduler output inside
`CAMPAIGN`. Set `TMPDIR`, `DRBX_CACHE_DIR`, `XDG_CACHE_HOME`, and
`PYTHONPYCACHEPREFIX` there before starting Python. All campaign results,
receipts, scratch and caches belong beneath that same directory. Inputs and the
software environment may remain outside it. No output symlinks.

Resume the same commands and folder after interruption. Completed chunks have
atomic receipts, configuration/input/source identities, exact coverage, finite
array and SHA-256 checks. Interrupted chunks without a completed receipt are
recomputed. A mismatched/corrupt completed checkpoint fails explicitly; retain
its evidence for local follow-up. Keep one active controller per folder.

The maintainer-only `build_manifests.py` freezes the campaign; remote execution
uses the committed manifests unchanged. `replay_bounded.py` is a local
extraction validation helper requiring the old bounded-study outputs, and is
not part of the remote run.

## Return artifacts

Return the entire campaign directory, including:

- campaign/input/source manifest, localized reference sidecar and invocation
  history, environment/commit identity;
- N32/N48/N64 preflight selection, q3/q5/q7 arrays and comparisons;
- per-chunk face fluxes, oracle fluxes, constant actions, donor IDs/counts,
  expansion levels and constraint diagnostics; cell reference numerators and
  physical volumes, plus hash receipts;
- full owner actions/references, global/regional errors, orders and flags;
- execution receipts with requested/effective workers, timing, setup time and
  peak worker RSS; independent final assembly validation;
- logs, scheduler receipts, `last_exit.json`, and any failure evidence.

The remote worker runs computation and prescribed validation only. Scientific
interpretation, repairs and subsequent milestones are handled locally.
