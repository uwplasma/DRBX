# P08 step 1: unit-parallel row artifact + replay gate campaign

## What this package does

Builds the per-grid, schema-v3 row artifact (`drbx.stencils.artifact`,
`NeumannRowChunk` tagged by `(request, radial_degree)`) and replays it,
**unit-parallel**, against the six frozen accepted-campaign oracles: P05,
P05N (both catalogues: `frozen_v1`/`05be9063` and `upwind_v1`/`43250ccf`),
P06N, P06 legacy (with its seam double count), P07, P07N. See
`work/p08_step1_consolidation_design_20260928/design.md` for the design this
implements, and its "Replay gate" section (5) in particular.

This used to replace a single-process, whole-artifact-in-memory
`scripts/p_shared/replay.py` reference implementation with a
`p_shared.runner.run_stage` unit pipeline: every unit opens exactly the one
artifact build-chunk file it needs (never the whole artifact), computes a
small, sparse, unnormalized owner-space partial sum, and a reduction stage
sums those partials and applies the same Tier-B/pointwise comparison
(`p_shared.replay_support.compare_owner_term`/`compare_pointwise_only`).
`replay.py` has since been retired (deleted): it was a monolithic, never-
run-to-completion full-grid reference, and every one of its per-campaign
functions was already dead code once this unit-parallel path existed. The
small set of generic helpers it defined that this path still needs (the
comparison core, the campaign `Environment`, a few per-campaign trace
helpers) now live in `scripts/p_shared/replay_support.py`; see that
module's own docstring.

## Stages

1. **artifact** -- `p_shared.build_artifact.run_full_build` (through this
   package's one small adapter, `campaign._build_artifact` -- update only
   that function if `build_artifact.py`'s entry point changes; this package
   never edits `build_artifact.py`/`runner.py`/`provider.py`). Produces
   `<output>/artifact/N{n}/` (manifest, census, geometry, chunked rows) --
   geometry chunk sizes (4096/4096) are numerical parameters (the
   finite-difference-derived P06 K / P07 divergence fields shift by up to
   ~4.5e-9 relative when a metric-evaluator batch boundary moves), fixed in
   `configuration.json` and hashed into the identity, never exposed as a
   resource flag.
2. **cells / faces / p07** (replay units) -- `p_shared.replay_units
   .compute_cells_unit`/`compute_faces_unit`/`compute_p07_unit`, one per
   artifact build chunk (`p_shared.replay_units.artifact_plan_units` reads
   the artifact's own `plan.json` so a replay unit's `(start, stop)` is
   bitwise the build unit's own). Each unit's own output is a small,
   sparse `(touched_owner_ids, partial_value)` file (never a dense
   `(owners, ...)` array) -- see `replay_units`'s module docstring, "Owner
   -space additivity", for why summing these across every unit reproduces
   the full-grid scatter bitwise.
3. **reduction** -- `p_shared.replay_units.reduce_grid`: sums every unit's
   partial, finalizes the (fixed- or accumulated-denominator) owner-level
   arrays, and applies `p_shared.replay_support`'s comparison/report
   functions. Writes `<output>/replay/N{n}/replay.json` + `report.md`.
4. **validate** -- requires a passing `preflight.json` (same identity) and
   every grid's `replay.json`, writes `validation.json`. Per this project's
   own convention (see e.g. `p06n_field_derived_global`), this returns pass
   /fail data unchanged; the user decides acceptance from the reported
   Tier-B ratios.

## Row lookup: everything read from the artifact, nothing rebuilt

Every unit reads its own R1/R2/R3 (`PointRowChunk`), R4
(`IntegratedRowChunk`), *and* Neumann companion (`NeumannRowChunk`) rows
straight from the artifact -- **never rebuilt** -- looking Neumann rows up
by `(request, entity_id, quad_node)` from that same build chunk's
`rows/neumann_<stage>_<i>.npz` file (`load_neumann_chunk_for_unit`/
`row_index_from_neumann_chunk` in `replay_units.py`). This replaced an
earlier on-the-fly rebuild (`p_shared.replay_support.NeumannSource` +
`prepare_neumann_point_rows`, mirroring the artifact builder's own call) that
had a real bug: it rebuilt at `PointRows.trace_target_points`, the wall
-projected anchor the Dirichlet lift's own boundary correction uses, instead
of the row's true query point -- see the P08 step-1 task report for the
root-cause finding and its bitwise-verified fix. `NeumannSource`/
`prepare_neumann_point_rows` are kept only for the bounded preflight check
below, which rebuilds at the true query point on purpose, to cross-check the
stored rows.

## The bounded preflight: rebuild a few rows, compare bitwise

`p_shared.replay_units.neumann_rebuild_compare_check` rebuilds a small,
explicit sample of Neumann rows at their true query point (derived directly
from geometry -- the raw midpoint for a cells/R1 row, the census face's own
q3 quadrature node for a faces/R2-R3 or p07/R4 row -- never from
`trace_target_points`) and compares them bitwise against the artifact's
stored rows. This is the only place in the package that still calls
`prepare_neumann_point_rows`; it is never part of the per-grid replay
itself.

## Two domain bugs this found (fixed directly in `replay_units.py`)

Building small real units and replaying them against a real artifact (this
package's own local validation, see below) surfaced two cases where a naive
full-grid domain selection raises `ValueError` against a real artifact --
both are small, documented fixes applied directly in `replay_units.py`'s
own per-campaign blocks (not inherited from elsewhere):

1. **P05 live jump**: `stencil_builder.p07_row_selection`'s domain includes
   the collapsed r=0 census face (1024 rows at N32), for which R2/R3 never
   build a row (confirmed empirically). Fixed by excluding
   `census.collapsed_r0` from that domain -- the same convention
   `build_artifact.py` already uses for P07's own family-0 integrand (zero).
2. **P06-legacy face domain**: `~census.collapsed_r0` alone also keeps the
   periodic theta/eta alias slots (2048 rows at N32), which the artifact
   never stores a row for either. Fixed by also excluding
   `census.legacy_alias_slots` -- verified empirically that no alias-slot
   row ever satisfies the "double the seam" `double_mask` condition, so
   this changes nothing about *which* rows get doubled.

## Inputs

`input_manifest.json` is an identical copy of
`p07n_field_derived_global/input_manifest.json`: the same 17 immutable
files (the canonical continuum reference sidecar, the RLP
`32x32x32`/`48x48x48`/`64x64x64` geometry pairs, the metric-cache `.npz`,
the mgrid `.nc`, and the frozen `hsx_fci_64x64x64` artifact directory), so
this package's `--input-root` reuses exactly the same immutable input root
the P05N/P06N/P07N remote campaigns use (e.g. a `/pscratch/.../hsx-midpoint
-inputs-<stamp>` root on a remote allocation). `verify-inputs` hash
-verifies every one of those 17 files under `--input-root` (size + sha256),
failing with the first missing/changed path, before anything else runs.

The geometry itself is read from `--input-root` + `configuration.json`'s
`geometry_input_subdir` (`geometry_artifacts/rlp_convergence_32_48_64_20260917`)
-- never from a local absolute path or a `work/` scratch folder (only
`oracle_default_paths`, remapped by `--oracle-root`, and this package's own
`work/p08_step1_campaign_dev_*` **output** folders are workspace-`work/`
-relative by design; every *input* path resolves under `--input-root`).

**Sidecar localization.** `configuration.json`'s `canonical_sidecar_relative`
names the canonical continuum reference sidecar's own input-root-relative
path (`input_manifest.json`'s own entry for it). `verify-inputs` (via
`campaign.verify`/`campaign.localize_sidecar`, the same approach and
function semantics `p06n_field_derived_global/campaign.py`'s
`localize_sidecar` and `p07n_field_derived_global/campaign.py`'s equivalent
inline code use) reads that canonical file and rewrites its
`metric_cache`/`makegrid`/`artifact` paths to point under `--input-root`,
writing the result to `<output>/localized_sidecar.json` and recording its
sha256 as `localized_sidecar_sha256` in `campaign_manifest.json`. Every
later command (`preflight`, `run`, `validate`, `run-stage`) resolves the
sidecar through `campaign._sidecar_path`, which always returns
`<output>/localized_sidecar.json` -- never a local `work/` scratch path
(the previous, pre-remote-ready version of this package pointed at
`work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json`, a
sidecar localized to one laptop's own paths; that is no longer read).
`verify-inputs` refuses (raises) if the on-disk localized copy would
change relative to a freshly-recomputed one (the canonical source changed,
or the localized copy was hand-edited), or if `campaign_manifest.json`'s
own recorded `localized_sidecar_sha256` no longer matches the on-disk
localized copy.

A sidecar localized by this code from `--input-root "/Users/yxie/Desktop
/HSX drbx"` was checked byte-identical to the old, pre-remote-ready
`work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json`, and
`ref._metric` evaluated bitwise identically from both at several sample
points (see the task report for the exact check).

## Identity, lock, resume

One lock per output folder (`campaign.lock`, advisory, `fcntl.flock`).
Identity (`campaign.verify`) covers: `configuration.json` (every numerical
policy parameter, including the geometry chunk sizes), `input_manifest.json`
(every immutable input file's own sha256), every listed source file's hash
(`campaign.SOURCE_FILES` -- this package's own modules, `p_shared`'s shared
modules, the frozen package builders/kernels this design pins as oracles
(including `build_artifact.py`'s own hash even though this package never
imports its private internals), and the committed `oracle_manifest.json`
itself), the git commit, and the **committed oracle manifest**'s own content
(every oracle file's recorded sha256 -- see below; never a manifest rebuilt
from local file state). The localized
sidecar's own sha256 is recorded alongside the identity in
`campaign_manifest.json` (see "Inputs" above) and checked on every
subsequent `verify-inputs` call, but is not itself hashed into `identity`
-- the same convention `p06n_field_derived_global`/`p07n_field_derived_global`
use. `run`/`validate` refuse to proceed without a `preflight.json` recorded
under the same identity. Every stage is resumable through
`p_shared.runner`'s chunked-checkpoint convention (a unit's chunk file +
JSON receipt, both required, sha256-verified).

## Oracle inputs (`p_shared/oracle_manifest.py`)

Lists **exactly the files the replay reads** per campaign and grid (never
"a whole chunk folder" beyond the one campaign -- P05's own "upwind" oracle
-- that genuinely has no single merged file on disk).

**The manifest is committed, not rebuilt.** `scripts/p08_step1_global/
oracle_manifest.json` is a *frozen* copy of this manifest (extracted once
from a `pack-oracles` tarball -- see `campaign.committed_oracle_manifest`),
checked in and hashed into `campaign.SOURCE_FILES`/the campaign identity.
`verify()`/`verify-inputs` load this committed file and hash-verify every
listed file under `ORACLE_ROOT` -- **`--oracle-root` if given, else
`--input-root`** (which suits local use, where the workspace root is both)
-- failing, naming every missing/mismatched file (not just the first),
without ever touching `--input-root`'s own (possibly oracle-file-free, e.g.
the 17-file immutable `INPUT_ROOT`) tree to *build* a manifest. This fixes
two remote-portability bugs the old "rebuild on every `verify()` call" design
had: (1) the old rebuild path resolved every path relative to `<repo>.parent`
(an assumed local workspace layout), which crashed (`relative_to`
`ValueError`) whenever the checkout did not sit inside that same workspace
(a remote run, or a `git archive` export elsewhere); (2) rebuilding against
`--input-root` alone silently produced an empty/all-missing manifest on a
remote run, since the remote's immutable `INPUT_ROOT` holds none of the
17,739 oracle files. Every later command (`preflight`, `run`, `validate`,
`run-stage`) resolves oracle paths under the same `ORACLE_ROOT` (via
`campaign.oracle_paths`) and is refused (a new `--output` folder is
required) if `ORACLE_ROOT` or the committed manifest's identity changed
relative to a previous run recorded in that output's `campaign_manifest.json`
(which records `oracle_root` and `oracle_verified` alongside the identity).

A *fresh* build (`campaign.oracle_manifest_for`, backed by
`om.build_manifest(..., root=...)` -- `root` is always passed explicitly,
never guessed) still exists, used only by `pack-oracles` (below) and tests;
nothing on `verify()`'s own runtime path calls it any more.

**Oracle delivery.** Rather than point a remote run at each frozen
campaign's own `work/` folder over the network, `pack-oracles` tars exactly
the manifest's files (plus the manifest itself, as `oracle_manifest.json`)
into one tarball, preserving every file's workspace-relative path; the
remote run's `--oracle-root` is wherever that tarball gets extracted.
Before packing, `pack-oracles` asserts that a manifest freshly built against
`--input-root` still equals the *committed*
`scripts/p08_step1_global/oracle_manifest.json` over the requested
campaigns/grids slice -- refusing to pack (and deliver) a tarball
`verify()`'s committed-manifest hash checks would not recognize.
`pack-oracles` was tested only on a tiny synthetic file set (not the real,
~9 GB oracle data) -- see the task report for what was and was not run for
real; the committed manifest itself was checked to equal a fresh local
build byte-for-byte (every path, size, sha256) as part of the P08 step-1
remote-portability fix.

## Preflight

Bounded (seconds to a few minutes per grid), two modes:

- **A grid with an existing (full) `geometry.npz`/`census.npz`** (true of
  N32 in this local workspace, where a prior serial build already produced
  them): builds a handful of *real* artifact build units --
  boundary/wall-containing tail units of each stage (cells/faces/p07),
  auto-selected via `bounded_build.boundary_and_family_units`, since the
  census's own row order (`radial_block_then_theta_then_eta`) puts every
  wall/boundary layer and every P07 family (including 1/2/4) at the *end* of
  each domain -- runs those units through the real
  `p_shared.replay_units` pipeline and `reduce_grid`, plus the bounded
  Neumann rebuild-and-compare check.
- **A grid with no existing geometry** (true of N48/N64 in a from-scratch
  run, and a **hard local constraint**: never run a full-grid geometry
  stage locally): `bounded_build.light_subset_requests` builds
  `GeometryArrays` and R1-R4 rows **only for a small, explicit raw-id/
  census-row subset**, in memory, via `drbx.stencils.builder`'s own
  functions directly -- no chunked artifact, no `build_artifact.py`
  geometry stage at all. This mode runs the bounded Neumann-rebuild
  determinism check and structural checks (row/family counts, residual/
  condition bounds) against that light subset. It **also** runs the real
  `p_shared.owner_closure` oracle-comparison gate at full strength (not
  restricted to structural sanity): every frozen campaign's oracle arrays
  are saved per-grid (N32/N48/N64 -- see the oracle manifest below), and
  `owner_closure.build_owner_rows` computes `GeometryArrays` only for the
  selected dozen owners' own raw cells/incident faces directly from the
  provider (batched at the production geometry stage's own 4096-row chunk
  size -- `owner_closure._owner_geometry_arrays`), independent of the
  separate light subset built above and never touching this grid's
  (nonexistent) production `geometry.npz`. See the task report for the
  N48/N64 validation tables this produced.

## CLI

```
python -m p08_step1_global.campaign verify-inputs --input-root <workspace> --output <dir> [--oracle-root <dir>]
python -m p08_step1_global.campaign preflight      --input-root <workspace> --output <dir> --resolutions 32 [48 64]
python -m p08_step1_global.campaign run            --input-root <workspace> --output <dir> --resolutions 32 48 64 \
    --workers 4 --memory-budget-gib <B> --worker-memory-gib <W> --memory-reserve-gib 1.0
python -m p08_step1_global.campaign validate       --input-root <workspace> --output <dir>
python -m p08_step1_global.campaign run-stage      --input-root <workspace> --output <dir> --n 32 --stage artifact
python -m p08_step1_global.campaign run-stage      --input-root <workspace> --output <dir> --n 32 --stage faces
python -m p08_step1_global.campaign pack-oracles   --input-root <workspace> --output <dir> --pack-output <tar path>
```

Resource flags (`--workers --memory-budget-gib --worker-memory-gib
--memory-reserve-gib`) follow the same convention as
`p05n_field_derived_global`/`p06n_field_derived_global`: the effective
worker count is `min(workers, (memory_budget_gib - memory_reserve_gib) //
worker_memory_gib)` when both memory flags are given, else `workers` as
given.

## Memory target

Design section 6's target is ≤2 GiB per worker at N64. This package's own
local validation (N32, a bounded real-unit subset) measured worker RSS well
under that (see the task report for the exact figures); N64's per-unit
memory is extrapolated from the N32 measurement using the artifact's own
n²/n³ row-count scaling, not measured directly (a full N64 build/replay was
explicitly out of scope for local validation).

## Tests

`tests/test_p_shared_replay_support.py` (the generic Tier-B/pointwise-cap
core, unchanged in behavior) plus `tests/test_p08_step1_campaign.py` (CLI,
identity, preflight gating, resume, reduction on synthetic units, the
`input_manifest.json` byte-identity check, and sidecar-localization
identity/guard-rail tests against a synthetic workspace -- no multi-GB
fixtures, no real oracle data). `tests/test_p_shared_owner_closure.py` and
`tests/test_p_shared_replay_units.py` cover the shared owner-closure/replay
machinery this package's `preflight`/`run` call into.
