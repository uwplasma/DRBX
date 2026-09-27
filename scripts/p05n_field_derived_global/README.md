# P05N field-derived global campaign

Global (all-owner, all-resolution) qualification of the P05 E×B-like bracket
under physical-normal Neumann rows, batched for tractable wall-clock cost.
Frozen catalogue: `p05n_catalogue.json`, vendored byte-identically in this
package directory (sha256 recorded in `configuration.json`).

This directory is fully portable/self-contained (everything it reads at
runtime is either tracked in this package or resolved through `--input-root`;
nothing under a local scratch `work/` directory is ever read by
`campaign.py`, `preflight.py`, or any of the numerics modules -- see "Inputs"
below and `tests/test_p05n_field_derived_campaign.py`'s
`test_package_sources_never_reference_a_local_work_directory`). It has two
layers:

- **The oracle** (do not modify its numerics): `fields.py` (field values,
  excluding `zero_trace_generator`, which is new here), `rows.py`,
  `operator.py`. Exercised per-owner, one (owner, pairing) at a time.
  `tests/test_p05n_neumann_rows.py` covers it.
- **The batched campaign** (this task's deliverable): `core.py` (batched
  kernels + the frozen 10-role catalogue table), `campaign.py` (resumable CLI,
  mirrors `scripts/p07n_field_derived_global/campaign.py`), `preflight.py`
  (small-scale, non-chunked gate checks against the oracle and a vendored
  replay fixture), `preflight_fixtures/` (the vendored fixtures + the local
  extraction script that built them). `tests/test_p05n_field_derived_campaign.py`
  covers it.

## Why batching, and how it stays bit-faithful to the oracle

Profiling of the per-owner path (`operator.owner_centered`/`owner_jump`)
showed ~0.1-0.4s per (owner, pairing) call, ~75% of it in `ref._metric` being
called on a handful of points at a time, plus the same target cell's
reconstruction row being rebuilt from scratch for every one of the ~10-12
catalogue pairings that touch it.

The key fact that makes batching safe without touching any numerics: every
row object (`StructuredReconstruction.rows`'s `PointRows`, and
`prepare_neumann_point_rows`'s `NeumannPointRows`) is **already
field-independent geometry** -- `PointRows.apply(owner_values, trace)` and the
Neumann row's `value @ owner_values[donor_ids] + boundary_value @ g_N`
arithmetic are already generic over a trailing "fields" axis (confirmed by
`perpendicular_structured/reconstruction.py`'s `PointRows.apply`, which takes
`owner_values` of shape `(owners, fields)` and `trace(q) -> (v (Q,F), g
(Q,3,F))`). So the campaign never reconstructs a *different* row per field or
per pairing: it builds **one** row per target point (or once per face, over
that face's quadrature points -- exactly the oracle's own granularity), then
applies **every** catalogue field to it in a single matrix product
(`core.batched_cell_values` / `batched_face_common_gradient` /
`batched_side_values`, each returning both the Dirichlet-role and
Neumann-role reconstruction for the full field axis at once). `ref._metric`
is called once per chunk (thousands of points), not once per (owner,
pairing). `direct_pair_actions` and `p05_scalar_face_jump` were already
vectorized over an arbitrary list of (generator, transported) index pairs
into a shared trailing field axis, so all 10 catalogue pairings (plus their
10 matched-Dirichlet-diagnostic counterparts) come out of **one** call per
chunk.

Net effect: instead of ~12 independent row constructions per target (one per
pairing) plus ~12 tiny `_metric` calls per owner, there are 2 row
constructions per target (Dirichlet-role and Neumann-role) applied to a
6-column field matrix, and one `_metric` call per whole chunk. See "Timing
measurement" in the task report for the measured speedup.

## The frozen 10-role catalogue table (`core.py`)

`core.ROLES` maps a role name (e.g. `"b1_N"`, `"b1_D"`) to `(physical field
name, boundary condition)`. `core.PAIRINGS` maps each of the 10 catalogue
pairing names (`a_main1`, `a_main2`, `a_heldout`, `a_control1`, `a_control2`,
`b_main1`, `b_main2`, `b_heldout`, `b_control1`, `b_control2`) to
`(generator role, transported role)`, matching
`p05n_catalogue.json`'s `pairings` block field-for-field.
`core.DIRICHLET_COUNTERPART` gives every role's Dirichlet-role twin (used to
build the matched-Dirichlet diagnostic D for every pairing, exactly as the
bounded construction-verification run's own `DIRICHLET_COUNTERPART` does for
its smaller catalogue). `core.R_PAIR_INDEX` indexes
the underlying physical fields directly (BC-role-independent): R depends only
on the physical field pair, so e.g. `R("a_main1") == R("b_main1")` exactly,
both being `(field_b1, field_e3)`.

### Constant role decision

The frozen catalogue does not suffix `constant` with `:dirichlet` anywhere,
even though it appears as both a generator and a transported field across
both pairings. Per the task spec's explicit latitude ("pick one, state it in
the README"), **`constant` is treated as Dirichlet-role in every pairing it
appears in**, mirroring the bounded construction-verification run's
`CATALOGUE_ROLES = {..., "constant": ("constant", "dirichlet")}` exactly (not
the alternative of switching it to Neumann-role specifically when it is the
transported field in an `a_neumann_f_neumann` control). This keeps a single,
already-validated role table and satisfies the required "constant action
<= 1e-8" check (see `configuration.json`'s
`constant_action_absolute_maximum`).

## Stages

Three stages (simpler than P07N's six, because P05N's reference R is a cheap
analytic point-bracket computed alongside the centered action, not a
separate MMS residual stage):

1. `observations`: live owner-averaged raw-volume-weighted values of the six
   physical fields (`field_b1`, `field_e3`, `field_e12`, `heldout_field_b2`,
   `constant`, `zero_trace_generator`).
2. `raw`: per raw-midpoint chunk, the centered N action (candidate 1) and its
   matched-Dirichlet diagnostic D, for all 10 pairings at once, plus the
   analytic reference R, antisymmetry, exact-input defect, and Neumann
   condition/residual bookkeeping.
3. `faces`: per global owner-boundary face (from
   `p07_combined_global.topology.census`, the same lo/hi convention
   `operator.owner_boundary_faces` documents), the live jump N and D for all
   10 pairings, plus per-face-kind (physical wall / radial n-1 / transverse
   last-two-layers) zero-jump bookkeeping.

Candidate 2 (`direct_centered_plus_live_U_minus_A`) is `raw.N + faces.N`
(and its D diagnostic likewise), assembled at `reduce-stage`/`run`.

`O ≡ R` for this pointwise operator is a structural fact (the exact-input
action equals R by code identity, checked via `exact_input_defect` at every
raw chunk), not a separately stored global array -- matching the task spec.

## Inputs

Everything the campaign reads at runtime comes from exactly one of two
places: this tracked package directory, or `--input-root` (resolved the same
way, from the same manifest, as `scripts/p07n_field_derived_global`):

- `input_manifest.json` is a byte-identical copy of P07N's own manifest: the
  same `geometry_artifacts/rlp_convergence_32_48_64_20260917/{n}x{n}x{n}`
  base/topology npz files per resolution, the pre-built HSX metric-cache npz,
  the raw makegrid `.nc` file (hashed by `verify-inputs` for parity with
  P07N's own contract even though it is never read at this stage -- the
  metric cache already makes it unnecessary), and
  `DRBX/work/perpendicular_second_order_hsx_p01_p03/continuous_reference_sidecar.json`
  (a path *inside the tracked DRBX input layout*, resolved through
  `--input-root`, not a local scratch directory). Any remote `INPUT_ROOT`
  that already passes P07N's `verify-inputs` passes P05N's unchanged.
- Geometry resolves via `input_root/geometry_artifacts/rlp_convergence_32_48_64_20260917/{n}x{n}x{n}`
  (`perpendicular_structured.reconstruction.load_context`, the same
  `p07_combined_global.kernels.configure`/`.load` P07N itself uses) and the
  global face census via `p07_combined_global.topology.census` on the same
  root.
- The continuum reference is built exactly as
  `scripts/p07n_field_derived_global/campaign.py`'s `verify()` builds it:
  read the raw sidecar from `input_root`, patch its `metric_cache`/`makegrid`/
  `artifact` paths and `metric_query_batch_size` to `input_root`-relative
  absolute paths, write the result to `OUTPUT/reference_sidecar.json`, and
  re-verify it is byte-identical to that copy on every subsequent invocation
  (`campaign.verify`/`localize_sidecar`). Verified locally to give a
  bit-identical `ref._metric` to the previously used, separately-localized
  sidecar (compared directly on a batch of points; see the task report).
- The ~54-owner preflight selection and the accepted-P05-replay fixture are
  vendored byte-identically/derived into `preflight_fixtures/` (tracked; see
  "Preflight" below) -- never read from a local `work/` directory at runtime.

`--input-root` is the HSX-drbx workspace root locally (the parent of `DRBX/`)
or the equivalent remote `INPUT_ROOT` on Perlmutter.

## Running it

From `DRBX/scripts`, with `DRBX/scripts` (not this package's own directory)
on `sys.path` -- e.g. `python -m p05n_field_derived_global.campaign ...` from
`DRBX/scripts`, or any driver that does `sys.path.insert(0, ".../DRBX/scripts")`
before `from p05n_field_derived_global import campaign`. Never import this
package's own directory first: it contains `operator.py`, which would shadow
the stdlib `operator` module for anything imported after it.

The runner is a **single controller process with one node-local
`ProcessPoolExecutor`** per `run-stage`/`run` invocation (no MPI, no
cross-node coordination): `--workers` caps the pool size, `--memory-budget-gib`
and `--worker-memory-gib` cap the *effective* worker count
(`min(--workers, floor((--memory-budget-gib - --memory-reserve-gib) / --worker-memory-gib))`,
`--memory-reserve-gib` defaults to 1.0), and each worker enforces its own
`--worker-memory-gib` cap at every chunk (`MemoryError` if exceeded). Chunks
are content-addressed NPZ + JSON receipts under `OUTPUT/chunks/N{n}/{stage}/`,
bound to a campaign identity digest (configuration + input manifest + every
package source file's sha256, including `preflight_fixtures/` and this
README, + the input-root manifest + the git commit); resuming re-verifies
every existing chunk's hash and re-runs only what's missing or invalid.

Remote command sequence (same shape as P07N's own, one Perlmutter node,
`INPUT_ROOT` the already-verified P07N input root):

```
python -m p05n_field_derived_global.campaign verify-inputs --input-root INPUT_ROOT --output OUT
python -m p05n_field_derived_global.campaign topology      --input-root INPUT_ROOT --output OUT --resolutions 32 48 64 --workers 1 --memory-budget-gib 4 --worker-memory-gib 2
python -m p05n_field_derived_global.campaign plan          --input-root INPUT_ROOT --output OUT --resolutions 32 48 64
python -m p05n_field_derived_global.campaign preflight     --input-root INPUT_ROOT --output OUT --resolutions 32 48 64
python -m p05n_field_derived_global.campaign run           --input-root INPUT_ROOT --output OUT --resolutions 32 48 64 --workers 3 --memory-budget-gib 16 --worker-memory-gib 2.5 --memory-reserve-gib 1
python -m p05n_field_derived_global.campaign validate      --input-root INPUT_ROOT --output OUT --resolutions 32 48 64
```

(`run` requires a complete, identity-matching `preflight.json` first. Any
individual stage can instead be driven directly with `run-stage --stage
{observations,raw,faces} [--max-units K]` and `reduce-stage --stage {...}`,
e.g. to interrupt/resume a long stage or to smoke-test a handful of chunks.)

`OUTPUT/` after a full run contains: `campaign_manifest.json` (identity,
`input_root`, git commit, source hashes), `reference_sidecar.json` (the
localized sidecar), `N{n}.topology.npz`/`.json`, `plan.json`,
`preflight.json`, `chunks/N{n}/{stage}/*.npz`+`.json` (one receipt per
chunk), `N{n}.owner_values.npz` + `.observations.reduction.json`,
`N{n}.raw.npz`/`.raw.reduction.json`, `N{n}.faces.npz`/`.faces.reduction.json`,
`summary.json` (the global order gate and per-pairing/candidate stats),
`validation.json`, plus `progress.json`/`executions/*.json`/`invocations/*.json`
bookkeeping and a `.campaign.lock` file.

**Wall lattice.** Every Neumann wall node is exactly
`(1, t.centers[1][i], t.centers[2][k])`, so `core.WallLattice` evaluates the
physical normal and every field's g_N once per grid (one metric call on n²
points) and looks them up by exact coordinate. Evaluating them per patch and
per target dominated the near-wall cost. Lattice values agree with direct
evaluation to about 2e-14 (batch-size roundoff of the metric evaluator), and
the face stage makes one metric call per chunk, not one per face.

**Cardinal memo.** `StructuredReconstruction._tensor` evaluates the frozen
trigonometric cardinal `p07_combined_global.kernels.cardinal(nodes, [theta])`
once per point, layer and eta plane, and identical arguments recur
constantly. `core.memoize_single_target_cardinal()` caches single-target calls
by the exact bytes and dtype of both arguments. It is installed on import, in
this process only; the shared source is unchanged. Face chunks with and
without it were checked once to be bitwise identical on every face family
(interior radial/theta/eta, transition, near-wall).

**Cost (measured).** Local N32 smoke run, 6 workers:
- the campaign preflight took about 2 minutes;
- observations, raw and faces took about 35 CPU-minutes (faces 1,969 CPU-s,
  raw 150 CPU-s), or about 8 minutes wall.

Volume faces dominate and scale with n³, so all three grids are projected at
roughly 7 CPU-hours. The remaining hot spot is the Python loop in the shared
`_tensor` (rows whose radial stencil touches an aggregated ring, about the
inner third of the layers). Peak worker RSS was 0.88 GiB, so
`--worker-memory-gib 2.5` leaves ample headroom. The one-off
`verify-equivalence` at N32 took about 5 minutes and passed at 1.2% of
tolerance.

## Preflight (`preflight.py`)

Runs on the same ~54-owner selection vendored in `preflight_fixtures/`
(wall-adjacent, seam, and two controls per resolution), one process per
resolution. The campaign `preflight` runs gates (ii) and (iii), all through
the batched path, in seconds per grid. Gate (i) is a code-equivalence check
that does not depend on where the campaign runs, so it is a separate command,
`verify-equivalence` (writes `equivalence.json`), run once per code change
rather than on every campaign:

- **check1_equivalence** (gate i): batched raw/face kernels vs.
  `operator.owner_centered`/`owner_jump` on the 14-owner oracle subset
  (`preflight._oracle_subset`: every wall-adjacent radial layer, a second
  angular position in the last two layers, all seam owners and both
  controls), every one of the 10 frozen pairings
  and their matched-Dirichlet diagnostics, atol=rtol=1e-12 (`numpy.isclose`
  headroom convention: `max(headroom) <= 1`).
- **check2_replay** (gate ii): the accepted P05 8-field catalogue
  (`p05_structured_global.numerics.FIELDS`/`PAIRS`/`boundary_trace`), all
  Dirichlet, through the **batched** kernels (`core.batched_cell_values`
  etc., not the oracle), against the vendored
  `preflight_fixtures/N{n}.accepted_p05_replay.npz`'s expected
  `centered`/`jump` rows at the same tolerance. That fixture carries the
  accepted P05 campaign's `observations` only for the closure of donor
  owners the replay's own row construction touches (computed once, locally,
  by `preflight_fixtures/extract.py`); every other owner is NaN at runtime,
  so any row construction needing an unanticipated donor fails loudly
  instead of silently comparing against stale/wrong data.
- **check3_linearity** (gate iii): a Neumann row's value/gradient is linear
  in (owner values, g_N), checked on a 5-owner sample through the batched
  path.
- **check4_constant**: max |action| over the four `CONSTANT_PAIRS`, both
  candidates, across the selection <= 1e-8.
- **check5_antisymmetry**: max swapped-argument defect over the raw chunks
  touching the selection <= 1e-12.
- **check6_exact_input**: max |R - (R computed through the candidate's own
  `direct_pair_actions` code path on exact gradients)| <= 1e-12 (a structural
  identity check; expected to be ~0 by construction).
- **check7_zero_jump**: max |N| and |D| on physical-wall / radial-n-1 /
  transverse-last-two-layer faces touching the selection <= 1e-12.
- **check8_condition**: max Neumann elimination condition number over the
  selection <= 1e8 (reported alongside the max constraint residual).

### `preflight_fixtures/`

- `N{n}.selection.json`: byte-identical copy of the bounded
  construction-verification run's 54-owner selection.
- `N{n}.accepted_p05_replay.npz`: `selected_owners` (54,), `expected_centered`
  / `expected_jump` (54,8), `donor_owner_ids` (D,), `donor_values` (D,8),
  `owner_count` (scalar) -- see gate (ii) above.
- `fixtures_manifest.json`: sha256 of every source file `extract.py` read and
  every fixture it wrote, plus `extract.py`'s own sha256.
- `extract.py`: the one file in this package allowed to reference a local
  `work/` directory (run manually, locally, never by `campaign.py`/
  `preflight.py`); re-run it if the bounded selection or the accepted P05
  campaign's outputs ever change.

## Known simplifications (documented, not hidden)

- The Neumann-role wall-patch batching (`prepare_neumann_point_rows`) is done
  once per raw-midpoint chunk (thousands of points at once) for the `raw`
  stage, and once per face's own quadrature-point set for the `faces` stage
  (matching the oracle's own per-face granularity) -- not further batched
  across many faces in one call. The dominant costs identified by profiling
  (`ref._metric` per (owner, pairing), and per-pairing row-rebuild) are both
  eliminated; batching the Neumann patch construction across many faces at
  once would be a further optimization, not required to reach the ~20x
  target (see the task's timing measurement).
- `batched_cell_values`/`batched_face_common_gradient`/`batched_side_values`
  always compute *both* the Dirichlet-role and Neumann-role reconstruction
  for every target, even though a given catalogue pairing only needs one
  variant for a given role. This trades a small (roughly 2x, not 10-12x)
  redundant compute for a substantially simpler, more uniform, and easier to
  verify implementation.
