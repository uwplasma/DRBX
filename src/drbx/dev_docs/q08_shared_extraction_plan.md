# Q08 shared reconstruction extraction plan

Status, 4 October 2026: **shared extraction and full-grid GPU replay passed;
h/32 and five eta planes selected; Q08 closeout pending**. The APIs below are
reusable research-library code, without a production solver selector. See the
[current roadmap freeze](parallel_second_order_roadmap.md#current-q08-configuration--user-freeze-4-october-2026)
for the selected numerical contract, static evidence, remaining diagnostic and
resource work, and explicitly untested coverage. The stages below retain the
original implementation/replay requirements.

[Implementation and independent review](../../../../work/q08_implementation_20261002/report.md)
records source snapshots, literal C3 replay, tests and measured resource costs.
The detailed stages below retain the complete acceptance criteria; a bounded
implementation result does not imply every full-domain criterion is closed.

[Code audit and bounded reuse evidence](../../../../work/q08_extraction_audit_20261002/report.md)
and [source inventory](../../../../work/q08_extraction_audit_20261002/source_inventory.json).
The audited base is `b1746484` plus the inventoried working tree, including
the uncommitted Q05–Q07 modules. Recheck source hashes before implementation
because P and production development continue concurrently.

## Objective and frozen contract

Make one reusable field-independent Q reconstruction plan serve traced G,
tube D, material/characteristic transport, vorticity/current–φ and accepted
cap-gradient diffusion. Reuse P's existing storage, batching and execution
infrastructure wherever semantics agree. Keep P/Q policy builders separate;
a universal face algorithm or identical donor policy is not the objective.

For the extraction replay, freeze:

- `compact_c3` magnetic evaluator, RK4-64 saved endpoints and their geometry
  identity; keep the package's spline default and frozen old campaigns intact.
- Q compact28 with rank repair and selective geometry-only gradient guard,
  structured outer support, last-agglomerated-ring switch and common five-plane
  η polynomial. No new basis, support search or selection thresholds.
- Material inner total separation h/32, outer characteristic separation h/16.
  The user-selected common baseline now uses h/32 diffusion as well. Keep
  spans explicit in preparation/application identity and retain h/16 diffusion
  replay coverage. Alternative h/n plans must prepare consistent endpoints,
  nested outer sampling and coefficients and carry their own qualification.
- Current scalar values at primitive slots, raw coefficients/products before
  complete-owner physical-volume projection, matched Ti/φ force once.
- Separate coefficient records for geometry-consistent material divergence
  and frozen cap-gradient diffusion. Do not overwrite either with the other.
- Prescribed D and physical-normal N data, mixed per-field kinds and current
  accepted lifts. No density floors, positivity clipping, new sheath law,
  endpoint SAT, or new elliptic solve is introduced by extraction.
- Existing characteristic frozen-projector AD contract; do not promise full
  eigensystem differentiation from ordinary state/BC JVP tests.

## What to reuse

| Component | Existing authority | Decision |
|---|---|---|
| Polynomial/cardinal algebra | `geometry/_fci_perpendicular_point_primitives.py`, `stencils/q_parallel_primitives.py` | Move common identical functions behind compatibility imports; keep policy selection outside. |
| Owner/member topology | `geometry/fci_perpendicular_reconstruction.PointRowContext`, Q topology builder | Reuse one validated owner/member representation through adapters; preserve existing coordinate increments, centroids and reduction order. Do not substitute P evolution volumes. |
| Exact query tables | `stencils/artifact._QueryTable`, loader deduplication | Expose shared exact-bit table utilities. Preserve units, query roles and boundary kind separately from coordinate identity. |
| Source-major dense rows | `native/fci_perpendicular_source_rows` | Reuse gather-once/value/gradient contraction with field batching. The bounded Q D bridge already replays to 8.5e-14. |
| Normal-data linear responses | `native/fci_perpendicular_neumann_rows` | Reuse the dimension-generic apply kernel; add a Q lowering for 35-node maps. Do not reuse the 28-node P lowerer unchanged. |
| Factored rows | `stencils/tensor_rows`, `native/fci_perpendicular_tensor_rows` | Reuse exact-factor capture, table deduplication and fallback strategy. Generalize η extent 4→parameter only after dense Q replay; never pad/interpolate Q into four planes. |
| Artifact/checkpoint mechanics | `stencils/artifact`, `scripts/p_shared/runner` | Reuse atomic writes, locks, streaming hashes, fixed work units and checked resume. Keep P v2/v3 readable and Q v2 readable; introduce a distinct Q plan schema. |
| Runtime plan pattern | `stencils/operator_plan`, `native/fci_perpendicular_reconstruction_state` | Copy the architectural contract, not the P face census or its physics: arrays as pytree data, small immutable shapes/policies static. |
| Plane layout/halo exchange | `native/fci_perpendicular_sharding` | Extract plane permutation and halo exchange into common owner utilities with compatibility exports; implement a Q-specific plan localizer. |
| Magnetic evaluation | `geometry/Bfield_evaluator`, `jax_bfield_evaluator`, `jax_metric_evaluator` | Reuse evaluator objects and explicit provenance. Keep Q derivative recipes/factors unchanged. A package Q provider must not import research MMS code. |

The P source kernel currently accepts owner-major `(owners, fields)` data;
Q public calls accept `(..., fields, owners)`. Use a documented axis adapter
and `vmap` over leading batch axes. Do not silently reinterpret manufactured
case axes as coupled physical fields.

P's `C3` inner-support name means `fixed_radius`; magnetic `compact_c3` is
unrelated. Q keeps its own accepted compact/repair support. Record both choices
under separate policy keys in every identity.

## Proposed data flow and API

The workflow is:

`checked geometry + owner topology + saved traces/choices`
→ `one paired preparation`
→ `checksummed source-row bank + operator coefficients`
→ `lowered array-data Q plan`
→ `one batched reconstruction per state/BC update`
→ `existing operator algebra`
→ `complete-owner projection`.

The implemented dense workflow uses the following interfaces (boundary values
remain live arrays in the established `QBoundaryData` layout):

```python
from drbx.stencils.q_parallel import prepare_paired_chunks
from drbx.stencils.q_bank import build_q_bank
from drbx.stencils.q_artifact import save_q_bank
from drbx.stencils.q_parallel_rhs import prepare_six_field_rhs
from drbx.stencils.q_plan import lower_q_plan
from drbx.native.q_plan import stage_q_plan, apply_q_plan

outer, inner = prepare_paired_chunks(
    topology, owners, saved_endpoints, provider.geom, provider.jacobian,
    source_identity=trace_identity, geometry_identity=provider.identity,
    frozen_choices=choices, choice_provenance=choice_receipt)
bank = build_q_bank(outer, inner, identity=provider.identity_record())
save_q_bank(bank, artifact_path)
authority, audit = prepare_six_field_rhs(
    inner, outer, provider.geom, diffusion_span=explicit_span)
plan = stage_q_plan(lower_q_plan(
    bank, diffusion_span=explicit_span, material_runtime=authority))
terms = apply_q_plan(plan, state, inner_boundary, outer_boundary,
                    phi, phi_boundary, diffusion_coefficients,
                    kinds=field_kinds, phi_kind=phi_kind, tau=tau, mu=mu)
```

`QGeometryProvider` is an optional host callback/provenance adapter; existing
bare callbacks remain supported. It preserves query batches and makes no
normalization or range-clipping change. `reconstruct_q_state` exposes the five
scalar slots and homogeneous part for individual consumers; traced G and
balanced material tube D adapters consume these slots. The historical Q06
unbalanced D remains available through its original API. Full coordinate-
gradient decoding requires `include_gradients=True`; omitted gradients are
never silently replaced by zero-valued gradients.

The compact bank uses shared padded dense storage with explicit family/width
descriptors. All sampled Q donor widths were 140, with no donor padding;
separate bucket storage would not reduce those sampled buffers. Exact outer
tensor capture/decoding is available as an optional host representation, but
the measured direct host action was slower, so the runtime remains dense.
P's historical four-plane tensor schema is unchanged. Q's sum reduction and
expanded boundary response remain separate from P's einsum/donor-subtraction
kernel, preserving exact legacy Q actions while sharing polynomial algebra,
query tables, owner layout and halo utilities.

`native.q_sharding.shard_q_plan` and `sharded_q_rhs` provide the array-stacked
full RHS wrapper with shared periodic halo exchange and explicitly trimmed
padding. Both global and sharded plans accept their numerical arrays as JAX
data. The 1/2/4 forced-CPU tests cover populated and empty shards, but do not
replace full-domain actual-HSX or real-GPU resource/scaling measurements.

The bank contains source/anchor IDs, complete raw-member closure, donor lists,
families/repair choices, slot IDs, BC query maps and source diagnostics.
There are five unique physical scalar positions: outer−, inner−, center,
inner+, outer+. The original builder evaluates six slots because it repeats
the center. First reuse its original six-slot geometry call and accumulation
order, then deduplicate **stored identical** centers. Do not change batch
shapes of geometry evaluation as a hidden optimization.

Keep operator coefficient blocks separate from reconstruction rows:

- scalar caps/center feed traced G, D, current, material and vorticity;
- diffusion consumes its existing precontracted D/N action rows and lifts;
- optional coordinate-gradient diagnostics are retained on the host, loaded
  only when needed; scalar-only runtime users do not stage all three gradients;
- common donor/projection storage may be shared by reference/index, while
  operator numerical normalization remains explicit.

Boundary tables contain unique physical/logical query coordinates plus
explicit request roles: D values, D tangential derivatives, N physical-normal
derivatives. The same coordinates may serve multiple fields and operators;
their values remain live inputs. Interior buckets carry no wall arrays.
Wall rows use 35-node response maps and separate arbitrary-query D values.
Repeated locations can be deduplicated exactly; distinct boundary laws or
normalizations must not be merged merely because coordinates match.

Runtime arrays should be registered pytrees passed as arguments, with only
small shape/mode metadata static. Full provenance dictionaries, source hashes,
condition histories and timers remain in host artifacts. No fits, tracing,
geometry queries, hashing or file reads occur inside a warmed RHS call.

## Staged implementation

### 1. Common primitives and paired preparation

This is the first implementation slice. It should be independently reviewable
and leave every public application interface working.

1. Move the 14 identical primitive definitions to a neutral private geometry
   module, for example `geometry/_reconstruction_primitives.py`. Keep P/Q
   modules as compatibility exports. Preserve aliases (`residual`/`resid`),
   long-double intermediates, rounded tie ordering, SVD cutoff and summation
   order. Verify the small `fit` alias difference explicitly.
2. Add a private paired preparation core behind `prepare_chunk`; compute
   magnetic six-slot geometry, support rows, wall elimination and owner
   projection once, then make both span views. Keep the old single-span call
   as a wrapper and the old artifact schema readable.
3. Reuse bounded topology/ring/fit caches across the paired views, keyed by
   complete geometry/topology/policy identity. Cache a wall elimination by
   its actual angular indices/degree/normal identity. Do not reuse a fit under
   another geometry or deduplicate approximately equal floating coordinates.
4. Retain the existing callback API first. Define a narrow package geometry
   provider for `(J, b_contra, |B|, coordinate_jacobian)` with evaluator,
   normalization, validity range and batching provenance. Migrate research
   adapters without importing `scripts/` or MMS sources into the package.

Gate: donor IDs/order, choices, slot coordinates, complete-owner weights and
prepared rows reproduce the legacy call; bitwise where the unchanged arithmetic
permits it. Replay both diffusion spans, D/N/mixed, scalar slots, gradients
and the six-field action. Include the portable fixture **and** the N32/N48/N64
C3 audit owners spanning core, repaired inner, transition, bulk and wall.
Re-run affected P primitive/row and Q extraction suites. Record setup-stage
times; do not call a setup speedup a timestep speedup.

### 2. Compact Q bank and shared artifact utilities

1. Introduce `stencils/q_plan.py` (Q bank/plan types) and a small Q artifact
   adapter. Reuse existing exact query-table and receipt primitives, exposing
   them through neutral helpers rather than copying them again. Keep P's
   request tags R1–R4 and Q's cap/center tags in their own schemas.
2. Bucket by family, donor width and required channels. Store common interior
   D/N rows once and allocate BC responses only for conditioned wall sources.
   Store both span operators against one donor/projection bank; share center
   rows only after exact equality checks.
3. Separate replay diagnostics from minimal runtime arrays. Report logical
   bytes, unique-buffer bytes, padded bytes, compressed bytes and maximum
   load temporaries. Do not infer device residency from NPZ size.
4. Reuse `p_shared.runner` for research orchestration: deterministic complete-
   owner work units, atomic chunk/receipt writes, content hashes, one writer,
   bounded spawn workers and validated restart. Extract neutral runner helpers
   only if package callers need them; package numerics must not import it.
5. Use a field-independent bank identity covering canonical input content,
   magnetic evaluator/coordinate map separately, normalized geometry,
   trace steps/endpoints/crossing diagnostics, support and choice receipts,
   source hashes, dtype, spans and observation/projection rules. Keep campaign
   catalogue identity separate. Legacy freeze lineage is not a substitute
   for the actual C3 geometry/action identity.

Gate: encode/decode reproduces stored arrays and maps exactly; corrupted,
incomplete, stale, wrong-geometry and wrong-span artifacts fail before apply.
Preserve the Perlmutter regression for same-size content changes with unchanged
timestamps: hashes must be of content, not a stat-only cache assumption.
Generate a stratified real-grid byte census before any full-domain build.

### 3. Shared runtime reconstruction and six-field reuse

1. Lower the bank to an array-data pytree, with static family/shape descriptors.
   Reuse P source-major dense contraction and N apply machinery through small
   adapters. If extracting neutral kernels, preserve P wrappers and pytree
   registration exactly once; avoid a circular P↔Q dependency.
2. Preserve both supported D representations: P's donor subtraction plus
   query addition, and Q's expanded node/query lift. A contraction reorder
   must pass the declared replay budget, not be called bitwise by assumption.
   Keep N's pre-eliminated normal response; no wall solve at runtime.
3. Reconstruct all needed physical fields with batched gathers, sharing five
   scalar positions across consumers. Reuse the same donor gather for the
   precontracted diffusion rows when profiling confirms benefit.
4. Reuse homogeneous/lift components for current diagnostics. Its physical
   current is `n*(Vi-Ve)` at primitive slots; physical-minus-zero-BC current
   contains cross terms. Do not substitute `I(n*(Vi-Ve))`, a linear current
   lift, or a second endpoint current contribution.
5. Retain material/characteristic/vorticity/current–φ algebra in their existing
   modules; expose an internal apply-from-slots path. Apply B²/n before owner
   projection and maintain the matched electron Ti compensation once.
6. Keep legacy wrappers, shapes and diagnostic outputs. An explicit static
   diagnostic mode may omit unused intermediates from the ordinary call;
   finiteness/admissibility flags must remain meaningful.

Gate: every scalar/action component and BC response replays, not only the
sum. Validate JIT, vmap, permitted real/complex scalar paths, live state/BC/
coefficient JVP/VJP, fixed-shape coefficient updates without recompilation,
constant reproduction and invalid-input checks. Keep the inherited restricted
characteristic AD contract explicit. Recheck P kernels after any shared edit.

### 4. Exact factoring, only where it helps

Keep the compact dense bank as a trusted fallback. First measure which arrays
dominate memory and warmed application. Reuse P's exact-factor strategy:
capture factors while building the rows; never fit factors back from weights.

- Parameterize the tensor representation for 4 or 5 η planes, including
  donor products, table indices, ring entry shape and checksums. Existing P
  v3 encodings must retain their historical four-plane meaning and load path.
- Q outer structured rows are 4 radial × 7 angular × 5 η factors. Preserve
  anchor-fixed supports and accumulation order; gate expanded decoding
  bitwise with dense fallback if it differs.
- Q inner per-plane quartic maps are different from outer tensor rows. Keep
  them dense initially. Reusing one 15×28 fit per anchor/plane may help many
  targets but can also store more than five evaluated value rows; decide from
  measured byte/operation counts, not polynomial terminology.
- Keep wall N elimination dense initially. Capture radial/angular factors
  later only if it reduces measured runtime/storage with full BC replay.
- Direct factor application changes reduction order. Gate actions separately
  from the exact-decode check and report cancellation floors.
- Avoid P's current all-η tensor temporary on a Q shard: localize contexts to
  owned/halo planes and only the requested fields/channels.

Gate: exact decode, fixed support/choice invariance, P/Q action replay and
measured memory/time improvement. Do not require a particular compression
ratio, and do not retune numerics to obtain one.

### 5. Matched η sharding

Extract/reuse P's plane-major permutation and ring halo exchange with its
existing tests and compatibility imports. Q gets a separate localizer:

- Partition by complete owners; retain all raw members on the owner's plane.
  Assert the equal-owner-per-plane and single-plane-owner properties rather
  than assuming them for arbitrary future meshes.
- Compute donor reach from actual nonzero rows, including repaired and wall
  supports and both spans. Current fixed k−2…k+2 Q support predicts halo 2;
  prove it at every lowered source. P's RHS halo 3 is not copied blindly.
- Remap donor, projection and boundary query indices to local owned+halo
  storage. Only owned output owners are returned. Padding must carry safe
  indices/zero weights and cannot hide missing raw members.
- Share a halo exchange over the necessary fields when P/Q are eventually
  composed; choose the maximum required reach explicitly. Do not wire the
  production combined solver during this step.

Gate: bounded actual-HSX owners on every η plane, periodic seams, all regions
and D/N/mixed data; matched single-device versus 1/2/4 forced CPU devices.
Then measure the identical plan on actual target GPUs. Host-device emulation
does not establish GPU memory, performance or scaling.

### 6. Pilot, full replay and handoff to the remaining Q08 work

Before a full build, measure representative owner-complete chunks by family
at N32/N48/N64. Record a source-identified benchmark command and fixed hardware/
memory/disk budgets. Measure:

- cold/warm geometry/support/wall preparation, writing/loading and transfer;
- persistent host/device payload and peak temporary/compile allocation;
- synchronized first-call and warmed calls per operator and all six fields,
  BC refresh cost, diagnostics on/off and repeated-call cache growth;
- actual one-state RHS cost separately from many-MMS-state batching, and
  estimated integrator-stage cost labelled as a projection;
- sharded communication and replicated-table bytes, measured and projected
  quantities separately. No A100 forecast based solely on the local CPU.

Only build full artifacts after the byte census fits the declared resources.
Reuse saved C3 traces and qualified choices. First perform full-domain action
replay of the new representation; implementation-only refactoring does not
require repeating analytic-reference/tracing campaigns. Keep old scientific
coverage limits: current/φ, vorticity and C3 diffusion transfer still need their
remaining term-resolved global static evidence before full RHS certification.

The subsequent span and three-versus-five-plane comparison is complete. The
user selected h/32 and five planes on 4 October, including h/32 diffusion,
and skipped unequal-resolution checks as a Q08 prerequisite. Alternative
actions require their own scoped qualification. Q09 confirms the selected
configuration in full-domain evolution; patch spectra are not a substitute.

## Replay and review policy

- Snapshot the currently passing source plus immutable oracle receipts before
  moving functions; do not silently rewrite archived manifest hashes.
- Compare array storage/relabeling bitwise. For floating application reorder,
  freeze tolerances from existing component replay and measured conditioning
  **before** seeing the new results. The existing Q diffusion replay gate is
  1e-8; it is not a generic scientific accuracy gate or permission to relax
  any other gate. Use P's explicit cancellation-floor/control treatment where
  applicable, without deriving tolerance from a nearly zero residual alone.
- Cover D/N uniform and mixed, core/first ring, repaired inner, both transition
  sides, bulk, last two rings and η seam. The portable two-owner probe is
  insufficient on its own. Include both diffusion spans and all inherited
  smooth/short-wave states in bounded C3 replay.
- Compare old and new N and unchanged O/R separately. No new global order
  claim follows from bounded replay. Record source/input identity and preserve
  original errors, rather than tuning support to make a replay gate pass.
- Coordinate edits to shared P modules against the current source hashes.
  Preserve concurrent work and keep P compatibility regressions attached to
  each extraction slice; do not require P to wait for Q's physical wall work.

## Completion of this extraction, versus completion of Q08

Extraction is complete when the shared bank/plan drives every selected Q
consumer, P compatibility and Q replay pass, artifacts are resumable and
portable, bounded/sharded checks pass and the target representation has a
measured memory/runtime budget. It need not introduce a new generic polynomial
framework or change any accepted operator.

Full Q08 additionally requires its selected spans/η support, remaining static
term coverage, frozen six-equation source-paired RHS and performance gates.
Physical grazing-sheath modeling, exterior crossings/SAT/wall work, outstanding
traced Q04 evolution and Q09 time-dependent qualification remain explicit.

## Packaged remote implementation gate

The [Q08 replay campaign](../../../../DRBX/scripts/q08_extraction_global/README.md)
implements the full-domain replay/resource stage above using saved C3 traces.
It combines CPU legacy-versus-extracted preparation and action checks with
actual one/four-A100 full-RHS replay and timing. The campaign includes no fresh
tracing or continuum MMS reference work. Independent accepted bounded actions
are bundled, and resume requires unchanged content identities. A passed local
preflight is preparation evidence only; full-grid and actual GPU outcomes must
be returned and reviewed before this extraction gate can close.
