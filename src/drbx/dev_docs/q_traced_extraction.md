# Frozen traced Q preparation and application

The research-library path in `drbx.stencils.q_parallel` prepares complete-owner
chunks from saved RK4 endpoints. `drbx.native.q_parallel.apply_q` applies the
fixed diffusion rows to current compact owner values. No production selector,
state layout, or restart format uses this path yet. The accepted method and
exceptions are specified in `q_traced_diffusion_frozen_contract.md`.

## Preparation

`topology_from_arrays` constructs the frozen compact owner order, raw member
volumes, and physical-volume centroid observations from canonical RLP arrays.
`prepare_chunk` takes sorted complete owners, their raw members in owner order,
all four saved endpoints per raw seed, and two host geometry callbacks:

- `geom(points)` returns positive Jacobian determinant, contravariant unit-field
  components, and magnetic magnitude, using the frozen evaluator behavior.
- `jacobian(points)` returns the coordinate-to-physical Jacobian used for the
  physical wall normal and the 35-dimensional Neumann elimination.

Callers must supply the accepted campaign identity, source trace hash, and
canonical geometry/magnetic input identity. The bundle records the explicit
span, topology and trace hashes, weights, selected support, degree-5/6 guard
scores, directional value/gradient rows, magnetic/divB coefficients, and wall
sidecars. The Q-specific host support algebra preserves the frozen nearest28,
rank repair, always-balanced and local40 candidates; it selects according to the
frozen field-independent gradient guard. Outer layers and the last two wall rings
use their separate frozen rows. No fit failure changes runtime support.

`save_chunk` and `load_chunk` provide a checksummed v2 NPZ bundle with identity
rejection. V1 artifacts are rejected because they can contain non-wall
Dirichlet tangent coefficients; rebuild from saved endpoints. The checksum
manifest must cover every stored array. `PreparedQ.runtime_view()` drops directional rows, wall query geometry,
and diagnostics before device staging; it retains only fixed diffusion and
boundary action rows. Full directional data stay in the host artifact for later
Q06 work. An owner alias is a raw member for observation/projection, never an
extra state. The owner projection retains every physical-volume-weighted raw
member, with deterministic stored member order.

## Boundary and runtime contract

`QBoundaryData` has four arrays with field batch axes before the raw axis:

| Member | Trailing axes | Meaning |
|---|---|---|
| `dirichlet_trace` | `(n_raw,35)` | prescribed wall-lattice values |
| `dirichlet_query_value` | `(n_raw,3)` | values at wall-projected cap queries; retained for directional value reuse |
| `dirichlet_tangent` | `(n_raw,3,2)` | theta/eta derivatives at those queries |
| `neumann_normal` | `(n_raw,35)` | outward physical-normal derivative at wall nodes |

The diffusion action uses gradients, so the Dirichlet query value does not enter
`apply_q`; it is still an explicit required member of the boundary contract.
The Dirichlet gradient lift is the frozen interior gradient applied to
`d - tile(trace,4)` plus the query tangential derivatives. The Neumann lift uses
the frozen metric-dependent elimination response and supplied physical-normal
data. Interior and boundary contributions are affine and can vary with every
application. `homogeneous_boundary` supplies zeros for the simple homogeneous
case; a spatially constant Dirichlet trace uses constant wall-node values and
zero tangential derivatives.
All boundary action coefficients are exactly zero on non-wall rows. Finite
values in unused non-wall boundary-array entries do not affect the action.

`apply_q(prepared.runtime_view(), owner_values, boundary, kind='D'|'N')` accepts
real64 or complex128 owner values with shape `(..., n_owner)` and returns
`(..., n_chunk_owner)`. Batch axes of boundary data must match the field batch
axes. The function contains only JAX gather, multiplication, reduction, and
fixed owner projection. JIT captures a fixed prepared view and can be reused as
field and prescribed boundary values change. JVP differentiates the fixed
state action; preparation and donor topology are discrete host work.
Use `drbx.native.q_parallel.stage_q(prepared.runtime_view())` once to obtain
the device-backed view actually passed to application or captured by JIT.

The direct reconstructed cap gradients here are the accepted Q diffusion action.
They do not certify the separate owner-sampled `D_h(G_h f)` transfer planned for
Q06. The saved directional rows support that investigation without promoting a
new runtime operator now.

## Direct gradient research interface

`drbx.stencils.q_parallel_gradient.prepare_parallel_gradient(prepared_q)` checks
the schema-v2 source and contracts its three stored logical derivative slots
with the matching contravariant unit-field components. Slot order is minus cap,
plus cap, raw midpoint. It retains source, geometry, topology, trace, and span
identities and the complete raw-member owner projection. It prepares separate
affine D wall-node/tangential and physical-normal N lifts. Dirichlet query
values remain required in `QBoundaryData` but do not contribute to `G`.

`gradient.runtime_view()` is the lean midpoint-only payload.
`gradient.runtime_view(include_caps=True)` supplies all three slots for a
separately named diagnostic; cap outputs are raw rows and are never averaged as
owner midpoint values. Stage the selected view with
`drbx.native.q_parallel_gradient.stage_parallel_gradient`, then use
`apply_parallel_gradient(view, owner_values, boundary, kind='D'|'N')` for
`(..., n_chunk_owner)` midpoint values or
`apply_raw_slot_gradients(three_slot_view, ...)` for `(..., n_raw, 3)`.
All field batch axes and BC axes agree with `apply_q`; fixed coefficients support
real64/complex128 state, changing affine data, JIT, and JVP. No gradient bundle
is persisted: the checked PreparedQ deterministically supplies it at setup,
and the gradient metadata contains a distinct source-bound identity. This is a
research operator, with bounded evidence in
`work/q06_direct_gradient_20260929/`; it is not a production selector or a
divergence/composition implementation.

## Bounded divergence candidate interface

`drbx.stencils.q_parallel_divergence.prepare_parallel_divergence(prepared_q, geom)`
reuses the checked value and gradient rows and the frozen geometry evaluator at
raw midpoints. It prepares `kappa=div(b)` with fourth-order logical-coordinate
differences and records independent `div(B)/B-b·grad(log B)` and step-halving
diagnostics. `prepared.runtime_view('tube')` precontracts the three scalar slots
with the saved `magnetic_L`, including its midpoint divB correction.
`runtime_view('direct')` precontracts the reviewed midpoint G plus kappa times
the reconstructed midpoint scalar; this remains a diagnostic baseline. The
three logical gradient components are already one prepared runtime row, rather
than three separately executed derivative operators.

Stage either lean view with `stage_parallel_divergence`, then call
`apply_parallel_divergence(view, owner_values, boundary, kind='D'|'N')`. It
returns `(..., n_chunk_owner)`. `apply_raw_parallel_divergence` returns raw
midpoint actions before owner projection. For term audits,
`prepared.scalar_view()` and `apply_raw_scalar_slots` expose reconstructed
minus-cap, plus-cap and midpoint values. Dirichlet scalar values include both
wall-node traces and explicit slot query values on wall rows; G still uses query
tangential derivatives, while Neumann data are physical-normal wall values.
The runtime is fixed-coefficient JAX gather, multiply, reduction and projection;
state and BCs may vary under JIT. Neither candidate is a production selector,
and bounded accuracy does not establish conservation, adjointness or global
qualification. The comparison evidence is in `work/q06_bounded_divergence_20260929/`.

For tube-only setup, `prepare_tube_divergence(prepared_q)` returns the lean tube
runtime directly, with no new geometry calls, direct-G contraction or kappa
audits. Its coefficients match the bounded two-candidate builder exactly.

The optional `prepare_chunk(..., frozen_choices=..., choice_provenance=...)`
path rebuilds only the support selected by accepted, checksummed per-raw choices.
The caller must verify those choices against the frozen campaign; it is not an
automatic policy selector. It preserves rank/reproduction checks and requires
0/1/2 on inner rows and -1 elsewhere. Indicator/noise arrays are unused zero
placeholders with explicit `selection_diagnostics` metadata identifying that
they were not reevaluated. The global campaign preflight compares donors and
rows to saved full-selection preparations and checks frozen diffusion actions.
This saves alternative-support work while retaining the original numerical rows.

The [local h/32 tube qualification runner](../../../scripts/q06_tube_global/README.md)
streams complete-owner blocks from the saved traces. It evaluates fresh analytic
divergence midpoint references, retaining N/O/R separately, and stores compact
results rather than a full directional-coefficient archive. Global qualification
is pending; pilot chunks are not complete-domain evidence.

## Replay

`DRBX/scripts/q05_traced_replay.py` is the thin research adapter. It verifies
the accepted frozen source archive, source files, canonical magnetic/geometry
inputs, plan, and saved trace hashes. It supplies the frozen evaluator and MMS
boundary data only outside the package, then compares all 26 field actions for
both boundary types with the accepted global arrays. `--all --n 64` streams
complete owner units and writes resumable receipts, with no giant assembled
artifact. Run each grid size separately with BLAS threads one and CPU JAX. The
corrected bounded validation and performance receipts are under
`work/q05_traced_extraction_20260929/review_v2/`; the original worker evidence
is retained in the parent directory.

Replay receipts bind the current Q package and replay scripts, canonical input
hashes, frozen-source manifest, accepted result checksum, plan, exact owner ids,
trace, span, environment versions, and requested checks. Both the bounded
driver and full replay reject stale or nonfinite evidence. Summaries validate
those same requests before aggregating. A request for JIT, source-row comparison
or an artifact cannot be satisfied by a receipt that omits it. Replaced JSON
evidence is retained in `superseded/`. Full-domain action replay was not performed. On 29 September 2026 the user
closed Q05 on the corrected bounded evidence and waived full replay as a Q06
entry prerequisite. This is an acceptance decision, not a full-replay pass.
The current next step is the [Q06 direct-gradient assignment](q06_direct_gradient_assignment.md).

Within a replay process, verified file hashes are reused only while device/inode,
size, mtime and ctime remain unchanged. Receipts still contain content hashes,
not filesystem timestamps. One resolution's read-only field observations and
geometry context are cached under the full replay identity; changing that
identity or resolution replaces the cache. This avoids repeated 5.8GB makegrid
hash passes and repeated whole-grid field evaluation for each small patch.

The portable actual-HSX patch in `tests/data/q05_actual_hsx/` checks both spans,
both BCs, archived actions, wall masking, staging and artifact integrity without
requiring an external campaign directory. Larger local fixtures add core,
rank-repair and complete-unit checks.


## Symmetric-eta traced gradient

`drbx.stencils.q_traced_gradient.prepare_traced_gradient(checked_q)` builds a
lean fixed action from the existing scalar cap-value rows. The formula is
`b_eta_center*(f_plus-f_minus)/(span*deta)`, with symmetric signed eta legs.
The preparation checks slot symmetry and sampled eta monotonicity; no field
coordinate-gradient rows or magnetic tube-divergence weights are contracted.
`drbx.native.q_traced_gradient.stage_traced_gradient` stages only this runtime.
`apply_raw_traced_gradient` returns raw-member actions and
`apply_traced_gradient` projects all members to complete owners.

State arrays are `(..., n_owner)`. The existing `QBoundaryData` supplies D
wall-node/query values or N physical-normal data with matching batch axes.
Dirichlet tangential derivative data are not read. Owner/state and BC values
can change under a fixed JIT; no geometry evaluation, tracing, fitting or donor
selection occurs during application. Complex/real floating state is supported;
qualification uses float64/complex128. The shared Q05 preparation still retains
its gradient diagnostics for diffusion and direct-gradient users.

The [bounded actual-HSX report](../../../../work/q06_traced_gradient_20260930/report.md)
records 21 owners, both frozen spans and BCs, all 26 fields and JIT/JVP checks.
This reusable research implementation is not a production default or a global
gradient certification. It does not modify accepted cap-gradient diffusion.
