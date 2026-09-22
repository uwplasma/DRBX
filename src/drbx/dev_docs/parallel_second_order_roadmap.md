# Roadmap: second-order parallel operators on HSX angular RLP grids

Approved research roadmap, 18 September 2026. This document specifies planned
work and acceptance gates, not implemented or verified capabilities. It is the
authoritative Q00–Q09 progress ledger. Keep run logs and detailed numerical
evidence in linked research artifacts. The separate
[perpendicular roadmap](perpendicular_second_order_roadmap.md) uses P00–P09.

## 1. Objective and shared acceptance contract

### Current result: whole-support exchange improves accuracy but fails the gate

The [returned Q03 remote campaign and independent local analysis](../../../../work/q03-exchange-tNlxO86M_analysis/report.md)
are complete at pinned revision `afbb1ace`. The frozen 64-swap whole-support
cubic candidate gives G3 global orders radial `1.528/1.949`, angular
`2.231/1.634`, and mixed `2.019/1.515` on N32/N48/N64. No field passes both
intervals. N64 errors improve by `4.76%/13.44%/11.24%` relative to the preceding
enriched candidate, but this is not a qualification pass. Q03 remains open;
Q04, structural/solution certification and production promotion remain pending.

Exact-row orders `1.346/2.016`, `2.266/1.494`, `1.912/1.591` establish a remaining
return/assembly accuracy limitation even without G3 endpoint error. Rechecked
reference budgets are `0.22–2.86%` of candidate error and cannot rescue the
failed intervals within the recorded empirical uncertainty. Source/input and
all 2,197 chunk identities, full incidence assembly, and weighted errors were
verified locally. All interior fits retain rank 19 with cubic residual at most
`6.84e-13`; this is a scientific order failure, not an execution failure.

N64 ordinary interiors carry `59.08%/85.91%/78.60%` of G3 squared error and
boundary-adjacent owners `40.89%/14.07%/21.34%`; all axis/RLP strata combined
carry under `0.07%` in each field. Prescribed continuum boundary flux remains
the diagnostic boundary treatment. Global exact-flux cancellation improves
for radial/mixed and is almost unchanged for angular, while the absolute
face-error budgets shrink slowly. Do not attribute this result generically to
RLP interfaces, incorrect wall flux, G3 alone, or worsening cancellation.

The completed [weak directional-observation mode audit](../../../../work/parallel_q03_weak_modes_20260922/report.md)
finds physical near-nullness with material target-recovery amplification, not
a harmless transverse nullspace or floating-point breakdown.  On its bounded
complete-owner N32/N48/N64 sample, the weakest fixed cubic projector group
carries only `0.10--0.57%` of squared target projection but `43--67%` of
weighted coefficient-norm squared; its physical parallel-gradient energy is only
`6--10%` of the strong group's.  N64 hotspots have median completed physical
quartic residual `2.34e-1` versus `5.88e-3` for residual-blind ordinary/RLP
controls.  Matched ordinary-hotspot quartic residual grows `2.51x` from N48 to
N64 while controls fall to `0.36x`.  Removing the weakest six modes loses
`5.1%` median cubic target reproduction and worsens control-field RMS by
`9.2--44x`, so no singular cutoff is supported.  Fresh degree-five diagnostics
do not displace the leading quartic result.  The smallest follow-up internal to
this hypothesis would be one bounded mapped/area-integrated observation control
on the same supports, holding degree, q9 target, G3, shared incidence and
assembly fixed.  It is not another target-quadrature, regularization,
support/cap, global-campaign, Q04 or production authorization.

The parent's parallel [direct cubic owner-to-face comparison](../../../../work/parallel_q03_direct_owner_flux_20260921/report.md)
is complete on 32/36/36 selected owners at N32/N48/N64, with all 379/523/620
incident subfaces. The field-independent 120-owner scalar fit reuses qualified
q9 integrated face moments and the actual raw-midpoint owner sampling. On N64,
complete-cell RMS improves approximately 463/519/449 times over the returned
G3 candidate; error-blind controls improve 54/23/42 times. Independent q9/q11
HSX reference checks are below 4.4% of the smaller errors; source sampling,
geometry identity, polynomial reproduction, and flux/action replay checks pass.
These are bounded results, not global convergence or structural certification.
The sampled radial error is nonmonotone at N32/N48; no sample slope is a gate.
Direct scalar recovery is a promising alternative, but its cross-field
pollution, dissipation, minimum principle and global accuracy remain unqualified.

The completed [direct-cubic anisotropy audit](../../../../work/parallel_q03_direct_anisotropy_20260922/report.md)
extends the sample to 43/45/41 complete owners. Two smooth probes have over
99.94% perpendicular gradient energy in actual HSX geometry. Their N64 errors
are `1.25e-4/1.07e-4`, versus old G3 `0.0486/0.0505`; the original three fields
retain 435--516x improvements. Reference quadrature differences are at most
1.6% of the new probe errors. These remain bounded results, not global orders.
The homogeneous rows have material negative off-diagonal coefficients (about
53% of entries): the unrestricted linear action fails positivity/minimum
principle for some nonnegative states. This does not prove energy growth;
global dissipation is unqualified. An energy argument alone cannot repair the
established positivity defect.

**Current authorized next step:** prepare the remote N32/N48/N64 static global
accuracy qualification of the frozen direct cubic scalar-owner candidate.
Use the [direct cubic remote campaign](../../../scripts/q03_direct_campaign/REMOTE.md),
with the original three nonconstant fields plus constants, 120 donors, cubic
moments, fresh continuous q9/q11 references and prescribed analytical boundary
flux. The two additional anisotropy probes remain bounded diagnostics. This
experiment may establish accuracy of the unrestricted high-order component;
it does not pass Q04 or relax positivity, dissipation or solution requirements.
Do not require structural repair before measuring this component's global
accuracy. Conversely, no passing accuracy result promotes it as a complete
diffusion method. Conservative positivity protection is a subsequent design
experiment, not an authorized remote modification.

The [direct runner](../../../scripts/q03_direct_campaign/README.md) uses indexed
exact donor searches, reusable centered owner moments, batched SVD, bounded
process parallelism and identity-checked restart. Frozen scalar inputs are
exported locally; remote reference production uses the actual corrected d58
metric cache and continuous MAKEGRID evaluator without historical workspace
imports. The older Q01 catalogue metric hash is retained as historical
source-state provenance, not substituted for this campaign's evaluator hash.
No global computation has yet run for this candidate. P and Q remain separate.

### Remote global qualification and worker ownership

Global qualification campaigns should use the remote CPU allocation once
their bounded implementation checks and authorized candidate are ready.
**Completed execution decision, 21 September:** the user stopped the planned
local global campaign and requested remote handoff of the frozen whole-support
cubic exchange N32/N48/N64 comparison. That remote computation has now finished;
the scientific decision is recorded above. The local exception remains superseded.
The completed exchange study used the [exchange runner](../../../scripts/q03_exchange_campaign/README.md).
The newly authorized direct study uses the separate remote path above; the old
exchange workflow remains stopped and the local Q bounded audit is complete. Remote
allocation, environment setup, parallelism and monitoring belong to the remote
worker's setup skill; remote scientific interpretation is excluded.
Resource selection belongs to the remote setup skill, with separate reference
and reconstruction worker settings and no prerequisite scaling study. The remote task receives
repository commands, not environment, scheduler, transfer, or launch-setup
instructions. It owns its run and monitoring. P and Q are separate campaigns;
all coordination goes through the parent task. O is archived.

The perpendicular campaign now has a clean versioned
[remote command path](../../../scripts/hsx_remote_qualification/README.md).
Its donor policy and inputs are not automatically the Q implementation.
Apply the same provenance principle here: freeze the geometry, owner/leg
functionals, selected supports, numerical policy, reference qualification,
and source identity. A changed donor graph requires rebuilding dependent
outputs under a new identity, not forcing historical tie outcomes or mixing
old and new caches. Reuse genuinely independent geometry/reference inputs.
Exact historical donor equality is not a gate for a new candidate; consistency
within its own declared policy and the HSX global accuracy gate are required.

**Current Q scope:** the 22-owner N64 cubic-enrichment diagnostic, frozen-selector
optimization, bounded benchmarks, and authorized serial N32/N48/N64 global
cached-rank-one campaign are complete. The global candidate fails the frozen-G3
two-interval `>=1.8` order gate in all three fields, so it is not promoted and
Q03 remains in progress. Do not launch or operate P's remote/global study from
Q. No Q04 certification, N128 extension, limiter stage, or production promotion
is implied.

The [whole-support qualification work package](q03_whole_support_qualification_plan.md)
has delivered its authorized global computation and local interpretation.
The broader proposed local sample was not a prerequisite and was superseded
by the full remote comparison. The interim local launch remains stopped;
no new campaign should start from these historical instructions. Preserve the
frozen inputs, returned outputs, and negative accuracy result under their own
identities. Regional regressions and sample slopes do not add acceptance gates.

### Scientific contract

First certify the lean owner-overlap parallel diffusion method currently under
MMS investigation. Only after diffusion passes, promote its qualified owner
measures, mapped interface geometry, reconstruction conventions, and conservative
assembly to the remaining parallel operators and a coupled parallel-system MMS.
A symmetric diffusion matrix is not an advection operator: preserve the
directional geometry needed to construct gradient, divergence, and characteristic
fluxes independently.

Both roadmaps use the following contract:

- **Real HSX geometry is the acceptance basis from the first numerical audit.**
  Use the actual metric, field-line traces, angular RLP topology, axis treatment,
  and domain boundaries. No idealized-map, slab, or simple-polar prerequisite
  convergence gate. Small algebra/bookkeeping tests remain useful but do not
  establish HSX accuracy or complete an operator milestone.
- **Numerical tests target HSX only:** new or revised accuracy, convergence,
  reconstruction, interface, and boundary tests in Q00–Q09 must use actual HSX
  artifacts, not idealized slab, straight-field, circular-polar, or synthetic
  map campaigns, including preparatory diagnostics. Cheap regressions may
  extract bounded regions of real HSX artifacts while retaining their metric,
  topology, measures, traces, and boundary semantics, with parent provenance.
  Local extracts cannot certify global order. Geometry-independent unit tests
  are limited to algebra/bookkeeping; preserve historical idealized evidence
  without rerunning it for milestone completion. Missing HSX inputs are an
  explicit producer dependency, never a reason to substitute idealized fixtures.
- Require global **physical-volume-weighted operator-error L2 order at least
  1.8 on both finest refinement intervals**, separately for each certified
  operator and nontrivial manufactured field. Count each physical volume once;
  RLP aliases are representation, not extra mass. Use actual spacing ratios.
- Regional and maximum-norm orders are diagnostic. Report axis, RLP transition,
  ordinary interior, and boundary errors, physical volumes, and contributions
  to global squared error. Maintain a disjoint partition for accounting even
  when additional diagnostic masks overlap.
- Keep **midpoint MMS as the inexpensive default**. State sampling and averaging
  conventions explicitly. Use bounded checks on selected actual HSX cells when
  representation or reference accuracy needs qualification, not mandatory
  full-domain higher-order quadrature. Routine midpoint error may contribute
  to the overall second-order error budget. If bounded evidence demonstrates
  mesh-harmonic midpoint aliasing large enough to obscure the operator error,
  use a qualified independent integrated physical-face reference for that
  certification study while retaining midpoint results as separate sampling
  diagnostics.
- Retain independent fixed-physical-time solution checks with the same L2 order
  criterion. Reference, temporal, and applicable solver errors must each be
  below 10% of the corresponding finest spatial error. A solution pass does
  not waive an operator failure; algebraic invariants do not prove accuracy.
- Begin with the existing **32³/48³/64³** geometry artifacts and one continuous
  geometry reference across the sequence. Nonmonotone results remain
  inconclusive. Extending resolution is an explicitly scoped follow-up, not
  an automatic escalation of resource use. Do not infer slopes at a numerical
  floor; exact-zero controls verify identities rather than empirical order.
- **Evaluate physical face geometry continuously at the face quadrature
  points.** Stored owner membership, logical bounds, and connectivity define
  where a face or agglomerated boundary tile lies; they are not the default
  source of its metric, embedding derivatives, normal, measure, or magnetic
  factors. Derive those quantities consistently from one analytic/continuous
  evaluator call, share/cache that canonical evaluation across both
  incidences, and use a qualified regular chart or evaluator limit at the
  axis. A missing serialized face metric is not a producer dependency when
  the continuous evaluator supplies it.

**Pinned boundary baseline for both roadmaps: the existing frozen MMS
configuration**, `physical_wall_model="legacy-velocity-trace"` with
`parallel_velocity_wall_bc="neumann"`. This supersedes the interim no-flow
and rung-2 choices. The legacy compatibility model supplies owner-extrapolated
ion/electron velocity traces and thermodynamic Neumann conditions, with the
existing potential/vorticity Dirichlet conditions. It is neither the rung-1
no-flow wall nor the rung-2 conducting sheath. Preserve
`neumann_ghost_scheme="physical"`,
`parallel_boundary_pairing="characteristic-sat"`, and
`parallel_characteristic_wall_law="energy-absorbing"` from the frozen MMS.
The numerical characteristic wall law does not turn the legacy trace model
into a conducting sheath.

The source of truth is `simulate_hsx_mms.py`'s `_production_configuration`,
`_production_selector_args`, and runtime assertions, checked against the
32/48/64 frozen-run manifests in the
[HSX RHS summary](../../../work/perpendicular_second_order_hsx_p01_p03/perpendicular_rhs_summary.json).
`native/fci_physical_wall.py` resolves the selected physical traces. Keep
periodic eta, actual HSX wall/termination geometry, and existing axis topology.
Physical sheath-law certification or migration is outside this P/Q scope.

Choose manufactured fields and independent sources/boundary pairing compatible
with this fixed configuration. Explicitly record each operator's effective
scalar/diffusive flux, normal-derivative, polarization, and terminated-leg
closure: the physical-wall selector alone does not specify every numerical
boundary functional or justify dropping wall contributions. Preserve existing
operator-specific support rules. Bounded tests with other manufactured boundary
loads remain labelled diagnostics. Include selectors, effective closures, and
boundary-contract identity in provenance/cache keys. Preserve historical
results; no boundary rerun is required solely to repair the earlier rung label.
Check actual closure differences before certification reuse. This pins the
research configuration, not production or blob-driver defaults.

The final scope is the current parallel system on the actual HSX domain under
this frozen-MMS boundary contract. Blob-driver synchronization and new transport physics
remain out of scope.

**Positivity is mandatory for homogeneous diffusion**, alongside conservation,
dissipation, constants, and the discrete minimum principle. Test those separately
from forced MMS, whose source can legitimately change extrema. Qualify the
positivity behavior of the selected time integrator at its stated timestep;
semidiscrete properties alone do not certify every timestep. If accuracy and
these structural requirements cannot be achieved together, record a blocked
numerical-design milestone rather than silently relaxing either requirement.

These are acceptance requirements for the final physical method, not restrictions
on every intermediate diagnostic. Do not require all reconstruction coefficients,
intermediate matrices, or nonlinear Jacobians to have the current graph's sign
pattern. For a nonlinear candidate, establish the applicable action-level
conservation, dissipation, and minimum-principle properties; an M-matrix is a
sufficient construction for the existing linear graph, not a mandatory format
for every candidate. Clearly labelled unlimited or signed diagnostic actions
are allowed to isolate consistency errors, but cannot be promoted without the
final structural checks. Missing producer data, or failure of a global scaling
control, is not proof that the numerical requirements are incompatible.

### Donor coverage and conditioning (shared P/Q lesson)

Both paths have exposed reconstruction failures caused by insufficient or
poorly distributed donors. Donor count alone is not the criterion: nearest
owners can cluster along too few angular columns or eta planes, leaving the
target derivative weakly determined. A full-rank solve and accurate polynomial
reproduction can still produce large coefficients and inaccurate operator
actions. This is a support-design issue to check before rejecting the
reconstruction family or increasing polynomial degree.

- Preflight the actual HSX owner-observation matrix with centered/scaled
  coordinates and rank-revealing solves. Preserve the actual stored owner
  functional; replacing averages by representative point values is a separate
  consistency error that better conditioning cannot repair.
- Inspect directional coverage, donor count, support radius, scaled condition
  numbers, and coefficient amplification. For derivative rows report
  mesh-scaled amplification with explicit coordinate/component conventions;
  regular Cartesian y is not logical theta. Condition numbers depend on
  scaling and basis, so compare them under a stated common convention.
- Do not stop support expansion solely because rank and reproduction pass.
  Compare a modest increase or directionally balanced support on cached HSX
  faces, holding degree, observation functional, and assembly fixed. Select
  support from geometry and algebra, not exact MMS values or errors. Prefer
  compact, adequately distributed donors: wider support can reduce
  amplification while increasing approximation bias and runtime.
- Rerun the complete bounded operator action after repairing support, including
  the relevant gradient/return stages and cancellation-sensitive contributions.
  Better conditioning is evidence of a repaired fit, not proof of operator
  accuracy. No universal condition-number cutoff, formal conditioning-to-error
  bound, or improvement at every sampled owner is an additional acceptance
  requirement. Global HSX operator convergence and the existing structural
  gates remain decisive.
- Record the support policy, scaling, expansion/fallback activity, cost, and
  comparison identity. Invalidate affected reconstruction caches when these
  change; retain independent geometry/reference caches and historical results.

Evidence from both paths supports this workflow. The parallel
[reconstruction audit](../../../../work/parallel_q03_state_refinement_patch_20260919/Q03_RECONSTRUCTION_IMPLEMENTATION_AUDIT.md)
reduced the maximum scaled condition from `6.86e5` to `38.9` with a
geometry/algebra-selected reconstruction using better donor support and
substantially reduced bounded propagated errors. The perpendicular
[supervisor support check](../../../../work/perpendicular_owner_face_supervisor_20260920/report.md)
found that 12 donors per eta plane occupied only two angular owner columns at
representative failing faces; 18 introduced a third, reducing condition numbers
from thousands to about five and sharply improving sampled vorticity
derivatives. These are bounded results, not universal donor counts or global
certification; the full perpendicular support rerun is pending.

### Efficient diagnostic execution (shared P/Q workflow)

Use the [supervise-long-runs skill](/Users/yxie/.codex/skills/supervise-long-runs/SKILL.md)
for reusable stage execution, recovery, and quiet adaptive supervision. The local
supervisor chains authorized stages immediately; the heartbeat provides oversight.
Default heartbeat wakeup is **10 minutes**, adapting to **20 or 30 minutes** for
measured longer stages, with a **30-minute maximum** unless the user changes it.
Scientific scope and acceptance requirements remain in this roadmap.

**Task handoff design requirement:** new bounded assignments must specify the
mathematical actions/functionals, coordinate and component conventions, fixed
inputs/support, independently checked quantities, stage dependencies, acceptance
and stopping decisions, and exact scope of any resulting claim. Require a small
preflight through the actual HSX path before scaling. A prerequisite failure must
not block an independent diagnostic without a demonstrated dependency. Do not
replace shared multi-owner assembly with a one-row balance identity and label it
equivalent. Record deviations explicitly before interpreting their results.

**Implementation flexibility:** fix the scientific question, representation,
boundary contract and acceptance criteria; treat suggested donor counts, patch
sizes and solver choices as starting points unless they define the comparison
itself. Workers may expand local support, improve scaling/rank detection, change
equivalent solvers, repair indexing, and make modest evidence-driven scope
adjustments without another authorization. Formal rank/compatibility is not a
usable-stencil guarantee: failed reproduction or excessive numerical amplification
justifies adapting the support. Preflight failures before expensive rebuilds,
record changes and preserve comparison identities. Report a blocker only when
the scientific requirement or a real resource limit prevents progress, not merely
because an initial implementation recipe failed. This does not authorize global
campaigns, production promotion, or weakening accuracy/structural requirements.

Apply these practices to both roadmaps without weakening the HSX accuracy,
reference-independence, complete-support, or structural acceptance requirements.

- **Scope and time the work before scaling.** State the hypothesis, decision to
  be made, actual owner/face/donor support, reference-point count, and proposed
  resolutions. Run a tiny actual-HSX preflight through the intended coordinate,
  boundary, sampling, and reporting path. Time one representative batch and
  separate setup/compilation, reference evaluation, operator action, and output
  costs. Use these measurements for a provisional runtime/memory estimate;
  refine it from progress. Do not add a separate benchmark campaign or require
  user approval for routine runs already within the assigned scope.
- **Compute only the reference quantities needed.** Prefer narrow Jacobian,
  face-velocity, vorticity, or gradient evaluations over preparation of unrelated
  RHS terms. Batch and deduplicate shared points; reuse metric evaluations at
  common differentiation locations. Avoid computing unused Hessians and
  repeating whole-grid preparation for each bounded candidate. Preserve the
  independent continuum definition and verify optimized paths on bounded HSX
  samples before relying on them. Profile before a broad performance rewrite.
- **Reuse qualified data, not stale conclusions.** Cache geometry/reference
  samples and complete bounded support across candidate comparisons. Key reuse
  by geometry and relevant implementation hashes, field/source and time,
  coordinates/selection/weights, boundary contract, precision, reference
  controls, and schema; key certification additionally by requested checks and
  acceptance rules. Record dependencies and invalidate affected results when
  they change. Preserve historical evidence and independent reference checks.
- **Qualify expensive controls economically.** Midpoint MMS remains the default.
  Use bounded, stratified actual-HSX quadrature, differentiation, or tracing
  refinement to estimate uncertainty in the corresponding completed operator
  residual. Fixed choices such as 128 tracing substeps or full-domain high-order
  quadrature are not requirements in themselves. Use cheaper qualified controls
  when evidence supports them; do not infer sufficiency from endpoint shifts
  alone, truncate necessary support, or weaken the agreed error budget.
- **Checkpoint and chain authorized stages automatically.** Use persistent
  stdout/stderr, stage timings/progress counts, atomic versioned checkpoints,
  process/launch identity, and captured exit status. A launcher should run the
  next authorized chunk or stage immediately after successful completion,
  including final assembly/analysis where already specified. A successful
  chunk exit is not completion of the whole experiment. Resume only missing
  valid stages and guard against duplicate launches. Stop the chain on a real
  failure or a numerical decision requiring review. Do not use an agent
  heartbeat as a scheduler for each small chunk.
- **Match monitoring to expected duration to save tokens.** For short runs,
  inspect completion directly with bounded tool waits instead of creating a
  recurring monitor or repeatedly polling. For runs expected to outlast an
  active turn, use a quiet long-interval heartbeat (normally the user's
  10-minute default, adapting up to 30 minutes), informed by measured runtime. Reuse/update one existing
  monitor per task; do not create duplicates or shorten the interval merely
  to dispatch the next chunk. Stay quiet on unchanged healthy progress;
  resume on completion, actionable failure, or a required decision. Pause or
  remove obsolete monitors when their work finishes. Monitoring cadence must
  not insert idle gaps between authorized computations.
- **Recover failures before repeating work.** Check exit code and logs, validate
  stage inputs, and fix the identified problem with a small HSX preflight.
  SIGKILL alone does not prove memory exhaustion. Preserve useful checkpoints,
  record unknown causes explicitly, and confirm a restarted process is alive
  before reporting it running. Do not interrupt a healthy expensive stage just
  to retrofit logging or optimization; apply changes at a safe stage boundary.
- **Report scientific progress and cost separately.** Summarize completed and
  remaining stages, measured errors/orders, unresolved mechanisms, compute
  versus idle time, and a provisional ETA when useful. A local regression is
  diagnostic, not a requirement for every region to improve. Keep the next
  experiment bounded by a specific decision; do not expand a sampled audit
  into a global campaign without the scope already being assigned.

## 2. Phase A — Establish and certify diffusion

### Q00 — Freeze the baseline and make result identity reliable

**Dependencies:** none.

- Freeze the current lean owner-overlap operator, coefficients, actual boundary
  treatment, precision, geometry/source hashes, revision and dirty-file hashes,
  and existing errors/orders. Identify the matrix and JAX actions separately.
- Correct result reuse to include geometry/graph identity, implementation,
  end time, field/source identity, acceptance-contract version, and requested
  checks. A cached case without requested timestep qualification cannot pass
  through reuse. Preserve historical outputs without relabeling them certified.
- Separate completion, invariant checks, reference qualification, operator
  convergence, and solution convergence. A disabled check is unqualified,
  not passed. Report static residual orders as well as solution orders.

**Gate:** a reproducible baseline with explicit certification state; stale
geometry or missing checks cannot produce a reused certified result.

### Q01 — Qualify HSX manufactured references

**Dependencies:** Q00.

- Retain the smooth-axis radial field and add smooth angularly varying and
  mixed fields using x=u*cos(theta), y=u*sin(theta), smooth radial envelopes,
  and eta harmonics. Include constants as identity controls. Require
  nontrivial variation within angular owners, not only functions of u and eta.
- Evaluate manufactured fields and independent continuum sources on actual
  HSX geometry. Match all imposed boundary data and distinguish the full
  torus period from a magnetic-coefficient field period.
- Qualify all resolutions against one continuous reference. Stratify bounded
  geometry/source samples across axis, transitions, interior, and boundaries.
  Check the actual source differentiation step and relevant coefficient/metric
  derivatives, not just positions and B at the finest resolution.
- Qualify midpoint/owner conventions with existing physical moments or bounded
  selected-cell integration. Keep geometry/source production separate from
  the trusted simulation consumer.

**Gate:** qualified sources, norms, sampling and boundary conventions, and a
nontrivial angular/mixed HSX field catalogue with a reference-error budget.

### Q02 — Localize the diffusion consistency error on HSX

**Dependencies:** Q01.

- Measure owner representation, mapped transfer/interface data, integrated
  fluxes, and completed owner residuals using matched HSX artifacts.
- Audit forward/backward coupling, conductances, interface measures, tracing,
  wall termination, boundary flux accounting, and topology changes. Record
  owner sizes and interface locations at every resolution.
- Diagnose error with regional residuals and controlled changes on the same
  HSX geometry. Use same-geometry alternative owner topologies only when
  supported and explicitly identified; do not replace the geometry with an
  idealized map.
- Add separate eta/perpendicular refinement experiments as diagnostics.
  Report the fixed-direction error floor; do not require an apparent pure
  refinement order when another spatial error remains fixed.
- Identify actual transition-adjacent owners from topology/interfaces, including
  the unagglomerated side; distinguish those masks from all agglomerated owners.
  Zero change under angular aggregation does not establish physical-average or
  mapped-transfer accuracy. Separate those questions explicitly.
- Instrument the producer on bounded actual HSX interfaces when the saved graph
  lacks directional contributions, quadrature locations, or weighted moments.
  Retain parent provenance and use separate diagnostic outputs; do not alter
  immutable artifacts or silently rebuild them in the consumer. Producer
  instrumentation needed for diagnosis is authorized here, before Q05's shared
  structure promotion.
- Use matched continuum/production transfer, geometry, and flux-integration
  comparisons to distinguish causes. Lack of separate-direction artifacts does
  not by itself block localization if these controls identify the mechanism;
  produce a bounded explicit diagnostic artifact only when necessary. Full
  32/48/64 payload regeneration, a passing convergence slope, and JAX/sharding
  certification are not prerequisites for completing this audit.

**Gate:** an HSX error budget identifies the failing stage and a bounded,
reproducible mechanism. This is an audit gate, not a repaired-convergence gate.
Global conservation or a region's error share alone cannot identify that
mechanism. Report unresolved contributions rather than ruling them out from
weak correlation or a single rescaling experiment.

### Q03 — Repair the identified mechanism

**Dependencies:** Q02.

- Apply the smallest correction supported by the audit. Consider improved
  interface integration, moment-consistent transfer/reconstruction, or
  conservative nonlinear limiting only where the evidence supports them.
- Compare candidates on identical HSX artifacts and manufactured fields;
  preserve independent continuum sources and historical baselines.
- Document the consistency and structural argument for each candidate,
  applicable literature precedent, stencil extent, conditioning, and fallback
  behavior. A theoretical or idealized result does not replace an HSX test.

**Current bounded evidence (19 September 2026):** the pooled-support control
in [`work/parallel_q03_matched_return_experiment_20260919/Q03_MATCHED_RETURN_REPORT.md`](../../../../work/parallel_q03_matched_return_experiment_20260919/Q03_MATCHED_RETURN_REPORT.md)
demonstrates that borrowing distant rows (up to roughly 50--74 scaled cell
widths) is not an acceptable local face map; it does **not** establish a
general return-only impossibility. The corrected, local eta-extension result
is in [`work/parallel_q03_local_eta_extension_20260919/Q03_LOCAL_ETA_EXTENSION_REPORT.md`](../../../../work/parallel_q03_local_eta_extension_20260919/Q03_LOCAL_ETA_EXTENSION_REPORT.md): adding balanced nearby eta intervals restores cubic rank 19 and reduces bounded matched-CV errors below `1e-3` for all three nonconstant fields. This remains a two-owner diagnostic. It does not change midpoint MMS acceptance, establish dissipation/positivity/global order, authorize Q04, or promote a production operator.

The subsequent [completed-action and coverage audit](../../../../work/parallel_q03_structural_coverage_20260919/Q03_STRUCTURAL_COVERAGE_REPORT.md)
finds robust negative off-diagonals and explicit nonnegative-state
minimum-principle counterexamples in both complete sampled rows, so this
unlimited linear candidate cannot be promoted unchanged. Accuracy remains
improved on four ordinary owners. The follow-up
[positivity and agglomerated-tile audit](../../../../work/parallel_q03_positivity_tiles_20260919/Q03_POSITIVITY_TILES_REPORT.md)
constructs a conservative one-row/exterior-ledger AFC prototype that removes
both sampled extremum violations while leaving all three smooth ordinary-owner
fields unchanged to roundoff; its guarantee is only the documented complete
one-row restriction, not a global conservation, dissipation, transient, or
minimum-principle proof. The same audit builds transition and axis owner
boundaries from existing topology and continuous geometry with no producer
change: 404 shared incidences agree, including regularized zero-measure axis
tiles. Thus the previous broad tile-producer blocker is withdrawn.

The later [corrected-reference, exact-observation, and shared-patch audit](../../../../work/parallel_q03_reference_shared_patch_20260919/Q03_REFERENCE_SHARED_PATCH_REPORT.md)
corrects a Cartesian/logical derivative mismatch in that first agglomerated
field reference while preserving its topology evidence. The repaired helper
matches the established ordinary reference within `2.08e-16`; q6/q8
sensitivity is at most `2.29e-6`. Geometry-only saved observations, independent
of numerical G, are degree-1/2/3 compatible on all four selected transition and
axis owners and reproduce cell-average targets within `9.91e-5`. Therefore an
ordinary-only numerical-G donor policy is not an exact-observation blocker; an
actual-G follow-up instead needs a qualified agglomerated state functional.
The prescribed two-receiver projection uses one shared canonical face and
repairs simultaneous sampled extrema without changing MMS actions. Its sampled
two-cell energy rates remain negative, but most witness correction exits via
artificial exterior faces, and the unlimited return has 7--10% errors on the
frozen nonpolynomial smooth extrema. The next bounded Q03 decision is therefore
an actual agglomerated state-functional design plus a closed multi-row
conservative correction that cannot discharge through an artificial exterior
ledger. Q04/global refinement remains unauthorized.

The [revised state-observation and internal-patch audit](../../../../work/parallel_q03_state_refinement_patch_20260919/Q03_STATE_REFINEMENT_PATCH_REVISION_REPORT.md)
supersedes two conclusions in the preserved original report. Centered moments
and geometry/algebra-driven support expansion repair the numerical G failure:
maximum condition and coefficient L1 fall from `3.75e14`/`1.55e14` to
`6.86e5`/`8.70e4`, and the constant mapped-functional error falls from
`8.05e-2` to `2.72e-11`. Actual-state propagated errors nevertheless remain
`1.7e-2`--`1.80e-1` on the transition owners while exact-observation return
errors remain below `9.91e-5`; stable reconstruction therefore does not resolve
the state-to-observation/local-accuracy defect. The fixed physical field result
is preserved unchanged: N32/N48/N64 RMS errors `3.818`, `3.594`, `1.365` and
diagnostic slopes `0.149`, `3.366` are not an asymptotic or global-order result.

The earlier 38-owner stop was an identity error: it used distant receivers
119698/172115 instead of the assigned adjacent pair 119698/123794. The correct
closure has 13 connected ordinary owners, 17 internal faces, and 44 exterior
faces. Its internal-only minimum-change projection is feasible and independently
validated for every required sampled state, leaves exterior corrections exactly
zero, removes all sampled extremum violations, and conserves the active-patch
correction to `1.36e-19`. It is inactive on the MMS and smooth controls; all ten
compact sampled energy rates are nonpositive. This is bounded patch evidence,
not global positivity/dissipation certification. Next bounded work should first
diagnose transition-support state-to-observation error and return locality, then
broaden the internal-only patch coverage only under a separate authorization.

The latest [reconstruction implementation audit](../../../../work/parallel_q03_state_refinement_patch_20260919/Q03_RECONSTRUCTION_IMPLEMENTATION_AUDIT.md)
supersedes the prior endpoint and practical-G conclusions. Frozen initial states
are `1 + amplitude*f`; the earlier order-one endpoint diagnostic omitted the
unit background. Corrected saved-point maxima are only `3.96e-6`--`1.24e-5`.
A geometry/algebra-selected quadratic reconstruction now compares local nearest
and directionally balanced 24-owner supports, with 48 owners selected in only
7,567/784,785 cases. Its maximum condition and coefficient L1 are `38.9` and
`4.06`, versus `6.86e5` and `8.70e4`. Four-owner propagated sample RMS errors
fall from `3.52e-2`--`1.03e-1` to `5.20e-5`--`2.42e-4`; unchanged exact-return
sample RMS is `2.83e-5`--`4.93e-5`. Constants/quadratics pass to `1.98e-14`.
This is a material bounded repair, not a global result. Stage B and the 13-owner
internal correction were preserved without rerun. The next bounded decision is
a stratified ordinary/transition/axis coverage audit of this frozen candidate,
using the same F and references, before any Q04 or production promotion.

The subsequent [candidate coverage and patch-integration audit](../../../../work/parallel_q03_state_refinement_patch_20260919/Q03_CANDIDATE_COVERAGE_PATCH_REPORT.md)
tests 11 frozen ordinary/transition/axis owners and rebuilds the same 13-owner
internal correction with candidate G. Transition/axis gains extend strongly,
but sparse local ordinary row sets require broader support (maximum extent
`11.35`--`13.55` scaled cells) and candidate G is not uniformly better: it
improves 5/11 radial, 10/11 angular, and 4/11 mixed owner cases. Across all 11,
candidate propagated sample RMS is `3.63e-4`--`1.11e-3`, versus unchanged
exact-return RMS `9.84e-4`--`1.62e-3`. On the sharp smooth patch field,
exact-return RMS `2.77` dominates candidate-G RMS `0.108`; the apparent total
change is partly cancellation and does not repair F. The candidate-action patch
remains feasible/KKT-valid with 17 internal variables, zero exterior correction,
`9.49e-20` conservation residual, no post-correction sampled violations, no
positive sampled energy rate, and no activation on MMS/smooth controls. The next
bounded step is the already specified candidate-G 32/48/64 decomposition using
existing row/F/reference artifacts (estimated 26 seconds of bounded G work),
while any return repair should remain exact-lambda-only and localized to the
same smooth patch faces. Q04 remains unauthorized.

The latest [repaired-G refinement and exact-return audit](../../../../work/parallel_q03_state_refinement_patch_20260919/Q03_RETURN_REFINEMENT_REPORT.md)
completes that bounded step. On the frozen six-cell N32/N48/N64 smooth sample,
candidate-G propagation RMS is `0.244`, `0.544`, and `0.00900`, while unchanged
exact-return RMS is `3.818`, `3.594`, and `1.365`. Total-CV RMS is therefore
`3.581`, `4.137`, and `1.373`, with diagnostic slopes `-0.356` and `3.833`:
the N32-to-N48 regression precludes a convergence claim despite the small N64 G
term. Historical G is unavailable in these frozen resolution artifacts and was
not fabricated. An exact-lambda-only audit on all 61 existing patch faces tests
a deterministic cubic-compatible, locality/extent/amplification/quartic-proxy
F selector. It improves individual face-flux RMS but only reduces smooth
completed-action RMS from `3.085` to `2.970` and worsens the worst MMS completed
RMS by `1.416x`; it fails the predeclared promising-candidate gate, so no paired
correction rerun is claimed. The remaining blocker is a conservative completed-
divergence/face-cancellation return formulation, not further field-specific G
tuning. Q03 stays in progress and Q04 stays unauthorized.

The subsequent [joint completed-divergence return experiment](../../../../work/parallel_q03_state_refinement_patch_20260919/Q03_JOINT_RETURN_REPORT.md)
tests that hypothesis on the unchanged 13-owner/61-face patch. A single common
regular-x/y/unwrapped-eta basis replaces incomparable face-centered quartics.
The scaled completed quartic map is full row rank 195 with condition `1.23e3`.
A cubic-nullspace fit using 256--295 rows per face reduces the completed quartic
defect from `5.847` to the q6--q8 discrepancy floor `3.19e-5`, while retaining
common-basis cubic reproduction at `1.06e-13`. Median dimensionless L1
amplification grows `1.25x` (worst face `3.43x`), with no larger mapped-leg
extent than the prior independent alternative. Exact-return RMS improves for
radial/angular but worsens for mixed and smooth; with repaired G, total-CV RMS
improves for all three MMS fields by 25--68% but smooth worsens 14%. The paired
internal-only correction remains feasible/KKT-valid with zero exterior
correction, `1.56e-19` conservation residual, no sampled post-correction
violation or positive energy rate, and no MMS/smooth activation. This is useful
mixed feasibility evidence, not a uniform repair: the next bounded test is a
small degree-five/six geometry-only completed-remainder model on the same patch.
Q03 remains in progress and Q04 remains unauthorized.

The [degree-five/six completed-defect diagnosis](../../../../work/parallel_q03_state_refinement_patch_20260919/Q03_MULTIDEGREE_DIAGNOSIS_REPORT.md)
does not support an upward error-shift mechanism. In the unchanged common basis,
quartic joint F reduces the degree-five completed norm by 21% and degree-six by
1.6% relative to F0; it does not enlarge either block. The prior independent
Fstar, by contrast, enlarges them by 39% and 60%, so its behavior cannot be
attributed to the joint quartic objective. Degree-three constraints remain at
`1.38e-14`, and q8--q10 higher-degree target sensitivity is about one millionth
of the measured defect. No multi-degree fit, field validation, or redundant
correction rerun was performed. The focused next Q03 alternative is a localized
geometry-only support/observation leverage audit on the same faces, not further
polynomial-degree escalation. Q04 remains unauthorized.

That aggregate-only interpretation is superseded by the independent parent
mode audit and the [regularized degree-four/five/six comparison](../../../../work/parallel_q03_state_refinement_patch_20260919/Q03_MULTIDEGREE_COMPARISON_REPORT.md).
Although the aggregate degree-five/six ratios are `0.792` and `0.984`, quartic
joint F increases 16/21 and 18/28 individual modes, with 6 and 5 modes above
`2x`; substantial within-block redistribution therefore justified the bounded
fit.  On the unchanged patch, the selected frozen quartic-anchor candidate
reduces 57/64 degree-four-through-six modes versus F0 and improves repaired-G
total-CV RMS for radial by 43%, mixed by 20%, and smooth by 55%, but angular
regresses by 65%.  The conditional correction retest remains conservative,
feasible, KKT-valid, and nonpositive in all sampled energy tests; simultaneous
minimum and maximum states now activate with correction L2 `4.02e-4`.  This is
useful mixed evidence, not a promotable repair.  Next is one localized
geometry-only observation/support leverage audit on these same faces targeted
at the angular tradeoff and conditioning; do not add polynomial degrees, reopen
G, launch a global campaign, integrate production code, or authorize Q04.

The independent [regularization and exact-angular audit](../../../../work/parallel_q03_parent_regularization_audit_20260919/REPORT.md)
supersedes that proposed broad support/conditioning audit.  Direct weighted
coefficient solves reproduce all three regularized candidates to relative
coefficient error `8.2e-9--8.6e-9`; MMS and smooth actions agree to `2.24e-11`
and `1.07e-8`.  The angular field is exactly degree nine, and its decomposition
shows the regression is a leading-quartic tradeoff: quartic RMS grows from
`2.622e-3` to `3.332e-3`, while degrees five/six and seven--nine improve.  The
two dominant owner changes are 78% and 88% theta-normal-face contributions.
Neither the eliminated high-degree lift nor missing degree-seven--nine content
explains the regression, and missing observations are not established.

The subsequent [quartic-protected comparison](../../../../work/parallel_q03_quartic_protected_20260919/Q03_QUARTIC_PROTECTED_REPORT.md)
uses the same supports and direct cubic-nullspace coordinates.  Its completed
quartic constraint map is full row rank 195 with condition `1.08e3`; both
predeclared ridge candidates preserve Fq's completed quartic response to at
most `2.78e-12`.  Alpha `0.01` remains the field-blind mathematical primary.
The already-frozen alpha `0.1` companion improves repaired-G total-CV RMS
versus F0 by 33% radial, 67% angular, 26% mixed, and 5% smooth, with median/max
amplification `1.48/2.72` times F0.  Its conditional correction retest preserves
zero exterior correction, `4.24e-20` conservation, constants, feasible
simultaneous extrema, and ten negative sampled energy rates, but the established
new-action minimum witness is infeasible with the 17 internal variables.
Therefore it fails the complete sampled structural gate.  The next bounded Q03
step is diagnosis of that single infeasible witness and, only if topology
supports it, one geometry-defined correction-space expansion.  Do not relax
quartic accuracy, add return degrees, reopen G, launch global/refinement work,
or authorize Q04.  Also preserve the selection-provenance correction: the
prior coefficients/family knees were field-blind, but correction candidate
index 7 was chosen later using physical-field geometric-mean validation and is
not an untouched holdout.

The preserved [one-layer correction-collar audit](../../../../work/parallel_q03_correction_collar_20260919/REPORT.md)
repairs the donor-115601 witness by growing the active set from 13 to 42 owners,
but it rejects repeated patch growth as the correction strategy.  Of 3,689
exterior donor columns, 1,018 have negative net patch budget; donor 115474 gives
the exact fixed-boundary contradiction `0 >= 7.24878e-4`.  The fourth compact
seed-260919 state also has positive energy `1.53457e-3` before and `7.58438e-4`
after the extrema projection.  These failures do not isolate the protected
return, because the collar is a mixed protected/baseline construction, but they
do show that an immutable artificial exterior ledger cannot provide the needed
general correction or dissipation route.  Do not expand the collar again.

The subsequent [domain-consistent shared-flux experiment](../../../../work/parallel_q03_shared_flux_limiter_20260919/Q03_SHARED_FLUX_LIMITER_REPORT.md)
implements the supported alternative on the actual N64 owner-overlap graph:
202,304 compact owners and 852,280 nonnegative pair links.  Its verified
low-order action is conservative, volume-weighted symmetric, constant
preserving, an M-matrix generator, and dissipative by the pairwise energy
identity.  A typed local union uses 111 low links incident to the original
13-owner core plus the frozen 61 protected high faces; every accepted flux is
scattered equally and oppositely to its two endpoints.  At unit limiter the
union reconstructs the frozen high core action within `5.68e-14`; at zero it is
the same verified global low-order fallback.  All 34 catalog states have a
feasible zero endpoint, solve successfully, satisfy the complete-support
sampled extrema constraints, conserve mass to `1.73e-18`, and have nonpositive
homogeneous quadratic energy.  This includes donor 115601, the donor-115474
net-budget obstruction, and collar random state 3.  The limiter remains unity
to numerical precision on the three nonconstant MMS fields and three smooth
probes, retaining core total-CV RMS `2.139e-3`, `6.651e-4`, `2.863e-3`, and
`2.898` for radial, angular, mixed, and smooth.  This closes the fixed
artificial-boundary correction blocker, but it is still a localized 61-face
high candidate with homogeneous boundary data.  It does not establish a global
high-order operator, finite-step positivity, nonzero-wall handling, global
convergence, JAX/shard certification, production promotion, Q03 completion, or
Q04 authorization.  The next bounded Q03 decision is a geometry-defined global
high-face extension under the same shared-edge, fallback, extrema, energy,
boundary, and provenance contract; completed-quartic preservation remains the
frozen candidate construction here, not a permanent acceptance gate.

The follow-on [whole-domain shared-flux experiment](../../../../work/parallel_q03_global_shared_flux_20260919/Q03_GLOBAL_SHARED_FLUX_REPORT.md)
builds that extension on the full N32 artifact: 25,376 compact owners, 110,492
verified low links, 63,688 repaired-G observation rows, and 90,880 canonical
physical subfaces across all three coordinate-normal families.  The F0 cubic
fit and global completed-quartic Fq remain sparse, conservative, and constant
consistent, but the documented global normalized-moment relaxation of the
protected ridge-0.1 degree-5/6 return is invalid.  Its maximum face-row
one-norm grows to `2.12e3`, its three nonconstant Q01 errors are `127–218`
versus `0.027–0.042` for the low graph, and a matrix-free symmetric-part search
finds a positive-energy mode.  The complete-support fallback activates on that
mode and passes sampled extrema, mass, and energy-tolerance checks, but leaves
smooth errors at `88–218`.  The frozen bounded N64 core replay agrees with the
saved nonconstant sparse-QP solutions to roundoff and confirms the important
qualification that all 33 saved nonconstant localized states already had
negative unlimited energy; only the new whole-domain action supplies an active
energy witness.  Because N32 validity failed, the conditional N48/N64 builds
were not launched.  Preserve the canonical-face/F0/Fq/G/boundary/replay
infrastructure, but revise or reject the protected return before any new
refinement.  This is a clear Q03 negative result, not Q03 completion or Q04
authorization.

The subsequent [supervisor audit](../../../../work/parallel_q03_global_shared_flux_20260919/SUPERVISOR_AUDIT_20260920.md)
corrects the interpretation of that negative result.  Incident face moments
were accumulated by same local index even though each face used a different
centered/scaled chart, instead of being binomially transformed into one fixed
owner chart; the protected identity penalty also acted on arbitrary reduced
moments rather than the declared weighted reconstruction coefficients.  The
reported amplification and errors remain valid for the executed protected
action, but they do not test the intended common-polynomial objective and do
not reject all higher-degree methods or the cubic F0 construction.

The [corrected compact whole-domain experiment](../../../../work/parallel_q03_compact_corrected_20260920/Q03_COMPACT_CORRECTED_REPORT.md)
implements the full face-to-owner polynomial transform and matching adjoint,
qualifies fixed q3 face quadrature on N32/N64 actual-HSX samples, and builds the
predeclared 19-mode cubic return on N32/N48/N64 with exactly 40 balanced rows
per face.  Conservation, constants, evaluator-free replay, complete-support
extrema, historical states, and the energy fallback pass.  The global weighted
L2 orders nevertheless fail: radial `1.481/-0.818`, angular
`-0.639/-1.299`, and mixed `0.921/-0.685`.  The optional coefficient-metric
ridge-0.1 quartic companion is numerically valid (bounded direct coefficient
equivalence at `6.03e-15`, maximum row L1 below `0.67`) but also fails, with
finest orders `-1.798/-1.858/-1.425`.  Thus the corrected implementations are
complete and structurally qualified, while convergence is not.  Do not launch
N128, promote either candidate, close Q03, or authorize Q04.  The next bounded
Q03 diagnosis should examine the nonmonotone exact-observation return and
reconstructed-G propagation together before selecting another repair; the
decomposition does not by itself identify G as the next change.

The follow-on [owner-observation correction](../../../../work/parallel_q03_owner_observation_correction_20260920/Q03_OWNER_OBSERVATION_CORRECTION_REPORT.md)
confirms that the old global G enforced representative-center moments even
though its input state is a raw-volume-weighted owner mean.  Weighted
centroid/covariance moments restore complete degree-2 owner secants to at most
`1.61e-13`; restoring the previously audited containing-owner and
nearest/directional donor selection materially reduces the smooth action,
whereas correcting moments on the old support does not.  With the compact F
fits unchanged byte for byte, the restored cubic errors at N32/N48/N64 are
radial `0.04785/0.04958/0.04803`, angular `0.06663/0.07900/0.06245`, and
mixed `0.06256/0.07484/0.06698`.  Their finest orders are `0.111`, `0.817`,
and `0.386`, still below the gate; the quartic radial and mixed errors remain
nonmonotone.  A bounded N64 signed attribution finds q3-q6 uncertainty small
and the independently re-aggregated Q01 owner reference identical to the
stored midpoint.  On that sample, the closed-cell q6-balance to qualified
owner-reference leg dominates angular and mixed exact-return error, while
radial is shared with the F-return leg.  Conservation, constants, complete
support, limiter energy, immutable-history, and evaluator-free replay gates
pass.  This corrects the G implementation diagnosis but does not justify a
wider F, N128, Q03 completion, or Q04 authorization.

The subsequent [bounded continuum/reference closure](../../../../work/parallel_q03_continuum_closure_20260920/Q03_CONTINUUM_CLOSURE_REPORT.md)
audits the exact operator rather than treating the prior nine-owner attribution
as a global result.  Deterministic N32/N48/N64 samples include actual
volume-weighted hotspots, ordinary-interior hotspots, regional controls, full
raw-cell unions, complete incident faces, and seven matched physical
neighborhoods.  The samples contain only `0.06–0.57%` of global
exact-observation squared error and `0.51–2.64%` of corrected numerical squared
error, so no global dominance is inferred.  At q10, signed analytical
closed-cell face balances agree with independently differentiated and
J-weighted continuum volume integrals within retained q8/q10 and half-step
uncertainty.  Q01's independent center formula reproduces the stored midpoint
to `1.05e-12`; chi, amplitude/background, eta, covector, J, normalization,
boundary, evaluator, and source identities agree.  The volume-average versus
midpoint term is legitimate sampling error, not a reference bug.  Same-support
q10 face retargeting barely changes the return, so a global q3-target rebuild
is rejected.  A bounded 12-worst-face comparison finds that expanding from 40
to 60/80 rows improves N32/N64 but worsens N48, so uniform widening is also
unjustified.  Corrected-G propagation and nonpolynomial F return both remain
material sampled reconstruction terms.  Next: on 8–12 matched ordinary
hotspot neighborhoods, separate G endpoint-value error from F face-remainder
error with geometry-local adaptive-support indicators and complete-incidence
replay before changing either policy.  No degree increase, global rebuild,
N128, Q03 completion, or Q04 authorization.

The replacement Q03 worker's [matched endpoint/adaptive-return comparison](../../../../work/parallel_q03_matched_endpoint_support_20260920/Q03_MATCHED_ENDPOINT_SUPPORT_REPORT.md)
then tests nine fixed physical neighborhoods across N32/N48/N64 with every
target-owner face and shared incidence retained.  A geometry-only cubic G3
uses actual degree-three owner moments and compact nearest/directional supports;
numerically weak primary fits expand to directional-48/nearest-96 using an
explicit floating-point roundoff indicator.  Final maximum condition is
`1.81e3`, coefficient L1 is at most `13.91`, and degree-three completed secants
reproduce to `7.59e-14`.  The identical-support quadratic control shows G3 is
better in 8/9 frozen-F0 physical comparisons, so the result is primarily a
degree effect rather than donor widening.  A field-blind cubic F selector over
the original 40-row and balanced 60/80-row supports improves exact-lambda
physical-face return in 8/9 cases; N48 mixed worsens by 13%.  Combined
Fadaptive+G3 improves 8/9 cases against both analytical face balance and the
frozen midpoint; N32 mixed regresses because the old F0+G2 cancellation was
favorable.  Constants, polynomial controls, shared-face conservation, and the
signed G2 source/target attribution pass, with target-endpoint error dominant
on this sample.  The sample contains only `0.0036%–0.924%` of preserved global
squared error, so this selects a candidate but does not establish global order
or structure.  Next: one frozen-policy global 32/48/64 static build/action
replay, estimated at 60–90 CPU minutes and about 5 GiB peak RSS; run the full
limiter/positivity/dissipation/JAX/sharding qualifications only if accuracy
passes.  No N128, production integration, Q03 completion, or Q04 authorization.

A supervisor audit then found that the adaptive-F selector averaged periodic
logical endpoint angles before conversion to the face-local regular chart.
The preserved original campaign is therefore superseded for selector decisions
by the [corrected bounded successor](../../../../work/parallel_q03_selector_corrected_global_20260920/bounded_report.md).
The successor converts each endpoint separately to the common regular chart and
only then averages the chart coordinates.  Whole-period theta/eta shifts change
the N64 seam witness by at most `1.78e-15`, and the optimized global builder
reproduces the corrected bounded 12/16-per-plane donor sets exactly on every
bounded face at all three resolutions.  `Fadaptive+G3` still regresses angular
and mixed fields at N32, but it improves the historical corrected `F0+G2`
action for all three fields at both N48 and N64 against both analytical
physical-face balance and the frozen midpoint.  At N64 the physical errors are
`0.00970/0.00975/0.01281`, versus `0.74195/0.66488/0.99171` for the historical
control.  Constants, cubic secants, endpoint attribution, shared incidence,
candidate validity, and historical replay pass.  This activates only the
previously authorized frozen global N32/N48/N64 static build and action replay;
it is not a convergence result and does not authorize N128, Q04, or production.

The resulting [frozen global static study](../../../../work/parallel_q03_selector_corrected_global_20260920/global_report.md)
rejects the corrected candidate.  Global `Fadaptive+G3` midpoint errors at
N32/N48/N64 are radial `0.03062/0.03916/0.02091`, angular
`0.04176/0.06492/0.03458`, and mixed `0.04495/0.05238/0.02762`.  All three
N32-to-N48 orders are negative (`-0.607/-1.088/-0.378`), below the frozen
two-interval `1.8` gate, although the N48-to-N64 orders are
`2.181/2.190/2.225`.  The exact-lambda adaptive-F control has the same N48 bump
(`-0.757/-1.100/-0.458` on the first interval), while N48 G3 propagation errors
are only `0.00321/0.00173/0.00340`.  This was initially interpreted as
localizing the rejection primarily to the nonmonotone global F return rather
than G3 propagation, but that attribution was unsupported because the
exact-lambda-minus-midpoint vector still contains the integrated-continuum-face
balance minus midpoint-sampling term.  Ordinary-interior
owners contribute `88.5–98.2%` of primary squared error, so it is not a
boundary- or axis-only effect.  Constants, conservation, reverse scatter,
candidate validity, historical replay, and evaluator-free replay pass, and N64
still improves the historical F0+G2 error by factors `2.30/1.81/2.43`.  Those
fine-grid benefits cannot waive the frozen gate.  The conditional structural
limiter stage was therefore not entered.  Next Q03 work, if authorized, should
diagnose the resolution-dependent ordinary-interior F exact-return bump around
N48 without reopening G, adding a candidate sweep, or launching N128.  Q04 and
production promotion remain unauthorized.

The [full-domain integrated-reference successor](../../../../work/parallel_q03_integrated_reference_global_20260920/report.md)
corrects that attribution without rebuilding F, G, geometry, or tracing.  It
constructs one independent continuous physical-face flux per canonical face,
using the qualified q5 rule at N32 and q4 at N48/N64, signed incidence, and the
frozen owner volume.  Selected-q saved-action checks agree to at most
`4.38e-16`; the constant flux/action is exactly zero; and evaluator-free replay
passes.  Against this reference, global `Fadaptive+G3` errors are radial
`0.01410/0.005565/0.005177`, angular `0.01611/0.006108/0.005050`, and mixed
`0.01766/0.007887/0.007092`.  The N32-to-N48 orders now pass at
`2.293/2.391/1.988`, confirming that the old N48 midpoint bump was sampling
aliasing, but the N48-to-N64 orders are only `0.251/0.661/0.370`; the frozen
two-interval gate therefore still fails.  The independently referenced
`Fadaptive_exact` fine-interval orders are `0.641/0.697/0.638`, so a genuine
exact-observation return plateau remains, with G3 changing but not solely
causing the primary error.  At N64 ordinary-interior owners contribute
`89.5–93.1%` of primary squared error.  The selected-reference uncertainty is
at most `1.92%` of measured full-domain primary error, so no global q10/q12
extension is indicated.  The limiter is not yet the next bounded test: first
localize the fine-interval ordinary-interior exact-return plateau and its signed
interaction with G3 from the saved actions.  This fixed-time correction does
not automatically make evolved MMS forcing or its time derivative consistent;
N128, Q04, Q03 completion, and production promotion remain unauthorized.

The follow-on [fine-grid ordinary-hotspot audit](../../../../work/parallel_q03_fine_slowdown_hotspot_audit_20260920/report.md)
uses 25 matched complete-owner witnesses and all 148 incident faces at N48/N64,
selected from the saved corrected actions without reopening geometry, tracing,
supports, or donors.  Replacing q3 face targets by q7 changes the sampled N64
primary RMS by only `0.58–1.80%`, does not improve the matched N48 sample, and
barely changes the deliberately local negative refinement orders.  The q4/q9
reference difference is at most `4.27e-5`, so target or reference quadrature is
not the sampled cause.  In contrast, unresolved quartic F response grows from
an N48 median `8.97e3` to an N64 median `2.97e4` and correlates with sampled
`|E_F|`; after a q5 owner mean, G endpoint reconstruction accounts for
`99.3–100.9%` of the signed saved-G projection while owner-state input bias is
negligible.  The bounded conclusion is a shared cubic reconstruction ceiling,
not a q3-target defect.  Next, test one field-blind quartic-completed F/G
reconstruction on the same frozen owners, faces, supports, and high-order
targets, with rank/condition/L1/extent, constants, and conservation gates before
any global rebuild.  No support sweep, limiter stage, N128, Q03 completion,
Q04, or production promotion is authorized.

The subsequent [direct-local quartic factorial experiment](../../../../work/parallel_q03_local_quartic_factorial_20260920/report.md)
tests F3/G3, F4/G3, F3/G4, F4/G4, and exact-observation controls on
the same complete-owner sample with q9 targets.  It is distinct from the prior
globally regularized quartic method.  Direct F4 is usable on all 144 internal
faces after a single nested field-blind support branch activates on 0 N48 and
45 N64 faces.  F4/exact improves all six sampled errors, but F4/G3 regresses
all N48 fields by `16.2–110.8%` and creates large errors at quiet N64 controls;
quartic complete-owner residuals vanish while the first unresolved degree-five
median grows by factors about `4.2` and `7.5`.  Direct G4 remains incompatible
at 427/794 endpoints after its one allowed expansion, leaving zero complete
G4 owners, so no combined result is inferred.  The useful positive control is
support rather than degree: the matched expanded F3 support reduces N64
hotspot RMS materially without changing the N48 or quiet-control actions.
Next perform one error-blind bounded F3 support-only validation with G3 frozen;
do not launch a global build from this selected sample.  Dissipation,
positivity, the global `>=1.8` contract, limiter work, N128, Q04, and production
promotion remain open and unauthorized.

The corrective [direct G assembly and angular-coverage experiment](../../../../work/parallel_q03_g_angular_coverage_20260920/report.md)
supersedes the preceding experiment's blanket quartic rejection.  Direct
centered/scaled raw-cell accumulation shows that all 427/794 former N48/N64 G4
failures are rank-14 systems rather than the previously reported rank-15
systems; the absolute-moment/binomial translation had suffered cancellation.
Their nonzero targets remain incompatible with the four-ray row space, which
misses a genuine angular direction.  One error-blind expansion adding the next
left/right angular columns with two radial owners each makes all 5,271/6,638
required endpoints usable; no second expansion is needed.  Corrected F4/G4
improves all six sampled overall q9 errors by `7.3–89.2%` and tracks F4/exact,
so quartic reconstruction remains a viable higher-complexity challenger.  It
also retains large N64 nearby-control errors already present in F4/exact.
Therefore the safest lowest-complexity candidate remains F3m/G3, while the next
single bounded action is a fresh error-blind complete-owner N48/N64 validation
of frozen F3m/G3 (primary) and F4/G4 (challenger) against F3/G3.  No global
build, limiter, N128, Q04, or production promotion is authorized.

While the subsequently authorized global comparison continued through N64,
the bounded [flux-cancellation and weak-mode audit](../../../../work/parallel_q03_flux_cancellation_audit_20260921/report.md)
replayed 15 N32 and 12 N48 owners with all 194/214 incident canonical faces.
Saved actions and the independent face-error decomposition close to
`1.33e-15`; q9/q13 uncertainty is at most `0.864%` of a non-negligible F3/F4
face difference.  The dominant F4/exact regressions have both larger
individual face errors and, on the strongest N48 boundary-footprint owners,
less cancellation.  Their actual physical-boundary face errors remain exactly
zero, so adjacent internal reconstruction—not the common boundary overwrite—
causes those witnesses.  F3m/exact is essentially identical to F3/exact while
F4/exact changes sharply on the same support.  On dominant faces the quartic
solve has target-coupled weak modes, larger coefficient amplification, and a
much larger first omitted-degree response than ordinary controls; coefficient
replay is roundoff-level, so floating-point solve error alone is not supported.
G4 can amplify or cancel the defect but cannot explain F4/exact.  This does not
yet prove that the same mechanism explains the aggregate ordinary-interior
global SSE.  The smallest supported follow-up is one geometry-only,
directionally balanced support layer on the same six diagnostic faces per
resolution, holding degree, q9 target, charts, exact observations, and G fixed;
do not automatically launch a global rebuild.

The resulting bounded [directional-support comparison](../../../../work/parallel_q03_directional_support_comparison_20260921/report.md)
retains those exact six faces per resolution and every original selected row.
It adds the same number of rows under a common coverage rule, comparing a
target-coupled rank-one information score with a distance-only control.  On
all 21 regression face/field cases, information-enriched F4 reduces the
independent exact-observation error.  Relative to original F4, regression
error-RMS ratios are `0.068/0.628/0.028` at N32 and
`0.0034/0.0116/0.0073` at N48 for radial/angular/mixed fields; the
target-weighted amplification ratios are `0.033` and `0.0053`.  The
same-count distance control helps but is weaker in every aggregate regression
field.  Ordinary controls remain mixed, especially at N48, and normalized
degree-five response is not monotone.  This supports inadequate directional
information as a causal part of the dominant local regressions, but it does
not generalize to the global ordinary population or authorize a global
rebuild.  Await the running N64 comparison before the next global decision.

The authorized [complete-cell directional-support comparison](../../../../work/parallel_q03_complete_cell_directional_support_20260921/report.md)
then applied the frozen information and same-count distance policies to all
194/214 incident faces of the original 15/12-owner N32/N48 samples.  Exact
information-quartic regression-owner RMS ratios are
`0.086/0.229/0.244` at N32 and `0.0072/0.0154/0.0095` at N48 for
radial/angular/mixed fields; fixed G3/G4 observations preserve the benefit and
do not reverse any original-regression owner.  Ordinary-control aggregate RMS
also does not regress, though individual and regional controls remain mixed.
Assembly reveals a qualification constraint absent from the face-local test:
the distance control beats the information score for N32 mixed and N48 angular
through stronger cancellation despite larger individual face-error budgets.
All candidate fits are full row rank with no fallback, and evaluator-free
complete-cell replay closes exactly.  This supports a staged N32-first global
qualification of both enriched cubic and quartic on identical support, with
distance quartic retained as an assembly control, only after the running N64
baseline is finalized.  It does not authorize that rebuild automatically.

#### Enriched-observation evidence and candidate-selection decision

Keep the following distinction explicit in future handoffs: the running
[32/48/64 baseline comparison](../../../../work/parallel_q03_global_quartic_comparison_20260920/)
uses the earlier F3/F3m/F4 supports, including limited usability-triggered
expansion. It does **not** include the new systematic information-based
enrichment. Its completed N32-to-N48 global orders are `2.232/2.363/1.907`
for F3/G3 and `0.321/1.684/-0.376` for F4/G4, in radial/angular/mixed order.
The second interval is still needed to qualify cubic; no baseline pass is
claimed here.

The linked face and complete-cell experiments establish bounded evidence for
improving directional information, not merely increasing donor count. The
enriched rule retains original observations and adds up to eight rows on each
of five existing eta planes, typically increasing support from 80 to 120 rows.
Selection uses the geometry, polynomial observation matrix, and requested
face-flux functional; it does not use MMS field values or errors. A same-count
distance control separates information selection from support size.

Both cubic and quartic were fitted on the **same quartic-informed enriched
support**. Enriched cubic wins all three N32 regression-owner exact-error
comparisons; enriched quartic wins all three N48 comparisons. This does not
test a cubic-optimized selector or establish either global order. Ordinary
control aggregate exact errors improve for all fields, but individual and
regional controls remain mixed. The normalized omitted degree-five response
can increase, and distance selection wins two assembled regression aggregates
through cancellation. Thus enrichment remains an evidence-backed alternative,
not a required component of every reconstruction or a certified global repair.

**Setup cost:** the current diagnostic selector makes up to 40 sequential
additions per face. At each addition it rebuilds selected-row moments,
recomputes an SVD, and scores remaining candidates in Python; several target
projections are repeated for every candidate. The complete-cell experiment
measured roughly 37--39 seconds of selection for 190--211 internal faces per
resolution, versus 5--6 seconds for fitting/evaluation. Extrapolating this
small sample gives about five hours of serial selection alone at N32; this
is an unvalidated estimate, not a global runtime measurement. Before any
enriched global campaign, profile representative batches, cache immutable
moment columns and selected coefficients, vectorize candidate scores, and
test bounded parallel face batches. Preserve the frozen numerical policy
and check selected supports/weights against the diagnostic implementation.
Selection is geometry/setup work for fixed geometry and policy; it must not
be repeated at every RHS evaluation. Larger sparse supports still increase
stored coefficients and action cost, which should be measured separately.

The bounded [one-off setup optimization](../../../../work/parallel_setup_optimization_O_20260921/report.md)
implements that preflight without changing the frozen numerical policy. On
fresh N32/N48 spawned processes including imports, artifact/context loading,
in-memory setup, assembly, and required writes, the representative pipeline
improves by `1.08x/1.20x`. Within the affected stages, face polynomial
arithmetic improves by `11.85x/14.07x`, endpoint-row setup by `2.23x/2.13x`,
and eight-face enriched selection by `6.27x/5.53x`; continuous evaluator work
still dominates face targets. Selected support IDs/order and CSR structure are
unchanged on the frozen samples, with action differences at roundoff scale.
Two warm workers improve face-batch throughput by `1.71x`, but startup makes
them slower for the small one-off sample. These are bounded CPU setup results,
not a global runtime, persistent-cache, N64, action-cost, or GPU-scaling claim.

**Recommended decision if baseline cubic qualifies:** if the unchanged
F3/G3 action passes the agreed global operator gate on both intervals for
every field with qualified references, freeze that complete configuration
as the working diffusion reconstruction baseline and proceed with the
remaining Q04 checks. Quartic failure is not a reason to require enrichment
of a passing cubic candidate. Record observation conventions, support rule,
scaling, geometry/flux functional, boundary treatment, precision, and assembly
alongside degree; "cubic" alone is not a reproducible design. This would
establish the accuracy milestone for that action, not certify arbitrary
cubic fits, other operators, or production readiness. Existing solution,
structural, and execution qualification remains in Q04; no additional
enrichment study is a prerequisite for a passing baseline. If cubic does
not qualify, the frozen enriched-support comparison remains the supported
next global candidate, subject to explicit authorization and runtime preflight.

The [completed baseline cubic qualification](../../../../work/parallel_q03_global_cubic_qualification_20260921/report.md)
now supplies the missing N64 result independently of the stopped quartic
branch. Original-support `F3/G3` errors at N32/N48/N64 are radial
`0.01384/0.005599/0.005154`, angular `0.01587/0.006088/0.005042`, and mixed
`0.01717/0.007927/0.007071`. Although all N32-to-N48 orders pass, the
N48-to-N64 orders are only `0.288/0.656/0.397`; all three fields therefore
fail the two-interval `1.8` gate. The analytic-observation `F3_exact` control
also has second-interval orders only `0.675/0.692/0.662`, and qualified
reference uncertainty is at most `0.26%` of the N64 candidate error. Ordinary
interior owners account for `89.7--93.1%` of N64 squared error. Computation,
incidence/constant bookkeeping, and evaluator-free replay pass, but the static
accuracy gate fails. This rejects unchanged cubic as the working baseline;
the next supported candidate remains the explicitly authorized N32-first
information-enriched comparison after setup-cost preflight. It does not
authorize an automatic launch, N128, Q04 completion, or production promotion.

The completed bounded [N64 cubic observation-enrichment experiment](../../../../work/parallel_q03_cubic_observation_enrichment_n64_20260921/report.md)
tests the failed original-support cubic directly rather than carrying the
earlier quartic-informed selector.  Its frozen 22-owner sample contains 12
ordinary exact/G3 hotspots balanced across all three fields, six deterministic
spatial ordinary holdouts, and axis/transition/agglomerated/boundary controls;
all 325 incident canonical faces are assembled once.  Cubic-target information
enrichment reduces all-owner exact RMS to `0.0485/0.1061/0.0686` of original
for radial/angular/mixed, versus `0.5516/0.5270/0.4926` for the equal-count
distance control.  Frozen-G3 ratios remain `0.0954/0.1039/0.0983`, so endpoint
replacement does not reverse the exact gain.  Hotspot and spatial-holdout
aggregates improve in every field; the four regional controls are mixed but
carry only about `3e-9--1e-7` of original exact global SSE.  Identical coverage
tiers and added counts, lower coefficient amplification, and much stronger
signed-face cancellation distinguish information scoring from donor count;
the median omitted quartic response is not improved monotonically.  Original
coefficients reproduce exactly, selected actions replay to `3.92e-15`, the
face-to-cell decomposition closes to `6.09e-17`, and evaluator-free replay is
exact.  This supports an explicitly authorized N32-only staged global test of
the frozen cubic-information rule with the distance control, but it is not a
global convergence result and does not automatically launch that test, close
Q03, authorize Q04/N128, or promote production code.

The subsequent [cubic-enrichment setup optimization](../../../../work/parallel_q03_cubic_enrichment_optimization_20260921/report.md)
is implementation evidence only and launches no global candidate.  Its cached
`F_M3` vector score reproduces the saved scalar support order, coefficients,
exact/G3 fluxes (including constants), and canonical shared-face actions
exactly on all 324 saved N64 internal faces.  On a cold deterministic 96-face
N32 cold trials it reduces selection by 3.90--7.40x and end-to-end setup by
1.51--1.91x.  Two workers remain below the
4 GiB cap at N32 but are slower cold, while the observed 2.23 GiB N64
single-process peak makes two N64 workers unsafe under that cap; the future
design is therefore serial, compact-chunked, and checkpointed.  Setup-only
serial working projections are 0.73/2.46/5.81 h for N32/N48/N64; the envelope
combining cold-trial variation and ±30% per-face uncertainty is
0.35--1.25/1.17--4.21/2.78--9.94 h.  These projections are not accuracy or convergence
evidence and do not authorize the staged global study.

The authorized [N32 cached-rank-one global enrichment campaign](../../../../work/parallel_q03_cubic_enrichment_global_n32_rank1_20260921/report.md)
completed all 90,880 canonical faces and passed exact coverage, finite-action,
shared-incidence, payload-hash, and constant checks (constant maximum
`2.16e-12`). Relative to original-support F3, enriched global N32 exact/G3 RMS
ratios are radial `0.945/0.713`, angular `0.663/0.659`, and mixed
`0.958/0.804`. Thus frozen-G3 error improves in all three fields by
`19.6--34.1%`, but the global gains are much smaller than the earlier bounded
22-owner result. This is encouraging first-resolution evidence only: it does
not establish an order, authorize N48 automatically, complete Q03, or promote
the research selector.

The subsequently authorized [N48/N64 continuation and combined N32/N48/N64
report](../../../../work/parallel_q03_cubic_enrichment_global_n48_n64_rank1_20260921/report.md)
completed all 307,152/726,528 canonical faces with exact evaluator-free action
replay, full coverage, finite actions, shared-face incidence, and constant
maxima `7.38e-12/1.26e-11`. Serial compute took `0.929/2.222 h` with measured
peaks `1.384/1.740 GiB`; no selector fallback, close-case rebuild, or ambiguity
replay occurred. Relative to original-support F3, N48/N64 exact/G3 RMS ratios
are radial `1.022/0.911` and `0.726/0.593`, angular `0.750/0.759` and
`0.656/0.638`, and mixed `0.809/0.753` and `0.671/0.591`. The enriched
candidate's exact orders are radial `1.406/1.863`, angular `2.050/1.160`, and
mixed `1.924/1.311` for N32-to-N48/N48-to-N64; frozen-G3 orders are
`1.628/1.779`, `2.014/1.263`, and `2.071/1.237`. Thus no field passes the
predeclared `>=1.8` gate on both intervals. Qualified reference uncertainty is
only `0.20--0.41%` of the N64 candidate errors and does not explain the failed
slopes. Boundary faces contribute `13.5--33.7%` of N64 frozen-G3 SSE while
ordinary faces carry the remainder. This is a validated negative result for
the frozen candidate: Q03 remains open, and no Q04, N128, limiter, positivity/
dissipation, fixed-time solution, or production claim follows.

The follow-on [complete-cell consistency audit](../../../../work/parallel_q03_enriched_consistency_audit_20260921/report.md)
replays the frozen candidate at six N64 exact-action hotspots and six
residual-blind holdouts, mapped to ordinary-interior N48 owners with complete
incident-face assembly. Saved actions replay to `4.68e-17`; qualified-reference
incidence closes exactly; common-owner-chart transforms close to `2.08e-15`
relative; and cubic cell moments close to `3.41e-14` relative. Physical support
extent contracts by a median factor `0.755`, but dimensionless extent grows by
`1.417` at hotspots versus `0.970` at holdouts. Physicalized common-cell
quartic response grows by `2.422` at hotspots and falls to `0.324` at holdouts;
relative to `h^2`, the ratios are `4.314/0.573`. Hotspot exact error, absolute
face-error budget, and surviving cancellation fraction worsen together by
median factors radial `7.423/4.298/2.194`, angular `4.417/2.160/2.045`, and
mixed `3.740/2.881/2.577`. Condition maxima grow (`3.527x` median), but maximum
weighted coefficient amplification is essentially flat (`0.990x` median) and
condition correlation with N64 error is weak/inconsistent. These data establish
a localized support-extent/leading-moment/assembly-cancellation interaction,
not an exclusive cause: the six hotspots cover only `1.03--1.49%` of N64 global
exact SSE, one N48 mapping is ambiguous, and symmetry-related pairs reduce
effective diversity. The smallest supported next comparison is an extent-matched,
same-count/same-tier control on only those hotspot incident faces, measuring
physicalized quartic response, absolute face budget, and signed cross terms.
It is a recommendation, not authorization for a changed candidate or global run.

The supervisor's [selector replay on these same hotspots and holdouts](../../../../work/parallel_q03_selector_hotspot_replay_20260921/report.md)
clears the cached rank-one optimization as the cause of the sampled defect.
On all 72 complete incident faces at each N48/N64, repeated-SVD vector ranking,
the original scalar full-SVD selector, and cached rank-one ranking select
identical supports and all 40 additions in identical order. Final coefficients,
exact/G3 fluxes, and complete-cell actions/errors agree bitwise. Intermediate
selected-score differences reach `3.03e-10`/`6.22e-9` relative but change no
selected row; this is sampled output equivalence, not a bound on every global
selection. Restoring full SVDs would reproduce the same sampled errors.
Before the recommended extent-control comparison, separate original-support
from enrichment extent and inspect full observation-leg endpoints as well as
midpoints. Preserve residual-blind holdouts; avoid forcing an infeasible
same-count/same-tier envelope. No changed candidate or global rerun follows
from this replay check.

The supervisor then completed the [support anatomy and bounded replacement
controls](../../../../work/parallel_q03_support_extent_investigation_20260921/report.md).
At all six N64 hotspots, original observations already set the largest x
midpoint/full-endpoint extent; additions-only compaction cannot remove that
extent. N48-derived full-endpoint envelope controls preserve the cubic fit and
observation counts while replacing outlying additions or any outlying row.
A follow-up control preserves exact source-plane/sector/direction joint counts.
That same-coverage control gives hotspot G3 RMS ratios `0.890/0.845/0.932`
(radial/angular/mixed), but holdout ratios `1.656/1.226/1.599`; across all 84
affected owners the ratios are `0.916/0.953/0.972`. Exact-observation results
show the same qualitative effect. All fits remain rank 19, with residuals below
`4e-13`, and neighbor changes preserve canonical shared-face incidence.
The same-coverage control needs 61 minimal envelope relaxations and retains a
maximum endpoint/cap ratio of `1.787`; it does not establish exact extent
matching. Hotspot quartic response improves only about 5% in median. These are
bounded placement effects, not a convergence repair or a reason to launch a
global study. Inspect the original-support/leg-geometry constraints behind the
remaining outliers before designing further field-blind whole-support exchanges.
Holdout regressions and condition numbers remain diagnostics, not added gates.

The subsequent [fixed-budget whole-support exchange](../../../../work/parallel_q03_whole_support_exchange_20260921/report.md)
retains those twelve locations and adds twelve residual-blind ordinary-interior
locations at each N48/N64. Each resolution updates all 144 incident faces of
24 owners, affecting 168 owners. Compare the frozen enriched support, exchanges
that protect original rows, and exchanges that may replace any row. Preserve
row/per-plane budgets and initially represented sector/direction coverage;
minimize the existing cubic information objective with SVD-verified exchanges.
The declared 32-to-64 swap extension triggers at both resolutions; only 4/11
faces remain capped. Whole-support exchange reduces median objective by
18.4%/16.7% and replaces a median 36 original rows, but is not a compaction:
median full-endpoint x extent grows 6.2%/9.2%. N48/N64 selected-cell G3 RMS
ratios are `0.931/0.579/0.860` and `0.696/0.693/0.674`; all-affected ratios
are `0.996/0.981/0.994` and `1.131/0.877/0.989` (radial/angular/mixed).
N64 affected exact-row ratios `1.098/0.881/0.984` show that the tradeoff is not
solely G3 reconstruction error. Cubic reproduction, replay, constants and
conservative increments check out. Original-holdout regressions remain
diagnostics, not new acceptance gates. The 144 neighboring owners receive only
partial face updates, so their regression cannot establish the outcome of a
uniform global selector. The next bounded comparison should complete every
incident face of these same 168 owners, reuse the 144 solved faces, and report
the fully updated region separately from the new partially updated fringe.
Keep the objective, cubic basis, candidate pools and fields frozen. Only if
tradeoffs persist under uniform local application should the next design
change address leading unrepresented moments/assembled cancellation. Neither
further donor growth nor stronger minimization of coefficient amplification
is currently a demonstrated repair. No new global run follows automatically.

The [complete-update comparison](../../../../work/parallel_q03_full_update_exchange_20260921/report.md)
then evaluates every incident face of the same 168 owners at each N48/N64:
864 faces, comprising 144 reused fits, 716 newly solved interior faces, and
four unchanged prescribed continuum boundary fluxes. Whole-support G3 RMS
ratios on those fully updated owners are N48 `0.998/1.132/1.029` and N64
`0.963/0.744/0.814`, versus the earlier partial-update N64 ratios
`1.131/0.877/0.989`. Exact-row N64 ratios `0.941/0.765/0.829` confirm that
the improvement is not solely G3 error cancellation. Thus the earlier N64
neighborhood regression substantially reflected mixing changed and unchanged
faces; it does not establish failure of the uniformly applied candidate.
The added residual-blind 84-owner patches improve in every G3 field at both
resolutions (`0.929/0.764/0.853` and `0.984/0.822/0.566`), while the older
N48 holdout patches regress. Local tradeoffs persist, with no requirement that
every cell or coarse-resolution error improve. The new 428-owner partial fringe
is reported separately and must not be confused with the fully updated region.
All reused results replay exactly, cubic fits retain rank 19, prescribed
boundary fluxes are unchanged, and constants/conservative increments check out.
Runs take 91/100 seconds with peaks below 2 GiB. Keep this frozen cubic
whole-support candidate alive; the next accuracy assessment should broaden
spatial coverage, not tune another objective to individual sampled errors.
A residual-blind HSX sample spanning axis/RLP/interior/boundary regions is an
inexpensive decision aid, not a new prerequisite gate; global qualification is
the decisive comparison when its setup cost is acceptable. No global order,
positivity/dissipation or Q04 certification follows from these local results.

**Gate:** the identified defect improves while conservation, dissipation,
constants, and the minimum principle remain valid. Record a structural blocker
if no candidate satisfies the contract; do not promote signed diffusion weights
merely to obtain a better slope.

**Tracking context:** steps 1–4 of the
[whole-support qualification plan](q03_whole_support_qualification_plan.md).
The planned sample is 96 residual-blind owners per resolution, stratified over
axis, transition-adjacent, agglomerated bulk, ordinary interior and boundary;
every selected owner receives a complete incident-face evaluation. This is a
decision aid toward global qualification, not a new local convergence gate.
The latest user decision supersedes local execution: keep the candidate frozen
and use the prepared remote runner for the full N32/N48/N64 comparison.

The parent subsequently completed a [selector implementation optimization](../../../../work/parallel_q03_exchange_optimization_20260921/report.md):
batch replacement scoring by source plane and reuse label/membership bookkeeping,
while retaining SVD verification and scalar ranking fallback near ties. Paired
speedup is 3.36–3.37x. Across 1,900 actual HSX faces at N32/N48/N64, donor paths,
final coefficients and exact/G3 fluxes are identical to the reference. This
changes implementation cost, not the candidate or scientific evidence. Q
should use the optimized implementation after integrated parallel/restart
checks, with explicit implementation identity and no global swap-history
retention. Estimated selector work falls from about 30 to 9 CPU-hours for
the three resolutions; integrated parallel timings must determine the ETA.

### Q04 — Certify the diffusion action

**Dependencies:** Q03.

- Require static global operator convergence for every nontrivial field, then
  independently qualified fixed-time conduction MMS.
- Verify the actual JAX action against the audited matrix/action, smooth-path
  JIT/JVP behavior where applicable, and single-device versus eta-sharded
  agreement. A nonlinear candidate must be checked as an action rather than
  assumed equivalent to a fixed matrix.
- Retain term/regional error budgets, invariant checks, positivity tests, and
  bounded temporal/reference qualification at the finest accepted resolution.

**Gate:** global operator and solution criteria both pass with structural
properties and error budgets qualified. **No shared-structure promotion before
Q04 passes.**

## 3. Phase B — Promote the verified structure

### Q05 — Extract the shared parallel interface contract

**Dependencies:** Q04.

- Extract qualified owner measures, mapped interface geometry, reconstruction
  conventions, and conservative assembly without changing certified diffusion.
- Supply oriented interfaces, source/destination owners, shared measures,
  distances, boundary classification, and qualified reconstruction data.
  Preserve directional information before diffusion conductance symmetrization.
- Reuse producer artifacts and host-side preparation with fixed-shape JAX
  runtime application. Add versioned geometry payload fields only when needed.

**Gate:** the shared implementation reproduces certified diffusion results
and invariants; consumers receive the directional information their operators
require. State and restart layouts remain unchanged.

### Q06 — Certify parallel gradient/divergence pairs

**Dependencies:** Q05.

- Construct the parallel gradient and conservative divergence on the shared
  structure. Distinguish b dot grad from div(b times field), including div(b).
- Audit metric factors, boundary terms, weighted pairing, individual actions,
  and relevant compositions independently on real HSX geometry.

**Gate:** each physical operator meets the global accuracy criterion and its
required conservation/pairing identities. Adjoint compatibility alone is not
an accuracy certificate.

### Q07 — Certify transport and coupled material terms

**Dependencies:** Q06.

- Integrate density flux, temperature advection/compression, velocity
  advection and pressure forces, vorticity transport, and current/electrostatic
  couplings. Reuse established characteristic physics while correcting the
  inconsistent geometric transfer or assembly stages.
- Extend the certified diffusion action to existing temperature and
  viscosity/diffusion channels. Do not introduce new transport physics.
- Use smooth positive MMS states that avoid unnecessary limiter activation;
  test positivity protection separately and report activation near smooth
  extrema. Record the exact selected schemes and fallback use.

**Gate:** individual terms and physically coupled blocks pass HSX global
operator gates. A sum cannot hide a failing component. Retain the applicable
balance and structural identities.

## 4. Phase C — Full parallel-system MMS

### Q08 — Certify the frozen coupled parallel RHS

**Dependencies:** Q07.

- Exercise density, Te, Ti, Vi, Ve, and vorticity, with manufactured phi
  prescribed first. These are six evolved fields plus an algebraic potential.
- Enable all parallel terms in the selected model and disable perpendicular
  evolution terms. Account explicitly for retained local algebraic terms.
- Derive independent term-resolved continuum sources; report individual terms,
  coupled blocks, and complete equation residuals. Use nontrivial fields for
  each certified contribution so zero/cancelled terms do not count as coverage.
- Use the same assembled stage path for source-paired and unforced evaluations
  and verify the source-addition identity. Retain regional and sharding checks.

**Gate:** every nontrivial equation and certified constituent meets the global
operator criterion with independent sources and qualified error budgets.

### Q09 — Certify evolved coupled MMS and integrate

**Dependencies:** Q08. The reconstructed-phi leg additionally depends on
independently certified polarization in perpendicular **P07**.

- Evolve to a fixed physical time with timestep refinement and the same global
  solution-order target. First prescribe manufactured phi; then repeat with
  the independently certified polarization closure. Keep the latter result
  explicitly pending if the cross-roadmap dependency is not ready.
- Verify conservation/balance identities, smooth-path JIT/JVP behavior, and
  single-device versus eta-sharded agreement; qualify temporal, reference,
  and polarization-solver errors separately.
- Promote the verified configuration into the shared model/MMS workflow only
  after the gates pass. Update current-behavior architecture documentation to
  describe the accepted implementation and its limits. Blob-driver
  synchronization remains deferred.

**Gate:** prescribed- and reconstructed-phi coupled solution MMS pass, Q04 and
Q06–Q08 operator gates remain satisfied, and the applicable structural and
execution checks pass under the pinned frozen-MMS boundary contract. This
certifies the numerical configuration; physical sheath-law certification
remains outside this roadmap.

## 5. Interfaces, evidence, and progress ledger

- Preserve evolved-state and restart layouts. Keep geometry production
  separate from simulation consumption; do not silently rebuild artifacts.
- Extend MMS reporting with static operator orders, per-field/per-term
  acceptance, regional contributions, reference qualification, fallback
  activity, complete provenance, and explicit qualified/unqualified checks.
- Each future task receives its ID, dependencies, bounded objective, acceptance
  gate, and required evidence. Record implementation/configuration identity,
  evidence links, measured orders, unresolved defects, and the next bounded task.
- Use `pending`, `ready`, `in progress`, `blocked`, and `passed`. Code or run
  completion is not a numerical pass. Changing geometry, implementation,
  field catalogue, or acceptance rules invalidates certification reuse.
- Preserve historical results. No idealized test, conservation check, or
  passing solution result substitutes for the HSX global operator gate.

The [initial conduction audit](../../../../output/literature/parallel_conduction_mms_audit_2026-09-18.md)
is background evidence, not completion of Q00–Q04. Its existing smooth-axis
32/48/64 results are nonmonotone and do not certify second order. Begin Q00 from
the current harness under
`work/heat_rlp_owner_boundary_overlap_20260916/convergence/` at workspace level
and the package overlap action in `native/fci_rlp_overlap.py`.

The [original Phase A task](thread://01a0b561-d18d-78b1-b85f-bfe3d67efe49?hostId=local), using GPT-5.6 Sol, owns the bounded Q02 amplitude-comparison correction. A separate [bounded Q03 experiment](thread://01a0b5ca-7e77-70f2-a0d9-0a696f29b1f0?hostId=local), also using GPT-5.6 Sol, is authorized concurrently. The correction is not a prerequisite to independent Q03 work. Neither assignment authorizes Q04 certification or Phase B promotion.

| ID | Work package | Dependencies | Status | Evidence / remaining work / next bounded task |
|---|---|---|---|---|
| Q00 | Baseline and result identity | None | passed | [Phase A task](thread://01a0b561-d18d-78b1-b85f-bfe3d67efe49?hostId=local), GPT-5.6 Sol; [evidence](../../../../work/parallel_phase_a_q00_q04_20260918/PHASE_A_REPORT.md#q00--frozen-baseline-and-reliable-identity). Revision `6c2b005`, action SHA `716c2bbe`. Reuse keys geometry/graph, implementation, end time, field/source, contract and requested checks. Wall operator orders -0.980/1.408 and solution -1.004/1.549; smooth-axis -1.007/1.375 and -1.029/1.511. Completion/invariants/time pass; historical references are unqualified and convergence fails. Q01 was next. |
| Q01 | HSX references and field catalogue | Q00 | passed | [Evidence](../../../../work/parallel_phase_a_q00_q04_20260918/PHASE_A_REPORT.md#q01--qualified-hsx-sources). Version-2 real-HSX midpoint sources cover radial–eta, angular-x, mixed-y-eta and constant fields at 32/48/64 with full-torus eta and explicit boundaries. Stratified requested-step sensitivity is at most `6.15e-8` relative; all-resolution position/B/J and bounded selected-owner quadrature are recorded. Q02 was next. |
| Q02 | HSX diffusion error localization | Q01 | passed | [Supplement](../../../../work/parallel_phase_a_q00_q04_20260918/Q02_LOCALIZATION_SUPPLEMENT.md): corrected masks establish 81.9–98.6% of squared residual in ordinary unagglomerated interior. Selected-interface quadrature changes are small. The point-transfer comparison omitted amplitude=0.2 and is being corrected by the original task; its worsening is not valid evidence. Retracing also changes the evaluator, so it is not a pure step-size estimate. The user accepts the existing localization as sufficient to start a bounded Q03 hypothesis test; an exclusive moment defect or asymptotic floor is not yet proven. |
| Q03 | Diffusion repair | Q02 | open — whole-support global accuracy failed | [Remote campaign and local analysis](../../../../work/q03-exchange-tNlxO86M_analysis/report.md): G3 orders radial `1.528/1.949`, angular `2.231/1.634`, mixed `2.019/1.515`; exact rows also fail both-interval qualification. N64 improves 4.76%/13.44%/11.24% over enriched baseline, but no field passes both intervals. Reference budgets, input/source/chunk integrity, and global assembly checks pass. Error lies mainly in ordinary interior/boundary-adjacent owners; omitted face-functional truncation response is the next bounded diagnostic question. Do not infer a G3-only, wall-only or RLP-only defect. No new run dispatched, no cap/degree increase or extra gate imposed; Q04 remains pending. |
| Q04 | Diffusion certification | Q03 | pending | Await a repaired candidate; no operator/solution or JAX/sharding certification pass is claimed. |
| Q05 | Shared interface structure | Q04 | pending | No promotion before certified diffusion. |
| Q06 | Gradient/divergence pairs | Q05 | pending | Independent HSX operator and pairing checks. |
| Q07 | Transport and material blocks | Q06 | pending | Term-resolved HSX gates and diffusion-channel integration. |
| Q08 | Frozen coupled parallel RHS | Q07 | pending | Six evolved equations, prescribed phi, all selected parallel terms. |
| Q09 | Evolved MMS and promotion | Q08; P07 for reconstructed phi | pending | Both phi legs, solution gates, structural and execution qualification. |
