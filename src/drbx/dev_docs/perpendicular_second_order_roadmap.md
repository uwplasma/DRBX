# Roadmap: second-order perpendicular operators on angular RLP grids

Approved research roadmap, 18 September 2026. This document specifies planned
work and acceptance gates, not a claim that the methods below are already
implemented or verified. It is the authoritative progress ledger for future
tasks. Keep numerical evidence and experiment logs in linked research artifacts.

## 1. Objective and acceptance contract

### Current execution decision: clean remote global qualification

The local P global campaign is paused and preserved as historical
evidence. Continue the matched-q3 centered-bracket N32/N48/N64 qualification
as a **new remote campaign**, using
[`scripts/hsx_remote_qualification`](../../../scripts/hsx_remote_qualification/README.md).
The remote handoff contains repository commands only, in
[REMOTE_COMMANDS.md](../../../scripts/hsx_remote_qualification/REMOTE_COMMANDS.md).
Use the available CPU allocation at maximum requested parallelism (64 workers
for the planned allocation); **no remote scaling study is required**. The
remote task owns execution and monitoring; the local P task remains paused.
Worker communications go through the parent task, never another P/Q/O worker.

The new `selection-v3` policy resolves near-equal candidate distances by a
documented anchored tolerance group and canonical owner ID, and applies one
half-open snapping convention to **all eight** angular-sector boundaries.
Do not special-case pi/4 or tune ties to reproduce historical donor IDs.
Rebuild the actual downstream dependency chain under this policy, including
physical-boundary cubic derivative rows and all face/cell reconstructions.
The global study does not consume historical bounded-fit, factor, cross, or
global-derivative caches. Geometry, original manufactured owner data, and
independent qualified references may be reused after content verification.
Historical N32/N48/N64 numerical outputs must not be mixed into the new
campaign, including already completed local global chunks.

Fresh HSX implementation checks remain required; replaying every archived
donor ID is **not** a prerequisite for a new campaign. Record donor identities,
polynomial reproduction, constants, and action diagnostics. Floating-point
agreement uses justified tolerances; a policy change receives a new numerical
identity rather than bypassing a stale-cache check. Candidate-specific
reference budgets and the existing two-interval >=1.8 global operator gate
remain unchanged. Remote execution is not P04/P05 completion or promotion.

### Scientific contract

Retain the RLP owner unknowns and regular chart, and establish second-order
convergence for the **current MMS configuration**:

- Material scalar upwind brackets and the centered vorticity bracket.
- Complete coupled curvature, including the potential remainder.
- Conservative perpendicular polarization and the shared evolved perpendicular
  diffusion action.

Other selectable formulations remain diagnostic controls. Parallel operators
have a separate [Q00–Q09 roadmap](parallel_second_order_roadmap.md).
Blob-driver synchronization remains outside both roadmaps.

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

**Real HSX geometry is the acceptance basis from the first numerical audit.**
Use the actual metric, field-line traces where applicable, angular RLP topology,
axis treatment, and domain boundaries. Idealized maps, slabs, and simple polar
geometries are not prerequisite convergence gates. Small algebra/bookkeeping
tests may remain, but their success does not establish HSX accuracy or complete
an operator milestone. Analytical manufactured fields are still appropriate:
evaluate them and their independent continuum sources on the real HSX geometry.

**HSX-only numerical test policy:** every new or revised numerical accuracy,
convergence, reconstruction, interface, return-map, or boundary test for this
roadmap must target actual HSX geometry. Do not add or run idealized slab,
straight-field, circular-polar, or synthetic-map campaigns as substitute or
preparatory numerical evidence. Cheap regressions may use a bounded extraction
of a real HSX artifact, retaining its metric, topology, measures, traces where
applicable, and boundary semantics; record the parent artifact and selection.
Such a local check diagnoses a mechanism but cannot certify global order.
Geometry-independent unit tests may check algebra, indexing, hashes, and norm
bookkeeping only. Preserve historical idealized results as background without
rerunning them to satisfy milestones. Missing HSX inputs must be reported or
produced through the explicit geometry producer, never replaced by an idealized
fixture. This policy applies to P01–P09, including tests added during repairs.

Begin numerical audits with the existing **32³/48³/64³** HSX artifacts. Keep one
continuous geometry reference across the sequence. Nonmonotone results remain
inconclusive; extending resolution is an explicitly scoped follow-up. Diagnose
failures with regional residuals and controlled changes on the same HSX geometry.

**The primary fast verification gate is second-order global operator convergence.** Retain the fast
midpoint-based MMS as the default convergence check, using independently derived
continuum sources and explicitly documented sampling/averaging conventions.
Use bounded actual-HSX average controls to qualify representation consistency; do not
require high-order quadrature over the full HSX domain. Require observed
global volume-weighted operator-error L2 order **at least 1.8 on both finest
refinement intervals**, assessed per certified operator and nontrivial field/case
rather than hidden in an average across fields. Retain independent elliptic and
fixed-physical-time solution MMS checks with the same global order criterion.

Second-order operator consistency is a stronger verification target than
second-order solution convergence alone; the two are not generally equivalent.
Good solution convergence does not automatically waive a failed global operator
gate. Document such a result and its mechanism for review rather than silently
relaxing the agreed requirement. Local truncation, face, and regional error
orders, including maximum-norm orders, remain diagnostics, not separate second-order gates. Required stability,
conservation, boundary, and algebraic properties remain part of certification.
This operator-plus-solution contract supersedes the interim solution-only rule.

Always report axis, transition, ordinary-interior, and physical-wall errors
separately: normalized RMS, maximum error, physical volume, and contribution to
global squared error. These regional norms need not individually achieve second
order under the selected acceptance contract; their behavior must remain visible.

Geometry-reference and, where applicable, temporal and linear-solve errors must
each be demonstrated to contribute less than **10% of the finest spatial error
in the corresponding operator or solution check**. When an optional
quadrature-qualified reference audit is used, apply the same bound to its
quadrature error. Routine midpoint sampling error may be part of the overall
second-order error budget; it need not satisfy this 10% bound. Qualify its order
with bounded controls on selected actual HSX cells, particularly near the axis and
agglomeration seams. Do not infer order from errors at an independently
established numerical floor.

Use stable task IDs below. Each task records: status, dependency, hypothesis,
implementation revision, configuration, evidence links, measured orders, and
remaining failures.

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

## 2. First audits: establish where order is lost

### P00 — Freeze the baseline and create the progress ledger

**Dependencies:** none.

- Maintain this durable roadmap and task ledger. Store numerical evidence and
  experiment logs in purpose-named research artifacts linked from the ledger.
- Record the exact MMS selections, enabled reconstruction payloads, source
  revision and dirty-file hashes, geometry artifacts, precision, boundary
  conditions, and sharding.
- Reuse the existing operator-only bracket audit and MMS term ledger; extend
  their diagnostics rather than creating a competing full simulation driver.
- Preserve baseline results before changing numerical formulas. Run one
  resolution per process to avoid accumulating geometry and JAX allocations.

**Deliverable:** reproducible baseline manifest and a results table with separate
bracket, curvature, diffusion, and polarization entries. Existing literature/source
audits count as background evidence, not completed numerical verification.

### P01 — Qualify manufactured fields, averages, sources, and norms

**Dependencies:** P00.

- Keep the existing midpoint-only full-HSX MMS as the fast default. Second-order
  convergence alone does not require higher-order quadrature everywhere.
- Qualify **physical owner-average** controls, boundary data, and continuum
  source averages on selected actual HSX cells/owners. Reuse qualified physical
  moments where possible; distinguish exact identities from approximate
  geometry moments. Use bounded optional quadrature on those cells/regions
  when necessary and cache reusable results. Idealized averaging fixtures are
  unit controls only, not reference qualification for HSX.
- Label midpoint and exact-average reference conventions explicitly. Use the
  latter to isolate representation/differentiation defects, not to replace every
  fast convergence test or make full-domain quadrature a setup dependency.
- Use smooth regular-chart fields containing linear, quadratic, cubic, mixed,
  and non-polynomial components. Include x and y, not only quadratic angular
  harmonics.
- Supply boundary data compatible with each manufactured field; for example,
  testing x with unrelated homogeneous boundary data would invalidate the
  diagnosis.
- Derive sources independently of the discrete operator being tested.
- Validate the regional masks, including overlapping diagnostic masks and a
  disjoint partition for error accounting.
- Fix the same continuum geometry reference across a refinement sequence and
  demonstrate its accuracy independently at every resolution with stratified
  samples covering axis, interfaces, interior, and boundaries.

**Deliverable:** reusable reference/measurement helpers and a qualified
manufactured-field catalogue.

### P02 — Audit reconstruction and differentiation stage by stage

**Dependencies:** P01.

Use the actual HSX geometry and automatic angular RLP topology as the primary
case. Where supported, use q=1 and axis-only owner topologies as controlled
diagnostics on the same continuous HSX geometry, with explicitly identified
topology-dependent artifacts. Do not substitute an idealized polar geometry
for the HSX stage audit or regenerate unrelated magnetic artifacts.

Measure separately:

1. Owner averages → reconstructed raw averages.
2. Owner averages → face values.
3. Owner averages → face derivatives.
4. Face data → integrated face contributions.
5. Integrated contributions → completed owner residuals.

Retain the existing analytical f=x first-face 8/9 result as supporting evidence.
Measure the corresponding average-to-derivative discrepancy and its propagation
through each complete operator on actual HSX cells; do not assume that the
idealized factor or its global effect carries over unchanged.

Test 1, x, y, x², xy, y², selected cubics, and smooth non-polynomial fields. For
brackets, include {x,y} and mixed-degree pairs using the implementation's sign
convention.

**Deliverable:** the earliest failing stage for each operator on HSX, including
whether a supported same-geometry control already fails without agglomeration.

### P03 — Audit geometry, interface coupling, return maps, and closures

**Dependencies:** P01; can proceed alongside P02.

**Current HSX follow-up assignment:** use the qualified continuous producer
metric/B reference to re-establish the unchanged production bracket baseline
on the existing 32³/48³/64³ HSX artifacts, and perform the bounded N32
matched-functional comparison below. This includes the necessary supplemental
P01 reference qualification and P02 stage localization. Update each milestone
only against its own evidence and deliver an explicit P04 readiness decision;
do not begin production P04 implementation in this follow-up.

1. Recompute midpoint-projected manufactured inputs and independent continuum
   bracket sources from the same continuous reference at every resolution.
   Cover manufactured vorticity and both existing smooth scalar controls with
   the same generator and frozen boundary contract. Report global physical-
   volume-weighted operator L2 errors and both interval orders, plus regional
   squared-error contributions. An exact common-face discrete oracle remains
   a stage diagnostic, not a substitute for the continuum operator reference.
   Qualify reference/differentiation uncertainty with bounded matched checks;
   retain midpoint MMS as the default and do not introduce full-domain
   high-order quadrature or new resolutions.
2. On the existing eight N32 owners and their complete face/donor support,
   compare the current geometric-moment candidate with a candidate whose
   polynomial observations use exactly the member sample locations and
   physical weights defining the stored midpoint-projected owner data.
   Verify that sampling convention from the implementation first. Keep owner
   values, reconstruction degree, donor-selection policy, production face
   velocity, center/compression term, assembly, and production eta
   reconstruction fixed so the changed observation functional is identifiable.
   Report any unavoidable support changes. Use low-degree reproduction through
   the actual observation functional as an algebraic check, then compare actual
   vorticity and both smooth fields on HSX, especially the non-axis transitions.
   Distinguish this MMS sampling diagnostic from a claim that evolved finite-
   volume averages are point samples. Existing bounded integrated-average
   results are a separate control, not justification for changing stored data.
3. Report face-state errors, signed radial/angular/eta residual changes,
   conditioning, weight amplification, fallback activity, complete support,
   and physical-volume-weighted sample errors. Preserve collapsed-face zero
   flux and retain axis-owner residuals. Update the detailed report, evidence
   identities, and ledger, then recommend whether a candidate warrants global
   qualification. Candidate global qualification is a subsequent scoped step;
   the production 32/48/64 baseline is authorized now.

**Diagnostic versus acceptance gate:** improvement at every sampled owner or
in every region is not required. Regional regressions require explanation and
error-budget reporting, not an automatic veto. The numerical acceptance gate
remains global physical-volume-weighted operator order >=1.8 on both finest
intervals for each nontrivial field, with the applicable structural and
reference requirements. Sample error reduction alone does not certify order.

- Perform the audit on actual HSX geometry from the outset. Use matched
  geometry/source artifacts and actual owner/interface regions for all
  counterfactual comparisons; preserve idealized results as supporting
  diagnostics only.
- Cross exact/production generator derivatives with exact/production transported
  traces to separate bracket errors.
- Compare nonlinear flux evaluated at common quadrature points with products of
  separately averaged quantities.
- Audit shared subface measures, coarse/fine flux summation, metric identities,
  and operator-specific mass factors.
- Separate curvature material updates, potential remainder, and radial/theta/eta
  contributions.
- Test R and H-star = M_owner^-1 H.T M_raw on exact smooth raw averages
  independently of the differentiated operator. Verify R H = I separately.
- Audit axis closure, periodic endpoints, physical Dirichlet and nonzero Neumann
  conditions, and eta-shard boundaries.
- Record reconstruction degree, stencil extent, conditioning, weight
  amplification, and positivity-fallback activation.

**Deliverable:** an HSX error-budget table identifying reconstruction, derivative,
geometry, quadrature, return-map, and boundary contributions.

**Audit gate:** do not begin a broad operator replacement until P02–P03 identify
a failing mechanism on actual HSX geometry and a bounded HSX reproducer. A failure already present on a same-HSX q=1 control
must not be attributed solely to agglomeration. Use stage-localized measurements
to identify what prevents the global operator target before committing to a
broad redesign. P02/P03 themselves remain audits; their completion does not
require operator repair or a passing global convergence result.

## 3. Literature-grounded implementation tasks

The primary approach is **moment-consistent face values and derivatives
evaluated with shared geometry and quadrature**. Keep the existing owner
representation; repair the operations applied to it.

| Literature precedent | Method to carry into this roadmap |
|---|---|
| [Purser, 1988](https://journals.ametsoc.org/view/journals/mwre/116/5/1520-0493_1988_116_1067_andnap_2_0_co_2.xml); [Lima–Peixoto, 2023](https://eartharxiv.org/repository/view/3192/) | Respect regular polar modes when reconstructing and differentiating across angular thinning. Retain the regular-chart basis and test its differentiated modes. |
| [Mignone, 2014](https://arxiv.org/html/1404.0537) | Derive interface quantities from geometric volume moments; distinguish cell averages, centroids, and point values. |
| [Fornax, Skinner et al., 2019, §VII](https://arxiv.org/html/1806.07390) | Coordinate restriction and transverse reconstruction across angular refinement interfaces. |
| [McCorquodale–Colella, 2011](https://msp.org/camcos/2011/6-1/camcos-v6-n1-p01-s.pdf) | Combine polynomial reconstruction, shared-face flux quadrature, and consistent time integration across refinement boundaries. |
| [Almquist–Wang–Werpers, 2019](https://arxiv.org/html/1806.01931) | Design value and derivative transfers jointly with quadrature and stability; adjoint compatibility alone does not preserve order. |
| [Barad–Colella, 2005](https://crd.lbl.gov/assets/pubs_presos/AMCS/ANAG/A158.pdf) | Assess elliptic solution convergence separately from interface truncation error. |

These papers establish applicable methods and precedents; none directly proves
convergence of the current DRBX composition.

### P04 — Build consistent owner-to-face functionals

**Dependencies:** P02–P03.

- Use the existing regular-chart moment machinery to evaluate **values and
  derivatives** at shared face quadrature points from the actual stored owner
  observation functional. The current candidate uses midpoint/raw-volume
  owner moments; do not silently reinterpret these as exact volume averages.
- Start with cubic reproduction. Differentiate the reconstructed polynomial
  analytically, then apply the chart/metric transformation consistently.
- Use common subfaces and geometry on both sides of an agglomerated interface;
  sum their contributions consistently for each owner.
- Cover every face family required by the certified operators. Audit
  ordinary/direct-face transitions explicitly rather than assuming a narrow
  transition patch is sufficient.
- Retain bounded value-row amplification; monitor mesh-scaled derivative-row
  amplification and stencil locality under refinement.
- Keep host-side compilation and fixed-shape JAX runtime application. Reuse
  existing moment-fit prototypes where their contracts match.
- Make unsupported degree, conditioning failure, and fallback behavior explicit
  in diagnostics.

**Interface changes:** extend internal reconstruction payloads with derivative
rows and quadrature provenance. Preserve evolved-state and restart layouts.
Candidate selection stays explicit in MMS/audit configuration; existing geometry
moments should be reused without regenerating unrelated magnetic artifacts.

**Gate:** reproduce the intended polynomial values/derivatives on actual HSX
owners to fitting tolerance for any new functionals and retain the applicable
structural properties before production integration. Bounded experiments
qualify implementation and motivate candidates; beating a baseline error or
improving every sampled owner is not a prerequisite for global qualification.
Repair only
what is needed for the global operator and solution targets; lower local order
alone does not require a repair if the global gates already pass.

### P05 — Repair and certify the brackets

**Dependencies:** P04.

**Current first milestone:** global qualification of the frozen general-cubic,
shared-face, matched-q3 centered bracket C on the original omega, regular,
and eta-varying fields. The [P worker](thread://01a0befd-521c-70f0-96f9-8be67df425fa?hostId=local)
is implementing/preflighting the authorized 32/48/64 study, including global
omega reference construction. No passing global result exists yet. There is
no established requirement to redesign the reconstruction before this study.
The outstanding acceptance items are complete global actions and an independently
qualified global omega reference. Record A/B constituents diagnostically; C is this
milestone's certified operator. Require global physical-volume-weighted L2
order >=1.8 on both refinement intervals for each original nontrivial field,
with the existing reference-error budget. A small predeclared held-out field
check tests generality without tuning or adding a per-owner acceptance gate.

**Scientific checks, not blockers for the current study:** constant annihilation,
argument antisymmetry, shared-face bookkeeping, and checks of actual boundary
branches are lightweight diagnostics reported alongside the convergence result.
Reuse valid evidence and existing assembly data; do not require separate global
runs, repetition at every resolution, or a separate boundary-convergence gate.
Missing or failed diagnostics do not automatically stop the study or veto the
reported global operator-order result. Record their status and scientific
implications separately. This does not make an incomplete domain, incorrect
reference, or corrupted computation valid evidence; correct concrete execution
or bookkeeping errors when they prevent a meaningful convergence measurement.
Structural certification and production suitability remain separate later
decisions, without adding them as prerequisites to this static study.

Material upwinding remains a separate bracket qualification within P05 and
does not prevent an independent centered-bracket milestone. Curvature,
polarization, full-system balance checks, evolved MMS, and production promotion
remain subsequent work; they are not additional prerequisites for this static
centered-bracket result. Completing this first milestone does not complete
all of P05 or certify the shared perpendicular infrastructure.

- Integrate consistent generator derivatives and transported traces into the MMS
  material and centered-vorticity paths.
- Evaluate the complete face contribution with matched geometry and quadrature.
- Preserve the compatible correction and centered bracket identities. A gradient
  change must be checked against the complete compatible algebra.
- Test both operands independently and together; include variable geometry and
  magnetic coefficients.
- Require the centered vorticity bracket to pass independently of material
  upwinding.

**Current operator-accuracy gate:** second-order global owner-residual convergence
for each bracket path being qualified, with qualified references. Report
constant/antisymmetry and shared-face identities as scientific diagnostics under
the current-study rule above, not additional blockers. Interface orders are
diagnostic, not independent second-order gates. Independent
fixed-time solution checks are retained in P09.

### P06 — Repair and certify complete curvature

**Dependencies:** P04.

- Reuse the direct radial quadrature infrastructure.
- Repair remaining face directions, axis/endpoints, and boundaries according to
  the audit.
- Treat the material characteristic update and potential remainder as separate
  diagnostic components, then certify their sum.
- Reconcile quadrature, coefficient products, and the intended V/B versus
  physical-volume normalization through the continuum derivation.
- Preserve the coupled path-conservative balance and compatible geometric
  identities.
- Use positive smooth MMS fields that keep positivity fallbacks inactive; test
  fallback behavior separately.

**Gate:** second-order global residuals for the complete curvature contribution
in each evolved equation, retaining coupled balance and compatible identities.
Report material and remainder residuals separately to expose errors in the
composition; directional and regional orders are diagnostic. Independent
fixed-time solution checks are retained in P09.

### P07 — Repair and certify perpendicular diffusion/polarization

**Dependencies:** P04 and P03's return-map audit.

- First retain the current conservative composition and replace the identified
  inconsistent derivative/flux stages.
- Verify the actual owner action against the owner-averaged continuum operator,
  including variable metric cross terms and physical boundaries.
- If the return map prevents the global operator or solution gate from passing,
  prototype an explicitly consistent mass/source formulation using the same owner unknowns. Keep this
  as a separately identified candidate.
- Derive boundary loads and distinguish physical normal derivatives from
  conormal fluxes.
- Use exactly the same selected physical action for phi inversion, Ti
  polarization, and evolved perpendicular diffusion; exclude solver
  regularization from physical diffusion.

**Gate:** second-order global operator residuals and independent elliptic
solutions, correct nullspace and conservation/boundary balances, and the
applicable energy property. Fixed-time diffusion solution MMS is retained in
P09. Regional operator orders remain diagnostic.

**Decision rule:** promote the smallest candidate that passes all gates. If both
the retained composition and the explicit mass/source candidate fail, record
the structural blocker and reopen the numerical design; do not redefine the
reference or weaken the success criterion.

## 4. MMS certification, integration, and progress tracking

### P08 — Certify the combined frozen HSX perpendicular RHS

**Dependencies:** P05–P07.

- Use the current MMS configuration with parallel terms disabled or reported
  separately.
- First prescribe exact manufactured phi; then repeat with reconstructed phi to
  expose polarization error.
- Report each perpendicular term and their sum.
- Use the same **32/48/64** HSX sequence established in the first audits. Any
  extension to resolve asymptotic behavior is an explicitly scoped follow-up;
  an inconclusive sequence remains incomplete.
- Recheck geometry accuracy and the qualified reference convention on the finest
  accepted mesh; use bounded reference spot checks rather than mandatory
  full-domain high-order quadrature.
- Test matched single-device and eta-sharded execution.

**Gate:** the certified operator contributions and the combined perpendicular
residual meet the global operator-order criterion, with qualified references
and matched sharding/source/phi diagnostics. Keep term-resolved and regional
error budgets so a summed residual cannot hide a failing operator. No separate
regional second-order gate is imposed. P09 independently checks solutions.

### P09 — Certify evolved MMS and promote the method

**Dependencies:** P08.

- Run fixed-final-time perpendicular MMS with independently controlled temporal
  and solver errors.
- Include diffusion-only evolution, bracket/curvature evolution, and the coupled
  perpendicular system.
- Recheck phi reconstruction and source/boundary pairing.
- Add compact regression tests for the discovered failure mechanisms; keep
  expensive convergence campaigns as reproducible research artifacts.
- Verify JIT, JVP/gradient behavior on the smooth path, sharding agreement, and
  record compilation/runtime/memory costs.
- Update architecture documentation to describe the accepted implementation
  and its verified limits.
- Promote the passing configuration in the MMS workflow. Blob-driver
  synchronization remains deferred.

**Gate:** independent fixed-final-time MMS solutions meet the global
volume-weighted L2 order criterion in Section 1 for the isolated controls and
coupled perpendicular system; elliptic solution controls also pass. Temporal,
linear-solve, and geometry-reference budgets are qualified. P05–P08 global
operator gates also pass; local and regional error orders remain diagnostic.

**Final deliverable:** a certification bundle containing commands, immutable
configuration manifests, machine-readable errors/orders, plots, regional
diagnostics, and passing regression evidence.

### Tracking and future-task rules

- P02/P03 may run independently after P01. P05/P06/P07 may become separate
  future tasks after P04.
- Each future task receives its ID, dependencies, bounded objective, acceptance
  gate, and required evidence.
- Use statuses: `pending`, `ready`, `in progress`, `blocked`, `passed`.
- Mark a task `passed` only when its numerical gate is satisfied and evidence
  is linked. Code completion alone is insufficient.
- Record implementation/configuration identity, geometry/source hashes,
  reference qualification, per-term and regional errors, fallback activity,
  unresolved defects, and the next bounded task. Changing geometry,
  implementation, field catalogue, or acceptance rules invalidates certification
  reuse; retain the historical evidence unchanged.
- Preserve failed candidates and their measured failure mechanisms in research
  artifacts, keeping current-behavior documentation free of experiment history.
- The roadmap is complete only when P09 passes; no conservation identity,
  interpolation test, frozen residual slope, or elliptic solution result alone
  substitutes for the agreed global operator-plus-solution contract.

### Progress ledger

P00–P01 evidence was produced by [Set up perpendicular convergence roadmap P00–P01](thread://01a0b506-1907-73d3-997d-a66637cd7e8c?hostId=local), using GPT-5.6 Sol.

**Current P worker:** this task owns the bounded P04 value/derivative cross and
cubic transported-value comparison that follows the negative global
balanced-cubic qualification. The P00–P03 task links and earlier
bounded/global diagnostic reports remain historical provenance; they are not
the current worker assignment.

**Historical HSX qualification assignment:** P01–P03 had completed their
original audit scope and were reopened for actual-HSX qualification. P00
remained baseline preservation evidence, and P04 was then pending on the
supplemental P02/P03 evidence; prior idealized success did not satisfy that
dependency. That P03 task included the required P01/P02 supplemental work and
stopped before P04 implementation. Preserve those results and task records;
the current P04/P05 status is recorded in the table below.

**Current corrected-reference evidence:** the [continuous-reference sidecar](../../../work/perpendicular_second_order_hsx_p01_p03/continuous_reference_sidecar.json)
qualifies the producer evaluator against 4096 artifact samples and records
bounded differentiation checks. The [corrected N32 orientation experiment](../../../work/perpendicular_second_order_hsx_p01_p03/face_candidate_orientation_continuous_N32.json)
uses that reference for manufactured inputs and face data. Prior quantitative
near-axis omega conclusions derived from serialized polar metric interpolation
are superseded, including the claim that the planar candidate barely improves
omega. The [orientation report](../../../work/perpendicular_second_order_hsx_p01_p03/candidate_orientation_report.md)
now leads with the corrected result and preserves its former rejection only as
historical evidence; that rejection is not the current design decision.
Historical common-face oracle and global error
results using the old reference cannot establish corrected continuum accuracy.
Algebraic identities remain separate supporting evidence.

The corrected planar-only candidate, retaining production eta reconstruction,
reduces physical-volume-weighted L2 error over the fixed eight-owner sample by
60.1% for omega, 16.7% for the regular scalar, and 52.9% for the eta-varying
scalar. Omega relative errors at rings 0/1/2/3 change from
0.05459/0.04652/0.05384/0.27486 to 0.02649/0.04702/0.02066/0.02810;
non-axis transitions regress from 0.00394 to 0.03986. These compare completed
residuals with a common-face midpoint reference, not global continuum operator
errors. Integrated donor averages give mixed results. The axis-placeholder
cleanup leaves owner residuals unchanged: representative u=0 state/metric
values are unavailable and excluded from face-state comparisons, collapsed
integrated flux is exactly zero, and axis-owner errors remain included.
The candidate was promising enough for the bounded matched-functional audit.
That audit is now complete: the midpoint/raw-volume observation functional
retains identical support on 258/268 rows, reproduces quadratics to at most
`1.67e-15`, and reduces aggregate selected-sample L2 error relative to
production by 41.1%/52.5%/49.7% for omega/regular/eta-varying fields. The
[corrected global baseline](../../../work/perpendicular_second_order_hsx_p01_p03/continuous_global_baseline/summary.json)
passes both intervals for actual omega (`1.9305`, `2.2761`) but fails the fine
interval for the smooth regular (`1.5764`) and eta-varying (`1.3302`) controls.
The [structured-omega uncertainty audit](../../../work/perpendicular_second_order_hsx_p01_p03/structured_omega_differentiation.json)
shows source-difference orders `4.80` and `3.96`, so reference differentiation
does not explain that failure. P03 is complete with a failed shared acceptance
gate. The [bounded N48/N64 fine-interval bridge](../../../../work/perpendicular_p03_fine_interval_design_20260919/report.md)
then refines the baseline into disjoint axis, boundary, true size-change,
agglomerated, and ordinary regions and tests frozen eight-owner samples before
candidate evaluation. True interfaces carry 94.6%/91.9% of scalar-upwind omega
squared error at N48/N64, whereas the centered omega budget is led by the axis
at 44.9%/51.1%; the scalar-upwind omega pass therefore does not certify the
centered operator. G improves every sampled scalar-upwind and centered field at
both resolutions. O improves both smooth upwind fields and all centered
combinations, but regresses scalar-upwind omega at both resolutions. Centered A
and B improve separately, with constant, orientation, decomposition, and
swapped-antisymmetry closures at roundoff. This closes the bounded P02/P03
mechanism audit and made P04 ready only for a separately authorized,
non-production P/G/O 32/48/64 global diagnostic qualification. The subsequently
authorized [P04 global diagnostic](../../../../work/perpendicular_p04_global_bracket_20260919/report.md)
is now complete. Production replay matches the qualified global L2 values
exactly. Of 36 per-field/operator/rule lanes, only production scalar-upwind
omega and G/O for the isolated centered-B eta-varying action pass both order
intervals. G/O fail every scalar-upwind lane and every complete centered-C lane;
their common-value scalar-upwind experiment explicitly does not preserve
production left/right stabilization. Structural, periodic, orientation,
constant, and bounded-equivalence checks close at roundoff, but the result does
not support a universal face rule or production promotion.

The subsequent [bounded P03 failure-localization report](../../../../work/perpendicular_p03_failure_localization_20260920/report.md)
reuses the frozen eight-owner N48/N64 samples and saved P04 rows with complete
incident support. On the centered-vorticity O/C sample, exact transported face
values do not improve the `0.761/0.819` numerical L2 error, while the exact
combined generator factor plus matching center correction reduces it to
`0.0540/0.0417`; both exact operands give `0.0312/0.0224`. Smooth controls also
favor the generator substitution, although cancellation prevents treating that
cross as a candidate operator. Center-state replacement is negligible, and
removing the production upwind jump produces only a modest change. Replays,
mean/jump algebra, reference-step sensitivity, and saved-row conditioning close
at their stated tolerances. That cross changed face geometry and the generator
gradient together and therefore did not by itself distinguish them. The
follow-on [four-way generator-factor report](../../../../work/perpendicular_generator_factorization_20260920/report.md)
separates HH, geometry-only EH, derivative-only HE, and combined EE while
preserving the face and matching center terms. EH does not remove the important
error, whereas HE reduces O/C vorticity from `0.761/0.819` to
`0.0430/0.0417` and reduces completed smooth-control errors for both O and G.
The interaction is small; N64 production and continuous one-forms agree to
roundoff. The supported next experiment is therefore matched owner-to-face
generator differentiation with the production one-form retained. This remains
bounded diagnostic evidence, not global convergence certification or a
production repair.

The initial [matched derivative experiment](../../../../work/perpendicular_owner_face_derivative_20260920/report.md)
used the actual midpoint/raw-volume owner functional and a full 3-D quadratic
in regular `(x,y,eta)`, but stopped support expansion as soon as rank and
polynomial reproduction passed. Its `597/1168` maximum regular-Cartesian `y`
amplification and O/vorticity regression from `0.761/0.819` to `3.754/5.152`
therefore reject that nearest-12 support rule, not degree-two differentiation.
The subsequent [donor-support correction](../../../../work/perpendicular_owner_face_derivative_support_20260920/report.md)
tests fixed 18 and 24 donors per eta plane and a geometry-only rule that starts
at 18 and expands to 24 when a plane has fewer than three distinct angular
owner columns. The coverage rule expands no N48 faces and two N64 faces,
reduces maximum condition to `12.3/13.6` and regular `x/y/eta` amplification to
`1.56/1.38/1.08` and `1.61/1.49/1.08`, and improves O/vorticity centered C to
`0.351/0.326`. However, G/vorticity and every smooth-control P/G/O centered-C
lane remain worse than production, with A/B failures exposed separately.
HH/HE replay exactly and no production selector changed. Global qualification
remains unwarranted; P05 needs bounded localization of the remaining stable-row
truncation/cancellation defect before a degree, regularization, or global-run
decision.

The follow-on [adaptive-support and truncation report](../../../../work/perpendicular_owner_face_derivative_adaptive_20260920/report.md)
keeps coverage18_24 as the baseline and localizes its remaining O/C error to
radial/angular generator differences with strong face/center cancellation.
Saved-owner rankings reproduce the independent concentration at true
interfaces and agglomerated interiors for the smooth controls and at the
axis/inner ring/interface for vorticity. A geometry-only sector-balanced
quadratic support leaves completed errors essentially unchanged, so support
placement is no longer the dominant defect. Actual owner observations of all
degree-three regular-chart monomials give persistent mesh-scaled quadratic-row
responses, while one adequately supported degree-three candidate (24 donors
per plane, five/six eta planes) reproduces degree three to `1.44e-12`, keeps
maximum condition at `50.7/87.9`, and improves all nine bounded centered-C
field/P-G-O lanes over production at N48/N64. O/vorticity becomes
`0.112/0.078`, regular O becomes `7.21e-5/4.80e-5`, and eta-varying O becomes
`4.20e-4/3.12e-4`. A/B improve separately and direct numerical-minus-HE
assembly closes below `7.2e-15`. This identifies degree-two truncation as the
dominant bounded mechanism, with residual compatible-composition sensitivity.
It supports a scoped global static qualification of the fixed cubic policy but
does not itself establish order or authorize production integration.

The authorized [global balanced-cubic qualification](../../../../work/perpendicular_cubic_global_qualification_20260920/report.md)
then applies that fixed degree-three policy at N32/N48/N64, with a single
geometry-only deficiency schedule used at every resolution. The optimized
construction reproduces saved bounded rows to roundoff, retains cubic
reproduction to `4.45e-10`, and closes replay and structural checks below
`7.82e-14`. O/centered-C vorticity passes both global order intervals
(`2.3048`, `2.3077`), but the smooth regular (`2.2140`, `1.3777`) and
eta-varying (`2.2527`, `1.4582`) controls fail the fine interval. O/A also
fails the fine interval for all three fields; O/B passes for vorticity and the
eta-varying control but fails for the regular control. At N64, 89.19% of the
regular-control and 72.06% of the eta-varying-control O/C squared error lies in
agglomerated interiors, with another 12.92% of the eta-varying error at true
size-change interfaces; neither the axis nor the physical boundary explains
the failure. This rejects the fixed global cubic candidate for certification
without invalidating its stable construction evidence. No production source,
boundary contract, evolved MMS, curvature, diffusion/polarization, or blob
driver changed.

The subsequent [bounded value/derivative cross](../../../../work/perpendicular_value_derivative_cross_20260920/report.md)
uses deterministic, geometry-stratified 28-owner N48/N64 samples with complete
incident neighborhoods and both base and expanded cubic supports. Holding the
globally qualified cubic derivative fixed, cubic planar face values regress
completed centered C relative to current O values at both resolutions:
vorticity ratios are `1.4487/1.3064`, smooth-regular ratios are
`1.0357/1.0356`, and eta-varying ratios are `1.0382/1.0601`; adding cubic eta
values is immaterial. Analytic values improve vorticity and eta-varying C, but
leave the regular C essentially unchanged at N48 and 7.10% worse at N64 even
though its A/B constituents improve, exposing cancellation/integration
sensitivity. Analytic derivatives reduce vorticity C from
`0.1534/0.04041` to `0.03722/0.01998`, so the cubic derivative family remains
useful diagnostic evidence rather than being rejected wholesale. For the
smooth controls, value and derivative substitutions both matter and their
completed-action behavior cannot be inferred from isolated A/B improvement.
Cross, decomposition, antisymmetry, derivative replay, value reproduction, and
full-operator replay close at or below `4.70e-13`. Under the midpoint target
used in that audit, the fixed cubic value candidates appeared rejected and the
workflow stopped before another global run. The integrated-reference result
below supersedes that target-dependent ranking. No production change was made.

The [integrated-reference requalification](../../../../work/perpendicular_bracket_reference_requalification_20260920/report.md)
supersedes that reference-sensitive interpretation while preserving its raw
midpoint evidence. The supervisor and P-campaign geometry paths resolve to
byte-identical payloads. Direct q3 physical-volume integration passes the 10%
reference budget for both smooth fields at every resolution; the global
O/centered-C orders become `2.1457/1.3143` for the regular control and
`2.1305/1.2130` for the eta-varying control, so the completed production
composition still fails the fine interval. On the frozen bounded cross,
cubic-planar values with the qualified cubic derivative now improve completed
C by 34.18%/48.79% for the regular control and 13.02%/2.42% for the
eta-varying control at N48/N64. The earlier blanket rejection of cubic values
is therefore withdrawn; it was not robust to integrated cell references.
Vorticity direct/IBP q3 passes the bounded budget at N32/N48/N64, but an N64
throughput sample extrapolates to 9.92 hours globally and the affordable
focused interpolant fails the N32 budget at 14.95%. No global integrated
vorticity result was built, so its historical midpoint pass is preserved but
not recertified. This motivated the matched face-and-volume experiment below.
No production source or frozen numerical input was changed.

That [bounded matched face-and-volume experiment](../../../../work/perpendicular_matched_face_volume_20260920/report.md)
is now complete on the frozen 28-owner N48/N64 samples. The corrected
supervisor prerequisite is preserved explicitly: O2/Dh vorticity C is
`0.075041/0.033573`, cubic-planar/Dh is `0.121014/0.026683`, O2 with the
analytical derivative is `0.111712/0.049318`, and analytical value/derivative
is `0.121491/0.050820`; analytical substitution alone therefore does not
repair midpoint assembly. Under matched q3 integration, the analytical volume
correction reduces face-only centered-C error by 93.6%/96.3% for vorticity and
99.7--99.8% for both smooth controls, supporting the continuum identity, sign,
and normalization. With values and derivatives from the same balanced cubic
fit, face-product integration reduces the two smooth-control errors by about
45--55%, but regresses vorticity. The numerical volume correction adds only a
0.9--6.5% smooth improvement and worsens vorticity by 2.9--5.0%. The remaining
vorticity error is concentrated in the boundary-footprint and axis owner strata;
the footprint label is not a radial-wall-face diagnosis. The smooth
error remains concentrated in agglomerated and true size-change strata. This
closes the requested bounded P04 integration audit with a mixed result: the
integration mechanism is real, but the complete numerical candidate is not
globally qualified.

The subsequent [supervisor face-factor audit](../../../../work/perpendicular_face_factor_audit_20260920/report.md)
supersedes the proposed volume-only, one-sided vorticity repair. With the
analytical volume correction held fixed, replacing only face values changes
vorticity C error from `0.144000/0.0305823` to `0.0907829/0.0191711` at
N48/N64; replacing only generator gradients gives `0.0545676/0.0139092`.
Both factors matter, while their bilinear interaction is only 0.84%/1.07%
of the total face-defect RMS. The same diagnosis survives the numerical
volume correction. There are zero physical radial-boundary faces in these
samples: the `physical_boundary_footprint` owner label includes short-leg
and double-hit topology masks and does not establish a wall-reconstruction
defect. No evidence justifies a field-specific repair, degree increase, or
automatic donor expansion. The user has authorized global 32/48/64
qualification of the frozen general cubic matched-integration candidate,
with the global vorticity reference explicitly costed and qualified; the
current P worker is preparing that study.
Bounded error magnitudes do not establish its order, and improvement at every
sampled owner is not a prerequisite. The audit itself changed no production
path; the subsequently authorized study remains research qualification.

Update this table in place. Link detailed evidence rather than appending a
chronological run diary here. Each evidence bundle must identify its hypothesis,
revision, configuration, measured results, and unresolved failures.

| ID | Work package | Dependencies | Status | Task / evidence / remaining failures |
|---|---|---|---|---|
| P00 | Baseline and reproducibility ledger | None | passed | Revision `6c2b005d`; immutable dirty-source and 32/48/64 geometry hashes, exact selectors/boundaries/precision/sharding, and a single-process N=32 bracket/curvature/diffusion/polarization baseline are in the [P00–P01 evidence bundle](../../../work/perpendicular_second_order_p00_p01/report.md) and [machine-readable manifest](../../../work/perpendicular_second_order_p00_p01/p00_p01_manifest_and_results.json). This is preservation evidence, not a convergence claim. |
| P01 | Independent references, averages, sources, norms | P00 | smooth global qualified; global vorticity reference work in progress | The [continuous-reference sidecar](../../../work/perpendicular_second_order_hsx_p01_p03/continuous_reference_sidecar.json) preserves the continuous geometry and full-torus manufactured eta period. The [integrated-reference requalification](../../../../work/perpendicular_bracket_reference_requalification_20260920/report.md) qualifies direct q3 physical-volume integration for both smooth fields globally and direct/complete-IBP q3 for vorticity on deterministic bounded HSX samples. The [matched audit](../../../../work/perpendicular_matched_face_volume_20260920/report.md) supports the analytical identity but does not supply a global omega reference. Constructing and qualifying that reference is included in the authorized P-worker study; the previous 9.92-hour N64 estimate is a cost to measure/optimize, not a numerical-design blocker. The failed focused interpolant and historical midpoint orders remain identifiable evidence, not substitutes. Frozen owner volume remains primary. |
| P02 | Reconstruction and derivative audit | P01 | complete; mechanism localized | [Task](thread://01a0b527-4aaf-7950-a06d-97a46ab517c4?hostId=local), GPT-5.6 Sol. N32 matched-functional evidence and the [N48/N64 bridge](../../../../work/perpendicular_p03_fine_interval_design_20260919/report.md) qualify the existing G/O degree-two mechanisms without changing degree or donor policy. G improves every bounded operator/field sample at both fine resolutions; O improves the failing smooth upwind controls but regresses scalar-upwind omega. This is bounded mechanism closure, not global convergence certification. |
| P03 | Geometry, interfaces, return maps, closures | P01 | complete; acceptance failed; mechanism localized | [Task](thread://01a0b544-d74f-71b0-a705-6b21e623a3cd?hostId=local), GPT-5.6 Sol. The [corrected global baseline](../../../work/perpendicular_second_order_hsx_p01_p03/continuous_global_baseline/summary.json), [fine-interval bridge](../../../../work/perpendicular_p03_fine_interval_design_20260919/report.md), [bounded failure localization](../../../../work/perpendicular_p03_failure_localization_20260920/report.md), and [four-way factorization](../../../../work/perpendicular_generator_factorization_20260920/report.md) preserve the boundary contract and frozen owner samples. Scalar-upwind omega passes (`1.9305`, `2.2761`), while scalar-upwind regular/eta (`1.5764`, `1.3302`) and centered omega (`1.4237`, `1.3153`) fail their gates. Independent EH/HE crosses localize the sampled completed-action failure to generator differentiation rather than face geometry; center state is negligible and the upwind jump is a smaller separate effect. This is bounded localization, not operator certification. Curvature and eta-shard evidence remain separate future qualifications. |
| P04 | Consistent owner-to-face functionals | P02, P03 | bounded audits complete; frozen candidate entering global qualification | The [matched integration audit](../../../../work/perpendicular_matched_face_volume_20260920/report.md) supports the analytical functional and bounded smooth-field benefit. The [supervisor face-factor audit](../../../../work/perpendicular_face_factor_audit_20260920/report.md) localizes finite-resolution error to both face values and gradients, with small bilinear interaction. It establishes neither a need for reconstruction redesign nor a wall/volume-only repair. Preserve the general cubic policy for the authorized global comparison; no production promotion yet. |
| P05 | Brackets | P04 | local study paused; clean remote qualification prepared; material upwinding pending | [P worker](thread://01a0befd-521c-70f0-96f9-8be67df425fa?hostId=local) is paused. The [pause receipt](../../../../work/perpendicular_matched_global_qualification_20260920/supervision_v1_4_parallel_2w/pause_receipt.json) preserves 566 validated N64 chunks and an inactive monitor. The next global study is the new selection-v3 campaign described above; no old reconstruction-dependent chunks enter it. Current acceptance requires complete per-field global operator orders and qualified reference errors. Actual-action identities, shared-face bookkeeping, and boundary-branch checks are scientific diagnostics alongside the study, not separate blockers or convergence gates. No reconstruction-design blocker has been established by the bounded error magnitudes. Historical midpoint omega orders `2.3048/2.3077` do not certify this integrated functional. Material upwinding needs its own later qualification; an independent centered pass is permitted but does not complete P05 or authorize production promotion. |
| P06 | Complete curvature | P04 | pending | — |
| P07 | Perpendicular diffusion/polarization | P04, P03 | pending | — |
| P08 | Combined frozen HSX perpendicular RHS | P05, P06, P07 | pending | — |
| P09 | Evolved MMS and promotion | P08 | pending | — |
