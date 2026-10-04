# Roadmap: second-order parallel operators on HSX angular RLP grids

Approved research roadmap, 18 September 2026. This document specifies planned
work and acceptance gates, not implemented or verified capabilities. It is the
authoritative Q00–Q09 progress ledger. Keep run logs and detailed numerical
evidence in linked research artifacts. The separate
[perpendicular roadmap](perpendicular_second_order_roadmap.md) uses P00–P09.

## 1. Objective and shared acceptance contract

### Current Q08 configuration — user freeze, 4 October 2026

The selected coupled verification configuration uses **h/32 for traced G,
tube D, material transport and every accepted cap-gradient diffusion channel**,
with common five-plane eta quartic reconstruction. Here h is one eta-plane
spacing and h/32 is the total inner cap separation: each cap is at +/-h/64.
Keep the existing paired characteristic construction: its outer samples have
total separation h/16, twice the inner separation. Selecting one base span
does not change that nesting or any qualified operator formula. Retain C3,
RK4-64, the accepted transverse supports/repair choices and D/physical-normal
N reconstruction, including mixed field-wise assignments.

Alternative h/n configurations remain allowed as explicit prepared-plan
choices. Trace endpoints, inner/outer pairing, magnetic coefficients, boundary
data locations and artifact/cache identity must match the requested span;
changing only an application denominator is invalid. Preserve existing h/16
diffusion evidence, but use h/32 for the common baseline. Other numerical
actions require their own applicable verification/qualification; configurability
does not transfer h/32 certification to arbitrary n. This freezes the research
configuration, not production defaults or time-integration qualification.

By user decision, the proposed unequal perpendicular/eta resolution check is
skipped as a Q08 prerequisite. The associated coverage is untested, not passed;
revisit it when qualifying a different resolution regime. Defer further
patch-eigenmode investigation and assess full-domain dynamics with the Q09
evolved path before production promotion. The [local Q08 closeout](../../../../work/q08_closeout_20261004/report.md)
completes host-memory calibration, harness repair and a local source/evidence
freeze. The focused corrected Ti diagnostic is replayed on bounded owners and
its references corrected globally; full-grid N replay still requires the
immutable banks on Perlmutter. Q08 remains open pending that result. The six assembled RHS equations and
full-grid CPU/GPU replay already have their documented evidence and exceptions.

The first focused Ti remote replay stopped before GPU execution on an overly
strict boundary-fixture equality check: four entries differed by at most
1.39e-17. The repaired preflight admits only active-wall float64 roundoff,
`32*eps*max(1,abs(expected))`, and records the observed discrepancy. Layout,
precision, finiteness and nonwall padding remain strict; numerical actions,
the final operator replay tolerance and scientific scoring are unchanged.
All 25 focused tests and the 22-state D/N bounded replay pass locally, with
scalar/full-RHS disagreement at most 5.69e-14. Preserve the failed run and use
a new source/output identity for recovery with the same immutable datasets.
GPU and corrected global Ti qualification are still pending.

### FCI traced parallel gradient: global N-O gate passed — 30 September 2026

User clarification selects a scalar-value difference along traced field lines
for Q06 G, rather than contracting derivatives of a 3D field reconstruction.
See the [superseding Q06 contract](q06_traced_gradient_divergence_contract.md#user-clarification--30-september-2026).
The initial candidate is `b0^eta*(f_plus-f_minus)/(alpha*deta)`, using existing
cap-value rows, complete-owner projection and saved h/16 and h/32 endpoints.
Direct G remains a diagnostic baseline; tube divergence retains its passing
global N-O evidence. The bounded all-field D/N N-O-R comparison is now complete across
21 actual-HSX owners and both spans. No new tracing or donor tuning was used.
The h/32 global campaign now passes the static N-O reconstruction gate;
unconditional continuum N-R order two does not pass. Q06 is now closed by
user acceptance with the explicit static-accuracy and balance exceptions below.

[Bounded traced-gradient report](../../../../work/q06_traced_gradient_20260930/report.md):
62 focused tests pass; 42 site/span combinations and 2,184 field/BC records are
finite and identity checked. Constant residual <=1.391e-12; explicit secant
versus precontracted action <=4.921e-13. Halving span reduces sampled O-R RMS
3.9867x (span order 1.995), while N-O remains close to direct G. Pooled
nonconstant N-O/O-R/N-R RMS at h/32 are 0.00576051/0.000264129/0.00577875;
direct G error is 0.00576235. Wall h/32 N-R is 0.00403837 versus direct
0.00395765. These are fixed-site summaries, not spatial/global orders. h/32
improves total error in 1,020/1,050 nonconstant comparisons; small cancellation
regressions remain. No sampled endpoint is exterior at these short spans:
boundary-conditioned rows are checked, actual exterior-cap coverage is not.
Carry that inventory into global preflight. Small CPU timings show no clear
runtime advantage over precontracted direct G. Diffusion remains unchanged.

**Global traced-gradient campaign completed, 30 September 2026:**
[review and recommendation](../../../../work/q06_traced_gradient_global_20260930/parent_review.md),
[all orders](../../../../work/q06_traced_gradient_global_20260930/orders.csv),
[runner](../../../scripts/q06_traced_gradient_global/README.md).
Identity `1c32df7eb69d0821da88594a9032113958b2fc77e17c0f83fb36f44f08fd50e4`.
All 793 chunks/313,696 owners, 26 fields and both BCs passed coverage, identity,
finite-value, replay and independent analysis checks in 38.90 minutes with two
CPU workers. Peak worker RSS 1.505 GiB; approximately 228.5 MiB results. Maximum
diffusion replay 1.712e-9 < 1e-8; constant gradient residual <=4.018e-12.
No new tracing, support tuning or production promotion occurred.

**N-O passes:** all 100 nonconstant global field/BC/interval RMS checks and all
reported regional RMS checks exceed second order. Global minimum orders
2.6547/3.1345; global maximum-norm minima 2.6875/2.5886. Inner, last two aggregate
rings, transition, bulk and wall RMS pass. Local inner N-O maxima retain a
12.1% worst fine-interval rebound (order -0.3977); transition and wall maxima pass.
**N-R remains conditional:** every nonconstant global and reported regional RMS
decreases, but 2/50 global checks at 32→48 and 24/50 at 48→64 fall below order
two (fine-interval minimum 0.9195). In the fine-interval failures, N64 O-R RMS is
9.44–39.07 times N-O. O-R is the demonstrated dominant discrepancy, not proof of
an analytic-reference bug. Its largest pointwise error still spikes at N48
(0.00584→0.02218→0.00496 across the worst fields). Retain that limitation.

**Exact-endpoint span audit, 30 September 2026:**
[bounded audit and reproducible results](../../../../work/q06_traced_gradient_span_audit_20260930/report.md)
cover the 21 prior owners plus control-field global O-R hotspots: 33 complete
owners/86 raw centers. At fixed actual-HSX sites, exact-cap O-R decreases with
spans h/16 through h/256. For the two constant-shift-equivalent controls,
successive h/32→h/64→h/128→h/256 sampled RMS orders are
1.998/2.000/2.000 at N32, 1.730/1.927/1.982 at N48, and
1.928/1.984/1.996 at N64. These are fixed-site span orders, not global spatial
orders. N48 hotspots have not reached the clean quadratic regime at h/32.
RK4-64 versus RK4-128 changes control actions by at most 1.126e-12;
tracer/reference infinitesimal derivatives agree within 1.095e-13 across all
26 fields. Saved/global O replay agrees within 5.306e-13. No sampled error floor
or evaluator mismatch is observed. A logical-coordinate linearized-field
decomposition attributes most sampled control O-R to the difference between
the traced secant tangent and the center field-line tangent; the nonlinear
scalar remainder is about 7%, 0.7%, 1.7% of total RMS at N32/N48/N64.
This supports finite-span differentiation error in the varying field-line
mapping, not an analytic-reference bug or a volume-average/midpoint mismatch.
It does not isolate coil ripple as the unique cause. Exact O and R share owner
weights and do not use BC reconstruction; both controls share derivatives and
each was tested with D and physical-normal N data.

The existing global control O-R RMS divided by `(h/32)^2` is
6.4345/4.1697/5.5680, a 33.54% rise in the effective coefficient on 48→64.
Different spatial samples/aggregates and nonasymptotic hotspots prevent the
fixed-site span result from proving global spatial order two. If reducing
this continuum discrepancy is required, the next accuracy experiment is a
bounded h/32 versus h/64 comparison with actual scalar reconstruction and
separate N-O/O-R/N-R scores. Exact-endpoint improvement alone does not justify
a production span change. The audit took 54 seconds; no frozen campaign,
diffusion operator, support policy or production default was changed.

**User decision, 30 September 2026:** retain h/32 as the provisional working
span for traced G and continue Q06 structural/boundary checks. Defer the
comparative span-selection experiment to the required
[Q08 traced-span selection step](#q08-traced-span-selection-before-freezing-the-coupled-rhs),
alongside the performance audit and before freezing the coupled RHS. Confirm
the selected configuration in Q09 evolved MMS before promoting a production
default. This does not choose a single diffusion span or change its accepted
h/16 and h/32 evidence.

Recommend retaining the traced-G candidate on its passing static reconstruction
criterion, with explicit review of the continuum and localized maximum
exceptions. The selected G/D boundary/structural audit is now recorded in Q06
below: boundary/algebra checks pass, but exact conservation/adjointness do not.
The completed smooth-packet balance measurement below gives monotonically
decreasing defects for all eight fixed-width bulk fields: 0.004–0.115% of
local divergence activity at N64. The user retains the current implementation;
this supports continued development with approximate balance explicitly
documented, not an exact-conservation claim. All saved paths
have zero detected crossings/re-entries/exterior endpoints: boundary-conditioned
wall reconstruction is checked, exterior ghost behavior is not. **Q06 is closed
by user acceptance on 30 September 2026 as a scoped static qualification, with
the documented accuracy and approximate-balance exceptions. Q07 is ready.**
Q04 traced evolution and production remain open. Monitoring is retired after
completion review.

The separately composed owner-field `D(G f)` diffusion alternative is deferred,
not required for the immediate path. Keep the accepted cap-gradient tube
diffusion unchanged, including its use of contracted reconstruction derivatives
and its documented static-accuracy exceptions. Then address G/D structural and
boundary checks, Q07 transport/material blocks, Q08 coupled RHS plus time/memory
audit, and Q09 evolved MMS/promotion; traced Q04 evolution remains outstanding.
This clarification supersedes older next-step recommendations below without
erasing their completed evidence.

### Accepted and frozen traced diffusion — user decision, 29 September 2026

**Status: static reconstruction accuracy accepted with documented exceptions.**
The user accepts the selective-repair diffusion operator and freezes it as the
baseline for the next implementation stage. See the
[frozen numerical contract and next Q steps](q_traced_diffusion_frozen_contract.md),
[checksummed freeze manifest](../../../../work/q_fci_diffusion_freeze_20260929/freeze.json),
and [source snapshot](../../../../work/q_fci_diffusion_freeze_20260929/frozen_source.tar.gz).
Accepted campaign identity:
`c0d64dba761087a3273d29cd4728689c4f4b5b67f838873a845d1cdb1a809f8b`.
The packaged base commit alone is insufficient: preserve the pinned selective
policy overlay. No production default, package action, commit or push changes
as part of this documentation freeze.

Freeze compact28 with existing rank repair plus the tested geometry-only
selective gradient guard; Cartesian transverse quartic/common five-plane eta
quartic in the inner region; unchanged structured outer and quartic D/physical-
normal N wall policies; both tested total spans h/16 and h/32; RK4-64 traces;
raw-volume midpoint observations/projection and nonzero-divB correction.
No ring20 special case, threshold tuning, higher degree or new trace campaign.

Accepted exceptions remain explicit: localized last-two-agglomerated-ring
truncation/cancellation loss, the coarse outer-envelope N-O order exception,
and the separate O-R limitation that prevents an unconditional continuum N-R
convergence claim. Regional/max orders are diagnostics, not a new order2 gate.
This accepts the current static reconstruction milestone; it neither transfers
historical direct-method evolution to this traced method nor completes all Q04
structural/evolved criteria or the full Q roadmap.

**Q05 closed by user acceptance; Q06 development history follows.**
The direct-gradient assignment and intermediate recommendations below record
the development sequence. The 30 September scoped Q06 closure supersedes
their pending-status statements; current next work is Q07.
The [corrective implementation and second review](../../../../work/q05_traced_extraction_20260929/review_v2/report.md)
close the non-wall Dirichlet tangent-lift leak and stale-receipt findings.
Schema-v2 artifacts, complete checksum manifests, actual device staging and
identity-bound replay have portable actual-HSX regression coverage. All 36
focused tests and 30 curated checks pass; 42 bounded site/span receipts and two
complete N64 chunks match archived actions to at most `4.467e-10` against the
unchanged `1e-8` tolerance. On 29 September 2026 the user accepted this evidence
and closed Q05. Full-domain saved-action replay was not performed; its completion
is waived as a prerequisite to Q06, not claimed as a passed verification.
No production promotion or traced Q04 evolved/structural pass follows from this
closure. The old [Q05 worker](thread://01a0ef54-27dc-7af0-9cdc-0d05d13f4d10?hostId=local)
is superseded by the new Q06 assignment.

The [frozen current Q06 contract](q06_traced_gradient_divergence_contract.md)
governs gradient/divergence and later composition work, superseding the older
integrated-face Q06 assignments for this traced path. The authorized initial
scope is [direct-gradient implementation and bounded verification](q06_direct_gradient_assignment.md):
contract existing midpoint/cap derivative rows with the unit magnetic field,
reuse affine D/physical-normal N lifts, and preserve complete-owner projection.
Direct divergence `G(f) + f div(b)` and tube divergence are subsequent comparison
candidates, not certified actions or part of this initial implementation task.
Worker: [Q06 direct parallel gradient implementation](thread://01a0f02e-8563-74d2-8bcc-4eb0f3abe5f1?hostId=local), GPT-6 Sol, high reasoning. The previous Q05 worker is archived.

Parent [implementation review](../../../../work/q06_direct_gradient_20260929/parent_review/report.md): no blocking numerical finding in direct G; independently reran 40 focused tests and verified all 42 current receipts finite/source-current. Before expanding the harness, enforce nonfinite failure gates and separate staged runtime timing from host-input conversion/transfer. Both harness findings are now [fixed and reverified](../../../../work/q06_direct_gradient_20260929/harness_corrections.md): 61 focused tests and all 42 bounded cases pass with unchanged numerical errors. The [bounded D_direct versus D_tube comparison](q06_bounded_divergence_assignment.md) is implemented and [reported for review](../../../../work/q06_bounded_divergence_20260929/report.md): 70 focused and 30 curated checks pass across 42 site/span cases. User preference is tube divergence for the implementation path, with direct divergence retained as a diagnostic baseline; accuracy and structural gates remain unchanged. No global Q06 pass is implied.
Parent [divergence analysis](../../../../work/q06_bounded_divergence_20260929/parent_review.md): independently reran 70 tests and verified current source/array checksums; no blocking bounded numerical finding. Tube h/32 sampled N-R RMS is .00571927 versus direct .00565045; halving span reduces tube O-R RMS 3.773x while N-O stays nearly unchanged. Favor h/32 for qualification preparation, with expanded bounded coverage and tube-only setup before any global launch; h/16 remains a control. This is span evidence, not global grid convergence.
**Global divergence campaign completed locally, 30 September 2026:** [runner](../../../scripts/q06_tube_global/README.md), [completion review](../../../../work/q06_tube_global_20260929/parent_review.md), [all field orders](../../../../work/q06_tube_global_20260929/orders.csv). Tube h/32, N32/N48/N64, all 26 fields and both BCs; 793 chunks/313,696 owners passed identity, finite, coverage and independent analysis validation in 29.59 minutes with two CPU workers. Reused RK4-64 endpoints and accepted choices; maximum diffusion replay 1.712e-9 < 1e-8. Freeze identity cb5ee38a9900d82ecd3d24b2fcb19ede20142340cda1ab0da2b89e091ddd1ad1.

**Static divergence reconstruction accuracy passes:** every nonconstant field and both BCs exceed second order globally and in every regional N-O RMS, including inner, last two aggregate rings, transition, bulk and wall. Global minimum orders 2.6547/3.1344; global maximum-norm minimum orders 2.6888/2.5879. Local inner maximum exceptions remain: worst 48→64 order −0.398, e.g. fresh_a15_lambda2 D maximum 1.01678e-4→1.14018e-4 while its RMS decreases. Transition RMS and maxima both pass. **Unconditional N-R accuracy does not pass:** 46/50 nonconstant field/BC checks fall below order two at 32→48 (44 rebound), while all pass at 48→64; O-R carries the dominant wall/N48 rebound. This is consistent with earlier discrete-span/geometry limitations, not proof here of a reference bug. The subsequent selected-G/D structural and smooth-balance audits are complete; the user closes Q06 with documented exceptions. Exact conservation is not obtained, and Q04 evolution and production remain pending; diffusion h/16 versus h/32 is unchanged.
Worker result: [bounded direct-gradient report](../../../../work/q06_direct_gradient_20260929/report.md).
The 21-site, two-span, two-BC, 26-field direct `G` implementation and checks are
reviewed; bounded divergence is also reviewed. Global tube-divergence N-O
accuracy passes as above; continuum N-R exceptions and approximate structural
properties are explicitly retained in the user-accepted Q06 closure.
Production qualification remains open.

Next proceed to Q07 transport/material blocks; retain traced diffusion
runtime/evolution checks and Q08/Q09 coupled RHS/evolved
MMS and final integration. Existing direct-method assignments/results below
remain historical evidence; the frozen contract governs the current traced
path. The freeze itself launched no work. The subsequent user-authorized [Q05 implementation assignment](q05_traced_extraction_assignment.md) defined the completed worker scope; the parent completed and re-reviewed corrections, and the user closed Q05 on bounded evidence while leaving complete-domain replay unperformed.

### Selective-repair interface audit — 29 September 2026

[Audit and recommendation](../../../../work/q_fci_repair_interface_audit_20260929/report.md).
Full cached accounting and11-owner/22-anchor replay localize the main penalty
to N64 ring20; ring21 changes little and the first two singleton rings are
unchanged exactly. Last-two-ring repair activation rises from3.44% of raw-anchor
physical volume at N48 to42.01% at N64. At owner26609 cap-gradient error improves
40%, but operator error increases54% because signed cap errors cancel less.
The leading pure transverse fifth-degree contribution increases; retained
quartic-tensor reproduction and signed decomposition pass near5e-12. This is
neither a rank failure nor a boundary/interface wiring defect.

At N64 the affected two rings contribute4.31% of global squared N-O error;
restoring compact there would improve global pooled RMS only another0.88%.
The fixed logical radial band0.28–0.36 has pooled orders3.95/2.21. These contextual
checks do not erase the localized moving-ring1.19 order or maximum rebound.

**Acceptance clarification:** the shared contract below makes regional and
maximum norms diagnostic, not independent mandatory second-order gates. The
previous “uniform regional qualification open” language must not be interpreted
as inventing an additional Q gate. Global coarse-envelope N-O exceptions and
O-R-driven N-R nonconvergence remain separate, explicit limitations.

**Accepted by the user:** freeze the selective repair as the static
reconstruction-accuracy baseline with documented localized/coarse/reference
exceptions, as recorded above. Further donor tuning is closed for now; shared
extraction/replay is next. The audit itself changed no numerical policy or
package default. Full continuum N-R and production/evolution certification are
not inferred from this scoped acceptance.

### Selective inner-support global rescore — completed 29 September 2026

[Interpretation](../../../../work/q_fci_selective_global_20260929/interpretation.md),
[full report](../../../../work/q_fci_selective_global_20260929/report.md), and
[all N-O-R orders](../../../../work/q_fci_selective_global_20260929/orders.csv).
Frozen identity `c0d64dba761087a3273d29cd4728689c4f4b5b67f838873a845d1cdb1a809f8b`.
Complete-domain N32/N48/N64 comparison, 26 fields, both D/physical-normal N and
h/16+h/32. All inner owners rescored; identical outer/wall actions and saved
GPU64 traces/O/R references reused. Computation/invariants passed in28.2minutes
on three local CPU workers; no source/default promotion or new tracing.

The geometry-only gradient guard retains compact plus its existing rank repair
unless both degree5/6 shell errors improve by10% without increased coefficient
amplification. Runtime count remains28 donors per inner plane. Pooled20-wave
inner N-O RMS changes +30.5%/-42.2%/-13.0% versus compact, with orders2.368/2.349
instead of0.360/3.771. Every nonconstant field's whole-inner RMS now exceeds
order2 on both intervals/spans (minimum2.053/2.166); original/fresh orientations
agree. Inner maxima improve59%/49% at N48/N64. Global pooled-wave RMS improves
5.1%/3.1% for D and3.6%/2.6% for N; core, first ring, outer and wall are unchanged.
This is a stronger overall accuracy candidate than always-balanced28, whose
N64 pooled20-wave inner RMS is57% above compact.

**Remaining regional exception:** last-two-agglomerated-ring RMS remains
monotonically decreasing, but fine order falls from compact's2.094 to1.187 and
N64 RMS is29.7% higher. Their pooled maximum increases5.7% from N48 to N64;
some individual wave maxima rebound. The wider interface including two outer
singleton rings still passes RMS order2, but does not erase this narrower
exception. Repair activation is15.77%/20.88%/14.33% of inner raw anchors.
N-O improvements do not change O-R or its known N48 rebound; global pooled D
N-R orders remain-2.960/3.854. The outer-envelope coarse-order exception remains.

**Decision (updated after the interface audit):** static reconstruction
accuracy is accepted with the documented exceptions and the operator is frozen.
The regional loss is diagnostic under the shared acceptance contract; it does
not impose a new mandatory local-order gate. No production default promotion
or further tracing/threshold tuning follows. Shared extraction/replay is next.

### Balanced-28 global comparison candidate — 29 September 2026

**Completed comparison campaign (historical setup):** always-balanced28 inner support in
`scripts/q_fci_layered_global/`, with unchanged ringwise angular7/radial-cubic
outer support, quartic D/physical-normal N wall treatment, common five-plane
eta quartic, h/16+h/32 spans and GPU RK4-64 tracing. The inner/outer switch stays
after the last aggregate rings 10/15/21. This is research qualification, not a
production change. The completed compact baseline is frozen at commit
`ad54c33207835814296f0c83fffa9db87c269e66`.

The [balanced-inner comparison](../../../../work/q_fci_balanced_inner_20260929/report.md)
reduces bounded pooled inner-wave N-O by38% at N48 and25% at N64, while
increasing it2.8x at N32 and worsening some fresh/join locations. The user accepts
N32 as a coarse stress case; evaluate fine-interval convergence and absolute
N64 accuracy alongside regional maxima. Preserve the outer envelope's observed
lower coarse order; increasing its polynomial degree is not authorized.

Balanced selection chooses exactly28 runtime donors from a geometry-only local
pool on every inner eta-plane fit, rather than activating only when nearest28
loses rank. Keep the tested QR choices, tie-breaking, pool expansion sequence
and polynomial basis frozen. A failed28-donor fit is an operational stop, not
permission to enlarge the runtime stencil or tune against MMS errors.

The [resolved-wave response](../../../../work/q_fci_balanced_wave_response_20260929/report.md)
and [short-wave extension](../../../../work/q_fci_short_wave_response_20260929/report.md)
show a genuine fidelity tradeoff: balanced attenuates shorter transverse waves
more, while compact can amplify underresolved values and reverse gradient
response. At N64 wavelength0.125 has8 radial widths but only about2 tangential
owner widths near the aggregate join. Neither method is certified for those
small scales. Global MMS consistency must not be relabelled as turbulence-scale
resolution or timestep damping/stability qualification.

Campaign scope: N32/N48/N64 all complete owners, both D/N, original18 fields
plus8 fresh orientation controls. Parallel exhaustive geometry preflight must
exercise the balanced rule everywhere before GPU tracing/pilot/global scoring.
Keep N-O/O-R/N-R separate, RMS and maxima, first ring, inner join, aggregate
join, ordinary outer and wall regions; retain signed arrays for local comparison
with the completed compact baseline. O-R and the magnetic-reference limitations
are unchanged. Fresh source identities require a new campaign folder.

Local bounded extraction/preflight receipts and remaining remote GPU/all-location
gates are recorded in the runner's `VALIDATION.md`. Publication and remote
execution require the prepared source revision to be committed and pushed;
preparing this candidate confers no new global numerical pass.

### Bounded compact-inner / ringwise-outer candidate — 29 September 2026

The [separate inner-rank and outer-accuracy comparison](../../../../work/q_fci_rank_outer_comparison_20260929/report.md)
supports repaired compact Cartesian-quartic reconstruction through the last
agglomerated ring, followed by P's transverse ringwise angular/radial-cubic
construction. Both retain Q's common five-plane quartic eta representation;
this borrows P's transverse machinery, not its complete operator. The switch
is after zero-based rings 10/15/21 at N32/N48/N64, as requested by the user.

Geometry-only balanced-pool QR selection repairs all 512 previously failing
sampled inner fits at N64 rings 15/16 with 28 selected donors per plane and
no larger-support fallback. Previously valid rows are unchanged. The candidate
pool reaches 56 owners during setup; this is not the runtime donor count.
Coverage is all theta indices on four eta planes, through radial index N-3;
the final two wall rings and all-eta coverage remain unqualified.

On 58 complete owners with 17 fields, the ringwise outer candidate reduces
N64 pooled outer-wave N-O RMS from `2.403e-3` to `2.274e-4` compared with the
four-ring structured Cartesian quartic. Its bounded pooled orders are
`3.72/2.97`, not global convergence orders. All five outer radial wave-RMS
groups decrease on both intervals, but two of 120 individual outer wave
tracks still rebound on the fine interval, and two neighboring singleton
roles have larger absolute error than the old quartic. The five-radial-layer
quartic passes geometric checks but worsens pooled outer accuracy and is not
preferred. h/16 and h/32 give the same ranking. Preserve the unchanged O-R
channel and report N-O-R separately; do not choose by cancellation in N-R.

**Next bounded candidate, not promotion:** retain the repaired compact-inner /
ringwise-outer split, confirm independent phases/orientations and full row
coverage, and audit the intended Dirichlet and physical-normal Neumann wall
closures before a global campaign. This research test changes no P-path or
production implementation and confers no new Q global qualification.

**Bounded wall integration completed:** the
[layered D/N comparison](../../../../work/q_fci_layered_dn_qualification_20260929/report.md)
retains this interior and adds radial-quartic wall rows on the outer two
layers. Dirichlet uses the prescribed trace and its tangential derivatives.
Physical-normal Neumann adapts P's local trace elimination to Q's fixed
seven-angular by five-eta wall lattice: 140 singleton donor values and 35
normal-data samples, with a setup-only 35-by-35 solve (maximum sampled
condition 6.55). Runtime is a linear owner/data row application. The full
metric normal, including tangential contributions, is enforced; no unknown
analytic wall value enters the Neumann action.

On 36 actual-HSX owners, three phases including the seam, distances 1–4,
and 18 fields, every selected wall-field pooled N-O norm decreases on both
intervals. h/16 wave fine orders on the outer two layers are 2.01/2.33 for
D and 4.22/4.00 for N; adjacent interior layers give 2.04/1.96. Preserve
seven individual D fine-interval wave rebounds and eight N coarse-interval
rebounds, and coarse orders below two in some non-wave controls. Neumann
node residuals are about 1e-12; held-out normal residual wave RMS decreases
`2.08e-3 -> 1.30e-4 -> 1.80e-5`. Algebra, tensor reproduction, constant,
affine-shift, periodicity and independent old-row replay checks pass.

All primary midpoint caps stay inside the wall. Separate near-wall probes on
the same HSX geometry produce six outside caps per resolution; extension
value and gradient errors decrease for both BCs. This is bounded extension
evidence, not all-crossing coverage. N-R remains dominated by the unchanged
O-R channel, so this is not a global midpoint-reference convergence pass.
Next: full geometry/conditioning coverage and independent phase/orientation
checks for the complete candidate, followed by global campaign preparation
under an explicit N-O-R/reference scope. No support retuning, P modification,
or production promotion follows from this bounded result.

**Remote campaign engineering:** `scripts/q_fci_layered_global/` packages the
frozen candidate with GPU RK4-64 tracing, parallel CPU all-location geometry
preflight/reconstruction, complete-owner projection, D/N cases, h/16+h/32,
resumable identity-checked chunks and separate N-O-R reductions. The prescribed
remote sequence includes exhaustive geometry coverage, numerical preflight,
CPU/GPU and RK4-step checks, a throughput pilot, then global N32/N48/N64.
Local CPU replay is evidence of extraction correctness; actual GPU execution
and exhaustive global coverage remain remote gates. See the runner README and
VALIDATION.md; no new numerical qualification is claimed by packaging it.

### Q Neumann MMS catalogue — 27 September 2026

**User decision:** retain Q's existing smooth and nonpolynomial wave fields,
prescribe their field-derived physical-normal Neumann data, and add a simple
analytic zero-Neumann control. Do not substitute the P catalogue for these Q
fields. This supersedes the geometry-corrected `oblique_zero_N` field as a
primary qualification requirement; preserve its historical results as a stress
diagnostic, without relabelling its reconstruction error as a reference error.

With `x=u*cos(theta)`, `y=u*sin(theta)` and the full eta period `2*pi`, retain

```text
s = 1 + 0.1*x*cos(eta) + 0.05*y*sin(eta)
      + 0.02*(x*x-y*y)*cos(2*eta)
common          = s
homogeneous_D   = (1-u*u)^2*s
simple_zero_N   = 1 + (1-u*u)^2*s
waves           = exp(i*(2*pi*x_or_y/lambda + eta))
constant        = 1
```

Pair `common` and each wave with their own Dirichlet values and Neumann
derivatives. Preserve the homogeneous Dirichlet control. For Neumann inputs use
the full outward physical normal,
`g_N = (g^{u j}/sqrt(g^{uu}))*partial_j f` at `u=1`, in physical derivative
units; it is not a radial derivative, parallel derivative or parallel wall
flux. No geometry-dependent term is added to the manufactured solution.
The common/wave Neumann pairs already used this convention in the
[corrected wall rescore](../../../../work/q_fci_wall_ghost_corrected_rescore_20260927/report.md).

`simple_zero_N` has wall value one and all first logical derivatives zero, so
its physical-normal datum is exactly zero. Its gradient and continuum operator
equal those of `homogeneous_D`; this supplies an additional matched D/N control
without a wall-metric correction. It does not exercise nonconstant tangential
wall values: the common/wave nonzero-Neumann pairs provide that coverage.
Keep the constant, affine-shift and owner/data row-linearity checks as well.

Use fixed `lambda=2,4` waves as the adequately sampled response controls and
retain `lambda=0.5,0.25` separately as short-scale stress probes. Wavelengths are
logical x/y units and remain fixed under refinement. Sine and cosine are the
real and imaginary parts, not a different boundary convention. Keep these
nonpolynomial controls alongside polynomial reproduction tests; do not infer
general reconstruction accuracy from low-degree transverse fields alone.

The [frozen catalogue and bounded cached-patch rescore](../../../../work/q_fci_field_derived_neumann_20260927/catalogue.json)
reuse the corrected h/8 candidate and h/4 diagnostic, inner legs, selected
owner/BC maps and physical geometry. Changing the field/data catalogue does not
require retracing, selecting supports or fitting new maps. Report the same
signed `N-E`, `E-P`, `P-R` and total `N-R`, with wall and both switch sides
separate. The locked midpoint target and global accuracy gate remain unchanged.
This decision changes Q's MMS catalogue, not the numerical operator, P's
roadmap, or production defaults, and is not a global qualification pass.

**Bounded rescore:** the [cached-patch report](../../../../work/q_fci_field_derived_neumann_20260927/report.md)
adds the new control on the existing 36 singleton targets without tracing or
refitting. At h/8, its relative reconstruction error on the wall is
`1.356% / 0.149% / 0.124%` across N32/N48/N64; the interior side of the switch
gives `0.121% / 0.0317% / 0.0185%`. Absolute RMS reconstruction error decreases
on both intervals in all three sampled regions. N64 total midpoint error is
`0.712%` at the wall and `1.463%` on the interior switch side, predominantly
outer differentiation. The retained smooth/wave definitions agree exactly,
common-field actions replay exactly, and zero-data/constant/linearity checks
pass. This removes the former geometry-corrected field as the immediate
primary-MMS obstacle; broader geometry and aggregate coverage remain required.

### Locked midpoint MMS reference — 25 September 2026

**User decision:** use the raw-cell midpoint projection as the primary MMS
reference for subsequent Q qualification. This supersedes the integrated
primary-reference requirements in older campaign descriptions below for new
work. Historical results retain their original method/reference identities and
pass/fail status; this decision is not a new convergence pass.

For the intended continuum parallel operator `L_parallel`, define

```text
V_o = sum_{c in owner o} V_c
R_o = sum_{c in owner o} V_c * (L_parallel f_MMS)(x_c) / V_o
```

Here `x_c` is each raw-cell logical midpoint and `V_c` is its frozen stored
physical raw volume. Evaluate the complete analytic operator, including its
geometry and coefficient derivatives, before projecting to owners. A singleton
reference is simply the analytic operator at its midpoint. An aggregate uses
every raw member, not one representative point, and retains one evolved owner
value. This matches the existing raw-volume-weighted midpoint observation
functional; it does not reinterpret observations as exact continuous averages.
Global and regional norms retain frozen physical owner-volume weights.

The numerical traced-FCI construction remains a conservative shared-face flux
sum divided by stored owner volume. Endpoint reconstruction, tracing, derivative
spans, numerical face quadrature/projection, opposite incidence signs and BC
channels are unchanged by this reference decision. That numerical action has a
natural volume-average interpretation; its consistency and convergence against
the chosen midpoint target must be measured, not assumed. No point evaluation
of a reconstructed operator replaces the numerical face-flux balance.

Use independent analytic MMS fields and derivatives for `R_o`, never numerical
endpoint values, reconstructed fields or the candidate face fluxes. No field-line
tracing or volume integration is needed to evaluate the midpoint target;
geometry differentiation may still have a computational cost. Retain bounded
nominal/half differentiation-step checks where derivatives are evaluated
numerically, with the existing 10% reference-uncertainty screen applied against
the numerical error on the same complete-owner sample. Keep the global L2 order
gate >=1.8 on both N32→N48 and N48→N64 intervals for each nonconstant primary
field. A bounded sensitivity check is not a rigorous global error bound.

Do not schedule midpoint-to-integrated comparisons or additional high-quadrature
integrated MMS references for this next qualification. Candidate face-quadrature
checks and existing exact-endpoint/exact-gradient diagnostics remain distinct
from reference integration. Reference-independent traced-span and reconstruction
findings remain evidence under the new target; changing the reference does not
repair those errors.

The next reference audit can evaluate this target against saved complete-owner
numerical actions without retracing or recomputing integrated references. Give
that rescore a new reference identity and separate outputs; never overwrite or
relabel historical integrated-reference results. General Dirichlet/Neumann BC
qualification remains required as specified below. This entry locks the roadmap
contract only: the existing Q runner is not yet migrated, and no computation,
worker assignment, or global launch is initiated by this edit.

### Next traced-Q global campaign: Dirichlet and Neumann walls — 25 September 2026

The user requires **both Dirichlet and Neumann boundary-condition families**
in the next global campaign. This is planned scope, not implemented or
qualified capability. The existing projected campaign prescribes zero wall
flux for specially compatible fields; shortening interior tracing intervals
does not establish general boundary reconstruction.

Before freezing a global runner, define and validate a bounded boundary
treatment using actual owner observations and supplied boundary data. Cover
homogeneous and smooth nonhomogeneous Dirichlet data and zero/nonzero Neumann
data, with manufactured fields that exercise nontrivial wall derivatives.
Specify whether Neumann data mean normal derivative, field-aligned derivative,
or the operator's outward normal flux, and derive any conversion from the
actual anisotropic operator and geometry. These quantities are not generally
interchangeable. Exact MMS interior values must remain diagnostics, not
numerical reconstruction inputs. Preserve complete-owner assembly and assess
boundary-layer and global errors separately under one frozen method.

The current [shared-support and special-owner audit](../../../../work/q_fci_fixed_support_coverage_20260925/assignment.md)
continues under its original zero-flux contract and resource cap. Its wall
results must not be presented as satisfying this new requirement. It should
identify what additional bounded BC construction/validation is needed before
global preparation. Reuse compatible traces and geometry across BC tests;
changed boundary data do not alone require retracing. Keep historical passes
and the current traced global non-pass unchanged.

### Projected-FCI global result — 25 September 2026

The [returned campaign and independent local analysis](../../../../work/q_fci_projected_global_analysis_20260925/report.md)
from source `ba754b32f85d045be6228e4c48febf3a57f21b5f` completed all three
resolutions, but **does not pass global static accuracy**. All 125,465 checkpoint
payloads/receipts, source/input identities and complete owner/face coverage were
verified. Independent canonical face-flux reassembly reproduced all ten saved
channels exactly. Constants and signed-incidence balance pass their frozen
checks; this is not evolved, energy or production certification.

| Field | RMS N32 | RMS N48 | RMS N64 | Order 32→48 | Order 48→64 |
|---|---:|---:|---:|---:|---:|
| radial–eta | 0.00822983 | 0.00214321 | 0.00115808 | 3.318 | 2.140 |
| angular-x | 0.00661012 | 0.00155418 | 0.00116250 | 3.570 | 1.009 |
| mixed-y–eta | 0.01050282 | 0.00216425 | 0.00133144 | 3.896 | 1.689 |

The gate requires order >=1.8 on both intervals for every field. All total
errors decrease; the maximum empirical reference fraction is only 0.597%.
N64 errors improve by about 4–8x versus the old single/four-seed return campaigns,
but this does not promote the projected method.

**Diagnostic result:** owner-transfer error (numerical minus exact-endpoint
action) has fine-interval orders 2.497/2.057/2.238. The traced finite-difference
contribution (exact-endpoint minus exact-gradient projection action) instead
grows by 12–28%, predominantly through eta-normal faces. More than 99.9% of
its N64 squared error lies outside the outermost two radial layers; 98.0–99.7%
lies on owners touching no shortened face. Wall-row total errors decrease.
Exact-gradient projection errors are only 6.3e-6–8.4e-6 at N64. These controls
isolate an unresolved traced-derivative contribution; they do not distinguish
finite-span truncation, trace/evaluator discrepancies, or integration of that
residual. Bounded q5/q7 candidate sensitivity remains relevant.

**Completed bounded follow-up:** the [exact-endpoint audit](../../../../work/q_fci_exact_endpoint_audit_20260925/report.md)
and [parent review](../../../../work/q_fci_exact_endpoint_audit_20260925/parent_review.md)
identify finite-span truncation at six complete matched N48/N64 owners.
64-to-256-step RK4 changes are negligible; reducing the span to h/4 improves
all eighteen nonconstant exact-endpoint residuals by at least 6.94x. This is
local diagnostic evidence, not a numerical-method or global convergence pass.
The [dispatched numerical shorter-span comparison](../../../../work/q_fci_short_span_numerical_20260925/assignment.md)
keeps owner reconstruction fixed, reuses saved trajectories and tests h, h/2,
h/4 with actual owner data, including a complete-owner q5/q7 control if the
resource preflight admits it. Face-quadrature sensitivity remains unresolved.
Preserve the returned dataset and previously passed direct Q03/Q04/Q05a
baselines. No global rerun or production change is implied.

### Shared reconstruction literature and resolved-scale response — 24 September 2026

The [annotated P/Q bibliography](perpendicular_second_order_roadmap.md#structured-reconstruction-and-resolved-scale-response--24-september-2026)
records the user-requested references on functional recovery, structured mapped
reconstruction, boundary consistency and turbulent-scale resolution. It covers
[Mirzaei–Schaback–Dehghan (2012)](https://doi.org/10.1093/imanum/drr030),
[McCorquodale et al. (2015)](https://doi.org/10.1016/j.jcp.2015.01.006),
[Du–Li (2018)](https://arxiv.org/abs/1801.00990),
[Lele (1992)](https://doi.org/10.1016/0021-9991(92)90324-R),
[Motheau–Wakefield (2021)](https://arxiv.org/abs/2106.06585),
[Denaro (2011)](https://doi.org/10.1016/j.jcp.2011.02.011),
[Ghosal (1996)](https://doi.org/10.1006/jcph.1996.0088),
[Kravchenko–Moin (1997)](https://doi.org/10.1006/jcph.1996.5597) and
[Almgren et al. (2013)](https://doi.org/10.1137/110829386).
They provide construction principles and scale-response diagnostics, not a
proof for the present projected-FCI scheme. Moment matching and smooth MMS
order do not certify turbulent spectra; finite-difference methods share
wavelength-dependent truncation and nonlinear aliasing concerns.

Two directly relevant plasma references are
[Giacomin et al.'s GBS description (2022)](https://arxiv.org/abs/2112.03573),
which includes numerical diffusion and grid-refinement studies, and
[Body et al.'s GRILLIX divertor treatment (2020)](https://arxiv.org/abs/1908.05398),
whose Sec. 2 explicitly discusses interpolation dissipation contaminating slower
perpendicular dynamics. The latter's support-operator treatment is a precedent,
not a reason to restart our previously rejected adjoint/return designs.

**Q application and sequencing:** preserve the completed projected-FCI
campaign and prioritize the exact-endpoint audit recorded above. Use the
all-orientation P evidence before choosing its
structured family as a Q endpoint challenger. A later bounded comparison would
reuse identical trajectories/full-half spans, face nodes, projection, geometry,
BC policy and owner observations; change only endpoint reconstruction. Report
endpoint amplitude/phase error, complete parallel-operator response, absolute
leakage on nearly field-aligned/null controls, and preparation/storage/application
cost separately. Include oblique and perpendicular variation because parallel
interpolation error can act across the field. Do not infer speed or accuracy
from the P stencil alone. Unavailable aggregate/axis supports remain explicit
rather than silently changed into independent point data. The full/half
combination can amplify interpolation error; test it as assembled. Physical
resolved wavelengths and reference quadrature must be checked on actual HSX.
No new tracing, campaign, endpoint replacement, filter, production change or
resolved-scale computation is launched by this reference update. Static response
precedes any claimed evolution damping rate or turbulence qualification.

### Literature basis and limits of the projected-FCI candidate — 24 September 2026

The current projected-FCI research direction is motivated by published
integration/projection approaches and the subsequent bounded HSX evidence.
It is **not a direct reproduction of a published algorithm**. The papers below
provide construction precedents; their convergence, stability and conservation
results do not automatically qualify our adaptation.

1. **Stegmeir et al., *The field line map approach for simulations of
   magnetically confined plasmas* (2016).**
   [Primary preprint](https://arxiv.org/abs/1505.02040).
   Develops field-line tracing with interpolation/integration and support-operator
   parallel diffusion. This supports the underlying FCI architecture, not our
   particular owner-to-face reconstruction or an obligation to reuse the earlier
   unsuccessful adjoint/return construction.
2. **Stegmeir et al., *Advances in the flux-coordinate independent approach*
   (2017), Computer Physics Communications 213, 111–121.**
   [Paper and DOI](https://doi.org/10.1016/j.cpc.2016.12.014).
   Combines integration and interpolation in the parallel-gradient construction
   to address distorted-map and convergence problems within a self-adjoint
   scheme. Its discrete operators and boundary treatment differ from ours;
   its structural and convergence properties cannot be inherited by citation.
3. **Wiesenberger and Held, *A finite volume flux coordinate independent
   approach* (2023), Computer Physics Communications 291, 108838.**
   [Paper and DOI](https://doi.org/10.1016/j.cpc.2023.108838),
   [open published PDF](https://backend.orbit.dtu.dk/ws/portalfiles/portal/332408589/1_s2.0_S0010465523001832_main.pdf).
   Sections 3.3.3 and 4 provide the closest architectural precedent: polynomial
   projection for numerical coordinate transformations and finite-volume flux
   construction in locally field-aligned representations. The paper also reports
   smoothing requirements and oscillation limitations. Our fixed physical-face
   projection of traced derivative samples is a distinct adaptation, not the
   paper's coordinate-transfer operator or full advection scheme.

**Our specific choices:** actual owner-moment cubic endpoint reconstruction,
the full/half-leg combination `(4 D_half - D_full)/3`, a ten-coefficient total-cubic
physical-area face projection, q5 (25-point) candidate quadrature, shared
coordinate-face fluxes, and the near-wall interval-halving policy. No matching
published prescription for this complete combination has been identified.
These choices require their own qualification. See the
[actual-HSX comparison](../../../../work/q_fci_hsx_face_projection_20260924/report.md),
[fourth-order extension](../../../../work/q_fci_hsx_fourth_order_extension_20260924/report.md),
and [frozen global campaign contract](../../../scripts/q_fci_projected_campaign/README.md).
The extension's eighteen regional RMS improvements motivated global testing;
individual-cell exceptions and boundary limitations remain, and this is not a
global convergence or production pass. Earlier stage restrictions below are
historical; the subsequently authorized global campaign is recorded in that
contract and remains under qualification.

**Cost/accuracy remains open:** neither these papers nor our recorded comparisons
establish that ten coefficients or 25 traced samples are necessary. A 4×4 rule
with the same cubic space, direct quadrature without this projection, and a
smaller face space have not been ruled out by an HSX comparison for the current
candidate. Nine samples cannot give rank ten in the present construction; that
algebraic fact is not evidence that every cheaper flux method fails. Higher-rule
checks assess sensitivity, not cost optimality. The expensive part is the traced
samples and endpoint maps, not the small mass-matrix solve. Any cheaper design
needs a separate bounded comparison with complete owner/recipient errors and
independent references; this entry does not change or interrupt the frozen run
or authorize an additional experiment.

**Deferred optimization: quadrature on shared owner-interface patches.** After
establishing convergence with the current potentially oversampled fine-face
construction, compare it against quadrature on larger shared interface patches
between agglomerated owners. The current q5 rule supplies 25 points per retained
fine-grid face, not per owner; internal faces of one aggregate are removed,
but its exterior boundary retains the fine-face pieces. This can create many
traced samples per evolved owner near the axis. Dense geometric integration
does not restore independent fine-cell solution values inside an aggregate.

A candidate is 25 points per shared owner-interface patch, with an appropriate
parameterization and quadrature for its geometry, rather than 25 points for an
entire owner's boundary. Keep interfaces to different neighbors separate, or
retain an explicitly conservative partition of their fluxes. Each shared flux
must enter its two owners with opposite signs; periodic seams, curved axis
interfaces and prescribed boundary fluxes must retain their proper treatment.
Twenty-five points on a larger patch is a hypothesis, not an established
accuracy requirement or guarantee. Hold endpoint reconstruction and derivative
span fixed, qualify patch quadrature by refinement, and compare complete-owner
errors, shared-flux balance, geometry/moment consistency and measured setup/
application cost against the converged fine-face baseline. Include axis-core,
first-ring, other aggregate and ordinary controls. This optimization is
sequenced after convergence; it does not alter the running shorter-span test
or authorize a new campaign now.

**Separate reference and candidate quadrature checks.** The returned global
reference compares analytic q11 face fluxes/continuous q7 owner volumes against
q9 faces/q5 volumes; its largest empirical change is 0.597% of numerical error.
That qualifies the stated reference budget, not q5 sampling of the numerical
candidate. Candidate q5/q7 means 25 versus 49 points per fine face with the
traced derivatives, endpoint reconstruction and face projection recomputed at
those points, against the same reference. The bounded N48 angular candidate
change reached 52.6% of its error. This is sensitivity evidence, not a proven
q5 error bound or evidence that q7 is exact. Reconstructed samples and traced
derivative residuals can vary more sharply than the analytic flux, so small
analytic-reference changes do not certify candidate integration. A material
q5/q7 difference must be localized and shown to settle before attributing a
remaining convergence defect solely to the derivative span or reducing sampling.

### Current result: analytic return comparison completed — 24 September 2026

The [bounded comparison](../../../../work/q_fci_return_architecture_comparison_20260924/report.md)
and [parent review](../../../../work/q_fci_return_architecture_comparison_20260924/parent_review.md)
complete all fourteen frozen configurations. The integrated physical-face
projection is conservative in this analytic model and approaches second order
with actual true-owner-mean transfer, but has larger RMS error than the current
return in 80/84 comparisons. The current method also improves on every balanced
refinement. No HSX failure or accuracy-improving replacement is established.

The nonlinear shear map has straight individual field lines, constant geometry
along each line and zero y-face flux. Five exact cubic null directions leave
only fifteen active modes in the variable case, versus nineteen in HSX.
Condition numbers grow while target-weighted gain stays bounded; this does not
reproduce the actual HSX remainder-amplification mechanism. The parent verified
all 197 manifest files, independently integrated the plane-wave reference and
replayed complete sparse actions, norms and conservation.

**User direction: use actual HSX geometry.** The proposed curved analytic
continuation is superseded. Q has been assigned the
[bounded actual-HSX face-projection comparison](../../../../work/q_fci_hsx_face_projection_20260924/assignment.md):
reuse the overlap experiment's two connected three-owner tracks at N32/N48/N64,
the original one-seed return, actual owner observations and q5 face oracle.
Compare the face-seeded projected construction with exact endpoint data and
legitimate owner transfer, using full HSX tensors, measures and curved field
lines. Include every incident face and all exterior-recipient increments.

Only the new face half-leg trajectories necessary for these small patches are
authorized: batched explicit RK4, N64-hotspot cost preflight, cached geometry,
at most 10,000 scalar legs and 1,200 total CPU seconds. This bounded allowance
supersedes earlier prohibitions on additional tracing for this task only; no
full-grid geometry/tracing/reference campaign is authorized. The assignment
specifies reduced optional scope if the measured cost requires it. Preserve
all negative results, qualify trace/reference/transfer errors, and distinguish
moving-patch refinement diagnostics from global convergence. No elliptic solve,
new quartic optimizer, parameter scan or production promotion. Earlier direct
milestones and the traced global failure retain their existing scope.

### Historical overlap decision and comparison assignment — 24 September 2026

The [two-face overlap report](../../../../work/q_fci_two_face_overlap_20260923/report.md)
and [parent review](../../../../work/q_fci_two_face_overlap_20260923/parent_review.md)
reject the independent full-weight quartic correction as a composable accuracy
rule. All twelve cases have valid isolated maps and unchanged exterior outputs.
Nevertheless, the newly tested B–C face already regresses on several fields,
the middle-owner SSE cross term reverses one net isolated improvement, and the
combined middle-owner l1 norm exceeds its original cap by 6.58–10.21% in all
six geometry-control cases. L2 still passes. Numerical certification and
conservation do not imply improved actual-field error or composed sensitivity.

The parent independently verified all original manifest files, complete saved
actions, cubic reproduction, signed repeated-row unions and the SSE interaction
identity. A fixed half-weight diagnostic, the convex mean of the A–B-only and
B–C-only operators, restores all three owners' l1/l2 caps without a joint solve.
Yet N32 hotspot radial numerical RMS still worsens by 4.25%/8.60% with one/four
seeds. This is an unqualified algebraic control, not a selected damping policy.
It separates the readily enforceable convex cap issue from the remaining
approximation/accuracy failure. All patches move with resolution; no new global
or fixed-location convergence result is claimed.

Following the [literature reset](../../../../work/rlp_literature_reset_20260924/research_memo.md),
the recommended next Q direction is a bounded **return-construction comparison**:
compare the present traced-observation-to-coordinate-face recovery with one
integrated/projection FCI construction on a controlled connected patch, using
exact flow maps where available, actual average/flux functionals, separate
parallel/transverse refinement and held-out smooth fields. State the transfer,
physical-face conservation and runtime contracts before implementation; a
published method's solution order or improved conservation is not our operator
qualification. Keep the existing direct method and a convex-average correction
as diagnostic controls with their limitations intact. Do not infer an FCI
impossibility from this particular failed return rule.

The worker's suggested joint two-face/three-owner quartic optimization could
test narrow feasibility, but is not the recommended priority: it retains an
objective that already failed on an isolated new face and leaves overlapping
patches unresolved. The user has now authorized and Q has been assigned the
[bounded analytic comparison](../../../../work/q_fci_return_architecture_comparison_20260924/assignment.md).
It explicitly audits earlier tube/projection attempts before implementing one
alternative with matched fixed-owner functionals and controlled refinement.
No global campaign, additional HSX tracing/seed scan, global coefficient solve,
elliptic repair or production promotion is authorized here. The traced global
static failures remain open; historical direct-method milestones remain valid
for their own actions. This direction supersedes older next-step recommendations
in the historical entries below. P's separately authorized boundary-conditioned
reconstruction is recorded in the perpendicular roadmap.

### Returned traced-FCI global result: both seed variants fail static accuracy

**Corrected bounded audit completed by the coordinator (23 September 2026):**
The [corrected report](../../../../work/q_fci_return_interior_audit_20260923/corrected_v2/report.md)
fixes the Taylor scaling and per-face SVD defects and completes the direct
remainder, modal, row-union sensitivity and signed cancellation analysis.
All six seed/resolution cases pass independent validation and replay the
returned global actions. At N64 the weakest three retained modes account for
about 100% of the signed sampled error; their N48-to-N64 changes account for
99.35–100.40% of the matched-track error change across fields and seeds.
Actual input remainders shrink while normalized target/return sensitivity grows
at several hotspots. Geometry-only tracks show the same mechanism unevenly;
angular error improves there on the fine interval. This is bounded evidence
of a sensitivity problem in the frozen return construction, not a global proof,
independent tracing validation, or a tested repair. Preserve the global static
failures. Next recommended bounded test: quantify minimum achievable weight
sensitivity subject to exact cubic target constraints on the same observations,
to distinguish intrinsic weak observability from the current weighting choice.
No mode truncation, new campaign, correction or follow-up task is launched.

**Historical parent review of the first bounded audit (23 September 2026):**
The [review](../../../../work/q_fci_return_interior_audit_20260923/parent_review.md)
accepts the sampled face-functional replay but marks the mechanism assignment
partially complete. Taylor coefficients are incorrectly scaled during owner-to-
face translation; the modal loop reuses the last face's SVD and saves no modal
products. Complete signed mode/cancellation and row-union sensitivity budgets
are absent. These are defects in the new diagnostic, not established campaign
bugs. The exact-endpoint channel still reproduces the accuracy failure, so the
return mechanism is not cleared. Finish the specified diagnostic before choosing
a correction or broadening to tracing; no follow-up is dispatched by this review.

**Bounded follow-up, now completed after correction (23 September 2026):** The user authorized the
[interior exact-data return-map audit](q_fci_return_interior_audit_assignment.md).
The parent froze nine ordinary-interior tracks and 54 complete incident faces
per resolution, shared between seed variants. Reuse archived traces and weights
to independently replay polynomial targets, resolve actual field-remainder
response through target-weighted SVD modes, and measure support locality and
signed complete-owner cancellation. This is bounded mechanism analysis, not a
new global qualification, tracing campaign or numerical-policy change.

The [independent single-/four-seed comparison](../../../../work/q_fci_seed_comparison_20260923/report.md)
of the returned 23 September 2026 campaigns verifies pinned source identities,
all global checkpoint hashes, complete face/owner coverage, face-map replay and
independent signed assembly. Both computations completed, but neither frozen
traced-return candidate meets the global order >=1.8 gate. Fine-interval orders
for radial-eta/angular-x/mixed-y-eta are `-0.906/-1.451/-0.778` with one seed and
`-1.082/-1.493/-0.843` with four. N64 error norms differ by less than 0.6%,
without implying identical operators or qualification of either seed policy.

The exact-endpoint-value secant channel already reproduces the failure. At N64,
the exact-secant-to-q5-face-return component accounts for 91.0–99.75% of the
signed projection onto total error; numerical endpoint reconstruction is a
smaller contributor. Empirical reference fractions are <=0.09182%, and all
analytical reference actions are bitwise identical between seed campaigns.
Removing the outer two radial layers still leaves negative fine-interval orders
for every field. This is predominantly an ordinary-interior return/assembly
accuracy problem, distinct from P's Dirichlet wall reconstruction investigation.
Rank-19 target reproduction passes, while condition numbers grow; causality
from conditioning, trace integration or support geometry is not yet established.

The next recommended numerical step is a bounded exact-data return audit on
complete ordinary-interior tracks and large interior contributors, reusing
saved traces, maps and analytical face oracles to resolve remainder/target
coupling and signed cancellation. Do not launch another broad seed-count
campaign or promote either variant on these results. Traced static accuracy
remains open; prior direct-method milestones remain valid for their own actions.
The user-preferred traced architecture remains under investigation. This review
does not launch new numerical work or change P's active assignment.

### Reopened architecture decision: traced FCI preferred; direct method retained

The user reopened the Q design choice on 22 September 2026. **Traced FCI is
strongly preferred, but is not an absolute constraint.** Do not abandon it on
the strength of the failed mapped-directional-observation-to-coordinate-face
return alone. A direct-method preference requires compelling evidence that
reasonable traced transfer and compatible assembly alternatives are infeasible
for the required HSX accuracy, stability and cost. This is an engineering and
scientific decision, not a request for a mathematical impossibility proof.

Two approaches remain open, as defined in
[the parallel architecture comparison](parallel_fci_direct_design_comparison.md):

- **Traced FCI:** reconstruct scalar values or moments at field-line mapped
  locations/footprints, differentiate or form fluxes along those traces, and
  return them through a compatible conservative/weighted assembly.
- **Direct reconstruction:** fit scalar owner observations in a spatial chart
  and apply magnetic differential/coordinate-face functionals directly. The
  successful Q03 diffusion candidate and Q05a/Q06 work belong to this branch;
  sharing FCI mesh containers does not make their derivative trace-based.

Existing direct static/evolved accuracy and extraction milestones remain valid
for their named implementations. They do not settle architecture selection or
automatically qualify a traced replacement. Direct-only global preparation is
held while the parent audits prior experiments and performs a bounded traced
comparison. Q00–Q09 identifiers, real-HSX global accuracy gates, current wall
baseline, deferred monotonicity repair and final production integration scope
are unchanged. Later historical recommendations to proceed solely with direct
G/D infrastructure are superseded by this section.

The recent integration audit establishes eta quadrature sensitivity and useful
composite integration, **not proven metric-knot crossings**: the toroidal metric
is Fourier–Zernike, and its stored eta sample planes are not interpolation
breakpoints. Preserve the numerical evidence while correcting that attribution.

The [parent's bounded traced-FCI audit](../../../../work/parallel_fci_reopening_20260922/report.md)
is complete. At eight complete interior/axis/RLP owners per N32/N48/N64,
cubic endpoint-transfer contributions are only 0.04–1.61% of the much larger
exact-data three-point diffusion errors. A provisional weighted-adjoint
assembly conserves and dissipates but does not fix consistency. Independent
retracing does not explain the dominant error. At four fixed N64 locations,
exact-data eta-spacing refinement approaches order 1.93–1.94; these off-plane
analytical probes are not additional resolved meshes or a global pass.
Changing from arc-length to eta-coordinate differentiation alone gives
similar errors. Selected wall owners were explicitly omitted;
wall qualification remains open.

**Next bounded numerical milestone:** establish a traced flux-tube measure
and matched transfer/projection formulation for diffusion, separately testing
exact-data along-line differentiation/integration and owner reconstruction.
Use complete recipient closure and qualified matching references. Compare
against the direct benchmark without conflating functionals. Do not restart
donor sweeps, increase trace substeps, or launch a full campaign merely from
the errors of the provisional controls. A failed plain weighted transpose
or coarse three-point stencil is not a rejection of FCI. Global accuracy,
wall and applicable structural/execution qualification remain future gates.

The [parent's complete-owner overlap comparison](../../../../work/parallel_fci_complete_owner_20260922/report.md)
now covers all saved graph links of ten selected owners at N32/N48/N64,
including wall owners separately. Cubic scalar transfer with the legacy
conservative overlap return remains inaccurate even with exact scalar data;
the reconstruction contribution is only 0.095–2.87% of the sampled exact-data
residual. q3–q5 integration of the frozen polygons is already well settled.
An independently traced mid-leg derivative/area oracle improves errors on
two complete ordinary N32 owners by 1.68–5.36 times but leaves a substantial
residual. Transporting the area alone does not further improve the result.
Conservation and negative sampled energy production are diagnostics, not
accuracy or global stability certificates. Complete graph coverage does not
certify the legacy terminated-cell wall coverage.

The [bounded three-/five-point trace study](../../../../work/parallel_fci_five_point_20260922/report.md)
is complete. Five-point differentiation changes the full-sample truncation
but is not uniformly better; scalar-transfer contributions remain small. It
is an underresolution control, not a promoted stencil.

The [matched N32 traced-tube study](../../../../work/parallel_fci_matched_tube_20260922/report.md)
now closes the exact tube cap/source identity on two ordinary owners and keeps
tube, fixed-owner, and raw-midpoint functionals separate. Tube/owner mismatch
is substantial but does not explain the legacy residual by itself; two- and
four-native-plane cap derivatives remain truncation dominated.

The [shared-interface N32 patch](../../../../work/parallel_fci_shared_patch_20260922/report.md)
then constructs one identical central cap partition across adjoining slabs.
Opposite-sign cap cancellation and overlap-volume return conservation pass to
roundoff, but bounded membership quadrature does not qualify the curved
tube/fixed-cell intersections: selected coverage moves by 2--4% and oracle
actions by `0.0018--0.0054` under the prescribed refinement. The conservative
piecewise-constant return is also visibly insufficient, but its accuracy is
not separable from that geometric uncertainty. Next qualify an adaptive
cut-cell oracle on the same frozen patch before widening support or adding
moments. Retain the direct benchmark and all existing global gates.

The subsequent [bounded integrated-gradient/return consistency test](../../../../work/parallel_fci_gradient_return_consistency_20260922/report.md)
uses the completed m1/m2 rows for one ordinary N32 owner and separates scalar
transfer, row averaging, and return. All contributing traces have positive
`b^eta`. Scalar transfer and endpoint-secant versus physical-volume-gradient
terms are only `O(1e-5--1e-4)`, compared with remaining action errors of
`0.011--0.031`. Ten geometry-selected rows agree with archived `F ds/B`
measures to at most `1.77e-5` relative under q5 traced-Jacobian integration;
q3--q5 sensitivity is at most `1.62e-3`. In contrast, exact prescribed-flux
return controls miss the independent fixed-owner divergence by `0.013--0.230`,
and field-wise inferred mass ratios span about `0.50--0.79`. A scalar mass fix,
orientation change, endpoint reconstruction, and settled longitudinal measure
quadrature therefore do not explain the dominant residual for this owner.
The evidence supports a mismatch between the present weighted-adjoint
return/row representation and the fixed-owner control-volume functional; it
does not reject the literature method or traced FCI generally.

**Authorized next bounded design test: shared-face flux return (option 2).**
The [new Q worker assignment](q_fci_shared_face_return_assignment.md) specifies
an N32 complete recipient patch centered on owner 5061. Reconstruct parallel
flux from observation-matched traced-leg moments, comparing linear and
quadratic flux polynomials; integrate one shared flux on each actual fixed
face and assemble owner divergence by signed incidence. Start with analytic
face/row flux controls, then analytic endpoint gradients and actual FCI G*T.
This retains traced FCI differentiation while testing an explicit conservative
return. The assignment includes geometric support qualification, integration
sensitivity, fixed-owner references and structural diagnostics. The direct
benchmark and existing global qualification gates remain unchanged.

The bounded Q-worker [shared-face return study](../../../../work/parallel_fci_shared_face_return_20260922/report.md)
is complete on the prescribed N32 patch around owner `5061`. Nine complete
ordinary owners, 42 unique fixed faces, and 486 saved m1/m2 observations were
used. Canonical shared-face incidence cancels exactly and the complete patch
balance residual is `6.94e-18`. Direct q5 face integration and degree-one
observation-matched reconstruction reproduce the analytic polynomial controls
at the target-owner level to `1.2--3.6e-4`; q5-to-q7 target changes are at
most `4.52e-4` across the bounded analytic controls. The independent
fixed-owner diffusion reference uses the `div(b*(b dot grad(T)))` volume
functional. Degree-one shared-face return errors for owner `5061` are
`(+1.48e-2,-2.00e-2,+7.00e-3)` for analytic endpoint secants and
`(+1.50e-2,-2.01e-2,+7.30e-3)` for saved numerical `G*T` rows. Relative to
the prior m2 weighted-adjoint errors `(+1.26e-2,-3.12e-2,+1.15e-2)`, angular
and mixed improve to about `0.65` and `0.63` of the old magnitudes, while
radial is `1.19` times larger. The labelled volume-mean path agrees closely
with the arc path. Quadratic fits are rejected for actual observations:
field-independent support conditioning reaches `5.92e7` despite
`2.56e-9` polynomial reproduction, and the numerical-row target errors grow
to approximately `(-31.6,-20.2,-62.0)`. This is a bounded partial
improvement, not a global, wall, positivity, dissipation, or mesh-order pass.
The next small decision is to retain fixed-face geometry and degree one while
testing a better-conditioned field-independent local stencil with the same
analytic and m1/m2 controls; do not launch mesh refinement or promote degree
two from this result.

The bounded [quadratic-support follow-up](../../../../work/parallel_fci_shared_face_return_quadratic_20260923/report.md)
added the requested outward source intervals `(4,-1)` and `(6,+1)`, with a
5x5 local transverse halo on each plane and separate m1/m2 degree-one/two
fits.  The actual patch accounting is 42 unique faces: 12 internal and 30
exterior; the earlier 21/21 statement is corrected.  Canonical face fluxes
were saved and independently replayed through signed incidences: shared-face
cancellation and owner-volume versus exterior-ledger residuals are zero in
stored double precision.  The added intervals reduce degree-two scaled
condition ranges to `310` (m1) and `378` (m2) at worst, from the baseline
`5.2e5--5.9e7`, with polynomial reproduction below `1e-14`.

This is a conditioning success but not an accuracy success.  At owner `5061`,
degree-one numerical errors remain approximately
`(+1.36e-2,-3.92e-2,+1.91e-3)` for m1 and
`(+1.39e-2,-3.94e-2,+1.91e-3)` for m2.  Degree-two errors are worse,
approximately `(-2.61e-2,-1.09e-2,-5.41e-2)` and
`(-3.10e-2,+2.72e-3,-5.21e-2)`.  The exact-face q5/q7 diffusion oracle
changes by at most `2.25e-4` on the patch, so the remaining discrepancy is
not explained by this bounded face quadrature sensitivity.  Keep fixed-face
assembly and the 12/30 incidence ledger as diagnostics. The subsequent
geometry-aware comparison below supersedes the initial recommendation to
retain degree one; neither free-polynomial degree is certified.

### Current bounded return candidate: geometry-aware cubic potential basis

The [basis and support design contract](q_fci_geometry_aware_return_design.md)
records the matching observation/face functionals, independent support
requirements, evidence limits, and the staged next bounded verification.

The [cached return-basis comparison](../../../../work/parallel_fci_return_basis_comparison_20260923/report.md)
supersedes the recommendation above to prefer a degree-one flux return.
Quadratic flux reconstruction improves the angular field but worsens the radial
and mixed fields; neither degree was qualified. A new FCI return uses basis
functions `b.grad(p_j)` for cubic regular-chart potential polynomials, including
continuous magnetic geometry in both leg observations and shared-face targets.
It still consumes traced endpoint-difference observations.

On the same nine-owner N32 patch, 96 existing leg observations per face give
numerical operator RMS `[4.46e-4,5.64e-4,5.56e-4]` for m1 and
`[2.07e-4,9.14e-4,1.41e-3]` for m2. These improve on free-quadratic flux fits
using the identical supports by approximately 8–51 times. The initial 24–27-row
geometry-aware fits fail badly; 48/96-row support expansion improves the actions
and reduces maximum scaled conditions from about `4e6` to `5e3–8e3`. Exact
pointwise-gradient controls support the approximation capacity of the new basis,
but are diagnostic only. Ninety-six is not a frozen production donor count.

The parent completed the [corrected local robustness test](../../../../work/parallel_fci_local_return_20260923/report.md)
on three new N32 ordinary, agglomerated, and transition complete owners. The
worker's earlier extension used distant observations; its errors were an
extrapolation result, superseded by 510 actual local m2 traced observations.
Original-patch replay agrees to `2.78e-16`. At 96 rows the geometry-aware
numerical RMS is `[7.790e-4, 1.163e-3, 2.137e-3]`, improving on free quadratic
returns on identical supports by `9.92/17.61/12.64` times. Endpoint reconstruction
is a smaller contribution; exact arc observations retain most of the error.
The 48-to-96 expansion improves radial/angular but modestly worsens mixed error,
so neither universal row-count optimality nor completed adaptive selection is
claimed. Exact potential endpoint differences qualify q5 leg basis moments:
using those moments changes actions by only `[1.94e-6, 1.88e-6, 5.66e-6]`.
The local comparison and validators took about 131 seconds at about 1.3 GiB.

The [bounded local-return refinement](../../../../work/parallel_fci_local_return_refinement_20260923/report.md)
completed N32/N48/N64 tracing, but the [parent review](../../../../work/parallel_fci_local_return_refinement_20260923/parent_review.md)
corrects its interpretation and completes the intended endpoint-matrix data paths.
The worker used the q5-integrated matrix for numerical/exact secants and applied
endpoint moments only to exact arc data. The parent applied the endpoint matrix
to all three channels from cached rows; all N32 endpoint-policy fluxes replay the
prior exact-leg-moment evidence exactly. All 28 incident faces and saved actions
were independently reassembled. Correct patch balance is below `2.12e-21`; the
worker's saved balance diagnostic contained an array-broadcasting error.

For the intended endpoint-matrix numerical path, per-field sampled RMS is
`[7.7926e-4,1.1623e-3,2.1320e-3]`,
`[1.5503e-4,8.1074e-5,4.2228e-4]`, and
`[8.3318e-5,1.4845e-4,1.5648e-4]` at N32/N48/N64. Radial local orders are
`3.982/2.159`, angular `6.567/-2.103`, and mixed `3.993/3.451`. The worker's
maximum-component slopes obscure the angular regression. Exact-secant angular
fine order is also negative (`-2.652`), so endpoint reconstruction does not
explain it; the ordinary track supplies 86.9% of N64 angular squared error,
with changed signed radial/theta/eta cancellation. These are local diagnostics,
not additional acceptance gates. Identical-support free-quadratic numerical
errors remain 54--130 times larger at N64, supporting the geometry-aware basis.

All intended maps retain rank 19, maximum conditions `7307/8678/14215`, and
target-relative reproduction defects below `1.04e-13`. Q7-face versus q9-volume
RMS is below 1% of numerical error at N64; N48 angular is about 9.54% and its
reference margin remains a caveat. Stored/continuous owner-volume ratios were
explicitly checked; normalizing the reference to stored volume does not remove
the angular regression. Conservation and reference checks remain bounded.

**Current authorized work (23 September 2026):** the user cancelled the Q
worker's background runner and asked the parent to prepare the remote handoff
using the remote-campaign-handoff skill. The abandoned runner and partial
outputs are preserved in `work/q_fci_global_preflight_20260923/`.
The standalone replacement is documented in
[the campaign README](../../../scripts/q_fci_return_campaign/README.md) and
[local readiness evidence](../../../scripts/q_fci_return_campaign/local_validation.md).
It retains the cubic-potential endpoint-moment method and provides deterministic
canonical-face supports, explicit compatible-MMS wall fluxes, complete axis/wall
incidence, checkpoints and measured CPU costs. Historical replay and full-domain
cheap topology audits pass at N32/N48/N64. The seven-owner N32 canonical preflight completed. The user stopped remaining
local N48/N64 preflight for cost; their completion is a mandatory remote
prerequisite, enforced by the runner. The linked receipt records this limit. Full-domain
numerical execution is reserved for the remote assignment. No global-order,
positivity/dissipation, general wall-closure, evolved or production milestone is
advanced by this preparation.

### Current result: direct cubic has high global order but fails the minimum principle

The [independently audited direct-cubic global campaign](../../../../work/q03_direct_66aea41b_KyVdek_analysis/report.md)
is complete at pinned revision `66aea41b`. The subsequent
[independent reference closure](../../../../work/parallel_q03_reference_closure_20260922/report.md)
qualifies static accuracy under the existing bounded empirical sampling
contract. N32/N48 retain q11 references; 2,304 N64 faces (0.317%) receive refined
analytical reference integration, with numerical targets/actions unchanged.
Physical-volume-weighted orders are now `3.212/3.011`, `3.509/3.285`, and
`3.346/3.168` for radial, angular and mixed fields. Constant reproduction,
signed-incidence assembly and saved-action replay pass. Reference budgets are
at most `0.990%` at N32, `5.159%` at N48 and `8.644%/9.836%/3.937%` at N64.
The N64 angular point estimate has a narrow margin: its diagnostic bootstrap
95% interval is `7.96–11.78%`. This is an empirical sampled qualification, not a
rigorous global bound or a high-margin result; sampling spread is not a new gate.
The original q9/q11 discrepancies of `12.00%/12.21%` remain in the historical
audit. Static accuracy now passes; structural and applicable execution
requirements still prevent declaring a complete Q03/Q04 pass.

The [Q1 fixed-time conduction MMS](../../../../work/parallel_q04_conduction_mms_20260922/report.md)
and [independent parent review](../../../../work/parallel_q_progress_review_20260922/report.md)
now also pass the evolved accuracy milestone. Stationary nontrivial fields,
independently forced and evolved to `t=1`, give solution orders radial
`3.890/2.551`, angular `3.564/3.715`, and mixed `3.592/3.389`. Maximum temporal
error fraction is `1.03e-9`; propagated reference difference is at most `3.823%`.
CPU/JIT/JVP checks pass. This is not coupled-system MMS or general
time-dependent spatial-forcing certification. The local actions differ from
the returned remote actions by up to `4.56%` of local static spatial error at
N64; their own independently recomputed static orders still pass and nearly
match the remote orders. Keep the two implementation identities separate and
localize that discrepancy during bounded extraction/replay preparation.

The [same-geometry N32 unforced heat evolution](../../../../work/parallel_q03_direct_heat_spot_20260922/analysis/report.md)
applies the unit-diffusivity sparse direct action to the current state with
`chi_parallel=0.1`, RK4 `dt=1/1500`, and 37,500 steps to `t=25`.  It completes
finite, conserves owner-volume heat to `3.7e-15`, and has nonincreasing sampled
variance energy, but violates the minimum principle on step 1.  The minimum
reaches `0.9982975` and ends at `0.9986894`; 152 owners are below the initial
minimum at `t=25`, while no temperature becomes negative.  The attached
owner-overlap control remains at or above `1`.  No clipping, limiter, or
coefficient symmetrization was applied.  Completion and scientific acceptance
are recorded separately; this run fails scientific acceptance and confirms
that the sampled sign defect is dynamically relevant for this smooth heat spot.

The requested [actual-48x48x48 unlimited heat evolution](../../../../work/parallel_q03_direct_heat_spot_N48_20260922/analysis/report.md)
is also complete with the same pulse, `chi_parallel=0.1`, RK4 `dt=1/1500`,
37,500 steps, and homogeneous boundary flux. No historical 48-cubed heat run
was found; the timestep provenance is the documented `48x48x32` heat control,
not the one-step forced-MMS `dt=0.01`. N48 conserves heat to `5.0e-15`, has no
one-step variance-energy increase or negative absolute temperature, reaches a
minimum `0.9992172`, and ends at `0.9995762` with 499 owners (`0.4148%` of
volume) below background. The undershoot is milder in amplitude than N32 but
persists. These two heat spots are diagnostics, not a solution-MMS order pair.
The validated result is published under
`prototype_runs/hsx_parallel_heat_spot_qhs_1T_main_only_48x48x48_direct_cubic_unlimited_t25_20260922`.

The earlier [returned Q03 remote campaign and independent local analysis](../../../../work/q03-exchange-tNlxO86M_analysis/report.md)
remains the negative result for the superseded whole-support exchange path.

It is complete at pinned revision `afbb1ace`. The frozen 64-swap whole-support
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

**Completed requested heat work:** monotonicity repair and limiter experiments
remain deferred at the user's request. The unlimited N32 and actual-48-cubed
unforced heat evolutions are complete and preserve the known minimum-principle
failure as diagnostic evidence. Do not infer spatial order from them or launch
N64 heat, a limiter study, or a changed reconstruction from these results.

**Completed Q05a extraction:** the bounded
[direct-cubic extraction and replay assignment](q05a_direct_cubic_extraction_assignment.md)
has passed. The [saved-result replay](../../../../work/parallel_q05a_direct_cubic_extraction_20260922/report.md)
reproduces frozen bounded N32/N48/N64 donors, coefficients and fluxes exactly,
and complete local saved-coefficient actions at roundoff scale through both
explicit shared-face and retained sparse paths. Monotonicity, general
dissipation and eta-sharding remain explicitly open. The unresolved
local/returned-remote difference is preserved without unsupported attribution.
This does not mark Q04 complete or promote a production/default operator.

**Completed accuracy/extraction work and next bounded recommendation:** reference closure is complete; use
the [qualified reference bundle](../../../../work/parallel_q03_reference_closure_20260922/reference_bundle.json)
without rebuilding reconstruction or the global reference campaign. Q1 has
completed fixed-time forced conduction MMS with independent temporal and
propagated-reference checks, and Q05a has completed extraction/replay against
the identified passing local baseline. The next bounded recommendation is Q06
planning for independently qualified gradient/divergence pairs using the
retained directional interface and observation data. This recommendation does
not dispatch Q06, confer certified promotion, or license reuse of the diffusion
matrix as a transport operator.
Do not rerun the full reconstruction campaign, raise degree or support size,
relax the reference budget, or silently introduce clipping/limiters. The known
minimum-principle failure remains recorded; deferral is neither a structural
pass nor production authorization. Q03/Q04 final certification and Q05
promotion retain their outstanding requirements. Full shared-model integration
remains at the end of the roadmap.

The [direct runner](../../../scripts/q03_direct_campaign/README.md) uses indexed
exact donor searches, reusable centered owner moments, batched SVD, bounded
process parallelism and identity-checked restart. Frozen scalar inputs are
exported locally; remote reference production uses the actual corrected d58
metric cache and continuous MAKEGRID evaluator without historical workspace
imports. The older Q01 catalogue metric hash is retained as historical
source-state provenance, not substituted for this campaign's evaluator hash.
The global direct computation and N32/N48 diagnostic heat evolutions are complete.
Neither promotes a production/default operator. P and Q remain separate.

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

### Integration sequencing: consolidate first, integrate at the end

**User decision, 22 September 2026:** full shared-model integration belongs at
the end of this roadmap, in Q09. First qualify diffusion, gradient/divergence,
and transport/material operators; progressively extract and reuse their common
machinery instead of integrating each prototype independently into the full
model. Q05 establishes the initial reusable interface contract after Q04;
Q06–Q07 may refine it as additional physical operators establish their needs.
Q05 is internal reuse of certified structure, not production/default promotion
or a commitment to an immutable final architecture.

Before Q08, consolidate shared owner measures, oriented interfaces, continuous
geometry queries, observation/reconstruction conventions, conservative
assembly, boundary classification, preparation, and runtime plumbing wherever
their contracts agree. Preserve operator-specific physics and directional
information: a symmetric diffusion matrix is not a transport operator.
Consider components already established by the
[perpendicular roadmap](perpendicular_second_order_roadmap.md), without forcing
identical P/Q reconstruction policies or creating a new cross-roadmap blocker.
Replay each affected certified action after consolidation; implementation-only
refactoring does not require repeating global accuracy campaigns, while a
changed numerical action needs separately scoped qualification.

Minimal opt-in integration in existing diagnostic harnesses remains allowed
before Q09 when needed to test an operator. In particular, the authorized N32
heat-conduction integration/run is a diagnostic adapter, not full shared-model
integration, a production default change, or Q04 certification. It remains
authorized with the existing run parameters. Q08/Q09 also need a common
verification-stage path for frozen/evolved coupled MMS; that test assembly
precedes deployment. Final shared-model integration follows Q09's gates.
Blob-driver synchronization remains deferred. Existing numerical, positivity,
dissipation, reference, and solution requirements are unchanged.

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

The user has authorized independent evolved-accuracy work while monotonicity
repair is deferred. [Q1's fixed-time conduction MMS assignment](q04_conduction_mms_assignment.md)
specifies the frozen direct-cubic action, stationary nontrivial spatial fields,
independent continuum forcing, a separate time-dependent uniform-source check,
and N32/N48/N64 temporal/reference qualification. It is separate from Q's
unforced N48 heat spot and does not itself assert complete Q04 certification.

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

**Current traced path: closed by user acceptance, 29 September 2026.** The
corrected extraction and second-review evidence above satisfy the user-accepted
Q05 milestone. Complete-domain saved-action replay was not run and is waived as
an entry prerequisite to Q06; this is not a full-replay pass. Traced Q04
structural/evolved checks and production promotion remain open. The historical
direct-path assignments below are separate.

The bounded [Q05a extraction/replay assignment](q05a_direct_cubic_extraction_assignment.md)
is a completed preparation exception while full Q04 structural certification
remains open. Its passing [replay evidence](../../../../work/parallel_q05a_direct_cubic_extraction_20260922/report.md)
is tracked separately from Q05 promotion and does not change the frozen
numerical action.

- Extract qualified owner measures, mapped interface geometry, reconstruction
  conventions, and conservative assembly without changing certified diffusion.
- Supply oriented interfaces, source/destination owners, shared measures,
  distances, boundary classification, and qualified reconstruction data.
  Preserve directional information in the qualified direct face-functional
  design; do not replace it with or infer a symmetrized conductance model.
- Reuse producer artifacts and host-side preparation with fixed-shape JAX
  runtime application. Add versioned geometry payload fields only when needed.
- Treat this as staged extraction for reuse by Q06–Q07. Consolidate the final
  shared contract after those operators expose their requirements; full model
  integration stays in Q09.

**Gate:** the shared implementation reproduces certified diffusion results
and invariants; consumers receive the directional information their operators
require. State and restart layouts remain unchanged.

### Q06 — Certify parallel gradient/divergence pairs

**Dependencies:** Q05.

**Current traced path:** [Q06 contract with 30 September clarification](q06_traced_gradient_divergence_contract.md).
Q05 is closed by user acceptance. Global tube-divergence N-O accuracy passes;
continuum N-R exceptions remain. The traced scalar-difference gradient now passes global N-O accuracy at h/32;
continuum N-R exceptions are documented above, with direct G retained as a diagnostic.
The separate owner-field D(G f) diffusion alternative is deferred. Boundary and
structural audit findings are recorded below. **Q06 is closed by explicit user
acceptance, 30 September 2026, with the operator unchanged.** This closes the
static traced-G/tube-D rung with documented N-R/local-maximum exceptions and
approximate conservation; it does not claim exact weighted adjointness or
conservation, exterior ghost qualification, evolved stability or production
readiness. Preserve the specialized current–phi/SAT compatibility contract.
The earlier Q06 assignments and results below are historical; they do not
override the current contract.

**Selected G/D structural audit completed — 30 September 2026:**
[report and reproducible evidence](../../../../work/q06_structural_boundary_audit_20260930/report.md).
No D(G f) diffusion was constructed or tested. All work retains h/32,
five-plane reconstruction, frozen support and existing diffusion.

- Affine D/N lifts, off-wall zeros, unused tangential inputs, constant-shift
  identities and independent cap-value assembly pass on 21 cached complete
  owners. Maximum explicit-action discrepancy 5.896e-13. Three actual wall
  patches enforce arbitrary nonzero physical-normal data at their 35 constraint
  nodes within 1.983e-12; Dirichlet residual rows vanish on the wall. This does
  not certify exterior caps, every between-node constraint, or a sheath law.
- Reused both complete global G/D campaigns: 26 fields, 676 ordered pairs and
  all four D/N combinations. Worst normalized pairing defects decrease from
  about 0.95% to 0.73% to 0.35%. Most discrepancy remains when analytical
  midpoint actions replace numerical operators in the same owner-volume sum.
  Keep reconstruction, finite-span and continuum-sum/boundary discrepancies
  separate. Independent wall integration through 512x512 points changes its
  matrix by 1.467e-5 on the final refinement, much less than the worst pairing
  defects. Some product boundary fluxes are nonzero; individual catalogue
  field fluxes are nearly zero and alone do not certify general conservation.
- **Exact weighted adjointness does not hold.** Homogeneous-Dirichlet owner
  basis vectors give nonzero `V_i*(D_ii+G_ii)`, including -6.595e-5 at N48
  wall owner 85888. This is a matrix counterexample, not an MMS reference error
  or a demonstrated dynamical instability.
- **Exact owner-weighted conservation does not hold.** Six bulk one-owner
  pulses have zero physical wall flux but nonzero summed tube divergence.
  All 140 possible recipients per pulse were included; inner/wall support
  exclusion was verified, so there is no missing patch-boundary contribution.
  Residuals relative to summed absolute column contributions are
  0.92–2.21% at N32, 0.10–0.94% at N48, and 0.025–0.379% at N64. These are
  grid-scale pulse tests at changing locations, not spatial orders or percent
  mass loss per timestep. The divB correction is retained. Local reconstructed
  tube evaluations do not automatically telescope as shared interface fluxes.

The audit used about 165 seconds of computation and less than 1.87 GiB peak
RSS, with no new tracing or full numerical campaign. Existing N-O accuracy
passes remain valid. This audit alone did not close Q06; the subsequent user
closure below accepts the measured limitations without an exact structural pass.

**Smooth resolved-packet balance measured — 30 September 2026:**
[review and full results](../../../../work/q06_resolved_conservation_20260930/parent_review.md).
The user requests retaining the implementation. Eight fixed-width smooth
profiles at two ordinary-bulk locations have exactly zero wall flux. Complete
incoming support covers 4,356 / 10,400 / 22,197 owners; inner/wall exclusions
and zero omitted exact-cap actions are checked. Both homogeneous D and
physical-normal N actions agree exactly. No new tracing or numerical changes.

- `abs(sum V*N) / sum V*abs(N)` ranges are 0.112–0.402% at N32,
  0.0206–0.192% at N48, and 0.00392–0.115% at N64. Medians are
  0.286%, 0.119%, 0.0447%. Every field improves monotonically in both this
  ratio and absolute net divergence. The six positive N64 packets are at
  most 0.0631%; the 0.115% case is a signed control.
- This is an instantaneous redistribution imbalance, **not percent mass
  lost per timestep**. For `df/dt = -c*D(f)`, the mass defect is
  `-c*sum V*N`; saved results include normalization by `sum V*abs(f)`.
- Keep signed `N-O`, `O-R`, and continuum midpoint-sum residuals separate.
  At N64 their maximum magnitudes relative to activity are 0.228%,
  0.00963%, and 0.293%, respectively (not necessarily the same case).
  Cancellation contributes to the small total; the defect is not simply
  an O-R reference issue, and shortening the span alone is not a remedy.
- Identity/coverage/finite/runtime gates pass. Independent compensated
  summation agrees within 1.10e-17. About 4.8 minutes of setup/computation
  including pilot, below 0.97 GiB peak RSS; validation/analysis overhead extra.
  This is not a smooth axis/transition/wall survey or an evolution result.

**Closure decision — accepted by the user, 30 September 2026:** close Q06 as
static qualification of the unchanged h/32 traced scalar gradient and tube
divergence, retaining the five-plane reconstruction and frozen donor policy.
Accept approximate balance within the measured scope above; exact conservation
and weighted adjointness are explicitly not properties of this pair. Preserve
N-R/local-maximum exceptions and the exterior-crossing exclusion. No new
threshold, operator modification or production promotion accompanies closure.
Q07 may now build transport/material blocks from these operators, but must not
assume an exact discrete product rule or use the generic pair in place of the
specialized compatible current–phi/SAT construction. Track term/block balance
in Q07 and accumulated drift in subsequent evolution. Exact-conservation
requirements, if introduced later, would require a separate numerical change.

The h/d and three-versus-five eta-plane choices are deferred to the Q08 step
below; they do not block this structural audit. No new owner-field D(G f)
diffusion is required. Traced Q04 evolution, Q07 coupled material terms, Q08
performance/span selection and Q09 production promotion are separate work.

The detailed bounded numerical contract is frozen in the
[Q06 gradient/divergence assignment](q06_gradient_divergence_assignment.md).
The subsequent bounded reference/transfer follow-up and the corrected
interpretation of the archived derived-gradient diagnostic are frozen in the
[Q06 reference/transfer follow-up assignment](q06_reference_transfer_followup_assignment.md).
The next bounded common-integration and derived-scalar diagnostic is frozen in
the [Q06 integration/derived audit assignment](q06_integration_derived_audit_assignment.md).
The completed [integration/derived audit](../../../../work/parallel_q06_integration_derived_audit_20260922/report.md)
and linked [correction to the earlier follow-up](../../../../work/parallel_q06_reference_transfer_followup_20260922/report_integration_addendum.md)
supersede the matched-integration interpretation below where they differ.
Its subsequent [source-level correction](../../../../work/parallel_q06_integration_derived_audit_20260922/report_source_correction.md)
supersedes the audit's metric-knot causal wording. The authorized reusable
implementation and campaign-preparation contract is the
[Q06 parallel-operator implementation assignment](q06_parallel_operator_implementation_assignment.md).
The completed [bounded follow-up](../../../../work/parallel_q06_reference_transfer_followup_20260922/report.md)
finds matched q9/q11 exact-face/volume unit-field RMS of
`5.04e-5/4.32e-5`, `5.59e-5/1.66e-5`, and `8.32e-7/3.62e-7` at
N32/N48/N64. Targeted matched q8-subcell checks reach
`7.72e-8/1.84e-6/1.03e-8`. The direct-divergence difference from
`-b·grad(log|B|)` is accounted for by `div(B)/|B|` with identity closure below
`5e-13` RMS. Independent N64 raw-midpoint observations replay the archived
`D_h O_mid(G_cont f)` action to `1.81e-16`; the angular derived-scalar node
reconstruction error is `1.31e-3` versus `1.82e-7` for the original scalar on
identical supports, and its action error is `1.10e-2` versus `1.86e-6` for
direct exact-gradient face integration. This localizes the bounded N64
amplification to transfer of the derived scalar through D, not a discrete
`G_h`-then-`D_h` composition. No package action changed. These frozen hotspots
are not a representative global sample, so global qualification remains
pending.

The final two-owner-per-resolution A/B/C/R audit classifies the coarse hotspot
discrepancy as shared operator/reference face-integration error, not a
bookkeeping defect. Replacing the archived unsplit q9 functional with
sample-plane-composite GL8 changes the unit action by
`1.206e-3/1.251e-3/2.274e-6` at N32/N48/N64. Under the common rule, exact-face
minus independently integrated volume RMS falls to
`3.685e-6/3.383e-7/4.725e-10`, inside the independent Fejer10 face-plus-volume
checks. Directional refinement localizes the sensitivity to eta, but the
metric evaluator is Fourier--Zernike and its stored eta sample planes are not
piecewise interpolation knots. The established cause is under-resolved eta
integration; breakpoint alignment remains unproven. A smaller field-dependent
reconstruction residual remains. On six frozen N64 faces, fitting the original scalar and then
differentiating reduces the derived node RMS error from `1.459e-3` to
`1.720e-5`; on four near-axis faces it reduces `2.029e-3` to `8.373e-8`, and
its flux matches the direct-diffusion control to `1.43e-18`. This remains a
face-only mechanism diagnostic. No package source or numerical policy changed.
Implement the bounded equal-cost integration-policy comparison and reusable
G/D preparation, then prepare but do not launch representative global G/D
qualification after freezing its implementation identity and independent
references.

- Construct the parallel gradient and conservative divergence on the shared
  structure. Distinguish b dot grad from div(b times field), including div(b).
- Audit metric factors, boundary terms, weighted pairing, individual actions,
  and relevant compositions independently on real HSX geometry.

**Gate:** each physical operator meets the applicable global accuracy and
structural requirements, or has explicitly accepted exceptions. For the current
traced path, the 30 September user closure above governs: static N-O accuracy
and bounded BC/algebra pass; N-R/local-maximum limits, approximate balance and
non-adjointness are retained. Adjoint compatibility alone is not an accuracy
certificate. Historical gates below/above do not revoke this scoped acceptance.

### Q07 — Certify transport and coupled material terms

**Dependencies:** Q06, closed with documented exceptions on 30 September 2026.
**Status:** five-field static material block **qualified by user acceptance,
1 October 2026**, with documented native-grid order, short-wave and local-maximum
exceptions. Selected background/action: compact C3, RK4-64, h/32 inner cap
separation and h/16 outer characteristic separation, geometry-consistent centered
tube weights, unchanged accepted reconstruction/repair and D/physical-normal
N/mixed conditioning. [Global review](../../../../work/q07_c3_global_review_20261001/report.md)
and [velocity attribution audit](../../../../work/q07_c3_velocity_audit_20261001/report.md)
are the current acceptance evidence. This is scoped static qualification, not
a claim of uniform second-order measured N-R or full Q07 closure. Next:
bounded actual-HSX current–phi/SAT integration audit; align and replay remaining
vorticity/diffusion terms on the selected C3 background before a coupled RHS
freeze. Production defaults remain unchanged.

The following records the earlier implementation history. The first centered
density-flux block and shared scalar-slot interface were implemented;
[contract](q07_transport_contract.md)
and [bounded review](../../../../work/q07_density_bounded_20260930/parent_review.md).
26 focused tests and 672 bounded field/BC/site records pass implementation and
artifact checks; replay <=1.180e-13, constant factors <=6.697e-13, zero velocity
exactly zero. Primitive products are formed at caps/center before divergence
and complete-owner projection; continuity uses electron velocity.
Smooth-control maximum N-O decreases across the samples, but new short-wave
wall controls are sensitive, especially for Neumann velocity. The worst N64
N-O is 0.15684, dominated by velocity reconstruction; the nonlinear cross term
is only 1.821e-7. Sampled wall wavelengths have roughly 2–4 angular samples per
cycle. Those first samples were not matched-location or global convergence results.

The subsequent [matched-location wavelength audit](../../../../work/q07_density_matched_20260930/report.md)
now covers 2,592 records at 27 owners. All predeclared smooth/ladder groups show
decreasing pooled N-O RMS and maxima at wall and both transition sides over
both intervals, with individual rebounds retained. Short-wave accuracy limits
remain; no support/span retuning or global-order claim.
The [characteristic-correction prototype](../../../../work/q07_characteristic_bounded_20260930/report.md)
reuses the active DAE production matrix (including mu*tau coupling) and existing
h/16+h/32 values for five-point eta one-sided-minus-centered derivatives.
168 bounded records have zero eigensystem fallback; smooth correction N-O
maxima decrease, while short-wave sensitivity and up to 8.688e-9 reconstructed-
constant roundoff remain. It returns a correction only and is not production-wired.
The [five-field assembly audit](../../../../work/q07_material_bounded_20260930/report.md)
now implements those centered terms plus matched psi=phi+tau*Ti force and the
separate correction. 41 focused tests and 1,008 bounded records pass wiring,
identity, mixed BC, force cancellation, JIT/AD and replay checks; density replay
is exact, zero spectral fallback. Smooth combined N-O maxima decrease
0.02100 / 0.004741 / 0.003177, but exact-stencil characteristic modification
increases smooth total N-R versus centered-only (combined maxima
0.3470 / 0.08986 / 0.10385). Short-wave electron-row reconstruction remains
sensitive; fixed sites differ across N and cannot establish global orders.
Next compare the full material block at corresponding wavelength/phase locations,
with component-normalized electron force decomposition and characteristic mode
response, before selecting the correction for a global campaign. Specialized
current–phi/SAT integration remains separate and open. No Q07/global/production pass.


The [remaining-term bounded implementation](../../../../work/q07_remaining_bounded_20260930/report.md)
adds scalar vorticity advection/correction and all six existing constant diffusion
channels, plus a separate prescribed-coefficient cap-flux audit. 52 focused Q
tests and 1,680 HSX records pass; constant scaling is exact and unit coefficient
replays frozen diffusion within 3.826e-12. No diffusion span default is selected.
Smooth sample errors decrease, with short-wave/local exceptions retained.
The [current/phi inventory](../../../../work/q07_remaining_bounded_20260930/current_phi_integration.md)
separates reference-pair identity, live generalized force, affine current lift
and unqualified physical wall-work trace. Three prior captured regression tests
cannot replay because their saved NPZ arrays are missing; no new current/phi
pass is claimed. Next parent task is the actual-HSX integration audit; the new
Q worker completed the matched material/characteristic response audit, reviewed below.


Parent review of the [matched material/characteristic audit](../../../../work/q07_material_matched_audit_20260930/report.md),
30 September: frozen evidence hashes verified, 5,832 action records, zero
fallback, exact density replay. Every nonconstant pooled field/region group
has decreasing N-O RMS at both intervals; individual exceptions remain.
Keep the unchanged correction as the candidate: its ideal/local response is
dissipative in both orientations, with expected second-order phase error.
Smooth N64 wall electron combined N-R is 0.1098% of pooled continuum RMS;
short-wave lambda=0.35 remains 9.068% (primarily reconstruction), versus 0.1606%
at lambda=0.7 and 1.378% at lambda=0.5. The smooth exact-stencil/reference error
still rebounds N48-to-N64; corresponding sites are not identical physical
points and neither global orders nor a reference bug is established. Actual
eta reconstruction has larger phase error than the ideal short-span stencil:
wall m4/N32 1.1751%, m4/N64 0.07662%; retain eta support/span selection in Q08.
This review supports preparing term-resolved global static comparisons of
centered/corrected material, vorticity and existing diffusion channels with
prescribed phi while completing the distinct current/phi integration audit.
Current/phi boundary-work compatibility, evolved stability, exterior behavior
and full Q07 closure remain open; do not promote production or retune support.


The authorized [local Q07 global static campaign](../../../scripts/q07_material_global/README.md)
is prepared for N32/N48/N64 on 30 September, with frozen identity
`80d3d8c880f6baf6c8423098e26b0714877f03526d4e60ad0331632567be0554`.
It retains all 18 matched-audit states, four D/N combinations, prescribed phi,
centered/correction/combined material and vorticity actions, and six constant
diffusion channels at both accepted spans. All 33 preflight owners passed;
bounded action replay is within 5.87e-12 and frozen diffusion replay within
4.23e-10. The 36 focused campaign/material/channel tests pass. Representative
pilot batches project 2.18–3.42 hours with two CPU workers and approximately
0.46 GiB of checkpoint output; pilot peak RSS is 1.76 GiB, versus 2.56 GiB in
the broader preflight. The checkpoint supervisor chains all grids, validation,
analysis and independent final reduction in
`work/q07_material_global_20260930`; quiet supervision uses the existing task
heartbeat. No new traces or numerical policy changes are involved.
Scientific results are now reviewed below. Complete-owner physical-volume N-O/O-R/N-R
RMS, maxima and signed integrals are retained for nine regions, with smooth
and wavelength stress controls reported separately. Neither current–phi/SAT,
exterior crossings, evolution nor production is qualified by this campaign.

Supervision check, 30 September 19:05 UTC: N32 completed and validated; N48
stopped after two saved chunks at the existing 1e-12 absolute center-b replay
gate. A bounded diagnostic reproduces a 1.0703e-12 discrepancy in b^theta
(relative 3.0085e-13) at raw cell 362, N48 axis core. Re-evaluation using the
original six-slot batch layout matches exactly, with identical center
coordinates. This establishes batch-shape floating-point sensitivity rather
than different geometry inputs; it is not a measured Q07 convergence failure.
Evidence is in `work/q07_material_global_20260930/center_geometry_audit.json`.
At that halt, completed chunks remained intact; a documented harness
revision was needed before resume, with
explicit source provenance and replay validation. No gate was relaxed and no
operator/reconstruction policy changed at the halt. The subsequent authorized recovery is below.

The user subsequently authorized relaxing this constraint and continuing.
Q07 now explicitly uses center-b absolute replay tolerance 1e-10 with rtol=0;
the package default stays 1e-12. Reference/operator arithmetic is unchanged.
The revised campaign identity is
`9dc56499db7428e4d5c8e7fbddbfb8ad6522a8d96f70b35a596863beea0ba274`.
The original source, design and receipts are retained under
`tolerance_revision_v2/original`; migration checks preserve all 66 completed
chunk arrays bit-for-bit and record their original computation identity.
N32 was revalidated. Thirteen focused tests verify the configurable guard,
unchanged reference arithmetic and continued rejection of larger mismatches.
Only this geometry replay guard is relaxed; scientific, finite-value,
diffusion replay, constant, coverage and resource gates remain unchanged.

**Completed global static review, 30 September 2026.** The revised campaign
completed all 793 chunks and 313,696 owners across N32/N48/N64. Independent
reduction, coverage and artifact hashes passed. See the
[scientific review](../../../../work/q07_material_global_20260930/scientific_review.md),
[frozen report](../../../../work/q07_material_global_20260930/report.md) and
[full orders](../../../../work/q07_material_global_20260930/orders.csv).
Maximum accepted-diffusion replay difference is 1.7141e-9 (<1e-8); constant
N-O maximum is 5.5101e-8 (<1e-7); characteristic fallback count is zero,
minimum thermodynamic slot 0.8466 and peak reported worker RSS 2.927 GiB.
The explicit center-b tolerance migration above remains part of provenance.

The five combined material fields and vorticity show smooth global N-O orders
3.859–3.997 across all four uniform/mixed D/physical-normal-N cases; every
smooth region exceeds 3.816. All wave global N-O orders exceed 2.918 and none
of these regional RMS errors rebounds. Seven distinct phase-pi/2 inner wave
output/field cases (28 with BC repetitions) have N48-to-N64 order 1.832–1.996;
transition and wall combined orders remain above two. The shortest wavelength
0.35 is still a stress control: N64 Ve relative N-R reaches 4.42% globally and
7.45% at the wall. Both-span constant diffusion channels replay the accepted
operator; smooth global N-O orders are 3.837–4.034 and smooth regional orders
exceed 3.465. Wave diffusion has local exceptions down to 1.024 in the last
two aggregate rings and 1.511 in the transition, despite global order >2.781.

**Continuum accuracy is a separate qualification limit.** Smooth density,
Te and Ti global N-R rebound at N48 through O-R. Smooth Vi/Ve N-R decrease,
but N48-to-N64 orders are only approximately 1.04/1.25; smooth vorticity has
2.06/3.01 (uniform D). N64 relative global N-R is about 0.58% for density/Ti,
0.53% Te, 0.014% Vi, 0.056% Ve and 0.017% vorticity. For Ve the characteristic
correction increases N64 N-R from centered 0.00514 to combined 0.03311
(6.44 times), principally in the exact-slot correction action; this is a
numerical discretization cost, not automatically a reference bug. O-R also
carries the density/temperature/diffusion N48 rebounds; the campaign alone
does not prove coil-ripple causation for every exception. Local maxima can
rebound despite decreasing RMS. Small signed error integrals do not establish
exact conservation. Full N-O/O-R/N-R norms, orders, maxima and signed integrals
are retained in the linked review and analysis.json.

This supports retaining the individual static material/vorticity implementations
and existing constant diffusion channels with these explicit limitations; it
is **not** a uniform second-order N-R pass or full Q07 closure. The next work
is the bounded current–phi/SAT integration and boundary-work audit using the
specialized compatible pair. Exterior crossings, evolved stability/accumulated
drift, Q08 span/support selection and time/memory audit, and production remain
open. No donor/degree/correction policy or diffusion default changed; no
production promotion occurred. Campaign supervision is complete.

**O-R mechanism audit, 30 September 2026.** The subsequent
[fixed-location exact-value audit](../../../../work/q07_or_audit_20260930/report.md)
replays all twelve selected global maximum owners within 1.14e-11 and varies
seven trace spans on actual HSX geometry. On ordinary fixed lines, centered
and corrected velocity actions recover approximately second-order span
scaling; native eta-grid sampling alone produces low orders/rebounds. On the
line through the N48 density maximum, holding the N64 span fixed while
sampling 32/48/64 eta phases gives centered density O-R RMS
6.5745e-5 / 7.4849e-3 / 8.6188e-5. These are bounded unweighted line diagnostics,
not new global orders. Mode 48 and its sidebands dominate ordinary-line error
spectra, linking part of this limitation directly to the P roadmap's ripple
band. Extreme hotspot lines have finer geometric structure; its physical
versus interpolation origin is not yet isolated.

The dominant centered density/compression mechanism is the constant-flux
geometry defect epsilon = D_delta(1) - div(b). At the actual N48 density
maximum, centered O-R is -0.0545201, its -n*Ve*epsilon contribution is
-0.0546284, and the remainder is +0.00010827. Subtracting this diagnostic
contribution leaves combined error -0.0134371, so the characteristic correction
still matters. Velocity correction splits into an O(delta^2) dispersive and
O(delta^3) dissipative term; rapidly varying geometry and changing cancellation
explain additional local non-asymptotic behavior. The N48 density hotspot's
fixed-location span order approaches 2.08 only at shorter spans. RK64-to-128
and reference-step differences are at most 5.9e-9 and 8.0e-9 in the original
audit, far below these errors. This is not evidence of noisy R or poor donor
reconstruction, and is not an unconditional continuum/stability pass.

A concrete next bounded candidate is a geometry-only center-weight correction
D_delta(F) + [div(b)-D_delta(1)]*F_center, preserving the cap machinery. It was
only decomposed diagnostically here, not adopted, reconstructed, or globally
qualified. Assess its constant/product consistency, geometry sensitivity,
replay and balance before altering the accepted tube action. Keep velocity
span/correction sensitivity separate and retain Q08 selection; no default,
donor, production implementation or completed campaign artifact changed.
Full current–phi/SAT integration and evolved stability remain open.

**Geometry-consistent tube candidate tested, 30 September 2026.** The
[derivation and bounded comparison](../../../../work/q07_geometry_balanced_20260930/report.md)
clarifies that the existing tube already includes nonzero div(B)/B. The new
research-only form is D_bal(F)=D_old(F)+[div(b)-D_old(1)]*F_center, equivalently
B_center*b_eta/(2*delta)*[(F_plus-F_center)/B_plus -
(F_minus-F_center)/B_minus] + div(b)*F_center. Only the center weight changes;
cap traces, donor/BC rows and field-gradient-free FCI application are retained.
Geometry div(b) is prepared and checked against div(B)/B-b·grad(log B), with
no div(B)=0 assumption or runtime elliptic solve. A G(F)+div(b)*F product-rule
control was also evaluated; it has no consistent accuracy advantage.

The prototype completed 57 complete-owner samples (27 matched, 21 prior,
nine distinct global hotspots), 18 states, four D/N/mixed combinations and
centered/combined actions: 24,624 records, 50.9 seconds, 1.670 GiB peak RSS.
Baseline replay <=5.64e-13, independent action algebra <=2.67e-15, reconstructed
constant centered N-R <=7.53e-9, zero fallback, finite positive states. Both
velocity rows are bitwise unchanged and the characteristic correction is
unchanged up to subtraction roundoff. Geometry/reference sensitivity remains
bounded; exact-scalar constant consistency is with the prepared div(b), not
a claim of exact geometry derivatives.

On the matched smooth N64 sample, centered density/Te/Ti N-R RMS improves
8.54x/5.08x/8.33x. At N48 hotspots, centered improvements are about
452x/391x/471x. N-O stays essentially unchanged. The combined matched N64
errors instead increase 1.88x/1.69x/1.80x through loss of cancellation with the
unchanged characteristic correction; N48 hotspot combined errors decrease.
Wave-group errors are mostly unchanged. These are bounded comparisons, not
global convergence or universal accuracy claims. Retain this candidate for
further coupled-correction and balance/compatibility audit before deciding on
a global rerun. No accepted divergence/diffusion, donor policy or production
default was changed; the existing Q06 qualification is not silently amended.

**Paired balanced-tube global comparison completed, 1 October 2026.**
[Independent review and full orders](../../../../work/q07_balanced_global_review_20261001/report.md),
[validation](../../../../work/q07_balanced_global_review_20261001/validation.json).
The remote run completed after the metadata-hash cache repair at commit
`a5638ac34b17333576b264710a55d038acfd5ce0`, identity
`117daaec352ad2817b3dc9a4c4db5b9f6ee939fb9c4fb1b98a3404fed646694b`.
All 793 chunks and 313,696 complete owners at N32/N48/N64, 18 states and four
D/N/mixed combinations were independently reduced and their bundled hashes
verified locally. Original global replay is within 1.17e-9 (maxima), below
1e-8. Zero characteristic fallbacks; peak worker RSS 1.973 GiB. The remote
used 128 CPU workers in a GPU allocation; three numerical stages took 15m08s.
No new tracing, donor changes, velocity/diffusion changes or production promotion.

The centered geometry correction is strongly supported: every smooth regional
and global RMS improves at each resolution/BC; N64 centered N-R decreases by
112x/57x/119x for n/Te/Ti. All smooth centered regions are monotone, but global
48→64 N-R orders remain 1.52/1.68/1.42, with approximately 0.16–0.19
fine-interval density/Ti orders in the last-aggregate/transition regions.
Constant centered N-R is at roundoff against the prepared geometry target.
Smooth N-O remains about fourth order globally and >3.817 in all regions;
all affected-field regional wave N-O orders exceed 2.000. These claims concern
n/Te/Ti only; excluded velocity equations receive no new qualification.

Combined smooth RMS also improves everywhere compared with the old operator,
by N64 factors 2.65/1.47/2.63, but retains the N48 rebound: global D N-R RMS
n = 1.218e-4/3.104e-4/8.692e-5, Te = 1.502e-4/3.845e-4/1.054e-4,
Ti = 7.260e-5/1.847e-4/5.164e-5. The unchanged characteristic correction
now dominates O-R and N-R; 86.35–86.37% of N48 smooth combined squared error
is in the two wall rings. The residual maximum moves to owner 82867 (radial
index 46, theta 30, eta 19). Exact-slot O exhibits the same effect, so this is
not evidence of a wall donor/BC reconstruction defect. Short-wave limits remain:
N64 combined wave relative RMS reaches 4.36% globally and 7.33% at the wall;
combined λ=1.4 controls also rebound at N48. O-R is not automatically a
reference bug, and physical coil ripple versus interpolation structure is not
uniquely separated by this campaign.

Keep the balanced centered formulation as the preferred candidate, with no
automatic default change or uniform second-order/Q07 closure. Next bounded
work should isolate the exact-slot characteristic correction at fixed physical
points (including the new hotspot), consistent inner/outer span refinement,
and odd/dispersive versus even/dissipative contributions with the balanced
centered action. No further donor tuning or identical global rerun is indicated.
Current–phi/SAT compatibility, balance and evolved stability remain open.

**Exact-slot characteristic span audit completed, 1 October 2026.**
[Detailed report](../../../../work/q07_characteristic_span_audit_20261001/report.md),
[validation](../../../../work/q07_characteristic_span_audit_20261001/validation.json).
At 696 actual HSX points, all 18 MMS states and five material components,
consistently shorten inner/outer legs from h48/16 through h48/512, holding
physical points and h48=2π/48 fixed. These are bounded span orders, not global
grid orders. Native hotspot exact-slot actions replay the global oracle maxima.
No donor, degree, boundary policy, accepted operator or production default changed.

At N48 owner 82867, the even/dissipative characteristic term contributes
99.96–99.998% of the smooth n/Te/Ti correction; Ve is instead about 97%
odd/dispersive. An exact product decomposition of the smooth MMS on the real
traces attributes 99.987–99.988% of that scalar dissipative contribution to
transverse trace motion, rather than explicit eta trigonometry. The coupled
matrix includes fast local eigenvalues approximately ±61.5, amplifying the
fourth-difference signal. This does not uniquely separate physical coil ripple,
magnetic interpolation structure and coordinate-map structure.

The standard span is preasymptotic at the hotspot: smooth density correction
is 0.039785/0.010461/0.001502 at h48/32,/64,/128. Ve initially worsens at /64.
At /256→/512 the separated smooth dispersive/dissipative orders approach
1.94–1.95 / 2.993–2.994; balanced centered orders approach 1.989.
The bulk-line controls reach the expected second/third orders much earlier.
All wave controls on the fixed 192-point hotspot line also approach these
orders for the separate pieces. Individual total corrections can rebound by
loss of signed cancellation even while both pieces decrease; retain these
exceptions rather than claiming universal monotonicity.

Sampling alone produces a large rebound: at fixed u,theta and fixed h48/32
span, 48 eta samples yield 124–130 times the n/Te/Ti line RMS of 32 samples.
This identifies a sharp eta-localized feature, not the entire global error
budget; the 192-point line is not a certified quadrature reference. RK64→128
changes the smooth hotspot scalar corrections by ≤4.14e-11 (Ve 5.88e-9),
far below the observed errors. Constants give zero correction; decomposition
replay is within 8.72e-13; all states/eigensystems/traces pass their checks.
Since correction R=0 exactly, its rebound cannot be attributed to numerical
differencing of the continuum reference. Exterior wall crossing remains untested.

Retain the balanced centered tube as preferred and the correction as provisional.
Before another global run or span decision, compare actual reconstructed actions
at h/32,/64,/128 with both legs scaled together, fixed donors/degrees/BCs,
N-O/O-R separation, row amplification and centered/correction/combined outputs.
Include maxima, nearby and fresh wave controls, inner/transition and wall rows.
Do not adopt h/512 from this oracle-only audit. Q07/current–phi/SAT/evolution
and production remain open; no uniform global second-order claim follows.

**Actual reconstructed shorter-span comparison completed, 1 October 2026.**
[Report](../../../../work/q07_reconstructed_span_comparison_20261001/report.md),
[validation](../../../../work/q07_reconstructed_span_comparison_20261001/validation.json).
42 complete owners at N32/N48/N64 cover core, inner/transition, bulk, the
native maxima and neighbors, and both wall layers. All 18 original states plus
four held-out waves, all five material fields and four D/N/mixed combinations
were evaluated at inner h/32,/64,/128 with outer spans twice the inner.
Degrees, donor IDs, frozen repair choices and BC machinery are identical.
The frozen runtime was not relabeled or generalized: a separate analysis
adapter calls its unchanged row builders at new points, with true span weights.
Nine original-span scalar replays agree within 4.76e-11; independent full
five-field/phi public-runtime replays at the N48 wall hotspot and complete
64-member N64 core agree within 3.55e-9 (gate 1e-8).

The oracle improvement survives reconstruction. N48 hotspot smooth combined
N-R for n is 0.039829/0.010480/0.001507 at h/32,/64,/128, while N-O stays
around 4e-9. Te/Ti also improve about 26x at /128. Owner-data gradient-row
L1 amplification changes by only -1.32% to +1.39%; boundary-node rows by
about -2.26% to +2.01%. Independent endpoint/Dirichlet-query noise bounds
still scale inversely with span. Constant N-O max 1.413e-8 passes the unchanged
1e-7 stiff-row gate; all positivity, finite, eigensystem and trace gates pass.
Numerical run 136 seconds locally, peak RSS 3.11 GiB. No exterior crossings.

This is not uniform improvement or global convergence. Reconstruction error
remains approximately fixed: original wave combined N-O RMS changes by only
about 0.2% or less; held-out scalar wave RMS increases 1.3–1.6%. Losing
cancellation can worsen N-R: a N48 wave Ve error grows 1.17→26.53 as N-O
remains about -22.8 and O-R changes +21.8→-3.72. Held-out velocity sample
relative RMS remains about 26% on this deliberately difficult sample; these
are not global norms. Sampled N48 smooth hotspots remain larger than the
sampled N32/N64 maxima even at /128, so a monotone global sequence is not
predicted or assumed. No coil-ripple-only attribution follows.

Prefer h/128 as the next paired global *comparison candidate*, retain h/32
baseline and h/64 sensitivity, and keep both legs scaled together. New shorter
endpoints are required; geometry, owner states, repair choices and continuum
references can be reused. Do not promote a default or launch automatically.
The next global question is domain-wide absolute errors/remaining rebound,
not an assumed second-order pass. Short-wave reconstruction response,
current–phi/SAT, conservation, exterior crossing and evolution remain open.

**Next sequence for the current traced path:**

1. Freeze a term-by-term contract against the existing parallel RHS: evolved
   variables, signs, normalization, product locations, BC data and selected
   characteristic correction. Keep h/32 traced G/D, five eta planes and donor
   choices fixed; leave diffusion's accepted formulation/spans unchanged.
2. Start with a bounded density-flux block on actual HSX. Declare whether the
   scalar flux is formed from owner data then reconstructed, or from separately
   reconstructed factors at caps; these operations are not interchangeable.
   Select and record that contract before evaluating accuracy. Apply the
   current tube divergence including its divB term. Derive product wall data
   consistently (e.g. physical-normal derivative of n*u uses the product rule),
   rather than reusing either factor's BC. Keep prescribed data separate from
   physical sheath-law certification.
3. Add temperature advection/compression and velocity advection/pressure
   forces, followed by vorticity and current/electrostatic blocks. Reuse the
   existing characteristic physics; qualify centered terms and any selected
   characteristic correction separately. Preserve specialized current–phi/SAT
   compatibility; generic G/D are not substituted as an adjoint pair.
4. Integrate the accepted cap-gradient diffusion into the existing thermal and
   viscous channels in the verification path, auditing variable coefficients,
   signs and normalization without introducing an owner-field D(G f) operator.
5. Run bounded N32/N48/N64 tests covering core, agglomeration, transition, bulk
   and wall, with positive smooth states, independent wave controls and both
   D/physical-normal N data. Report each term and coupled-block N-O, O-R, N-R,
   constants, BC lifts, limiter/fallback activation and net balance separately.
   Define O using exact fields in the same declared product/stencil assembly;
   R is the analytic continuum midpoint term with the established owner
   projection. Never let cancellation in a block hide a failing term.
6. After bounded review, prepare term-resolved global static qualification,
   reusing endpoints and applicable prepared reconstruction data. Scale the
   execution location from measured time/memory rather than assuming a new
   tracing campaign. Full coupled RHS, span/support selection and performance
   audit remain Q08; evolved MMS/accumulated drift and promotion remain later.

The first implementation assignment should cover items 1–2 and their bounded
checks, not all transport physics or a new global campaign at once.

- Implement and qualify density flux, temperature advection/compression, velocity
  advection and pressure forces, vorticity transport, and current/electrostatic
  couplings. Reuse established characteristic physics while correcting the
  inconsistent geometric transfer or assembly stages.
- Extend the certified diffusion action to existing temperature and
  viscosity/diffusion channels in the verification path. Do not introduce new
  transport physics or perform full model/default integration at this stage.
- Use smooth positive MMS states that avoid unnecessary limiter activation;
  test positivity protection separately and report activation near smooth
  extrema. Record the exact selected schemes and fallback use.

**Gate:** individual terms and physically coupled blocks pass HSX global
operator gates. A sum cannot hide a failing component. Retain the applicable
balance and structural identities.

## 4. Phase C — Full parallel-system MMS

### Q08 — Certify the frozen coupled parallel RHS

**Dependencies:** Q07.

**Engineering work started, 2 October 2026.** The
[shared extraction plan](q08_shared_extraction_plan.md) and
[current-code/P08 reuse audit](../../../../work/q08_extraction_audit_20261002/report.md)
are complete. This starts the engineering portion while the documented Q07
coverage and physical-wall work remain open; it does not close either rung.
Reuse P's artifact/batched-row/halo infrastructure with separate P/Q policies.
The bounded actual-HSX bridge replays Q D/N values and gradients through P's
existing kernels within 8.5e-14, and 22 relevant P infrastructure tests pass.
First implementation slice: common polynomial primitives and one paired-span
Q preparation pass, preserving existing rows, choices, lifts and coefficients.
Then compact storage, array-data runtime plans, shared six-field reconstruction,
measured factoring, matched eta sharding and a memory/runtime pilot. P's tensor
format assumes four eta planes and its wall lowerer 28 nodes; Q needs five and
35, so those formats require explicit adapters/extensions. No numerical-policy,
default, production or span-selection change is authorized by this extraction.

**Full-grid representation and actual-GPU replay subgate passed, 3 October
2026.** The [returned Q08 review](../../../../work/q08-verification-v2-149644b9-Apdb1a/local_report.md)
validates all 1,056 records on N32/N48/N64, all 22 states, four D/N/mixed
patterns, both diffusion spans and one/four A100 execution. Shared extraction
now has full-domain implementation replay and measured device-resource
evidence. Numerical span/eta-support selection, the remaining static scientific
coverage, final coupled-RHS freeze and evolution remain open. The polynomial
split remains opt-in; this pass does not change production defaults. See the
returned-campaign entry below for timing and the host-memory estimate caveat.

**Next scientific scope selected, 3 October 2026.** The user selected global
static N-O-R qualification of the current six-field RHS before the pending
span/eta-support comparison. The campaign at
[`scripts/q08_rhs_mms_global`](../../../scripts/q08_rhs_mms_global/README.md)
reuses the completed C3 banks, RK4-64 endpoints and polynomial GPU implementation.
It preserves material h/32, outer h/16 and both diffusion spans. This does not
settle the final span/support choice or close Q08.

#### Q08 traced-span selection before freezing the coupled RHS

**Required step — added by user decision, 30 September 2026; bounded comparison
completed 3 October; h/32 and five planes selected by the user on 4 October.** The
[span/support/stiffness report](../../../../work/q08_span_support_20261003/report.md)
supports retaining five eta planes and identifies h/64 as a possible future
accuracy alternative. The selected h/32 configuration includes diffusion;
the existing twice-inner characteristic separation is preserved. The user
waives the unequal-resolution check as a Q08 prerequisite, retaining that
coverage limitation. Restricted-patch spectra do not certify a production
timestep; full-domain dynamics remain for Q09. The design criteria below
continue to govern any later alternative-span qualification.
Investigate a suitable denominator `d` for total traced cap separation `h/d`
(`h = delta_eta`, each symmetric leg `h/(2d)`). Use `d` here to distinguish
the denominator from grid resolution N. Keep h/32 as the working traced-G
choice through the current Q06/Q07 work. Conduct selection early in Q08,
when the selected G, D and transport blocks are available, alongside the
time/memory audit and before freezing the coupled RHS. Q09 then confirms
the chosen configuration in evolved MMS and records the production decision.
This deferred selection step was not a prerequisite to the scoped Q06 closure.

- Compare a small predeclared set, initially h/16, h/32 and h/64, on the same
  actual-HSX owners, fields and boundary data. Extend to a shorter span only
  if the error balance warrants it. Reuse verified endpoints where available;
  keep the magnetic evaluator, RK4 policy, polynomial degree and donor policy
  fixed to isolate span effects. Start bounded, then qualify the selected
  candidate globally if its numerical action changes. Do not run a full
  global campaign for every trial denominator.
- In a separate controlled reconstruction comparison, evaluate a common
  symmetric three-plane eta quadratic on k-1..k+1 against the current common
  five-plane eta quartic on k-2..k+2. Hold transverse support and operator
  formula fixed, compare at a common trace span first, and then check the
  interaction with span selection. Three-plane quadratic reconstruction is
  not ruled out by a second-order target; polynomial reproduction tests must
  match the declared degree. Compare accuracy, resolved-wave amplitude/phase,
  coefficient sensitivity, boundary consistency, memory/gather and setup cost.
  Recheck wall conditioning if the eta support changes there. The historical
  [common-polynomial comparison](../../../../work/q_fci_common_eta_polynomial_20260928/report.md)
  favored a common quartic over the old piecewise/common cubic controls; it
  did not establish that three planes fail. Preserve one common polynomial
  across both caps and nested stencil evaluations: do not reintroduce the
  moving-four-plane first-derivative interface. This is a deferred design
  comparison, not authorization to change the current five-plane default.
- Assess the multi-eta-plane reconstruction as well as the finite-span
  difference. Measure endpoint value errors, their signed correlation across
  the two caps, the resulting N-O action error, row/derivative amplification,
  polynomial reproduction and conditioning. A shorter span does not add
  independent information between eta planes or narrow the donor footprint.
  For traced G, the relevant error is
  `b_eta*(e_plus - e_minus)/(h/d)`: common reconstruction errors may cancel,
  while differential errors may be amplified. With a common smooth polynomial,
  the small-span limit is its directional derivative error, not necessarily
  an unbounded inverse-span growth. Verify this behavior rather than assuming
  either improvement or loss of accuracy from cap proximity alone.
- Report N-O, O-R and N-R separately, with identical owner projection, regional
  RMS and maxima, for D and physical-normal N data. Include inner/agglomerated,
  transition, bulk and wall sites, prior O-R hotspots, fresh sites and
  nonpolynomial wave orientations. Preserve signed cancellation diagnostics;
  do not select a span from favorable cancellation in one manufactured field.
- Deferred by user decision on 4 October; not a Q08 prerequisite: test
  representative production resolution ratios by refining perpendicular
  resolution at fixed eta spacing, and vary eta spacing with perpendicular
  resolution controlled. Include resolved-scale amplitude/phase response and
  eta variation; a shorter evaluation span cannot recover unsampled parallel
  structure. Distinguish reconstruction bandwidth from stencil truncation and
  from rapid variation in the HSX field-line mapping. Retain the exact-endpoint
  span audit as a baseline, not a replacement for numerical-state tests.
- Evaluate G, tube D and accepted cap-gradient diffusion according to their
  own formulas; one denominator need not be optimal for all three. Check the
  selected coupled terms and Q06 structural/boundary properties together.
  Requalify any changed span/action explicitly; the current diffusion freeze
  is not silently replaced by the gradient choice. Record boundary crossings,
  exterior caps and reentries; absent events do not qualify ghost behavior.
- Compare setup/tracing and warmed RHS application costs, memory, precision
  sensitivity and bounded stability/timestep evidence using the Q08 performance
  methodology below. Use available traced-Q04 evolution evidence where relevant;
  retain its outstanding gates. Do not infer production timestep feasibility
  or a speed improvement from a smaller geometric span alone.

**Deliverable/gate:** a reproducible span-versus-error/cost report and a frozen
per-operator span configuration or explicitly justified geometry-only policy,
with its qualified resolution range and remaining limits. Base selection on
total accuracy, reconstruction robustness, structural behavior and cost, not
exact-endpoint O-R alone. Replay unchanged actions; repeat the required static
qualification for changed actions before freezing Q08. Q09 must validate
evolved accuracy/stability with this choice before production promotion.

**Required Q08 performance and memory audit — added by user decision,
29 September 2026.** Measure the reusable Q05–Q07 implementation and coupled
parallel RHS, following the separation of setup/artifacts/JAX application in
P08. This audit is part of Q08 completion; it is not launched by this plan and
does not replace numerical qualification.

- Inventory per-grid artifact bytes, nonzeros, index/mask overhead, duplicate
  data, raw versus owner/factored layouts, field-independent geometry and
  boundary storage. Estimate N32/N48/N64 resident memory before full builds.
  P08's approximately75GB N64 plain-CSR artifact is a lesson to avoid, not a
  memory budget for Q. Prefer exact factoring, deduplication and bounded chunks
  when useful, with replay proving unchanged actions.
- Report host setup stages separately: cached/new tracing when applicable,
  magnetic coefficients, donor selection/fit, wall preparation, artifact
  writing/loading and device transfer. Separate cold and warm caches. Do not
  include analytic MMS reference work in claimed timestep cost.
- Separate compile/first-call latency from warmed, device-synchronized timing
  of each operator and the coupled RHS. Include batching over physical fields,
  boundary-data refresh, per-RHS cost and estimated per-step cost for the stated
  integrator/stage count; later Q09 measures actual evolution. Report repeated
  samples and variation, not unsynchronized JAX enqueue time.
- Measure peak host RSS (per worker and aggregate), persistent device payload,
  device peak/temporary workspace, executable/compile memory and allocations
  during repeated application. Distinguish measured live allocations from
  allocator reservation and nbytes estimates. Check cache growth, retention and
  unintended recompilation as state/BC values change at fixed shapes.
- On available CPU and actual target GPU hardware, record device/model/count,
  versions, precision, affinity, worker/thread counts, problem size, field
  count, chunk/layout identity and warmup. Report CPU-only or extrapolated
  quantities honestly when GPUs are unavailable. Do not infer A100 throughput
  or memory from local CPU tests.
- Compare matched single-device and eta-sharded execution with the same
  operator, catalogue and precision: communication/gather volume, synchronization,
  peak memory, throughput and scaling efficiency. Keep numerical replay and
  source/boundary invariants attached to every performance result.
- Profile the dominant kernels and memory traffic before optimizing. Prefer
  semantics-preserving reuse/fusion/factoring; support, polynomial, precision,
  tracing or BC changes require a separately qualified numerical decision.
  Compare meaningful baselines only; campaign setup speed is not RHS speed.
- Deliver a reproducible benchmark command and memory/time tables with measured
  versus projected costs, bottlenecks and actionable recommendations. Freeze
  target hardware and resource budgets before a full audit. Record whether the
  intended production configuration fits them or remains an engineering blocker;
  do not invent arbitrary timing pass thresholds after measurement. No default
  promotion solely because accuracy passes if the runtime representation is
  not feasible.


- Consolidate and replay shared Q05–Q07 machinery before assembling the coupled
  verification path. Reuse components across operators rather than copying
  their geometry, reconstruction, or runtime infrastructure.
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
operator criterion with independent sources and qualified error budgets. The
Q08 traced-span selection and time/memory audit are complete, reproducible and
qualified for the frozen configuration, with
resource feasibility or remaining engineering blockers explicitly recorded.

### Q09 — Certify evolved coupled MMS and complete final integration

**Dependencies:** Q08. The reconstructed-phi leg additionally depends on
independently certified polarization in perpendicular **P07**.

- Evolve to a fixed physical time with timestep refinement and the same global
  solution-order target. First prescribe manufactured phi; then repeat with
  the independently certified polarization closure. Keep the latter result
  explicitly pending if the cross-roadmap dependency is not ready.
- Confirm the Q08 traced-span choices against evolved solution accuracy and
  timestep/stability behavior at their stated resolution ratios before making
  them production defaults. If a span changes, requalify affected static and
  structural actions as well as the evolved test; do not silently retune it.
- Verify conservation/balance identities, smooth-path JIT/JVP behavior, and
  single-device versus eta-sharded agreement; qualify temporal, reference,
  and polarization-solver errors separately.
- Promote the verified configuration into the shared model/MMS workflow only
  after the gates pass, using the consolidated preparation/application path
  verified in Q08–Q09. This is the full integration milestone; record final
  selector/default decisions and avoid reimplementing per-operator machinery.
  Update current-behavior architecture documentation to describe the accepted
  implementation and its limits. Blob-driver synchronization remains deferred.

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
| Q03 | Diffusion repair | Q02 | direct static accuracy qualified; frozen traced static reconstruction accepted with exceptions; minimum-principle repair deferred | [Reference closure](../../../../work/parallel_q03_reference_closure_20260922/report.md): orders radial `3.212/3.011`, angular `3.509/3.285`, mixed `3.346/3.168`; all bounded empirical reference-budget point estimates are below 10%, with narrow N64 angular margin and sampling spread explicitly reported. Numerical actions unchanged; only 0.317% of N64 reference faces refined. [N32 heat](../../../../work/parallel_q03_direct_heat_spot_20260922/analysis/report.md) undershoots to `0.9982975`; [actual N48-cubed heat](../../../../work/parallel_q03_direct_heat_spot_N48_20260922/analysis/report.md) undershoots to `0.9992172`. Both conserve heat, remain finite/nonnegative and have nonincreasing variance, but neither is a solution-order test or structural pass. User defers repair. |
| Q04 | Diffusion certification | Q03 | direct evolved accuracy passed; traced static reconstruction accepted with exceptions; traced evolved/full certification pending | [Q1 results](../../../../work/parallel_q04_conduction_mms_20260922/report.md), [parent review](../../../../work/parallel_q_progress_review_20260922/report.md): solution orders radial `3.890/2.551`, angular `3.564/3.715`, mixed `3.592/3.389`; temporal fractions ≤`1.03e-9`, propagated reference fractions ≤`3.823%`. CPU/JIT/JVP checks pass; local static orders independently pass despite measurable local/remote differences. Monotonicity remains deferred; general dissipation and eta-sharding are not established. |
| Q05a | Bounded direct-cubic extraction/replay | Q04 accuracy exception | passed | [Evidence](../../../../work/parallel_q05a_direct_cubic_extraction_20260922/report.md): reusable geometry/observation/functional/sidecar and pure-JAX shared-face assembly extracted; frozen bounded N32/N48/N64 donors, coefficients and fluxes reproduce exactly; complete saved-coefficient actions replay at roundoff and agree with sparse actions. Returned-remote cause remains unresolved; no production promotion, Q04 structural pass or Q06 launch. |
| Q05 | Shared traced preparation/application | Accepted traced-static preparation exception | closed by user acceptance | [Corrected extraction and second review](../../../../work/q05_traced_extraction_20260929/review_v2/report.md): 36 focused + 30 curated checks; 42 site/span replays and two full N64 chunks; max discrepancy 4.467e-10 < 1e-8. User closes Q05 on this evidence; complete-domain action replay remains unperformed and is waived as a Q06 prerequisite. No production promotion or Q04 structural/evolved pass. |
| Q06 | Certify parallel gradient/divergence pairs | Q05 closed by user acceptance | closed by user acceptance; scoped static qualification | [Contract](q06_traced_gradient_divergence_contract.md), [global traced G](../../../../work/q06_traced_gradient_global_20260930/parent_review.md), [global tube D](../../../../work/q06_tube_global_20260929/parent_review.md), [structural audit](../../../../work/q06_structural_boundary_audit_20260930/report.md). Exact adjointness and owner-weighted conservation do not hold. [Smooth bulk balance](../../../../work/q06_resolved_conservation_20260930/parent_review.md) improves for all eight fields; N64 imbalance/activity 0.004–0.115%. User closes Q06 on 30 September with unchanged operators and documented approximate-balance/non-adjointness exceptions; no evolved or production pass. Continuum N-R/local maximum limits retained. Exterior ghosts unexercised; D(G f) diffusion unnecessary for closure and deferred. |
| Q07 | Transport and material blocks | Q06 closed with exceptions | five-field static material qualified; bounded prescribed-boundary six-field assembly passed; full closure pending | [Contract](q07_transport_contract.md), [C3 h/32 global review](../../../../work/q07_c3_global_review_20261001/report.md), [six-field assembly](../../../../work/q07_six_field_assembly_20261002/report.md). User accepts the five material fields with documented exceptions. Six-field owner-state assembly replays material, current/phi, vorticity and constant diffusion channels, including D/N/mixed BCs. Keep C3, RK4-64, h/32 material inner and h/16 outer separation, balanced material tube and frozen donors; diffusion spans remain explicit. Next: Q08 engineering consolidation/pilot and remaining C3 static coverage for current/phi, vorticity and diffusion transfer. General basis/enrichment searches are deferred. Physical sheath/SAT, exterior crossings, evolution and production remain open. |
| Q08 | Traced-span selection, frozen coupled parallel RHS and performance audit | Q07; engineering audit may proceed alongside remaining Q07 coverage | in progress: h/32/five planes frozen; local closeout complete; focused global Ti N replay pending | [Shared extraction plan](q08_shared_extraction_plan.md), [GPU review](../../../../work/q08-verification-v2-149644b9-Apdb1a/local_report.md), [global MMS review](../../../../work/q08_rhs_mms_return_analysis_20261003/report.md), [local closeout](../../../../work/q08_closeout_20261004/report.md). All 1,056 implementation replays pass; N64 warmed RHS is 34.74/10.30 ms on one/four A100s. All base operator spans use h/32, with paired twice-inner outer characteristic sampling. Unequal-resolution check waived as a prerequisite, not tested. Memory guard calibrated against both remote campaigns; 21 harness/closeout tests and bounded Ti replay pass; signed O/R corrected at all 313,696 owners. Full numerical Ti replay requires the existing Perlmutter banks and actual GPU, without another complete RHS campaign. Existing scientific exceptions, full-domain dynamics and production limits remain. |
| Q09 | Evolved MMS and promotion | Q08; P07 for reconstructed phi | pending | Both phi legs, solution gates, structural and execution qualification; confirm Q08 span choices in evolution before production defaults. |
| Q traced-return research | Frozen selective-repair traced diffusion | Completed layered/balanced/selective campaigns and interface audit | static reconstruction accuracy accepted with documented exceptions; operator frozen | [Frozen contract](q_traced_diffusion_frozen_contract.md), [global result](../../../../work/q_fci_selective_global_20260929/report.md), [interface audit](../../../../work/q_fci_repair_interface_audit_20260929/report.md). Whole-inner nonconstant RMS order>2 for both intervals/spans; localized join/coarse-envelope and O-R limitations retained. Current implementation choice is selective gradient repair, not always-balanced28. Q05 extraction closed by user acceptance on reviewed bounded evidence; Q06 static traced G/D closed with exceptions; next Q07 bounded density-flux block. Production defaults and broader Q04 certification remain unchanged. Earlier projected-return failures remain historical evidence. |

### Q bounded interior return-map audit — 2026-09-23

**Completed after coordinator correction:** The
[corrected v2 audit](../../../../work/q_fci_return_interior_audit_20260923/corrected_v2/report.md)
supersedes the partial mechanism result below. Independent direct polynomial,
remainder-gradient, per-face modal and complete-owner checks pass for both seed
variants on all three resolutions. Weak target-coupled modes explain the
dominant sampled error and fine-interval change. The next recommendation is
same-row cubic-preserving sensitivity feasibility, not unqualified mode removal.
No new global run or policy change follows automatically.

Parent qualification: **partial**, superseding the completion implication below.
See the [parent review](../../../../work/q_fci_return_interior_audit_20260923/parent_review.md)
for confirmed Taylor scaling and per-face SVD bugs and missing diagnostic budgets.
Retain face replay and sampled error norms; do not treat the unavailable modal
analysis as evidence or the proposed correction as the supported next action.

Evidence: `work/q_fci_return_interior_audit_20260923/report.md` and its
saved arrays/validator. The fixed nine-owner/54-face audit replayed both
returned seed variants at N32/N48/N64 without new tracing. Independent
face-functional replay agrees with archived rows to at most `7.95e-18`; the
selected tracks reproduce the increasing exact-secant-minus-q5 channel toward
N64 in both variants. Source and endpoint extents show that source locality
does not imply endpoint locality. This is bounded mechanism evidence only:
the traced global static gate remains failed, direct-method evidence remains
separate, and no production or structural change is promoted. The next
bounded task, if needed, is a same-face-set single-owner return-functional
probe with traces, q5 target, geometry and wall policy frozen.


### Q07 paired h/128 global preparation — 1 October 2026

The existing `scripts/q07_balanced_global` campaign now compares balanced h/32
with balanced h/128; outer characteristic spans shorten consistently from h/16
to h/64. It retains the previous frozen source bundle, canonical geometry,
saved GPU RK4-64 baseline endpoints, geometry-only repair choices, polynomial
basis, donor support and D/physical-normal N/mixed boundary machinery. Only the
additional short endpoints are traced, with parallel CPU-batched RK4-64.

Coverage is all complete N32/N48/N64 owners (793 chunks), all five material
fields, centered/correction/combined terms, four boundary combinations and 22
states: the original 18 plus four held-out waves. N-O/O-R/N-R, regional maxima,
physical-volume norms, signed integrals and h/32 replay remain separate. No
other parallel operator or production default changes. The source identity is
`0a16837968506693afdbc530216dc24911821cb8c7b4558589c870f833bbc659`.

The bounded evidence suggests a predominantly geometry-driven O-R improvement
at the smooth hotspot: transverse trace motion accounts for 99.987–99.988% of
the density/temperature dissipative contribution. This is variation of the
manufactured field along the geometry-dependent path, not an improvement to its
reconstruction. Attribution specifically to coil ripple rather than magnetic
spline or coordinate-map structure remains unproved. Short-wave N-O limitations
and possible cancellation-driven total-error regressions remain open.

Readiness: 42 complete-owner preflight sites, all cases/BCs/spans, reproduced
saved bounded N/O/R within 1.28e-11; relocated inputs/source verified and ten
focused tests passed. The four-region N64 pilot projects 131–201 minutes with
two local workers, above the local 30-minute target; use remote execution.
Pilot peak RSS was 1.21 GiB. Preflight, pilot, tracing and global scoring are
parallel; checkpoints support identical-identity resume. Reuse the old small
input and canonical archives plus the new 1.6 MB supplement.

The first pilot exposed a constant-only bulk electron residual 1.85e-7 at
h/128 versus 4.71e-8 at h/32, from constant slot residuals <=1.38e-14. Supplying
exact constant owner data gives the same result. Before global launch, the
h/128 constant roundoff gate is explicitly scaled by inverse span to 4e-7;
h/32 stays at 1e-7, scientific/baseline replay at 1e-8 and center-b at 1e-10.
Arithmetic is unchanged. Preserve the original failed-pilot and audit evidence
in `scripts/q07_balanced_global/verification/h128/`; this is not a convergence
acceptance threshold. Scientific regressions must be reported without tuning.

The global comparison is prepared, not scientifically qualified. Diffusion,
vorticity, exterior ghosts, current–phi/SAT, conservation, evolution and
production promotion are not closed by this campaign. See the campaign README
and `verification/h128/` for the frozen contract and operational evidence.


### Q07 sharp geometry feature: toroidal magnetic-spline influence — 1 October 2026

[Bounded real-HSX attribution audit](../../../../work/q07_geometry_feature_audit_20261001/report.md)
identifies a specific mechanism at the N48 scalar hotspot. Jacobian conditioning
and b^eta remain regular. Freezing the coordinate Jacobian retains the density/
temperature dissipative spike; freezing cylindrical B removes about 99.7%.
Changing only toroidal B interpolation to a compact four-point control removes
99.79% of the density dissipative contribution at h/32, while keeping the
original R/Z interpolation. Local cubic/quintic controls change sampled B values
by approximately 0.2%; full-grid quintic instead increases the correction ~7.5x.
These are exact-slot diagnostics with the characteristic matrix held fixed,
not complete alternate-operator or magnetic-reference qualifications.

The periodic cubic prefilter couples all MAKEGRID toroidal planes. An exact
raw-plane decomposition reproduces B within 5.6e-16: planes outside the nearest
four supply 97.07%/99.42% of the projected along-trace B third-difference signal
at h/32 and h/128. Plane 68 degrees (5.208 degrees from the query) has a small
weight -5.787e-4 but very sharp field variation along the sampled R,Z path.
That R,Z location is 5.89 cm outside the modeled wall on the displaced plane.
Thus nonlocal interpolation carries sharp exterior-field structure into the
interior query. This sharpens the earlier generic coil-ripple hypothesis; it
does not show that all physical ripple or every P/Q O-R error is artificial.
A direct/finer independent magnetic reference remains necessary to quantify
physical field error conclusively.

NumPy/JAX RK4-64 endpoints replay within 1.12e-16, raw nodal values match direct
MAKEGRID reads within 1.34e-15, source/input hashes verify, and RK64/128
sensitivity is far smaller than the observed scalar change. Electron dispersive
error remains distinct. Next consider a bounded locality-and-regularity audit
of compact toroidal magnetic interpolation, with derivative continuity at
source-plane changes and periodic seam, div(B), and consistent trace/reference
recalculation. Moving-stencil Lagrange controls are not automatically promotable.
The already prepared h/128 campaign, canonical inputs, accepted Q operators,
and production remain unchanged; no new global run or commit is authorized by
this diagnosis alone.

### Compact magnetic evaluator: bounded qualification (1 October 2026)

The explicit experimental `toroidal_method="compact_c3"` magnetic evaluator
is the leading candidate for subsequent P/Q integration checks. It uses
shared first-through-third nodal derivatives from seven raw toroidal planes
and a septic Hermite interval, giving eight-plane total support and C3 phi
continuity, including the periodic seam. No toroidal prefilter is applied;
R/Z retain their previous cubic spline preparation. `compact_c2` is a
six-plane C2 comparison. Neither is divergence-free. The default remains
`spline`; canonical geometry and the frozen h/128 campaign are unchanged.

Evidence: `work/compact_bfield_qualification_20261001/report.md`, `audit.json`,
`holdout.json`, `broad.json`, and `completion.json` in the workspace. All 27
focused verification tests passed. The real-HSX audit uses 61 locations,
22 states, h/32 and h/128, consistently regenerated traces/factors/references,
and RK4-64/128 comparison. It measures exact-slot O-R, not reconstructed N-O
or N-R. C3 lowers the known smooth density correction from 3.9785e-2 to
6.0913e-5 at h/32 (about 653x); electron and fresh short-wave errors remain
substantial and are not explained entirely by this interpolation artifact.

An independent 1,024-point sample shows C3 RMS div(B) falling from 4.191e-3
to 8.775e-4 T/m, maximum from 0.12393 to 0.011135 T/m, with median/p95/p99
also improved. Native-plane holdout (2-degree training predicts omitted
1-degree samples) reduces RMS B error from 1.533e-3 to 6.006e-4 T. C2 wins
holdout RMS/outliers but worsens median/p95 divergence; C3 provides the better
regularity/divergence tradeoff. This holdout is not proof of physical accuracy
between existing full-resolution samples. Actual native-node replay is within
1.2e-14, host/JAX within 2.0e-15. Coefficient memory is unchanged; warmed CPU
field evaluation is about twice the old spline cost (C3 only about 5% over C2
in the bounded benchmark), not a measured plasma RHS slowdown.

Bounded evaluator implementation qualification passes; production adoption,
full-grid operator qualification, and physical magnetic accuracy remain open.
Next: matched old/C3 P/Q N-O/O-R/N-R replay at core, transition, bulk and wall,
with D and physical-normal N; include magnetic knot neighborhoods and longer
trace legs. Regenerate endpoints and magnetic factors for C3 rather than
relabeling old traces. Before canonical adoption, include the interpolation
option in geometry/cache identity and explicitly version affected artifacts.
Keep the current h/128 campaign separate; this task does not promote defaults
or change Q's accepted reconstruction/span policies.


### Q07 compact C3 regenerated-trace integration — 1 October 2026

The bounded matched comparison in
`work/q_c3_regenerated_trace_bounded_20261001/report.md` validates 46 complete
N32/N48/N64 owners (the prior 42 plus four deduplicated N48 native-knot/seam
controls), all 22 states/five fields/four D/physical-normal N/mixed BCs,
and paired h/32,h/128 spans. Old spline and explicit compact C3 each use fresh
RK4-64 traces, consistent magnetic factors/div(b)/continuum references,
unchanged donors/repair choices and rebuilt frozen Q rows. N-O/O-R/N-R and
each numerical background's R remain separate; this is not a common physical
magnetic-reference error or a global order claim.

The smooth N48 hotspot combined density N-R falls in magnitude from
3.9829e-2 to 6.8829e-5 at h/32 and 1.5072e-3 to 3.8280e-7 at h/128.
Selected-owner physical-volume smooth scalar RMS ratios C3/old are about
0.053 at h/32 and 0.018 at h/128; N-O remains approximately unchanged.
Held-out and short-wave reconstruction floors persist, including roughly
25–30% held-out sample-relative errors. Losing cancellation worsens the N48
hotspot short-wave Ve total from -1.1729 to -25.3416 at h/32 despite reducing
O-R; wall h/128 short-wave velocity RMS also increases by up to about 1%.
Knot-control smooth h/32 Ve RMS increases about 2.2%. These are retained
accuracy exceptions, not reasons to change supports or acceptance policy.

All constant/replay/center-b gates pass unchanged: baseline all-field action
replay <=2.394e-9 versus 1e-8; constant maxima 4.081e-9/1.519e-8 versus
1e-7/4e-7. Eight independent N48 owner/method RK64/128 audits retain action
sensitivity separately (maximum C3 reconstructed action 1.440e-8). All 200
trace caches, source/input identities and N-O-R arithmetic validate; no rank,
exterior query, crossing or reentry failure occurs. Full bounded run: 475 s,
3.32 GiB peak RSS, one CPU x64 worker.

Evidence supports a larger matched C3/old operator qualification retaining
both spans and signed/localized exceptions. Longer legs, exact native-knot
crossings, exterior wall ghosts, current–phi/SAT, conservation, evolved
stability, diffusion composition and an independent physical magnetic
reference remain open. Production defaults, canonical artifacts and the
frozen remote h/128 campaign are unchanged.

### C3 long-leg boundary gate — 1 October 2026

The proposed remote comparison was **h, h/2, h/32, h/128**, with h
one eta-plane spacing: inner caps +/-h/(2d), outer correction points +/-h/d.
The user selected resolving long-leg boundaries before packaging that campaign.
Evidence is in workspace `work/q07_c3_longleg_boundary_20261001/report.md`,
`analysis.json` and checksum-validated `completion.json`. This is bounded
research; no production or frozen-campaign source was changed or promoted.

A 512-seed/resolution/span geometry-only survey of the last eight rings finds
h crossings from the fifth ring inward at N32 and N64 (zero-based rings 27
and 59). Routing solely by the starting owner's last two/four rings is
insufficient. First-hit termination must also handle trajectories that later
re-enter; invalid extended-chart trajectories must never supply action data.

The tested quartic continuation of primitives and inverse B to nominal exterior
points is rejected: three h cases fail positivity, including two negative
inverse-B continuations; maximum continuation row L1 is 1333.36. A first-hit
cut-leg alternative uses actual eta nodes [a,a/2,0,b/2,b], quadratic-exact
centered/backward/forward derivatives and the same geometry-consistent tube
D/characteristic split. Physical queries stay inside/on the wall; RK4-64 base
stepping refines the first-wall event. All 45 original owner/span cases and
16 fresh fifth-ring cases are finite and positive, covering all 22 states,
five fields and four D/physical-normal N/mixed combinations. Symmetric-limit
replay is 1.09e-11; constant gates pass. RK64/128 preserves all audited hits
and changes physical points by <=4.02e-9 m; the largest exact-slot action
sensitivity is 1.39e-4 (smooth-only 3.78e-6), retained separately rather than
relabelled as a 1e-8 replay pass.

Wall-normal elimination satisfies its physical-normal sample-node constraints
within 8.11e-12 algebraically and 2.98e-12 when evaluated as derivatives.
At actual off-node hits, smooth residuals reach 5.15e-5 and wave residuals
9.47: interpolation accuracy remains distinct from enforcing the nodal BC.

The cut-leg geometry is promising, but its long-span accuracy is not qualified.
Reusing the four-ring wall polynomial for fifth-ring crossings extrapolates
poorly on waves. An endpoint-local support comparison (61 owner/span cases,
unchanged degrees/counts; exact h32/N and O/R replay) lowers fresh h wave Ve
N-O RMS from 841 to 470 while worsening density 2.43->4.83 and original-wall
Ve 289->422. It is not a uniform repair and is not selected for production.
Furthermore, smooth density O-R RMS in the original wall panel is 0.546/0.538
at h/h2, versus 0.000419 at h32; the long-span contribution is predominantly
the exact-slot characteristic correction, not reconstruction. These are pooled
selected-owner physical-volume norms, not global norms or convergence orders;
O-R is not being declared a reference bug.

Before the four-span remote handoff: isolate that exact-slot long-span correction
with first-hit geometry fixed; qualify a geometry-only common support over the
reachable interval against these retained candidates; test fresh support switches,
seams/re-entry and physical-normal data; implement/validate batched event tracing.
Keep the accepted short-span operator unchanged. Report actual cut lengths and
row/Jacobian amplification in the span/stiffness audit; finite values alone do
not establish either accuracy or timestep stability. No new remote campaign is
ready from this bounded boundary investigation.

### Working-span decision — 1 October 2026

The user selected **h/32 for now**, superseding the proposed four-span remote
comparison above. For this Q07/C3 qualification direction, retain total inner
cap separation h/32 (caps at +/-h/64) and outer characteristic samples at
+/-h/32, with h one eta-plane spacing. Retain the accepted reconstruction and
boundary policy; do not adopt the experimental cut-leg or endpoint-local
support candidates. Defer long-leg boundary development and the h/h2 campaign.
The h/128 results remain comparison evidence, not the selected working span.

The next qualification target is the compact C3 evaluator with consistently
regenerated h/32 traces, factors and continuum references. Preparation must
still check full-domain crossings and input/replay gates; absence of crossings
in bounded samples is not exterior-boundary qualification. This decision does
not launch a campaign, promote the magnetic evaluator to production, change
the separately frozen diffusion span contract, or alter an existing remote run.
Final span/eta-support selection and measured stiffness remain in the Q08 audit.

### C3 h/32 campaign preparation — 1 October 2026

The replacement static material campaign is `scripts/q07_c3_global`, frozen
identity `ab607bf62cda21dd67935334da9fdea35c2ad0a0c529ff555006aa3a8ed83bc3`.
It retains accepted support/BCs and regenerates compact C3 RK4-64 traces,
magnetic factors and continuum references. All three full-domain trace-only
crossing gates precede full scoring; no exterior ghost or cut-leg experiment
is enabled. All complete N32/N48/N64 owners, 22 states, four BC combinations
and 15 centered/correction/combined outputs are included. Old spline results
are not equality targets; 46 independent C3 bounded owners supply preflight.
Fifteen tests, relocated extraction, all 46 preflights (max 9.611e-12 replay),
a four-region pilot, and two complete N64 bulk/wall chunks with checkpoint
reuse passed. The local two-worker pilot projects 148–228 minutes; remote
execution uses a GPU allocation with CPU computation. Full-grid scientific
qualification remains pending. The latest retired-campaign failure was not
provided and is not claimed diagnosed by this preparation.
Prepared campaign committed and pushed on `2D_fci` as
`bc6f5c817ad8eefe37557989412addfa3c1c3888`; remote branch SHA verified.
Self-contained handoff: workspace
`work/q07_c3_global_preparation_20261001/REMOTE_HANDOFF.md`.

### C3 h/32 full material campaign returned — 1 October 2026

[Scientific review](../../../../work/q07_c3_global_review_20261001/report.md),
[independent validation](../../../../work/q07_c3_global_review_20261001/validation.json).
The prepared `bc6f5c81` / `ab607bf6` campaign completed all 793 chunks and
313,696 complete owners over N32/N48/N64, 22 manufactured states, four D/N/mixed
BC assignments and all five centered/correction/combined material fields.
Locally verified frozen source/input and payload hashes, all associated trace
hashes, and reproduced all totals exactly by independent chunk reduction.
All full-domain short-trace gates passed with zero crossings/re-entries;
exterior ghost behavior remains unexercised. Constant N-O <=7.983e-9, bounded
remote replay <=3.586e-9, reference-step sensitivity <=1.749e-8, zero
characteristic fallbacks. Remote allocation took 29m33s with 48 CPU workers
on a GPU allocation; scientific computation was CPU-only.

The C3 change removes the smooth scalar N48 RMS rebound. Compared with the
previous balanced old-spline campaign, N64 combined n/Te/Ti N-R decreases by
5.81x/5.69x/5.76x while N-O is effectively unchanged. Smooth combined global
N-O orders remain 3.86–4.00. Combined D N-R orders (32→48 / 48→64) are
n 1.803/3.645, Te 1.785/3.680, Ti 1.788/3.662, Vi 3.022/0.983 and
Ve 2.858/1.145; N and mixed BCs show the same trends. Every combined smooth
and wave global/regional RMS decreases. Smooth global n/Ti maxima still
increase 6.54%/0.94% from N48 to N64; monotone RMS is not pointwise monotonicity.

Velocity errors improve only modestly under C3 and remain dominated by O-R.
Exact-slot velocity correction fine-interval orders are 0.945/1.158; centered
Vi O-R also has order 0.950. This is not evidence of a donor or BC fit failure,
and not a demonstrated reference bug. Smooth scalar centered fine-interval
N-R orders remain about 0.93–1.17, despite much better combined scalar orders.
C3 therefore resolves an important scalar geometry pathology but does not
establish uniform second-order continuum accuracy for every material block.

Wave combined global N-O orders are >=3.038/4.443 and N-R >=1.772/1.494.
The whole-inner region retains twelve distinct case/field fine-interval N-O
exceptions at 1.899–1.996; transition N-O is >3.68. Worst N64 combined wave
relative N-R is 4.56% globally and 7.64% at the wall (held-out shortest-wave
Te with N data), predominantly N-O. Two wall correction-only velocity rows
rebound slightly on 32→48, but their combined actions remain monotone.
Retain these exceptions rather than describing a universal second-order pass.

Recommendation: retain C3 h/32 as the preferred research candidate; next audit
exact-slot velocity centered/correction errors at fixed C3 global hotspots and
bulk/transition locations, separating consistent span refinement from eta-phase
sampling. No additional global campaign or donor tuning is warranted solely
by these results. No automatic production promotion or full Q07 closure;
current–phi/SAT, evolution, exterior crossings and Q08 span/support/stiffness
selection remain open. Diffusion and vorticity were not tested in this campaign.

### C3 velocity fixed-location and sampling audit — 1 October 2026

[Report](../../../../work/q07_c3_velocity_audit_20261001/report.md),
[validated artifacts](../../../../work/q07_c3_velocity_audit_20261001/completion.json).
The frozen `ab607bf6` material action was audited with exact manufactured slots
at 708 actual HSX locations, including 24 complete owners containing global,
bulk and transition velocity maxima, all 22 states, fixed-coordinate eta lines
and hotspot neighbors. All 36 selected maximum records replay within
3.243e-11. RK64/128 combined sensitivity <=3.693e-9; algebra and finite/positive
state checks pass, no crossings/reentries or characteristic fallback. Main
run: 271 s / 1.383 GiB; no reconstruction, operator or default changes.

Unlike the old sharp scalar hotspot, the tested C3 velocity hotspots are
already in the second-order span regime at h/32. Holding physical locations
fixed and consistently halving both legs, smooth centered orders across the
24 owners are 1.997–2.001, dispersive 1.988–2.004, combined 1.984–2.006 on
the first halving; combined final-halving orders approach 1.999–2.001. Fixed
hotspot/bulk line RMS shows span² centered/dispersive/combined and span³
dissipative behavior, also for wave controls. Tiny individual dissipative
terms retain cancellation-sensitive ratios, without spoiling combined trends.

A controlled sampling comparison explains low apparent native-grid orders:
at fixed h48/32, changing only eta samples from 48 to 64 increases combined
line RMS by factors 1.761/1.484 (Vi/Ve hotspot line) and 2.585/2.484 (bulk
line). Scaling these measured RMS values by (48/N)^2 predicts independently
traced native-span combined line RMS within 0.46%. N48 samples favorably
underestimate the local error-coefficient RMS. Thus changing samples can yield
subsecond-order intervals or line rebounds even with second-order local
consistency. This is bounded causal evidence, not a full global budget closure
or permission to relabel the measured global orders.

About 85.4–88.0% of smooth combined line-error spectral power lies in eta
modes 38–58, including sidebands of 48 seen in earlier P/Q ripple audits.
Increasing line sampling 192→384 changes smooth combined RMS only 1.10–1.21%
at the hotspot and 0.095–0.098% in bulk. The smooth primitive field's explicit
eta variation is not the main source: transverse motion of the exact trace
dominates the hotspot dispersive correction. Diagnostic freezing of cylindrical
B leaves only 6–12% of the dispersive magnitude at four global hotspot
locations; freezing the cylindrical-frame coordinate Jacobian retains most
of it. Bulk controls exhibit metric–magnetic cancellation, so the attribution
is not universal. These are factor-isolation diagnostics, not alternative
physical geometries. Source-cell crossings occur but no dominant fixed-point
consistency failure is seen. Physical ripple versus residual C3/RZ interpolation
error remains unproved without an independent magnetic reference.

Retain C3 h/32 and the existing donor/BC policy. A shorter span lowers absolute
O-R but is not required to enter the demonstrated local second-order regime,
and does not resolve changing-sample bias by itself. For stricter asymptotic
qualification use shared, adequately sampled physical locations or controlled
eta refinement rather than another identical N32/N48/N64 global run. Static
velocity consistency evidence is substantially stronger; current–phi/SAT,
evolution, conservation, production promotion and Q08 span/support/stiffness
selection remain separate. No full Q07 closure is declared by this audit.

### Static material qualification accepted — 1 October 2026

The user explicitly accepts the C3 h/32 five-field centered and characteristic-
corrected material operator as **qualified**, on the complete global evidence
and subsequent fixed-location velocity audit. Freeze campaign commit `bc6f5c81`,
identity `ab607bf62cda21dd67935334da9fdea35c2ad0a0c529ff555006aa3a8ed83bc3`,
and the current numerical choices as the accepted static material baseline.
Retain measured orders as measured: the geometry/ripple-band sampling diagnosis
supports the scoped acceptance; it does not replace approximately first-order
native-grid velocity intervals with a claimed global second-order result.
Short-wave reconstruction errors, regional maxima, approximate conservation,
non-adjointness and unexercised exterior caps remain explicit limitations.

Next Q07 work is a bounded current–phi/SAT compatibility audit using real HSX
and complete incoming support. Expose the existing homogeneous maps and affine
boundary lifts separately; audit physical masses, current support, one-time
endpoint lift, raw B²/n weighting before owner projection, and matched Ti/psi
force cancellation. Keep the reference pair and opt-in live-adjoint prototype
separate; diagnose compatibility with the newly accepted material force rather
than assuming the generic G/D pair is adjoint. Restore matching prior captures
or create an explicitly scoped fresh actual-HSX capture if they remain missing.
Do not introduce an elliptic solve or reopen the failed Rung-3 repair merely to
obtain an algebraic identity. Report any design decision needed before changing
the accepted force/action.

Before Q08 coupled-RHS consolidation, replay remaining vorticity/diffusion
blocks consistently on C3; their prior evidence is retained but this material-
only campaign does not qualify their magnetic-background transfer. Full Q07
closure follows the remaining integration checks. Q08 retains shared
implementation consolidation, span/eta-support selection, and time/memory/
timestep-feasibility audits; Q09 retains evolved qualification and production
promotion. No full Q07 or production closure is implied by this scoped decision.


### Q07 bounded current/phi assembly and C3 transfer — 1 October 2026

[Report and retained results](../../../../work/q07_current_phi_audit_20261001/report.md).
The research-only `native/q_parallel_current_phi.py` accepts explicit raw
homogeneous current divergence, affine lift, phi/Ti gradients, density and B.
It exposes one-time current lift, raw B²/n multiplication, and matched Ti/psi
force terms before complete-owner projection. It does not select a production
pair, infer current BC from density, or replace the accepted material action.

The audit reuses checked C3 RK4-64 traces and frozen donor choices at 21 complete
HSX owners, all 22 fields and four uniform/mixed BC combinations: 12,012
current/potential/vorticity records. Force replay differs by at most 3.411e-13,
Ti cancellation by 5.685e-14, and current normal-data affine decomposition by
4.828e-15. Current/phi raw-input JVP and mass identities pass. Twenty-three
distinct focused tests passed, including the existing density-payload isolation
regression; this is not a fresh actual-HSX replay of the missing old pair NPZs.

New-map sparse work probes include all divergence donors and all raw members
of every nonzero-current owner. The adjoint-control work residual is below
2.9e-20; the traced map has a nonzero interior as well as wall defect. These
impulse/windowed/random probes confirm accepted Q06 non-adjointness, not a new
MMS failure or a percentage of physical conservation loss. The constructed
adjoint control is not the old production pair. Do not import its identity into
the accepted force or enforce a new transpose just to make this diagnostic zero.

The bounded C3 transfer of scalar vorticity advection and both-span cap-gradient
diffusion is complete. All six constant-channel scalings replay exactly over
11,088 channel records; 42 preparations retain the frozen repair choices.
Smooth sampled reconstruction discrepancies decrease, while sampled O-R/N-R
rebounds and wave controls remain. These are bounded transfer results, not a
new global diffusion qualification or a change to its span default.

**Remaining Q07 integration:** explicitly map the canonical resolved physical
sheath-current state to the new current boundary/lift interface, keeping the
accepted traced force and its non-adjoint contract explicit. Prescribed-normal
MMS lifts are not physical-endpoint SAT lifts. Current short-leg traces have no
wall crossings, and the old actual-HSX pairing captures remain missing, so
neither physical wall-power closure nor old-pair replay is claimed. Freeze the
boundary adapter and audit its actual current/force response before promoting
the complete RHS. No new elliptic solve, production default, or Q07 closure.

### Q07 option-1 physical wall-current test — 1 October 2026

[Bounded report](../../../../work/q07_option1_wall_current_20261001/report.md).
48 actual-HSX C3 wall patches at N32/N48/N64, two wall rings, both magnetic
orientations and shallow incidence; 28 conducting-sheath/no-flow trial states.
All use existing short traces, donor support and wall fits. Query-local physical
wall targets reproduce the canonical physical-boundary-state current exactly;
particle-current and live voltage/density-response checks pass. Four focused
existing wall tests pass. The selected production interface exports nonlinear
current from the complete physical target, unlike its older modal selectors.
An independent current reconstruction or extra current lift is not required
merely to obtain the correct wall current for this tested contract.

Interpolating nodal targets first is not equivalent: branch mixing gives normal
current discrepancies up to 0.0695 and wrong-sign ion-flux query instances.
Retain continuous query-local magnetic orientation and wall-law evaluation.
The wall-normal Neumann fit is exact at its nodes, not at every intervening
point; short-wave off-node derivative residuals remain explicitly reported.

The unchanged nonconstant Dirichlet lift has a separate interior-extension
limitation. Its unattenuated `T(query)-T_interpolated(query)` component can
carry a sheath branch discontinuity into interior caps, even though the short
traces do not hit the wall. Six projected-cap stencils change branch. At N64
cap speeds reach 5.482 for donor Vi=2, while a diagnostic lift using the existing
wall radial basis times the query target reduces the branch-case maximum to
2.007 and preserves wall values. The exact old/new difference is
`(1-ell_wall)*(T(query)-T_interpolated(query))`, verified to 3.24e-15. One
raw current-divergence action changes from -293.654 to -0.2163; these trial
states have no matched continuum current reference, so this is sensitivity
evidence, not a claim that the smaller action is the physical solution.

Continue option 1 by bounding this primitive Dirichlet-lift modification on
smooth previously qualified controls and branch-changing patches, with cap
value/gradient accuracy and voltage response. Do not add a current SAT to
compensate for the primitive continuation issue. The radial lift remains a
research diagnostic; coupled MPE normal data, physical wall-work balance,
evolution, and full Q07 closure are still open. No package/default change.

### Q07 paired radial wall-lift comparison — 1 October 2026

[Report](../../../../work/q07_radial_lift_test_20261001/report.md), with validated
paired actions and sample reductions. Tested the proposed `If + ell*T_query`
on the same 48 actual-HSX C3 patches, all 22 frozen manufactured states and four
material BC combinations. Reused RK4-64 traces; no source/default change.
The wall-specific five-point eta interpolation is quartic Lagrange, not cubic;
this corrects the preceding conversational description without changing degree.

Exact wall values, donor nodes, representable tensor polynomials, constants,
frozen old-adapter replay and independent value/gradient checks pass. Physical
sheath fixed-branch gradients, conductor response and gauge checks pass; prior
branch-cap overshoot reduction is retained. Neumann material actions are exactly
unchanged. These are not exterior-crossing or time-stability tests.

**Do not promote the radial lift as a blanket replacement.** On wave controls,
pooled sampled Vi/Ve traced-G N-O RMS increases from 0.03587/0.006615/0.002202
to 0.13247/0.03289/0.01735 at N32/N48/N64. Smooth absolute errors remain small
but also increase. The material comparison changes only D Vi/Ve; at N64 the
all-D combined wave N-O RMS for n/Te/Ti changes from
0.002843/0.002419/0.001305 to 0.02158/0.02108/0.01263. Electron-force scale
hides these regressions in a pooled five-field norm. Candidate pooled errors
still decrease with resolution; these are bounded sample statistics, not global
norms or a global disqualification. O-R is identical between the two lifts.

With `Delta=T_query-T_interp`, the gradient change contains
`grad(ell)*Delta + (ell-1)*grad(Delta)`. The radial derivative amplifies tangential
interpolation error by an inverse grid spacing. The old residual lift exactly
reproduces arbitrary radially constant targets; the candidate loses this
property, as confirmed by explicit nonpolynomial controls. The new construction
therefore resolves one interior-extension symptom at a measurable accuracy cost.

Retain the qualified smooth-data lift. Next investigate a branch-consistent
physical sheath extension preserving smooth residual reproduction, with policy
based on geometry/wall physics rather than error-driven switching. A compatible
physical-boundary accuracy/evolution test is needed to choose the sheath adapter;
lower D(j) on arbitrary incompatible trial states is not proof of accuracy.
Coupled MPE, wall work, evolution and full Q07 integration remain open. No extra
current SAT, package numeric change, commit or production promotion follows.

### Q07 same-branch support and split-lift comparison — 1 October 2026

[Report](../../../../work/q07_branch_extensions_20261001/report.md) and validated
artifacts cover the same 48 actual-HSX patches, 22 manufactured fields/states,
28 physical sheath/no-flow trial states, and shared-query neighboring-patch
checks. Existing C3 traces and field degree retained; no package numerics or
production default changed.

Option 1 used the nearest up to 48 same-sign wall nodes from a fixed 11x9
geometry pool in the existing 35-dimensional tangential space, consistently
for the four interior radial planes and wall residual. Thirty of 96 branch
supports fail rank/conditioning; further physical-target extrapolation failures
leave only 498/1344 physical scenario-patch cases valid. On common valid wave
subsets, sampled traced-G RMS worsens (N64 0.002627 -> 0.08859). Reject this
specific support policy; no degree reduction or fallback was silently applied.

Option 2 uses `I(f-S_nodes)+S_query+ell*(T_query-S_query)`: S continues one
sheath sign, chosen at the source wall projection and held fixed across the
stencil; T retains the actual physical query sign. It exactly replays the old
smooth prescribed-data lift (S=T) and preserves radially constant nonpolynomial
reproduction. All 1344 physical cases are valid. In the N64 projected-branch
subset, max cap speed drops 5.482 -> 2.007 and max abs D(j) 293.65 -> 4.636.
Wall values, no-flow, fixed-branch derivatives and conductor response pass.
Arbitrary sheath trial states have no compatible continuum reference, so the
smaller D(j) is not an accuracy measurement. This is the leading research
candidate, not a production or global qualification.

Neighboring reconstructions still differ. With common positive smooth wall
plasma inputs and actual HSX geometry, max sampled value mismatch is
0.4017/0.03638/0.005088 for option 2 versus about 0.993/1.018/1.024 for the old
lift at N32/N48/N64. A shared continuation-sign diagnostic does not uniformly
remove it; differences between smooth local interpolants also matter. The
separate live reconstructed-input seam test has explicit extrapolation
exclusions, retained in the report. No global orders are inferred from these
sampled mismatches. The ion max/clamp remains piecewise smooth even after
removing the magnetic sign jump from the continuation.

Next: a bounded physical-boundary-compatible reference/state family for the
split lift, including grazing neighborhoods, neighboring-patch errors and
current/force response; then short evolved wall-response checks before
promotion. Off-node normal data, coupled MPE, physical current/phi wall work,
exterior hits, evolution and full Q07 closure remain open. No additional current
SAT is inferred solely from primitive boundary-extension sensitivity.

### Q07 split lift with compatible sheath fields — 1 October 2026

[Report](../../../../work/q07_split_compatible_20261001/report.md), with 17
verified artifact hashes. Tested option 2 on the same 48 actual-HSX C3 wall
patches and saved RK4-64 h/32 caps: 54 sheath states plus two no-flow controls,
16,128 paired action records. No tracing, donor/degree changes or production
selection. A fixed-resolution-independent continuation uses
`s = bn/sqrt(bn^2 + gamma^2*(1-u)^2)` in the signed sheath velocities, with
smooth residuals. It satisfies the non-grazing velocity wall law; constant
thermodynamics also satisfy zero physical-normal data. The varying-thermodynamic
control uses prescribed nonzero normal data, not the complete simple zero-normal
thermal sheath model. This is one admissible manufactured interior extension,
not a prediction of the physical solution. Exact wall grazing has no continuous
limit; no classical global smooth-MMS order is claimed there.

Wall values and the package sheath resolver agree to roundoff. Checked reference
gradients, current/phi coefficient assembly and Ti-force cancellation pass.
Nevertheless exact-target sampled current-divergence N-O RMS is
3.836/5.367/7.895 at N32/N48/N64, versus O-R 0.215/0.610/0.297. Option 2 reduces
the old N64 N-O of 39.889 substantially, but its remaining error is comparable
to continuum activity (R RMS 7.58). Live target production adds further error;
its provisional Vi extrapolation is itself unqualified. These equal-weight
bounded statistics are not global volume norms or convergence orders.

The scored wall rings approach grazing as N increases, so reference activity
also grows. A separate fixed-physical-query test removes that confound and
still finds nonuniform error reduction. Supplying exact transverse data on the
same four radial planes reduces N64 value RMS 0.1680 -> 0.001249 and gradient
RMS 44.90 -> 11.51. Transverse reconstruction is the main limitation, with
remaining radial derivative error near grazing. A local transition-width
indicator finds median eta-spacing/width ratios 7.42/4.44/3.92 near grazing;
the theta medians are below one. This points to inadequate eta information for
this reference family, not the earlier O-R coil-ripple mechanism or proof of
an evaluator bug. No-flow current N-O RMS decreases
6.51e-7 -> 8.41e-8 -> 2.10e-8.

Keep option 2 as a research candidate and preserve the accepted smooth-data
operators. Next: bounded anisotropic eta sampling at fixed physical queries on
the same continuous HSX geometry, with exact-plane controls, to separate missing
resolution from support/degree limitations. Do not infer a universal physical
grazing width from this manufactured continuation or silently smooth the sheath
law. Defer evolution and global promotion until compatible-field accuracy is
supported. Off-node normal enforcement, coupled MPE, physical wall work,
exterior crossings and full Q07 closure remain open.

### Q07 option-2 anisotropic sampling and grazing limits — 1 October 2026

[Report](../../../../work/q07_split_eta_sampling_20261001/report.md). Same actual
HSX C3 geometry and 56 compatible reference cases: fixed physical queries at
N32/N48/N64 and both saved N64 wall-ring tube stencils. 35,840 records cover
eta-spacing factors 1/2/4/8/16/32, independent theta refinements, exact-plane
controls and an eta phase check. Four radial layers, seven theta nodes, five
eta nodes, degree, continuation sign and physical traced endpoints stay fixed.
Additional observations are exact manufactured samples, not new information
recoverable by interpolating current owners. No tracing or production change.

Exact-density current N-O RMS at N64 decreases
7.895 -> 6.904 -> 5.087 -> 3.411 -> 1.665 -> 0.559 as eta spacing decreases;
max decreases 55.54 -> 6.20, while common O-R stays 0.297. Exact eta gives
0.464 and exact theta+eta 0.0554. Fixed N64 value RMS decreases 0.970 -> 0.0959.
This confirms a substantial transverse-information deficit, separately from
O-R. Constant-thermodynamic zero-normal controls show the same mechanism.
No-flow remains accurate. These are bounded sample statistics, not global
orders or a prescription for 32-fold production eta refinement.

Residuals matter: theta factor 2 at eta32 increases current RMS to 1.239,
principally at fixed_2; factor 4 reduces it to 0.516. Exact transverse data
still leave fixed-query gradient RMS about 10.1. The half-step eta phase puts
inner caps on fine donor planes and nearly matches the exact-eta result;
retain this as alignment sensitivity, not a selected support policy.

**Separate finite-grid extension defect established:** in a single fixed-sign
patch, `I(f-S_nodes)+S_query` is continuous, so option 2 has the one-sided jump
`ell(u)*(T_plus-T_minus)` across a projected grazing line. Actual HSX roots
at all eight anchors confirm this. At N64 u=.97/.98, limiting ion-velocity
jump magnitudes are 0.23556/0.20031, while the exact smooth interior field
changes by less than 9e-7 across eta separation 2e-9. At radial donor centers
ell=0 and the value jump vanishes. Measured-minus-predicted jump is at most
1.35e-6, consistent with the finite separation. Angular refinement cannot
remove this term at fixed radial grid, even though it removes much of N-O.

Replay, independent tensor assembly, constants and derivative checks pass;
main runtime 213 s, peak RSS 0.913 GiB. An auxiliary support check initially
rejected a permutation of identical theta nodes; it was corrected to compare
node sets, retaining the original failure and unchanged tolerance. Scientific
actions and selection policy were not tuned.

Next compare continuous interior extensions of the physical wall data,
requiring smooth-data replay, one-sided interior continuity, current N-O and
explicit wall-value/trace checks. Collocated wall constraints and a continuous
distance-dependent extension have different off-node wall contracts; disclose
that distinction rather than silently smoothing the physical sheath law or
selecting a manufactured-field-dependent width. Do not promote the current
split lift on eta refinement alone. Previously accepted smooth prescribed-D/N
operators remain unchanged; live target production, normal enforcement, wall
work, MPE, exterior hits, evolution and full Q07 remain open.


### Q07 wall observations and GBS-style smoothing — 1 October 2026

The [paired bounded comparison](../../../../work/q07_wall_observations_20261001/report.md)
tests the split lift against `I(f-S_nodes)+S_query+W*(T_nodes-S_nodes)`, with
hard sheath sign and fixed 0.5°/1°/2° tanh incidence widths. Same actual C3 HSX
geometry, saved RK4-64 h/32 caps, 140 donors/35 wall constraints, 56 compatible
cases, eight anchors, both wall rings, N32/48/64 and fixed physical queries.
Each smoothed wall law has its own matching interior reference. Norms are
bounded sample RMS, not global convergence orders. Current uses exact density
to isolate velocity reconstruction; live-target sensitivity is separate.

The new construction removes the fixed-patch interior jump: the hard split
limit is 0.238148, whereas observation differences shrink proportionally to
point separation (6.94e-8 at eta separation 2e-9). All smoothed split versions
also remove the jump and retain exact arbitrary-query wall targets. Observation
wall-node and donor checks pass, but off-node wall values are inaccurate:
N64 hard wall RMS 1.222, maximum 4.446; 2° RMS 0.667, maximum 3.089. Wrong normal
parallel-ion flow is substantial at some queries (e.g. -0.0793 instead of the
prescribed +0.0590). These are query samples, not wall-area fractions.

At N64 current N-O is 7.895 -> 7.884 for hard split -> observations, almost
unchanged. Observation N-O for widths 0.5°/1°/2° is 5.436/4.032/2.896; O-R is
0.02588/0.01015/0.003568, versus 0.29739 for hard sign. Lower absolute errors
also accompany a smaller reference action; relative N-R remains above 100%
in this demanding matched sample. The hard observation N-O trend is
3.831/5.365/7.884; only the broadest smoothing improves monotonically over both
intervals (4.424/3.249/2.896). Near-grazing angular underresolution remains;
the plotted actual wall target changes sign twice within one eta-plane spacing.
No global qualification is warranted by this candidate.

[GBS section 2.4](https://arxiv.org/abs/2112.03573) motivates velocity smoothing
near tangency; this audit's tanh function/widths are explicit research choices,
not GBS's exact function or full coupled MPE derivative treatment. On the
64x64 HSX wall, widths 0.5°/1°/2° change velocity by >10% over about
21%/42%/83% of physical area; constant-state projected particle-flux reductions
are 1.46%/5.83%/19.14%. These are geometry diagnostics, not evolved losses.
Do not transfer GBS's small reported affected-area fraction to this HSX wall
or select smoothing solely for smaller manufactured errors.

All 32,256 primary records and expected wall/live/seam/jump coverage validate.
Main runtime 125 seconds, peak 0.918 GiB; independent tensor, derivative,
constant, smooth replay and conductor-response checks pass. One ancillary
analytic conductor check was corrected to use projected-wall temperature;
original failure is preserved, with main actions and tolerances unchanged.

Retain exact arbitrary-query wall enforcement as the next design requirement.
If smoothing is physically chosen, use the smoothed split lift as the next
bounded baseline and assess resolution together with wall-flux impact. If the
hard law is retained, investigate a continuous geometry-aware interior
extension with correct wall limit and independent compatible radial profiles.
Do not promote wall observations at this support or launch another global
campaign. Accepted smooth D/physical-normal N operators are unchanged; live
wall-law production, coupled MPE, physical wall work and Q07 integration remain
open.


### Q07 bounded eta refinement with both boundary lifts — 1 October 2026

[Report and plots](../../../../work/q07_eta_refinement_20261001/report.md).
Nr=Ntheta=64, Neta=64/128/256/512, two eta phases, all 56 compatible fields,
eight actual HSX anchors, both outer rings, both split/observation candidates,
hard and fixed 0.5°/1°/2° tanh targets. Three distinct arms retain old physical
caps, shorten caps at the same seeds, or move seeds onto refined grid planes.
All finer donor observations are genuine exact manufactured samples. These are
bounded sample norms, not global orders or a full anisotropic simulation.

Eta information clearly helps both candidates. With 1° smoothing and fixed
traces, observation current N-O is 4.032/1.536/0.518/0.145; split gives
4.032/2.180/0.562/0.150. Common O-R stays 0.01015. Observation off-node wall
RMS improves 0.842/0.455/0.103/0.0453; half-phase 512 gives 0.0225. Primary
512 reversed parallel-ion contributions have maximum magnitude 2.03e-6,
versus 0.0713 at 64. The previous severe smooth-target wall-observation failure
was therefore substantially underresolution, not an intrinsic rejection of
collocated wall data. Hard-sign wall error remains large (512 RMS 0.78–0.86),
and the hard split interior jump remains 0.238 despite eta refinement.

Do not select the most favorable alignment: 1° fixed-trace N-O at 512 changes
0.145 -> 0.305 with half-plane phase. Actual refined-grid h/32 seeds give
N-O 0.623 / 0.309 across phases, about 16% / 8.5% relative N-R. Constructions
are effectively identical there. Shortening at fixed seeds lowers O-R from
0.01015 to 0.0001586 but barely changes N-O (0.145 -> 0.147 primary phase).
The dominant gain is finer donor sampling, not a shorter trace: the donor
count stays fixed while its angular spacing and physical support shrink.

At 1°/512 primary phase, exact eta yields N-O 0.1425, exact theta 0.05829,
both exact 0.03228. At half-phase the corresponding finite/exact-eta/exact-theta
values are 0.3048/0.1425/0.2788. Remaining accuracy depends jointly on eta
alignment and poloidal/radial reconstruction; do not assert a universal minimum
eta resolution from this manufactured field set. Lower-mode and constant-thermo
controls show the same issue. A bounded joint eta/poloidal comparison is more
informative next than a global campaign or blanket eta-only refinement.

Retain the smoothed split lift as primary research baseline for any chosen
smooth wall law: exact arbitrary-query wall target, similar refined-grid cap
accuracy. Keep wall observations as a comparison, with explicit off-node error.
No smoothing width is selected or production change made; broad smoothing's
physical wall-flux change is not removed by numerical refinement. Previously
qualified smooth D/N operators remain unchanged. Exact-target velocity audits
do not close live wall production, coupled MPE or current-phi/SAT work.

200,704 primary / 28,672 wall / 7,168 seam / 8,192 grazing records validate;
86,016 directional controls independently replay tensor values/gradients within
2.91e-14 / 4.95e-12. 768 fresh RK4-64 trajectories have no crossings/reentries,
N64 saved endpoint replay 8.88e-16. Main action replay 3.28e-11, derivative FD
2.05e-8; all original gates pass. Main runtime 505 seconds, peak 1.37 GiB;
directional controls about 184 seconds. No gate relaxation, support tuning,
commit, push or production promotion.

### Q07 joint eta/poloidal refinement — 1 October 2026

[Report and full tables](../../../../work/q07_joint_angular_20261001/report.md).
Nr=64 fixed; Ntheta=128/256 crossed with Neta=128/256/512, both eta phases,
additional theta-half-phase controls, and validated reuse of Ntheta=64 results.
Both lifts, all 56 compatible fields, eight HSX anchors, two wall rings and
hard/0.5°/1°/2° targets. Every fit retains 140 interior donors and 35 wall
nodes with unchanged degree; refinement changes spacing, not donor count.
These remain bounded equal-sample norms, not global orders.

At 1° and fixed N64 physical traces, Neta=512 observation N-O for
Ntheta=64/128/256 is 0.14462/0.05841/0.05827 at primary eta phase and
0.30483/0.27818/0.27882 at half phase. Finite theta refinement reaches the
previous exact-theta controls (0.05829/0.27884), identifying diminishing
returns beyond Ntheta=128 for these samples. Prior exact-both-angular
N-O=0.03228 shows a possible later radial limitation; error norms are not
additive and these fixed-cap controls cannot diagnose every moved grid seed.

The actual refined-grid (256,512) seeds still give 1° N-R RMS 0.241–0.650,
or 6.7–17.1% relative, across phases. O-R is only 0.00015–0.00025; N-O
dominates. Both constructions are essentially equal there. Primary maximum
N-O=4.591 is at fixed_3, ring 63, case 47; that anchor supplies 92.5% of the
sample squared error (75% at half eta phase). Both wall rings and low-mode/
constant-thermodynamic fields retain error. This is reconstruction of the
sharp grazing continuation, not evidence for the old coil-ripple O-R account.

At the finest pair, observation wall RMS for 1° is 0.0087–0.0367 and the
maximum inward parallel-ion contribution is 2.01e-6; split enforces arbitrary
wall targets to roundoff. Hard-law observation wall RMS remains 0.65–0.73.
Theta velocity seams are about 1e-6–3e-6, while eta seams remain 0.029–0.046.
Keep smoothed split as the primary research baseline for a chosen smooth law,
observations as a comparison; do not choose smoothing or a universal minimum
resolution from these manufactured errors. Accepted smooth D/N gates stay
unchanged; no physical sheath or full Q07 qualification is added.

Next bounded discriminator: exact-eta and exact-both-angular controls at the
actual refined-grid cap locations (fixed_3 and shallow anchors), both phases,
before further eta/radial refinement or a global campaign. The current test
does not establish whether the moved-grid residual is entirely eta error.
Live wall-state production, coupled MPE, current-phi/SAT, wall work and evolution
remain open. 322560 action / 64512 wall / 32256 seam records validate; 1152
RK4-64 trajectories have no crossings/reentries. Main runtime 24.4 minutes,
peak 0.920 GiB; replay/constraint/derivative gates pass unchanged. No source
promotion, commits or pushes.

### Q07 exact-angular audit at actual refined-grid caps — 1 October 2026

[Report, tables and figure](../../../../work/q07_grid_angular_oracle_20261001/report.md).
All eight actual HSX anchors, two outer rings, 56 compatible fields and both
lifts at Nr64/Ntheta128/Neta512, both eta phases. All hard/0.5°/1°/2° targets
are retained. Saved endpoints, span, radial rows and geometry are unchanged;
finite/exact-eta/exact-theta/exact-both-angular information is compared.

For the 1° split lift, N-O 0.66122/0.21026 becomes 0.04501/0.05448 with
exact eta, 0.66007/0.20016 with exact theta, and 0.03727/0.03076 with both.
Common O-R is only 0.000247/0.000140. Exact eta removes 74–93% of the RMS;
relative N-R falls from 17.72%/5.98% to 1.20%/1.55%. The exact-both result
is 1.00%/0.87% relative. This isolates the leading eta reconstruction error
at the actual moved-grid hotspot, not merely the old fixed physical caps.

Primary fixed_3 N-O RMS falls 1.7965 -> 0.08434, maximum 4.394 -> 0.2854.
Both rings and low-mode/constant-thermo controls improve. Center directional
error terms have RMS radial/poloidal/eta 0.0373/0.0274/0.6664 at primary
phase and 0.0308/0.0474/0.2009 at half phase. Exact eta removes its derivative
error while leaving the other terms unchanged. Center-error and finite-cap
N-O RMS differ by only 0.000250/0.000053, so shortening the trace cannot
remove the leading reconstruction derivative error. Correlated RMS terms are
not additive error fractions.

At these grid/donor centers, both exact angular directions remove their
derivative errors to roundoff; the remaining radial derivative error agrees
with independent one-dimensional radial interpolation. Both lifts give the
same direct-center mechanism. Keep smoothed split as the research baseline:
the evidence concerns representing the sharp interior grazing transition,
not incorrect enforcement of its prescribed wall value. Hard-target defects
and live physical-wall integration remain open; accepted smooth D/N gates stand.

The proposed 512/1024/2048-plane diagnostic refinement was superseded by the
user's production-resolution constraint on 2 October. The next authorized audit
uses Neta32/48/64, phase controls and fixed donor counts/degrees, comparing
smoothed-wall reconstruction accuracy with its physical wall-flux impact.
Smoothing is part of the candidate boundary model; increasing resolution to
hundreds/thousands of planes is not the practical acceptance route. No physical
smoothing width is selected by the exact-angular audit. 57344 records
validate, baseline action replay 1.10e-11, radial-only gradient replay 1.27e-12,
gradient FD 1.26e-8. Main runtime about 134 seconds, peak 0.915 GiB. A JSON
longdouble serialization fix is preserved and its first-anchor arrays replay
bitwise; no numerical formula or tolerance changed, no production promotion.


### Q07 production-resolution smoothing feasibility — 2 October 2026

[Report, tables and figure](../../../../work/q07_production_feasibility_20261002/report.md).
The magnetic evaluator was committed as opt-in `8dadf696` (27 focused tests);
the user explicitly retained the old package default. This audit explicitly
uses C3. Nr64, Ntheta64/Neta32/48/64 plus Ntheta128/Neta64 control, two eta
phases, eight actual HSX anchors, both outer rings, all 56 states, hard and
0.5/1/2/4 degree targets, both constructions; fixed 140/35 donor/wall counts
and unchanged degree. These are bounded sample norms, not global orders.

Smoothing alone does not produce an accurate production-resolution candidate
for this prescribed compatible sheath interior continuation. At Neta64,
1 degree split N-O is 4.032/4.228, O-R 0.0102/0.0652, and N-R relative RMS
111.3%/99.2%. At 4 degrees N-O falls to 1.832/2.953 but N-R is still
123.5%/96.1%. The reduction in absolute error accompanies a weaker target
variation. Both rings, thermodynamic families and low/wave modes retain large
errors; Ntheta128 does not rescue them. Smooth zero-wall-flow controls have
N-O 0.00271% and N-R 0.1266% relative. This is predominantly reconstruction
of the sheath angular/interior variation, not the old O-R account.

Split wall values remain exact to 2.89e-15 with no inward contribution.
Observation-only has comparable cap errors but off-node wall RMS 0.477/0.581
and inward maxima 0.0275/0.0163 even at 4 degrees/Neta64. Keep smoothed split
as the research baseline. Sampled model flux reductions at 1/2/4 degrees are
6.7%/20.2%/42.4% on a 128x128 wall-area grid (64x64: 5.8%/19.1%/41.6%).
No smoothing width is selected. The manufactured interior continuation is not
uniquely imposed by the wall law: this does not prove evolved physical error
or revoke accepted smooth D/N gates. Next investigate the interior representation
of known sheath angular dependence at affordable resolution, using held-out
continuations before degree/support changes; no new experiment is launched.

All 71680 action, 71680 wall and 17920 seam records validate. 512 new C3
RK4-64 trajectories have no crossings/reentries; saved trace replay 4.44e-16.
Main runtime 247 seconds, peak 0.951 GiB. Original pilot/repair is preserved:
a zero-wall-flow state with a nonzero interior residual is a scientific control,
not a roundoff gate; replace that invalid check with an exactly zero field.
Squared-gradient-norm replay uses normalized error; direct action gate remains
1e-8 and observed N/O/R replay is <=2.06e-10. Numerical arithmetic unchanged.
Physical sheath, live wall producer, MPE/current-phi/SAT and evolution stay open.


### Q07 manufactured-continuation audit — 2 October 2026

[Report and full reductions](../../../../work/q07_continuation_audit_20261002/report.md).
Nine profiles compare original gamma .5/1/2, a constant-radial branch,
exponential attenuation, alternative broadening, and smooth-bulk blending with
fixed logical-u scales .02/.08. Same 1°/2° wall targets, 18 matched states plus
two zero-wall-flow controls, eight actual C3 HSX anchors, both outer rings and
both original eta/site phases. Nr64/Ntheta128. Saved Neta64 physical caps and
h/32 span are identical across profiles and donor Neta64/512; no retracing,
changed support, degree or production policy. Bounded equal-sample norms only.

Manufactured wall compatibility is 2.22e-16, split wall error 1.64e-14,
reference derivative FD <=9.97e-8, candidate FD 2.73e-9. Prior N/O/R replay
is <=1.50e-12; O/R exact. Thus no incompatible target or demonstrated analytic
reference bug explains this failure. At 1°/Neta64, original gamma=1 N-O=4.689,
N-R/RMS(R)=104%; column=5.076/107%; exp L=.02=2.669/85.9%; broaden L=.02=
4.852/104%. All profiles span 85.9–107.2% at 1° and 79.6–103.0% at 2°.
Ion velocity relative RMS is ~11%, parallel-gradient error ~89–105% (1°),
so current subtraction alone is not responsible. Both rings, phases, low/wave
modes and constant thermodynamics retain the issue. Smooth no-flow controls
retain N-O 2.11e-8 and N-R about 0.0623% relative.

Exact eta reduces the 1° N-O RMS by 98.7–99.4% across every profile; the
original 4.689 becomes .0334 and column 5.076 becomes .0527. Exact both angular
directions reduce column to 1.92e-6 but leave profile-dependent radial error:
original gamma=1 .00454, gamma=2 .0515, exp/broaden L=.02 about .0092/.0080.
Actual 512-donor refinement improves all cases but retains 16.0–25.5% relative
N-R at 1° and 12.1–15.7% at 2°. These fixed-coarse-cap results are not the
prior native-Neta512 sample locations or spans; O/R remain bitwise unchanged.

The original continuation is not the unique cause; nearby interior profiles
sharing the rapid wall angular dependence are difficult for the current sparse
polynomial representation. This does not establish a self-consistent physical
presheath or exhaust all possible continuations. Keep the physical question of
applying the presheath law over the whole shallow-incidence wall separate/open.
Next design a known-angular-factor representation with smooth amplitudes and
flexible radial dependence, retaining these profiles as held-out controls;
do not insert exact manufactured gamma/L or tune a profile to obtain a pass.
No such candidate is implemented or promoted here; all accepted smooth D/N
and static material gates, live wall/current-phi/SAT and evolution status stand.

69,120 records validate, about 295 s and 0.67 GiB RSS. Independent retained-value
action replay 1.33e-14, RMS reduction 8.88e-16. The pilot JSON longdouble issue
was fixed only in serialization; all 1744 original pilot arrays replay bitwise.
Full source, checkpoint, coverage and completion receipts are retained.

### Q07 prescribed-boundary six-field assembly — 2 October 2026

The user retains the general reconstruction and moves on from basis/enrichment
research. The [bounded assembly audit](../../../../work/q07_six_field_assembly_20261002/report.md)
provides one research owner-state entry point for n/Te/Ti/Vi/Ve/omega, prescribed
phi, centered/characteristic material action, scalar vorticity advection,
slot-product current drive and the six existing constant diffusion channels.
No production default, collision/source model, physical sheath law or SAT
endpoint map is introduced.

All 21 complete C3 HSX owners at N32/N48/N64, 22 states, four uniform/mixed
primitive D/physical-normal N combinations, and both diffusion spans pass.
103488 term/field records validate in 77 s, peak process RSS 2.94 GiB; 37
focused tests pass. Component replay <=4.36e-11, prior current/phi replay
2.20e-11, prior diffusion replay 5.39e-13, retained-record replay exact and
independent component-sum reduction 3.41e-13. Constant N-O <=1.59e-9. JIT,
live centered+diffusion state/BC JVP and full phi/coefficient response pass;
the inherited frozen characteristic spectral-projector AD scope remains.

Preparation keeps the qualified geometry-consistent material/current center
weight separate from the unchanged diffusion weights. The diagnostic current
boundary response includes primitive product cross terms, so it cannot be
replaced by a boundary-only product or a density-derived current BC. B²/n is
applied before projection and Ti/psi compensation appears once.

For h/32 diffusion, bounded smooth N-O RMS decreases across all three grids
for all six assembled equations. N64 smooth N-R/RMS(R) is 0.023%, 0.082%,
0.022%, 0.042%, 0.218%, 0.004% for n/Te/Ti/Vi/Ve/omega respectively. Native
N-R rebounds and short-wave reconstruction limits persist; these are equal-
weight sampled norms, not global convergence orders. Keep the separate terms
and BC/region reductions visible rather than selecting by total cancellation.

The next work is Q08 engineering consolidation and memory/runtime feasibility,
followed by remaining C3 static term coverage using the existing maps. Five-
field C3 material global evidence is accepted; current/phi, vorticity and C3
diffusion transfer remain bounded evidence, so this audit does not close all
Q07 gates. Physical grazing-sheath validity/resolution, live wall/SAT and
exterior endpoints remain a separate open workstream. Traced Q04 evolution,
Q08 span/eta-plane selection, Q09 coupled evolution and production are pending.

### Q08 shared extraction implementation — 2 October 2026

The [implementation review](../../../../work/q08_implementation_20261002/report.md)
records the bounded implementation by three GPT-6.1 Sol agents and independent
parent replay/review. Common primitives, paired preparation, exact query tables,
compact QBank artifacts, the array-data QPlan and shared owner-plane utilities
are implemented. The six-field action reuses five scalar positions and one
batched donor gather; homogeneous current no longer requires a second zero-BC
reconstruction. Material/current balancing and frozen diffusion stay separate.
Legacy Q/P interfaces and artifacts remain supported.

All 31 prepared arrays replay bitwise against the pre-extraction source for
seven complete C3 owners at each N32/N48/N64, including core, repaired inner,
transition, bulk and both wall rings. With 22 states, four uniform/mixed D/N
patterns and both diffusion spans, every eager six-field output/diagnostic
matches the legacy action exactly; persisted accepted component actions also
match exactly. JIT remains within the predeclared component replay gate.
State/BC/coefficient AD retains the frozen characteristic projector contract.
These are implementation replays, not new N-O/O-R/N-R orders.

Paired setup took approximately half the old two-call time on these chunks.
The staged plan uses 0.360/0.551/0.670 MB versus the old runtime's
0.999/1.563/1.911 MB. Device-resident one-state CPU timings were essentially
unchanged. Exact outer tensor factors are captured and decode bitwise, but
direct host factor application was slower, so dense runtime remains selected.
This does not establish full-grid compression, GPU residency or speedup.

Q localization measures actual nonzero donor reach (-2,2), checks full owner
closure and frozen topology content, and retains Q halo2 independently of P's
halo3. Detailed CPU sharding coverage and validation receipts are in the report;
bounded C3 owners occupy one eta plane per grid and do not establish full-domain
HSX distributed coverage. Review added rejection of malformed padding,
fractional identities and stale raw/topology receipts.

The next gate is a full-grid representation/resource pilot and replay using
saved C3 data, followed by actual GPU measurements. Full Q08 is not closed:
remaining term-resolved global static coverage, span/eta-plane selection,
physical wall/SAT/exterior behavior and Q09 evolution retain their scope.
No production default, numerical policy, commit or push changed in this step.

### Q08 full-grid extraction/GPU replay campaign — 2 October 2026

The authorized implementation gate is packaged in
[`scripts/q08_extraction_global`](../../../../DRBX/scripts/q08_extraction_global/README.md).
It reuses the complete saved compact-C3 RK4-64 endpoints and accepted support
choices, without tracing or continuum-reference evaluation. CPU processes
compare all 31 prepared arrays against literal pre-extraction code and all
six-field output leaves for 22 states, four D/N/mixed patterns and both frozen
diffusion spans on every N32/N48/N64 owner. Material remains h/32 with h/16
outer samples. The same campaign then checks complete RHS execution on one
and four actual A100 GPUs and measures compilation, staging, synchronized
warm calls, host/device memory and checkpoint storage separately.

The floating action replay gate is `1e-8 + 1e-11*abs(expected)`, inherited from
the bounded extraction review; same-CPU preparation rows remain bitwise.
Independent accepted C3 actions are retained as preflight inputs. Content
identities, complete owner/raw coverage, all cases/leaves and actual GPU
backend proof are mandatory. Forced CPU multi-device tests do not satisfy
the GPU gate. Source/input changes and failed gates return for local review.

Preparation of this campaign does not mean its global or A100 gate has run.
The evaluator remains opt-in, dense execution remains selected, and no
production selector is promoted. Remaining physical/static transfer gates,
span/eta support selection and Q09 evolution retain their existing scope.

Local release preflight for the Q08 campaign passed at N32/N48/N64: all 31
arrays are bitwise equal and all legacy/extracted output-leaf discrepancies
are zero for the bounded samples. Independent saved actions pass the frozen
budget. Twenty campaign tests and eleven operational gate tests pass; actual
A100/full-grid execution remains pending. On the user's cleanup request,
40.1 GiB of older bulk artifacts were retired while preserving current C3
P/Q and Q08 inputs plus lean historical records. Retirement inventories and
per-directory markers are in `work/workspace_cleanup_20261002/`.

### Q08 A100 component bottleneck audit — 3 October 2026

The returned [component profile and local analysis](../../../../work/q08-component-profile-ff1bff86-20261003T024800Z/local_report.md)
identify the general nonsymmetric 5x5 characteristic eigensolve as the dominant
GPU cost. For one full N32 smooth all-Dirichlet case, synchronized warm medians
are 0.192 s CPU versus 18.066 s A100 for eig alone, and 0.469 s versus 18.005 s
for the full six-field RHS. Reconstruction takes 1.51 ms on A100, and the
centered/diffusion/current subset takes 2.19 ms. Four GPUs reduce full RHS to
4.542 s (3.94x relative to one shard); sharding is not the leading bottleneck.
Saved HLO and the profiler trace identify `cusolver_geev_ffi`; the pinned
JAX 0.9.2 implementation invokes the general solver separately for each small
matrix, with repeated kernels and solver-internal transfers.

Full CPU/GPU and one/four-shard action replay pass the unchanged
`1e-8 + 1e-11*abs(expected)` gate; maximum direct difference is 1.835e-9.
Eigen residuals are approximately 1e-15. All 153 returned files were verified
against their archive members. This single-case profile does not close the
full Q08 matrix or resource gate. The report records incomplete CUPTI buffers,
a repaired split-container validation harness, and the isolated split's unit
normal (full RHS comparisons use actual geometry).

The profile motivates replacing the general eigensolver while preserving the
characteristic matrix, spectral split and stopped-gradient contract. The
user subsequently ruled out CPU offload; the device-native candidate below
is the selected investigation. No donor, degree, span or physical upwind
policy change follows from this performance evidence.

### Q08 device-native characteristic candidate — 3 October 2026

Following the P06 closed-form implementation and the user's explicit choice
to keep numerical computation on GPUs, the next candidate replaces Q's
generic nonsymmetric eigensolve with its exact characteristic polynomial.
The five-field symbol has one exact root Vi and a quartic in lambda−Ve.
Scaled derivative-cubic bracketing, 64 fixed batched bisections, algebraic
eigenvectors and a small device LU inverse produce the sign projectors. The
existing live-matrix/stopped-projector AD convention and device-side
Frobenius/Rusanov fallback are retained. The candidate has no general eig
call or host callback; the CPU-LAPACK proposal above is superseded by the
user's GPU-only requirement.

`characteristic_method="polynomial"` is opt-in in the compact QPlan and its
sharded RHS; the accepted `"eig"` remains the default. Candidate validity
adds physical-domain and residual checks, so extreme-state equivalence is
not assumed. Unit controls cover reversed/zero normals, equal flow speeds,
zero Ti/tau, nonhyperbolic states, a singular alternative eigenvector formula,
conditioning, JVP/VJP and one/four-device mechanics including empty shards.

The [bounded saved-C3 replay](../../../../work/q08_polynomial_20261003/c3_replay.json)
covers 22 states, four BC patterns, both diffusion spans and seven stratified
owners per N32/N48/N64. All output leaves pass the unchanged
`1e-8 + 1e-11*abs(expected)` budget; largest absolute difference is 2.274e-13,
largest budget fraction 1.010e-5, with identical valid flags. These are
implementation replays, not new global orders or GPU speed measurements.

The [bounded A100 benchmark](../../../../DRBX/scripts/q08_polynomial_profile/README.md)
reuses the original complete N32 CPU banks without new tracing/preparation.
Both numerical baseline and candidate run on GPUs, with a fixed 12-case
replay matrix and six timing variants. Its frozen overlay isolates this
change from concurrent P/production edits. Full Q08, larger-grid performance,
production selection and evolved physics remain open; do not resume the
large slow audit until this performance comparison returns.

### Q08 polynomial split A100 profile passed — 3 October 2026

The returned [profile analysis](../../../../work/q08-poly-characteristic-profile-f4821f02-20261003T051443Z/local_report.md)
passes the pinned completion/hash validator with candidate identity
`10ce76b4096cccfb7dd5dc6aacafbee15853685fe0ed2c5b4b421c7a96f6f849`.
On the complete N32 grid, one-A100 synchronized full-RHS median falls from
17.896 s to 3.750 ms (4,772x); isolated characteristic splitting falls from
17.977 s to 1.833 ms (9,808x). Candidate one/four-shard medians are
3.918/2.174 ms. Compiled candidate code has device LU/GEMM but no general
eigensolver or host callback. No numerical CPU offload is introduced.

All twelve smooth/constant/held-out-wave x D/N/mixed direct replays pass the
unchanged component gate: maximum full-RHS difference 9.095e-13, maximum
budget fraction 1.585e-5, identical validity flags. Centered/diffusion/current
leaves are exactly equal. Smooth all-D one/four-shard replay each differs
by at most 1.833e-9 (0.1809 of the budget). This profile covers h/16 diffusion
and N32 only; it does not close the full 22-state, both-span, three-grid gate
or promote the polynomial production default. Whole-process peaks are
4.30 GiB host and 4.32 GiB GPU0 live allocations (8.06 GiB allocator pool),
including baseline/candidate compilation and execution.

Evidence supports continuing the full-grid qualification with the polynomial
candidate. First revise and pin the GPU-stage identity and explicitly select
the method; the unchanged old runner still selects eig. Reuse validated CPU
banks and input receipts without relabeling their identities, preserve old
partial GPU records, and store revised GPU outputs separately. Host
manufactured-boundary setup is now the avoidable audit cost: 13.64 s/state
at N32, redundantly called across two spans and four BC patterns. Reuse one
state's boundary arrays across these combinations with bounded memory,
retaining full correctness coverage. Larger-grid memory/timing remain to
be measured. The global campaign was not resumed by this analysis.

The continuation is packaged in
[`scripts/q08_polynomial_global`](../../../../DRBX/scripts/q08_polynomial_global/README.md).
It independently revalidates the original CPU receipts without rewriting them,
and retains the original CPU eig action as the comparison reference. Candidate
one/four-A100 calls explicitly select polynomial, with compiler checks excluding
general eigensolvers and host callbacks. The complete 352-record matrix per
grid and seven warm repeats remain. Boundary reuse caches exact wall rows and
bitwise-uniform nonwall templates under a declared memory bound, rather than
retaining 22 dense states. New results have their own frozen identity/output
folder; old partial eig timing records are not reused as polynomial records.
N32 completes and validates first, then checked resume continues N48/N64.

### Q08 verification overhead optimization — 3 October 2026

The reported N32 polynomial GPU matrix took 708 s, while its 2,464 warm
operator calls took only 8.4 s. This is verification/setup overhead, not a
measured integrator timestep. The harness now evaluates only the requested
manufactured state in bounded wall batches and builds compact wall data
directly, preserving exact nonwall padding. Validated literal chunk host plans
are reused across the four BC combinations; every eager CPU eig action and all
eight independent merge audits remain. Compiler proofs are read/hashed once
per validation invocation and hardware inventory once at preflight; per-record
allocator measurements remain. Exclusive timing now accounts for merge,
reference/audit work, output comparison, proof inspection, I/O and cleanup;
shared CPU/BC timings are counted once per one/four-device pair.

[Bounded boundary audit](../../../../work/q08_verification_optimization_20261003/boundary_audit.json):
all 22 states on the saved C3 N32/N48/N64 banks replay bitwise, including
physical-normal data, tangential data, affine omega padding and cached restore.
Median cold-cache BC speedups were 5.55/4.30/3.73x on these local patches under
load. These are not full-grid or A100 campaign speedups. A runtime bitwise gate
also compares all 22 states against the old producer on distributed actual wall
queries before new GPU work. The complete 352-record matrix, seven warm repeats,
operator arithmetic and replay tolerances are unchanged. No running remote
source/checkpoint is edited or relabeled; deployment requires the new frozen
verification identity in its own output namespace. This optimization does not
close full Q08 or promote the production default.

Validation: 17 focused pytest checks and 8 portable checks pass. The four-host-
device orchestration/resume test passed in 97.12 s on isolated retry; its first
attempt completed all 16 replay records in 281.39 s but exceeded the 300 s
subprocess deadline during the additional resume check. No gate/deadline was
changed. The [literal reference audit](../../../../work/q08_verification_optimization_20261003/literal_audit.json)
also reproduces every output leaf bitwise on the saved C3 N32 smooth-state
patch for all four BC combinations and both diffusion spans, with one lowering
and four retained eager actions per span. Actual A100/full-grid speedup of the
revised verification harness remains to be measured.

### Q08 optimized full-grid GPU replay passed — 3 October 2026

The [returned campaign analysis](../../../../work/q08-verification-v2-149644b9-Apdb1a/local_report.md)
and [independent local reduction](../../../../work/q08-verification-v2-149644b9-Apdb1a/local_analysis.json)
pass at commit `149644b9bda68a137f24a325b8550aa5eebf16d9`, identity
`1a953c60d7352b2a74e6a737d7c2be7dcca9ee194808db3154e5f840d4a9b087`.
All 1,187 completion-file hashes and the committed source match. Local checks
independently reduce the complete record/leaf/shape/timing matrix, 24 literal
CPU merge audits and 48 compiler proofs. Large immutable baseline inputs remain
remote; their source/input and full coverage checks are attested by the pinned
remote validators, rather than numerically rerun locally.

Coverage is 313,696 owners / 405,504 raw cells over N32/N48/N64, 22 states,
four D/N/mixed patterns, both diffusion spans and one/four actual A100s:
1,056 records and 7,392 synchronized warm calls. All required Boolean flags
match and are valid; sampled full-catalogue BC data replay bitwise. Maximum
floating replay differences are 2.503e-9 / 6.270e-9 / 6.368e-9, using
0.233 / 0.627 / 0.627 of the unchanged component budget. Diffusion is within
1.092e-12; literal merged/chunk audits use at most 0.00614 of the budget.
These are implementation differences, not new N-O/O-R/N-R convergence norms.
The polynomial candidate has CUDA LU/GEMM/triangular-solve targets and no
general eigensolver, CPU LAPACK or host callback.

Median warmed full-RHS calls are 4.67 / 13.92 / 34.74 ms on one A100 and
2.29 / 4.94 / 10.30 ms on four, with matched speedups 2.05 / 2.88 / 3.37x.
These are synchronized calls with plans/input arrays already on device, not
P+Q integrator steps or live sheath updates. GPU stages take 6.14 / 16.90 /
38.34 min; the numerical Slurm step completes in 63:03. N32 improves from the
earlier reported 708 s to 368.6 s with unchanged coverage. The new ledger does
not double-count shared BC/reference work. At N64, eager literal chunk audits
take 1,085 s and merged CPU references 684 s (77.6% combined), while all 2,464
warm GPU calls take 54.6 s (2.39%). Remaining verification cost is mainly the
independent CPU audit; it is not production RHS work.

Measured host-process peaks are 8.72 / 19.66 / 49.55 GiB, versus preflight
estimates 4.22 / 14.26 / 34.90 GiB. The estimate excludes external runtime
overhead and must not be used as a total-process bound on tighter allocations;
calibration/headroom is a remaining engineering follow-up. Actual usage stays
within the 224.4 GiB budget (Slurm whole-step MaxRSS about 53.9 GiB). At N64,
GPU0 peaks at 7.01 GiB live / 8.50 GiB pool and the other GPUs at 3.38 / 4.13
GiB each. These peaks include setup and sequential one/four-device variants,
not isolated production-only storage; compressed BC cache is 1.46 GiB.

Record the full-domain representation and actual-GPU implementation replay
subgate as passed and retain the polynomial implementation as the working Q08
candidate. Continue with the predeclared bounded span/eta-support comparison
and remaining static coupled-term evidence before final RHS freeze. Full Q08,
production-default selection, evolved stability/timestep feasibility and the
documented physical-wall/exterior-crossing limitations remain open. No new
tracing, numerical tuning, source promotion or production change occurred in
this analysis.

### Q08 six-field global static MMS preparation — 3 October 2026

The next campaign reuses the completed extraction inputs and GPU replay receipts
without retracing, rebuilding rows or rerunning the old performance matrix.
All 22 states, four D/N/mixed patterns, N32/N48/N64 and complete owners remain.
The 31 independently scored outputs include six-field centered, correction,
diffusion and combined actions, primitive-product current, omega advection and
current drive, phi force, electron material, Ti compensation and generalized
force. Candidate N runs on four A100s with the unchanged polynomial split.
Independent O/R references are computed once per chunk on a bounded CPU pool
and reused across BC patterns. A steady S=-R source is added in the same GPU
assembly call and its residual is checked against N-R. No evolved MMS or full
physical-sheath/SAT claim is added.

R retains the previous raw-center target and physical-volume owner projection.
The three fourth-order coordinate-flux reference steps are retained, including
per-term sensitivity. O/R are not integrated volume references and the first
step remains the target; no favorable step is selected after seeing errors.
Global/regional RMS, relative RMS, maxima with owner IDs, signed integrals and
both refinement orders are returned for local scientific interpretation.
Scientific rebounds do not trigger tuning or abort the computation.

Local replay of 21 complete actual-HSX owners, all states/BCs/spans reproduces
persisted O/R within 1.71e-13/1.42e-13 and polynomial N within 4.26e-11. These
are bounded implementation checks, not new convergence evidence. A separate
spawn-worker/checkpoint test and four forced-CPU-device sharding test reproduce
source addition to 6.94e-18; actual CUDA preflight is mandatory remotely. The
host merge guard now includes three times the old estimate plus 8 GiB, and
CPU reference workers are pinned before JAX initialization with a 4 GiB cap.
The previous seven repeats, one-GPU scaling matrix and full literal CPU N
sweeps are omitted because those engineering gates already passed. Full Q08,
span/support selection, production promotion and Q09 evolution remain open.

The initial remote launch at `19136040` passed input verification and actual
CUDA bounded replay on N32/N48/N64, then stopped before the pilot produced any
reference chunks. This was an import collision between the MMS controller and
the older polynomial campaign, not a numerical failure. Explicit MMS package
imports replace the ambiguous bare names; the regression suite exercises a
cold controller and spawned worker with the conflicting module present.
Recovery uses a new source identity and separate RUN, retaining the same
immutable banks and implementation receipts and repeating the prescribed
gates. No formulas, tolerances, support, traces or qualification claims change.

### Q08 current six-field global MMS returned — 3 October 2026

[Local scientific report](../../../../work/q08_rhs_mms_return_analysis_20261003/report.md),
[artifact audit](../../../../work/q08_rhs_mms_return_analysis_20261003/validation.json),
[regional results](../../../../work/q08_rhs_mms_return_analysis_20261003/local_analysis.json).
Campaign `bd3410af` / identity `e8303a1a...389652` completed all 3,174 reference
chunks and 528 four-A100 action records for 313,696 complete owners, 22 states,
four D/N/mixed patterns and both diffusion spans. Its final completion receipt
was not written: CSV output uses CRLF, but the validator compares it with
Path.read_text, which normalizes to LF. Local independent reduction reproduces
all primary sums, RMS, relative RMS, maxima, owner IDs and signed integrals
exactly; cross-platform log evaluation changes orders by at most 8.88e-16.
The report and all non-order CSV cells match exactly. All scientific artifact
hashes pass; one operational runtime log was appended after its verification
snapshot, retained explicitly in the audit. Original remote artifacts remain
unchanged, and no numerical campaign rerun is needed for this text defect.

All six assembled equations have decreasing global and regional N-O and N-R
RMS for every nonconstant state, all BC patterns and both diffusion spans.
Smooth combined N-O is approximately fourth order (global 3.82–4.06; every
regional interval >=3.817). With h/32 diffusion and all-D data, combined global
N-R orders (32→48 / 48→64) are n 1.851/3.693, Te 1.764/3.731,
Ti 1.910/3.786, Vi 3.012/0.993, Ve 2.858/1.145, omega 1.823/2.379.
N64 smooth relative global errors are 0.011–0.061%. The velocity intervals
continue to be O-R dominated and match the accepted material sampling
limitation; this is not a new GPU or reconstruction failure, nor proof of a
reference bug. D/N/mixed patterns preserve these conclusions.

For h/32, combined wave global N-O orders are >=3.038/4.428; N-R orders are
>=1.814/1.494. Worst N64 relative combined wave N-R is 4.483% globally,
7.549% in the wall region and 7.778% in the outermost ring. Genuine N-O
exceptions remain: inner minimum 1.868, last-two-aggregate minimum 1.681
(held-out wavelength .35/110-degree omega), while the wider transition
minimum is 2.027. Isolated short-wave diffusion N-O can reach 0.967 in the
last two aggregate rings (Te, same held-out state); RMS still decreases.
Isolated smooth/wave diffusion global fine-interval N-R is >=2.035/>=2.232.
h/16 retains a small isolated diffusion core/first-ring coarse-interval
rebound. Maxima and signed integrals need not decrease with RMS; e.g. one
combined wave omega wall maximum increases 2.46x on 48→64. Do not claim
universal regional second order or pointwise monotonicity.

One standalone diagnostic must be excluded pending a harness correction:
`electron_ti_compensation` returns the negative material Ti column in N, but
the O/R diagnostic was packed with the positive sign. The generalized force,
electron assembly and six combined equations are unaffected. Bounded replay
at 21 owners, D/N, smooth and held-out fields verifies the sign identity to
1.42e-14; N64 smooth D sampled N-O drops from 47.69 to 3.71e-5 after aligning
the diagnostic sign. These are bounded norms, not corrected global results.
Norm-only global payloads cannot recover its exact corrected N-O/N-R without
a focused replay. Correct this column and CSV comparison before the next
harness use; do not modify the operator to repair diagnostic output.

Constant N-O is <=5.69e-9; steady source-pair discrepancy is exactly zero.
The conservative global RMS bound on measured reference-step sensitivity is
<=8.49e-8 at N64 over all terms/states (pointwise maximum 1.75e-6), much smaller
than the leading errors. Allocation elapsed about 44m52s; reference stages
took 47/67/117s and GPU stages 117/268/611s. N64 host peak was 33.03 GiB,
device live/pool peaks 3.38/4.12 GiB. The complete invocation after initial
preflight/pilot took about 23m46s.

Evidence supports retaining the current static shared-RHS assembly and C3
transfer, with the above exceptions. Full Q08 remains open. Next: the planned
bounded span/eta-support, resolved-scale and stiffness assessment, including
the short-wave aggregate-ring limitations. Physical wall/SAT, exterior caps,
perpendicular coupling, Q09 evolved MMS and production promotion stay separate.

### Q08 harness repairs and bounded span/support/stiffness audit — 3 October 2026

[Bounded report](../../../../work/q08_span_support_20261003/report.md),
[harness review](../../../../work/q08_harness_repair_20261003/report.md).
The requested GPT-6.1 Sol worker repaired byte-exact CSV completion validation,
the standalone O/R Ti-compensation sign, and the mutable operational runtime-log
hash treatment. Parent review and 14 focused tests pass. All 31 output identities
are checked, retaining the immutable first-28 persisted numerical replay.
The assembled equations, generalized force and numerical tolerances are
unchanged. The old manifest intentionally rejects these local edits; a future
campaign needs a newly reviewed source identity. Returned remote receipts and
the scientifically excluded old global Ti-diagnostic norms remain untouched.

The bounded numerical identity starts `c1899e35bd9544f4`. The comparison covers
33 complete owners / 221 raw members across N32/N48/N64, all 22 states, D/N,
consistent inner h/16/h/32/h/64 with twice-inner outer characteristic separation,
and common five-plane quartic versus three-plane quadratic eta support. The
magnetic evaluator, RK4-64, transverse donors and accepted repair choices are
fixed. Baseline rows replay exactly and actions within 4.66e-9 (0.409 of the
unchanged replay budget). Degree-appropriate reproduction is within 2.10e-15;
physical-normal enforcement is within 1.96e-12. The primary run took 255.9 s,
peaked at 1.36 GiB per process and produced about 92 MiB including checks.

Retain five planes: at N64 three planes saves 40% of local donor/wall entries,
but increases smooth combined N-O by 104–521 times, often against a small
baseline. At eight eta points per wavelength the bulk traced-G amplitude
response is 0.988 with five planes versus 0.900 with three; at four it is
0.849 versus 0.637. These are derivative response measurements, not evolved
damping rates. No optimized three-plane runtime speedup is claimed.

Five-plane h/64 reduces N64 smooth combined selected-owner N-R to 0.138–0.270
of h/32 with almost unchanged N-O. Wave N-R ratios are 0.659–1.013; the
outermost-wall Neumann held-out Ve maximum increases about 1.9% as favorable
N-O/O-R cancellation shrinks. Thus shortening does not cure the wave
reconstruction limitation. These are bounded norms, not global orders. The
h/16 material variant crosses once at its positive outer sample at N48;
inner samples remain inside, so this is not a new failure of previously
qualified h/16 diffusion caps. Exterior continuation remains diagnostic-only.

The reduced actual-HSX periodic eta-line Jacobians show no universal inverse-
span stiffness increase: interior/Neumann spectral radii vary less than 0.1%
over a fourfold span change; N64 Dirichlet wall changes +2.68% on h/32 to h/64.
G/diffusion row L1 norms change at most about 1.32%/2.14% on that comparison.
However all reduced line systems have growing modes, strongest at the
Dirichlet wall. Independent-transverse 30/140-owner wall patches also have
positive eigenvalues for D/N. Their strong patch-size sensitivity and artificial
fixed patch-edge data preclude a full-grid instability claim, but growth cannot
be dismissed solely as the transverse-constant line restriction. Nonlinear
directional checks reproduce the patch Jacobians to relative 2.37e-9. RK4
approaches the same growing matrix-exponential solution; this is not a stable
evolution pass. `modal_limits.json` explicitly corrects an original diagnostic
that admitted tiny positive modes into a nominal non-growing-mode timestep
limit; saved matrices and scientific actions remain unchanged.

The requested bounded comparisons and measurements are complete. Keep h/32
and five planes as the working qualified baseline; h/64 would require affected
global qualification if selected. Next isolate material/correction/current and
diffusion growth with the actual boundary-consistent operator and suitable
background/closure before timestep signoff. Final per-operator span selection,
anisotropic resolution ratios, Q08 final freeze and Q09 evolved stability remain
open. No operator promotion, new campaign, commit or push was performed.

### Q08 growing-mode investigation reviewed — 3 October 2026

[Investigation](../../../../work/q08_growing_modes_20261003/report.md),
[independent parent review](../../../../work/q08_growing_modes_20261003/parent_review.json).
Identity `04d17c944e2ece65...`: all 27 source, 55 input, 25 prior-evidence and
40 result hashes pass parent verification. The parent independently reproduced
representative same-mode term contributions and lifted-mode residuals. No
packing, parameter-orientation or electron Ti/force-pairing defect was found;
the ten old matrix cases replay within 1.24e-9. Nonlinear perturbation evolution
and time refinement reproduce the restricted-system growth.

Small periodic interior and Neumann-wall line rates disappear in a diagnostic
control subtracting only the centered div(b) coefficient. This implicates
geometric compression of the forced constant background and/or its transverse
restriction, not a reason to remove physical div(b). Here b is the unit field;
this is distinct from an assumption that div(B) vanishes. The large rates are
dominated by centered transport through restricted derivatives. For the same
N64 Dirichlet-wall line eigenvector, centered/correction/current/diffusion
contributions are +2859.56/-121.56/+0.10/-316.81, totaling +2421.29. With phi
fixed, current-to-omega coupling is triangular and cannot change material
eigenvalues. Correction is not uniformly damping in every restricted patch;
N64 patch diffusion-only spectra have negative real maxima.

The strongest evidence concerns artificial closure. A 252-owner bulk patch
without physical wall rows also grows at +1041.13. About 74% of that mode's
norm lies on artificial edges. The 252-owner D wall mode is concentrated more
at the patch's inner artificial edge than at the physical wall. Lifting wall
patch modes by zero and applying the actual Jacobian on their 1014-owner
reverse donor footprint produces exterior/interior action ratios 0.991 D and
0.701 N. Relative full lifted eigen-residuals are 0.704/0.574 despite inside
residuals near 1e-14: these are not approximate global eigenmodes. The complete
action does have positive initial growth in the diagnostic fluctuation norm;
this may be transient growth and is neither a thermodynamic-energy claim nor
proof of full-grid modal instability. Geometry still affects the rows; no
MMS O/R reference enters this diagnostic.

Keep the five-plane h/32 policy and existing static gates unchanged. The next
discriminating bounded test is to evolve the exterior response: begin with
J²v on the next reverse-support collar, then short matrix-free propagation
with verified support/tail error and no repeated pinning of halo values.
Compare against the original fixed-exterior trajectory. This is preferable
to tuning reconstruction or interpreting another isolated patch spectrum as
global stability. Full-grid stability, suitable physical closure/equilibrium,
final Q08 selection and Q09 evolution remain open.
