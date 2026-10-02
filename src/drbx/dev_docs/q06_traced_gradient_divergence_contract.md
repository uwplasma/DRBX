# Q06 traced gradient/divergence contract

Status: Q06 closed by explicit user acceptance on 30 September 2026, with
documented exceptions; original research contract authorized 29 September 2026.
This governs the current selective traced reconstruction. Earlier Q06 direct
face/integrated-reference assignments are historical evidence, not instructions
to restore their numerical scheme. Closure qualifies the unchanged static
traced G/tube D within the scope below; no production pass is implied.

## Entry decision

### User clarification — 30 September 2026

The selected Q06 gradient target is now an **FCI traced scalar-difference
gradient**, avoiding derivatives of the reconstructed field in all three
coordinates. This supersedes the direct-gradient choice and the prohibition on
endpoint secants below; those paragraphs record the previous implementation
contract. Retain the reviewed direct gradient as a diagnostic baseline.

For the existing symmetric eta-parametrized traces, the initial candidate is

```text
G_trace(f) = b0^eta * (f_plus - f_minus) / (alpha*deta).
```

Here alpha*deta is the total signed eta separation, not either leg length.
Reconstruct scalar values at the existing caps, form the raw action, then apply
the unchanged complete-owner projection. The center b^eta factor converts the
eta derivative along the field line to the parallel derivative. This candidate
does not use the tube-divergence B weights or divB correction. Multidimensional
value reconstruction and prescribed physical-normal wall constraints remain;
neither entails applying a full field-gradient contraction for G at runtime.

The conversion is the chain rule along the traced curve: `dx/ds=b`,
`deta/ds=b^eta`, and hence `df/ds=b^eta*d[f(x(eta))]/deta`. The total derivative
along this curve includes the changing radial and poloidal positions; it is
not the coordinate partial derivative at fixed transverse coordinates. No
assumption of eta-dominated field direction is made. Eta must remain a valid
parameter (no zero/turning b^eta); the existing trace producer owns path validity
and the new preparation checks nonzero, same-sign b^eta at all three slots.

Symmetric eta caps are chosen to reuse the existing RK4-64 traces and fixed-leg
ghost-endpoint boundary reconstruction. They do not assert superior accuracy
to arc-length differencing. Smooth symmetric eta traces can also give a
second-order total-arc-length secant; arbitrary wall-truncated unequal legs
require the appropriate centered unequal-spacing formula. No leg is shortened
at the wall in this implementation. Dirichlet data enter scalar wall-node and
query-value lifts; physical-normal Neumann data enter the existing scalar
extension. Tangential Dirichlet derivative inputs are not used by this value
operator. The normal derivative is not identified with a parallel derivative.

The reusable implementation is `stencils/q_traced_gradient.py` (checked source,
slot symmetry, value-row/BC contraction and identity) and
`native/q_traced_gradient.py` (pure-JAX raw/owner actions and device staging).
Its runtime stores no field-gradient, magnetic-gradient or tube-divergence
arrays. The original Q05 source bundle still contains gradient diagnostics;
this change does not claim that shared preparation is value-only throughout.

Bounded test design (now executed): reuse both saved h/16 and h/32 spans, frozen donor/repair
choices, all 26 fields, both D/physical-normal N conditions and stratified
actual-HSX core, inner, transition, bulk and wall owners. N uses reconstructed
cap values; O uses exact manufactured cap values in the same traced difference;
R uses the analytic midpoint parallel gradient with identical owner projection.
Report N-O, O-R and N-R separately, constants, boundary lifts, maxima and staged
runtime/memory. Compare against the existing direct G without tuning supports
or retracing. A bounded pass precedes global qualification using the saved maps;
the winning span was not assumed in advance. The [completed bounded report](../../../../work/q06_traced_gradient_20260930/report.md) favors h/32 for global preparation: O-R falls 3.9867x on halving span and N-O stays close to the direct-gradient baseline. All 62 focused tests and 42 site/span checks pass. This is not a global accuracy claim. No selected cap is exterior, so explicit ghost-cap coverage remains untested by this bounded matrix.

The [h/32 global traced-gradient campaign](../../../../work/q06_traced_gradient_global_20260930/parent_review.md)
has now completed: all nonconstant global and regional N-O RMS checks exceed
second order (global minima 2.6547/3.1345). All nonconstant N-R RMS decrease,
but some fine-interval orders remain below two and are dominated by O-R.
Localized inner maximum rebounds persist. This records a static reconstruction
gate pass, not unconditional continuum accuracy. The subsequent user closure
accepts the documented continuum/local-maximum limitations. No exterior crossings occur in the saved traces.

The [30 September structural/boundary audit](../../../../work/q06_structural_boundary_audit_20260930/report.md)
passes affine D/N handling, constant identities, explicit value assembly and
physical-normal wall-node constraints. Full-domain smooth-field pairing defects
decrease with refinement and are mostly present in the analytical midpoint
volume sum. Separate owner-basis tests nevertheless disprove exact weighted
adjointness, and complete incoming-support bulk pulses disprove exact
owner-volume conservation of tube D. These are discrete structural findings,
not O-R reference errors. The subsequent
[smooth-packet balance audit](../../../../work/q06_resolved_conservation_20260930/parent_review.md)
measures eight fixed-width bulk profiles with zero wall flux and complete
incoming support. All absolute and activity-normalized defects decrease
monotonically; N64 imbalance is 0.004–0.115% of divergence activity (median
0.0447%). This is not percent mass loss per timestep; cancellation between
reconstruction and continuum-sum errors is retained in the reported budget.
The user retains the implementation and explicitly closes Q06 as a scoped
static qualification. Approximate balance and non-adjointness are accepted
limitations; later accumulated-drift assessment remains open. No D(G f) diffusion was
evaluated; no operator or production default changed.

Keep tube divergence as the selected divergence path and the accepted direct
cap-gradient tube diffusion unchanged. The separate owner-field D(G f)
diffusion comparison is deferred and is not an immediate prerequisite. It is
not a replacement for the frozen diffusion. Individual traced G/D accuracy and
the applicable weighted pairing, conservation and wall audits are complete
with the documented exceptions accepted. Q07 is ready; preserve the specialized
current–phi/SAT compatibility requirements instead of assuming generic G/D
are an exact adjoint pair. Q04 traced
diffusion evolution and Q08 performance gates remain open.

The user closes Q05 on the corrected extraction and second-review evidence:
36 focused tests, 30 curated checks, 42 bounded site/span replays and two complete
N64 chunks, maximum action discrepancy 4.467e-10 against the unchanged 1e-8 gate.
Complete-domain saved-action replay was not performed. It is explicitly waived
as a prerequisite to this Q06 work, not recorded as passing or complete.
See [review](../../../../work/q05_traced_extraction_20260929/review_v2/report.md).
Traced Q04 evolved/structural checks and production promotion remain separate.

## Frozen numerical inputs

Reuse the [diffusion contract](q_traced_diffusion_frozen_contract.md) and corrected
schema-v2 preparation described in [the extraction interface](q_traced_extraction.md).
Campaign identity: c0d64dba761087a3273d29cd4728689c4f4b5b67f838873a845d1cdb1a809f8b.
Actual canonical HSX N32/N48/N64, one value per compact owner, physical-volume
weighted raw-midpoint observations and complete raw-member projection remain
fixed. Keep selective geometry-only gradient repair, inner transverse quartic
and common five-plane eta reconstruction, structured outer and current D/N wall
rows. Do not retune support, degree, ranks, repair criteria or reference fields.
Use saved GPU RK4-64 endpoints. Both alpha=1/16 and 1/32 remain in the comparison;
cap separation is alpha*deta and legs are signed alpha*deta/2.

## Operators

Let b=B_vector/|B_vector| and B=|B_vector|. For logical coordinates q^a, b^a
denotes the contravariant unit-field components supplied by the frozen evaluator.

1. Direct gradient: G(f)=b^a partial_a f. Evaluate derivatives of the original
   scalar reconstruction at each raw midpoint and project to owners. Also expose
   raw cap gradients using the same rows for diffusion/composition diagnostics.
   No endpoint secant and no derivative of an already sampled gradient replaces G.
2. Direct divergence candidate: D_direct(f)=G(f)+f div(b), with
   div(b)=div(B_vector)/B - b dot grad(log B). Form both terms at raw midpoints
   before owner projection. The raw scalar evaluation must be declared explicitly
   (the existing center value row is the initial candidate), not silently replaced
   by an owner centroid value or a product of separately averaged quantities.
3. Tube divergence candidate:
   D_tube(f)=B0*b0^eta*((f/B)_+-(f/B)_-)/(alpha*deta)
             +(div(B_vector)_0/B0)*f0.
   Use the same selected reconstruction for cap and center values and the frozen
   magnetic coefficients. Do not assume div(B_vector)=0.

The initial gradient assignment covered item 1; the subsequent
[bounded divergence assignment](q06_bounded_divergence_assignment.md) authorizes
items 2 and 3. User direction: favor tube divergence as the implementation path;
retain direct divergence as a diagnostic baseline. This preference does not
waive accuracy or structural gates, and discrepant results must be reported.
Direct G remains useful independently and for accepted cap-gradient diffusion.
Direct divergence combines all three logical derivatives into a precontracted
runtime row; tube divergence instead differences reconstructed scalar cap values.
Both still use multidimensional reconstruction, and the tube's divB correction
still requires prepared magnetic derivative information.
Neither algebraic product identity nor tube nomenclature certifies
exact discrete conservation or weighted adjointness. Audit these independently.

## Boundary, owner and runtime semantics

Reuse explicit Dirichlet wall-node values and query tangential derivatives, and
physical-normal Neumann wall-node data. Normal derivative is not merely radial
coordinate derivative. Boundary lifts are affine, field/batch dependent and
exactly zero on nonwall rows. Keep values on all raw members of an agglomerated
owner in evaluation/projection; aliases are not extra evolved unknowns.
Runtime is pure JAX with fixed prepared coefficients, float64/complex128,
changing state and boundary data, no geometry calls, fits or searches.
Preparation stays under drbx.stencils; application under drbx.native.

## References and comparison sequence

Use N (implemented action), O (same discrete evaluation with exact manufactured
values/derivatives) and R (analytic continuum raw-midpoint action, projected with
the identical owner weights). Keep N-O, O-R and N-R separate. For direct G at
midpoints, exact derivative O equals R when using the same geometry and
observation convention: report that fact; do not invent an outer-stencil error.
At caps report reconstruction gradient error separately, not as midpoint N-R.
Tube D has a genuine finite-span O-R term. Do not add costly integrated references.

Start with bounded actual-HSX core, agglomerated, repair-active, interface, bulk,
wall and known hotspot locations, both spans, both BCs and all 26 frozen fields.
Use G(1)=0 with compatible BCs and D(1)=div(b), not D(1)=0. Test explicit nonzero
BCs and nonpolynomial wave controls. Bounded site sets are mechanism diagnostics;
their apparent orders are not representative global convergence certificates.

The earlier owner-field D_h(G_h f) comparison is **deferred and not required
for Q06 closure**, as clarified by the user. Do not perform it as a hidden
structural prerequisite or replace the accepted cap-gradient diffusion.

Audit the selected traced G and tube D directly for weighted pairing,
conservation and boundary terms. The individual global G/D campaigns are now
complete; reuse those actions and references where applicable instead of
repeating their accuracy campaigns. No composed diffusion is involved in
these audits.
The user closes Q06 on the completed global N-O and bounded BC/algebra
evidence, accepting documented N-R/local-maximum, non-adjointness and
approximate-balance limitations. Exterior ghost behavior remains unqualified.
Q07 implementation and production changes require separate authorization. Q04 evolution and Q08 performance qualification remain
open; this contract does not close them.

## Current implementation assignment

The [direct-gradient implementation plan](q06_direct_gradient_assignment.md)
defines the initial bounded worker scope and deliverables.
