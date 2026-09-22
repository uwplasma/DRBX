# P06 curvature: P1 bounded formulation audit and reconstruction comparison

## Boundary-functional follow-up result (2026-09-22)

The [bounded A/B/C follow-up](../../../../work/p06_boundary_functional_20260922/report.md)
corrects the earlier mechanism description.  Under the frozen wall contract,
the physical-wall characteristic correction is exactly zero: the wall solver
receives the same reconstructed interior and boundary state, so an imperfect
trace cannot create a wall-face fluctuation.  The measured wall-owner error is
instead in the smooth reconstructed volume gradient.  A boundary-aware repair
must therefore act on wall-adjacent cell reconstruction and, for a complete U
action, on incident central/biased face fits whose support reaches the wall.

A reusable versioned package interface now compiles geometry-only wall value,
physical-normal derivative, tangential derivative, and physical surface-
measure rows together with a separate dynamic linear relation.  Scaled,
rank-revealing SVD/nullspace preparation exports fixed maps
`c=M_owner owner_data + M_bc boundary_data`; eager/JIT/JVP actual-HSX tests
cover changed boundary data.  The sidecar geometry contains no manufactured
values or wall physics, and vorticity receives no independent constraint.

The bounded ladder retained the parent four wall owners and two ordinary
controls per resolution and both predeclared fields.  A is the original cubic,
B the single-wall-center cell constraint, and C the three physical-wall moments
`{1,s_theta,s_eta}` applied to cells and affected face fits with unchanged
degree, support, weights, bias, recentering, and characteristic solve.  C
substantially improves A, but B remains better in the aggregate for both the
centered and complete-U scores.  All physical-wall corrections remain zero;
M+R, constraint, dynamic reproduction, control invariance, and central-face
checks close at roundoff.  Therefore retain the reusable interface, but do not
promote this three-moment candidate to a global campaign.  The next bounded
step is to localize the B-to-C regression before changing moment scaling or
face activation.  No wall-law change, donor/bias scan, global run, or
production integration is authorized by this result.

## Authorized P06-D correction amendment (2026-09-22)

The parent review preserves the completed N32/N48/N64 total-action evidence but
requires a bounded correction pass before any global qualification proposal.
P06-D therefore reuses the frozen 26-owner selections, two manufactured states,
selection-v3 cubic support, bias 0.75, characteristic matrices, continuous
geometry, and legacy wall baseline. It must not repeat the complete study solely
to change the recorded N64 peak RSS or the historical hash result.

This amendment authorizes only the following corrections:

- replace the unstable monkey-patch-dependent implementation hash with explicit
  selected source/implementation identities, independently reconcile the saved
  N64 identities and artifacts, and preserve the original failed receipt;
- extract production material and potential-remainder contributions from the
  same original-state curvature evaluation and unchanged halo/boundary traces,
  verifying their directional and total reconstruction at walls and interiors;
- retain midpoint owner observations while saving independent q1/q3/q5 physical
  and evolution-measure reference arrays, qualifying q5 with bounded higher-order
  checks only where q3-q5 is decision-relevant;
- derive and test a matched owner functional for the diagnostic C/U action,
  keeping raw observations, the discretization evolution functional, and the
  physical-volume reporting norm distinct; and
- replace the automatic aggregate-improvement rule with a term-resolved,
  reference-qualified scientific assessment. A concrete unresolved formulation
  issue or negative bounded result is an acceptable completion.

Deliver corrected evidence under a new revision directory linked to the
original artifacts. Separate computational completion, artifact integrity,
resource compliance, and scientific conclusions. Do not launch a global
campaign, change production defaults, start evolved runs, or edit the shared
roadmap while other work is active; instead provide a ready-to-apply P06 status
paragraph in the final report.

## Purpose and status

P1 owns the next curvature investigation, alongside P's P05 package extraction and replay. P03/P04 and the centered/material P05 static passes remain established. This assignment does not depend on completion of P05 structural or evolved tests.

Deliver a bounded actual-HSX comparison identifying whether the successful cubic owner reconstruction and matched integration provide a consistent complete curvature action, and precisely what remains before global qualification. Do not claim convergence from samples or launch a global campaign.

Read the perpendicular roadmap, package architecture/testing guidance, and the returned centered/material campaign analysis. Preserve the current production characteristic physics and boundary baseline. The bracket's scalar upwind correction must not be substituted for the coupled curvature characteristic action.

## Work sequence

### P06-A: Establish the continuum and discrete contracts

Trace the curvature contribution through:
- fci_drb_EB_rhs._curvature_rhs_contributions;
- fci_operators.local_curvature_production_path_op and local_curvature_conservative_op;
- fci_curvature_production_flux characteristic matrix/fluctuation primitives;
- existing geometry coefficient generation, radial quadrature, within-cell fluctuations, owner restriction, and continuum curvature sources.

Write an explicit equation/units/sign/measure table for density, Te, Ti, and vorticity. Confirm whether Vi/Ve curvature is identically absent in the selected model; do not invent new channels.

Define psi=phi+tau*Ti and decompose total curvature into material M and potential remainder R. The current remainder coefficient is (-2n,-4Te/3,-4Ti/3,0)/B multiplying C(psi). Independently derive M from the continuum equations, not by subtracting a numerical remainder from a numerical total. Verify algebraically that M+R reproduces the independently derived complete equations.

Reconcile C(f)=K.grad(f) with the conservative representation B/J*d_alpha[(J*K^alpha/B)f]. Track any divergence-of-coefficient defect rather than assuming its discrete cancellation. The reference evaluator currently constructs K=B/(2J)*curl(b_cov/B); verify its actual conventions and prefactors. Trace all factors of B, J, rho_star and tau through the material characteristic matrix, Q=J*K/B face coefficient, quadrature weights, within-cell action, and owner sum(raw_volume/B). Do not assert that this measure is wrong merely because the reporting norm uses physical volume.

Distinguish:
1. owner observation convention (raw-midpoint physical-volume-weighted means);
2. discrete evolution measure and the continuum functional it approximates;
3. physical-volume-weighted reporting norm.
Show the map to a common physical owner-residual target. Where physical-volume and V/B averages genuinely differ, quantify this as a separate bounded comparison rather than comparing unlike targets.

Audit the nonlinear product in R: coefficient times an averaged derivative is generally different from averaging the coefficient times derivative. Record where production performs each operation. Include the ion-temperature chain-rule/cancellation diagnostic.

Freeze the boundary from the roadmap, not a guessed rung: physical_wall_model="legacy-velocity-trace", parallel_velocity_wall_bc="neumann", thermodynamic physical Neumann treatment, and existing phi/vorticity Dirichlet treatment. Retain the recorded numerical characteristic pairing/law. No no-flow or sheath-law redesign.

### P06-B: Freeze bounded HSX fixtures and qualify references

Use existing N32/N48/N64 real HSX artifacts, the same continuous geometry reference across resolutions, and selection-v3 owner observations. Record source/geometry/configuration hashes before calculations.

Select up to four owners per category per resolution, spread across eta deterministically, and deduplicate: axis-adjacent, agglomerated bulk, size transitions, ordinary interior, radial wall, theta seam, eta seam. Use geometry-only choices, saved before errors are computed. Each owner includes every constituent raw cell, all incident faces, and required donor support. Do not substitute face-only samples.

Use two predeclared states:
1. Existing corrected frozen MMS field catalogue and parameters. Audit its axis/metric evaluation lineage before reuse; do not resurrect historical interpolant-based vorticity references.
2. An independent smooth positive regular-chart state. Let x=u*cos(theta), y=u*sin(theta), zeta=2*pi*(eta-eta_origin)/eta_period, E=(1-x*x-y*y)^2:
   n=1+E*(0.08*x+0.03*y*sin(zeta));
   Te=1+E*(0.06*y+0.02*x*y*cos(zeta));
   Ti=1+E*(0.05*x*cos(zeta)+0.02*(x*x-y*y));
   phi=E*(0.04*x*y+0.03*y*cos(zeta));
   omega=E*(0.05*(x*x-y*y)+0.02*x*sin(zeta)).
   Prescribe phi independently; this is a curvature test, not polarization closure. Use existing tau and normalizations. The envelope makes thermodynamic normal derivatives and phi/omega values compatible at u=1; confirm compatibility with the actual boundary adapter. Vi/Ve are background values since they are not active curvature inputs.

Evaluate field derivatives analytically, and geometry from continuous evaluators at the actual required points. For the complete continuum sources implement an independent field-gradient expression. Sharing the qualified geometry evaluator is appropriate; invoking the discrete production action as the source is not.

Retain midpoint owner observations. For this bounded diagnostic only, compare midpoint, q3 and q5 owner source integrations and a geometry-derivative step refinement on a stratified subset (at least one owner/category). Reuse prior qualified metric checks. Report reference uncertainty relative to observed differences; if a comparison is below that uncertainty, call it unresolved. Do not require high-order global quadrature or near-zero relative-error divisions.

### P06-C: Bounded decomposition and candidate comparison

Build a research driver isolated from P's package extraction. Initially use the frozen qualified research reconstruction through a small adapter; adopt a completed shared package interface only when available and replay-compatible. Do not duplicate a second production reconstruction implementation.

First replay the current complete curvature action on the bounded owners, including actual within-cell fluctuations, interface jump corrections, physical walls, periodic endpoints, remainder and owner normalization. Compare term-resolved continuum M, R and total; retain directional and regional diagnostics.

Then compare these controlled variants on identical inputs:
- L: current production geometry/state representation and complete action.
- G: same representation/characteristic formulas but continuously evaluated q3 geometry, coefficient products and consistently integrated terms. This isolates geometry/integration from reconstruction.
- C: selection-v3 central cubic values/gradients at all required q3 face and cell nodes, with the coupled smooth material term and potential remainder evaluated consistently. Suppress the interface dissipative jump only as a centered diagnostic; retain within-cell physical transport. Never infer a complete wave-propagation action from the jump term alone.
- U: C plus the curvature characteristic interface correction from frozen bias-0.75 cubic side fits on the same donor supports. Recenter the two states around the common central state using their side-fit difference. Evaluate the existing live characteristic matrix/absolute action at the common state and actual node B, then integrate the fluctuations. Do not use scalar abs(U_phi) for this four-field system. Preserve physical-wall characteristic treatment; no exterior polynomial fit.

For C/U, derive the smooth within-cell and nonconservative product integration from P06-A before coding. The approved change is representation/integration, not a new characteristic model or arbitrary flux. If the face-minus-cell form and direct volume form require additional terms for equivalence, show and include those terms. A continuum-volume evaluation with numerical cubic gradients can be an explicit diagnostic control, but must not be mislabeled as a proven conservative/path-compatible replacement.

Use exact analytic field values/derivatives at those same actual HSX quadrature locations for ONE bounded oracle substitution to separate reconstruction error from assembly/coefficient error. Label it diagnostic only.

If L cannot be sampled without a prohibitively expensive full-domain build, reproduce the exact existing local formulas on complete owner patches and validate a small genuine-action fixture where feasible. Record this limitation; do not stall the entire audit or accidentally launch a global geometry campaign.

Report separately the material smooth term, characteristic correction, remainder, total, radial/theta/eta contributions, raw/owner measures, boundary contribution and fallback activity. Report signed error vectors and weighted squared-error contributions, so cancellation cannot conceal a component failure. Check constants, coefficient divergence and applicable balance identities as scientific diagnostics. Smooth-state positivity fallbacks should be observable; do not tune fields/support/bias to hide activation or invent a new limiter.

There is no per-owner improvement gate, conditioning cap, or required regional order. Complete all valid planned comparisons even when one is worse; stop only for invalid inputs, unresolved formulation needed for a meaningful calculation, resource limits, or work beyond scope. A negative bounded result is a useful completion.

## Deliverables, acceptance, and next stages

Write configuration/provenance, frozen selections, resumable arrays, a compact machine-readable summary, and report under work/p06_curvature_bounded_<date>/. Separate setup/application time and peak memory. Record exact changed source identity, commands, exit/validation status, and reference uncertainty.

Completion means the contracts are reconciled (or a concrete contradiction is isolated), bounded comparisons are reproducible, and the report chooses the next single correction or qualification step supported by the error decomposition. It does not require repaired convergence.

Future sequence, proposed only:
- P06-D: one targeted correction if required, with the same bounded replay.
- P06-E: freeze candidate/field/reference identities and prepare global N32/48/64 complete-curvature qualification. Physical-volume operator L2 order >=1.8 on both intervals for each nontrivial equation and each declared field; report M/R separately. Nonmonotone results are inconclusive; no automatic resolution extension.
- P06-F: reusable curvature integration and later coupled/fixed-time verification under P08/P09. A bracket pass or bounded oracle does not certify curvature.

No new idealized accuracy gates; no global campaign, remote setup, production default changes, evolved runs, additional physics, or commit/push in this assignment.

## Ownership and supervision

P1 owns this plan, new p06-specific research scripts/tests and its evidence directory. P owns P05 extraction/package files and the shared roadmap. Do not modify P's new modules or the shared roadmap concurrently. Provide a ready-to-apply P06 status paragraph in the report for the parent to merge. Existing completed research helpers may be imported read-only.

No direct messages to P, Q, or the parent; progress belongs in this task and evidence files. Parent checks on demand.

Use supervise-long-runs only for actual long stages, with one controller and checkpointing, default ten-minute heartbeat adapting to twenty/thirty minutes maximum, quiet when unchanged. Keep obsolete monitors inactive. Preflight small geometry batches and reuse valid setup across variants/fields. Target P1 peak <=4 GiB and respect the existing 10 GiB total local computation allowance including other active campaigns; inspect resource use and serialize/defer heavy stages rather than overcommitting. Short calculations need bounded waits, not recurring monitoring. Keep source/reference/evaluator sampling bounded and avoid cache growth from unique query arrays.
