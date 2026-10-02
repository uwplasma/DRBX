# Q07 transport contract: material blocks and prescribed-boundary six-field assembly

Status: **five-field static centered and characteristic-corrected material
block qualified by explicit user acceptance, 1 October 2026**, with documented
native-grid order, short-wave and local-maximum exceptions. Evidence:
[C3 h/32 global review](../../../../work/q07_c3_global_review_20261001/report.md)
and [velocity consistency/sampling audit](../../../../work/q07_c3_velocity_audit_20261001/report.md).
The accepted baseline is commit `bc6f5c81`, campaign identity
`ab607bf62cda21dd67935334da9fdea35c2ad0a0c529ff555006aa3a8ed83bc3`.
Current–phi/SAT integration remains open; full Q07 is not closed. Q06 remains
closed with its accepted approximate-conservation and non-adjointness exceptions.
This is research/static qualification; no production selector/default is changed.

## Current next step after the six-field assembly audit

The [bounded six-field audit](../../../../work/q07_six_field_assembly_20261002/report.md)
passes on 21 complete C3 HSX owners at N32/N48/N64, 22 manufactured states,
four D/N/mixed combinations and both accepted diffusion spans. The research
entry points are `stencils/q_parallel_rhs.prepare_six_field_rhs` and
`native/q_parallel_rhs.apply_six_field_rhs`. They compose material/characteristic,
scalar vorticity advection, current/phi and six constant diffusion channels,
with prescribed phi. Collisions, perpendicular terms, external sources and
physical endpoint SAT are outside this selected differential block.

Material/current preparation explicitly balances the tube center coefficient
as `div(b)-Lminus-Lplus`; diffusion retains its frozen cap-gradient weights.
Current is a product of reconstructed primitive slots. Its diagnostic
physical-minus-zero-BC response includes product cross terms and is not an
affine map of primitive data or an independently imposed current boundary law.
Apply B²/n before complete-owner projection, and include the material Ti
compensation and its generalized-force partner once.

Component replay is 4.36e-11; prior C3 current/phi replay 2.20e-11 and diffusion
replay 5.39e-13. The 103488 records, constants, prescribed BCs, JIT and declared
JVP checks pass; 37 focused tests pass. Full corrected-state AD retains the
existing frozen spectral-projector convention. Bounded smooth N-O decreases
for all six assembled equations, while inherited O-R/short-wave exceptions
remain. No global orders or full Q07 closure are inferred from these samples.

By the user's decision, retain the general reconstruction and move on from
basis/enrichment investigations. The physical grazing-sheath model and its
underresolution remain open, separate from smooth prescribed-BC verification.
Next: Q08 engineering consolidation and a representative memory/runtime pilot,
then remaining term-resolved static coverage using existing C3 maps. The five
material fields have accepted C3 global evidence; current/phi, vorticity and
C3 diffusion transfer still have bounded evidence. Close the applicable
numerical coverage gaps explicitly before full RHS qualification; physical
wall/SAT, traced Q04 evolution and Q09 production gates remain open.

Q08's [code audit and shared extraction plan](q08_shared_extraction_plan.md)
are now complete (2 October). The first implementation slice is common
polynomial primitives plus paired Q preparation; subsequent compact storage
and batched application reuse P infrastructure while retaining Q's support,
five eta planes, 35-node wall maps and separate material/diffusion coefficients.
No replacement boundary lift or numerical policy is part of this extraction.

## Current integration interface and evidence

`native/q_parallel_current_phi.py` is an explicit research-only raw-action
adapter: physical current divergence is the supplied homogeneous action plus
one supplied affine lift; the vorticity coefficient B²/n is applied before
owner projection. The caller owns the current BC/endpoint producer and its
normalization. The paired Ti compensation replaces the material Ti column;
it is not added a second time to the existing five-field action.

The [bounded C3 integration audit](../../../../work/q07_current_phi_audit_20261001/report.md)
passes assembly, live raw-input derivatives, complete-owner projection and
bounded vorticity/diffusion background transfer. New-map work probes retain the
accepted non-adjointness exception; they do not reproduce the old production
reference pair. Normal-data reconstruction lifts and physical-endpoint SAT
lifts are distinct contracts. Physical sheath-current endpoint integration
remains open because the saved short legs never cross the wall. No current-pair
selector or production RHS has been changed.

The [option-1 wall-current audit](../../../../work/q07_option1_wall_current_20261001/report.md)
confirms query-local primitive wall targets reproduce the selected canonical
physical-boundary-state current on conducting-sheath/no-flow HSX patches.
The existing Dirichlet lift nevertheless extends an unresolved wall-target
branch discrepancy into interior caps. A wall-radial-weighted lift is a
bounded diagnostic candidate, not an accepted replacement. The
[paired lift test](../../../../work/q07_radial_lift_test_20261001/report.md)
passes wall/polynomial/derivative/live-response checks but increases smooth and
wave N-O errors. Changing only Dirichlet velocities also worsens density and
temperature transport. Retain the accepted lift; investigate a branch-consistent
physical extension before introducing a separate current correction. Coupled
MPE and physical wall-power qualification remain open.

The [branch-extension comparison](../../../../work/q07_branch_extensions_20261001/report.md)
identifies smooth branch continuation plus a radially weighted jump as the
leading research candidate: `I(f-S_nodes)+S_query+ell*(T_query-S_query)`, with
one continuation sign per stencil and the physical query sign retained in T.
It replays the old smooth-data lift exactly and reduces sheath cap excursions.
The tested same-branch support alternative has rank/admissibility failures and
worse wave accuracy. The split lift still has neighboring-patch mismatch.
The subsequent [compatible-boundary audit](../../../../work/q07_split_compatible_20261001/report.md)
passes wall-law, derivative and current/phi assembly checks but does not support
accuracy qualification at N32/N48/N64. Its deliberately demanding grazing
continuation has large N-O errors, principally from transverse reconstruction;
O-R is much smaller. Fixed physical queries retain the problem. Exact transverse
data greatly reduce the error, and a local width diagnostic identifies eta
underresolution. This manufactured continuation is not the unique physical
sheath solution and has no continuous limit at exact wall grazing.

The [anisotropic sampling and grazing-limit audit](../../../../work/q07_split_eta_sampling_20261001/report.md)
confirms that additional eta information improves N64 current N-O RMS
7.895 -> 0.559 on unchanged traced stencils; O-R remains 0.297. This diagnostic
uses new exact manufactured observations, not interpolation of existing owners.
However, a separate same-patch one-sided limit establishes an interior jump
`ell(u)*(T_plus-T_minus)` wherever the projected wall sign changes. At N64,
u=.97/.98, ion jumps of 0.236/0.200 persist for a smooth compatible interior
field. Finer eta sampling alone cannot remove that jump at fixed radial grid.
It vanishes at radial donor centers, which can hide it in center-only checks.
Keep the smooth-data operators; the physical extension must next address
interior continuity while preserving the actual signed wall law and reporting
any off-node boundary approximation. Neither candidate is promoted; live wall
target production, wall work, coupled MPE and evolution remain open.

The [wall-observation/smoothing comparison](../../../../work/q07_wall_observations_20261001/report.md)
removes the split lift's hard-sign interior jump by interpolating wall data,
but does not justify promotion: off-node wall errors are order unity and can
reverse the parallel contribution to normal ion flow. Hard N64 current N-O is
nearly unchanged (7.895 -> 7.884). Explicit tanh smoothing also makes the split
lift continuous while preserving the smoothed target exactly at arbitrary wall
queries; its lower absolute errors do not establish relative accuracy. Widths
0.5°/1°/2° modify constant-state projected HSX particle-flux estimates by
1.46%/5.83%/19.14%, so the physical smoothing model and angular resolution must
be evaluated together. GBS-style here means velocity smoothing only, not the
full coupled MPE boundary set. Keep exact arbitrary-query wall enforcement as
a design requirement; neither smoothing width nor physical-sheath candidate is
selected. Accepted smooth D/N gates and production operators are unchanged.

The subsequent [bounded eta-only refinement](../../../../work/q07_eta_refinement_20261001/report.md)
shows that smooth-target wall-observation errors strongly improve with genuine
finer plane data: at Nr=Ntheta=64 and 1° smoothing, fixed-cap N-O drops
4.032 -> 1.536 -> 0.518 -> 0.145 for Neta=64/128/256/512; wall RMS drops
0.842 -> 0.455 -> 0.103 -> 0.0453. This keeps observations viable as a research
comparison, but the smoothed split remains the primary baseline because it
preserves exact arbitrary-query wall enforcement with comparable cap accuracy.
Phase and actual refined-grid samples retain larger errors; hard-sign wall data
remain problematic. Directional controls identify joint eta/poloidal/radial
limits, so neither 128 nor 512 is declared a universal minimum or a qualified
physical-sheath resolution. Next bounded work should cross eta and poloidal
refinement, retaining both phases, before a global campaign. No physical
smoothing width, production lift or coupled MPE/current-phi closure is selected.

The [joint eta/poloidal refinement](../../../../work/q07_joint_angular_20261001/report.md)
has now completed with Nr=64, unchanged donor counts/degrees, both eta phases
and additional theta-phase controls. At 1°/Neta512, fixed-cap observation N-O
for Ntheta64/128/256 is 0.1446/0.0584/0.0583 at primary phase, and
0.3048/0.2782/0.2788 at half eta phase. Agreement with the earlier exact-theta
controls indicates little further benefit from poloidal refinement on these
samples. Actual refined-grid current-divergence N-R remains 6.7–17.1% at
(256,512), overwhelmingly N-O; both lifts are nearly identical there.
Off-node observation wall RMS is 0.0087–0.0367, while split preserves the
chosen target to roundoff. Retain smoothed split as primary research baseline;
no physical smoothing width or minimum resolution is selected. Next isolate
eta versus radial reconstruction at the moved-grid hotspot using exact-eta
and exact-both-angular diagnostic controls, before another global campaign.
All gates pass; this bounded exact-target audit does not close live sheath
production, coupled MPE, current-phi/SAT or evolution.

The [actual-grid exact-angular audit](../../../../work/q07_grid_angular_oracle_20261001/report.md)
now identifies eta reconstruction as the leading error for the 1° case at
Nr64/Ntheta128/Neta512. Across eta phases, N-O 0.6612/0.2103 becomes
0.0450/0.0545 with exact eta; exact theta alone barely changes it. Both exact
directions leave the independently verified radial residual 0.0373/0.0308.
The large error is already present in the reconstructed derivative at the
grid center; changing the short trace span would not cure it. Both lifts give
the same mechanism, so retain smoothed split and its exact wall enforcement.
The user subsequently rejected hundreds/thousands of eta planes as a practical
solution. The next audit is production-resolution feasibility at Neta32/48/64,
with phase controls and fixed donor counts/degrees: compare smoothing's accuracy
benefit against changes to prescribed particle flux/current. Do not choose a
width from favorable numerical errors alone. No live physical boundary or full
Q07 qualification is added; no smoothing width or universal minimum eta
resolution is selected.

The [production-resolution feasibility audit](../../../../work/q07_production_feasibility_20261002/report.md)
now shows that widening incidence smoothing alone does not qualify this
manufactured sheath continuation at Neta32/48/64. At Neta64, 1 degree split
N-R is 99–111% relative, dominated by N-O; 4 degrees remains 96–124% while
reducing sampled prescribed projected flux by about 42%. Both rings/families
are affected and Ntheta128 does not rescue accuracy. Split wall values remain
exact and are preferable to the off-node errors/inward contributions of the
observation-only comparison. No physical width is selected. Keep the accepted
smooth D/N operators; next address representation of the sheath angular/interior
variation at affordable resolution with held-out continuation controls. The
physical interior is not uniquely set by the wall law; no evolved-error claim
or full Q07 closure follows. The C3 magnetic evaluator is committed as opt-in
8dadf696; the package default remains unchanged by explicit user direction.

## Frozen inherited choices

Use the traced scalar G and geometry-consistent material tube D on compact C3,
total inner cap separation h/32 and outer characteristic separation h/16, common
five-plane eta reconstruction, frozen support/repair choices, physical-volume
owner observations and complete raw-member projection. Reuse verified C3 RK4-64
endpoints and checked prepared rows; old-evaluator traces are not C3 inputs. Leave the accepted cap-gradient
diffusion and its span evidence unchanged. Do not form owner-field D(Gf).
Span/support selection and whole-RHS performance consolidation remain Q08.

## Term inventory and sequence

The current term-resolved definitions are read from
`native/fci_drb_EB_rhs.py`: `parallel_derived_state_traces`,
`parallel_characteristic_matrix`, and the stage-parallel/RHS assembly.
The primitive material state order is `(n, Te, Ti, Vi, Ve)`. Density is carried
by **electron velocity** in the current model. Set `j = n*(Vi-Ve)`,
`Pe = n*Te`, `P = n*(Te+tau*Ti)` and `mu = mi/me`.

The following centered continuum material terms guide the staged implementation;
existing production-path characteristic residuals rearrange them as coupled
blocks and must be compared term-by-term before transfer, not added twice.

| Field | Centered parallel contribution / subsequent block |
|---|---|
| n | `-D(n*Ve)` — first bounded assignment |
| Te | `-Ve*G(Te) + 2*Te/(3*n)*(0.71*D(j) - n*D(Ve))` |
| Ti | `-Vi*G(Ti) + 2*Ti/(3*n)*(D(j) - n*D(Vi))` |
| Vi | `-Vi*G(Vi) - G(P)/n` |
| Ve | `-Ve*G(Ve) - mu*G(Pe)/n - 0.71*mu*G(Te)`, with electrostatic, collisional and generalized-potential contributions audited separately |
| vorticity | `-Vi*G(omega)` plus the specialized current-divergence/electrostatic block; preserve current–phi/SAT compatibility |

These are term definitions, not a decision to multiply separately projected
owner outputs in every future block. Before implementing each nonlinear term,
freeze its raw/slot product and coefficient placement and then its complete-owner
projection. The generic Q06 G/D are not weighted adjoints and do not have an exact
discrete product rule. Their acceptance does not permit substituting them for
the specialized compatible current–phi construction.

Subsequent Q07 work adds the existing characteristic correction using its
coupled eigensystem, then temperature/velocity and vorticity/current blocks,
and accepted diffusion in existing thermal/viscous channels. No new transport
physics, arbitrary scalar wave speed, coupled solve or time evolution is part
of the first assignment.

## Selected density product contract

For each raw member r of owner I, reconstruct each primitive with its own
prescribed BC data at the same slots s = minus cap, plus cap, midpoint:

```
n[r,s]  = R_kind_n[n_owner, boundary_n](r,s)
Ve[r,s] = R_kind_Ve[Ve_owner, boundary_Ve](r,s)
F[r,s]  = n[r,s] * Ve[r,s]
raw_rhs[r] = -sum_s L[r,s] * F[r,s]
rhs[I] = sum_{r in I} V[r]/V[I] * raw_rhs[r]
```

The qualified material block uses the geometry-consistent center correction
`D_bal(F) = D_old(F) + [div(b)_0 - D_old(1)]*F0`. Equivalently,
`D_bal(F) = B0*b0^eta*((F_plus-F0)/B_plus - (F_minus-F0)/B_minus)/(h/32) + div(b)_0*F0`.
Constant scalar F gives `div(b)_0*F`; no div(B)=0 assumption is made.
This is the accepted Q07 material weight, not a silent replacement of the
separately frozen Q06 divergence or diffusion contract. Velocity zero with
compatible zero boundary data gives zero transport exactly.

The selected order is primitive reconstruction, slot multiplication, raw action,
owner projection. Neither `D(n_owner*Ve_owner)` nor a discrete product-rule split
replaces it. Products of owner averages lose within-owner covariance in
agglomerates, and multiplication and interpolation also do not commute in the
bulk. The bounded harness reports the former as an unselected diagnostic, not
a candidate tuned to the observed MMS errors.

Dirichlet wall values and physical-normal Neumann derivatives are prescribed
independently for n and Ve; mixed D/N choices are supported. Normal derivatives
use the saved physical-normal contraction, not radial derivatives alone.
Reconstructing the primitives first supplies the boundary-conditioned product
without demanding an independently prescribed product value or normal derivative.
For comparison, a product-based fit would need
`normal(n*Ve) = Ve*normal(n) + n*normal(Ve)` and wall values of both factors;
Neumann data alone do not supply those values. Exact MMS wall values are used
only in the explicitly marked product-owner diagnostic.

No limiter, positivity clipping or characteristic dissipation is applied in this
smooth centered block. Positive manufactured density and reconstructed slot
minima are checked. Separate protection/upwind checks remain necessary before
transport-block qualification. Exterior ghost behavior remains unexercised.

## Shared implementation

- `stencils/q_parallel_divergence.prepare_scalar_slots` exposes the existing
  cap/center value rows without a kappa audit, new geometry evaluation or fitting.
- `stencils/q_parallel_transport.prepare_density_transport` captures that shared
  scalar view and unchanged magnetic weights with an identity covering both.
- `native/q_parallel_transport` stages the arrays once, applies primitive BCs,
  multiplies slot values and returns raw or complete-owner density RHS.
- `scripts/q07_density_bounded.py` supplies independent analytic states,
  midpoint references and bounded comparisons; the package imports no MMS code.

Runtime inputs are equal-shaped `(..., n_owner)` primitive states and their
live `QBoundaryData`; BC kinds are static for JIT. Runtime is pure JAX and
supports differentiation of both states and prescribed data. Host prepared
rows and weights remain fixed. Five-field characteristic splitting is not
claimed by this interface.

## Bounded verification and limits

Use the 21 pre-existing complete-owner Q05 sites across actual HSX N32/N48/N64:
core/agglomerated, repair/transition, ordinary bulk and wall. Eight pairs include
constant flux, constant density, constant velocity, smooth coupled fields,
nonpolynomial transverse/eta waves, sign-changing velocity, a radial envelope
and zero velocity. Four primitive BC combinations are DD, NN, DN and ND.
All positive density profiles are >=0.85 analytically, with no limiter needed.

N is the implemented action above. O substitutes exact primitive values at the
same slots, multiplies there, and uses the identical L and owner projection.
R is the independently differentiated continuum midpoint expression
`-[Ve*b.grad(n) + n*b.grad(Ve) + n*Ve*div(b)]`, with identical raw projection.
The existing three-step physical `div(b)` diagnostic tracks reference sensitivity.
Report N-O, O-R and N-R separately, zero/constant controls, positive slot minima,
BC effects, selected support identities and artifact/source/input receipts.

Portable actual-HSX tests additionally verify explicit action assembly, complete
projection, JIT, state-and-BC JVPs against bilinearity and finite differences,
physical-normal boundary response, ignored off-wall data, shape errors, span
rejection and identity changes. These algebra tests are distinct from MMS.

The fixed site selections differ across resolutions: pooled trends are bounded
screening, not global orders. Local volume-weighted residuals are not global
conservation sums. No global campaign, production promotion, Q07 closure or
new physical wall law is authorized by this first implementation step.

## First bounded result and next decision

[Reviewed evidence](../../../../work/q07_density_bounded_20260930/parent_review.md):
26 focused tests and 672 bounded field/BC/site records pass implementation,
identity and finite-value checks. Runtime replay <=1.180e-13, constant-factor
reduction <=6.697e-13, zero velocity exactly zero, minimum density 0.85354.
The smoother controls have maximum N-O errors 3.592e-5 / 8.128e-6 / 5.461e-6.
New short-wave controls expose larger wall errors, especially with Neumann
velocity; the worst N64 N-O magnitude is 0.15684 and is dominated by primitive
velocity reconstruction, not the nonlinear cross term. The chosen wall waves
have about 2–4 angular samples per local cycle. These differing site selections
are not convergence orders or a global gate.

The subsequent fixed-policy matched-location comparison and correction-only
prototype are recorded below. No donor, degree or accepted G/D operator change
is justified by the initial bounded result alone.


## Matched-location wavelength result

[Evidence](../../../../work/q07_density_matched_20260930/report.md): 2,592 records,
27 complete owners, three fixed logical angular anchors, last-aggregate / first-
outer / wall rings, all four primitive BC combinations. The predeclared logical
wavelength ladder 1.4/0.7/0.5/0.35 uses two orientations and two phases; all
original fields are retained. Selected centers are nearest available owners and
corresponding interface rings, not exactly coincident physical points.

Every ladder group and smooth-control group has decreasing pooled N-O RMS and
maxima over both refinement intervals in all three regions. Wall RMS at lambda
1.4 is 1.074e-3 / 1.277e-4 / 2.239e-5; at lambda 0.35 it is
1.087 / 0.3569 / 0.08951. Thus the earlier pooled rebound depended on disparate
sample locations, but short waves remain much less accurate. Individual phase/
BC rebounds remain and are enumerated in analysis.json; this is not a global
order certificate. Accepted support/repair choices and all endpoints are reused.
Assembly/decomposition max 2.160e-14; minimum density 0.85211. The check took
14.4 seconds and 0.780 GiB peak RSS. Continue without reconstruction retuning.

## Characteristic correction prototype: selected bounded construction

[Evidence](../../../../work/q07_characteristic_bounded_20260930/report.md).
The reusable prototype is `native/q_parallel_characteristic.py`; it returns
only the correction and validity flags. It is not added automatically to the
density API or any production RHS. No full material-block qualification yet.

Use **the active production DAE-reduced matrix** from
`fci_parallel_production_flux.parallel_matrix_from_state`, not the older helper
in `fci_drb_EB_rhs`. The active matrix contains `A[Ve,Ti] = mu*tau` and is
associated with `psi = phi + tau*Ti`. Before full coupling, preserve its matched
generalized-potential force and the specialized current–phi/SAT treatment; the
continuum cancellation does not authorize mixing unrelated discrete G operators.
The earlier term table describes centered continuum terms before this rearrangement.

At each raw center, form one common five-point field-line stencil from the
existing h/32 caps (eta offsets +/-delta), center, and h/16 caps (+/-2delta),
where delta = deta/64. Check identical center, owner projection and frozen
support/repair choices across the two prepared span views. The saved scalar
values are point evaluations, so do not apply the old cell-average face-lift
formula to them. This prototype instead uses second-order one-sided point
formulas along eta:

```
d_minus = (3*U0 - 4*U_minus + U_minus2)/(2*delta)
d_plus  = (-3*U0 + 4*U_plus - U_plus2)/(2*delta)
d_center = (U_plus - U_minus)/(2*delta)
A_eta_plus, A_eta_minus = split(b0^eta * A(U0))
correction = -A_eta_plus*(d_minus-d_center)
             -A_eta_minus*(d_plus-d_center)
```

Splitting the oriented matrix handles either sign of b^eta. Subtracting the
centered part prevents adding the material principal action twice. There is no
new div(b) source or electrostatic force in this correction. Center-state
linearization and these point derivatives are explicit prototype choices,
not a claim to reproduce the old canonical-face scheme bitwise. No accepted
G/D or diffusion span default changes; the h/16 values are additional upwind
stencil points. The corrected full material block still requires qualification.

The existing splitter freezes its eigenbasis for differentiation, retains live
matrix multiplication, and reports an inadmissible-eigensystem fallback to a
matrix-norm Rusanov split. This fallback must be reported, not hidden or confused
with the normal coupled characteristic action. Positivity/finite flags do not
clip inputs or certify a positivity-preserving time update.

Bounded actual-HSX checks: 21 complete owners, D/N data, four five-field states,
tau=1 and mu=1836; 168 records. No eigensystem fallback, no nonpositive
thermodynamic state, no new tracing. Correction N-O for smooth controls has
maxima 4.266e-5 / 8.817e-6 / 5.542e-6; the density-component maxima are
3.838e-8 / 1.564e-9 / 6.482e-10. Short-wave sensitivity remains. Exact-value
correction O is nonzero and can be substantial in the stiff electron row;
it must be assessed against the total material RHS, not mislabeled as N-O.
Reconstructed constants leave at most 8.688e-9 projected roundoff after stiff
coefficient amplification; no artificial zeroing was introduced. Algebra
controls cover constant/linear/quadratic cancellation, independent eigensystem
assembly, orientation reversal, cubic step-squared scaling, JIT and JVP with
center/eigenbasis fixed. Those are distinct from real-HSX numerical evidence.

## Centered five-field assembly and combined-action audit

Implemented in `stencils/q_parallel_material.py` and
`native/q_parallel_material.py`. Checked paired h/32 and h/16 sources must agree
on source/geometry/trace identity, complete-owner projection, raw centers and
frozen repair choices. Runtime accepts five independent D/N kinds and a separate
prescribed phi state/BC. Primitive fields are reconstructed once per span;
products are evaluated at scalar slots, coefficients at the inner raw center,
and the full nonlinear raw action is projected only afterwards.

The term table above is now implemented. The material electron row additionally
contains `-mu*tau*G(Ti)` for the active DAE convention. Its separate force is
`mu*(G(phi)+tau*G(Ti))`, sharing exactly the same `G(Ti)` scalar values and weights.
This avoids assuming that differently boundary-conditioned independent psi and
Ti fits commute. The centered sum recovers the stated physical electron terms.
`MaterialAction` exposes material, generalized force, centered, correction and
combined arrays, with raw positivity/finite/eigensystem flags. No clipping or
implicit solve is added. The caller supplies live prescribed phi; specialized
current–phi/SAT production integration remains open.

The characteristic correction adds only the difference between one-sided and
centered principal actions. The centered flux-product remainder is retained:
there is no exact discrete product rule and no claim that this sum equals a
pure principal-matrix upwind operator. Geometry sources and phi force are not
added again by the correction. Separate owner-field D(Gf) remains deferred.

[Bounded evidence](../../../../work/q07_material_bounded_20260930/report.md):
21 complete actual-HSX owners, four five-field MMS sets, uniform D/N and two
mixed primitive/phi combinations, three actions; 1,008 records. 41 focused tests
pass, including active principal symbol/signs, geometry sources, Ti cancellation,
phi force exactly once, mixed BC mapping, density replay, complete projection,
JIT and centered live-state/BC JVPs. Correction JVP uses the inherited frozen
spectral-basis contract. Density replay is exactly zero; decomposition residual
5.685e-14; no spectral fallback. Continuum div(b) step sensitivity contributes
at most 8.761e-11. The warmed campaign took 15.6 seconds and 1.067 GiB peak RSS.

Smooth centered N-O maxima are 0.02096 / 0.004732 / 0.003172; combined maxima
0.02100 / 0.004741 / 0.003177. Reconstruction remains promising on these samples.
However smooth combined N-R maxima (0.3470 / 0.08986 / 0.10385) exceed centered
N-R (0.04730 / 0.005665 / 0.01098). The exact-value characteristic correction,
especially its stiff electron component, causes the main increase; it is not a
reference bug or duplicated material term. Short-wave lambda=0.35 combined
N-O maxima are 583 / 200 / 429, reflecting severe electron-row reconstruction
sensitivity at the sampled walls, including mixed BCs. Different sites preclude
global orders. The error at the worst smooth N64 site is about 0.070% of that
site's continuum electron RHS; this local ratio is not a global relative norm.
Constant correction is analytically zero, while reconstructed roundoff yields
up to 1.621e-8 combined electron residual; no zeroing or gate tuning occurred.

**Next bounded decision:** repeat the full material comparison at corresponding
locations with the wavelength/orientation/phase ladder; separate electron
pressure, thermal and phi contributions and normalize by component. Audit the
frozen-center characteristic mode response to distinguish dissipation/dispersion
from reconstruction. Decide whether this correction is accurate enough before
a global material campaign, then implement vorticity and specialized current/phi
integration and remaining thermal/viscous channels. Q07 remains in progress;
no full material accuracy, evolved stability, exterior crossing or production
qualification is inferred from these implementation checks.

## Remaining terms: bounded vorticity and diffusion channel implementation

`native/q_parallel_vorticity.py` now supplies centered `-Vi*G(omega)` and a
separate scalar second-order point upwind correction using `b_eta*Vi` at the
raw center. It shares checked h/32+h/16 scalar views, applies independent omega
and Vi BCs, then projects complete raw owners. No tube-compression/current
source is included in this advective term.

`native/q_parallel_channels.py` applies the unchanged frozen diffusion to the
six existing channels `(density,Te,Ti,Vi,Ve,vorticity)`, scaled by their existing
constant-in-space parameters. Those parameters already define the respective
parallel diffusivity/thermal diffusivity/viscosity: there is no additional
`2/(3n)`, mass ratio or temperature power. Both accepted diffusion spans remain
available; no default is selected. State, prescribed BC and scalar coefficients
remain differentiable; negative/nonfinite coefficients are flagged, not clipped.

A separate `stencils/q_parallel_channels.py`/native coefficient-slot interface
exposes `L_s*[chi_s*(b.grad f)_s]` with explicit prescribed live chi values at
each cap and center. It reuses directly reconstructed cap gradients, not
owner-field D(Gf). It does not select a physical temperature-dependent closure
or implement reconstruction of chi from evolved owners.

[Bounded evidence](../../../../work/q07_remaining_bounded_20260930/report.md):
52 focused Q tests pass, including 11 new tests. 1,680 records on the 21 frozen
HSX complete-owner sites cover four field sets, both diffusion spans and D/N;
advection also covers mixed BCs. Constant channel scaling replay is exact;
unit chi agrees with accepted diffusion within 3.826e-12. Largest constant
action is 1.029e-11. Three-step continuum diffusion reference sensitivity is
1.709e-9. Runtime 16.5 seconds, peak RSS 1.030 GiB. Smooth h/32 vorticity centered
N-O maxima are 1.605e-6 / 3.640e-7 / 2.428e-7. Constant-channel smooth N-O
maxima are 1.580e-5 / 8.672e-7 / 3.893e-7. These are sample trends, not global
orders; short-wave/local rebound cases remain. Variable chi was supplied
analytically, so that extension isolates coefficient-slot assembly only.

The [current/phi integration inventory](../../../../work/q07_remaining_bounded_20260930/current_phi_integration.md)
records the remaining paired-operator/affine-lift contract. The old reference
pair is adjoint, but the existing live generalized-potential force is a different
map; the opt-in live-adjoint prototype does not yet certify physical wall power.
The new prescribed-phi material block is not already connected to either one.
Three captured regression replays were attempted but their saved NPZ arrays are
missing; retained reports/source are context only, not a fresh numerical pass.
No generic Q operator is substituted, no production pairing is promoted.

The new Q worker owns the matched material/characteristic audit in
`work/q07_material_matched_audit_20260930`. Parent next work is a bounded
actual-HSX current/phi integration audit with explicit homogeneous/affine split,
physical masses and traces, coefficient-before-projection placement and common
Ti/psi force balance. Restore matching prior captures or produce an explicitly
scoped new actual-HSX capture; do not assume the missing reference-pair replay
has passed. Full Q07/global/evolved qualification remains open.

## Parent review: matched material and characteristic response

The [worker audit](../../../../work/q07_material_matched_audit_20260930/report.md)
is complete and its artifact hashes were checked by the parent. 5,832 action
records and independent validation pass; density replay is exact, zero fallback.
Across the 27 corresponding owners, all nonconstant pooled per-field N-O groups
improve over both refinement intervals. This supports keeping the current
reconstruction and characteristic correction unchanged as candidates.

Smooth N64 wall electron combined relative N-R is 0.1098%, while centered-only
is about 0.01155%. The difference is predominantly the exact correction's
second-order phase/truncation contribution, with mass-ratio amplification;
Ti compensation and force assembly are correct. Both ideal orientations damp,
and sampled fixed-phi coupled eigenvalues have no positive growth beyond
roundoff. These are local semidiscrete checks, not a global evolved stability
proof. The short-wave lambda=0.35 N64 wall error is still 9.068% for Ve and
4.77–5.81% for the other fields, dominated by reconstruction. Do not hide these
controls in a pooled smooth-field pass. The smooth O-R rebound persists and
is not automatically a reference error.

Proceed to engineering term-resolved global static comparisons for the assembled
material/vorticity/constant-diffusion blocks, retaining centered and corrected
outputs separately with prescribed phi. In parallel, complete the separate
current/phi integration contract/audit before declaring full Q07 closure. No
further donor/correction tuning is justified by this audit. Eta reconstruction
response, support count and h/n selection remain the planned Q08 decisions;
the donor transfer can dominate phase error even when the evaluated span is short.

## Local global campaign prepared — 30 September 2026

The [checkpointed runner](../../../scripts/q07_material_global/README.md) freezes
the unchanged 18-state matched catalogue, four uniform/mixed D/N combinations,
44 term actions and nine reporting regions. Material uses h/32 with paired
h/16 outer upwind points; constant diffusion retains both spans separately.
Preflight replays all 27 matched owners plus six fresh core/bulk owners and
passes within 5.87e-12 for material, 4.23e-10 for accepted diffusion. 36 focused
tests pass. The pilot supports an authorized local two-worker launch, estimated
2.18–3.42 hours, with 0.46 GiB checkpoint output. Source/input identity and
receipts are in `work/q07_material_global_20260930`; frozen identity is
`80d3d8c880f6baf6c8423098e26b0714877f03526d4e60ad0331632567be0554`.
Results remain pending. Checkpoint statistics cover every owner and preserve
physical-volume RMS, signed integrals, maxima and maximizing owner IDs for
N-O/O-R/N-R. Short waves are stress controls; scientific regressions do not
trigger source or tolerance tuning. Current–phi/SAT compatibility, exterior
crossings and evolved stability remain separate open work.


## Manufactured continuation scope — 2 October 2026

The [nine-profile continuation audit](../../../../work/q07_continuation_audit_20261002/report.md)
confirms wall compatibility/derivative checks and shows that the large Neta64
velocity/current derivative error persists across different compatible radial
extensions, including no radial broadening. Same wall targets and saved physical
caps, unchanged split reconstruction; bounded samples only. At 1° the profiles
retain 85.9–107.2% relative current N-R; exact eta removes 98.7–99.4% of N-O RMS.
The surviving radial error does depend on the continuation. Donor Neta512
improves all profiles but does not qualify them; fixed caps/span distinguish
this information test from native-grid refinement. No reference profile is
claimed to be a self-consistent physical presheath solution. No source/default
or accepted operator gate changes.

Keep the whole-wall physical presheath-model validity question open, especially
at shallow incidence. A known-angular-factor representation was identified as
an untested design lead, but the user subsequently chose to retain the general
reconstruction and move on. Enrichment and a general Fourier/Zernike basis
search are deferred. The failed continuation controls remain evidence of a
resolution limitation for this family, not a qualified physical sheath model.
