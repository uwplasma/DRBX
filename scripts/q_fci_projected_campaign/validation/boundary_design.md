# Geometry-only dyadic interval reduction near the wall

Authorized by the user after the fixed-span N32 coverage check found 115 invalid
legs among 13,800 tested legs. One bounded numerical candidate, not a sweep.
For each physical face quadrature point start with span H=2*pi/N and the four
endpoint legs -H/2,+H/2,-H/4,+H/4. If any leg leaves 0<r<1, halve the span
for that point alone, reusing the old half legs as new full legs and computing
only the new half legs. At most eight halvings. Bad/nonpositive b_eta is a hard
failure, never a reason to shrink. Freeze the per-node span from geometry only.
No field values enter that decision. Retain 64 RK4 substeps per distinct leg.

Use the unchanged endpoint OwnerMoments/four-eta-plane cubic transfer at every
accepted endpoint. D4=(4 D_half-D_full)/3 with each point's actual full/half
span; the face projection fits the physical along-line derivative with the
unchanged physical-area P3 projection. Prescribed wall flux is the same zero
integrated flux for the frozen fourth-flat MMS fields. No imposed scalar wall
value, exact interior endpoint, extrapolation outside geometry, omitted seed,
or fallback to the old return is allowed. Unshortened points reproduce the
original candidate. Changes in line spacing are a new explicit boundary-adjacent
policy requiring this bounded comparison before campaign promotion.

Six fixed wall/inward owner pairs at N32/48/64: theta/eta nearest to (0,0)
and to (35.5*2*pi/48,20.5*2*pi/48), radial indices N-1 and N-2. Resolve periodic
ties by smallest raw index. Enumerate all their incident canonical faces from
actual topology. Score both complete owners and preserve outer increments.
Original four fields including constant. q5 numerical candidate, exact endpoint
and exact-gradient projection channels; q9/q11 exact face references. q7
candidate on all incident faces. Compare old archived return if a matching
trusted baseline is directly available, otherwise compare the pinned direct
cubic gradient diagnostic and absolute reference errors without inventing a
baseline. Report that limit. Also compare adaptive fourth/full/half on the same
accepted per-point spans. No claim of observed mesh order for moving cells.

Freeze all six q5/q7 maps before candidate error evaluation. Store invalid
original counts, accepted span levels, every trace and map identity, cubic
endpoint/algebra checks, constants, canonical incidence, sparse replay,
reference and q7 effects, sensitivities, per-owner errors and gradient/transfer
components. Test line formula using a polynomial in eta and an intentional wrong
span denominator. Independent host/128 checks on the shortest accepted node
in each case (both signs and both distances) use the same frozen span. Keep
geometry-query/tracing validation separate from physical model accuracy.

One process, single-thread BLAS, <=800 total CPU seconds including 120 seconds
for report/audit, <=3 GiB RSS and <=250 MiB artifacts. No global run or parameter
search. If invalid legs remain or assembly/algebra fails, stop and report rather
than choosing another method. User priority remains static accuracy, with
sensitivity recorded separately from an energy certificate.
