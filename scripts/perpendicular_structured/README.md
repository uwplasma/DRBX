# Shared structured point reconstruction

`reconstruction.py` extends the tracked P07 structured tensor/ringwise/coupled
quartic construction to point values and logical gradients, including raw-cell
quadrature. It is research infrastructure, not a production promotion.

`load_context(n,input_root)` loads canonical geometry. Construct
`StructuredReconstruction(t)` then `rows(key,points,location='face'|'cell')`.
Face keys are `(axis,i,j,k)`; cell keys `(i,j,k)`. Points are `(Q,3)` logical
coordinates. Returned `PointRows` contains owner `donor_ids`, `value(Q,D)`,
`gradient(Q,3,D)`, separate prescribed-trace coordinates and support diagnostics.
`rows.apply(owner_values,trace)` accepts `(owners,fields)` and a callback
`trace(points)->(values(Q,fields), gradients(Q,3,fields))`. Only tangential trace
derivatives are used. Owner observations are raw physical-volume-weighted
member-center samples. No manufactured field values enter coefficient assembly.

The wall lift is g(theta,eta) constant in radius: subtract g at donor coordinates,
reconstruct the residual with the prescribed zero wall node, then add g and its
tangential derivatives at targets. Nonzero radial wall derivatives are recovered
from interior data. Compatible Neumann fields evaluated with their analytic trace
are consistency controls, not qualification of a separate Neumann closure.

Coupled rows use the P07 bounded cubic-then-quartic support expansion and SVD
cutoff, now checking every requested point value/gradient functional. This is a
stronger target than P07's integrated gradient functional; unsupported rows fail
without substitution. No field-dependent support selection is permitted.

`side_rows(key,points)` returns adjacent-cell anchored structured reconstructions
(left,right); an exterior radial state is None. Angular and eta support selection
is anchored at each adjacent cell center. The common rows stay unchanged. This
is a new structured two-state correction, not the old selection-v3 biased WLS.
Each campaign must separately test and qualify its upwind/characteristic action.
The physical-wall exterior state is supplied directly by the prescribed trace.
Collapsed-axis flux is exactly zero and needs no singular metric query.
