# Physical-wall and characteristic-face design

This document is the development contract for the plasma-facing boundary of
the drift-reduced Braginskii system. The production target is a warm-ion
magnetic-presheath-entrance (MPE) closure. The development ladder has exactly
four core rungs; electrical and thermal choices are orthogonal policies and
are not additional rungs.

The central numerical decision is:

> The physical wall model constructs an admissible wall target and physical
> face fluxes. The existing live characteristic/Riemann flux combines those
> data with the owner state. One canonical resolved boundary flux is consumed
> by material transport, the current/vorticity SAT, and the local implicit
> solve.

The boundary is therefore a flux interface, not a prescription that solves a
set of arbitrary primitive values by modifying whichever incoming
characteristics happen to be available. A complete target state may be useful
to construct the physical flux, but it is not itself a ghost state and is not
required to satisfy every composite quantity after characteristic splitting.

## Why the previous characteristic-residual design is superseded

At a partially incoming hyperbolic boundary, outgoing waves are supplied by
the plasma and only the missing incoming information is supplied by the wall.
The old implementation instead treated physical wall equations as hard
residuals in the incoming amplitudes. In particular, it attempted to impose
an exact ion Bohm equality while preserving every outgoing mode. This is not
well posed when the live incoming subspace cannot control that residual.

The following mechanisms are explicitly **not** part of the production
design:

* hard incoming-amplitude residual solves for arbitrary wall equations;
* releasing an outgoing characteristic lane to make an overconstrained solve
  square; and
* generic minimum-residual, least-squares, or completion metrics that silently
  violate a physical wall law.

The live eigensystem remains essential for the numerical characteristic flux,
orientation, dissipation, and diagnostics. It does not determine how many
physical equations a wall model is allowed to impose. A zero crossing is
handled by the matrix split and the appropriate physical flux branch, not by
changing the residual dimension.

## Boundary contract

### Physical wall model

A `PhysicalWallLaw` owns the physical closure and returns a fixed-shape,
batched description containing, as applicable:

* an admissible target state or target primitive/flux data;
* particle, momentum, current, and heat fluxes;
* scalar normal values or derivatives for diffusive/elliptic operators;
* the electrical-wall policy and any genuinely independent auxiliary
  variables (for example a conductor potential or sheath drop);
* polarization/vorticity data;
* magnetic incidence, wall orientation, branch, and admissibility metadata;
  and
* derivatives of the physical flux with respect to the owner and independent
  auxiliary variables.

The wall law does not receive a preclassified list of incoming modes and does
not release outgoing modes. It may construct a target using the owner state,
wall geometry, and model parameters, but physical constraints must be encoded
in the returned physical flux/trace contract rather than handed to an
incoming-amplitude completion algorithm.

### Characteristic numerical interface

The numerical boundary adapter evaluates the same stage-local normal
characteristic operator used by the bulk material flux. It combines the owner
state with the wall-model target or flux through the selected stable
characteristic/Riemann (or weak Lax/Rusanov) boundary flux. This automatically
handles incoming, outgoing, and glancing modes, including changes of sign.

The adapter returns one canonical wall-data payload containing:

* the owner-directed material fluctuation/flux;
* the canonical face state when one is defined;
* ion/electron particle fluxes and current;
* pressure, momentum, and heat fluxes required by other operators;
* the live oriented eigenvalues and incoming/outgoing/glancing diagnostics;
* wall-law branch and admissibility information; and
* the total owner Jacobian needed by the local backward-Euler solve.

The current/vorticity SAT, material transport, and local implicit solve must
all consume this same resolved object. No subsystem may reconstruct current,
pressure, or particle flux from an unrelated primitive ghost trace.

Physical wall constraints that are genuinely auxiliary or globally coupled
(such as a single conductor potential) may use a small separate solve. That
solve is owned by the wall model and is not an incoming-characteristic
residual solve.

### Operator-specific reconstruction-support contract

The physical wall face and the support used to reconstruct a nearby plasma
face are separate pieces of the interface.  The wall law first supplies the
operator-specific physical face trace (or normal derivative/flux) on the
masked physical wall face.  That trace is the wall-face law; it is not a
replacement for the plasma-side support cells used by a high-order stencil.

For the compatible Poisson bracket, each of the six advected fields—`density`,
`Te`, `Ti`, `Vi`, `Ve`, and `vorticity`—gets a separate homogeneous
physical-normal Neumann support halo.  That PB-only support halo is used for
the PB gradients, conservative stencils, third-order face states, and the
compatible-vorticity generator route.  It is deliberately distinct from the
production physical halos/traces, which remain the inputs for all non-PB
consumers and for the actual physical wall face.  At the first interior face
next to a physical wall, the PB candidate that would otherwise read its
physical ghost is replaced by the canonical plasma-only candidate

```text
q_face = (2 q_0 + 5 q_1 - q_2) / 6,
```

where `q_0`, `q_1`, and `q_2` are the three adjacent plasma cells.  This is
the exact face value for a quadratic on a uniform coordinate and is
independent of the arbitrary physical ghost value.  The physical wall face
itself continues to use its wall trace and adjacent owner state according to
the selected wall flux law.  Lower and upper wall masks are applied per
tangential column: an unmasked column remains the ordinary high-order
reconstruction, while a masked column receives only the wall-specific
closure.

The electrostatic potential `phi` is the compatible bracket's generator, not
one of those six advected-field support halos.  Its generator action uses the
physical potential closure and the supplied physical `phi` face trace; it does
not receive the homogeneous-Neumann support override used for `density`,
`Te`, `Ti`, `Vi`, `Ve`, and `vorticity`.  This changes only the potential
generator representation; the six advected-field support rules and the wall
laws remain unchanged.

This support rule is operator-specific, with `phi` as the bracket's
electrostatic operand.  In the `material-scalar-vorticity-compatible-upwind`
selector, the same PB support is used by the compatible-vorticity generator
route; vorticity is not given an unrelated analytic wall value.  Separately,
the production curvature path has migrated to operator boundary traces and
canonical face states rather than consuming a strong physical ghost as its
high-order support.

The lower radial axis-regular side is topological, not a physical wall.  Its
axis-regular half-turn/parity fill is left untouched, as are periodic seams
and shard seams.  If an axis has fewer than three owned plasma cells, the
third-order candidate cannot be formed: the adjacent owner states are then a
true first-order fallback.  The reconstruction marker returned by the
operator records that the physical-boundary closure was applied at an
`n >= 3` adjacent face; it must not be interpreted as an order reduction.

This repair changes only the relevant high-order support closure.  The
physical wall face law/trace, parallel material transport and its implicit
short-leg solve, perpendicular diffusion, and polarization/inverse solve
remain on their existing operator-specific paths.  In particular, the
support repair does not alter the wall target, the characteristic incoming
subspace, the diffusion normal closure, or the discrete polarization gauge.

## Four-rung core ladder

### Rung 1 — No-flow verification wall

This is the machinery baseline, not a production plasma-wall model. It uses a
reflecting/no-flow target with

```text
Vi_wall = 0,  Ve_wall = 0
```

and consistent zero normal particle/current fluxes. The purpose is to verify
wall orientation, FCI topology, characteristic flux splitting, SAT wiring,
and IMEX handoff. Any unused incoming freedom is handled by the numerical
flux, not by a hard residual completion policy.

The validated 48^3 short replay remained finite for 32 steps from the staged
restart, with 33 finite frames, final ranges

```text
n      [0.9835355, 1.0302540]
Te     [0.9843459, 1.0067774]
Ti     [0.9898584, 1.0044951]
Vi     [-0.00145265, 0.00111411]
Ve     [-0.3256469, 0.2375784]
omega  [-0.0752558, 0.0647477]
phi    [-0.00548446, 0.00819162]
```

The final maximum residual was `9.36e-8`; no period-two wall packet was
observed. This validates common machinery only and is not evidence that a
physical sheath closure has passed.

### Rung 2 — Simple conducting sheath entrance

This is the first physical rung. It is a simplified conducting sheath
entrance, not yet the magnetic-presheath entrance. The wall model:

* extrapolates `n`, `Te`, and `Ti` from the plasma side;
* uses weak logical Bohm ion outflow, for example
  `u_i^* = max(c_B, u_i_owner)` in the outward orientation;
* prescribes the wall potential `phi_wall` through the selected electrical
  policy;
* obtains the electron loss flux from the exponential sheath response;
* uses zero normal thermodynamic derivatives,
  `d_n Te = d_n Ti = 0`; and
* permits nonzero current when the wall is grounded or biased.

The ion Bohm condition is an outflow/inequality selection in the numerical
flux, not an exact equation to be forced through incoming amplitudes. The
electron response and potential determine the electron flux; `j_n` is then a
physical output. This makes a boundary solution available without assuming
two incoming modes or imposing pointwise `Vi=Ve`.

The compatible initial state must include the electrical/sheath relation as
well as any velocity matching used for startup.  The default prescribed wall
potential is therefore expressed in the simulation's shifted potential gauge:

```text
phi_wall = -Te0 log[sqrt(mu Te0 / (2 pi)) / sqrt(Te0 + tau Ti0)] .
```

With `phi_face=0`, `Te=Te0`, and `Ti=Ti0`, this makes the electron loss speed
equal the Bohm speed and permits the existing `Vi=Ve` compatible startup.  A
user-supplied wall potential represents a different grounded or biased wall
and need not be current-free initially.

### Rung 3 — Simplified GBS warm-ion MPE

This rung keeps the conducting sheath particle/electron model but adds the
coupled warm-ion magnetic-presheath-entrance relations used by the simplified
GBS model. Tangential-gradient corrections are intentionally omitted. The
first implementation uses the prescribed conducting-wall electrical policy
defined below; global floating and circuit coupling remain later orthogonal
extensions.  With ``sigma`` denoting the outward field-line orientation and
``c_B = sqrt(Te + tau Ti)``, the intended simplified physical-normal laws are

```text
D_n Te  = 0,
D_n Ti  = 0,
D_n n   = -sigma (n / c_B) D_n Vi,
D_n phi = -sigma (Te / c_B) D_n Vi.
```

The signs and normalization above are part of the wall-law contract and must
be tested for both wall orientations.  ``D_n Vi`` is evaluated from the
stage-local plasma owner to the physical wall target with the same
metric-aware owner-to-face normal-derivative stencil used by the other
physical Neumann closures.  The coefficients are collocated plasma-side face
values from that same wall-law evaluation.  It must not be reconstructed from
an independently filled velocity ghost or from the dissipative part of the
characteristic numerical flux.

The current primitive physical-normal closure uses the shared face-trace
affine map ``f_face = b_f + r_f g_f``.  Te and Ti use their actual zero
Neumann face values, while gVi is obtained from the weak-Bohm target using
those actual Te/Ti values.  With ``k = sigma*gVi/cB``, the coupled targets are

```text
n_face = b_n / (1 + r_n*k),   g_n = -k*n_face,
g_phi = -Te_face*k,            phi_face = b_phi + r_phi*g_phi.
```

The electron velocity target is then evaluated from the actual ``phi_face``
and ``Te_face`` Maxwellian data. One required topology-prefilled set of five
halos (n, phi, Vi, Te, Ti) is constructed and shared; the old build-refine/
private fallback path has been removed. Invalid faces and singular affine
denominators raise eagerly; no clipping is applied.

The shared metric/topology/actual-trace construction verifies the isolated
physical-face discrete law, including skew metrics and mixed-gradient states.
This does not claim continuum or refinement accuracy, nor independent
prescription at coupled physical corners. The frozen 48³ face-trace
verification is complete. The current face-collocated-``Ve`` replay
(``replay_one_step_dt16_physical_normal_20260905.json``) stopped at
``stopped_nonfinite_updated2`` after its short leg (``41.226`` s), so boundary
correctness is verified but is not a stability fix: there was no completed
step and no ten-step qualification. With ``dt=1.46484375e-5``, stage-1 phi
ranged over ``[-0.4265226197, 0.6707013621]``, with
``lambda=-0.02565367514`` and inversion relative residual
``4.56093e-8``. Stage-2 phi ranged over
``[-5591.8791607, 6026.4007740]``, with ``lambda=943.7905072`` and
relative residual ``1.71888025e-8``; its base density minimum was
``0.4866953364`` before subsequent implicit nonfinite fields.
Focused validation and trajectory qualification remain separate gates; no
production qualification is claimed here.

#### Current implementation audit note

The production isolated physical-face path uses shared Lagrange weights through
``neumann_face_trace_physical_affine``; this construction has no midpoint
fallback.

The new coupled data therefore include:

* density and ion-flow normal-derivative coupling;
* the corresponding normal potential derivative;
* the polarization/vorticity relation propagated through the same discrete
  polarization operator; and
* `d_n Te = d_n Ti = 0` for the simplified thermal closure.

The density, potential, and vorticity conditions are one coupled physical
boundary model. They must not be independently replaced by `phi=0`,
`omega=0`, or unrelated primitive ghosts. In particular, vorticity receives
no independent analytic wall value.  The Rung-3 ``phi`` and ``Ti`` face/halo
closures are inserted into the selected discrete Boussinesq polarization map

```text
omega = L_perp,h(phi + tau Ti).
```

Any wall-facing vorticity trace or source needed by a downstream operator is
derived by this forward discrete map, with the same metric, RLP aggregation,
and boundary stencil used by the inverse solve.  The evolved owner vorticity
remains the volume right-hand side of the inverse problem.  This is what is
meant by propagating vorticity through the polarization operator: it is not a
copy of a continuum generalized-vorticity wall formula and it is not a second
independent vorticity boundary condition.  These derived data and the
material face fluxes must be exported through the same canonical wall-data
contract used in Rung 2.

This is the first rung that closes our evolved five-field model as a
simplified MPE entrance. It is not a separate incoming-characteristic solve;
the live characteristic operator supplies the stable numerical interface for
the physical GBS target and fluxes.

### Rung 4 — Full warm-ion magnetic-presheath entrance

Starting from Rung 3, add the remaining MPE physics:

* tangential density and potential gradients;
* total normal drift, including the adopted `E x B`, diamagnetic, and
  curvature contributions;
* magnetic-incidence dependence through `B dot n`;
* a physically declared grazing/tangent-field branch; and
* localized smoothing of the sign transition near `B dot n = 0`, only after
  the unsmoothed branch is verified.

The same material numerical interface, current SAT, polarization relation,
and implicit Jacobian contract remain in place. Rung 4 enriches the wall law;
it does not reintroduce hard residual solves or a different characteristic
machinery. Heat-transfer, secondary-emission, and inverse-sheath effects are
added only when selected by the orthogonal policies below.

## Orthogonal electrical policies

Electrical choices are policies attached to a rung, not numbered rungs:

* **Prescribed grounded/biased wall:** `phi_wall` is supplied and the local
  sheath response determines a generally nonzero current.
* **Globally floating conductor:** one potential per connected conductor is
  determined from the integrated current condition
  `integral(j_n dA) = 0`. This is distinct from imposing `j_n=0` independently
  at every face.
* **External circuit:** a circuit equation supplies the conductor potential
  and receives the integrated plasma current.
* **Local insulating approximation:** a local sheath-drop/current relation may
  be used only when that approximation is physically intended and has its own
  auxiliary variable. It is not silently attached to every MPE closure.

The electrical policy owns the potential/sheath-drop auxiliary solve and its
Jacobian. It does not alter the number of incoming characteristic lanes or
release an outgoing lane.

For the initial Rung-3 implementation the selected policy is a prescribed
conducting wall.  It imposes no pointwise or integrated zero-current equation.
The potential reference uses the standard equilibrium Maxwellian warm-ion
sheath drop

```text
Delta_phi_sheath,eq =
    Te0 log[sqrt(mu Te0 / (2 pi)) / sqrt(Te0 + tau Ti0)],
g^T phi = phi_wall + Delta_phi_sheath,eq.
```

Here ``g`` is the normalized physical-area-weighted plasma-side wall-face
functional.  The existing shifted convention ``g^T phi=0`` and
``phi_wall=-Delta_phi_sheath,eq`` is exactly the same voltage difference and
may be retained internally, but provenance must identify the convention.
This is a fixed equilibrium calibration for a conducting wall, not a dynamic
local-floating or global-current solve.

### Current/potential affine boundary pairing and wall power

The current/vorticity SAT is an affine boundary lift, not a hard residual
solve and not a tunable damping penalty.  Let ``j = n (Vi - Ve)`` and let
``D_phys`` denote the FCI discretization of the B-weighted parallel current
divergence used by the vorticity equation.  At a material FCI hit, the
resolved sheath state supplies a generally nonzero endpoint current
``j_wall*``.  Therefore the physical map is affine in the interior current:

```text
D_phys(j) = D0(j) + S_sheath .
```

``D0`` is the same FCI current-divergence map with a zero *reference*
current inserted only at physical FCI endpoints; ordinary mapped endpoints
retain a separate homogeneous zero-normal **current support** closure.
Only the physical masks/topology are borrowed from the density BC template;
density's prescribed values and boundary kinds are not current data. This is
essential for MPE, where density has a nonzero physical-normal derivative.
This reference value is not a no-current physical boundary
condition.  ``S_sheath = D_phys - D0`` is the injected contribution of the
resolved sheath current.  The vorticity RHS must use the complete physical
operator ``D_phys``.

The reference map is kept because it is homogeneous and linear.  The
parallel potential-gradient operator is constructed as its physical-volume
weighted negative adjoint,

```text
G0 = - M^-1 D0^T M .
```

``M`` is the diagonal fine-cell physical-volume measure (the raw control
volume under RLP, or the metric volume times volume fraction and cell
spacings otherwise). Consequently this reference pair has an exact
homogeneous Green identity. In the original Rung-2 design, ``G0(phi)`` was the
electrostatic force and ``D_phys(j)`` the current-divergence drive. The live
production generalized-potential force has since changed to a different
support-core/legacy gradient; the reference identity alone does not qualify
that live force. The Rung-3 reconciliation below distinguishes these paths.

The Rung-2 reference operator/polarization condition is ``phi_face = 0``. It is
important not to conflate that trace with the sheath-law ``phi_wall`` used in
the electron-loss exponential.  Changing the latter changes ``j_wall*`` and
therefore ``S_sheath``; it does not create a potential-side lift while
``phi_face`` remains zero.

Physically, ``phi_face`` is the plasma/sheath-entrance-side trace used by the
drift-fluid and polarization operators, whereas ``phi_wall`` is the
material-conductor-side potential used to define the unresolved sheath drop.
They are collocated at one computational boundary hit; the Debye sheath
between them is not spatially resolved.  A grounded or biased electrical
policy fixes the material-side potential, not in general the sheath-entrance
potential.  Independently imposing both values as physical Dirichlet data
would normally overconstrain the sheath drop.

The current Rung-2 choice, ``phi_face = 0`` together with a prescribed or
equilibrium-compatible shifted-gauge ``phi_wall``, is a reduced calibrated
closure.  It must not be interpreted as a self-consistent solution for both
sides of the sheath or as an identification of the two potentials.  Rung 3
is intended to supply the coupled fluid/MPE-entrance conditions, including the
normal-potential relation; a grounded, floating, or circuit electrical policy
still supplies or determines the material-side potential independently.

For nonzero prescribed *operator* potential face data, the corresponding
gradient map is also affine:

```text
G_phys(phi) = G0(phi) + g_wall(phi_face).
```

That policy must export both lifts and account for the corresponding
discrete wall power (with the chosen outward-current sign convention),
``integral(phi_face * j_wall dA)``, in energy diagnostics and, when present,
the circuit or wall-energy model.  This extension must preserve the same
canonical resolved wall-current/flux payload used by material transport, SAT,
and the local implicit solve.

#### Rung-3 generalized-potential reconciliation

Rung 3 supplies a physical-normal potential relation, not the Rung-2 zero
operator-face Dirichlet value. The plasma-side trace generally depends on
the volume state and on the normal derivative. The material conductor
potential in the sheath exponential remains a separate quantity.

The material characteristic block is written after eliminating
``psi = phi + tau Ti``. Its electron row retains ``A[4,2] = mu*tau``;
the external force is ``+mu*G(psi)``. At the continuum principal level this
combines ``-mu*tau*grad(Ti)`` from the material block with the generalized
force. Thus ``G0`` does not intrinsically omit ion-temperature physics: a
linear gradient can act on ``psi`` as well as on ``phi``. Applying unrelated
maps to the compensating terms, however, need not preserve the same discrete
balance.

The live ``_compose_parallel_phi_ti_gradient`` applies one support-core map
to ``psi`` and adds legacy physical/transition gradients on excluded target
rows. This ensures a common map for phi and Ti in that force, but it does
not make the resulting gradient the adjoint of the separately assembled
current divergence. Nor does applying one gradient once instead of twice
repair two genuinely different linear maps.

A complete SAT extension must identify its trace operators as well as its
lifts. In a fixed-boundary-type linearization, a Neumann reconstruction has
the schematic form

```text
psi_Gamma = T_psi psi + r_Gamma(g_phi + tau*g_Ti),
G(psi; g) = G_h psi + L_g g,
D(j; j_Gamma*) = D_h j + L_j j_Gamma*.
```

The homogeneous volume pair, the canonical resolved current lift, and the
potential-side trace/lift must satisfy one declared Green/SAT boundary-work
identity. A coordinate-face trace, the plasma-side FCI trace entering the
sheath law, and the dual trace induced by ``L_j`` are distinct objects until
their compatibility is established. A zero coordinate trace alone therefore
does not eliminate every possible SAT trace contribution.

The density-to-current payload leak is repaired in the production current
support closure. The captured 5x4x3 mapped-wall probe also establishes that
the difference between the live homogeneous gradient and ``G0`` cannot be
represented solely by lifts of the current coordinate, plasma-endpoint, or
current-lift-dual traces, even when all three trace sets are combined. A unit
variation with all these traces below ``3.2e-16`` retains a gradient mismatch
of ``0.2654`` in Euclidean norm. This is a fixture-scoped obstruction to a
trace-only extension using those existing traces, not a proof against other
SBP/SAT discretizations or an estimate of a production growth rate.

A work-only completion ``D_h = -M^-1 G_live^T M`` satisfies the volume Green
identity and zero-endpoint integrated-current conservation to roundoff on
that fixture. Keeping the canonical endpoint lift still induces a dual
potential trace different from the plasma trace entering the sheath law;
physical face quadrature and boundary work remain to be reconciled. No
adjoint completion is promoted to production merely because its volume
pairing holds by construction. See the
[reconciliation report](../work/boundary_load_audit/current_potential_sat_reconciliation.md)
for reproducible matrices, the density fix, and the remaining qualification.

The candidate now has an explicit native model selection,
``parallel_current_pairing="live-gradient-prototype"``, with the reference
variant still the default. It constructs the full homogeneous live gradient
and its matrix-free weighted-adjoint divergence, retaining the canonical
endpoint-current lift once. It is restricted to the tested FCI/support-core,
production-path, characteristic-SAT, simplified-GBS-MPE configuration. This
is an implementation prototype, not a completed physical wall-power
qualification. See the
[implementation and static-test note](../work/boundary_load_audit/live_current_divergence_prototype.md).
No timestep reruns accompany this implementation. A successful later test
of the old FCI-leg-only IMEX path would remove the need for the extended
implicit system for this problem.

### Rung-3+ all-Neumann polarization gauge

#### Support-paired polarization operator

The all-Neumann solve must not use the historical projected-owner operator
unchanged.  Although its RLP owner restriction is already the correct
volume-adjoint map, its independently reconstructed face gradient and
finite-volume divergence are not exact weighted adjoints.  The resulting
homogeneous operator is weakly nonnormal; on the cached ``48^3`` HSX/RLP
state that small skew component was sufficient to stall the quotient solve at
500 iterations.

The implementation separates the full physical action from its SPD energy
preconditioner. Let

```text
Draw(u; g) = D0 u + d_g,
F(u; g) = M_owner^-1 { D0^T W_face P Draw(u; g)
                       - T0^T Sf[P Draw(u; g); g_flux] }.
```

Here ``T0`` is the scalar boundary trace of homogeneous variations (not the
mean projector), and is zero on Dirichlet faces. For fixed boundary types and
masks, ``s_g = F(0; g)`` and ``A0 u = F(u; g) - s_g``. Both the
gradient-affine and surface-affine contributions are therefore retained.

Here ``D`` is the complete owner-to-materialized-fine-grid *raw coordinate*
face-gradient map, including RLP prolongation, halo topology, and wall
stencils, but stopping before metric projection. ``P`` is the perpendicular
face projector and is applied exactly once. ``W_face`` is the matching
open-face metric/quadrature measure: the three full-vector face-family
samples carry one-third weights, while first/last normal-face duplicates use
trapezoidal half-weights (including periodic and shared-shard seams); a
physical boundary face consequently carries its half dual-cell measure.
The transpose is therefore the exact weighted scatter of the same raw
coordinate gather used to form ``D``; no independently assembled divergence is
assumed to be its adjoint.

The physical Neumann datum prescribes a normal derivative, not a boundary
trace; the boundary value and hence the unknown ``q_0`` contribution remain
in ``Sf``. The full surface term includes normal and oblique tangential
contributions. ``BC_NORMALFLUX`` instead prescribes the ``J``-weighted
coordinate flux in ``Sf`` and likewise does not prescribe a trace.

The SPD energy action is distinct:

```text
E = M_owner^-1 D0^T W_face P D0 .
```

Only ``E`` has a guaranteed weighted-SPD property. The physical ``A0`` is
generally nonsymmetric and is the action used for RHS/residual evaluation.
For an all-Neumann boundary this physical action is restricted to the
mean-zero quotient,

```text
Q x = x - 1 (1^T M_owner x) / (1^T M_owner 1),
Aq = Q A0 Q .
```

The two-sided projection makes the constant the exact two-sided null mode.
If any Dirichlet face is active, that physical boundary removes the constant
null and the unprojected support action is retained. The same selected action
is used for both ``phi`` and ``Ti`` in the polarization equation and to derive
the wall-facing vorticity trace.

For Dirichlet data, the affine gradient component is explicitly

```text
s_D = M_owner^-1 D_0^T W_face P d_D,
```

where ``D_0`` is the gradient map from the same halo/stencil construction used
by the unknown-field action and ``d_D`` is the Dirichlet affine gradient. The
surface term is not differentiated independently: it is part of ``F``
and its affine contribution is included in ``s_g`` exactly once. ``W_face``
retains one-third family averaging and endpoint half-weights, while ``Sf``
uses full physical surface measure with no such volume-quadrature factors.

#### Frozen Dirichlet-load audit (48^3)

The work-only regression ``work/boundary_load_audit/check_frozen_boundary.py``
exercises the production model setup against the frozen ``current_*`` state in
``capture_stage2_base.npz`` (no timestep or full-stage compile). With
``c = 0.19007102146317228`` and Dirichlet data on every active physical
coordinate-face mask, the full-support gradient of the constant is
``8.881784197001379e-16``, the mass norm and maximum of ``A(c)`` are
``3.135126454476723e-13`` and ``9.511448009749886e-12``, respectively, and
the nonconstant shift identity has relative defect
``1.1873839198098055e-15``. The all-Neumann homogeneous quotient check also
measures only roundoff-level constant action. The
machine-readable result is written to
``work/boundary_load_audit/frozen_boundary_check.json``.

This confirms the paired Dirichlet source and constant preservation for this
frozen production state. The external frozen matched-Neumann control and
full-LU coarse comparison are complete: a tighter bounded continuation reached
corrected relative residual ``8.9802e-11`` with
``lambda=-0.4542872917`` and zero weighted wall-gauge residual. The production
JAX/frozen replay execution completed unsuccessfully: both one-step replays
stopped at ``updated2`` with nonfinite material fields and are not
stability-qualified. The scoped
axis/periodic-corner refresh fix is included in this boundary audit context;
the default periodic-before-polar leaves cross-corner halo values at zero (the
masks are unchanged). A
physical normal derivative is distinct from a natural conormal flux: at an
oblique boundary the tangential gradient contributes to the conormal quantity
even for homogeneous normal data. Replacing one with the other therefore
requires a full matched surface/trace audit and may require a nonsymmetric
operator; no Neumann fix or production-readiness claim follows from this
constant test.

The current selector is deliberately opt-in:

```text
--polarization-operator-form support-paired
```

The earlier ``weighted-symmetric`` form remains a diagnostic comparison. It
symmetrizes the historical conservative action, but it does not define the
operator from one primary support map and therefore is not the selected
Rung-3 contract. The support-paired energy contraction has small-grid PSD
evidence on aligned fixtures; the full physical action is generally
nonsymmetric and is not claimed PSD. The support-paired form has constant-null
and affine-source-once tests; trajectory qualification remains blocked by the
failed one-step ``48^3`` replays.

The candidate Rung-3+ design uses an all-Neumann plasma-side polarization
operator.  The scalar augmented form below is conditional on verifying the
discrete ``ker(A) = span{1}`` for the homogeneous operator (called ``A0``
below).  In straight-field or aligned geometry, an all-Neumann operator can
have additional field-aligned (for example,
eta-only) null families.  Those cases require a constraint matrix and a
quotient by a basis of the full nullspace, rather than one scalar multiplier
and one gauge row.  The cached ``48^3`` HSX/RLP audit found ``A0 1 = 0``
exactly and finite responses for the sampled eta Fourier modes, supporting a
single exact constant null direction for this geometry.  This is a targeted
nullity probe, not yet a bound on every near-null singular mode.  Subject to
the one-dimensional nullspace condition, ``A0`` has the additive constant
nullspace, so a separate
compatibility condition is required in addition to choosing a gauge.  The
multi-nullspace case uses a basis ``Z`` and matching independent constraints
``C`` in the block ``[A0, Z; C, 0]`` (the scalar form below is ``Z=1`` and
``C=g^T``), or an equivalent full-nullspace quotient.  The physical gauge is
the normalized, area-weighted average over plasma-side wall faces, tied to the
material-wall potential and the equilibrium warm-ion floating-sheath
reference:

```text
g^T phi = phi_ref,    g^T 1 = 1,
phi_ref = phi_wall + phi_sheath,eq,warm-ion,float .
```

This reference is an equilibrium calibration of the potential offset.  It is
not a local dynamic zero-current condition and must not be reinterpreted as a
pointwise floating-sheath solve.  The exact constrained form is

```text
[A0, 1; g^T, 0] [phi; lambda] = [r - s_Gamma; phi_ref] .
```

The physical affine equation is ``A0 phi + s_Gamma = r``.  Thus the quotient
RHS is ``b = r - s_Gamma``.  Do not assume that volume weights are a left-null
functional for nonsymmetric RLP.  For a right-null vector ``z`` and any
quotient functional ``c`` with ``c^T z != 0``, define

```text
P = I - z c^T / (c^T z) .
```

The equivalent factorized implementation solves

```text
P A0 P phi_q = P b,    c^T phi_q = 0,
```

then imposes the physical gauge with the analytic nullspace shift

```text
phi = phi_q + ((phi_ref - g^T phi_q) / (g^T z)) z .
```

For the scalar constant-nullspace case, ``z=1`` and the normalized gauge
weights satisfy ``g^T 1 = 1``, recovering the simpler shift shown above.  The
compatibility multiplier is then computed without explicitly knowing the
left-null vector:

```text
lambda = c^T (b - A0 phi) / (c^T z),
verify A0 phi + lambda z - b = 0 .
```

When ``rank(A0) = N-1`` and the actual left null ``q`` satisfies
``q^T z != 0``, ``lambda = 0`` diagnoses the compatibility condition
``q^T b = 0``.  The actual ``q`` can still be audited for conservation, but it
is not used in this factorization.  Choosing volume weights for ``c`` gives a
zero-volume-mean quotient; that computational choice is only for validation
or fallback and is not the physical final gauge.

The preconditioner must project both every residual and every correction with
``P``. The full matched physical-Neumann action uses FGMRES. The SPD energy
contraction is a diagnostic reference; Rung 3 uses the positive full-axis
owner Jacobi action ``diag(K_owner)^-1 M_owner``, while the coarse correction
factors the full nonsymmetric action with LU. It is finite, positive, and
self-adjoint in the same owner-volume product as the support-paired operator.
All support-paired preconditioner choices use FGMRES because the physical
action is nonsymmetric.
The older line-``u`` action remains an FGMRES-only development
smoother/fallback. A stronger future preconditioner may combine
projected line-``u`` smoothing with an operator-derived, RLP-aware coarse
angular/eta correction; a separable eta correction must not be assumed.  The
factorized form still needs no full augmented-block preconditioner.  The
nonzero Neumann source must enter the raw affine assembly exactly once;
neither compatibility measurement nor the analytic shift may inject it a
second time.

#### Coarse-additive selector (qualification status)

The experimental ``coarse-additive`` selector is currently restricted to the
support-paired polarization form with the simplified GBS-MPE wall model. It
builds one factorized coarse payload at startup from the full matched,
projected owner action on the captured geometry and boundary topology. The
payload is closed over by the local runtime solver, not assembled inside a
Krylov iteration or time-step JIT. Setup must be repeated if geometry, owner
masks or volumes, face projectors, wall/Neumann masks, stencils, or RLP
topology changes. Physical affine boundary values remain in the
right-hand-side/lift path exactly once.
The bounded coarse setup currently produces rank ``383``. Its ``H`` factor is
now assembled from the full physical action and solved with LU; this is an
integration result, not a 48^3 production qualification. The SPD energy factor
remains a diagnostic/preconditioning reference.

The coarse correction is only an additive quotient-space correction: residuals
and corrections are first projected with the computational projector
``P = I - 1 c^T``, where normalized ``c`` is the owner-volume functional. It
does not itself solve an augmented system. The physical wall gauge can
equivalently be written as

```text
[ A0       1 ] [phi]   [b]
[ g_wall^T 0 ] [lam] = [phi_ref - g_affine]
```

where ``g_wall`` is the area-weighted wall functional and ``g_affine`` is its
known lift contribution. In the constant-null case the computational solve
is instead ``Aq phi_q = P b`` with ``c^T phi_q = 0``; the physical wall shift
then sets ``g_wall^T phi + g_affine = phi_ref``. In this one-constant-null
case, the projected solve plus constant wall-reference shift is algebraically
equivalent to the augmented system; the recorded ``lambda`` is the removed
compatibility/source term, not a new physical source. The coarse factor
accelerates that quotient solve; it neither supplies the physical gauge shift
nor changes affine-source accounting. Because conducting ``phi_wall`` is
fixed, choosing a plasma mean as reference is an electrical-model assumption,
not a harmless gauge relabeling unless the wall potential is shifted
consistently too.

The bounded CPU benchmark and focused-kernel tests do not qualify a full
trajectory. The first production ``48^3`` staged ten-step attempt accepted
``imex1`` but rejected ``stage2-base`` on nonfinite phi-solver input before
step 1, so no full-run, GPU, multirank, or production-scalability
qualification is claimed.

For the first support-paired trajectory gate, use FGMRES for all
support-paired preconditioner choices; owner Jacobi is the baseline, while
``coarse-additive`` remains experimental. The full matched physical-Neumann
action is generally nonsymmetric, so the SPD energy action is retained only
as a preconditioner. If the
quotient RHS is exactly zero
(``P b == 0``), skip the Krylov solve, set ``phi_q = 0``, and apply only the
physical constant gauge shift; do not rely on a tolerance-triggered null
solve.

``lambda`` is retained as a record-only diagnostic, together with the raw and
normalized compatibility defects and full residual.  It does not accept or
reject a stage and has no acceptance threshold; it must not be used to conceal
a compatibility, sign, duplicate-source, or gauge-assembly error.  The
coarse-additive RHS plumbing is present, but full Rung-3+ trajectory
qualification remains pending.

The split-vorticity-boundary helper regression is separately fixed and tested:
``_vorticity_from_polarization`` now receives all four required arguments,
with no legacy single-BC signature. The focused
``tests/test_vorticity_split_boundary_affine.py`` regression passes (one test,
17.25 s); its first RHS primitive is identical to baseline and only omega
differs at roundoff. That fix therefore was not observed to explain the
one-step blow-up.

Acceptance tests are:

* a manufactured all-Neumann problem recovers the field modulo a constant and
  satisfies the independent compatibility check;
* the area-weighted plasma-side wall-face average equals ``phi_ref`` after
  the analytic shift, while a volume-mean value is accepted only as quotient
  validation/fallback;
* a nonzero Neumann source has the expected sign and magnitude and is counted
  exactly once, while ``lambda`` and the raw/full-residual diagnostics are
  recorded without a lambda acceptance threshold;
* projected preconditioner residuals and corrections remain in the quotient,
  including under FCI single-hit/double-hit rows and shard boundaries; and
* the equilibrium warm-ion floating-sheath calibration and eager/compiled
  short replays agree before any further trajectory qualification.

## Orthogonal thermal policies

Thermal choices are likewise independent:

* the initial simplified models use `d_n Te=d_n Ti=0`;
* a sheath heat-transmission policy may prescribe electron and ion heat
  fluxes;
* a transcollisional/kinetic policy may replace those coefficients; and
* thermal policies must export the same resolved heat flux to diffusion,
  material energy transport, and diagnostics.

No thermal policy should be smuggled into the rung number or inferred from a
primitive wall value.

## Empirical reason the old Bohm rung is superseded

The former ion-only and local-floating attempts are retained as diagnostic
baselines but are superseded by this four-rung design. They imposed hard
residuals on the incoming characteristic amplitudes and therefore tested an
incompatible boundary problem.

In the compatible `48^3` one-step audit, `1,700` rank-one required faces per
leg had only one incoming control lane. The worst projected ion-Bohm residual
control gain was `0.00841031`. Enforcing the equality generated
`|Ve| ~= 70.8018` and face current `|j| ~= 36.1872`, despite `Vi=Ve` in the
compatible initialized owner. A full-state completion produced the same
rank-one response because completion cannot create a missing control
direction. The first invalid face was a backward single-hit face with
incoming rank one and inadmissible thermodynamic state.

The earlier local-floating experiment compounded this by treating pointwise
`j=0` as another hard material equation. A physical floating sheath can be
well posed when its sheath drop or conductor potential is an independent
unknown, but that is not what the old incoming-amplitude solve supplied. The
observed failure therefore does not show that a physical conducting or
floating sheath is impossible; it shows that the old residual formulation is
not its implementation.

The `physical-boundary-state` selector is the current implementation of the
new interface: it passes a complete model target through the live
characteristic split.  `primitive-least-residual` and `energy-absorbing`
remain legacy comparison paths; they are not physical rungs and may be
retired once the four-rung regressions replace their remaining uses.

## Verification gates

Every rung must verify:

* physical admissibility and finite resolved face fluxes;
* orientation and live incoming/outgoing/glancing diagnostics;
* unchanged outgoing content in the characteristic numerical interface;
* equality of material, SAT, local-BE, and diagnostic current/flux outputs;
* finite-difference or JVP checks for the total owner Jacobian;
* lower/upper walls, single/double-hit FCI rows, periodic seams, and shard
  interfaces;
* stagewise positivity, current, particle, energy, and free-energy budgets;
  and
* eager short diagnostics before any full production run.

A passing Rung 1 replay validates common machinery only. A physical-rung
production claim requires the selected wall model, compatible initialization,
fresh replay from the initial state, and spatial/timestep refinement.

## Migration order

1. Keep the live characteristic/Riemann flux and reduce the production wall
   interface to one canonical physical-target/flux payload. Retain legacy
   residual paths only for comparison diagnostics.
2. Migrate and preserve the Rung 1 no-flow regression, including common
   material/SAT/local-BE consistency tests.
3. Implement the analytic Rung 2 conducting sheath target and weak Bohm/electron
   flux, with a prescribed-wall-potential policy first. Do not implement a
   new hard incoming-amplitude solve.
4. Add the Rung 3 simplified GBS coupled `n/phi/omega` derivative closure.
5. Add the Rung 4 full MPE tangential, incidence, grazing, and smoothing
   branches.
6. Add global floating, circuit, and advanced thermal policies as independent
   modules and retire the old ion-only/local-floating experiments.

The four-rung core is intentionally stable as the physics becomes richer:
only the wall model and its orthogonal policies change; the characteristic
numerical interface and all downstream consumers retain the same contract.

Historical no-flow behavior remains reproducible from commit
`29ac064d802c6a048f7c4041db21b63a71def52f`. The raw-state Bohm and
vorticity-upwind runs remain diagnostic baselines, not implementations of the
revised four-rung contract.

## Coupled-boundary IMEX status (2026-09-05)

The standalone `fci_boundary_imex` engine provides an eager, matrix-free
damped-Newton stage solve and a transactional SSP222 host orchestrator.
The model adapter is now host-wired, but remains unqualified for production.
Here `F = E + I` denotes the physical RHS split, not the nonlinear stage
residual.  For a stage, with `h = gamma*dt`, the residuals are

* `R_U = U - U_base - h I` for the complete six-field material block;
* `R_phi = A phi + tau A Ti + omega + lambda*active` for MPE, with
  non-augmented `lambda = 0` and unchanged potential BCs; and
* `R_g = g^T phi + affine - target` for the MPE gauge.

The stage-aware wall vorticity trace is
``omega_Gamma = T[-Aphi(phi) - tau A Ti(Ti) - lambda]``. It uses the same
face payload and metric trace as the polarization action, with ``lambda``
available explicitly in the stage. A saved state can recover the compatible
multiplier as ``-mean_M(Aphi + tau A Ti + omega)`` without another inverse;
the raw-image helper and the non-augmented path are unchanged.

The implicit block includes the complete parallel five-field material action,
geometric/electron-potential/omega-advection terms, current and dissipation or
collision terms, and the physical-minus-reference perpendicular contribution.
The explicit/reference split includes prescribed sources and uses physical phi with plasma-side `g`,
homogeneous values with unchanged BC kinds for diffusion, and plasma
extensions for curvature.  The reference model reuses the already-resolved
physical face bundle. Reference evaluations may resolve the physical wall model
on the actual state; algebraic support extensions are never supplied to that
physical wall model.

The unknown is `(n, Te, Ti, Vi, Ve, omega, phi, lambda)`.  Newton is capped at
12 iterations, with 300 FGMRES iterations, restart 50, and a full step plus at
most 12 halving trials.  For each SSP222 stage `h = gamma*dt`.  Material
targets are `rtol=1e-8`, `atol=1e-10`; polarization uses the existing
configured acceptance tolerances and reports both target and acceptance
diagnostics; gauge tolerance is 1e-10.  Norms are owner-volume weighted and
field-scaled.  Lambda is diagnostic and is not itself an acceptance gate.

Every trial must be finite, positive on active density/Te/Ti owners, and pass
retarding sheath-drop admission (roundoff slack on non-grazing faces; grazing
faces are exempt).  There is no blanket clipping or timestep retry.  Failure
returns the original state transactionally.  The checkpoint remains the seven
state fields; lambda belongs in stage diagnostics JSON.

`--imex-split coupled-boundary` is an opt-in eager one-device experiment; the
historical default is not promoted.  No multistage production replay or 48^3
qualification has completed.  Remaining gates are fresh compatible
initialization, RLP/non-grazing 48^3 coverage, and timestep refinement.  The
local-backward-Euler path described earlier is a
historical control, not the new coupled split.  Confirmed interface coverage
is the 4-test main split batch (203.76 s) and 6 passing baseline short-leg
tests; these do not constitute production qualification.

The current small-stage gates are positive but bounded: the no-flow control
completed one coupled stage with 1 Newton update and 19 Krylov iterations, and
the MPE control completed two Newton updates with 127 Krylov iterations.
Twenty-four traced physical wall hits were covered across two oblique split
fixtures; the reported physical-minus-reference perpendicular ``Ve`` delta
was ``9.288e-5`` (these are not trajectory-admission passes).  The isolated residual kernel required about 37 s for its first
compile and then reused in about 0.0185 s; this is a kernel benchmark, not a
trajectory result.  Its current compiler policy disables only
``constant_folding`` through per-JIT compiler options; a 48^3 process sample
found constant-folding HLO in 652/665 samples, with an estimated 8.2 GB
footprint.  This is an execution diagnostic, not a production setting.  The
latest focused suites recorded 90 passing tests in 187.45 s (cache-write
warnings only) and a subsequent 36 passing tests in 24.72 s.  The direct host smoke using the annular mapped fixture was
not a solver failure: it was rejected by the existing FCI validation because
that fixture lacks toroidal topology (``parallel_operator_scheme='fci'``).
The ``48^3`` ``dt/16`` v3 control terminated at full
``dt=1.46484375e-5`` with initial residual ``8.684430682e-4``.  It accepted
zero Newton corrections; the 300-iteration FGMRES solve reported
``linear_solve_failed``.  The multiplier was ``-0``, the initial state was
admissible, and the failure checkpoint remained at step 0, time 0.  No linear
residual history was recorded, so this does not establish that the
preconditioner alone caused the failure or that the physical closure is
invalid.  Independent checkpoint inspection found all seven owner fields
unchanged on 86,016 owners; whole-array differences were alias
materialization, not rollback error.  No ``dt/256`` or full trajectory run
was launched, and no replay process remains active.

The coupled-boundary rollout is stopped.  The focused next step is a frozen
linear JVP refinement and block-residual/preconditioner study, with no
automatic trajectory rerun.  Actual 48^3 qualification and timestep
refinement remain pending and unqualified.
