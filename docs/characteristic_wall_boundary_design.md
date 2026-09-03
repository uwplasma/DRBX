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
new coupled data include:

* density and ion-flow normal-derivative coupling;
* the corresponding normal potential derivative;
* the polarization/vorticity derivative relation derived from the same
  discrete polarization operator; and
* `d_n Te = d_n Ti = 0` for the simplified thermal closure.

The density, potential, and vorticity conditions are one coupled physical
boundary model. They must not be independently replaced by `phi=0`,
`omega=0`, or unrelated primitive ghosts. Their discrete face fluxes and
derivatives must be exported through the same canonical wall-data contract
used in Rung 2.

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
* eager and compiled short replays before any full production run.

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
