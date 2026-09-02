# Physical-wall and characteristic-flux design

This document is the authoritative design for physical-wall boundary
conditions in the parallel five-field system. The end goal is a full warm-ion
magnetic-presheath entrance closure. Simpler wall models are validation rungs
that must use the same numerical interface as that final model.

The central decision is:

> A physical wall model supplies a complete boundary bundle. The material
> boundary state is passed through the same live characteristic numerical
> flux used at an interior face. We do not solve a fixed-size system for a
> preselected number of incoming characteristic amplitudes.

## Separation of physical and numerical responsibilities

### Physical wall model

At every physical face and integrator stage, a `PhysicalWallModel` constructs
a `PhysicalWallBundle`. The bundle is the single physical source of truth and
must eventually contain:

* a material face trace or exterior state in the primitive order
  `(n, Te, Ti, Vi, Ve)`;
* density and temperature derivative or flux data;
* electrical-wall data for `phi` (grounded, biased, locally floating, or a
  globally coupled conductor);
* polarization/vorticity boundary data;
* wall geometry and orientation, including magnetic incidence angle; and
* admissibility and branch diagnostics.

The physical model may solve its own local nonlinear equations to construct
this bundle. That solve is determined by the physical closure, not by the
instantaneous number of incoming eigenvalues of the material Jacobian.

### Boundary-state adapter

A physical face trace and an exterior reconstruction state are not always the
same object. A boundary adapter must convert the wall model's trace/flux data
to the representation expected by each operator. For example, a second-order
Dirichlet face value may require

`q_exterior = 2 q_face - q_owner`

rather than passing `q_face` directly as a ghost value. This conversion must
be explicit and tested for every reconstruction order and wall orientation.

### Live characteristic material flux

Once the owner state `q_L` and physical exterior state `q_R` are known, the
wall is treated as a boundary Riemann face. The production flux uses the same
live eigensystem and wave splitting as the bulk, for example

`F* = 0.5 (F(q_L) + F(q_R)) - 0.5 |A_hat| (q_R - q_L)`,

or the equivalent `A+`/`A-` form. The outgoing characteristic content is then
selected from the plasma-side state and the incoming content from the
physical wall target by the flux itself. The wall state is not first projected
onto an assumed incoming subspace.

The characteristic matrices are evaluated live at every stage and face, just
as they are for the bulk upwinded subsystems. The boundary implementation must
not own a hard-coded list of two acoustic modes.

## Eigenvalue crossings and incoming-mode count

The five-by-five positive and negative matrix parts have fixed array shape even
when their ranks change. Therefore a mode crossing zero does not require a
dynamic residual dimension:

* an outgoing mode is controlled by the interior side;
* an incoming mode is controlled by the exterior wall state;
* a glancing mode has vanishing normal characteristic speed and its boundary
  influence turns on or off continuously through the matrix split.

Incoming, outgoing, and glancing counts remain important diagnostics, but they
do not size a boundary solve. A count change is not an error by itself.
Failures should instead report loss of hyperbolicity, an unusably conditioned
eigenbasis, a nonfinite or physically inadmissible wall bundle, or unresolved
mode chatter under the chosen zero-speed tolerance.

Physical branches are selected from physical criteria, not `N_in`. Examples
include an intersecting-wall sheath branch, a grazing/tangent-field branch,
grounded versus floating electrical response, and eventually inverse-sheath
regimes.

## One boundary bundle for every subsystem

The material flux, characteristic SAT/current coupling, local short-leg
implicit solve, diffusion, potential solve, and vorticity/polarization closure
must consume consistent views of the same `PhysicalWallBundle`.

In particular:

* the SAT wall current must be derived from the same boundary numerical flux
  or state used by material transport;
* a selected local backward-Euler wall leg must either differentiate the wall
  map with respect to the owner state or embed the wall model in its implicit
  residual; and
* no subsystem may silently reconstruct a different wall state from an
  unrelated equilibrium reference.

This consistency is more important than making every equation use the same
primitive boundary operator. Each consumer receives the trace, exterior
state, flux, or normal derivative appropriate to its differential equation.

## Retired fixed-two-mode design

The former `velocity-no-flow` implementation classified the spectrum at
`(n, Te, Ti, 0, 0)`, selected exactly two acoustic eigenvectors, and solved
the two residuals `Vi=0` and `Ve=0` for two incoming amplitudes. It rejected
any state whose classified incoming count was not exactly two.

That design is retired because it made the numerical characteristic count
part of the physical wall-law interface. It cannot naturally handle sonic or
glancing crossings and is not the architecture needed by the magnetic
presheath model. The selector, fixed-two-mode solver, and their tests are
removed before implementing the replacement.

The generic `dirichlet-zero` primitive face trace is retained. Under the new
framework it will become an input produced by the no-flow physical wall model,
not a special incoming-characteristic solve. Historical no-flow results remain
reproducible from commit `29ac064d802c6a048f7c4041db21b63a71def52f`.

The existing `primitive-least-residual` projection and the normalized
equilibrium `energy-absorbing` map are compatibility paths only. Neither is
the target architecture for new physical closures.

The replacement infrastructure begins with
`parallel_characteristic_wall_law="physical-boundary-state"`. It passes the
complete five-component physical trace directly into the live `A+`/`A-`
boundary action, uses the live incoming rank only as a diagnostic, and exports
the current of that same state to characteristic SAT. Combined with the
`dirichlet-zero` velocity wall model, it is the new no-flow replay path.

## Boundary implementation ladder

### 0. Boundary-flux infrastructure

Implement `PhysicalWallModel`, `PhysicalWallBundle`, boundary-state adapters,
and a live material boundary flux that accepts arbitrary incoming rank without
changing array shape. Route the same bundle to SAT/current and selected
short-leg implicit consumers. Initially adapt the existing primitive traces
to this interface for controlled comparison.

Acceptance gates:

* exact bulk-face equivalence when `q_R` is an ordinary neighbor state;
* correct orientation for both wall ends;
* smooth behavior through synthetic eigenvalue crossings;
* trace-to-exterior reconstruction tests;
* consistent material flux, SAT current, and implicit wall linearization; and
* eager/JIT, batched, sharded, and invalid-spectrum coverage.

### 1. No-flow validation wall

The model supplies `Vi=Ve=0` at the material face, with explicitly chosen
simple density and temperature traces/derivatives. The live flux determines
which characteristic information enters. This is a reflecting machinery test,
not a production plasma-wall model.

Acceptance gates include exact zero velocity at the physical trace, no fixed
incoming-count assumption, crossing tests, wall-flux/SAT/implicit consistency,
and the existing short 48^3 stability replay at the validated timestep.

### 2. Zero-current Bohm--Chodura wall

The wall model supplies an oriented warm-ion sonic target such as

`Vi = Ve = sigma sqrt(Te + tau Ti)`,

with the precise normalization, density/temperature closure, and orientation
defined by the physical model. Equality of the two velocities gives zero
parallel current. This is a useful intermediate sheath-entry model, not the
full magnetic presheath.

### 3. Bohm ion flow plus electron/electrical response

Retain the warm-ion Bohm target for ions and add an electron response to the
plasma-to-wall potential drop. Implement grounded or biased walls first, then
local floating and globally floating conductor policies. The electrical wall
choice supplies data to the bundle; it is not an extra local characteristic
amplitude equation.

### 4. Full warm-ion magnetic-presheath entrance

Implement the coupled physical boundary bundle at the magnetic-presheath
entrance, including the selected model's incidence-angle dependence,
normal-flow relation, electron response, density/temperature or heat-flux
conditions, potential condition, and polarization/vorticity derivative data.
Zero current must be a selected electrical-wall policy rather than silently
assumed by every magnetic-presheath closure.

The physical model and normalizations should follow the equations actually
adopted from Loizu, Ricci, Halpern & Jolliet, *Boundary conditions for plasma
fluid models at the magnetic presheath entrance*
([EPFL manuscript](https://infoscience.epfl.ch/bitstreams/c693bbf8-aa98-4bcc-8c56-2ded2e106038/download)),
and Giacomin et al., *Journal of Computational Physics* 463 (2022) 111294
([DOI](https://doi.org/10.1016/j.jcp.2022.111294)).

## Cross-rung verification

Each rung must record:

* live incoming/outgoing/glancing counts and eigenbasis condition;
* wall-model branch, nonlinear residual (if the physical model has one), and
  bundle admissibility;
* material boundary flux and SAT/current consistency;
* local-BE Jacobian/closure error on selected wall legs;
* density, temperature, current, particle, heat, and electrical-wall budgets;
* stagewise positivity and finite-state checks; and
* spatial/timestep refinement of wall-localized modes.

A passing no-flow or Bohm replay validates the common machinery only. A
production claim requires the full selected physical wall model, a fresh run
from the initial state, and convergence evidence.
