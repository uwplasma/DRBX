# Non-Axisymmetric Stellarator SOL Implementation Plan

This document records the staged path for native non-axisymmetric
scrape-off-layer calculations in `jax_drb`. The goal is not only to produce a
visually plausible 3D movie, but to make the geometry, operators, validation
metrics, differentiability boundary, and documentation strong enough for
research use.

The plan follows the verification pattern used in the stellarator edge/SOL
literature: first validate metric tensors and traced field-line maps, then
validate parallel and perpendicular operators on manufactured fields, then run
reduced transport and turbulence-like dynamics, and only then promote device or
equilibrium-specific claims. The current implementation therefore uses an
analytic non-axisymmetric stellarator geometry as the first native lane. Real
equilibrium and wall data should be added only after the same validation gates
are clean on the analytic lane.

## Literature-Anchored Requirements

The relevant literature gives several constraints that should shape the code:

- Non-axisymmetric SOL simulations need geometry methods that remain valid
  through islands, stochastic regions, and open field lines. A field-line-map
  formulation is the practical near-term route because it does not require a
  single global field-aligned coordinate system.
- Stellarator SOL results should report more than a final field image. The
  recurring evidence classes are Poincare or field-line context, connection
  length or target mapping, steady profiles, fluctuation levels, skewness,
  spectra, radial flux proxies, and target or wall heat-load localization.
- Operator validation should include manufactured or analytic fields with
  resolution studies. The important numerical claims are map interpolation
  accuracy, parallel-gradient accuracy, parallel-diffusion dissipation, metric
  tensor consistency, and boundary behavior near short connection lengths.
- Reduced dynamics must be labeled as reduced dynamics until full sources,
  sheath/target closures, neutral physics, vorticity inversion, and real wall
  geometry are in place.

Useful public references for the scientific pattern are:

- [Coelho et al. 2022](https://arxiv.org/abs/2201.10871), which presents a
  global, flux-driven, electrostatic two-fluid stellarator island-divertor
  calculation and analyzes coherent mode structure, radial particle/heat
  transport, and wall interaction.
- [Shanahan et al. 2024](https://arxiv.org/abs/2403.18220), which emphasizes
  island-SOL fluctuation levels, mode content, skewness, correlation structure,
  and curvature-drive interpretation.
- [The 2024 Journal of Plasma Physics article](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/global-fluid-turbulence-simulations-in-the-scrapeoff-layer-of-a-stellarator-island-divertor/BA86AE2B67AE1F224800F2A0BB7193C1),
  which provides a compact template for methods, geometry, grid, source,
  boundary-condition, and results exposition in stellarator SOL turbulence.
- [The 2025 unified edge-fluid solver paper](https://www.sciencedirect.com/science/article/pii/S0010465525003765),
  which shows the current standard for metric verification, island-profile
  flattening tests, segmented target treatment, 3D visualization, and
  continuous-integration verification of non-axisymmetric edge calculations.

## Implemented First Lane

The first native lane now consists of:

- `src/jax_drb/geometry/metric_tensor.py`, which stores covariant and
  contravariant metric components and checks finite values, positive Jacobian,
  positive magnetic field, and inverse consistency.
- `src/jax_drb/geometry/fci_maps.py`, which stores forward and backward traced
  plane-intersection maps plus boundary masks and plane spacing.
- `src/jax_drb/geometry/stellarator.py`, which builds an analytic
  non-axisymmetric stellarator geometry with five field periods, helical
  mirror modulation, rotating elliptical cross-sections, island-like radial map
  shifts, full 3D metric tensors, curvature proxy, and connection-length proxy.
- `src/jax_drb/native/fci.py`, which implements JAX-native bilinear
  interpolation, forward/backward map application, central traced
  parallel-gradient, traced parallel diffusion, compact perpendicular
  diffusion, conservative metric-weighted parallel diffusion, and conservative
  metric-weighted perpendicular diffusion. It also now exposes the full
  \(J^{-1}\partial_i(JK g^{ij}\partial_j f)\) scalar operator with all
  contravariant cross terms for the non-axisymmetric manufactured-solution
  gate.
- `src/jax_drb/native/fci_sheath_recycling.py`, which implements the first
  non-axisymmetric target closure gate: traced endpoint masks, Bohm ion loss,
  zero-current electron particle reconstruction, sheath heat transmission, and
  exact recycled-neutral source accounting.
- `src/jax_drb/native/fci_neutral.py`, which implements the first
  non-axisymmetric neutral diffusion, ionisation, recombination, and
  charge-exchange reaction gate.
- `src/jax_drb/native/fci_vorticity.py`, which implements the first
  metric-weighted perpendicular vorticity operator and mean-free conjugate
  gradient potential inversion.
- `src/jax_drb/native/fci_drb_rhs.py`, which assembles the first
  transformable PyTree RHS combining sheath/recycling, neutral
  diffusion/reactions, vorticity diffusion, and the potential-inversion seam.
- `src/jax_drb/native/recycling_fixed_residual.py`, which now exposes
  `linearize_fixed_residual_action` and `fixed_residual_jvp_action` for
  JAX-native Jacobian-vector products on fixed-layout residuals.
- `src/jax_drb/geometry/biot_savart.py`, which loads ESSOS-format Fourier
  coil JSON files, expands field-period and stellarator symmetries, evaluates
  the filament Biot-Savart magnetic field with JAX, and exposes a
  JVP-transformable observation-point field map.
- `src/jax_drb/validation/stellarator_fci_geometry_campaign.py`, which writes
  a metric/map/connection-length validation bundle.
- `src/jax_drb/validation/stellarator_fci_suite_campaign.py`, which runs the
  same metric/map gate on baseline island, stronger island-map, and high
  mirror/shear analytic configurations.
- `src/jax_drb/validation/stellarator_fci_operator_campaign.py`, which writes
  interpolation convergence, parallel-gradient convergence, and diffusion
  dissipation diagnostics.
- `src/jax_drb/validation/stellarator_metric_mms_campaign.py`, which writes
  the full 3D metric scalar-operator manufactured-solution convergence gate
  plus synthetic-stellarator constant-state, dissipation, and cross-term
  activity diagnostics.
- `src/jax_drb/validation/stellarator_sheath_recycling_campaign.py`, which
  writes the first target/sheath/recycling source-balance artifact on the same
  non-axisymmetric field-line maps.
- `src/jax_drb/validation/stellarator_neutral_physics_campaign.py`, which
  writes the neutral diffusion/reaction conservation artifact.
- `src/jax_drb/validation/stellarator_vorticity_campaign.py`, which writes
  the vorticity inversion and radial \(E\times B\) proxy artifact.
- `src/jax_drb/validation/stellarator_drb_pytree_campaign.py`, which writes
  the fixed-layout PyTree RHS, JVP, batched-objective, and multi-device
  profiling artifact.
- `src/jax_drb/validation/stellarator_sol_showcase.py`, which writes a reduced
  3D SOL dynamics package with snapshots, an opened toroidal 3D poster, and a
  README-ready GIF.
- `src/jax_drb/validation/essos_biot_savart_campaign.py`, which writes the
  first coil-produced Landreman-Paul QA field gate: Fourier-coil ingestion,
  Biot-Savart \(\mathbf{B}\), annular field-line Poincare/residence
  diagnostics, closed-like/open-SOL-like annular FCI maps, and compact reduced
  turbulence response on both regions.

The current generated artifact bundle lives in
`docs/data/stellarator_fci_validation_artifacts/`.

The first coil-produced field artifact lives in
`docs/data/essos_biot_savart_landreman_paul_qa_artifacts/` and is documented in
`docs/essos_biot_savart_landreman_paul_qa.md`. It currently uses the
Landreman-Paul QA Fourier-coil JSON, expands `16` coils from four base coils,
and separates the annular FCI response into a closed-like region with boundary
fraction about `0.413` and an open/SOL-like region with boundary fraction about
`0.681`. The companion field-line trace shows longer inner-annulus residence
than outer-annulus residence, with mean annular exit times about `0.381` and
`0.155` toroidal turns. This is a coil-field FCI gate, not yet an imported-wall
predictive stellarator edge simulation.

The current suite gives three passing 3D analytic configurations rather than a
single showcase-only geometry. Their mirror ratios span about `0.45` to
`0.61`, mean connection-length proxies are about `23.8` to `24.7`, and maximum
radial map shifts are about `0.17` to `0.36` cells.

The newly promoted sheath/recycling gate gives exact source accounting on the
same maps. The current generated campaign has target fraction about `5.37e-2`,
total normalized ion loss about `5.94e2`, total recycled neutral source about
`5.76e2`, total target heat load about `7.04e2`, particle recycling relative
error about `5.9e-16`, neutral-energy relative error about `8.6e-16`, and
zero-current balance error below display precision.

The current conservative-operator, neutral, vorticity, and PyTree-RHS gates
also pass. The conservative diffusion probe has constant-state residual about
`9.35e-17` and monotone energy decay. The full 3D metric scalar gate observes
about `1.90` order on the identity-metric manufactured solution, annihilates a
constant field on the synthetic stellarator metric, dissipates metric-weighted
energy monotonically, and shows about `17%` cross-term contribution relative to
the full operator. The neutral gate closes particle reaction balance to about
`1.2e-18` relative error and momentum balance below display precision. The
vorticity inversion reconstructs the manufactured potential with relative L2
error about `1.30e-3` and residual about `2.01e-4`. The compact PyTree RHS is
`jax.jvp` transformable on the combined component state, with documentation-grid
JVP relative error about `6.4e-14`, `vmap` serial mismatch about `8.9e-16`, and
a passing two-device GPU smoke profile on the same fixed-layout objective.

## Geometry And Metric Equations

Let the logical coordinates be \(x^1=x\), \(x^2=y\), and \(x^3=z\), with
physical position \(\mathbf{r}(x,y,z)\). The covariant basis vectors are

```text
e_i = partial r / partial x^i
```

and the covariant metric, Jacobian, inverse metric, and magnetic-field unit
vector are

```text
g_ij = e_i dot e_j
J = e_1 dot (e_2 cross e_3)
g^ij = (g_ij)^(-1)
b = B / |B|
```

The native metric gate checks:

```text
allfinite(g_ij, g^ij, J, |B|)
min(J) > 0
min(|B|) > 0
max_ijk |sum_m g^im g_mj - delta^i_j| < tolerance
```

The current analytic campaign gives an inverse residual of about
`1.44e-14`, which is well below the promoted metric gate tolerance.

The geometry currently uses a rotating elliptical stellarator surface with a
five-period magnetic modulation. It is deliberately analytic so that tests can
be deterministic and small enough for CI while still exercising the
non-axisymmetric bookkeeping that an imported equilibrium will need.

## Field-Line Map Operators

For a scalar field \(f(x,y,z)\), the map data store the intersection of a field
line with the next and previous toroidal planes:

```text
f_up(i,j,k) = I[f(:,j+1,:)](x_fwd(i,j,k), z_fwd(i,j,k))
f_dn(i,j,k) = I[f(:,j-1,:)](x_bwd(i,j,k), z_bwd(i,j,k))
```

where \(I[\cdot]\) is bilinear interpolation on the target plane. The compact
central traced parallel derivative is

```text
grad_parallel_f = (f_up - f_dn) / (2 dphi)
```

and the compact traced parallel diffusion operator used in the first gate is

```text
laplace_parallel_f = (f_up - 2 f + f_dn) / dphi^2
```

The current operator campaign validates three properties:

- FCI interpolation converges with observed order about `1.96`.
- The traced parallel gradient converges with observed order about `1.54`.
- Parallel diffusion dissipates energy monotonically over the reduced test,
  with a final energy drop of about `1.35e-2`.

The current full-metric scalar gate validates the differential form needed for
non-axisymmetric perpendicular diffusion:

```text
L_K f = J^(-1) partial_i(J K g^ij partial_j f)
```

The remaining operator promotion is to turn this centered strong-form gate into
a face-conservative finite-volume production kernel with the same cross-metric
terms and with short-connection-length boundary interpolation:

```text
div_parallel(K grad_parallel f)
  = J^(-1) partial_parallel(J K partial_parallel f)
```

The compact parallel diffusion gate and the full metric scalar MMS gate now
provide the reference behavior for that production promotion.

The coil-produced annular FCI gate uses the same map abstraction, but builds
the maps directly from a Biot-Savart field. With cylindrical projections
\((B_R,B_\phi,B_Z)\), it advances field lines plane-to-plane using

```text
dR / dphi = R B_R / B_phi
dZ / dphi = R B_Z / B_phi
```

and maps the endpoint back to annular \((\rho,\theta)\) coordinates. This
gives the next bridge from analytic non-axisymmetric geometry to imported coil
fields: first a small coil-field gate, then Poincare/connection-length
diagnostics, then wall/target-aware source terms.

## Reduced SOL Dynamics Benchmark

The first 3D dynamics benchmark is intentionally compact. It evolves a
fluctuating scalar on the analytic non-axisymmetric geometry:

```text
partial_t n_tilde
  = chi_parallel L_parallel(n_tilde)
  + chi_perp L_perp(n_tilde)
  + alpha C(x,y,z) R[n_tilde]
  + N[n_tilde]
  - beta n_tilde
  - gamma n_tilde^3
```

Here \(L_parallel\) is the traced parallel diffusion operator, \(L_perp\) is a
compact radial/poloidal diffusion operator, \(C\) is the curvature proxy,
\(R[\cdot]\) is a radial-gradient drive, \(N[\cdot]\) is a conservative
nonlinear transfer proxy, and the final two terms bound the reduced dynamics.

The current showcase is not a full device-predictive turbulence run. It is a
native 3D geometry/operator/dynamics gate that proves the artifact pipeline,
movie generation, and reduced non-axisymmetric transport diagnostics. The
current metrics are:

- energy growth factor: about `1.13`
- final RMS fluctuation: about `8.58e-2`
- final skewness: about `5.14e-1`
- final kurtosis: about `6.81`
- connection-length-weighted RMS: about `8.53e-2`
- positive-fluctuation radial center: about `0.698`
- low-mode spectral-power fraction: about `2.99e-2`
- dominant poloidal/toroidal mode indices: `4` and `25`
- mean and maximum cell RMS fluctuation: about `9.99e-3` and `1.17e-1`
- radial-flux proxy: about `-1.20e-4`

Those quantities are the first documentation-facing physics metrics. The next
step is to add sustained flux drive, target loss, sheath response, and a
potential/vorticity solve so that the radial-flux proxy becomes a real
\(E \times B\) transport diagnostic.

The current media now follow the more useful review pattern: R-Z panels at
several toroidal angles, RMS/skewness/flux/spectrum diagnostics, and opened
traced surfaces with radial cuts for the 3D movie. This is a quality gate for
the visualization pipeline, not a replacement for the physics gates below.

## Full Drift-Reduced Braginskii Model To Promote

The production non-axisymmetric lane should promote the same physical state as
the existing open-field and recycling code paths, but with traced parallel
operators and metric-weighted perpendicular operators. In normalized form, the
minimum electrostatic single-ion model is:

```text
partial_t n
  + div_perp(n v_E)
  + div_parallel(n u_i)
  = div_perp(D_n grad_perp n)
  + S_ion - S_rec + S_core

partial_t(n u_i)
  + div_perp(n u_i v_E)
  + div_parallel(n u_i^2)
  = -grad_parallel(p_e + p_i)
  - n grad_parallel(phi)
  + div_parallel(eta_i grad_parallel u_i)
  + S_mom,atomic + S_mom,src

partial_t p_e
  + div_perp(p_e v_E)
  + div_parallel(p_e u_e)
  = -gamma_eos p_e div(v_E + u_e b)
  + div_parallel(kappa_e grad_parallel T_e)
  + div_perp(chi_e grad_perp T_e)
  + Q_ei + Q_e,atomic + Q_e,src + Q_e,sheath

partial_t p_i
  + div_perp(p_i v_E)
  + div_parallel(p_i u_i)
  = -gamma_eos p_i div(v_E + u_i b)
  + div_parallel(kappa_i grad_parallel T_i)
  + div_perp(chi_i grad_perp T_i)
  - Q_ei + Q_i,atomic + Q_i,src + Q_i,sheath

Omega = div_perp((n / B^2) grad_perp phi)

partial_t Omega
  + div_perp(Omega v_E)
  = div_parallel(J_parallel)
  + C(p_e + p_i)
  + div_perp(mu_Omega grad_perp Omega)
  + S_Omega
```

Here \(n\) is plasma density, \(u_i\) and \(u_e\) are ion and electron
parallel velocities, \(p_s=nT_s\), \(\phi\) is electrostatic potential,
\(v_E=b\times\nabla\phi/B\), \(J_\parallel=n(u_i-u_e)\) in singly charged
normalization, \(C(\cdot)\) is the magnetic-curvature drive, and
\(\Omega\) is the generalized Boussinesq or non-Boussinesq vorticity selected
by the solver tier. The production documentation must state which terms are
active in each model tier and which are deliberately disabled in compact
validation cases.

Neutral transport should be promoted in two tiers. The first tier is a
diffusive neutral model,

```text
partial_t n_n
  = div(D_n0 grad n_n)
  + S_recycle - S_ion + S_rec,vol

partial_t p_n
  = div(chi_n grad T_n)
  + E_recycle S_recycle
  - E_ion S_ion
  + Q_cx + Q_wall
```

followed by the mixed neutral momentum tier already present in the
axisymmetric/open-field source tree:

```text
partial_t(n_n u_n)
  + div(n_n u_n u_n)
  = -grad p_n
  + div(mu_n grad u_n)
  + R_cx + R_ion + R_wall
```

The target/sheath closure now has a native non-axisymmetric gate. For every
traced endpoint, the current implemented identities are:

```text
c_s = sqrt((T_e + T_i) / m_i)
Gamma_i,target = N_endpoint n c_s
Gamma_e,target = Gamma_i,target
q_e,target = gamma_e Gamma_i,target T_e
q_i,target = gamma_i Gamma_i,target T_i
Gamma_n,recycle = R_recycle Gamma_i,target
Q_n,recycle = E_recycle Gamma_n,recycle
```

where \(N_endpoint\) is the number of forward/backward FCI exits from a cell.
The campaign verifies particle recycling, neutral-energy accounting, and
zero-current balance to roundoff. The next production step is to distribute the
source with a finite target interaction length, include wall geometry and
target normals, and replace the compact target scalar source with the same
field-specific density, momentum, and pressure source slots used by the
recycling residual.

## Validation Gates To Add Next

The full metric scalar MMS gate is now implemented and documented. The next
gates should be added in this order:

1. Conservative traced parallel diffusion with variable \(J\), variable
   \(K\), forward/backward map masks, and boundary-distance fallback.
2. Promote the current target/sheath response gate into the full field RHS,
   including density, momentum, electron pressure, ion pressure, and neutral
   source slots.
3. Reduced seeded-filament propagation in the rotating-ellipse geometry,
   reporting center-of-mass motion, skewness, fluctuation amplitude, and
   connection-length localization.
4. Flux-driven reduced turbulence with source, damping, target losses,
   spectra, radial-flux proxy, and mode-number diagnostics.
5. Imported equilibrium-map ingestion from a clean NetCDF geometry bundle,
   including metric consistency, wall/target masks, and selected-field
   visualization.
6. Coil-produced Biot-Savart field maps with Poincare plots, connection-length
   histograms, closed/island/open region classification, and target-aware
   annular or imported-surface FCI maps.
7. Differentiability gates for geometry parameters, source amplitude,
   damping/transport coefficients, and objective functions based on radial
   flux, RMS fluctuation, target load, and connection-length-weighted
   observables.

## Production Physics Closure Still Required

The first native lane deliberately stops before claiming a complete
drift-reduced Braginskii edge solver on non-axisymmetric geometry. The
remaining production closures should be implemented as separate gates:

1. Convert `SyntheticStellaratorGeometry` and imported geometry bundles into
   explicit PyTrees with static metadata and dynamic arrays separated for JIT,
   `vmap`, `grad`, `jvp`, and multi-device execution.
2. Promote the centered full-metric scalar gate into a face-conservative
   production \(J^{-1}\partial_i(JK g^{ij}\partial_j f)\) operator, including
   coefficient weighting, boundary-distance fallback, and full-field RHS
   integration.
3. Expand the current `fci_sheath_recycling.py` fixed-layout RHS bridge from
   density and pressure source arrays to the full momentum and target-normal
   boundary-condition slots used by the production recycling residual.
4. Expand the current neutral density/pressure/reaction gate to mixed neutral
   momentum, finite wall interaction length, recycling source deposition, and
   imported target masks.
5. Promote the current vorticity/potential solve from Boussinesq inversion to
   the non-Boussinesq density-weighted operator and add an implicit solve
   preconditioner gate.
6. Expand the compact `fci_drb_rhs.py` PyTree RHS into the production
   fixed-layout residual with target-normal boundary terms, full parallel
   momentum advection, pressure advection/compression, curvature drive, and
   optional implicit/IMEX staging; use the new JAX-native residual
   linearization helpers for Jacobian actions rather than sparse
   finite-difference assembly.
7. Add seeded-filament and flux-driven turbulence gates that compute true
   \(E\times B\) radial particle/heat flux, target heat-load maps, skewness,
   spectra, correlation lengths, and mode-number diagnostics.
8. Add imported equilibrium/wall/target bundles only after the analytic
   geometry gate passes the same metric, map, sheath, neutral, and operator
   tests.
9. Add CPU/GPU performance gates on batched geometries and source scans, using
   persistent compilation cache and shape-stable PyTree residuals.
10. Update README, docs, and the manuscript plan only with claims that have a
    JSON/NPZ/PNG validation artifact and a focused regression test.

## Documentation And README Targets

Every promoted gate should produce:

- a JSON report with thresholds, observed metrics, and pass/fail status;
- an NPZ bundle containing only compact fields needed to regenerate figures;
- at least one publication-grade PNG figure;
- when the run is 3D and time-dependent, a small GIF or MP4 with colorbar,
  time label, and a visible opened toroidal/radial cut;
- a docs page that states the equations, implementation functions, validation
  metrics, and reproduction command.

The current docs target is
`docs/stellarator_fci_validation.md`. It should remain the main public page for
this lane until a full imported-equilibrium case has passed the same gates.

## Code Refactor Targets

The implementation should stay split by responsibility:

- `geometry`: metric tensors, field-line maps, analytic and imported
  equilibrium adapters;
- `native`: transformable JAX operators and reduced model kernels;
- `validation`: campaign builders, plots, movies, and JSON/NPZ reports;
- `examples`: small reproducible scripts that generate docs artifacts;
- `tests`: metric gates, operator convergence, dynamics metrics, and artifact
  creation.

The next refactor should avoid placing stellarator-specific logic inside the
general runner. Instead, geometry objects should become PyTrees consumed by
small native kernels. That keeps the path compatible with `jax.jit`,
`jax.vmap`, `jax.grad`, `jax.jvp`, and future multi-device execution.

## Ship Criteria For This Lane

This lane is ready for a release claim only when:

1. all current geometry, operator, and reduced dynamics tests pass;
2. docs and README show the generated figures and movie;
3. public docs avoid claiming device-predictive turbulence before target,
   sheath, source, and imported-equilibrium gates are complete;
4. generated artifacts stay small enough for a fast clone;
5. the next-step validation gaps are explicit and tracked in this plan.
