# Requirements for full 3D energy-conserving DRB (FCI track)

This page is a **living checklist** for the end goal: a fully nonlinear, 3D, energy-conserving,
multiphysics drift-reduced Braginskii (DRB) solver for edge/SOL turbulence in realistic diverted
geometries.

The **FCI (flux-coordinate independent)** approach is the intended path to X-points and island-divertor
topologies, so the requirements below are organized around FCI building blocks and the benchmark gates
needed to keep them reviewer-auditable.

Checkboxes are interpreted as:

- `[x]` implemented **and** covered by tests/gates in this repository
- `[ ]` not yet implemented (or implemented but not yet benchmarked to a gate)

## A. Geometry and field-line map pipeline

- [x] Analytic slab map (constant-$B$ / constant shift) for controlled MMS tests
  - code: `src/jaxdrb/fci/map.py`
  - examples: `examples/09_fci/fci_slab_parallel_derivative_mms.py`, `examples/09_fci/fci_hello_world.py`
- [x] Curved-map regression with spatially varying in-plane shifts (non-constant mapping)
  - example: `examples/09_fci/fci_curved_map_regression.py`
  - test: `tests/test_fci_curved_map_regression.py`
- [x] ESSOS field-line map generation on toroidal planes with target-intersection metadata
  - code: `src/jaxdrb/fci/builder.py` (`build_fci_maps_essos_toroidal_planes`)
  - tests: `tests/test_fci_essos_toroidal_builder.py`
- [ ] Map generation from full production equilibria (VMEC / coils / near-axis) into a validated FCI map set
  - target: offline map build + runtime load (indices, weights, $\Delta l$, masks)
  - related infrastructure: `src/jaxdrb/geometry/essos.py`, `examples/07_essos_geometries/`
- [x] Open-field-line termination detection and target metadata in the map (milestone)
  - maps now carry `hit`, `dl_hit`, `hit_R`, `hit_Z`, `hit_phi`, `hit_target`
  - required to impose sheath closures consistently near plates and to apply one-sided parallel stencils

## B. Interpolation and map-error controls

- [x] Differentiable in-plane interpolation (bilinear on structured grids)
  - code: `src/jaxdrb/fci/parallel.py`
  - tests: `tests/test_fci_parallel.py`, `tests/test_fci_parallel_integral.py`
- [x] Map/interpolation regression thresholds in CI (curved-map convergence/consistency)
  - tests: `tests/test_fci_curved_map_regression.py`
- [ ] Higher-order or structure-preserving interpolation options (when map smoothness demands it)
  - design constraint: remain end-to-end differentiable and compatible with JIT batching

## C. Parallel operators (derivatives, integrals, and BCs)

- [x] Centered FCI $\partial_\parallel$ operator + MMS convergence checks
  - example: `examples/09_fci/fci_slab_parallel_derivative_mms.py`
- [x] Line-integral mapping utilities (needed for parallel closures and diagnostics)
  - example: `examples/09_fci/fci_hello_world.py`
- [x] One-sided/target-aware parallel stencils near targets (Appendix-B style B/C/X handling)
  - code: `src/jaxdrb/fci/parallel.py` (`parallel_derivative_target_aware_3d`)
  - tests: `tests/test_fci_parallel_target_bc.py` (MMS convergence + point classification)
- [x] Parallel boundary-condition framework milestone for 3D slab DRB
  - implementation: target-aware one-sided `\partial_\parallel` + sheath-budget channel diagnostics
  - tests: `tests/test_fci_parallel_target_bc.py`, `tests/test_fci_drb3d_full_model.py`
- [ ] Full production boundary-condition framework across realistic 3D geometries (divertor targets, symmetry planes, etc.)
  - requirement: energy/particle budgets remain auditable and differentiable across map families

## D. Perpendicular operators (3D-ready discretization choices)

What exists today is a verified 2D milestone:

- [x] Conservative Arakawa bracket kernel on periodic 2D grids
  - code: `src/jaxdrb/operators/brackets.py`
  - gate: `tests/test_hw2d_conservative_gate.py`, `tests/test_drb2d_conservative_gate.py`

What is required for full 3D SOL turbulence:

- [x] A first perpendicular operator suite compatible with FCI planes (FD/FV + periodic/spectral) with:
  - code: `src/jaxdrb/fci/drb3d_full.py`, `src/jaxdrb/nonlinear/fd.py`, `src/jaxdrb/nonlinear/fv.py`
  - tests: `tests/test_fci_drb3d_full_perp_bc.py`
- [ ] Complete perpendicular operator suite (including DG) with:
  - conservative advection kernels,
  - controllable dissipation (hyperdiffusion, viscosity, slope limiting as needed),
  - non-periodic wall/plate boundary conditions.
- [ ] Convergence + budget gates for each operator family (not just “no NaNs”).

## E. Polarization / Poisson: toward real non-Boussinesq

Verified milestones:

- [x] Spectral periodic Boussinesq inversion (exact up to roundoff)
- [x] FD + CG Poisson verification for Dirichlet/Neumann domains
  - test: `tests/test_fd_poisson_cg.py`
  - figure: `docs/assets/images/poisson_cg_verification_panel.png`
- [x] Preconditioner benchmark gate (residual + runtime) for the Poisson solve
  - benchmark: `benchmarks/check_poisson_preconditioner_gate.py`

Required for quantitative SOL DRB:

- [ ] **Non-Boussinesq polarization** in real space:
  $$
  -\nabla_\perp\cdot(n\,\nabla_\perp \phi)=\Omega,
  $$
  which is a **variable-coefficient SPD elliptic problem** when $n>0$.
- [ ] A dedicated SPD preconditioner that remains fast in JAX (circulant/FFT, multigrid, or hybrid),
  with CI gates for both residual and runtime.
- [ ] Energy-rate gates that include the variable-coefficient polarization operator (no “leakage”).

## F. Sheath / targets (full boundary-condition set in 3D)

Verified 1D field-line closures:

- [x] MPSE/sheath-entrance closures (Bohm/Loizu-style) with quantitative gates
  - implementation: `src/jaxdrb/models/sheath.py`
  - tests: `tests/test_sheath_quantitative_gate.py`, `tests/test_mpse_loizu2012_consistency.py`

Required in 3D SOL turbulence:

- [x] Target handling in the FCI map (masking + distance-to-target + target geometry metadata)
  - implementation: map fields `hit`, `dl_hit`, `hit_R`, `hit_Z`, `hit_phi`, `hit_target`
  - tests: `tests/test_fci_essos_toroidal_builder.py`, `tests/test_fci_map_io_roundtrip.py`
- [x] Full-branch 3D slab DRB target/sheath budget coupling milestone with hot-ion/EM/neutrals toggles
  - tests: `tests/test_fci_drb3d_full_model.py`, `tests/test_fci_drb3d_full_essos_biotsavart.py`
  - CI gate: `benchmarks/check_fci_drb3d_full_multiphysics_gate.py`
- [ ] Full sheath closure set in 3D (current closure + heat transmission, optional SEE) with:
  - energy-consistent discrete budget,
  - coupled EM/hot-ion closure consistency,
  - regression gates on sheath-limited limits (published proxies and/or analytic limits).

## G. Conservative DRB formulation and invariant gates

Verified milestones today:

- [x] Hard conservative gate on the *field-line* cold-ion DRB conservative subset
  - tests: `tests/test_drb_nonlinear_conservative_gate.py`, `tests/test_drb_operator_rates.py`
  - CI gate: `benchmarks/check_drb_conservative_gate.py`
- [x] DRB2D conservative and energy-budget gates (curvature + drives)
  - tests: `tests/test_drb2d_conservative_gate.py`, `tests/test_drb2d_energy_budget.py`
- [x] Minimal DRB3D slab conservative + sheath budget gates (FCI milestone)
  - tests: `tests/test_fci_drb3d_conservative_gate.py`, `tests/test_fci_drb3d_sheath_budget.py`

Required for a full 3D solver:

- [ ] A conservative 3D discretization of the chosen DRB form (e.g. conservative formulation) with:
  - energy functional implemented as a *first-class diagnostic*,
  - term-by-term budget closure checks,
  - hard regression gates (finite-time drift + instantaneous operator-rate residuals).
- [x] Long-time turbulence-statistics regression gate for the FCI DRB3D full milestone branch
  - test: `tests/test_fci_drb3d_full_turbulence_regression.py`
- [ ] Full production-level long-time turbulence-statistics gate set (multi-geometry / multi-physics).

## H. Target benchmark gates (what should be in CI)

Current CI-gated benchmarks are documented in `docs/validation.md`. For the 3D FCI track, a minimal
reviewer-proof set should include:

- [x] FCI MMS observed order gate (slab)
- [x] Curved-map regression threshold gate (non-constant map)
- [x] 3D slab conservative-rate gate (minimal operator)
- [x] 3D slab sheath-budget gate (minimal closure)
- [x] One-sided $\partial_\parallel$ accuracy gate near targets (manufactured solution)
  - test: `tests/test_fci_parallel_target_bc.py`
- [x] Full-branch 3D slab DRB target/sheath budget gate with hot-ion/EM/neutrals enabled
  - benchmark: `benchmarks/check_fci_drb3d_full_multiphysics_gate.py`
- [ ] Non-Boussinesq polarization residual + runtime gate (SPD solve + preconditioner)
- [ ] 3D DRB energy/mass/charge/current/momentum drift gates for representative parameter sets
- [ ] Published-proxy curvature/sheath benchmarks for EM/hot-ion branches (tight tolerances)

## I. How this connects to other edge/SOL codes

Modern SOL turbulence codes (e.g. GBS, BOUT++/Hermes-3, GRILLIX) typically emphasize:

- method-of-manufactured-solutions (MMS) and order-of-accuracy studies,
- conservation/budget closure checks (especially for nonlinear runs),
- published benchmark problems (interchange, resistive ballooning, drift-wave turbulence),
- regression tests on turbulence statistics and long-time behavior.

`jaxdrb` follows the same philosophy, but adds an explicit constraint: keep the implementation
**JAX-first** (JIT/VJP-friendly) so that the full workflow can remain differentiable.

## Next steps

- For the high-level project plan: `docs/roadmap.md`
- For what is already benchmarked today: `docs/validation.md`
