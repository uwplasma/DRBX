# jax_drb Rewrite Plan (Hermes-Parity Foundation)

## 1) Goal and Decision

Build a **new JAX-native, differentiable, CPU/GPU-efficient DRB solver** that first matches Hermes for ES/cold/Bouss/open-field/sheath, then extends to full conserving DRB physics and advanced geometries.

### Decision
- We will **not keep patching parity into the current broad core** as the main path.
- We will create a **new strict finite-volume parity core** and keep current implementation in `legacy`.
- We will only promote the new core to default after parity gates pass.

## 2) Audit Summary (Completed)

### Hermes facts (source/docs)
- Parallel transport is FV + slope limiter + Lax flux (`solver_numerics.rst`, `evolve_density.cxx`, `evolve_pressure.cxx`).
- Sheath BCs are component-level boundary physics with target flux handling (`boundary_conditions.rst`, `sheath_boundary_simple.cxx`).
- Vorticity/potential path includes boundary/guard-cell semantics and Laplacian solve details (`vorticity.cxx`).
- Normalization is explicit (`Nnorm`, `Tnorm`, `Bnorm`, `rho_s0`, `Omega_ci`).

### Current jax_drb issues for parity
- Mixed discretization paths (bracket-centered + multiple term schedulers) make strict parity brittle.
- Sheath/current/vorticity semantics are close but not structurally identical.
- Too many toggles in one core slow down parity debugging.

## 3) Repository Restructure Plan

## 3.1 New active core
- Add: `src/jaxdrb/parity_fv/`
- Add: `src/jaxdrb/parity_fv_operators/`
- Add: `src/jaxdrb/parity_fv_geometry/`
- Add: `src/jaxdrb/parity_fv_diagnostics/`

## 3.2 Move current implementation to legacy (staged)
- Move to `src/jaxdrb/legacy_v1/`:
  - `src/jaxdrb/core/`
  - `src/jaxdrb/operators/`
  - `src/jaxdrb/fci/`
  - legacy benchmark scripts tied to old core behavior
- Keep shared utilities in active package:
  - `src/jaxdrb/io/config.py`
  - `src/jaxdrb/normalization.py` (if reused after cleanup)
  - `src/jaxdrb/cli/main.py` (rewired to new core)

## 3.3 Backward-compatibility policy
- No compatibility guarantee during rewrite.
- `legacy_v1` kept for traceability and reference only.

## 4) New Core Architecture (Target)

```text
src/jaxdrb/parity_fv/
  params.py
  state.py
  geometry.py
  metrics.py
  bc.py
  flux_reconstruct.py
  flux_parallel.py
  flux_exb.py
  pressure.py
  vorticity.py
  poisson.py
  sheath.py
  sources.py
  rhs.py
  integrator.py
  diagnostics.py
```

### Design rules
- Single state layout and single RHS assembly.
- Component-like terms mapped 1:1 with Hermes equation blocks.
- Geometry and normalization are explicit adapters, not hidden in term code.
- No duplicate equation implementations.

## 5) Numerical Strategy (JAX-specific)

## 5.1 Performance and memory
- Use fixed-shape arrays and `lax.scan` for full-step JIT.
- Keep field layout stable (choose one canonical memory layout and never transpose in time loop).
- Precompute metric factors and face coefficients once.
- Cache Poisson/preconditioner objects by `(shape, bc, metric hash)`.
- Avoid host transfers inside the loop; diagnostics downsampled and optional.

## 5.2 Solver strategy
- Short-term parity runs: explicit RK path matching Hermes short windows.
- Stiff/open-field runs: IMEX path with Diffrax implicit stiff block (Kvaerno/ImplicitEuler + Lineax GMRES).
- Poisson:
  - FFT/spectral where periodic and valid.
  - Metric-consistent CG/GMRES for non-periodic with reusable preconditioners.

## 5.3 Differentiability
- No non-JAX side effects in RHS.
- Differentiable linear solves (`jax.scipy`/Lineax); avoid custom non-diff branches.
- Use remat/checkpoint toggles for long differentiable runs.

## 6) Physics Parity Roadmap

## Phase A: strict ES/cold/Bouss/open-field/sheath parity
- Implement only Hermes-equivalent terms:
  - density FV transport
  - pressure FV transport (`vgradp` and `pdivv` forms)
  - vorticity + Poisson boundary semantics
  - Bohm/current sheath fluxes and energy flux closures
- Gate: one-step term-by-term RHS parity at `t=0.01`.

## Phase B: short-window parity (`t<=0.1`)
- Gate: fluctuation RMS and dominant PSD peak within 10–20%.
- Reject runs with finite-run gate (spikes/non-finite).

## Phase C: medium window (`t<=0.5` then `t<=1.0`)
- Gate: RMS trend, spectra slope, radial flux profile agreement band.

## Phase D: production turbulence window
- Extend to long SOL turbulence only after A/B/C pass.

## 7) Geometry Plan

## 7.1 Immediate
- Field-aligned tokamak with `bxcv`, `J`, `gxx/gxy/gyy`, sheath/open-closed masks.
- Match Hermes `tokamak.nc` parsing and axis mapping exactly.

## 7.2 Next
- Axisymmetric analytic adapters (s-alpha, Miller) built on same operator contracts.
- FCI path added only after parity core is stable.

## 8) Testing Matrix (new, future-proof)

## 8.1 Unit tests
- Reconstruction limiters (minmod/MC) and face states.
- FV flux divergence identities and BC semantics.
- Poisson operator SPD/symmetry and gauge handling.
- Sheath flux signs and energy transmission consistency.

## 8.2 Physics tests
- Linear DW/ballooning proxy checks.
- Conservation/invariants (advection-only and selected reduced systems).
- Sheath target flux sanity and open-field transport trends.

## 8.3 Regression tests
- Strict one-step term parity CSV gate.
- `t<=0.1` parity gate (RMS/PSD/PDF/coherence minimal set).
- Performance regression (time/step + peak memory on fixed small case).

## 8.4 CI policy
- `ruff`, `black`, `pytest` mandatory.
- Parity gates run on small meshes with deterministic seeds.
- Longer turbulence parity as optional/nightly CI.

## 9) Documentation Rewrite Plan

- Rewrite docs around new core first; move old pages under `docs/legacy_v1/`.
- Keep README concise: what code solves, quickstart, links to docs.
- Full docs include:
  - equations and term mapping table (Hermes ↔ JAX)
  - normalization derivation
  - geometry and BC policy
  - solver numerics and algorithmic choices
  - parity/validation dashboards and reproducible scripts

## 10) Execution Checklist

- [x] Audit Hermes docs/source and identify structural mismatch classes.
- [ ] Freeze current core as `legacy_v1` in repo.
- [ ] Scaffold `parity_fv` package with state/params/geometry contracts.
- [ ] Implement density FV term parity.
- [ ] Implement pressure FV term parity.
- [ ] Implement vorticity + Poisson parity path.
- [ ] Implement sheath boundary component parity (particle, momentum, energy).
- [ ] Add strict one-step term audit gate in CI.
- [ ] Pass `t<=0.1` parity gate.
- [ ] Pass `t<=0.5` parity gate.
- [ ] Build long-window benchmark panel and movies.
- [ ] Refactor docs/README to new structure.
- [ ] Move superseded tests/examples/docs to `legacy_v1`.

## 11) Stop/Go Criteria

- **Go to long turbulence runs only if**:
  - term-level parity passes for dominant channels,
  - short-window finite-run gate passes,
  - no unresolved normalization/geometry mismatch remains.
- **If not passing**, continue structural parity fixes; do not tune random parameters.

## 12) Immediate Next 5 Tasks

1. Create `src/jaxdrb/legacy_v1/` and move current core/operator modules there (no behavior edits).
2. Add `src/jaxdrb/parity_fv/{params.py,state.py,geometry.py,rhs.py}` scaffolding with strict field layout.
3. Implement Hermes-equivalent FV parallel flux kernel and unit tests.
4. Implement strict Poisson/vorticity guard-cell boundary semantics and one-step parity test.
5. Wire CLI to select `engine = "parity_fv"` and run the existing audit scripts against it.
