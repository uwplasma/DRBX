# Validation & Test Matrix

This project is **validation-first**. The unified DRB system is exercised through
unit, regression, and physics tests anchored to the literature, and these tests
are designed to **execute the actual PDE RHS** (not proxy models). This page
summarizes the current validation surface and literature anchors.

---

## Core Principles

1. **Single unified system**: all toggles (EM/ES, hot/cold ions, linear/nonlinear,
   Boussinesq/non‑Boussinesq, sheath/no-sheath, 1D/2D/3D) are subsets of the same
   core RHS.
2. **Physics anchoring**: whenever possible, tests reference specific equations
   and regimes from published work.
3. **Numerical accountability**: regression gates check stability, Poisson
   iteration counts, and performance ceilings.

---

## Physics Anchors

- Conserving DRB formulation (local note/paper)
- Ballooning / s‑alpha context: Halpern 2013
- DW/BM regime map: Mosetto 2012
- s‑alpha geometry context: Ricci 2012

---

## Test Matrix (Current)

| Category | Test | Purpose | Anchor |
|---|---|---|---|
| Unit | `tests/test_region_bc.py` | Region masks + BC policy application | Boundary-conditions design |
| Unit | `tests/test_bc_relaxation.py` | Log vs linear variables, Neumann/Dirichlet relax targets | Boundary-conditions design |
| Unit | `tests/test_normalization.py` | Physical → normalized scaling | Normalization scheme |
| Unit | `tests/test_fluctuation_diagnostics.py` | `rms_*_fluct` and `equilibrium_*` consistency | Fluctuation diagnostics |
| Unit | `tests/test_benchmark_schema.py` | Benchmark bundle schema round-trip (normalized + SI) | Cross-code reproducibility |
| Unit | `tests/test_benchmark_diagnostics.py` | Shared diagnostics (PSD/PDF/coherence/flux + finite gates) | Hermes parity diagnostics |
| Unit | `tests/test_benchmark_compare.py` | Relative-L2 comparison of short-window diagnostics with axis-aware interpolation | Hermes short-window parity harness |
| Unit | `tests/test_make_compact_reference_bundle.py` | Reproducible trimming of full bundles into compact reference fixtures | Hermes short-window parity harness |
| Regression | `tests/test_hermes_short_window_compact_fixture.py` | Compact Hermes reference fixture load + self-consistency regression | Hermes short-window parity harness |
| Unit | `tests/test_parity_fv_scaffold.py` | Parity-FV state/geometry/RHS shape contracts | rewrite safety gate |
| Unit | `tests/test_parity_fv_parallel_flux.py` | FV parallel reconstruction + boundary-flux balance | Hermes solver numerics |
| Unit | `tests/test_parity_fv_poisson_vorticity_guards.py` | INVERT_SET + radial/y-guard semantics | Hermes vorticity boundary semantics |
| Unit | `tests/test_parity_fv_engine.py` | `engine = "parity_fv"` build/run/audit scheduling compatibility + `coeff_path` metric ingestion | parity rewrite integration |
| Unit/Physics | `tests/test_parity_fv_term_gates.py` | density/pressure/vorticity parity-term assembly + coefficient scaling | Hermes parallel/vorticity structure |
| Regression | `tests/test_parity_fv_poisson_solver.py` | spectral Poisson/ω(φ) round-trip and solver-mode regression | parity Poisson/vorticity path |
| Unit/Physics | `tests/test_parity_fv_sheath.py` | open-field sheath particle/momentum/energy boundary channels | sheath boundary component parity |
| Regression | `tests/test_parity_fv_one_step_audit_gate.py` | strict one-step term-array regression for parity_fv (step0 and step1 channels) | Phase A one-step parity gate |
| Regression | `tests/test_parity_fv_short_window_gate.py` | deterministic `t<=0.1` short-window RMS/PSD/finite-run gate for parity_fv | Phase B rewrite-local gate |
| Regression | `tests/test_parity_fv_hermes_short_window_gate.py` | Hermes-coupled `t<=0.1` compact-fixture regression for RMS/PSD mismatch signature | Hermes parity progression gate |
| Unit/Physics | `tests/test_diamagnetic_terms.py` | Diamagnetic drift form mixing + pressure→temperature conversion | Hermes diamagnetic drift |
| Unit/Physics | `tests/test_equilibrium_drive.py` | Equilibrium-profile gradient drives (`ω_n`, `ω_T`) | SOL background-gradient physics |
| Unit/Physics | `tests/test_braginskii_terms.py` | Braginskii heat exchange, friction, classical diffusion | Braginskii closures |
| Unit | `tests/test_energy_budget_new_terms.py` | Energy-budget entries for diamag polarisation + Braginskii | Energy diagnostics completeness |
| Unit | `tests/test_energy_budget_remaining_terms.py` | Energy-budget entries for remaining dissipative/closure terms | Energy diagnostics completeness |
| Unit | `tests/test_em_energy_budget.py` | EM ψ energy contribution in energy-rate | EM coupling consistency |
| Unit | `tests/test_em_psi_dissipation.py` | EM ψ diffusion/resistive dissipation is energy‑dissipative | EM energy-budget closure |
| Unit/Physics | `tests/test_operator_mms_convergence.py` | FD operator MMS-style convergence (`O(Δx²)`) | Hermes/GRILLIX verification practice |
| Unit/Physics | `tests/test_mms_diamag_braginskii.py` | MMS convergence for diamag polarisation + Braginskii diffusion | Hermes/GRILLIX verification practice |
| Unit/Physics | `tests/test_parallel_limiter_mms.py` | Open-field parallel limiter MMS convergence | Hermes parallel numerics |
| Unit/Physics | `tests/test_neutrals_terms.py` | Neutral ionization/recombination + drag terms (energy-exchange placeholder skips until available) | SOL neutral closures |
| Regression | `tests/test_full_stack_em_hot_neutrals.py` | Short-run regression with EM + hot-ions + neutrals enabled together | full-stack toggle validation |
| Physics/Regression | `tests/test_sheath_sol_parity_gate.py` | Sheath flux + SOL parallel loss gate | Open-field SOL parity |
| Regression | `tests/test_benchmark_panel_script.py` | Canonical side-by-side benchmark panel render | Reproducible benchmark figures |
| Unit | `tests/test_arakawa_bracket_invariants.py` | Arakawa bracket invariants (energy/enstrophy) | conservative DRB operators |
| Unit | `tests/test_parallel_z_mode.py` | `vmap` vs `scan` parallel-z modes | Geometry implementation |
| Physics | `tests/test_energy_conservation.py` | Energy conservation (advection-only) | conserving_drb |
| Physics | `tests/test_sheath_flux_sanity.py` | Open-field Bohm-current target flux sanity (finite, positive) | sheath/open-field validation |
| Physics | `tests/test_nonlinear_stats_window.py` | Finite nonlinear stats window gate on unified RHS | SOL turbulence sanity |
| Regression | `tests/test_poisson_iter_stats_regression.py` | Warm-start reduces CG iterations | Numerical solver stability |
| Regression | `tests/test_performance_regression.py` | Max time/step on 16×16 slab | Runtime guardrail |
| Physics | `tests/test_ideal_ballooning.py` | Ideal ballooning proxy check | Halpern 2013 |
| Physics | `tests/test_mosetto_regime.py` | DW/BM regime thresholds | Mosetto 2012 |
| Physics | `tests/test_curvature_energy_budget.py` | Curvature energy budget closure | conserving_drb |
| Physics/Regression | `tests/test_linear_growth_salpha.py` | Unified-core linear growth (s‑alpha) | Halpern 2013 + DRB |

---

## Linear Growth Regression (Unified RHS)

This test is a **small s‑alpha run** that uses the *actual unified RHS* to anchor
analytic proxies to a PDE‑level signal. We define the growth rate as:

\[
\gamma = \frac{d}{dt} \ln\left(\mathrm{RMS}[n]\right).
\]

The test (`tests/test_linear_growth_salpha.py`) runs a small field‑aligned
s‑alpha case with curvature + density‑gradient drive and asserts a positive
growth rate in a mid‑time window. This ties the analytic linear proxies directly
to the **PDE solver** and anchors later multi‑code comparisons.

---

## Energy Budget Closure

Energy consistency is checked directly on the discrete operators. The unified
system exposes an energy budget:

\[
\dot{E} = \dot{E}_\mathrm{curvature}
        + \dot{E}_\mathrm{drive}
        + \dot{E}_\mathrm{transport}
        + \dot{E}_\mathrm{sheath}
        + \dot{E}_\mathrm{sources}
        + \mathrm{residual}.
\]

The curvature‑only test (`tests/test_curvature_energy_budget.py`) asserts that
the discrete budget closes to numerical precision, consistent with the
conserving DRB formulation.

---

## Term Provenance (Equations)

Each term in the unified RHS is anchored to a physical operator or closure. The
equation references below correspond to the bundled literature PDFs
(e.g., `conserving_drb.pdf`, `Loizu_2013...pdf`, `Ricci_2012...pdf`,
`Stegmeir_2018...pdf`) and the Hermes documentation that codifies numerical
fluxes.

| Term | Physics/Operator | Provenance |
|---|---|---|
| `advection` | E×B bracket (Arakawa) | conserving_drb |
| `diamagnetic` | ∇·(p/B×∇) form + gradient form mixing | Hermes drift‑reduced model |
| `parallel` | ∇∥ transport (conservative flux form) | Braginskii + Hermes numerics |
| `curvature` | C(f) curvature drive | Ricci 2012, Halpern 2013 |
| `drive` | background gradient drive from equilibrium profiles | SOL gradient physics |
| `volume_source` | explicit volumetric sources | SOL/turbulence practice |
| `sol_sources` | SOL‑localized particle/heat sources | Ricci 2012, SOL models |
| `neutrals` | neutral coupling source terms | edge‑plasma closures |
| `diffusion` | perpendicular diffusion / hyper‑diffusion | standard turbulence models |
| `classical_diffusion` | Braginskii classical diffusion | Braginskii closures |
| `braginskii_friction` | e–i frictional momentum exchange | Braginskii |
| `braginskii_heat_exchange` | e–i heat exchange | Braginskii |
| `extra_dissipation` | numerical φ/ω damping + φ BC relaxation | numerical stabilization |
| `sol_sinks` | SOL sinks (n, Te, ω) | SOL transport models |
| `sol_parallel_loss` | Bohm/parallel loss model | SOL sheath losses |
| `sol_sheath_phi` | sheath‑current φ damping | Bohm/Loizu sheath |
| `sol_sheath_omega` | ω damping in open field | SOL sheath closures |
| `sol_omega_bc` | ω boundary relaxation | SOL boundary physics |
| `sol_vpar_bc` | v∥ boundary relaxation | SOL boundary physics |
| `sol_edge_relax` | edge relaxation (n, Te) | SOL boundary physics |
| `region_bc_relax` | region‑policy BCs | boundary‑condition design |
| `field_bc_relax` | per‑field BC relaxation | boundary‑condition design |
| `perp_bc_relax` | perpendicular BC relaxation | boundary‑condition design |
| `sheath` | Bohm/Loizu sheath closure | Loizu 2013 + Hermes docs |
| `line_bcs` | 1D line BC relax | flux‑tube line models |

---

## Diamagnetic Drift Validation

The Hermes‑style diamagnetic drift is validated through a dedicated unit test
(`tests/test_diamagnetic_terms.py`) that checks:

1. **Form mixing**: divergence form vs gradient form, with spatially varying
   curvature.
2. **Pressure→temperature conversion**: verifies that

\[
\dot{T}_e = \frac{\dot{p}_e - T_e \dot{n}}{n}
\]

is enforced for the diamagnetic energy flux. This guarantees the temperature
update is consistent with the pressure form used in the conservative DRB model.

---

## Boundary‑Condition Enforcement

Region‑policy BCs are tested for **log vs linear variables**, as well as
Neumann/Dirichlet relax targets (see `tests/test_bc_relaxation.py`). This is
critical for matching open‑field‑line setups where boundary behavior controls
SOL transport and sheath losses.

---

## Reproducing Physics Gates

Each physics gate has a direct pytest entry point. Examples:

- `tests/test_energy_conservation.py`: `pytest -q tests/test_energy_conservation.py`
- `tests/test_curvature_energy_budget.py`: `pytest -q tests/test_curvature_energy_budget.py`
- `tests/test_diamagnetic_terms.py`: `pytest -q tests/test_diamagnetic_terms.py`
- `tests/test_mms_diamag_braginskii.py`: `pytest -q tests/test_mms_diamag_braginskii.py`
- `tests/test_equilibrium_drive.py`: `pytest -q tests/test_equilibrium_drive.py`
- `tests/test_braginskii_terms.py`: `pytest -q tests/test_braginskii_terms.py`
- `tests/test_em_energy_budget.py`: `pytest -q tests/test_em_energy_budget.py`
- `tests/test_em_psi_dissipation.py`: `pytest -q tests/test_em_psi_dissipation.py`
- `tests/test_parallel_limiter_mms.py`: `pytest -q tests/test_parallel_limiter_mms.py`
- `tests/test_neutrals_terms.py`: `pytest -q tests/test_neutrals_terms.py`
- `tests/test_sheath_flux_sanity.py`: `pytest -q tests/test_sheath_flux_sanity.py`
- `tests/test_sheath_sol_parity_gate.py`: `pytest -q tests/test_sheath_sol_parity_gate.py`
- `tests/test_nonlinear_stats_window.py`: `pytest -q tests/test_nonlinear_stats_window.py`
- `tests/test_ideal_ballooning.py`: `pytest -q tests/test_ideal_ballooning.py`
- `tests/test_mosetto_regime.py`: `pytest -q tests/test_mosetto_regime.py`
- `tests/test_linear_growth_salpha.py`: `pytest -q tests/test_linear_growth_salpha.py`

---

## Poisson Solver Regression

Poisson warm‑start is tracked via mean/max CG iterations. The regression test
(`tests/test_poisson_iter_stats_regression.py`) asserts that warm‑start does not
increase the mean iteration count on a small periodic grid.

---

## Performance Guardrail

`tests/test_performance_regression.py` provides a **tiny runtime ceiling** for
a 16×16 slab on a JIT‑compiled RK4 scan. This does not replace profiling, but it
prevents accidental algorithmic regressions (e.g., unintentional host transfers,
extra Poisson solves, or Python‑side loops).

---

## How to Run

```
pytest -q tests
```

For profiling, see `docs/profiling.md`.

---

## Scope

The public validation surface focuses on **internal consistency, conservation,
and physics‑anchored checks** of the unified DRB system. Validation is kept
self‑contained and reproducible within `jax_drb`.

## Benchmark Gating (Hermes Parity Workflow)

The staged benchmark workflow applies finite-run gating at every stage:

1. short linear window
2. onset window
3. saturated window

Each stage is rejected if any RMS fluctuation channel is non-finite or exceeds
growth/peak thresholds. See:
- `tools/run_staged_benchmark.py`
- `tools/build_benchmark_bundle.py`
