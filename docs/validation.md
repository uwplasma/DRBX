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

## Physics Anchors (Local PDFs)

- Conserving DRB formulation: [conserving_drb.pdf](/Users/rogerio/local/tests/drb_literature/conserving_drb.pdf)
- Ballooning / s‑alpha context: [Halpern 2013](/Users/rogerio/local/tests/drb_literature/Halpern_2013_Nucl._Fusion_53_122001.pdf)
- DW/BM regime map: [Mosetto 2012](/Users/rogerio/local/tests/drb_literature/mosetto_2012.pdf)
- GBS s‑alpha geometry context: [Ricci 2012](/Users/rogerio/local/tests/drb_literature/Ricci_2012_Plasma_Phys._Control._Fusion_54_124047.pdf)
- Hermes‑3 model reference: [2303.12131v2](/Users/rogerio/local/tests/drb_literature/2303.12131v2.pdf)

---

## Test Matrix (Current)

| Category | Test | Purpose | Anchor |
|---|---|---|---|
| Unit | `tests/test_region_bc.py` | Region masks + BC policy application | Boundary-conditions design |
| Unit | `tests/test_bc_relaxation.py` | Log vs linear variables, Neumann/Dirichlet relax targets | Boundary-conditions design |
| Unit | `tests/test_normalization.py` | Physical → normalized scaling | Normalization scheme |
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
to the **PDE solver** and is the first anchor for later Hermes/GBS comparisons.

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

## Boundary‑Condition Enforcement

Region‑policy BCs are tested for **log vs linear variables**, as well as
Neumann/Dirichlet relax targets (see `tests/test_bc_relaxation.py`). This is
critical for matching Hermes/GBS open‑field‑line setups where boundary behavior
controls SOL transport and sheath losses.

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
pytest -q /Users/rogerio/local/jax_drb/tests
```

For profiling, see `/Users/rogerio/local/jax_drb/docs/profiling.md`.

---

## Benchmark Link

Full Hermes‑3 vs GBS vs jax_drb s‑alpha benchmark notes and plots live in:
`/Users/rogerio/local/jax_drb/docs/benchmarks/salpha_full.md`.
