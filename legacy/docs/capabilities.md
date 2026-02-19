# Capabilities (what `jaxdrb` can do today)

This page is a compact “map” of the repository: **linear vs nonlinear modules**, where the code lives,
and which tests/examples exercise each capability.

For the long-form narrative (equations, algorithms, and references), start at:

- [Home](index.md)
- [Validation & benchmarks](validation.md)

## Linear field-line workflows (flux-tube / ballooning representation)

**What you can do**

- Scan growth rates/frequencies vs $(k_x,k_y)$ in slab / s–$\alpha$ / circular tokamak / tabulated geometries.
- Compute growth rates by initial-value evolution of the linearized system.
- Compute leading eigenvalues/eigenmodes with matrix-free Arnoldi using Jacobian–vector products `J·v` from JAX AD.
- Apply open-field-line MPSE/sheath-entrance closures (Bohm/Loizu-style) with optional heat/SEE knobs.

**Core code**

- Cold-ion electrostatic DRB: [`src/jaxdrb/models/cold_ion_drb.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/models/cold_ion_drb.py)
- Hot-ion extension (adds `Ti`): [`src/jaxdrb/models/hot_ion_drb.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/models/hot_ion_drb.py)
- Electromagnetic extension (adds `psi ~ A_parallel`): [`src/jaxdrb/models/em_drb.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/models/em_drb.py)
- Boundary conditions + MPSE/sheath closures: [`src/jaxdrb/models/sheath.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/models/sheath.py),
  [`src/jaxdrb/models/bcs.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/models/bcs.py)
- Geometry interface + built-ins: [`src/jaxdrb/geometry/base.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/geometry/base.py),
  [`src/jaxdrb/geometry/slab.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/geometry/slab.py),
  [`src/jaxdrb/geometry/tokamak.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/geometry/tokamak.py),
  [`src/jaxdrb/geometry/tabulated.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/geometry/tabulated.py),
  [`src/jaxdrb/geometry/essos.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/geometry/essos.py)
- Matrix-free linear algebra:
  - `J·v` construction: [`src/jaxdrb/linear/matvec.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/linear/matvec.py)
  - Arnoldi: [`src/jaxdrb/linear/arnoldi.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/linear/arnoldi.py)
  - Initial-value growth estimator: [`src/jaxdrb/linear/growthrate.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/linear/growthrate.py)
  - Ideal-ballooning (Halpern 2013 Eq. 16): [`src/jaxdrb/linear/ideal_ballooning.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/linear/ideal_ballooning.py)
- CLI entry point:
  - `jaxdrb-scan`: [`src/jaxdrb/cli/main.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/cli/main.py)
  - 2D scans helper: [`src/jaxdrb/cli/scan2d.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/cli/scan2d.py)

**Key tests (verification / gates)**

- Growth-from-time vs growth-from-eigs: [`tests/test_growth_vs_eigs.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_growth_vs_eigs.py)
- Slab drift-wave sanity checks: [`tests/test_slab_dispersion.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_slab_dispersion.py)
- Tiny Arnoldi vs dense Jacobian: [`tests/test_arnoldi_dense_compare.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_arnoldi_dense_compare.py)
- MPSE/sheath closure consistency + quantitative gates:
  - [`tests/test_mpse_loizu2012_consistency.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_mpse_loizu2012_consistency.py)
  - [`tests/test_sheath_quantitative_gate.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_sheath_quantitative_gate.py)
- Hot-ion / EM model checks:
  - [`tests/test_hot_ion_model.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_hot_ion_model.py)
  - [`tests/test_em_model.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_em_model.py)
- Literature-aligned reduced-proxy gates:
  - Mosetto (2012) calibrated 4-regime gate: [`tests/test_mosetto_regime_quantitative_gate.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_mosetto_regime_quantitative_gate.py)
  - Halpern (2013) ideal-ballooning gate: [`tests/test_ideal_ballooning.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_ideal_ballooning.py)

**Key examples**

- Literature workflows: [`examples/06_literature_tokamak_sol/`](https://github.com/uwplasma/jax_drb/tree/main/examples/06_literature_tokamak_sol)
- Sheath/MPSE examples: [`examples/03_sheath_mpse/`](https://github.com/uwplasma/jax_drb/tree/main/examples/03_sheath_mpse)
- ESSOS geometries: [`examples/07_essos_geometries/`](https://github.com/uwplasma/jax_drb/tree/main/examples/07_essos_geometries)

## Nonlinear 2D workflows (HW2D + DRB2D testbeds)

**What you can do**

- Run nonlinear HW2D and DRB2D simulations (periodic domains) as verification milestones.
- Enforce hard conservative gates (energy and other quadratic invariants) in ideal subsets.
- Compute energy budgets from the discrete RHS (term-by-term closure checks).
- Compare fixed-step vs adaptive Diffrax solvers and track solver-dependent drift.
- Run DRB2D hot-ion and EM branches and verify split/parity and curvature-drive benchmarks.

**Core code**

- HW2D model: [`src/jaxdrb/nonlinear/hw2d.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/nonlinear/hw2d.py)
- DRB2D model: [`src/jaxdrb/nonlinear/drb2d.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/nonlinear/drb2d.py)
- DRB2D hot-ion / EM variants:
  - [`src/jaxdrb/nonlinear/drb2d_hot_ion.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/nonlinear/drb2d_hot_ion.py)
  - [`src/jaxdrb/nonlinear/drb2d_em.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/nonlinear/drb2d_em.py)
- Neutral coupling (milestone model): [`src/jaxdrb/nonlinear/neutrals.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/nonlinear/neutrals.py)
- Conservative diagnostics/gates: [`src/jaxdrb/nonlinear/conservative/checks.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/nonlinear/conservative/checks.py)
- Conservative bracket (Arakawa): [`src/jaxdrb/operators/brackets.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/operators/brackets.py)
- Time integration helpers: [`src/jaxdrb/nonlinear/integrate.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/nonlinear/integrate.py)

**Key tests (verification / gates)**

- HW2D invariants + budget closure:
  - [`tests/test_hw2d_conservative_gate.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_hw2d_conservative_gate.py)
  - [`tests/test_hw2d_validation.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_hw2d_validation.py)
- DRB2D conservative and budget gates:
  - [`tests/test_drb2d_conservative_gate.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_drb2d_conservative_gate.py)
  - [`tests/test_drb2d_energy_budget.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_drb2d_energy_budget.py)
- DRB2D linear-phase benchmark (2D → 1D mapping):
  - cold-ion: [`tests/test_drb2d_linear_phase_match.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_drb2d_linear_phase_match.py)
  - hot-ion: [`tests/test_drb2d_linear_phase_match_hot_ion.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_drb2d_linear_phase_match_hot_ion.py)
  - EM: [`tests/test_drb2d_linear_phase_match_em.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_drb2d_linear_phase_match_em.py)
- DRB2D curvature-drive benchmark proxies:
  - cold-ion: [`tests/test_drb2d_curvature_benchmarks.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_drb2d_curvature_benchmarks.py)
  - hot-ion + EM: [`tests/test_drb2d_curvature_benchmarks_hot_em.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_drb2d_curvature_benchmarks_hot_em.py)
- DRB2D split parity (full RHS vs split components):
  - hot-ion: [`tests/test_drb2d_hot_ion_split_parity.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_drb2d_hot_ion_split_parity.py)
  - EM: [`tests/test_drb2d_em_split_parity.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_drb2d_em_split_parity.py)
- Neutrals exchange invariants: [`tests/test_drb2d_neutrals_exchange.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_drb2d_neutrals_exchange.py)

**Key examples**

- HW2D movies + validation: [`examples/08_nonlinear_hw2d/`](https://github.com/uwplasma/jax_drb/tree/main/examples/08_nonlinear_hw2d)
- DRB2D movies + gates + budgets: [`examples/08_nonlinear_drb2d/`](https://github.com/uwplasma/jax_drb/tree/main/examples/08_nonlinear_drb2d)
- Verification bundle: [`examples/10_verification/`](https://github.com/uwplasma/jax_drb/tree/main/examples/10_verification)

## FCI / 3D preparation (maps + parallel operators + 3D slab DRB milestones)

The FCI stack is a **preparation milestone**: it validates the geometry-agnostic building blocks needed
for diverted tokamak and island-divertor geometries (where flux-surface coordinates break down).

**Core code**

- Map data structures: [`src/jaxdrb/fci/map.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/map.py)
- Map IO format (`.npz`, format v2): [`src/jaxdrb/fci/io.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/io.py)
- Map builders (z-plane + ESSOS toroidal plane): [`src/jaxdrb/fci/builder.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/builder.py)
- Parallel derivative/integration operators: [`src/jaxdrb/fci/parallel.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/parallel.py)
- Minimal 3D slab operator + budgets: [`src/jaxdrb/fci/drb3d.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/drb3d.py)
- Full 3D slab DRB milestone (`n, Omega, vpar_e, vpar_i, Te` + split API + sheath budgets):
  [`src/jaxdrb/fci/drb3d_full.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/drb3d_full.py)

**Key tests (verification / gates)**

- Parallel derivative + line-integral mapping tests:
  - [`tests/test_fci_parallel.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fci_parallel.py)
  - [`tests/test_fci_parallel_integral.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fci_parallel_integral.py)
- Target-aware derivative (Appendix-B style B/C/X handling) + MMS:
  - [`tests/test_fci_parallel_target_bc.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fci_parallel_target_bc.py)
- ESSOS toroidal-plane builder + IO metadata roundtrip:
  - [`tests/test_fci_essos_toroidal_builder.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fci_essos_toroidal_builder.py)
  - [`tests/test_fci_map_io_roundtrip.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fci_map_io_roundtrip.py)
- Curved-map interpolation regression: [`tests/test_fci_curved_map_regression.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fci_curved_map_regression.py)
- Conservative + sheath budget gates on 3D slab milestone operators:
  - [`tests/test_fci_drb3d_conservative_gate.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fci_drb3d_conservative_gate.py)
  - [`tests/test_fci_drb3d_sheath_budget.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fci_drb3d_sheath_budget.py)
- Full 3D slab DRB milestone gates:
  - [`tests/test_fci_drb3d_full_model.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fci_drb3d_full_model.py)
  - [`tests/test_fci_drb3d_full_essos_biotsavart.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fci_drb3d_full_essos_biotsavart.py)
  - [`tests/test_fci_drb3d_full_perp_bc.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fci_drb3d_full_perp_bc.py)
  - [`tests/test_fci_drb3d_full_turbulence_regression.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fci_drb3d_full_turbulence_regression.py)

**Key examples**

- FCI demos + MMS + regression: [`examples/09_fci/`](https://github.com/uwplasma/jax_drb/tree/main/examples/09_fci)

For the “full 3D energy-conserving DRB” checklist and target benchmark gates, see:
[`docs/fci/requirements.md`](fci/requirements.md).

## CI benchmark gates and reproducibility

`jaxdrb` includes “physics gates” (benchmarks that must pass in CI) for conservative invariants and solver quality.

- CI workflow: [`.github/workflows/ci.yml`](https://github.com/uwplasma/jax_drb/blob/main/.github/workflows/ci.yml)
- Field-line conservative operator gate: [`benchmarks/check_drb_conservative_gate.py`](https://github.com/uwplasma/jax_drb/blob/main/benchmarks/check_drb_conservative_gate.py)
- Poisson preconditioner benchmark gate: [`benchmarks/check_poisson_preconditioner_gate.py`](https://github.com/uwplasma/jax_drb/blob/main/benchmarks/check_poisson_preconditioner_gate.py)
- FCI DRB3D full multiphysics target/sheath budget gate:
  [`benchmarks/check_fci_drb3d_full_multiphysics_gate.py`](https://github.com/uwplasma/jax_drb/blob/main/benchmarks/check_fci_drb3d_full_multiphysics_gate.py)

## Next steps toward production 3D SOL turbulence

The medium/long-term target is a fully nonlinear, 3D, energy-conserving, multiphysics DRB solver
for edge/SOL turbulence (sheath + closures + hot ions + EM + realistic geometry).

- Project roadmap: [`docs/roadmap.md`](roadmap.md)
- FCI/3D requirements checklist + benchmark gates: [`docs/fci/requirements.md`](fci/requirements.md)
