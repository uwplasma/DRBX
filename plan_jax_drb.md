# JAXDRB Execution Plan

This file is the single authoritative plan for `jax_drb`. It replaces
`docs/research_grade_execution_plan.md` (5,935 lines, now an archival stub; full
text in git history). Written 2026-07-12 after a full audit of the source tree,
test suite, docs, hermes-3, SOLVAX, and the 2024–2026 literature; revised the
same day to commit to full hermes-3 component coverage, hardened
non-axisymmetric gates, and a closed/open × tokamak/stellarator flagship
example matrix.

## Product definition

`jax_drb` is a JAX-based, end-to-end differentiable drift-reduced Braginskii
(DRB) code for edge and scrape-off-layer plasma turbulence, on both closed and
open field lines, in axisymmetric (tokamak) and non-axisymmetric (stellarator)
geometry via the flux-coordinate-independent (FCI) approach. It must be
research-grade: literature-anchored verification and validation, honest
whole-package test coverage of at least 95%, the full hermes-3 component menu
reproduced on a documented case ladder, strong-scaling evidence on CPU and
GPU, a small codebase, and documentation a new user can act on in minutes.

Strategic fact (verified 2026-07): no published JAX or otherwise end-to-end
differentiable DRB SOL/edge turbulence code exists. The nearest published
differentiable codes are TORAX (1D core transport, arXiv:2406.06718) and
flux-tube gyrokinetics (gyaradax, arXiv:2604.06085). A lean, benchmarked,
differentiable DRB SOL code with FCI stellarator support occupies an empty
niche; hermes-3 itself has no FCI. That combination — differentiability plus
FCI non-axisymmetric SOL — is the paper.

## Ground truth (audit, 2026-07-12)

What is genuinely strong and must be preserved:

- Finite-volume/operator kernels verified against independent scalar reference
  loops at rtol 1e-12; real MMS convergence-order tests (order > 1.5 asserted).
- A deep 1D/2D recycling + neutral (AFN-style `neutral_mixed`) implementation
  with AMJUEL/ADAS rate data, sheath boundary conditions, target recycling,
  feedback controllers, and implicit BE/BDF2 time stepping.
- Golden-array parity infrastructure: committed `.npz` field dumps produced by
  a real external hermes-3 binary, compared per-term for ~8 physics families
  (drift wave, neutral_mixed, recycling, tokamak cases), with the D/T/He and
  `NVh` term-level offenders closed to 1e-9–1e-12 on their gates.
- FCI machinery for imported geometry (ESSOS coils, VMEC, hybrid), with a
  Jacobi-preconditioned potential solve and measured interpolation order ~1.96.
- A TOML-deck CLI (`jax_drb run deck.toml`) plus `run_input_case` Python API,
  restart bundles, and structured run logs.
- Incorporated from PR #3 (Aiken Xie, branch `3D_fci`, merged 2026-07-12): a
  new cell-centered FCI stack — `geometry/fci_geometry.py` (FciGeometry3D,
  halo layouts, shard specs), `native/fci_operators.py` (consistent FV
  operators, single-device + sharded), `fci_halo.py` (halo exchange),
  `fci_boundaries/helpers/model.py`, 2-field/4-field/EM RHS modules, an RK4
  integrator, and MMS/operator/domain-decomp/halo/multigrid/blob test suites.
  It coexists with the original FCI lane; the ESSOS-import rewiring from that
  PR was left out (its author marked it work-in-progress and it does not run).
  Phase 6 migrates the imported-geometry examples onto this stack and then
  deletes the original FCI modules; Phase 7 builds strong scaling on its
  shard/halo machinery.

What is broken or oversold and drives this plan:

- Weight: 80,044 LOC in `src/jax_drb` of which 42,065 (53%) is
  `validation/` campaign/figure harness code never referenced by the CLI;
  `native/recycling_1d.py` alone is 7,883 lines. Tests are ~60k LOC across 147
  files, dominated by harness/artifact-existence/mock tests. Docs are 491
  files. The old plan was 404 KB.
- The "95% coverage" badge measures ~30 hand-picked files out of 163; hosted CI
  runs a 7-file test slice; `test_closeout_coverage.py` only asserts the gate's
  own configuration.
- No test ever runs hermes-3. "Live rerun" campaign tests monkeypatch the
  binary and fabricate comparison numbers. The TCV-X21 "benchmark record" is a
  fabricated scaffold string, not the public dataset. blob2d and drift-wave
  "benchmarks" assert regression-locked outputs of prior runs, not literature
  values. `test_validation_alfven_wave.py` hard-codes an absolute path that
  exists only on one machine.
- Dependency metadata is wrong: `diffrax` and `equinox` are declared but never
  imported; `lineax` is imported but not declared; `solvax` (same org, v0.6.1,
  GMRES/GCROT/PCG/block-Thomas/banded/preconditioners/implicit-diff) is not
  used at all while `solver/implicit.py` hand-rolls ~2,900 LOC of the same
  machinery.
- The heavy recycling path is mixed JAX/NumPy/SciPy and is not end-to-end
  differentiable; its own docs say so. Current GPU evidence is negative
  (12.3x slower on the promoted gate).
- hermes-3 works locally: `build/hermes-3` (arm64, RelWithDebInfo, SUNDIALS,
  MC slope limiter) runs directly; Docker is available for reproducible runs.

## Hard rules

1. This file is the only plan. No new roadmap/status/ledger documents.
2. Claims trace to committed tests or documented commands. No "promotion
   ledger" / "claim boundary" / "capability tier" vocabulary anywhere in
   README, docs, or code. A figure without a reproduction command does not
   ship.
3. Coverage means whole-package: `pytest --cov=jax_drb` over `src/jax_drb`,
   one number, enforced in CI at 95% after Phase 1. No curated file lists.
4. Heavy artifacts (NetCDF, npz baselines, movies, profiler bundles) live in
   GitHub releases with a manifest, never in git history.
5. Every commit is authored as rogeriojorge with no Co-Authored-By or
   Generated-with trailers.
6. No version-suffixed module names (`_v2`, `_new`). Canonical names only.
7. Generic numerics belong in SOLVAX; jax_drb keeps physics. Host-side SciPy
   paths are labeled as such and never marketed as differentiable.
8. Prefer deleting to relocating: anything not needed for the product
   definition, the benchmark ladder, or a shipped example is removed (git
   history preserves it).

## Target architecture and budgets

```text
src/jax_drb/                      ≤ 38,000 LOC (full hermes-3 menu included;
  config/       TOML decks, normalization             still <50% of today)
  geometry/     mesh, metrics, FCI maps, ESSOS/VMEC/VMEC-extender import
  physics/      DRB models: RHS terms, closures, drifts, sheath,
                vorticity/potential, electromagnetic
  neutrals/     neutral_mixed, parallel diffusion, AMJUEL/ADAS rates, recycling
  solver/       thin adapters over solvax (Newton, preconditioner aliases)
  linear/       linearized-DRB operator + eigensolver (NEW, Phase 3)
  runtime/      jit/precision/devices, state, restart, artifacts
  diagnostics/  profiles, spectra, fluxes, movie writers (small)
  cli.py        run / inspect / cases / compare
tests/                            ≤ 25,000 LOC, honest coverage ≥ 95%
examples/
  tutorials/    flat SIMSOPT-style scripts: params at top, no main(),
                build deck → run → plot, CI-smoke-tested
  benchmarks/   one script per benchmark-ladder rung (regenerates the figure)
  tokamak/      closed + open field-line flagships (see example matrix)
  stellarator/  closed + open field-line flagships (see example matrix)
dev/            NOT installed, NOT counted: golden regeneration vs hermes-3,
                figure factories, profiling harnesses (small, curated)
docs/           ≤ 30 markdown pages + small images; campaign reports deleted
```

Deletion targets (from the audit; git history preserves everything):

- `src/jax_drb/validation/`: 36 `*_campaign.py` files (~28.3k LOC) plus
  movie/scaffold/audit/profile harnesses (~12.7k LOC). Keep only the analysis
  functions used by benchmark tests (drift_wave, blob2d, alfven_wave,
  neutral_mixed, fluid_1d_mms), consolidated into `diagnostics/`.
- `scripts/diagnose_*.py` (~24 one-off forensics scripts), coverage-gate
  scripts, `test_closeout_coverage.py`, synthetic "hermes live rerun" tests,
  artifact-existence campaign tests, mock-saturated controller-campaign tests.
- The recycling test mega-family (~15k LOC) collapses to per-behavior unit
  tests plus one golden-array parity test per case.
- `examples/engineering/` (22 campaign wrappers) and scaffold demos.
- Docs: ~60 campaign/status pages; release-notes pages fold into CHANGELOG.md.

## Flagship example matrix

Four quadrants, each with a flagship example that must fully work: a flat
SIMSOPT-style script, a CI smoke run at tiny size, a documented full-size
reproduction command, a committed figure, and both a physics gate
(literature-anchored) and a numerics gate (convergence and/or conservation).

| | Closed field lines | Open field lines |
|---|---|---|
| **Tokamak** | **DONE (Phase 3)** — JAX-native Hasegawa-Wakatani drift-wave turbulence (`native/hasegawa_wakatani.py`): single-mode linear growth matches the B2 eigenvalue to ~1e-14, develops outward particle flux, and is differentiable end-to-end (grad of final energy vs FD). Example `examples/tokamak/drift_wave_turbulence_demo.py`. | (i) 1D `1D-threshold`-class SOL with sheath, recycling, AMJUEL reactions, conduction — detachment-capable (B5/B6); (ii) blob2d with velocity-scaling gate (B4); (iii) 2D diverted transport + recycling vs hermes-3 goldens. Land Phases 3–5. |
| **Stellarator** | VMEC closed-field-line turbulence on an imported QA equilibrium plus a rotating-ellipse closed control: zero endpoint masks, closed-map operator conservation gate, profile/spectrum diagnostics. Lands Phase 6. | Island-divertor open-SOL turbulence on an analytic non-axisymmetric field with endpoint masks, sheath + recycling + neutral sources and closed source accounting (B7/B8); hybrid VMEC/coil open-SOL as the imported-geometry variant. Lands Phase 6. |

Quadrant exit criteria: the script runs from a clean clone (plus documented
release-asset fetch where needed), its gates run in CI (smoke size) or as a
documented manual lane (full size), and the docs page shows the figure with
the exact command that made it.

## hermes-3 capability parity matrix

Goal: every hermes-3 component has a jax_drb equivalent with a test, or an
explicit stretch/deferred label in this table. "Have/partial" statuses below
are from the 2026-07 audit; Phase 4B step 1 verifies each one against a
hermes-3 golden case and locks the matrix in `docs/parity.md`. Component names
are hermes-3's (CPC 296, 108991; local checkout `/Users/rogerio/local/hermes-3`).

| Group | hermes-3 component | jax_drb status | Target |
|---|---|---|---|
| Density | `evolve_density` | have | 4A |
| | `fixed_density`, `set_temperature`-style prescribed fields | partial (deck expressions) | 4B |
| | `quasineutral` | missing | 4B |
| | `fixed_fraction_ions` | partial (impurity fraction in radiation lane) | 4B |
| Pressure/energy | `evolve_pressure` | have | 4A |
| | `evolve_energy` | missing | 4B |
| | `isothermal`, `fixed_temperature` | partial | 4B |
| Conduction | `braginskii_conduction` (Spitzer-Härm + flux limit) | partial (conduction closure in recycling lane) | 4A |
| | `simple_conduction` | partial | 4B |
| | `snb_conduction` | missing | stretch |
| Momentum | `evolve_momentum` | have | 4A |
| | `zero_current` | partial | 4A |
| | `electron_force_balance` | partial | 4A |
| | `fixed_velocity` | partial | 4B |
| | `braginskii_ion_viscosity` | partial (D/T/He closure) | 4B |
| | `braginskii_electron_viscosity` | missing | 4B |
| | `braginskii_thermal_force` | partial (D/T/He closure) | 4B |
| Collisions | `braginskii_collisions` (multispecies AND braginskii modes, per-pair toggles) | partial | 4A |
| | `braginskii_friction`, `braginskii_heat_exchange` | partial | 4A |
| | `sound_speed` | partial | 4B |
| Fields/drifts | `vorticity` (Boussinesq + non-Boussinesq) | have (2D + FCI) | 3 |
| | `relax_potential` | missing | 4B |
| | `electromagnetic` (A∥, finite-β, Alfvén) | partial (EM selected fields) | 3 (B3) |
| | `sheath_closure` (2D drift-plane) | partial (blob2d) | 3 (B4) |
| | `diamagnetic_drift` | partial | 4B |
| | `polarisation_drift` | partial (non-Boussinesq gates) | 4B |
| Sheath BCs | `sheath_boundary_simple` (Bohm-Chodura, γ_i/γ_e, extrapolation modes) | have | 4A |
| | `sheath_boundary` (Tskhakaya multi-ion) | missing | 4B |
| | `sheath_boundary_insulating` | missing | stretch |
| | `noflow_boundary`, `neutral_boundary` | partial | 4B |
| | `decaylength` radial BC | missing | 4B |
| Neutrals | `neutral_mixed` (AFN modes, flux limiter, `neutral_lmax`) | have | 4A |
| | `neutral_parallel_diffusion` | have | 4A |
| | `neutral_full_velocity` | missing | stretch |
| | SOLKiT comparison variants | missing | stretch |
| Reactions | AMJUEL H iz/rec/CX, isotope-resolved h/d/t | have | 4A |
| | AMJUEL He (2.3.9a/2.3.13a) | partial (D/T/He lane) | 4B |
| | ADAS neon (all charge states) | partial (tables bundled) | 4B |
| | ADAS carbon, lithium | missing (same pipeline as neon) | 4B |
| | fixed-fraction radiation curves (c/n/ne/ar/kr/xe/w + simplified Ar) | partial | 4B |
| | `rate_multiplier` / `radiation_multiplier` | verify | 4B |
| Recycling | `recycling` (target/sol/pfr/pump multipliers, recycle energies) | have (target lane) | 4A |
| | fast-recycle fractions/energies per region | verify | 4B |
| | `simple_pump` | missing | 4B |
| Transport | `anomalous_diffusion` | have | 4A |
| | `classical_diffusion` | missing | 4B |
| | `binormal_stpm` (stellarator two-point model) | missing | 6 |
| Controllers | `upstream_density_feedback` | have | 4A |
| | `temperature_feedback` | have | 4B |
| | `detachment_controller` | have | 5 |
| Utility | `scale_timederivs` (steady-state acceleration) | missing | 4B |
| | `transform`, reaction diagnostics channels | partial (run-log diagnostics) | 4B |

Rules for this matrix: a row moves to "done" only with (i) implementation,
(ii) a unit/limiting-case test, (iii) one hermes-3 golden comparison in which
the component is active and its term is checked, and (iv) a row in the docs
parity table. Stretch rows (SNB, insulating sheath, full-velocity neutrals,
SOLKiT variants) are tracked but never block a release; everything else is in
scope for the 2.x series.

## Benchmark ladder

This ladder is the validation backbone. Each rung gets: one test (or manual
gate with a committed artifact), one example script under `examples/benchmarks/`,
and one docs page section. Tier V1 = verification (analytic), V2 = code-code,
V3 = validation (experiment/public dataset).

| # | Tier | Case | Anchor | Pass criterion |
|---|------|------|--------|----------------|
| B1 | V1 | MMS convergence, 1D fluid + 2D operators on curvilinear metric | Riva et al., PoP 21, 062301 (2014); Dudson et al., PoP 23, 062303 (2016), arXiv:1602.06747 | observed order → 2 (assert ≥ 1.9 on operators, ≥ 1.5 on integrated 1D) |
| B2 | V1 | Resistive drift-wave dispersion (adiabaticity scan) | BOUT++: Dudson et al., CPC 180, 1467 (2009), arXiv:0810.5757 | **DONE** — `jax_drb.linear` Hasegawa-Wakatani operator: adiabatic limit → ω\* = κk_y/(1+k⊥²); finite-α resistive instability with growth rising toward the hydrodynamic regime (tests/test_linear_dispersion.py) |
| B3 | V1 | Shear-Alfvén wave dispersion incl. electron inertia (k⊥ scan) | Stegmeir et al., PoP 26, 052517 (2019), arXiv:1904.09230 | **DONE** — `jax_drb.linear` shear-Alfvén operator reproduces ω = k∥ v_A/(1+k⊥²d_e²)^½ to machine precision, matching the code's Alfvén-benchmark deck |
| B4 | V2 | Seeded blob / 2D filament: velocity-vs-size scaling (inertial + sheath-limited branches) | Riva et al., PPCF 58, 044005 (2016); Easy et al., PoP 21, 122515 (2014), arXiv:1410.2137 | reproduce v_r(t) and the two-branch velocity scaling; cross-check vs hermes-3 `blob2d` |
| B5 | V2 | hermes-3 parity ladder (see next section) | Hermes-3: Dudson et al., CPC 296, 108991 (2024), arXiv:2303.12131 | per-case tolerances from regenerated goldens with provenance |
| B6 | V2 | 1D detachment: target-flux rollover under upstream-density scan; recombination dominant below T_t ≈ 1 eV; Lengyel-scaling comparison | SD1D: Dudson et al., PPCF 61, 065008 (2019), arXiv:1812.09402; Body et al., NME 41, 101819 (2024), arXiv:2406.16375 | rollover reproduced; detachment-onset scaling matches Lengyel within ~2x, matching hermes-3's own agreement level |
| B7 | V1/V2 | FCI verification in non-axisymmetric field: rotating-ellipse parallel-operator convergence; filament propagation | BSTING: Shanahan et al., PPCF 61, 025007 (2019), arXiv:1808.08899 | parallel-operator order ~2 with bounded perpendicular pollution; toroidally nonuniform filament dynamics |
| B8 | V2 | Stellarator island-divertor isothermal turbulence (analytic field) | Shanahan et al., JPP 90 (2024), arXiv:2403.18220; GBS: Coelho et al., NF 64, 076057 (2024) | dominant island-mode structure and outboard positive skewness; open-SOL source accounting closes |
| B9 | V3 | TCV-X21 diverted L-mode, χ agreement metric over the 45 public observables | Oliveira, Body et al., NF 62, 096001 (2022), arXiv:2109.01618; dataset Zenodo 5776286 (github SPCData/TCV-X21); hermes-3 result arXiv:2506.12180 | χ computed with the dataset's own Python package; compare against published GBS/GRILLIX/TOKAM3X/hermes-3 values (stretch goal; paper-scale) |
| B10 | V1 | Differentiability: JVP/VJP vs finite differences on every promoted model; sensitivity, inverse design, detachment-front-position gradient | TORAX (arXiv:2406.06718) as the style anchor | derivative checks at ≤1e-6 rel.; three worked autodiff examples |

Neutral-model anchors for docs: Wersal & Ricci, NF 55, 123014 (2015) (kinetic,
GBS); Zholobenko et al., NF 61, 116015 (2021) (diffusive neutrals, GRILLIX,
AUG validation); hermes-3 AFN + AMJUEL (CPC 296, 108991).

## hermes-3 parity: definition and mechanics

Parity means reproducing hermes-3's component menu (matrix above) on a
documented case ladder, with goldens regenerated from real runs — never
mocked.

- Tier-1 target (the flagship): the `1D-threshold` stack —
  `evolve_density` + `evolve_pressure` + `evolve_momentum` (D+, D, e),
  `zero_current` / `electron_force_balance`, `braginskii_collisions` +
  `braginskii_conduction`, `sheath_boundary_simple` (Bohm-Chodura, γ_i/γ_e),
  `recycling`, AMJUEL ionization/recombination/CX, `neutral_parallel_diffusion`.
  Long-window profile parity (n, T_e, T_i, target flux), not just one-step.
- Tier-2: `blob2d` (+Te+Ti variants), 2D single-null transport with neutrals
  (hermes-3 CPC paper case 2), anomalous-diffusion tokamak cases, D/T/He
  recycling, fixed-fraction impurity radiation (Ne/Ar cooling curves), and one
  golden case per matrix row as 4B closes them.
- Tier-3 (tracked, not blocking): vorticity/EM turbulence cases, TCV-X21 stack.
- Numerical-parity knobs to match and document per case: MC slope limiter,
  multispecies-vs-braginskii collision closure, AFN neutral-diffusion mode and
  flux limiter, sheath γ coefficients and extrapolation modes, rate-table
  provenance (AMJUEL H.4 2.1.5, 2.1.8, H.3 3.1.8, H.10; ADAS acd/scd/plt/prb/ccd).
- Mechanics: a `dev/parity/` harness runs the local `build/hermes-3` binary or
  the Docker image, writes goldens + a provenance manifest (hermes-3 commit,
  BOUT++ commit, deck, dt, limiter, solver), and uploads to a release asset.
  Tests consume goldens via the existing downloader. One manual CI lane
  actually reruns hermes-3 in Docker to refresh goldens; the mocked "live
  rerun" test family is deleted.
- Claim wording: "reproduces hermes-3 within stated per-case tolerances on the
  committed ladder" — never bare "parity with hermes-3".

## Non-axisymmetric program: physics, numerics, gates

The stellarator/FCI lane is the differentiator and gets explicit gates at
three levels. Geometry providers: analytic rotating ellipse, analytic
island-divertor field (Dommaschk-class), ESSOS coil import, VMEC import,
hybrid VMEC/coil, VMEC-extender finite-beta artifacts (import contract only
until upstream exporters stabilize).

Numerics gates (all in CI at smoke size):

- FCI parallel-operator MMS on the rotating ellipse: observed order ≈ 2,
  perpendicular numerical pollution bounded and reported (B7; Shanahan 2019,
  Stegmeir CPC 198, 139 (2016) support-operator scheme as reference points).
- FCI map interpolation order ≥ 1.9 (currently measured 1.96 — keep locked).
- Vorticity/potential inversion on FCI maps: Boussinesq vs non-Boussinesq
  consistency in the constant-n/B² limit (existing gate, promoted).
- Conservation: closed-field-line cases conserve particles/energy to solver
  tolerance; open-field-line cases close the source accounting
  (volume sources = sheath + recycling + neutral sinks on the consumed
  endpoint masks; the existing 1e-15-class sheath identities are the model).
- Grid/timestep refinement: flagship stellarator cases demonstrate stable
  statistics under one grid and one timestep refinement.

Physics gates (literature-anchored):

- Rotating-ellipse seeded filament: toroidally nonuniform propagation
  matching the BSTING phenomenology (B7).
- Island-divertor isothermal turbulence: dominant island-localized mode
  structure and outboard positive skewness (B8; Shanahan JPP 2024; Coelho
  NF 64, 076057 (2024) report an m=4-class coherent mode in GBS — use as
  qualitative cross-check).
- Open stellarator SOL: sheath + recycling + neutral sources active on
  endpoint masks with closed accounting; connection-length and target-flux
  maps as diagnostics.
- `binormal_stpm`-style stellarator 1D transport for the two-point-model
  limit (matrix row, Phase 6).

Context anchors to cite in docs: GRILLIX-stellarator (Stegmeir et al., CPC
318, 109874 (2026)), BOUT++ stellarator methods (Bold & Shanahan,
arXiv:2603.28221), GENE-X stellarator extension (SSRN/CPC preprint, 2025).

## SOLVAX extraction (uwplasma/SOLVAX, PyPI `solvax`)

jax_drb's `solver/implicit.py` (2,867 LOC) is physics-agnostic and moves to
SOLVAX as new modules; jax_drb keeps residual builders, stencil/coloring
choices, and the physics-named preconditioner alias table. Commit/push to
SOLVAX is authorized.

| New/extended solvax module | Content (from jax_drb) |
|---|---|
| `solvax.integrators` | `backward_euler_residual`, variable-step `bdf2_residual` |
| `solvax.sparsity` | stencil-radius CSR patterns, modulo graph coloring, JVP direction plans/workspaces |
| `solvax.jacobian` | colored finite-difference and colored-JVP sparse Jacobians |
| `solvax.newton` | the three Newton drivers (assembled-sparse, host newton_krylov behind the `native` framing, JAX-linearized with backend dispatch + line search + telemetry) |
| `solvax.precond` (extend) | sampled-from-matvec constructors: diagonal, local block, line-Schur, parallel-line |
| `solvax.krylov` (extend) | `bicgstab` |
| `solvax.pcg` (extend) | optional inner-product/projection hooks (absorbs the FCI vorticity CG) |
| `solvax.elliptic` (new) | Fourier–Helmholtz spectral+tridiagonal solve |

Result: `solver/implicit.py` → ~400–600 LOC of adapters; jax_drb sheds
~2,400 LOC; solvax gains the features with its own ≥95% coverage gate and a
minor-version release. jax_drb then depends on `solvax` (and drops unused
`diffrax`/`equinox`; `lineax` becomes an optional extra or arrives via solvax).

## Phases

Each phase ends with: full fast suite green locally, honest coverage not
decreased, and a short entry appended to CHANGELOG.md. No execution log in
this file.

### Phase 0 — Truth and hygiene (small, do first)

1. Land this plan; stub the old plan doc; update mkdocs nav.
2. Fix `pyproject.toml` dependencies (rule 7 list above).
3. Replace coverage machinery with one whole-package `pytest --cov=jax_drb`
   lane; delete the two curated-gate scripts and `test_closeout_coverage.py`;
   record the honest baseline number in CI output (badge later, once real).
4. CI runs the full fast suite (`-m "not slow"`) on 3.10–3.12; heavy/live
   lanes become manual workflows. Mark genuinely slow tests with the existing
   `slow` marker instead of curated file lists.
5. Fix the hard-coded `/Users/rogerio/...` path in the Alfvén test (skip →
   committed fixture).
6. Purge local working-tree junk (`tmp/` 2.0 GB, `profiles/` 127 MB,
   `artifacts/`, `site/`) and add ignores where missing.

Gate: CI green on the full fast suite; one honest coverage number known.

### Phase 1 — The big cut

1. Execute the deletion targets list (architecture section). Move the few
   keepers to `dev/` (uninstalled) or `diagnostics/`.
2. Trim `cli.py` to `run / inspect / cases / compare`; drop campaign
   subcommands.
3. Reorganize `examples/` into `tutorials/ benchmarks/ tokamak/ stellarator/`;
   rewrite the survivors as flat SIMSOPT-style scripts (constants at top, no
   argparse for tutorials, print + plot + save under `output/`); each is
   smoke-run in CI with tiny sizes.
4. Consolidate tests per the audit keep-list (operator kernels, MMS, solver,
   config/IO, golden-array parity per family, comparison-engine spec); delete
   the theater/harness families.
5. Prune docs to the ≤30-page tree; untrack generated `docs/data/**` JSON
   (release assets instead); fold release notes into CHANGELOG.md.
6. Split `native/recycling_1d.py` into `neutrals/recycling/{state, layout,
   operators, closures, reactions, collisions, neutral_diffusion, targets,
   boundaries, residual, stepping, diagnostics}.py` — behavior-preserving,
   golden parity tests must not move.
7. Rewrite README.md (~150 lines): what it is, install, a 60-second first run,
   the benchmark-ladder table with 3–4 figures, the example matrix, docs
   links, citation. All hedging audit language removed.

Gate: budgets met (package ≤38k, tests ≤25k, docs ≤30 pages); whole-package
coverage ≥ 95% now enforced in CI; README/docs contain no dead links.

### Phase 2 — SOLVAX extraction

Execute the extraction table; release solvax; adapt jax_drb; delete the
duplicated linearized-update in `recycling_fixed_residual.py` and the
hand-rolled CG in `fci_vorticity.py`. Golden parity tests unchanged.

Gate: jax_drb imports solvax for all generic numerics; parity suite green;
solvax coverage ≥95% including the new modules.

### Phase 3 — Physics completeness and the linear DRB solver

1. New `jax_drb.linear`: linearize any registered model about a given
   equilibrium with `jax.jacfwd`/`jax.linearize`; assemble dense (small grids)
   or matrix-free operators; eigensolve (dense `eig` small, Arnoldi via
   SciPy/solvax off-JIT for larger); return γ, ω, eigenvectors. This is both a
   user feature ("linear solver of the DRB equations") and the engine for B2
   and B3.
2. Land B2 (resistive drift-wave dispersion scan) and B3 (SAW dispersion,
   completing the `electromagnetic` matrix row) as analytic-assert tests +
   benchmark example scripts; retire the regression-locked drift-wave scalars.
3. Promote vorticity/potential (Boussinesq and non-Boussinesq) with
   equation-to-code documentation in one `docs/physics.md`; add 2D curvilinear
   operator MMS (B1 completion).
4. Tokamak-closed flagship: s-alpha/flux-tube drift-wave turbulence example
   (linear phase B2-verified, outward transport) — first quadrant of the
   example matrix done end-to-end.

Gate: B1–B3 pass analytically; physics docs complete; tokamak-closed flagship
shipping.

### Phase 4 — Honest hermes-3 parity (4A flagship, 4B full menu)

4A. Execute the parity mechanics section: `dev/parity/` regeneration harness
(local binary + Docker), provenance manifests, regenerated goldens for the
tier-1 `1D-threshold` stack, long-window profile parity, manual Docker CI
lane, delete remaining mocked-parity remnants, publish the per-case tolerance
table in docs (B5). The tokamak-open 1D flagship ships here.

4B. Close the capability matrix: first verify/lock every "partial"/"verify"
row against a hermes-3 golden case, then implement the "missing" rows in this
order — quasineutral + evolve_energy + isothermal/fixed_* (model surface);
electron viscosity + thermal force + collision-mode toggles (closures);
diamagnetic/polarisation drifts + relax_potential + sheath_closure variants
(fields/drifts); Tskhakaya multi-ion sheath + noflow/neutral/decaylength BCs;
ADAS carbon/lithium + He completion + fixed-fraction radiation curves + rate
multipliers (reactions); fast-recycle + simple_pump; classical_diffusion;
scale_timederivs. Each row lands with its unit test + golden case + docs row.

Gate: every parity number traces to a regenerated golden with provenance;
matrix has no unverified rows; stretch rows explicitly labeled.

### Phase 5 — Neutrals, recycling, detachment (B6)

1. Upstream-density scan driver on the 1D recycling model → target-flux
   rollover benchmark with figure; recombination-sink diagnostic; Lengyel
   comparison.
2. Cross-check against hermes-3 1D detachment runs (same harness as Phase 4).
3. Differentiability showcase: gradient of detachment-front position with
   respect to impurity fraction / upstream density through the implicit
   solver (B10 example 3). This is the flagship autodiff demo.
4. Complete the tokamak-open quadrant: blob2d velocity-scaling gate (B4) and
   the 2D diverted transport + recycling example against tier-2 goldens.

Gate: B6 test + figure committed; detachment-front gradient verified vs finite
differences; tokamak-open quadrant fully shipping.

### Phase 6 — Non-axisymmetric SOL (B7, B8) and the stellarator quadrants

1. Rotating-ellipse analytic field: FCI parallel-operator MMS gate (B7), then
   the seeded-filament example.
2. Island-divertor open-SOL turbulence flagship on an analytic field (B8):
   endpoint masks, sheath + recycling + neutral sources with closed
   accounting, profile/spectrum/skewness diagnostics, grid/timestep
   refinement check — the stellarator-open quadrant.
3. Stellarator-closed quadrant: VMEC closed-field turbulence on imported QA +
   rotating-ellipse closed control, with conservation gates and zero endpoint
   masks.
4. `binormal_stpm` stellarator 1D transport row; ESSOS/hybrid import examples
   kept as the imported-geometry variants, stripped of ledger machinery.

Gate: all numerics + physics gates in the non-axisymmetric section pass; both
stellarator quadrants fully shipping; example matrix complete.

### Phase 7 — Performance and strong scaling

1. Protocol per the code-paper conventions: strong scaling (fixed grid,
   wall-time per step vs devices, ideal line), cost breakdown by term
   (potential inversion vs RHS vs solve), and time-per-simulated-ms for a
   benchmark case. CPU (MacBook + office 36-core) and GPU (office 2x A4000,
   `ssh office`).
2. GPU strategy: batched `vmap`/`shard_map` ensembles and batched linear
   solves — single dispatch-bound iterative solves lose on GPU (measured here
   and in sfincs_jax); claims only from same-fidelity kernels.
3. Multi-device `shard_map` on the 2D turbulence case; document scaling
   figures with commands.

Gate: scaling figures reproducible via `examples/benchmarks/`; no unbacked
GPU claim anywhere.

### Phase 8 — Docs, release, paper

1. Final docs pass on the ≤30-page tree; mkdocs strict build in CI.
2. Release 2.0.0: honest coverage badge, benchmark gallery, PyPI, Zenodo DOI,
   CITATION.cff.
3. Paper skeleton (CPC style, hermes-3 paper as template): equations and
   numerics; verification (B1–B3); blob benchmark (B4); hermes-3 parity (B5);
   detachment (B6); FCI stellarator (B7–B8); performance; differentiability
   (B10, with the detachment-front gradient as the centerpiece); TCV-X21 (B9)
   if ready, else stated as ongoing.

## Standing decisions

- Time integration stays hand-rolled BE/BDF2 (moving to solvax.integrators);
  diffrax is not a dependency.
- The stable compatibility BDF path remains the default for heavy recycling
  until the JAX-linearized path beats it at same fidelity; the JAX path is the
  research lane, exposed but not default. Re-evaluate at Phase 5 exit.
- Preconditioner development is closed except line/Jacobi already proven (FCI
  potential solve); new sweeps only with a materially different operator
  splitting (bordered-Schur class), recorded win-or-lose.
- TCV-X21 (B9) is paper-scale work and never blocks releases.
- Hosted CI is unavailable this month (GitHub Actions spending limit): all
  gating is LOCAL. Run tests in parallel to keep iteration fast — the full fast
  suite drops from ~8 min to ~3:49 with `pytest -q -m "not slow" -n 4
  --dist worksteal` (needs `pytest-xdist`, in the `dev` extra). Per-change,
  run only the affected test files (seconds); run the full parallel suite
  before each commit. The authoritative local gates are: that parallel fast
  suite, `JAX_DRB_DISABLE_REFERENCE_ROOT=1 pytest -q -m "not slow"` (reproduces
  the no-reference-checkout condition), and `mkdocs build --strict --clean`.
  Mark genuinely heavy tests `slow` so they stay out of the fast gate.
- CI memory note: the full fast suite peaks at ~6.9 GB RSS (driven by the
  recycling/neutral tests, not the FCI stack). The coverage job splits the FCI
  stack into a second `--cov-append` process so coverage instrumentation
  overhead does not exhaust the runner.
