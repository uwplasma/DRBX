# Testing Strategy

This page explains how `drbx` should be tested during the refactor and what
counts as meaningful evidence for a research-grade release.

The testing philosophy follows the verification and validation split discussed
in [Roy 2005](https://www.sciencedirect.com/science/article/pii/S0021999104004747)
and the practice used in major edge/SOL codes such as
[GBS](https://www.sciencedirect.com/science/article/pii/S0021999122003280) and
the [TCV-X21 benchmark program](https://arxiv.org/abs/2109.01618).

## Evidence Layers

`drbx` should be tested in layers. No single layer is sufficient by itself.

### Verification

Verification asks whether the equations and operators are implemented
correctly.

This layer includes:

- manufactured-solution convergence
- operator identities
- restart equivalence
- pack/unpack and state-layout invariants
- Jacobian and linearization consistency
- `grad` versus finite-difference agreement on promoted differentiable lanes

### Regression

Regression tests ask whether known behavior changed unexpectedly.

This layer includes:

- fixed numerical baselines
- bounded history and summary comparisons
- CLI and artifact-schema checks
- output/log/provenance checks for public surfaces

### Code-to-code validation

This layer compares `drbx` against curated reference outputs for the kept
models.

This is the main current bridge for:

- same-machine native rerun matrices across representative
  1D and 2D lanes
- selected-field geometry comparisons

### Benchmark and experiment-facing validation

This layer compares the promoted workflows against broader literature benchmarks
and diagnostics, such as:

- TCV-X21
- geometry-portability studies

This is the layer that most directly supports publication figures.

## Target Test Taxonomy

The suite should converge toward explicit logical groups:

- unit/operator tests
- regression tests
- autodiff/JAX transformation tests
- publication/campaign tests
- smoke/tutorial tests

The current on-disk tree is still mostly flat, but the refactor should make the
distinction explicit in filenames, markers, and CI slices before or during any
directory migration.

For `drbx`, the high-value test taxonomy is now:

- operator identity tests for each term in the density, pressure, momentum,
  vorticity, sheath, recycling, neutral, and collision equations;
- manufactured-solution convergence for metric-weighted FCI
  operators, including nonorthogonal metric cross terms;
- transient ladders for RHS, one-step, short-window, restart, and
  selected longer diagnostic windows;
- physics gates for conservation, source balance, target power, core/SOL
  volume integrals, connection length, endpoint masks, radial flux, spectra,
  skewness, and profile lineouts;
- differentiability gates for `jit`, `vmap`, `jvp`, `grad`, finite-difference
  agreement, and matrix-free/JVP solver actions;
- performance gates that separate compile time, execute time, memory proxy,
  CPU scaling, GPU scaling, and artifact provenance.

## Coverage Policy

The `95%` target should mean:

- strong coverage on the promoted solver surface
- direct coverage of extracted operators and closure branches
- public-surface coverage for CLI, campaigns, and artifact generation

It should not mean:

- relying on smoke tests to color large files green
- excluding the hard solver modules from the measurement
- claiming broad coverage when only compact differentiable examples are tested
- counting tutorial execution as a substitute for operator, physics, and
  convergence assertions

The release standard from the consolidated execution plan is:

- at least `95%` on the promoted solver and public-surface slice
- no critical operator family below `90%`
- no monolithic module left effectively untested except through one large
  integration case

The explicit whole-package coverage entry point is now:

```bash
pytest -q -m "not slow" --cov=drbx --cov-branch
coverage report
```

This is the same whole-package coverage job enforced by `coverage.yml` in CI. It
covers the native mesh/metric/deck-runner surface and CLI entry points, and
exists to prevent the project from confusing a narrow coverage slice with
meaningful solver coverage. When the extraction work has enough direct unit,
operator, and artifact-producing tests, the whole-package number must satisfy
the `95%` threshold before the solver surface is called research-grade.

The first measured baseline for this slice was `73%` total coverage with all
selected tests passing. The current promoted solver/public-surface audit now
passes the `95%` gate at `95.19%` over `554` promoted tests. The next coverage
work should therefore
be treated as architecture hardening, not as percentage chasing. The remaining
high-value targets are:

- `deck_runner.py`: split setup, execution policy, logging/provenance, restart,
  and artifact writing into directly tested helpers
- `solver/implicit.py`: keep the finite-difference sparse path, fallback
  diagnostics, and JAX-linearized path covered as the solver backend boundary
  is split further
- `cli.py`: exercise subcommand branch behavior through focused argument-parser
  and command-dispatch tests rather than only end-to-end CLI runs

## Figure-Producing Tests

Some tests and campaigns are not only correctness checks; they are also figure
generators for the paper and docs. Those should produce:

- machine-readable JSON
- optional NPZ arrays
- publication-grade plots
- direct regression checks for artifact completeness

Publication-grade plots should follow a stricter standard than ad hoc notebook
figures. In practice that means:

- high-resolution export from the validation package itself
- axis labels and titles that name the physical or numerical quantity directly
- support-window cropping when the scientifically meaningful signal is localized
- summary panels such as ratios, residuals, integrated totals, or peak values
  when a full-domain lineout is effectively flat
- figure logic that is derived from the same validated artifact bundle checked
  by the tests, rather than from separate paper-only plotting code

Any test family that demonstrates one of the following should have a paired
artifact-producing campaign, even if the test itself remains assertion-only:

- literature-anchored numerics such as MMS, convergence, or operator studies
- benchmark-facing physics comparisons
- differentiability results that would appear in the paper

The assertion test and the artifact campaign should share the same source logic
wherever possible. The test proves correctness; the campaign proves the result
is communicable and publication-ready.

This applies directly to:

- reconstruction rules and guarded-boundary formulas
- collision closures
- electron-force-balance pressure-gradient stencils
- electron parallel-force source updates, including the use of
  boundary-conditioned ion densities in the electric-force momentum source
- benchmark validation surfaces

If a surface is strong enough to be discussed in the paper, it should already
exist in the docs as a reproducible artifact rather than only as a hidden test
assertion.

Priority figure-producing families are:

- MMS convergence
- neutral parallel-diffusion closure campaign
- collision/conduction closure campaign
- tokamak anomalous-diffusion campaign
- sheath-response campaign
- neutral short-window comparisons
- 3D runtime and convergence campaigns
- differentiability, uncertainty, inverse design, and local throughput

When a campaign includes cross-code relative errors, it should also expose the
absolute-error context needed to interpret near-zero reference fields honestly.

The current promoted example of this policy is:

- [fluid_1d_mms_convergence.md](fluid_1d_mms_convergence.md), which turns the
  manufactured-solution refinement study into the same JSON/NPZ/plot artifact
  surface used by the other publication-facing campaigns

The supported runtime-profiling workflow for those campaigns is now:

- [profiling_runtime.md](profiling_runtime.md)

## CI Gate Target

The final automated gate should be tiered rather than one monolithic slow job:

- fast PR gate: packaging, release surface, unit/operator tests, runtime
  precision/import checks on Python 3.10, 3.11, and 3.12
- research-fast gate: the default slices in
  [scripts/run_fast_research_checks.py](../scripts/run_fast_research_checks.py)
- coverage gate: promoted solver and public-surface slice with the meaningful
  `95%` target
- artifact gate: schema and metric checks for lightweight committed validation
  artifacts
- nightly/manual heavy gate: convergence campaigns, memory profiling, and
  selected performance campaigns

Hosted CI runs the scheduled public slice; heavy profiling campaigns remain
explicit manual/self-hosted campaigns because they need a larger runtime budget.

The hosted GitHub Actions slice is a shipping-surface guard, not the final
research-code gate. Manual CPU/GPU profiling campaigns remain required before
making new research claims that depend on large runtime budgets.

## Immediate Refactor Priorities

During the first structural phase:

1. extract direct unit coverage for pack/unpack and layout logic
2. extract direct operator tests for neutral closures
3. extract direct tests for setup/runtime-model contracts such as field
   evaluation, source normalization, and species-template construction so deck
   interpretation is directly tested
4. extract direct tests for state-preconditioning rules such as density floors,
   guarded neutral reconstruction, and prepared-state assembly so sheath and
   collisional closures are not the only places where those branches are
   exercised
5. extract direct tests for distinct closure families such as neutral parallel
   diffusion and collision closure so those physics packages are directly tested
6. promote extracted tokamak anomalous-transport operators into a public
   artifact-producing campaign once the non-orthogonal metric contrast is
   stable enough to support manuscript-facing figures
7. extract direct tests for field sanitization, restart-policy selection, and
   other small execution rules that influence solver robustness and public
   artifact behavior
8. keep the existing native transient ladders green while files are
   being split
9. only then widen benchmark and literature-facing campaigns

That sequencing preserves scientific trust while the software architecture is
improved.
