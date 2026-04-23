# Testing Strategy

This page explains how `jax_drb` should be tested during the refactor and what
counts as meaningful evidence for a research-grade release.

The active cross-cutting execution plan is
[research_grade_execution_plan.md](research_grade_execution_plan.md). This page
keeps the testing policy narrower and should be read as the gate definition for
that plan.

The testing philosophy follows the verification and validation split discussed
in [Roy 2005](https://www.sciencedirect.com/science/article/pii/S0021999104004747)
and the practice used in major edge/SOL codes such as
[Hermes-3](https://www.sciencedirect.com/science/article/pii/S0010465523003363),
[GBS](https://www.sciencedirect.com/science/article/pii/S0021999122003280), and
the [TCV-X21 benchmark program](https://arxiv.org/abs/2109.01618).

## Evidence Layers

`jax_drb` should be tested in layers. No single layer is sufficient by itself.

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

This layer compares `jax_drb` against curated reference outputs, primarily from
Hermes-3-backed or staged reference bundles.

This is the main current bridge for:

- open-field recycling ladders
- direct tokamak recycling windows
- same-machine live native-versus-Hermes rerun matrices across representative
  1D and 2D lanes
- selected-field geometry comparisons
- controller and detachment history matching

### Benchmark and experiment-facing validation

This layer compares the promoted workflows against broader literature benchmarks
and diagnostics, such as:

- TCV-X21
- TORPEX or X-point blob validation
- detachment scaling
- geometry-portability studies

This is the layer that most directly supports publication figures.

## Target Test Taxonomy

The suite should converge toward explicit logical groups:

- unit/operator tests
- regression tests
- parity/reference tests
- autodiff/JAX transformation tests
- publication/campaign tests
- smoke/tutorial tests

The current on-disk tree is still mostly flat, but the refactor should make the
distinction explicit in filenames, markers, and CI slices before or during any
directory migration.

## Coverage Policy

The `95%` target should mean:

- strong coverage on the promoted solver surface
- direct coverage of extracted operators and closure branches
- public-surface coverage for CLI, campaigns, and artifact generation

It should not mean:

- relying on smoke tests to color large files green
- excluding the hard solver modules from the measurement
- claiming broad coverage when only compact differentiable examples are tested

The release standard from the refactoring roadmap is:

- at least `95%` on the promoted solver and public-surface slice
- no critical operator family below `90%`
- no monolithic module left effectively untested except through one large
  integration case

The explicit promoted-solver coverage entry point is now:

```bash
python scripts/run_promoted_solver_coverage.py --audit
```

That audit covers the native mesh/metric/open-field/recycling/runner surface,
portable parity helpers, and CLI entry points. It exists to prevent the project
from confusing the narrower release-closeout coverage gate with meaningful
solver coverage. During the refactor, use audit mode to identify the largest
uncovered modules and branches. When the extraction work has enough direct unit,
operator, parity, and artifact-producing tests, the same command should be run
without `--audit` and must satisfy the default `95%` threshold before the solver
surface is called research-grade.

The first measured baseline for this slice, run locally on April 23, 2026, is
`73%` total coverage with all selected tests passing. The top coverage targets
for the next implementation pass are:

- `runner.py`: split setup, execution policy, logging/provenance, restart, and
  artifact writing into directly tested helpers
- `recycling_1d.py`: extract residual assembly, continuation control, neutral
  reconstruction, pressure/source preparation, and target recycling branches
- `parity/diff.py`, `parity/compare.py`, and `parity/reference.py`: add direct
  tests for guard semantics, missing-field behavior, normalization modes, and
  failure reporting
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
- controller, detachment, or recycling transient histories that are
  scientifically interpretable
- differentiability results that would appear in the paper

The assertion test and the artifact campaign should share the same source logic
wherever possible. The test proves correctness; the campaign proves the result
is communicable and publication-ready.

This applies directly to:

- fits and tabulated-rate evaluations
- reconstruction rules and guarded-boundary formulas
- reaction and collision closures
- parity and benchmark validation surfaces

If a surface is strong enough to be discussed in the paper, it should already
exist in the docs as a reproducible artifact rather than only as a hidden test
assertion.

Priority figure-producing families are:

- MMS convergence
- open-field parallel-gradient, force-balance, and target-recycling operator
  identities
- reactions, collisions, and atomic-data closure campaigns
- neutral parallel-diffusion closure campaign
- neutral mixed boundary-mismatch campaign
- collision/conduction closure campaign
- tokamak anomalous-diffusion campaign
- target-recycling and sheath-response campaign
- live Hermès rerun matrix across representative 1D and 2D lanes
- direct tokamak recycling transient ladders
- neutral short-window comparisons
- 3D runtime and convergence campaigns
- differentiability, uncertainty, inverse design, and local throughput

When a campaign includes cross-code relative errors, it should also expose the
absolute-error context needed to interpret near-zero reference fields honestly.
The current live Hermès rerun matrix is the concrete example: integrated and
direct tokamak recycling still show bounded relative mismatch on `NVd`, but the
campaign now also exposes the tiny absolute max-errors so the paper and docs do
not overstate that class of discrepancy.

The current promoted example of this policy is:

- [fluid_1d_mms_convergence.md](fluid_1d_mms_convergence.md), which turns the
  manufactured-solution refinement study into the same JSON/NPZ/plot artifact
  surface used by the other publication-facing campaigns
- [hermes_live_rerun_campaign.md](hermes_live_rerun_campaign.md), which now
  ties same-machine native-versus-live-Hermès fidelity to runtime and absolute
  error on the same guarded compare surface

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
- nightly/manual heavy gate: live reference reruns, convergence campaigns,
  memory profiling, and selected performance campaigns

Until CI billing is available again, the narrow GitHub Actions slice is a
shipping-surface guard, not the final research-code gate.

## Immediate Refactor Priorities

During the first structural phase:

1. extract direct unit coverage for pack/unpack and layout logic
2. extract direct operator tests for recycling and neutral closures
3. extract direct tests for controller-state logic and compare-window helpers
   so validation/orchestration behavior is not only inherited through large
   runner and transient-solver tests
4. extract direct tests for setup/runtime-model contracts such as field
   evaluation, source normalization, species-template construction, and
   controller loading so deck interpretation is not only inherited through the
   full recycling solver
5. extract direct tests for state-preconditioning rules such as density floors,
   guarded neutral reconstruction, and prepared-state assembly so sheath and
   collisional closures are not the only places where those branches are
   exercised
6. extract direct tests for distinct closure families such as neutral parallel
   diffusion, collision closure, and controller-source assembly so those
   physics packages are not only exercised through the full recycling RHS
7. promote extracted tokamak anomalous-transport operators into a public
   artifact-producing campaign once the non-orthogonal metric contrast is
   stable enough to support manuscript-facing figures
8. extract direct tests for field sanitization, restart-policy selection, and
   other small execution rules that influence solver robustness and public
   artifact behavior
9. keep the existing Hermes-backed transient ladders green while files are
   being split
10. only then widen benchmark and literature-facing campaigns

That sequencing preserves scientific trust while the software architecture is
improved.
