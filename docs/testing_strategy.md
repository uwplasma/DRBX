# Testing Strategy

This page explains how `jax_drb` should be tested during the refactor and what
counts as meaningful evidence for a research-grade release.

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

## Figure-Producing Tests

Some tests and campaigns are not only correctness checks; they are also figure
generators for the paper and docs. Those should produce:

- machine-readable JSON
- optional NPZ arrays
- publication-grade plots
- direct regression checks for artifact completeness

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
- reactions, collisions, and atomic-data closure campaigns
- direct tokamak recycling transient ladders
- neutral short-window comparisons
- 3D runtime and convergence campaigns
- differentiability, uncertainty, inverse design, and local throughput

The current promoted example of this policy is:

- [fluid_1d_mms_convergence.md](fluid_1d_mms_convergence.md), which turns the
  manufactured-solution refinement study into the same JSON/NPZ/plot artifact
  surface used by the other publication-facing campaigns

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
6. extract direct tests for field sanitization, restart-policy selection, and
   other small execution rules that influence solver robustness and public
   artifact behavior
7. keep the existing Hermes-backed transient ladders green while files are
   being split
8. only then widen benchmark and literature-facing campaigns

That sequencing preserves scientific trust while the software architecture is
improved.
