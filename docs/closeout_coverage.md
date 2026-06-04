# Closeout Coverage

`jax_drb` now keeps two different coverage concepts separate:

- broad exploratory coverage for day-to-day iteration;
- bounded release-closeout coverage for the ship decision.

The broad exploratory path is still available through:

```bash
python scripts/run_fast_research_checks.py --with-coverage
```

That is useful for routine work, but it is not the release threshold.

The promoted solver and public API surface now has a separate audit/gate entry
point:

```bash
python scripts/run_promoted_solver_coverage.py
```

That command measures the core native solver, recycling, runner, parity, and
CLI surface that must ultimately carry the meaningful research-code `95%`
coverage claim. The default mode enforces the `95%` threshold. During risky
refactor work, `--audit` remains available when the desired behavior is to
report the current coverage without failing the iteration loop:

```bash
python scripts/run_promoted_solver_coverage.py --audit
```

As of the June 4, 2026 local audit, this promoted slice passes its tests
(`463 passed`, `7 deselected`, `1 xfailed`) and reports `95%` total coverage.
The remaining coverage deficits are concentrated in the hardest and most
valuable surfaces: runner orchestration, full-sheath lower-target branches,
legacy mixed residual helpers, target-recycling edge cases, and some CLI
subcommand branches. Future coverage work should still close those through
smaller extracted modules and operator tests, not through broad smoke coverage.

The release-closeout threshold is now explicit and reproducible:

```bash
python scripts/run_closeout_coverage.py
```

This script runs a bounded critical-path slice over:

- controller closeout packages;
- reactions/collisions and impurity/radiation campaigns;
- open-field operator verification for parallel gradients, force balance,
  target recycling, and autodiff sensitivity;
- native 3D runtime/convergence/profile audits;
- Hermes comparison summary and capability audit;
- packaging metadata and PyPI release-workflow checks;
- the public release-surface regression.

It then enforces a `95%` total coverage threshold on that exact closeout slice.
The latest local run reported `97%` total closeout coverage.

The point of this split is pragmatic: repo-wide monolithic coverage is still too broad and slow to be a credible local release gate, while the closeout slice is fast enough to run repeatedly and is aligned with the modules that currently decide whether `jax_drb` is ready to ship.

This does **not** mean repo-wide coverage is solved. It means the release thresholds are now:

- explicit;
- bounded;
- reproducible;
- tied to the actual closeout modules rather than an arbitrary giant command.

The distinction is important. The closeout gate is a bounded release-readiness
check over validation packages and public packaging behavior. The promoted
solver gate is the research-grade criterion for the native physics and runtime
surface. Both gates now enforce `95%`; neither replaces operator-level
validation, reference parity, profiling, or publication-grade figure review.
Both gates are required by the GitHub Actions coverage workflow on pushes and
pull requests, so the documented `95%` claim is enforced by CI rather than left
as a manual release step.
