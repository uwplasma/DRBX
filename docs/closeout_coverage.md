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
python scripts/run_promoted_solver_coverage.py --audit
```

That command measures the core native solver, recycling, runner, parity, and
CLI surface that must ultimately carry the meaningful research-code `95%`
coverage claim. During the refactor, `--audit` is the correct mode because it
reports the current gap without failing the iteration loop. Once the extracted
solver modules and direct operator tests are complete, the same script should be
run without `--audit` so the default `95%` threshold becomes the promoted-solver
gate:

```bash
python scripts/run_promoted_solver_coverage.py
```

As of the April 23, 2026 local audit, this promoted slice passes its tests
(`209 passed`, `7 deselected`, `1 xfailed`) but reports `73%` total coverage.
That is the actionable baseline for the next refactor pass. The largest
coverage deficits are concentrated in the runner orchestration, monolithic
recycling transient branches, parity compare/diff/reference helpers, and CLI
subcommand branches. Those gaps should be closed by extracting smaller modules
with direct tests, not by adding more broad smoke coverage.

The release-closeout threshold is now explicit and reproducible:

```bash
python scripts/run_closeout_coverage.py
```

This script runs a bounded critical-path slice over:

- controller closeout packages;
- reactions/collisions and impurity/radiation campaigns;
- native 3D runtime/convergence/profile audits;
- Hermes comparison summary and capability audit;
- packaging metadata and PyPI release-workflow checks;
- the public release-surface regression.

It then enforces a `95%` total coverage threshold on that exact closeout slice.

The point of this split is pragmatic: repo-wide monolithic coverage is still too broad and slow to be a credible local release gate, while the closeout slice is fast enough to run repeatedly and is aligned with the modules that currently decide whether `jax_drb` is ready to ship.

This does **not** mean repo-wide coverage is solved. It means the release threshold is now:

- explicit;
- bounded;
- reproducible;
- tied to the actual closeout modules rather than an arbitrary giant command.

The distinction is important. The closeout gate is a bounded release-readiness
check over validation packages and public packaging behavior. The promoted
solver gate is the future research-grade criterion for the native physics and
runtime surface. Both are needed before claiming that the code is ready for
external scientific use.
