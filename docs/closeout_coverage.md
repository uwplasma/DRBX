# Closeout Coverage

`jax_drb` now keeps two different coverage concepts separate:

- broad exploratory coverage for day-to-day iteration;
- bounded release-closeout coverage for the ship decision.

The broad exploratory path is still available through:

```bash
python scripts/run_fast_research_checks.py --with-coverage
```

That is useful for routine work, but it is not the release threshold.

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
