# Coverage

Coverage for `jax_drb` is a single whole-package number: branch coverage of
`src/jax_drb` measured by running the full fast test suite.

```bash
pytest -q -m "not slow" --cov=jax_drb --cov-branch --cov-report=term
```

The CI `coverage` workflow runs the same command. The target is 95% or higher
once the Phase 1 consolidation of the v2 plan lands (see
[`plan_jax_drb.md`](https://github.com/uwplasma/jax_drb/blob/main/plan_jax_drb.md));
until then the workflow records the honest baseline without failing the build.

There are no curated file lists, no "closeout" or "promoted-surface" coverage
subsets, and no coverage gates that measure anything other than the package as
a whole. The earlier bounded gates (`run_closeout_coverage.py`,
`run_promoted_solver_coverage.py`) were removed in the v2 plan's Phase 0
because they reported 95% over roughly 30 hand-picked files out of 163.
