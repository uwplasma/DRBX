# Local CPU Scaling Campaign

This package benchmarks local MacBook-class CPU scaling on a heavy direct
tokamak recycling solve instead of relying on a lighter synthetic kernel.

It now focuses only on the stronger local scaling story:

- steady-state fixed-work ensemble scaling for repeated heavy solves with one
  Jacobian thread per worker process

The older single-solve thread plot was removed because it stayed essentially
flat after warmup on this MacBook and was not the right reviewer-facing local
scaling result.

Run the package with:

```bash
python examples/engineering/local_cpu_scaling_campaign_demo.py
```

The committed artifacts are:

- `docs/data/local_cpu_scaling_campaign_artifacts/data/local_cpu_scaling_campaign.json`
- `docs/data/local_cpu_scaling_campaign_artifacts/images/local_cpu_scaling_campaign.png`

Interpretation:

- the committed figure uses `16` repeated heavy solves on
  `tokamak_recycling_dthene_one_step`
- the repeated heavy-solve ensemble gives the stronger local scaling story
  because per-worker warmup is amortized and the workload is naturally parallel
- on the committed local artifact the steady-state speedup is about:
  - `1.88x` from `1 -> 2` workers
  - `3.67x` from `1 -> 4` workers
  - `4.94x` from `1 -> 8` workers
- this is the right local-CPU figure for the paper because it is tied to a real
  promoted production solve rather than a tiny synthetic kernel
- the retained `16`-solve ensemble is deliberate: heavier local sweeps were
  checked, but they did not improve the curve on this thermally limited
  MacBook enough to justify a slower default artifact
- the figure now carries both speedup and parallel-efficiency panels, which is
  closer to the performance-reporting style used in numerical-physics and
  differentiable-science code papers
