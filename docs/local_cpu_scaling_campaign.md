# Local CPU Scaling Campaign

This package benchmarks local MacBook-class CPU scaling on a heavy direct tokamak
recycling solve instead of relying on a lighter synthetic kernel.

It produces two curves:

- single-solve scaling from `JAX_DRB_FD_JACOBIAN_THREADS=<N>` on
  `tokamak_recycling_dthe_one_step`
- steady-state fixed-work ensemble scaling for repeated heavy solves with one
  Jacobian thread per worker process

The first curve answers the user-facing question "how much faster does one
heavy solve get if I give it more local CPU threads?" The second curve answers
the stronger reviewer-facing question "how well do repeated heavy local solves
scale after warmup for UQ, parameter scans, and optimization workloads?"

Run the package with:

```bash
python examples/engineering/local_cpu_scaling_campaign_demo.py
```

The committed artifacts are:

- `docs/data/local_cpu_scaling_campaign_artifacts/data/local_cpu_scaling_campaign.json`
- `docs/data/local_cpu_scaling_campaign_artifacts/images/local_cpu_scaling_campaign.png`

Interpretation:

- the single heavy solve shows a real but bounded local speedup from threaded
  sparse finite-difference Jacobian assembly
- the repeated heavy-solve ensemble gives the stronger local scaling story
  because per-worker warmup is amortized and the workload is naturally
  parallel
- this is the right local-CPU figure for the paper because it is tied to a real
  promoted production solve rather than a tiny synthetic kernel
