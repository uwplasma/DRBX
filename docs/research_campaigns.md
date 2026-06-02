# Research Campaigns

This page defines the reproducible campaign layer used to keep the codebase
research-grade without turning every pull request into a multi-hour run.

The public entry point is:

```bash
python scripts/run_research_campaign_bundle.py --campaign scheduled-fast-research
```

The scheduled GitHub Actions workflow runs the same bounded public research
slice weekly. It does not require external reference checkouts and therefore
stays suitable for hosted CI. Longer live-reference and heavy profiling runs
are exposed through the same wrapper, but they are intended for local or
self-hosted machines where the reference checkout and heavier runtime budget
are available.

## Campaign Bundles

Use the CI-safe bundle for scheduled hosted checks:

```bash
python scripts/run_research_campaign_bundle.py --campaign all-ci
```

Use the local live-reference bundle when the external reference checkout is
available:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign live-reference \
  --reference-root /path/to/reference/root
```

Use the heavy recycling runtime bundle after solver changes:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign heavy-recycling-profile \
  --reference-root /path/to/reference/root
```

Use the fixed-layout D/T/He JAX-linearized residual gate when changing the
JAX-native recycling residual seam:

```bash
python scripts/run_research_campaign_bundle.py \
  --campaign dthe-jax-linearized-gate \
  --reference-root /path/to/reference/root
```

The `all-local` bundle runs the fast public slice, local CPU scaling, the
D/T/He JAX-linearized residual gate, the full heavy recycling cProfile/RSS
profile, and the live-reference matrix. It should only be used on machines
where multi-hour runs are acceptable.

## Current Evidence

The live-reference matrix remains the primary code-to-code fidelity dashboard.
It identifies the neutral mixed `NVh` operator mismatch as the main fidelity
offender and the heavy D/T/He recycling one-step path as the main runtime
offender. Those results are documented in
[hermes_live_rerun_campaign.md](hermes_live_rerun_campaign.md) and
[runtime_gap_remediation.md](runtime_gap_remediation.md).

The local CPU scaling evidence is the heavy fixed-work ensemble in
[local_cpu_scaling_campaign.md](local_cpu_scaling_campaign.md). It uses
repeated direct tokamak recycling solves rather than a synthetic microkernel,
and the committed artifact reaches about `4.79x` steady-state speedup from
`1 -> 8` worker processes on the retained `16`-solve ensemble.

The D/T/He fixed-layout JAX-linearized residual gate now has both CPU and GPU
profile summaries under `docs/data/runtime_profile_artifacts/`. On the current
small `950`-active-variable gate, the CPU run completes in about `4.74 s` with
about `5.0 GiB` sampled peak process-tree RSS. The office GPU run reaches the
same residual norm and cuts sampled peak RSS to about `1.4 GiB`, but the warm
wall time remains about `6.66 s`. That is useful evidence that the seam is
accelerator-executable and lower-memory; it is not yet a speedup claim because
this problem size is too small and still dominated by compile/launch overhead.

## Promotion Policy

A validation or performance campaign is promoted into the README or paper plan
only when it satisfies four conditions:

- it is tied to a named physics, numerical, or differentiability claim;
- it has a deterministic script entry point;
- it writes JSON evidence plus a publication-ready figure or profile bundle;
- it states the limitation of the result instead of extrapolating beyond the
  measured case.

For the remaining recycling solver work, the order is therefore fixed:

- keep the stable production BDF path as the default while it is the only
  fully validated output-window path;
- use `JAX_DRB_RECYCLING_BDF_JACOBIAN_MODE=jvp` only as an opt-in derivative
  experiment on transformable residual callbacks;
- continue moving source, closure, boundary, and target-recycling kernels into
  fixed-layout JAX functions with parity and JVP gates;
- promote matrix-free/JVP nonlinear solves only after the full heavy residual
  is transformable and has passed live-reference and runtime campaigns.
