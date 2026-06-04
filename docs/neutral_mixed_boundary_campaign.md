# Neutral Mixed Boundary Campaign

This campaign isolates the current `neutral_mixed_one_step` mismatch against a
live Hermès-3 rerun and turns it into a publication-grade artifact instead of a
single summary scalar. The figure follows the literature pattern used for
open-field parallel-profile comparisons: fixed cross-field lineouts for the
state variables, together with a separate absolute-error panel that shows where
the mismatch is localized.

The immediate purpose is twofold. First, it gives the refactor/runtime campaign
an auditable fidelity surface for the remaining neutral-mixed mismatch in the
same live rerun framework used by the broader Hermès matrix. Second, it creates
a manuscript-ready figure family that can sit next to that broader matrix and
show what the mismatch actually looks like in parallel space rather than only
as a dashboard scalar.

The generated artifact bundle is:

- [summary JSON](data/neutral_mixed_boundary_campaign_artifacts/data/neutral_mixed_boundary_campaign.json)
- [summary arrays](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_boundary_campaign_artifacts__data__neutral_mixed_boundary_campaign.npz)
- [summary figure](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_boundary_campaign_artifacts__images__neutral_mixed_boundary_campaign.png)

The figure reports:

- Hermès-3 versus JAX-DRB parallel lineouts for `Nh` at the `x,z` location of
  the worst `Nh` error
- the same for `Ph`
- the same for `NVh`
- the corresponding `max_{x,z} |Δ|(y)` profiles so boundary localization is
  visible even when a single centerline looks cleaner than the full field

The current package was regenerated after the neutral mixed history integrator
started using internal BDF substeps for the one-step and short-window parity
surfaces. The maximum `NVh` history error is now about `4.47e-6`, down from
the earlier `5.81e-4` and `3.37e-3` boundary-local mismatches. The worst
scalar neutral-state error is now on `Nh`, about `2.19e-4`, with `Ph` at about
`2.11e-5`. The campaign therefore separates the closed neutral-momentum
source-formula question from the remaining density/pressure boundary-history
fidelity work. The committed JSON is also used as a non-env-gated regression
surface, so future solver or boundary changes must either preserve these ranked
maxima or intentionally refresh the artifact with stronger parity evidence.

Run the package locally with:

```bash
PYTHONPATH=src .venv/bin/python examples/engineering/neutral_mixed_boundary_campaign_demo.py
```
