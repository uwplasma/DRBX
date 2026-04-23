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
- [summary arrays](data/neutral_mixed_boundary_campaign_artifacts/data/neutral_mixed_boundary_campaign.npz)
- [summary figure](data/neutral_mixed_boundary_campaign_artifacts/images/neutral_mixed_boundary_campaign.png)

The figure reports:

- Hermès-3 versus JAX-DRB parallel lineouts for `Nh` at the `x,z` location of
  the worst `Nh` error
- the same for `Ph`
- the same for `NVh`
- the corresponding `max_{x,z} |Δ|(y)` profiles so boundary localization is
  visible even when a single centerline looks cleaner than the full field

Run the package locally with:

```bash
PYTHONPATH=src .venv/bin/python examples/engineering/neutral_mixed_boundary_campaign_demo.py
```
