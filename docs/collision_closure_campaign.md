# Collision Closure Campaign

This campaign turns the extracted collisional closure family into a public
validation and publication surface. It evaluates the Braginskii-style
friction, ion-viscosity, and conduction closures on the prepared D/T/He
recycling reference state used elsewhere in the open-field validation ladder.

The main literature anchors are the collisional transport formulas introduced
by Braginskii and the multispecies edge/SOL closure practice documented in the
Hermes-3 model paper. The goal here is not another end-to-end transient match;
it is a closure-level audit that shows the extracted collision module produces
active, finite, and internally balanced physics terms on a realistic prepared
state.

The generated artifact bundle is:

- [summary JSON](data/collision_closure_campaign_artifacts/data/collision_closure_campaign.json)
- [summary arrays](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__collision_closure_campaign_artifacts__data__collision_closure_campaign.npz)
- [summary figure](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__collision_closure_campaign_artifacts__images__collision_closure_campaign.png)

The figure reports four closure summaries:

- species-resolved parallel ion-viscosity activity
- key collisional-friction activity for representative species pairs
- active-point conduction collision times for all pressure-evolving species
- frictional heating activity for the same representative pairs

The metrics in the JSON enforce the basic scientific contract for this closure
surface:

- the ion-viscosity forcing terms must remain active for the ion species
- selected friction diagnostics must satisfy action-reaction balance
- conduction collision times must remain finite for all pressure-evolving
  species

Run the package locally with:

```bash
PYTHONPATH=src .venv/bin/python examples/engineering/collision_closure_campaign_demo.py
```
