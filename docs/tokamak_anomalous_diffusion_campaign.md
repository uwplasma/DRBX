# Tokamak Anomalous Diffusion Campaign

This campaign promotes the extracted anomalous-diffusion operator family into a
public tokamak validation surface. It evaluates the non-orthogonal anomalous
transport closure on the evolved D/T/He recycling state used by the direct
tokamak short-window ladder and compares it against the same closure with the
non-orthogonal metric coefficients suppressed.

The main literature anchors are the mapped-grid tokamak transport practice used
in Hermes-3 and related edge/SOL fluid codes, together with non-orthogonal
tokamak transport formulations such as TOKAM3X. The point of the campaign is
not to claim a full end-to-end parity result on its own. It is to show that the
extracted anomalous operator produces a measurable, geometry-driven transport
contrast on a realistic evolved tokamak recycling state.

The generated artifact bundle is:

- [summary JSON](data/tokamak_anomalous_diffusion_campaign_artifacts/data/tokamak_anomalous_diffusion_campaign.json)
- [summary arrays](data/tokamak_anomalous_diffusion_campaign_artifacts/data/tokamak_anomalous_diffusion_campaign.npz)
- [summary figure](data/tokamak_anomalous_diffusion_campaign_artifacts/images/tokamak_anomalous_diffusion_campaign.png)

The figure reports four summaries:

- the configured anomalous coefficients for the active species
- the relative energy-transport contrast between orthogonal and
  non-orthogonal tokamak metrics
- a representative `d+` anomalous-energy lineout on the evolved recycling state
- a representative `t+` anomalous-energy lineout on the same state

The metrics in the JSON enforce the basic scientific contract for this surface:

- the electron anomalous `D` coefficient must match the `d+` literal-reference
  configuration on the tokamak deck
- the non-orthogonal metric terms must produce a material energy-transport
  contrast for `d+`, `t+`, and `he+`
- the electron density transport must retain a measurable non-orthogonal
  contrast

Run the package locally with:

```bash
PYTHONPATH=src .venv/bin/python examples/engineering/tokamak_anomalous_diffusion_campaign_demo.py
```
