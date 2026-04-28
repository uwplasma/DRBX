# Target Recycling Campaign

This campaign promotes the extracted target-recycling support layer into a
public closure and boundary-condition surface. It evaluates target recycling
and the associated boundary-conditioned electron response on the prepared
multispecies `1D-recycling-dthe` state reconstructed from the committed RHS
snapshot.

The main literature anchors are the sheath and wall boundary-condition
formulations used in reduced-fluid edge/SOL modelling together with the
multispecies recycling closures documented by Hermes-3. The goal is not another
end-to-end transient parity result. It is a prepared-state audit that shows the
recycling source partition and the electron boundary sink remain active,
finite, and interpretable on a realistic multispecies open-field state.

The generated artifact bundle is:

- [summary JSON](data/target_recycling_campaign_artifacts/data/target_recycling_campaign.json)
- [summary arrays](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__target_recycling_campaign_artifacts__data__target_recycling_campaign.npz)
- [summary figure](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__target_recycling_campaign_artifacts__images__target_recycling_campaign.png)

The figure reports four summaries:

- target recycling density-source lineouts for `d`, `t`, and `he` in the
  target-support window
- integrated recycling source totals for the same neutral species
- the boundary-conditioned electron energy sink in the same target-support
  window
- species-resolved peak target recycling strength on the same prepared state

The metrics in the JSON enforce the basic scientific contract for this surface:

- the `d`, `t`, and `he` target recycling sources must remain active on the
  prepared state
- the boundary-conditioned electron energy sink must remain active
- the current-free electron velocity reconstruction must stay finite even
  though it is carried as a numerical diagnostic rather than a plotted panel

Run the package locally with:

```bash
PYTHONPATH=src .venv/bin/python examples/engineering/target_recycling_campaign_demo.py
```
