# Detachment Controller Campaign

This package turns the reduced detachment-controller example into a bounded Hermes-backed validation lane instead of leaving detachment control only in source-code audit notes.

The current reduced lane is intentionally narrow and explicit:

- it stages the `tokamak-1D/extra/1D-recycling-with-detachment-control` example on a small `cvode`-compatible mesh;
- it strips the `beuler`-only solver options that would otherwise make the reduced deck fail input validation under the local non-PETSc reference build;
- it sets `settling_time = 0` so the bounded window actually exercises the controller law instead of spending the whole probe inside the original settling period;
- it validates the controller identities that are visible on the saved diagnostics:
  - proportional term from front-location error;
  - multiplier balance `control_offset + P + I + D`;
  - source balance `detachment_control_src_mult * detachment_control_src_shape`;
  - nontrivial control response over the bounded window.

Run it with:

```bash
PYTHONPATH=src .venv/bin/python examples/engineering/detachment_controller_campaign_demo.py
```

Artifacts:

- `docs/data/detachment_controller_campaign_artifacts/data/detachment_controller_campaign.json`
- `docs/data/detachment_controller_campaign_artifacts/data/detachment_controller_campaign.npz`
- `docs/data/detachment_controller_campaign_artifacts/images/detachment_controller_campaign.png`

Claim boundary:

- this is the first genuinely bounded `detachment_controller` lane on the local reference build;
- it is a reduced controller-validation surface, not a full detachment-production workflow claim;
- the broader impurity/radiation/detachment family still remains open beyond this reduced promoted lane.
