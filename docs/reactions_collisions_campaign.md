# Reactions And Collisions Campaign

This campaign turns the native reaction, collisionality, and atomic-data checks
into one reviewable artifact bundle instead of leaving them scattered across unit
tests.

The generated package contains:

- a summary JSON with named gates and pass/fail status;
- a compact NPZ payload with the metric values and targets;
- a summary summary plot.

The current gates cover:

- single-species hydrogen charge-exchange consistency;
- multispecies cross-isotope charge exchange;
- per-species `K_cx_multiplier` application;
- ionisation-rate consistency with reaction diagnostics;
- ion-parallel-viscosity collisionality closure;
- neon OpenADAS table loading.

Run the demo with:

```bash
python examples/engineering/reactions_collisions_campaign_demo.py
```

Committed artifacts:

- `docs/data/reactions_collisions_campaign_artifacts/data/reactions_collisions_campaign.json`
- `docs/data/reactions_collisions_campaign_artifacts/data/reactions_collisions_campaign.npz`
- `docs/data/reactions_collisions_campaign_artifacts/images/reactions_collisions_campaign.png`
