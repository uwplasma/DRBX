# Reactions And Collisions Campaign

This campaign turns the native reaction, collisionality, and atomic-data checks
into one reviewable artifact bundle instead of leaving them scattered across unit
tests.

The generated package contains:

- a summary JSON with named gates and pass/fail status;
- a compact NPZ payload with the scalar gates plus profile lineouts;
- a publication-grade summary plot.

The current gates cover:

- single-species hydrogen charge-exchange consistency;
- multispecies cross-isotope charge exchange;
- per-species `K_cx_multiplier` application;
- ionisation-rate consistency with reaction diagnostics;
- ion-parallel-viscosity collisionality closure;
- neon OpenADAS table loading.

The current lineouts make the package communicable as a paper-facing closure
verification figure rather than only as a scalar gate. The committed figure now
shows:

- single-species ionisation profile agreement between the assembled collision
  rate and the reaction diagnostic per neutral density;
- multispecies deuterium neutral charge-exchange decomposition into same-isotope
  and cross-isotope ion contributions;
- ion-parallel-viscosity total collisionality agreement between the assembled
  closure input and the explicit collision stack.

This is the first intended bridge between the refactored recycling operators and
future manuscript figures on reaction, collision, and atomic-data fidelity. The
unit tests prove the formulas; this campaign proves the same surfaces are
reviewable and publication-ready.

Run the demo with:

```bash
python examples/engineering/reactions_collisions_campaign_demo.py
```

Committed artifacts:

- `docs/data/reactions_collisions_campaign_artifacts/data/reactions_collisions_campaign.json`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__reactions_collisions_campaign_artifacts__data__reactions_collisions_campaign.npz`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__reactions_collisions_campaign_artifacts__images__reactions_collisions_campaign.png`
