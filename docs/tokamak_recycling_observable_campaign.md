# Tokamak Recycling Observable Campaign

This campaign turns the direct-tokamak `D/T/He` recycling parity surface into a
profile-level validation artifact. The goal is to follow the evidence style used
in TCV-X21, SOLPS-ITER, and Hermes-3 validation studies: profile observables at
target-index rows, neutral buildup along the parallel coordinate, and
observable-level error bars rather than only a single scalar parity metric.

The generated artifact bundle is:

- [summary JSON](data/tokamak_recycling_observable_campaign_artifacts/data/tokamak_recycling_observable_campaign.json)
- [summary arrays](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_recycling_observable_campaign_artifacts__data__tokamak_recycling_observable_campaign.npz)
- [summary figure](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_recycling_observable_campaign_artifacts__images__tokamak_recycling_observable_campaign.png)

![Tokamak recycling observable campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_recycling_observable_campaign_artifacts__images__tokamak_recycling_observable_campaign.png)

The figure reports:

- charged `D+`, `T+`, and `He+` density profiles at the lower target-index row;
- `|NV_s+|` target momentum-flux proxies at the upper target-index row;
- neutral `D`, `T`, and `He` density buildup along the parallel index;
- observable-level native/Hermes differences for density, flux, and target
  electron-temperature proxy profiles.

The lower-target density, lower-target flux, neutral-density, and target
electron-temperature proxy observables are tight on this surface. The upper
target-index `|NV_s+|` profiles remain the visible diagnostic outlier: their
absolute amplitudes are small compared with the lower target, but the relative
profile error is large enough that it should stay in the offender register until
the direct-tokamak target-boundary update is tightened further.

The target and neutral-observable focus is deliberate. TCV-X21 established that
outer-midplane profiles can be substantially cleaner than divertor and target
profiles, while subsequent SOLPS-ITER and Hermes-3 work emphasized the role of
neutral dynamics, target profile shifts, and ionisation-source localization in
interpreting remaining discrepancies. This campaign therefore acts as the
paper-facing bridge between the exact one-step compare surface and the physical
observables that reviewers expect for tokamak edge/SOL validation.

Regenerate the package locally with:

```bash
PYTHONPATH=src python examples/engineering/tokamak_recycling_observable_campaign_demo.py
```
