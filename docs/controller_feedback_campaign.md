# Controller Feedback Campaign

This package turns the native upstream-density feedback path into a bounded reference-backed gate instead of leaving controller semantics only in ad hoc diagnostics.

It currently covers the promoted single-species recycling controller surface:
- reference-vs-native dense-history controller multiplier `density_feedback_src_mult_d+`;
- reference-vs-native proportional and integral controller terms;
- reference-vs-native reconstructed controller integral history;
- reference-vs-native target recycling source history on the same controlled path.

Run it with:

```bash
PYTHONPATH=src .venv/bin/python examples/engineering/controller_feedback_campaign_demo.py
```

Artifacts:
- `docs/data/controller_feedback_campaign_artifacts/data/controller_feedback_campaign.json`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__controller_feedback_campaign_artifacts__data__controller_feedback_campaign.npz`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__controller_feedback_campaign_artifacts__images__controller_feedback_campaign.png`

Claim boundary:
- this is a real controller-oriented validation surface for the native density-feedback path;
- it does not overclaim `temperature_feedback` or `detachment_controller`, which remain open.
