# Native 3D Runtime Campaign

This package collects the current promoted native reduced 3D rungs into one runtime/scaling summary and adds small synthetic scaling sweeps for the non-tokamak reduction kernels.

Run:

```bash
PYTHONPATH=src .venv/bin/python examples/publication/native_3d_runtime_campaign_demo.py \
  --output-root docs/data/native_3d_runtime_campaign_artifacts
```

Committed artifacts:

- `docs/data/native_3d_runtime_campaign_artifacts/data/native_3d_runtime_campaign.json`
- `docs/data/native_3d_runtime_campaign_artifacts/images/native_3d_runtime_campaign.png`

Current scope:

- committed runtime summaries from the native tokamak one-step and short-window reduced rungs;
- committed runtime summaries from the traced-field-line and stellarator native reduced rungs;
- synthetic scaling sweeps for the traced-field-line and stellarator native reduction kernels.
