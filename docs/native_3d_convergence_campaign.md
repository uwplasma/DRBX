# Native 3D Convergence Campaign

This package adds an explicit convergence gate to the reduced native 3D program instead of leaving the 3D story at runtime and parity only.

It currently measures the observed order of the native traced-field-line radial-profile reduction on a smooth analytic field family whose exact radial average is known.

Run it with:

```bash
PYTHONPATH=src .venv/bin/python examples/publication/native_3d_convergence_campaign_demo.py
```

Artifacts:
- `docs/data/native_3d_convergence_campaign_artifacts/data/native_3d_convergence_campaign.json`
- `docs/data/native_3d_convergence_campaign_artifacts/images/native_3d_convergence_campaign.png`

Claim boundary:
- this is an operator-level convergence gate on the promoted reduced non-tokamak native surface;
- it does not replace full PDE convergence studies on future native 3D solver lanes.
