# Impurity / Radiation Campaign

This package is the first explicit validation bundle for the impurity/radiation family. It does not overclaim detachment or controller closure. It currently covers:

- neon OpenADAS ionisation and recombination table loading;
- finite native radiation-loss evaluation for the neon ionisation/recombination channels;
- exact direct tokamak `D/T/He/Ne` RHS closure on `Nne+`, `Pne+`, and `Pe`.

Run:

```bash
PYTHONPATH=src .venv/bin/python examples/engineering/impurity_radiation_campaign_demo.py
```

Committed artifacts:

- `docs/data/impurity_radiation_campaign_artifacts/data/impurity_radiation_campaign.json`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__impurity_radiation_campaign_artifacts__data__impurity_radiation_campaign.npz`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__impurity_radiation_campaign_artifacts__images__impurity_radiation_campaign.png`

Current claim boundary:

- impurity/radiation data loading and neon-enabled RHS closure are now explicitly gated;
- controller-oriented temperature-feedback and detachment-control workflows are still open and remain outside the promoted surface.
