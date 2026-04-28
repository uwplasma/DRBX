# Atomic Rate Differentiability Campaign

This campaign is the first focused differentiability artifact for the
reaction-source path. It audits the packaged AMJUEL, OpenADAS, and hydrogen
charge-exchange rate surfaces before those formulas are promoted into a fully
JAX-native recycling residual.

The generated package contains:

- a summary JSON with derivative parity gates;
- a compact NPZ payload with temperature grids, rates, autodiff derivatives,
  finite-difference derivatives, and derivative errors;
- a publication-grade summary plot.

The literature anchor is the same reaction model family used by Hermes-3:
hydrogenic AMJUEL ionisation/recombination, hydrogen charge exchange, and
OpenADAS impurity reaction tables. The numerical anchor is the JAX autodiff
pattern of differentiating scalar rate functions and batching those derivative
evaluations over the temperature grid.

The figure is intentionally a derivative-validation figure, not a full
recycling-performance claim. It demonstrates that the atomic data surface used
inside the reaction source can be differentiated reliably with JAX and checked
against centered finite differences before the larger residual is moved away
from the host/SciPy implicit path.

Run the demo with:

```bash
python examples/engineering/atomic_rate_differentiability_campaign_demo.py
```

Committed artifacts:

- `docs/data/atomic_rate_differentiability_campaign_artifacts/data/atomic_rate_differentiability_campaign.json`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__atomic_rate_differentiability_campaign_artifacts__data__atomic_rate_differentiability_campaign.npz`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__atomic_rate_differentiability_campaign_artifacts__images__atomic_rate_differentiability_campaign.png`
