# Neutral Parallel Diffusion Campaign

This campaign turns the extracted neutral parallel-diffusion closure into a
public validation and publication surface. It compares the `AFN` and
`multispecies` collision models on the same prepared D/T/He recycling state,
using the local Hermes-3 `1D-recycling-dthe` deck together with the committed
reference snapshot cache as the state source.

The main literature anchor is the Hermes-3 model paper:
[Dudson et al. 2024](https://www.sciencedirect.com/science/article/pii/S0010465523003363).
Hermes-3 and its documentation describe `AFN` as the modern neutral model and
retain `multispecies` neutral diffusion as a legacy comparison mode. This
campaign keeps that exact comparison and presents it as a species-level closure
study rather than only a hidden operator test.

The generated artifact bundle is:

- [summary JSON](data/neutral_parallel_diffusion_campaign_artifacts/data/neutral_parallel_diffusion_campaign.json)
- [profile arrays](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_parallel_diffusion_campaign_artifacts__data__neutral_parallel_diffusion_campaign.npz)
- [summary figure](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_parallel_diffusion_campaign_artifacts__images__neutral_parallel_diffusion_campaign.png)

The figure reports four closure summaries on the same prepared state:

- species-resolved effective neutral diffusivity under `AFN` and `multispecies`
- `AFN` collision-budget decomposition into ionisation and charge exchange
- `multispecies` collision-budget decomposition into aggregate multispecies
  collisions and charge exchange
- `multispecies / AFN` diffusivity ratio for the deuterium, tritium, and helium
  neutrals

The metrics in the JSON enforce the basic scientific contract for this closure
surface:

- AFN and multispecies diffusivities must remain finite
- the two collision models must produce measurable diffusivity contrast for all
  three neutral species
- charge exchange must remain a non-negligible part of the AFN deuterium
  collision budget on this prepared state

Run the package locally with:

```bash
PYTHONPATH=src .venv/bin/python examples/engineering/neutral_parallel_diffusion_campaign_demo.py
```
