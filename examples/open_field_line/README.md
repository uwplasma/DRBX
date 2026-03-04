# Open Field-Line Example

This example runs a field‑aligned s‑alpha configuration with an open/closed SOL
mask and poloidal‑plane visualizations that highlight the separatrix structure.
The SOL uses a **Bohm sheath‑loss closure** via `sol_parallel_loss_on=true`.

## Run

```bash
python examples/open_field_line/run.py --make-figures --make-movies
```

Outputs:
- `examples/open_field_line/output.npz`
- `docs/figures/open_field_poloidal_eq.png`
- `docs/figures/open_field_poloidal_fluct.png`
- `docs/figures/open_field_movie.gif`

## Hermes Benchmark Workflow

```bash
PYTHONPATH=src python tools/run_tokamak_hermes_benchmark.py \
  --jax-config examples/open_field_line/input_tokamak_bxcv_benchmark_es_cold.toml \
  --hermes-data runs/hermes_open_field_short/data \
  --out-dir runs/tokamak_benchmark_latest \
  --fig-dir docs/figures \
  --t-end-short 0.1 \
  --t-end-visual 1.2
```

This generates:
- short-window Hermes-vs-jax alignment panel (`docs/figures/tokamak_sol_benchmark_panel.png`)
- poloidal fluctuation snapshot (with open/closed overlay when `mask_open` is available in coefficients)
- poloidal and 3D tokamak turbulence GIFs
