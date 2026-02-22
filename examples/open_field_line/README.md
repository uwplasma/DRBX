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
