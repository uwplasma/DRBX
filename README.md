# jax_drb (Unified DRB)

Research‑grade **drift‑reduced Braginskii** solver centered on a **single unified system**.
All variants (ES/EM, hot/cold ions, sheath/no‑sheath, Boussinesq/non‑Boussinesq,
1D/2D/3D, linear/nonlinear) are **toggles** on the same core RHS.

![Tokamak SOL benchmark panel](docs/figures/tokamak_sol_benchmark_panel.png)
![Tokamak SOL movie](docs/figures/tokamak_sol_movie.gif)
![Tokamak SOL 3D cut movie](docs/figures/tokamak_sol_3d_movie.gif)

## Quick Start
```
jaxdrb path/to/input.toml
```

## Documentation
- Run + CLI: `docs/run.md`
- Inputs & outputs: `docs/inputs_outputs.md`
- Options & toggles: `docs/options.md`
- Normalization: `docs/normalization.md`
- Geometry models: `docs/geometry_models.md`
- Geometry consistency checks: `docs/geometry_compare.md`
- Validation & tests: `docs/validation.md`
- Benchmark workflow: `docs/benchmarks/open_field_alignment.md`
- Profiling: `docs/profiling.md`
- Figures & diagnostics: `docs/figures.md`
- Diagnostics utilities: `docs/diagnostics.md`
- Parity-FV rewrite track: `docs/parity_fv.md`
