# jax_drb (Unified DRB)

Research‑grade **drift‑reduced Braginskii** solver centered on a **single unified system**.
All variants (ES/EM, hot/cold ions, sheath/no‑sheath, Boussinesq/non‑Boussinesq,
1D/2D/3D, linear/nonlinear) are **toggles** on the same core RHS.

![Nonlinear DRB panel](docs/figures/nonlinear_panel.png)
![Blob movie](docs/figures/blob_movie.gif)
![Zonal flow](docs/figures/nonlinear_zonal_flow.png)
![Open field-line poloidal plane](docs/figures/open_field_poloidal_eq.png)
![Field-aligned 3D slices](docs/figures/three_d_slices.png)

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
- Profiling: `docs/profiling.md`
- Figures & diagnostics: `docs/figures.md`
- Diagnostics utilities: `docs/diagnostics.md`
