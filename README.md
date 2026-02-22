# jax_drb (Unified DRB)

Research‑grade **drift‑reduced Braginskii** solver centered on a **single unified system**.
All variants (ES/EM, hot/cold ions, sheath/no‑sheath, Boussinesq/non‑Boussinesq,
1D/2D/3D, linear/nonlinear) are **toggles** on the same core RHS.

![Nonlinear DRB panel](docs/figures/nonlinear_panel.png)

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
