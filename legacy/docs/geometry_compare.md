# Geometry Consistency Checks

`jax_drb` ships lightweight scripts to compare analytic geometry coefficients against
metric grid files (NetCDF) and ensure internal consistency.

## Metric grid comparisons

```
python tools/compare_geometry_metrics.py \
  --config /path/to/analytic_config.toml \
  --metric-grid /path/to/grid.nc \
  --mapping canonical \
  --x-index 0 \
  --radial-coordinate physical \
  --radial-from dr
```

Key options:
- `--mapping canonical`: canonical s-alpha log-B mapping (x=radial, z=field-aligned, metric scaling on)
- `--radial-coordinate {physical,flux}`: whether to use physical minor radius or flux-like `dx`
- `--radial-from {auto,dr,dx_btor,dx}`: how to build physical radius when using `physical`
- `--curv-x-axis {x,y,z}` / `--curv-y-axis {x,y,z}`: map curvature to grid coordinate conventions
- `--curv-sign-x`, `--curv-sign-y`: sign flips for alternative conventions
- `--use-metric`: include `gxx/gxy/gyy` metric scaling

Expected axis conventions:
- `x`: minor radial coordinate
- `y`: binormal (poloidal) direction
- `z`: field-aligned (ballooning) coordinate

## Notes

The metric comparison tool is designed for axisymmetric grids that provide
`logB`, metric coefficients (`gxx/gxy/gyy`), and the basic spacing metadata (`dx`,
`dy`, `dr`, `r0`). It reports RMS and relative errors for curvature and
`dpar_factor` so you can confirm analytic vs file‑based geometry alignment.
