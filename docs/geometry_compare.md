# Geometry Comparison Tools

`jax_drb` ships lightweight scripts to compare analytic geometry coefficients against
external grids and equilibrium files.

## Metric grid comparisons

```
python /Users/rogerio/local/jax_drb/tools/compare_geometry_metrics.py \
  --config /path/to/analytic_config.toml \
  --bout-grid /path/to/grid.nc \
  --mapping canonical \
  --x-index 0 \
  --radial-coordinate physical \
  --radial-from dr
```

Key options:
- `--mapping canonical`: canonical s-alpha log-B mapping (x=radial, z=field-aligned, metric scaling on)
- `--radial-coordinate {physical,flux}`: whether to use physical minor radius or flux-like `dx`
- `--radial-from {auto,dr,dx_btor,dx}`: how to build physical radius when using `physical`
- `--curv-x-axis {x,y,z}` / `--curv-y-axis {x,y,z}`: map curvature to BOUT coordinate conventions
- `--curv-sign-x`, `--curv-sign-y`: sign flips for alternative conventions
- `--use-metric`: include `gxx/gxy/gyy` metric scaling

Expected axis conventions:
- `x`: minor radial coordinate
- `y`: binormal (poloidal) direction
- `z`: field-aligned (ballooning) coordinate

## External equilibrium files

```
python /Users/rogerio/local/jax_drb/tools/compare_geometry_gbs.py \
  --config /path/to/analytic_config.toml \
  --gbs-file /path/to/results_*.h5 \
  --mapping canonical \
  --normalize
```

Key options:
- `--mapping canonical`: canonical s-alpha log-B mapping (no swap, positive signs)
- `--theta-range "min,max"`: theta range for the curvature arrays
- `--normalize`: compare shape (RMS-normalized) rather than absolute magnitude
- `--swap-xy`, `--curv-sign-x`, `--curv-sign-y`: axis/sign convention adjustments
