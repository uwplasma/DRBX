# Examples

The `examples/` folder contains runnable scripts that exercise the CLI on several geometries.

All examples write an output folder with `results.npz` and `gamma_ky.png`.

## Slab scan

```bash
python examples/run_slab_scan.py
```

## Circular tokamak scan

```bash
python examples/run_circular_tokamak.py
```

This uses `--geom tokamak` with a circular tokamak model (large aspect ratio).

## Cyclone (s-alpha) scan

```bash
python examples/run_cyclone_salpha.py
```

This uses `--geom salpha` with Cyclone-like defaults (q≈1.4, shat≈0.796, epsilon≈0.18).

## Tabulated geometry scan

```bash
python examples/run_tabulated_geom.py
```

This example generates a `.npz` file and then runs a scan with `--geom tabulated`.

## Tips

- Start with small `nl` (e.g. 32) and fewer `ky` points when iterating quickly.
- For small `ky`, Arnoldi may need a larger Krylov dimension to converge; the CLI adapts `m` up to
  `5*nl`.

