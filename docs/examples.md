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

## Literature workflows

The `examples/literature/` folder contains scripts that mirror common SOL/edge analysis steps used in
the literature:

```bash
python examples/literature/mosetto2012_driftwaves.py
python examples/literature/mosetto2012_ballooning.py
python examples/literature/halpern2013_gradient_removal.py
python examples/literature/cyclone_kxky_scan.py
```

These scripts write `out_*` folders with plots and `.npz` outputs. See the “Literature reproduction”
docs section for interpretation.

## Stellarator (pyQSC near-axis)

```bash
make examples-stellarator
```

This runs `examples/run_pyqsc_stellarator.py`, which:

- generates a tabulated geometry file from a near-axis pyQSC configuration,
- runs branch scans and a fixed-point $L_p$ estimate.

## Tips

- Start with small `nl` (e.g. 32) and fewer `ky` points when iterating quickly.
- For small `ky`, Arnoldi may need a larger Krylov dimension to converge; the CLI adapts `m` up to
  `5*nl`.
