# Examples

The `examples/` tree is organized by complexity:

- `examples/1_simple/`: quick "hello world" runs (fast, minimal parameters).
- `examples/2_intermediate/`: branch comparisons, parameter scans, and JAX-based workflows.
- `examples/3_advanced/`: literature-inspired reproduction scripts and stellarator (ESSOS) geometry.

Each script:

- writes an `out/...` folder with `.npz` data + publication-ready `.png` figures,
- prints progress so you can tell it is still running,
- is intended to be readable (docstrings + explicit variable names).

Most examples are pure-Python and can be run from the repo root, e.g.

```bash
python examples/1_simple/01_slab_ky_scan.py
```

The stellarator near-axis example uses ESSOS. If ESSOS is installed, you can run:

```bash
python examples/3_advanced/04_stellarator_nearaxis_essos.py
```
