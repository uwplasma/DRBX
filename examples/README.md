# Examples

Runnable example scripts live in:

`examples/scripts/`

Each subfolder focuses on a specific module/topic (geometry, sheath closures, literature workflows,
nonlinear HW2D, etc.). The structure is meant to stay stable as new physics and algorithms are added.

## Running

From the repository root:

```bash
python examples/scripts/01_linear_basics/slab_ky_scan.py
```

Most scripts write a small results folder (typically under `out/` or a user-provided `--out` path)
containing:

- `.npz` outputs with the raw scan data,
- publication-friendly `.png` plots.

## Index

See `examples/scripts/README.md` for a topic-by-topic index.

