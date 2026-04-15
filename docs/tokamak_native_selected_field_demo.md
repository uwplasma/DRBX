# Native Tokamak Selected-Field Demo

This page documents the first reduced native 3D selected-field parity rung in
the repository. It runs a promoted native tokamak one-step case and
compares the compact `Ne`/`Pe`/`phi` history surface against the committed
reference arrays on the same time grid.

It is intentionally narrow:

- it does not claim full native 3D benchmark closure;
- it does provide an honest native execution artifact bundle;
- it adds runtime and provenance reporting on top of the compact parity surface.

## Run It

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/tokamak-native/selected_field_demo.py \
  --reference-root /path/to/reference-root \
  --output-root docs/data/tokamak_native_selected_field_artifacts
```

## Output Files

- parity JSON: `data/tokamak_native_selected_field.json`
- parity arrays: `data/tokamak_native_selected_field.npz`
- observable report: `data/tokamak_native_selected_field_observable_report.json`
- runtime report: `data/tokamak_native_selected_field_runtime_report.json`
- parity plot: `images/tokamak_native_selected_field.png`

## Intended Use

This package is the next step after the benchmark-backed compact compare gates:

1. run a real native tokamak case on the promoted compact surface;
2. compare that native history against the committed reference arrays;
3. publish a compact machine-readable parity bundle;
4. record runtime/provenance metadata before widening the 3D claim.
