# Native Tokamak Selected-Field Demo

This page documents the reduced native tokamak selected-field parity rung in
the repository. It now covers both the first one-step execution surface and a
wider short-window history surface.

It is intentionally narrow:

- it does not claim full native 3D benchmark closure;
- it does provide an honest native execution artifact bundle;
- it adds runtime and provenance reporting on top of the compact parity surface.

## Run It

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/tokamak-native/selected_field_demo.py \
  --reference-root /path/to/reference-root \
  --case-label tokamak_native_selected_field \
  --output-root docs/data/tokamak_native_selected_field_artifacts
```

Short-window extension:

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/tokamak-native/selected_field_demo.py \
  --reference-root /path/to/reference-root \
  --case-name tokamak_isothermal_short_window \
  --case-label tokamak_native_selected_field_short_window \
  --field-name Ne \
  --field-name phi \
  --field-name Vort \
  --output-root docs/data/tokamak_native_selected_field_short_window_artifacts
```

## Output Files

- parity JSON: `data/tokamak_native_selected_field.json`
- parity arrays: `data/tokamak_native_selected_field.npz`
- observable report: `data/tokamak_native_selected_field_observable_report.json`
- runtime report: `data/tokamak_native_selected_field_runtime_report.json`
- parity plot: `images/tokamak_native_selected_field.png`

Second committed bundle:

- parity JSON: `docs/data/tokamak_native_selected_field_short_window_artifacts/data/tokamak_native_selected_field_short_window.json`
- observable report: `docs/data/tokamak_native_selected_field_short_window_artifacts/data/tokamak_native_selected_field_short_window_observable_report.json`
- runtime report: `docs/data/tokamak_native_selected_field_short_window_artifacts/data/tokamak_native_selected_field_short_window_runtime_report.json`

## Intended Use

This package is the next step after the benchmark-backed compact compare gates:

1. run a real native tokamak case on the promoted compact surface;
2. compare that native history against the committed reference arrays;
3. publish a compact machine-readable parity bundle;
4. record runtime/provenance metadata before widening the 3D claim.
