# TCV-X21 Reduced Selected-Field Parity Demo

This page documents the next explicit 3D promotion gate after the TCV-X21
scaffold bundle: a compact selected-field parity package on `Ne`, `Pe`, and
`phi`.

It does not claim a native 3D solver yet. It builds the compare/report surface
that the native 3D lane will need before broader benchmark claims:

- reduced selected-field error metrics over time
- saved JSON and NPZ parity artifacts
- a publication-style parity summary plot

## Run It

Preferred public benchmark-data demo:

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/tcv-x21/selected_field_parity_demo.py \
  --benchmark-data-root /tmp/tcv_x21_public_benchmark \
  --output-root docs/data/tokamak_tcv_x21_selected_field_artifacts
```

If you do not have the public benchmark files locally yet:

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/tcv-x21/selected_field_parity_demo.py \
  --download-public-benchmark-data \
  --benchmark-data-root /tmp/tcv_x21_public_benchmark \
  --output-root docs/data/tokamak_tcv_x21_selected_field_artifacts
```

Synthetic scaffold-to-scaffold fallback:

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/tcv-x21/selected_field_parity_demo.py \
  --output-root docs/data/tokamak_tcv_x21_selected_field_artifacts
```

If you have two real 3D workdirs to compare:

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/tcv-x21/selected_field_parity_demo.py \
  --reference-workdir /path/to/reference-workdir \
  --candidate-workdir /path/to/candidate-workdir \
  --output-root docs/data/tokamak_tcv_x21_selected_field_artifacts
```

## Intended Use

This package is the reduced 3D gate the plan calls for:

1. compare a compact field surface before widening claims;
2. save the result in machine-readable form;
3. make the benchmark-facing figure reproducible;
4. reuse the same field list when the first selected native 3D rung lands.

## Output Files

- parity JSON: `data/tokamak_tcv_x21_selected_field_parity.json`
- parity arrays: `data/tokamak_tcv_x21_selected_field_parity.npz`
- benchmark-data report: `data/tokamak_tcv_x21_selected_field_parity_benchmark_data_report.json`
- observable report: `data/tokamak_tcv_x21_selected_field_parity_observable_report.json`
- parity plot: `images/tokamak_tcv_x21_selected_field_parity.png`

The committed public bundle is now generated from the public TCV-X21 sample
files, using the benchmark data root as the reference side and a deterministic
derived candidate as the reproducible compare target.

## What It Does Not Do Yet

- it does not prove a native 3D solver path;
- it does not replace the full TCV-X21 observable package;
- it does not by itself count as benchmark validation.
