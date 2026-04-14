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

Synthetic scaffold-to-scaffold demo:

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
- parity plot: `images/tokamak_tcv_x21_selected_field_parity.png`

## What It Does Not Do Yet

- it does not prove a native 3D solver path;
- it does not replace the full TCV-X21 observable package;
- it does not by itself count as benchmark validation.
