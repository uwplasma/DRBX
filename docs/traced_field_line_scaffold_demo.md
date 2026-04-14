# Traced-Field-Line Scaffold Demo

This demo creates the second 3D geometry adapter scaffold in the current roadmap.
It is not a native stellarator solver claim. It is a geometry and metric
artifact bundle that exercises the general 3D infrastructure on a non-diverted,
traced-field-line family.

## Run It

Synthetic preview:

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/traced-field-line/scaffold_demo.py \
  --output-root docs/data/traced_field_line_scaffold_artifacts
```

With an explicit JSON mesh/metric specification:

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/traced-field-line/scaffold_demo.py \
  --mesh-spec /path/to/mesh_spec.json \
  --output-root docs/data/traced_field_line_scaffold_artifacts
```

With an external NetCDF FCI grid, including Zoidberg-style metric outputs:

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/traced-field-line/scaffold_demo.py \
  --mesh-spec /path/to/grid.fci.nc \
  --output-root docs/data/traced_field_line_scaffold_artifacts
```

## Artifacts

The scaffold writes:

- a manifest
- an input/mesh report
- a validation contract
- a metric summary JSON
- a compact metric NPZ bundle
- a publication-style metric summary figure
- a line-diagnostic JSON/NPZ bundle for radial, toroidal, and poloidal cuts
- a publication-style lineout summary figure

The input report records whether the source came from a JSON mesh spec or a
NetCDF FCI grid.

## Why It Exists

The current 3D program should not be defined by a single diverted tokamak
benchmark. This scaffold is the first explicit second adapter family, intended
to pressure-test the general mesh/metric and diagnostics interfaces against a
traced-field-line geometry family before broader 3D claims are made.

That includes real external metric grids from traced-field-line workflows, not
just synthetic preview metadata.
