# Traced-Field-Line Scaffold Demo

!!! note "Plan authority"
    This page documents a geometry scaffold artifact. The active execution plan
    is [Research-Grade Execution Plan](research_grade_execution_plan.md). If
    this page conflicts with that plan, follow the execution plan and update
    this page afterward.

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
- an observable report on the shared 3D geometry-adapter schema
- a metric summary JSON
- a compact metric NPZ bundle
- a summary metric summary figure
- a line-diagnostic JSON/NPZ bundle for radial, toroidal, and poloidal cuts
- a summary lineout summary figure
- a selected-plane summary JSON/NPZ bundle
- a summary selected-plane summary figure
- an animated GIF across the most informative radial, toroidal, or poloidal plane family for one selected metric field

Committed public JSON entry points:

- [traced_field_line_scaffold_manifest.json](data/traced_field_line_scaffold_artifacts/data/traced_field_line_scaffold_manifest.json)
- [traced_field_line_scaffold_input_report.json](data/traced_field_line_scaffold_artifacts/data/traced_field_line_scaffold_input_report.json)
- [traced_field_line_scaffold_validation_contract.json](data/traced_field_line_scaffold_artifacts/data/traced_field_line_scaffold_validation_contract.json)
- [traced_field_line_scaffold_observable_report.json](data/traced_field_line_scaffold_artifacts/data/traced_field_line_scaffold_observable_report.json)
- [traced_field_line_scaffold_metric_report.json](data/traced_field_line_scaffold_artifacts/data/traced_field_line_scaffold_metric_report.json)
- [traced_field_line_scaffold_line_report.json](data/traced_field_line_scaffold_artifacts/data/traced_field_line_scaffold_line_report.json)
- [traced_field_line_scaffold_slice_report.json](data/traced_field_line_scaffold_artifacts/data/traced_field_line_scaffold_slice_report.json)

The input report records whether the source came from a JSON mesh spec or a
NetCDF FCI grid. The slice report records which plane family was selected for
the summary and GIF, so the artifact stays useful even when a given geometry
has little or no toroidal variation.

## Why It Exists

The current 3D program should not be defined by a single diverted tokamak
benchmark. This scaffold is the first explicit second adapter family, intended
to pressure-test the general mesh/metric and diagnostics interfaces against a
traced-field-line geometry family before broader 3D claims are made.

That includes real external metric grids from traced-field-line workflows, not
just synthetic preview metadata.
