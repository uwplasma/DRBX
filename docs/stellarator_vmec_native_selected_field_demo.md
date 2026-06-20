# Stellarator VMEC Native Reduced Selected-Field Demo

This demo builds the first native reduced stellarator VMEC selected-field bundle on the shared 3D artifact path. It uses JAX-native profile handling on `iota`, `pressure`, and `toroidal_flux`, then writes parity, comparison, observable, and runtime reports plus summary figures.

Run:

```bash
PYTHONPATH=src .venv/bin/python examples/geometry-3D/stellarator-vmec/native_selected_field_demo.py
```

Optional inputs are configured as constants near the top of
`examples/geometry-3D/stellarator-vmec/native_selected_field_demo.py`: set
`REFERENCE_EQUILIBRIUM_PATH`, `CANDIDATE_EQUILIBRIUM_PATH`, or `OUTPUT_ROOT`
before running the file.

Default behavior:

- use the explicit external pair when `REFERENCE_EQUILIBRIUM_PATH` and
  `CANDIDATE_EQUILIBRIUM_PATH` are set, or when the legacy local files
  `/tmp/jax_drb_wout_reference.nc` and `/tmp/jax_drb_wout_candidate.nc` exist;
- materialize a candidate from the reference when only the reference path is
  provided;
- otherwise generate a synthetic preview pair.

The committed clean-clone artifact is intentionally the synthetic-preview
path. It validates the native selected-field adapter, report schema, plotting
surface, and provenance plumbing without committing VMEC NetCDF equilibria to
the lightweight repository. It is therefore excluded from the actionable
reference-parity queue until a real external VMEC reference/candidate pair is
provided through the constants above or through a release-hosted reference
bundle.

Committed artifacts:

- `docs/data/stellarator_vmec_native_selected_field_artifacts/data/stellarator_vmec_native_selected_field.json`
- `docs/data/stellarator_vmec_native_selected_field_artifacts/data/stellarator_vmec_native_selected_field_comparison.json`
- `docs/data/stellarator_vmec_native_selected_field_artifacts/data/stellarator_vmec_native_selected_field_observable_report.json`
- `docs/data/stellarator_vmec_native_selected_field_artifacts/data/stellarator_vmec_native_selected_field_runtime_report.json`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_vmec_native_selected_field_artifacts__images__stellarator_vmec_native_selected_field.png`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_vmec_native_selected_field_artifacts__images__stellarator_vmec_native_selected_field_comparison.png`
