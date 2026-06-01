# Stellarator VMEC Selected-Field Parity Demo

This demo adds the first reduced selected-field parity gate on the VMEC-style stellarator adapter.

The committed public bundle can now be regenerated from a real explicit external
VMEC pair, not only from the synthetic preview path. When the `/tmp`
`jax_drb_wout_reference.nc` and `jax_drb_wout_candidate.nc` fixtures exist, the
example prefers that pair automatically.

Run it with:

```bash
PYTHONPATH=src .venv/bin/python examples/geometry-3D/stellarator-vmec/selected_field_parity_demo.py
```

The script is configured by constants near the top of
`examples/geometry-3D/stellarator-vmec/selected_field_parity_demo.py`. Set
`REFERENCE_EQUILIBRIUM_PATH`, `CANDIDATE_EQUILIBRIUM_PATH`, or `OUTPUT_ROOT`
there when using an explicit VMEC pair or a non-default artifact directory.

If the local `/tmp/jax_drb_wout_reference.nc` and `/tmp/jax_drb_wout_candidate.nc`
fixtures exist, the example will use that explicit pair automatically instead of
falling back to the synthetic preview bundle.

The package writes:

- parity JSON and NPZ on `iota`, `pressure`, and `toroidal_flux`;
- a summary parity plot;
- a shared observable report on the generic 3D schema;
- a source report recording whether the run used a synthetic preview, a materialized external explicit pair, or a fully provided explicit pair.

Committed preview bundle:

- `docs/data/stellarator_vmec_selected_field_artifacts/data/stellarator_vmec_selected_field_parity.json`
- `docs/data/stellarator_vmec_selected_field_artifacts/data/stellarator_vmec_selected_field_parity_observable_report.json`
- `docs/data/stellarator_vmec_selected_field_artifacts/data/stellarator_vmec_selected_field_parity_source_report.json`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_vmec_selected_field_artifacts__images__stellarator_vmec_selected_field_parity.png`
