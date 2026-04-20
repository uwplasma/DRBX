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

Or point it at an explicit VMEC pair:

```bash
PYTHONPATH=src .venv/bin/python examples/geometry-3D/stellarator-vmec/selected_field_parity_demo.py \
  --reference-equilibrium-path /path/to/reference_wout.nc \
  --candidate-equilibrium-path /path/to/candidate_wout.nc
```

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
- `docs/data/stellarator_vmec_selected_field_artifacts/images/stellarator_vmec_selected_field_parity.png`
