# Traced-Field-Line Native Reduced Selected-Field Demo

This demo is the first honest native reduced rung on the non-tokamak 3D stack.
It does not claim a full plasma solve. It executes a JAX-native radial-profile
reduction on traced-field-line metric fields and compares the reduced profiles on
an explicit reference/candidate pair.

The generated package contains:

- parity JSON and compressed arrays;
- a comparison JSON with native versus reference radial profiles;
- a parity plot and a comparison plot;
- an observable report and a runtime report.

Run the demo with:

```bash
python examples/geometry-3D/traced-field-line/native_selected_field_demo.py
```

When the external FCI pair is available locally, the demo uses it directly and
marks the runtime report as an explicit pair. Otherwise it falls back to a tiny
synthetic pair so the workflow remains runnable, and the runtime report is
marked as `source_mode=synthetic_preview` with
`candidate_origin=synthetic_preview_pair`. The clean-clone committed artifact
therefore validates the adapter, plot, and report plumbing but is filtered out
of the actionable reference-parity queue until a real FCI reference/candidate
pair is supplied.

Committed artifacts:

- `docs/data/traced_field_line_native_selected_field_artifacts/data/traced_field_line_native_selected_field.json`
- `docs/data/traced_field_line_native_selected_field_artifacts/data/traced_field_line_native_selected_field_comparison.json`
- `docs/data/traced_field_line_native_selected_field_artifacts/data/traced_field_line_native_selected_field_observable_report.json`
- `docs/data/traced_field_line_native_selected_field_artifacts/data/traced_field_line_native_selected_field_runtime_report.json`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__traced_field_line_native_selected_field_artifacts__images__traced_field_line_native_selected_field.png`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__traced_field_line_native_selected_field_artifacts__images__traced_field_line_native_selected_field_comparison.png`
