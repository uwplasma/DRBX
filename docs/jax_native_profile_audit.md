# JAX Native Profile Audit

This package is the first explicit JAX profiling audit for the reduced native 3D kernels that are already part of the public `jax_drb` validation surface.

It does three concrete things:

- measures compile time separately from first and warm execution on the promoted traced-field-line and VMEC reduced native kernels;
- emits Perfetto-compatible profiler traces for both reduced native lanes;
- records the practical performance guidance that follows from those measurements.

The current artifact bundle is written to `docs/data/jax_native_profile_audit_artifacts/` and contains:

- `data/jax_native_profile_audit.json`
- `images/jax_native_profile_audit.png`
- `traces/traced_field_line/.../perfetto_trace.json.gz`
- `traces/stellarator_vmec/.../perfetto_trace.json.gz`

Run it with:

```bash
PYTHONPATH=src .venv/bin/python examples/engineering/jax_native_profile_audit_demo.py
```

Interpretation:

- the traced-field-line and VMEC reduced kernels now batch same-shape fields before entering the jitted reduction, so the compile surface is one batched kernel per geometry family instead of one tiny dispatch per field;
- the warm execution timings are the numbers that should inform summary reduced-kernel runtime summaries;
- the compile timings and Perfetto traces are the audit evidence for where JIT overhead still exists and where it does not.
