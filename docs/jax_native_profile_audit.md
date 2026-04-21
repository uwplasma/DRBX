# JAX Native Profile Audit

This package is the first explicit JAX profiling audit for the reduced native 3D kernels that are already part of the public `jax_drb` validation surface.

It does three concrete things:

- measures compile time separately from first and warm execution on the promoted traced-field-line and VMEC reduced native kernels;
- emits Perfetto-compatible profiler traces for both reduced native lanes;
- records the practical performance guidance that follows from those measurements.

After the latest reduced-kernel pass, the native selected-field comparisons now
batch the reference/candidate pair through one compiled reduction call instead
of dispatching the same reduced kernel twice. On the committed CPU audit that
cuts the measured profile surface materially:

- traced-field-line reduced kernel:
  - compile time dropped from about `8.77e-3 s` to about `7.84e-4 s`
  - first execution dropped from about `3.35e-4 s` to about `1.07e-4 s`
  - warm execution dropped from about `3.46e-5 s` to about `1.02e-5 s`
- stellarator VMEC reduced kernel:
  - compile time dropped from about `1.05e-3 s` to about `5.41e-4 s`
  - first execution dropped from about `5.68e-4 s` to about `1.04e-4 s`
  - warm execution dropped from about `2.65e-5 s` to about `1.07e-5 s`

The current artifact bundle is written to `docs/data/jax_native_profile_audit_artifacts/` and contains:

- `data/jax_native_profile_audit.json`
- `images/jax_native_profile_audit.png`
- `traces/traced_field_line/normalized/perfetto_trace.json.gz`
- `traces/traced_field_line/normalized/runtime.trace.json.gz`
- `traces/traced_field_line/normalized/runtime.xplane.pb`
- `traces/stellarator_vmec/normalized/perfetto_trace.json.gz`
- `traces/stellarator_vmec/normalized/runtime.trace.json.gz`
- `traces/stellarator_vmec/normalized/runtime.xplane.pb`

Run it with:

```bash
PYTHONPATH=src .venv/bin/python examples/engineering/jax_native_profile_audit_demo.py
```

Interpretation:

- the traced-field-line and VMEC reduced kernels now batch same-shape fields before entering the jitted reduction, so the compile surface is one batched kernel per geometry family instead of one tiny dispatch per field;
- the current implementation also batches the reference/candidate pair through the same reduced kernel, so parity/comparison work does not pay a second dispatch for the same shape;
- the warm execution timings are the numbers that should inform summary reduced-kernel runtime summaries;
- the compile timings and Perfetto traces are the audit evidence for where JIT overhead still exists and where it does not.
