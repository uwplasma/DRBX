# Hermes Capability Audit

This page records the maintained source-level capability audit against the local
`hermes-3` tree.

It is intentionally grouped by physics and workflow families rather than by
every individual C++ source file. The purpose is to keep the remaining
engineering work honest:

- which families are already closed on promoted native lanes;
- which families are only partially closed on selected benchmark surfaces;
- which families still need dedicated implementation or validation work.

The machine-readable artifact is:

- `docs/data/hermes_capability_audit.json`

Use the demo to regenerate it:

```bash
PYTHONPATH=src .venv/bin/python examples/engineering/hermes_capability_audit_demo.py \
  --output docs/data/hermes_capability_audit.json
```

Current highest-priority open families in that audit are:

1. `neutral_mixed`
2. `open_field_recycling`
3. `direct_tokamak_recycling`
4. `non_tokamak_3d_geometry_adapters`
5. `reactions_collisions_and_atomic_data`
6. `impurity_radiation_and_detachment_control`

The audit is not a publication figure. It is an engineering control surface for
closing the plan against real source families and real integrated-test
workflows.
