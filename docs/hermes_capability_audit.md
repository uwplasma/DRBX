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

Two of the previously broad open families now have stronger concrete evidence:

- `reactions_collisions_and_atomic_data` is no longer only covered indirectly by selected recycling lanes; the dedicated `reactions_collisions_campaign` package now writes an explicit JSON/NPZ/plot gate for charge exchange, isotope coupling, collisionality closure, and OpenADAS loading.
- `non_tokamak_3d_geometry_adapters` is no longer external-pair-only; native reduced traced-field-line and stellarator VMEC rungs now both exist on the shared 3D artifact model.
- `impurity_radiation_and_detachment_control` is no longer only a missing family label; the dedicated `impurity_radiation_campaign` package now covers neon OpenADAS loading, finite radiation-loss evaluation, and exact `D/T/He/Ne` RHS closure, while `controller_feedback_campaign` adds the first dense-history reference-backed controller gate on the native upstream-density feedback path. Temperature-feedback and detachment-controller workflows still remain open beyond that.
