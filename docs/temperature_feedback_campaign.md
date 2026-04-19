# Temperature Feedback Campaign

This package turns the `temperature_feedback` example into a bounded, reviewable controller gate. It does not claim that `jax_drb` already has a native `temperature_feedback` component. The scope is narrower and deliberate:

- run a reduced Hermes `1D-recycling-with-Tt-control` example on a bounded local window;
- use a clean auto-patched Hermes reference worktree when the local reference source still carries the known `temperature_feedback.hxx` permission bug, without modifying the user’s dirty reference tree;
- validate the controller algebra that is actually observable on the saved outputs:
  - exact multiplier balance,
  - exact proportional-term balance,
  - exact source-shape/source balance,
  - bounded output-time integral reconstruction,
  - bounded target-temperature error reduction;
- publish JSON, NPZ, and a documentation plot on the public artifact surface.

The summary JSON also carries a `timing_seconds` breakdown for input staging, reference execution, dataset load, and controller reconstruction, plus a `reference_provenance` section recording whether the run used the discovered local Hermes binary or an auto-patched clean reference worktree.

Run the demo with:

```bash
PYTHONPATH=src python examples/engineering/temperature_feedback_campaign_demo.py
```

The default output root is:

```text
docs/data/temperature_feedback_campaign_artifacts
```

The current committed bounded lane uses `ny=16`, `nout=4`, `timestep=100`, and `solver_type=cvode`. The figure is intended to answer one narrow question cleanly: does the reduced controller surface behave consistently on a real Hermes run, with explicit saved-diagnostic balance checks and a visible move toward the target temperature? That makes it a useful bridge between the already committed `controller_feedback_campaign` and any future native `temperature_feedback` or broader detachment-control implementation.
