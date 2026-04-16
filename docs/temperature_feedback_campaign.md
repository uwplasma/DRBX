# Temperature Feedback Campaign

This package turns the `temperature_feedback` example into a bounded, reviewable controller-law gate. It does not claim that `jax_drb` already has a native `temperature_feedback` component. The scope is narrower and deliberate:

- run a reduced Hermes `1D-recycling-with-Tt-control` example on a bounded local window under the current ten-minute validation budget;
- reconstruct the PI controller from the saved temperature history, gains, and time points;
- compare the reconstructed proportional term, integral term, multiplier, integral state, and electron energy source against the Hermes diagnostics;
- publish JSON, NPZ, and a documentation plot on the public artifact surface.

Run the demo with:

```bash
PYTHONPATH=src python examples/engineering/temperature_feedback_campaign_demo.py
```

The default output root is:

```text
docs/data/temperature_feedback_campaign_artifacts
```

The figure is intended to answer one narrow question cleanly: does the reduced PI controller law match the Hermes implementation on a real reference run? It is an honest bridge between the already committed `controller_feedback_campaign` and any future native `temperature_feedback` or detachment-control implementation. On this machine the bounded local Hermes example still exceeds the current ten-minute policy, so this package remains a scaffolded reduced gate rather than a promoted controller lane.
