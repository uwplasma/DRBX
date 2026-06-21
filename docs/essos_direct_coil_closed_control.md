# ESSOS Direct-Coil Closed And Near-Closed Control

This diagnostic separates closed or near-closed direct-coil field-line behavior
from open scrape-off-layer endpoint physics. It traces or imports field lines,
builds Poincare sections, computes same-section return distances, classifies
each seed as closed, near-closed, open-like, or no-return, and writes a report,
arrays, and a publication-style QA figure.

The control deliberately does not apply target, sheath, recycling, or neutral
semantics. Those closures belong to the open-field or hybrid SOL campaigns
after endpoint masks and connection-length diagnostics are validated.

## Run The Self-Contained Control

The default example uses manufactured non-axisymmetric traces so it can run
from a clean clone without an ESSOS checkout:

```bash
PYTHONPATH=src MPLBACKEND=Agg python \
  examples/geometry-3D/essos-field-lines/direct_coil_closed_field_demo.py
```

It writes:

- `artifacts/essos_direct_coil_closed_control/data/essos_direct_coil_closed_control.json`
- `artifacts/essos_direct_coil_closed_control/data/essos_direct_coil_closed_control.npz`
- `artifacts/essos_direct_coil_closed_control/images/essos_direct_coil_closed_control.png`

The JSON report contains the number of seeds, Poincare point count, toroidal
turn statistics, normalized same-section return-distance percentiles, closed
and near-closed fractions, and a `closed_control_passed` flag. The NPZ arrays
include the trajectories, Poincare points, per-line return distances, return
points, and integer line classifications.

## Run The Live Direct-Coil Control

To use the Landreman-Paul QA coil JSON from an ESSOS checkout, edit the top of
`examples/geometry-3D/essos-field-lines/direct_coil_closed_field_demo.py` and
set:

```python
RUN_LIVE_ESSOS = True
ESSOS_ROOT = Path("/path/to/ESSOS")
```

or pass the same information through `JAX_DRB_ESSOS_ROOT`:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src MPLBACKEND=Agg python \
  examples/geometry-3D/essos-field-lines/direct_coil_closed_field_demo.py
```

The live mode seeds a VMEC-shaped shell, asks ESSOS to trace the direct
Biot-Savart coil field, imports only arrays into JAXDRB, and runs the same
return-map classifier. If the closed or near-closed fraction is too small, the
report remains useful diagnostic evidence but is not promotion-ready.

## Interpretation

For a closed-field control, the relevant quantity is not target-to-target
connection length. The diagnostic instead evaluates whether a field line
returns close to its initial point on the same toroidal section after one or
more toroidal turns. The return distance is normalized by the inferred minor
extent of the seed shell and traced radial/vertical spans.

The default thresholds are:

- `closed_return_tolerance = 3e-2`
- `near_closed_return_tolerance = 1.5e-1`
- `minimum_closed_or_near_fraction = 0.20`

These values are diagnostic thresholds, not universal physics constants. They
are intended to separate a useful closed/near-closed control from a clearly
open or no-return trace bundle before any closed-field turbulence example is
promoted.

## Why This Gate Exists

Open-field stellarator SOL examples need endpoint masks, one-sided connection
lengths, target labels, sheath/recycling source accounting, and neutral
coupling. Closed-field examples instead need periodic field-line-map quality,
Poincare or return-map diagnostics, operator conservation, and profile or
spectrum diagnostics. This page exists so the two regimes are not mixed: a
closed or near-closed direct-coil control can be used to compare geometry and
field-line behavior, while target/sheath/recycling physics remains confined to
validated open-field or hybrid endpoint maps.
