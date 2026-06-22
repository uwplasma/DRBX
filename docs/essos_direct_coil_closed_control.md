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
- `artifacts/essos_direct_coil_closed_control/refinement/data/essos_direct_coil_closed_control_refinement.json`
- `artifacts/essos_direct_coil_closed_control/refinement/data/essos_direct_coil_closed_control_refinement.npz`
- `artifacts/essos_direct_coil_closed_control/refinement/images/essos_direct_coil_closed_control_refinement.png`
- `artifacts/essos_direct_coil_closed_control/transient/data/essos_direct_coil_closed_control_transient.json`
- `artifacts/essos_direct_coil_closed_control/transient/data/essos_direct_coil_closed_control_transient.npz`
- `artifacts/essos_direct_coil_closed_control/transient/images/essos_direct_coil_closed_control_transient.png`
- `artifacts/essos_direct_coil_closed_control/transient/movies/essos_direct_coil_closed_control_transient.gif`

The JSON report contains the number of seeds, Poincare point count, toroidal
turn statistics, normalized same-section return-distance percentiles, closed
and near-closed fractions, and a `closed_control_passed` flag. The NPZ arrays
include the trajectories, Poincare points, per-line return distances, return
points, and integer line classifications.

The refinement report reruns the same closed-control diagnostic at increasing
seed and trace samples. Its promotion gate checks that:

- every level passes the base closed-control gate;
- target, sheath, recycling, and neutral-source semantics are absent at every
  level;
- the closed-or-near-closed fraction stays above the configured floor;
- the closed-or-near-closed fraction is stable across levels;
- the closed versus near-closed split is not excessively sensitive to sampling;
- the 95th-percentile normalized return distance remains below the
  near-closed tolerance;
- the Poincare sampling density remains above the configured minimum.

This is a closed-field return-map stability gate. It is not a target-to-target
connection-length, sheath, recycling, or neutral-transport validation.

The transient package then runs a compact periodic scalar model along the same
closed or near-closed trace bundle. The model advances a line-following density
fluctuation with periodic advection, field-line diffusion, and a zero-mean
non-axisymmetric drive. Each timestep removes the per-line mean of the right
hand side, so the diagnostic is a closed-field profile/fluctuation control
rather than a source or sink balance. The report records fluctuation RMS, mass
drift, line-mean spread, positive-density checks, fixed-camera/fixed-color
movie flags, and a `closed_control_media_ready` flag. It also records
`open_sol_publication_ready = false`, because this trace bundle has no endpoint
mask, target sheath, recycling source, or neutral-loss channel.

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
return-map classifier and optional refinement gate. If the closed or
near-closed fraction is too small, the report remains useful diagnostic
evidence but is not promotion-ready.

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

## Current Live Landreman-Paul QA Result

The first live direct-coil run used 16 seed lines on a VMEC-shaped shell,
`maxtime = 240`, and 1200 trace samples. The control passed the current
return-map gate:

- Poincare points: `639`
- Mean toroidal turns: `10.11`
- Closed fraction: `0.125`
- Near-closed fraction: `0.875`
- Open-like fraction: `0.0`
- No-return fraction: `0.0`
- Median normalized return distance: `6.68e-2`
- 95th-percentile normalized return distance: `8.45e-2`

This is useful closed/near-closed direct-coil evidence, but it does not change
the open-SOL claim boundary. Open-field direct-coil media still requires
endpoint-label and adjacent-step refinement to pass. Closed-field physics
examples should use either this return-map control or the smoother VMEC map
lane, and they should continue to omit target, sheath, recycling, and neutral
losses unless an explicit endpoint mask is introduced.

The first compact live same-source refinement run used three levels,
`(3, 3, 256)`, `(4, 4, 384)`, and `(5, 4, 512)`, with `maxtime = 240`.
It also passed the current refinement gate:

- Levels: `3`
- Minimum closed-or-near-closed fraction: `1.0`
- Closed-or-near-closed fraction spread: `0.0`
- Maximum 95th-percentile normalized return distance: `8.62e-2`
- Minimum Poincare points per seed line: `39.9`
- Target semantics applied: `False`
- Sheath/recycling semantics applied: `False`
- Promotion rejection reasons: none

This closes the compact live return-map stability gate for the direct-coil
closed-control lane. The self-contained example now also writes a reduced
closed-trace transient and fixed-camera GIF. That media closes the local
closed-control tutorial gap, but it remains a closed-field diagnostic unless a
future live direct-coil run and frame-by-frame visual QA are promoted for the
README.

## Why This Gate Exists

Open-field stellarator SOL examples need endpoint masks, one-sided connection
lengths, target labels, sheath/recycling source accounting, and neutral
coupling. Closed-field examples instead need periodic field-line-map quality,
Poincare or return-map diagnostics, operator conservation, and profile or
spectrum diagnostics. This page exists so the two regimes are not mixed: a
closed or near-closed direct-coil control can be used to compare geometry and
field-line behavior, while target/sheath/recycling physics remains confined to
validated open-field or hybrid endpoint maps.
