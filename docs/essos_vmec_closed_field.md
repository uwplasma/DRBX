# ESSOS VMEC Closed-Field Control

This gate validates the smooth VMEC closed-field map used as the control for
non-axisymmetric stellarator simulations. It deliberately does not use open
target, sheath, recycling, or neutral-loss semantics. The purpose is to verify
that closed VMEC maps provide periodic field-line coupling, finite map
coordinates, non-axisymmetric magnetic-field modulation, and constant-state
FCI operator consistency before any closed-field transient or movie is
promoted.

## Run The Dry-Run Contract

The default example is self-contained and writes a live-run contract:

```bash
PYTHONPATH=src MPLBACKEND=Agg python \
  examples/geometry-3D/essos-field-lines/vmec_closed_field_demo.py
```

It writes:

- `artifacts/essos_vmec_closed_field/data/essos_vmec_closed_field_campaign_dry_run_contract.json`
- `artifacts/essos_vmec_closed_field/data/essos_vmec_closed_field_transient_dry_run_contract.json`

## Run The Live VMEC Gate

To generate the live Landreman-Paul QA VMEC closed-field artifact, edit the top
of `examples/geometry-3D/essos-field-lines/vmec_closed_field_demo.py` and set:

```python
RUN_LIVE_VMEC = True
ESSOS_ROOT = Path("/path/to/ESSOS")
```

or run with an environment variable:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src MPLBACKEND=Agg python \
  examples/geometry-3D/essos-field-lines/vmec_closed_field_demo.py
```

The live package writes:

- `artifacts/essos_vmec_closed_field/data/essos_vmec_closed_field_campaign.json`
- `artifacts/essos_vmec_closed_field/data/essos_vmec_closed_field_campaign.npz`
- `artifacts/essos_vmec_closed_field/images/essos_vmec_closed_field_campaign.png`

## Run The Reduced Closed-Field Transient

The same example can also generate a compact reduced transient, profile,
spectrum, and GIF movie on the periodic VMEC map. Set:

```python
RUN_LIVE_VMEC_TRANSIENT = True
ESSOS_ROOT = Path("/path/to/ESSOS")
```

The transient package writes:

- `artifacts/essos_vmec_closed_field/data/essos_vmec_closed_field_transient.json`
- `artifacts/essos_vmec_closed_field/data/essos_vmec_closed_field_transient.npz`
- `artifacts/essos_vmec_closed_field/images/essos_vmec_closed_field_transient.png`
- `artifacts/essos_vmec_closed_field/movies/essos_vmec_closed_field_transient.gif`

The reduced scalar model is

\[
\partial_t n =
-\alpha_E\{\phi,n\}_{x,z}
+ \nabla_\parallel(\chi_\parallel\nabla_\parallel n)
+ \nabla_\perp\cdot(\chi_\perp\nabla_\perp n)
+ S_0 .
\]

The parallel gradient and diffusion use the periodic VMEC FCI maps. The
perpendicular diffusion uses the conservative logical \(x,z\) metric-weighted
operator. The bracket \(\{\phi,n\}_{x,z}\) is the logical perpendicular
\(E\times B\) bracket used as a closed-field nonlinear transport control.
The source \(S_0\) is a zero-volume-mean drive:

\[
\int J S_0\,dV = 0,
\]

and the implementation removes the \(J\)-weighted mean of the full right-hand
side at every explicit substep. This makes the transient a profile/spectrum
and conservation control for closed maps. It is not a target-connected SOL
model: endpoint masks must be zero, target losses are disabled,
sheath/recycling terms are disabled, and neutral-loss terms are disabled.

## Current Live Landreman-Paul QA Result

The current compact live gate used a `(4, 6, 16)` VMEC map on the
Landreman-Paul QA input. It passed:

- Forward boundary fraction: `0.0`
- Backward boundary fraction: `0.0`
- Endpoint fraction: `0.0`
- Finite map-coordinate fraction: `1.0`
- Magnetic-field modulation: `1.247`
- Mean closed-map step length: `1.149`
- `grad_parallel(1)` \(L_\infty\): `0.0`
- `laplace_parallel(1)` \(L_\infty\): `0.0`
- conservative parallel diffusion of a constant field \(L_\infty\): `0.0`

These diagnostics are the closed-field counterpart to the open-field endpoint
checks. They prove that the VMEC map is a periodic closed-field control, but
they do not validate target heat flux, sheath losses, recycling, or neutral
detachment. Those quantities require an open or hybrid endpoint map.

## What The Figure Shows

The generated PNG contains:

- a VMEC poloidal cross-section;
- a magnetic-field section showing non-axisymmetric \(|B|\) modulation;
- a closed-map step-length panel;
- the forward poloidal shift of the VMEC map;
- a zero endpoint-mask panel;
- a text box with the closed-field invariant checks.

The zero endpoint-mask panel is intentional. For a closed VMEC map, target and
sheath semantics are disabled unless a separate artificial loss model is
explicitly selected and labeled.

The transient figure extends this control with a fixed VMEC cross-section,
final density, radial profile evolution, fluctuation and mass-drift traces,
and the final toroidal-poloidal spectrum. The GIF uses a fixed camera and
shows the density fluctuation on the closed section with a colorbar and time
annotation.

A fresh local live transient on the `(5, 8, 20)` Landreman-Paul QA VMEC map
also passed the closed-field control gate:

- Endpoint fraction: `0.0`
- Mass relative drift: `2.1e-15`
- Final fluctuation RMS: `3.38e-2`
- Spectrum finite: `True`
- Target semantics applied: `False`
- Sheath/recycling semantics applied: `False`
- Neutral-loss semantics applied: `False`

The report now records `fixed_camera = true`, `fixed_color_limits = true`,
and `movie_visual_qa_passed = true` when the closed-field transient gate
passes. A manually generated contact sheet showed no frame jitter or color
rescaling. The motion remains intentionally quiet, so this is closed-field
profile/spectrum/control evidence rather than promoted stellarator SOL
turbulence media.
