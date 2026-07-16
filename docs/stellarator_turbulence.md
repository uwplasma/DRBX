# Stellarator Turbulence: Closed and Open Field Lines

The four-field interchange model (density, vorticity, ion/electron parallel
velocity) runs turbulence-type dynamics on the genuinely non-axisymmetric
rotating-ellipse stellarator, on **closed** field lines and — with a toroidal
limiter — on **open** ones. A random-phase multi-mode density seed (no initial
vorticity or flow) is amplified by the curvature drive into interchange
dynamics that differ plane-by-plane because the flux surfaces rotate with the
toroidal angle.

| Closed field lines | Open field lines (limiter SOL) |
|---|---|
| ![Stellarator turbulence, closed](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/stellarator_turbulence_closed.gif) | ![Stellarator SOL turbulence, open](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/stellarator_turbulence_open.gif) |

## Open field lines: the limiter

`build_rotating_ellipse_geometry(..., limiter_radius=r)` opens every field line
with `x > r` on a toroidal limiter at the `zeta = 0` plane — their forward
field-line maps exit on the last toroidal plane and their backward maps on the
first — while `x <= r` remains a closed core. The FCI endpoint masks then mark
exactly the scrape-off-layer surfaces, and the Bohm sheath closure
(`compute_fci_sheath_recycling`) drains them at `n c_s` each step. The summary
figure of the demo shows the consequence: with the same seed, the open run
loses particle content through the limiter while the closed run conserves it.

![Stellarator turbulence summary](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/stellarator_turbulence_summary.png)

## What is checked

[`tests/test_stellarator_turbulence.py`](../tests/test_stellarator_turbulence.py)
pins, at reduced resolution:

- the limiter opens exactly the scrape-off-layer flux surfaces at the limiter
  planes and nothing else (the core stays closed);
- the closed run stays finite and positive and the curvature drive generates
  interchange vorticity from the pure-density multi-mode seed; and
- the open run has positive Bohm limiter flux and drains particle content
  faster than the closed run with the same seed — the open-field-line channel
  isolated by construction.

The shared driver lives in
[`tests/stellarator_turbulence_case.py`](../tests/stellarator_turbulence_case.py);
the physics is entirely the validated four-field stack in `jax_drb.native`.

## Honest scope

These are short, bounded, seeded runs showing turbulence-type interchange
dynamics on non-axisymmetric closed and open field lines — not statistically
saturated stellarator turbulence, which needs longer implicit-stepped runs.

## Reproduce

```bash
PYTHONPATH=src python examples/stellarator/stellarator_turbulence_demo.py
pytest -q tests/test_stellarator_turbulence.py
```
