# Stellarator Turbulence: Closed and Open Field Lines

The four-field interchange model (density, vorticity, ion/electron parallel
velocity) runs turbulence-type dynamics on the genuinely non-axisymmetric
rotating-ellipse stellarator, on **closed** field lines and — with a toroidal
limiter — on **open** ones. A random-phase multi-mode density seed (no initial
vorticity or flow) is amplified by the curvature drive into interchange
dynamics that differ plane-by-plane because the flux surfaces rotate with the
toroidal angle.

In three dimensions — the cutaway torus colored by the evolving density
fluctuation, and the field-line topology (closed core in blue, open
scrape-off-layer lines ending on the limiter in red):

![Stellarator turbulence in 3D](media/stellarator_3d_turbulence.gif)

![Closed and open field lines in 3D](media/stellarator_3d_field_lines.png)

| Closed field lines | Open field lines (limiter SOL) |
|---|---|
| ![Stellarator turbulence, closed](media/stellarator_turbulence_closed.gif) | ![Stellarator SOL turbulence, open](media/stellarator_turbulence_open.gif) |

## Open field lines: the limiter

`build_rotating_ellipse_geometry(..., limiter_radius=r)` opens every field line
with `x > r` on a toroidal limiter at the `zeta = 0` plane — their forward
field-line maps exit on the last toroidal plane and their backward maps on the
first — while `x <= r` remains a closed core. The FCI endpoint masks then mark
exactly the scrape-off-layer surfaces, and the Bohm sheath closure
(`compute_fci_sheath_recycling`) drains them at `n c_s` each step. The summary
figure of the demo shows the consequence: with the same seed, the open run
loses particle content through the limiter while the closed run conserves it.

![Stellarator turbulence summary](media/stellarator_turbulence_summary.png)

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
[`src/jax_drb/native/stellarator_turbulence.py`](../src/jax_drb/native/stellarator_turbulence.py)
(the whole-step jitted `run_stellarator_turbulence`, the phi-solver builder,
and the free-decay boundary conditions); the physics is entirely the validated
four-field stack in `jax_drb.native`. The movie scripts cache completed runs
in `closed_frames.npz` / `open_frames.npz` and reuse them when present, so
re-rendering the movies does not re-run the physics.

## Island divertor (B8)

Beyond the hand-placed limiter, the analytic **island-divertor** field
([`jax_drb.geometry.island_divertor`](../src/jax_drb/geometry/island_divertor.py))
carries a sheared rotational transform crossing the 2/3, 3/4, and 4/5 rational
surfaces, each opened by a resonant radial perturbation. The overlapping chains
make the edge stochastic, and the open scrape-off layer **emerges from the
field itself**: multi-transit field-line tracing
(`island_divertor_connection_length`, pure JAX) marks a cell open when its line
reaches the wall within a finite connection length — closed core, island
chains, stochastic SOL.

![Island divertor](media/island_divertor.png)

The gate [`tests/test_island_divertor.py`](../tests/test_island_divertor.py)
pins the topology (core surfaces closed over 40 transits, stochastic-edge lines
open with finite, ordered connection lengths), the emergent masks (closed core,
>90% open at the wall), and that the four-field turbulence drains through the
emergent divertor faster than the closed reference with the same seed.

```bash
PYTHONPATH=src python examples/stellarator/island_divertor.py
pytest -q tests/test_island_divertor.py
```

## Honest scope

These are short, bounded, seeded runs showing turbulence-type interchange
dynamics on non-axisymmetric closed and open field lines — not statistically
saturated stellarator turbulence, which needs longer implicit-stepped runs.

## Reproduce

```bash
PYTHONPATH=src python examples/stellarator/stellarator_turbulence.py
PYTHONPATH=src python examples/stellarator/stellarator_3d_render.py   # 3D movie + field lines
pytest -q tests/test_stellarator_turbulence.py
```
