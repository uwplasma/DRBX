# Jorge & Landreman (2021): near-axis stellarator geometry (ESSOS)

Jorge & Landreman (Plasma Phys. Control. Fusion 63, 014001 (2021), see `docs/references.md`) discuss how **near-axis**
expansions can provide geometric coefficients needed for turbulence simulations, and compare fitted
near-axis models to full equilibria.

`jaxdrb` is designed so that magnetic geometry is pluggable. This makes it straightforward to:

1. generate a **tabulated field-line geometry** from a near-axis solution (ESSOS),
2. run the same linear stability workflows (ky scans, branch toggles, $(\gamma/k_y)_{\max}$, and $L_p$ estimation).

## What the example does

Run:

```bash
make examples-stellarator
```

or directly:

```bash
python examples/scripts/07_essos_geometries/stellarator_nearaxis_essos.py
```

This script:

- builds a near-axis configuration using `essos.fields.near_axis`,
- chooses a radius `r=0.1` and a field-line label `alpha=0`,
- constructs a `.npz` geometry file containing:
  - perpendicular metric coefficients (`gxx`, `gxy`, `gyy`) used for $k_\perp^2(l)$,
  - curvature coefficients (`curv_x`, `curv_y`) from the field-line curvature,
  - a parallel-derivative factor `dpar_factor(l)` (here `l` is arclength, so `dpar_factor=1`),
- runs a ky scan using the same matrix-free eigen-solver used elsewhere in `jaxdrb`.

Outputs are written to `out/stellarator_nearaxis_essos/`.

![Near-axis stellarator ky scan (ESSOS geometry)](../assets/images/essos_nearaxis_scan_panel.png)

## How the geometry is constructed (implementation sketch)

The example currently uses a pragmatic two-step construction:

1. Use ESSOS to evaluate a periodic near-axis surface in cylindrical coordinates and build a
   **field line in xyz** (via interpolation on a precomputed $(\theta,\varphi)$ grid).
2. Define `l` as arclength and compute curvature coefficients from the geometry of the xyz curve.
3. Use a **local orthonormal perpendicular basis** around the field line (Frenet-like), so the
   stored metric coefficients are approximately:
   $$g^{xx}\approx 1,\quad g^{xy}\approx 0,\quad g^{yy}\approx 1.$$

This is sufficient for exploratory studies and for demonstrating ESSOS integration. A future
extension can replace the local-orthonormal metric with a Clebsch-like $(r,\alpha)$ metric derived
from the near-axis expansion (as in the NAQS paper).

## Practical notes

- This example requires ESSOS (`pip install essos` or a local editable install).
- The current `TabulatedGeometry` path uses periodic finite differences in `l`. The near-axis
  field line is treated as periodic over a single field period.
