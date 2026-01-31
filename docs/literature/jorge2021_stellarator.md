# Jorge & Landreman (2021): near-axis stellarator geometry (pyQSC)

Jorge & Landreman (Plasma Phys. Control. Fusion 63, 014001 (2021)) discuss how **near-axis**
expansions can provide geometric coefficients needed for turbulence simulations, and compare fitted
near-axis models to full equilibria.

`jaxdrb` is designed so that magnetic geometry is pluggable. This makes it straightforward to:

1. generate a **tabulated field-line geometry** from a near-axis solution (pyQSC),
2. run the same linear stability workflows (ky scans, branch toggles, $(\gamma/k_y)_{\max}$, and $L_p$ estimation).

## What the example does

Run:

```bash
make examples-stellarator
```

or directly:

```bash
PYTHONPATH=../pyQSC-main python examples/3_advanced/04_stellarator_nearaxis_pyqsc.py
```

This script:

- loads a near-axis configuration from pyQSC (`Qsc.from_paper("r1 section 5.1")`),
- chooses a radius `r=0.1` and a field-line label `alpha=0`,
- constructs a `.npz` geometry file containing:
  - perpendicular metric coefficients (`gxx`, `gxy`, `gyy`) used for $k_\perp^2(l)$,
  - an approximate curvature-drift-like coefficient (`curv_x`, `curv_y`),
  - a parallel-derivative factor `dpar_factor(l)` based on local arc length,
- runs two linear scans that illustrate resistive-like vs inertial-like branches,
- computes a fixed-point $L_p$ estimate using `jaxdrb.analysis.lp.solve_lp_fixed_point`.

Outputs are written to `out/3_advanced/stellarator_nearaxis_pyqsc/`.

## How the geometry is constructed (implementation sketch)

The example uses a finite-difference approach:

1. build flux-surface coordinates $x(r,\theta,\phi)$ from pyQSC for `r`, `r±dr`,
2. evaluate the chosen field line via $\theta(\phi) = \alpha + \iota\,\varphi(\phi)$,
3. compute covariant basis vectors $\partial x/\partial r$, $\partial x/\partial \alpha$,
   and $\partial x/\partial l$ (here $l$ is a toroidal-angle-like coordinate),
4. compute the perpendicular contravariant metric coefficients:
   $$g^{rr}=|\nabla r|^2,\quad g^{r\alpha}=\nabla r\cdot\nabla\alpha,\quad g^{\alpha\alpha}=|\nabla\alpha|^2,$$
5. store these arrays in a `.npz` file and run `jaxdrb` through `TabulatedGeometry`.

This procedure is *not* a substitute for a full GS2/GENE-compatible geometry module, but it is a
useful bridge between near-axis expansions and local linear stability studies.

## Practical notes

- This example requires pyQSC. If you do not have it installed, use the local checkout:
  `PYTHONPATH=../pyQSC-main`.
- The field-line mapping in a stellarator is not exactly periodic; the example selects a toroidal
  domain length that makes the mapping **approximately periodic**, compatible with the v1 periodic
  parallel derivative.
