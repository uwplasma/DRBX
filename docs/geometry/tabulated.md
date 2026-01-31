# Tabulated geometry (`.npz`)

`TabulatedGeometry` (`src/jaxdrb/geometry/tabulated.py`) loads a precomputed field-line geometry
from a `.npz` file.

This is the primary path for plugging in geometry from external equilibrium tools.

## Required arrays

The `.npz` file must contain 1D arrays over the same `l` grid:

- `l`: parallel coordinate array, shape `(nl,)`
- `gxx`, `gxy`, `gyy`: metric-like coefficients, each shape `(nl,)`

The file currently requires a **uniform** `l` grid.

## Optional arrays

- `curv_x`, `curv_y`: curvature coefficients, shape `(nl,)`
- `dpar_factor`: multiplicative factor for the finite-difference `d/dl`

If omitted, curvature defaults to zero and `dpar_factor` defaults to 1.

## How k_perp^2 is computed

Given `(gxx, gxy, gyy)`, we compute:

$$
k_\perp^2(l) = k_x^2 g^{xx}(l) + 2 k_x k_y g^{xy}(l) + k_y^2 g^{yy}(l).
$$

## How curvature is computed

The v1 tabulated curvature operator uses:

$$
C(f) = i\,(k_x\,\texttt{curv\_x}(l) + k_y\,\texttt{curv\_y}(l))\,f.
$$

## Example

See `examples/run_tabulated_geom.py` for a full working example that:

1. generates a `.npz` file from an analytic geometry,
2. runs a `ky` scan using `--geom tabulated`.

