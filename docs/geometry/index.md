# Geometry

`jaxdrb` is designed so that geometry lives behind a small interface.

The model core only requires:

- `kperp2(kx, ky) -> k_perp^2(l)`
- `dpar(f) -> ∇_|| f`
- `curvature(kx, ky, f) -> C(f)`

This makes it straightforward to:

- benchmark against analytic models,
- load tabulated coefficients from external equilibrium tools,
- later add VMEC/field-line tracer-derived coefficients without changing the model core.

## Built-in geometries

- **Shear slab**: `SlabGeometry` (`src/jaxdrb/geometry/slab.py`)
- **Circular tokamak**: `CircularTokamakGeometry` (`src/jaxdrb/geometry/tokamak.py`)
- **s-alpha**: `SAlphaGeometry` (`src/jaxdrb/geometry/tokamak.py`) with Cyclone-like defaults
- **Tabulated**: `TabulatedGeometry` (`src/jaxdrb/geometry/tabulated.py`) from `.npz`

## k_perp^2 in flux-tube form

The perpendicular structure is Fourier and geometry enters through:

$$
k_\perp^2(l) = k_x^2\,g^{xx}(l) + 2 k_x k_y\,g^{xy}(l) + k_y^2\,g^{yy}(l).
$$

Each geometry provider supplies the metric-like coefficients `(gxx, gxy, gyy)` either analytically
or from a tabulated file.

## Curvature operator

Curvature is represented as a linear operator. In the analytic geometries it is implemented in the
form

$$
C(f) = i\,(k_x\,\omega_{d,x}(l) + k_y\,\omega_{d,y}(l))\,f,
$$

but you are free to implement a different (still linear) operator as long as the interface matches.

## Parallel derivative

The default discretization is a periodic second-order centered finite difference:

$$
\frac{df}{dl}\Big|_j \approx \frac{f_{j+1} - f_{j-1}}{2\,\Delta l}.
$$

Geometries may optionally include Jacobian factors through a `dpar_factor(l)` coefficient or a
more sophisticated operator.
