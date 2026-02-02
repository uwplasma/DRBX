# Shear slab geometry

`SlabGeometry` (`src/jaxdrb/geometry/slab.py`) is the simplest analytic benchmark geometry.

## Definition

The field-line coordinate is `l ≈ theta` on a periodic domain `[-L/2, L/2)`. Metric-like
coefficients are:

$$
g^{xx}=1,\qquad g^{xy}= \hat{s}\,\theta,\qquad g^{yy}=1+(\hat{s}\,\theta)^2.
$$

This yields:

$$
k_\perp^2(\theta)=k_x^2 + 2 k_x k_y\,\hat{s}\,\theta + k_y^2\left[1+(\hat{s}\,\theta)^2\right].
$$

## Parallel derivative

The default is:

$$
\nabla_\parallel f \approx \frac{df}{d\theta}
$$

implemented with periodic finite differences.

## Curvature

In the slab geometry, curvature is represented with a simple cosine variation:

$$
C(f) = i\,(k_x\,0 + k_y\,\omega_{d,0}\cos\theta)\,f.
$$

Set `curvature0=0` for a curvature-free drift-wave-like benchmark.
