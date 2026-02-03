# Circular tokamak and s-alpha geometry

`jaxdrb` includes two related analytic tokamak models:

- `CircularTokamakGeometry` for a circular cross-section, large-aspect-ratio tokamak
- `SAlphaGeometry` (s-alpha model) with an `alpha` parameter
- open-field-line variants `OpenCircularTokamakGeometry` and `OpenSAlphaGeometry`

Both are implemented in [`src/jaxdrb/geometry/tokamak.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/geometry/tokamak.py).

## Circular tokamak (large aspect ratio)

The circular tokamak model uses:

- `theta` as the field-line coordinate
- `q` (safety factor) and `R0` (major radius) for the parallel derivative
- a simple inverse-aspect-ratio `epsilon = r/R0` to define a magnetic field variation

### Parallel derivative

We approximate:

$$
\nabla_\parallel \approx \frac{1}{q R_0}\,\frac{d}{d\theta}.
$$

For open-field-line variants, the same scaling is used but the derivative is evaluated with an
open-boundary finite-difference stencil (no periodic wrapping).

### Magnetic field strength

We use a simple circular tokamak field strength:

$$
B(\theta) = \frac{1}{1+\epsilon\cos\theta}.
$$

### Curvature operator

In these analytic tokamak geometries:

$$
C(f) = i\,k_y\,\omega_{d,0}\cos\theta\,B(\theta)\,f.
$$

## s-alpha geometry

The s-alpha model modifies the ballooning/shear structure by introducing a pressure-gradient-like
parameter `alpha` in the metric cross-term:

$$
g^{xy}(\theta) = \hat{s}\,\theta - \alpha \sin\theta,
\qquad
g^{yy}(\theta)=1+\bigl(g^{xy}(\theta)\bigr)^2.
$$

This produces an effective *field-line dependent* $k_\perp^2(\theta)$ even for `kx=0`:

$$
k_\perp^2(\theta)\big|_{k_x=0} = k_y^2\left[1+\left(\hat{s}\,\theta - \alpha\sin\theta\right)^2\right].
$$

The parallel derivative and curvature operator are the same form as the circular tokamak model.

## Cyclone base case

The “Cyclone Base Case” (CBC) is a common s-alpha benchmark configuration (often used for ITG
gyrokinetics). In CBC, `alpha = 0` (electrostatic / zero-beta) and typical geometric parameters are:

- `q ≈ 1.4`
- `shat ≈ 0.796`
- `epsilon ≈ 0.18`

See:

- `examples/1_simple/03_salpha_cyclone_ky_scan.py` (ky scan),
- `examples/2_intermediate/02_cyclone_kxky_scan.py` (kx–ky scan).
