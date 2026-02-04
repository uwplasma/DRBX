# 2D Hasegawa–Wakatani (HW2D) model

This nonlinear milestone implements a 2D periodic **Hasegawa–Wakatani-like** drift-wave model as a fast testbed.

## State

The HW2D state is

$$
y = (n, \omega),
$$

where:

- $n(x,y,t)$ is the fluctuating density (in normalized units),
- $\omega(x,y,t)$ is the vorticity,
- $\phi(x,y,t)$ is the electrostatic potential, obtained from a polarization closure.

## Polarization (Poisson solve)

On a periodic domain, we use the Boussinesq polarization closure

$$
\omega = \nabla_\perp^2 \phi,
$$

solved spectrally in Fourier space:

$$
\hat{\phi}(\mathbf{k}) = -\frac{\hat{\omega}(\mathbf{k})}{k_\perp^2},
\qquad \hat{\phi}(\mathbf{0}) = 0.
$$

## Evolution equations

The equations implemented in `jaxdrb.nonlinear.hw2d` are:

$$
\partial_t n + [\phi, n] = -\kappa\,\partial_y \phi + \alpha(\phi - n) + D_n \nabla_\perp^2 n,
$$

$$
\partial_t \omega + [\phi, \omega] = -\kappa\,\partial_y n + \alpha(\phi - n) + D_\omega \nabla_\perp^2 \omega.
$$

Here:

- $[\phi,f] = \partial_x \phi\,\partial_y f - \partial_y \phi\,\partial_x f$ is the $E\\times B$ Poisson bracket,
- $\kappa$ is a background-gradient drive parameter (a proxy for $R/L_n$),
- $\alpha$ controls resistive/adiabatic coupling,
- $D_n, D_\omega$ are stabilizing diffusion coefficients.

This system is not the full SOL drift-reduced Braginskii system; it is a nonlinear testbed with drift-wave-like physics that is useful for validating numerical kernels and preparing the nonlinear DRB transition.

## Implementation notes

- The bracket can be computed by:
  - pseudo-spectral derivatives + real-space product + dealiasing (fast),
  - Arakawa's conservative bracket on a finite-difference grid (robust/conservative),
  - a simple centered-difference bracket (for comparison only).
- Polarization uses an FFT Poisson solve with a zero-mean gauge for $\phi$.

