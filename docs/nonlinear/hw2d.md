# 2D Hasegawa–Wakatani (HW2D) model

This nonlinear milestone implements a 2D periodic **Hasegawa–Wakatani-like** drift-wave model as a fast testbed.

See `docs/validation.md` for the validation plots and tests anchored to drift-wave turbulence literature.

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

- $[\phi,f] = \partial_x \phi\,\partial_y f - \partial_y \phi\,\partial_x f$ is the $E\times B$ Poisson bracket,
- $\kappa$ is a background-gradient drive parameter (a proxy for $R/L_n$),
- $\alpha$ controls resistive/adiabatic coupling,
- $D_n, D_\omega$ are stabilizing diffusion coefficients.

This system is not the full SOL drift-reduced Braginskii system; it is a nonlinear testbed with drift-wave-like physics that is useful for validating numerical kernels and preparing the nonlinear DRB transition.

## Implementation notes

- The bracket can be computed by:
  - pseudo-spectral derivatives + real-space product + dealiasing (fast),
  - Arakawa's conservative bracket on a finite-difference grid (robust/conservative; default for long nonlinear runs),
  - a simple centered-difference bracket (for comparison only).
- Polarization uses an FFT Poisson solve with a zero-mean gauge for $\phi$.
- Optional hyperdiffusion terms $-\nu_4\nabla_\perp^4(\cdot)$ are supported (commonly used to stabilize the enstrophy cascade).
- Optional “modified HW” coupling can be enabled by applying $\alpha(\phi-n)$ only to non-zonal components (to avoid unphysical damping of zonal flows).

## Boundary conditions

The default nonlinear configuration is **periodic** in both $x$ and $y$, which enables the
fastest algorithms (FFT Poisson solve and pseudo-spectral bracket).

For development and benchmarking, `jaxdrb` also supports **Dirichlet** and **Neumann** boundary
conditions in $(x,y)$ using:

- finite-difference derivatives and Laplacian operators,
- a matrix-free conjugate-gradient (CG) Poisson solve for $\phi$.

This path is currently more limited and slower than the periodic/spectral path, but it is
useful for preparing the transition to open-boundary nonlinear SOL simulations.
