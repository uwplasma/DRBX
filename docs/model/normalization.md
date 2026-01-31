# Normalization and parameters (v1)

The v1 implementation is intentionally lightweight and uses a generic “drift-reduced SOL” style
normalization, where all coefficients are **dimensionless knobs**. This keeps the system usable
for qualitative stability studies without committing to a specific experimental mapping.

The model parameters live in `src/jaxdrb/models/params.py` (`DRBParams`).

## Background gradients

`omega_n` and `omega_Te` represent background-gradient drives that enter as:

$$
\mathcal{D}_n(\phi) = -i k_y\,\omega_n\,\phi, \qquad
\mathcal{D}_{T_e}(\phi) = -i k_y\,\omega_{T_e}\,\phi.
$$

These are consistent with a local approximation to `-[phi, n0]` and `-[phi, Te0]` for an equilibrium
with constant gradients and a single Fourier mode in the perpendicular directions.

## Parallel physics

- `eta`: resistive coupling coefficient between electron and ion parallel flows.
- `me_hat`: electron inertia parameter. Decreasing `me_hat` tends to push the system toward a more
  resistive response; increasing it tends to emphasize inertial effects.

## Curvature drive

Curvature can be toggled on/off (`curvature_on`). The magnitude and structure of curvature forcing
is geometry-dependent through the geometry provider’s `curvature(kx, ky, f)` implementation.

## Perpendicular diffusion

Simple perpendicular diffusion is included for numerical stability:

$$
D\,\Delta_\perp f \to -D\,k_\perp^2(l)\,f.
$$

These terms are not meant to be physically complete; they primarily regularize short-wavelength
behavior and improve robustness of the eigen/initial-value computations.

