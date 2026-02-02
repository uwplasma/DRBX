# Normalization and parameters

The implementation is intentionally lightweight and uses a generic “drift-reduced SOL” style
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

For hot-ion variants, an additional temperature-gradient drive is available:

$$
\mathcal{D}_{T_i}(\phi) = -i k_y\,\omega_{T_i}\,\phi.
$$

## Parallel physics

- `eta`: resistive coupling coefficient between electron and ion parallel flows.
- `me_hat`: electron inertia parameter. Decreasing `me_hat` tends to push the system toward a more
  resistive response; increasing it tends to emphasize inertial effects.

## Electromagnetism

The electromagnetic model variant uses:

- `beta`: normalized beta controlling the strength of inductive coupling,
- `Dpsi`: optional perpendicular diffusion applied to the inductive field `psi`.

See: `model/extensions.md`.

## Hot ions

The hot-ion model variant uses:

- `tau_i`: ratio $T_{i0}/T_{e0}$ controlling ion-pressure contributions,
- `omega_Ti`: ion temperature-gradient drive,
- `DTi`: ion-temperature diffusion coefficient.

See: `model/extensions.md`.

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
