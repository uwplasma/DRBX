# Normalization and parameters

The implementation is intentionally lightweight and uses a generic “drift-reduced SOL” style
normalization, where all coefficients are **dimensionless knobs**. This keeps the system usable
for qualitative stability studies without committing to a specific experimental mapping.

The model parameters live in [`src/jaxdrb/models/params.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/models/params.py) (`DRBParams`).

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

## Numerics and units (practical guidance)

### Numerical discretization (high level)

`jaxdrb` uses:

- a 1D grid in the parallel/field-line coordinate `l`,
- finite differences for $\nabla_\parallel$,
- a Fourier-perpendicular closure so $\nabla_\perp^2 f \to -k_\perp^2(l) f$.

For open-field-line geometries, boundary-localized terms (e.g. MPSE sheath relaxation) are applied
at the two endpoints via a mask.

### Dimensionalizing results

The code is intentionally **nondimensional**: most parameters are “knobs” in a reduced
normalization. If you want to map growth rates and frequencies to physical units, a robust way is
to choose:

- a reference sound speed $c_{s0}$ (e.g. $c_{s0}=\sqrt{T_{e0}/m_i}$),
- a reference parallel length scale $L_0$ (e.g. a connection length).

If `l` is interpreted as $l_\mathrm{phys}/L_0$, then:

$$
\gamma_\mathrm{phys} \approx \gamma_\mathrm{code}\,\frac{c_{s0}}{L_0},\qquad
\omega_\mathrm{phys} \approx \omega_\mathrm{code}\,\frac{c_{s0}}{L_0}.
$$

Similarly, if your perpendicular metric is normalized so $k_\perp^2 \sim k_x^2 + k_y^2$ (as in the
analytic slab models), then a common interpretation is:

$$
k_{y,\mathrm{code}} \sim k_{y,\mathrm{phys}}\,\rho_{s0},
$$

with $\rho_{s0}=c_{s0}/\Omega_{ci}$.

### Interpreting “knobs”

Some parameters (notably `eta` and `me_hat`) are designed to expose qualitative branch structure
(resistive-like vs inertial-like). Mapping them to a single experimental collisionality is
non-trivial and depends on the specific reduced model and normalization used in the target paper.
For quantitative studies, treat these as parameters to calibrate against the specific reference
model/equilibrium you want to match.
