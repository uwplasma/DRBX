# Braginskii / Spitzer closures

This page documents the **Braginskii-like transport closures** implemented in `jaxdrb`.

`jaxdrb` focuses on drift-reduced edge/SOL models, so we currently implement a **minimal, robust, and
matrix-free** subset of classical Braginskii/Spitzer-Härm transport intended to:

- improve physical realism for SOL studies (temperature-dependent parallel losses and resistive coupling),
- remain fast on CPU and compatible with matrix-free Arnoldi and Diffrax initial-value solvers,
- keep the full pipeline differentiable in JAX.

## Summary of implemented scalings

When enabled (`DRBParams.braginskii_on=True`), the following **equilibrium-based** scalings are applied:

1) **Spitzer resistivity**

$$
\eta \;\propto\; T_e^{-3/2}.
$$

2) **Spitzer–Härm parallel transport proxies** (heat conduction / viscosity-like diffusion)

$$
\chi_\parallel \;\propto\; T^{5/2},
\qquad
\nu_\parallel \;\propto\; T^{5/2}.
$$

In the hot-ion model, the ion-temperature-based scalings use the equilibrium ion temperature
$$
T_{i0} = \tau_i\,T_{e0}.
$$

## Why “equilibrium-based” in linear scans?

In a linear scan, `jaxdrb` evolves perturbations about a specified equilibrium profile:

$$
f(l,t) = f_0(l) + \tilde{f}(l,t).
$$

For matrix-free eigenvalue computations, we want the RHS to remain efficiently linear in the perturbation.
Therefore, the Braginskii coefficients are evaluated on the **equilibrium profiles** $(T_{e0}(l), T_{i0}(l))$
and treated as **spatially varying multipliers** on the linear operators.

This approach captures the most important SOL-relevant effect: **temperature-dependent parallel losses and
resistive coupling** through $T_{e0}(l)$.

Future nonlinear milestones will extend these to **state-dependent** coefficients evaluated on the evolving
fields, while preserving differentiability.

## Code mapping

Coefficient functions live in:

- `jaxdrb/models/braginskii.py`

and are used by all field-line model variants:

- `jaxdrb/models/cold_ion_drb.py`
- `jaxdrb/models/hot_ion_drb.py`
- `jaxdrb/models/em_drb.py`

The core helper functions are:

- `eta_parallel(params, eq)` for $\eta(T_{e0})$,
- `chi_par_Te(params, eq)` and `chi_par_Ti(params, eq)` for $\chi_\parallel$,
- `nu_par_e(params, eq)` and `nu_par_i(params, eq)` for $\nu_\parallel$.

To keep gradients well-defined, equilibrium temperatures are passed through a smooth positive floor:

$$
T \leftarrow T_{\min} + w\,\mathrm{softplus}\!\left(\frac{T-T_{\min}}{w}\right),
$$

implemented as `smooth_floor(...)`.

## Parameters and toggles

In `DRBParams`:

- Master switch: `braginskii_on`
- Per-effect toggles:
  - `braginskii_eta_on` (Spitzer $\eta$ scaling)
  - `braginskii_kappa_e_on`, `braginskii_kappa_i_on` ($\chi_\parallel$ scalings)
  - `braginskii_visc_e_on`, `braginskii_visc_i_on` ($\nu_\parallel$ scalings)
- Reference/scaling knobs:
  - `braginskii_Tref` (reference temperature)
  - `braginskii_T_floor`, `braginskii_T_smooth` (smooth positivity floor)

CLI flags:

- `--braginskii`
- `--braginskii-Tref`, `--braginskii-T-floor`, `--braginskii-T-smooth`
- `--no-braginskii-eta`, `--no-braginskii-kappa-e`, `--no-braginskii-kappa-i`,
  `--no-braginskii-visc-e`, `--no-braginskii-visc-i`

Equilibrium scalars (currently constant along the field line):

- `--eq-n0`
- `--eq-Te0`

## Example

```bash
python examples/2_intermediate/09_braginskii_closures_effects.py --out out_braginskii
```

This produces a panel comparing $\gamma(k_y)$ and $\max(\gamma,0)/k_y$ with and without Braginskii scalings,
and shows how changing $T_{e0}$ modifies the effective resistivity and parallel transport.

## References

For the full classical transport theory, see the standard Braginskii and Spitzer–Härm literature,
and the references cited in the SOL papers in `docs/references.md`.

