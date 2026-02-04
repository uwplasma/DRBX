# Known limits and qualitative checks

This page documents **qualitative** limits and trends that `jaxdrb` is designed to recover.
These checks are intended to answer “are we solving the equations consistently?” rather than to
claim quantitative agreement with any particular code or experiment.

## 1) No-drive limit: neutral stability

With:

- no background gradient drives ($\omega_n = \omega_{T_e} = 0$),
- curvature turned off,
- no explicit diffusion,

the system should not create free energy. In this limit, the linear operator should be
**neutrally stable**, with leading eigenvalues having $\Re(\lambda)\approx 0$.

This is enforced by a unit test:

- `tests/test_known_limits.py::test_no_drive_limit_is_neutrally_stable`

## 2) Curvature drive increases growth (interchange/ballooning-like)

When curvature is enabled, increasing the curvature coefficient in a simple geometry should
increase the growth rate of curvature-driven modes (all else equal).

This is checked in:

- `tests/test_slab_dispersion.py::test_ballooning_like_curvature_and_shear_trends`

## 3) Magnetic shear tends to stabilize curvature-driven modes

In field-line-following representations, increasing magnetic shear typically increases $k_\perp^2(l)$
variation and can reduce net drive by shearing apart radially localized structures.

This is also checked in:

- `tests/test_slab_dispersion.py::test_ballooning_like_curvature_and_shear_trends`

## 4) Resistive/inertial “branch” control knobs

Many SOL/edge papers separate low-frequency branches by the relative importance of:

- parallel resistive coupling (often “resistive drift wave”),
- electron inertia (“inertial drift wave”),

at fixed geometry and equilibrium gradients.

In `jaxdrb`:

- `eta` is a resistive coupling knob in Ohm’s law,
- `me_hat` controls electron inertia.

The literature-aligned examples in `examples/scripts/06_literature_tokamak_sol/` show how scanning these parameters produces
distinct growth-rate curves and different maximizers of $\max(\gamma,0)/k_y$.

## 5) Connection length effect (via `q` in analytic tokamak geometry)

In open-field-line SOL physics, longer parallel connection length generally reduces parallel losses
and can increase growth rates of modes that are otherwise stabilized by parallel dynamics.

In the analytic circular tokamak geometry, the parallel derivative is scaled as:

$$
\nabla_\parallel \approx \frac{1}{qR_0}\frac{d}{d\theta}.
$$

Increasing `q` decreases the parallel derivative strength (longer effective connection length),
and can increase growth rates in representative cases.

This is checked in:

- `tests/test_known_limits.py::test_connection_length_effect_via_q`

## 6) Non-Boussinesq polarization reduces to Boussinesq when $n_0=1$

When using the linearized non-Boussinesq closure
$$
\Omega=-k_\perp^2 n_0 \phi,
$$
and the equilibrium density is uniform with $n_0=1$, the closure is identical to the Boussinesq
form and the linear operator should match exactly.

This is checked in:

- `tests/test_polarization_models.py::test_non_boussinesq_matches_boussinesq_when_n0_is_one`

## 7) Hot-ion model reduces to cold-ion model as $\tau_i\to 0$

The hot-ion electrostatic variant adds an ion temperature field and ion-pressure couplings.
In the limit $\tau_i \equiv T_{i0}/T_{e0} \to 0$, it should reduce to the cold-ion model.

This is checked in:

- `tests/test_hot_ion_model.py::test_hot_ion_tau_zero_matches_cold_ion_leading_growth`

## 8) Electromagnetic no-drive limit: neutral stability

The electromagnetic extension adds an inductive field `psi` and an Ampère closure for $j_\parallel$.
In the no-drive, no-dissipation limit it should also be neutrally stable.

This is checked in:

- `tests/test_em_model.py::test_em_no_drive_limit_is_neutrally_stable`
