# Physics Models

This page is the technical map from the governing equations to the source tree.
It is meant to help both new users and developers find where a model lives
before they change a case, add a term, or debug a result. It also states the
main first-principles model forms and the numerical patterns actually used in
the current code, so the docs remain useful even without the manuscript.

## Model Families

`drbx` organizes its native physics into a small set of accuracy-tested
families:

- the JAX-native **Hasegawa-Wakatani drift-wave turbulence** flagship on a
  closed-field-line (periodic flux-tube) plane;
- the **flux-coordinate-independent (FCI) operator stack** for reduced and
  drift-reduced Braginskii dynamics on tokamak and non-axisymmetric
  stellarator geometry, including the 2-field and 4-field reduced models, the
  full electrostatic/electromagnetic drift-reduced Braginskii right-hand side,
  the 3-D FCI sheath closure, the neutral reaction-diffusion component, and the
  perpendicular vorticity inversion;
- the **linear stability / dispersion solver** (`drbx.linear`);
- the compact **1-D fluid, anomalous-diffusion, and electrostatic-vorticity
  deck models** that back the `drbx run` command.

Each family is documented below with its governing form and its kept source
modules.

## Drift-Reduced Braginskii Core

The FCI drift-reduced lanes are built around drift-reduced Braginskii-style
density, momentum, pressure, and potential evolution.

From first principles, these lanes start from the multispecies collisional
moment hierarchy and apply the standard edge/SOL drift ordering
`ω/Ω_i << 1`, `ρ_s/L_⊥ << 1`, and `k_∥ << k_⊥`, so that fast gyromotion is
removed while the dominant parallel transport, sheath losses, and
cross-field drift dynamics are retained. In the literature, this is the same
reduced-fluid family used by GBS, GDB, GRILLIX, and TOKAM3X.

Useful references for the model class and its scope:

- Braginskii transport review:
  [Braginskii 1965](https://link.springer.com/book/10.1007/978-1-4615-2808-1)
- reduced-fluid SOL model and GBS comparison class:
  [Ricci et al. 2012](https://iopscience.iop.org/article/10.1088/0741-3335/54/12/124047)
- multi-component edge/SOL fluid model:
  [Dudson et al. 2024](https://doi.org/10.1016/j.cpc.2023.108991)
- global tokamak edge fluid review:
  [Schwander et al. 2024](https://doi.org/10.1016/j.compfluid.2023.106141)

At the level exposed in the current native models, the code is solving discrete
forms of the following equations.

The resolved particle flux is represented schematically as

```text
Γ_s = n_s u_E + b n_s V_{∥,s} - D_{⊥,s} ∇_⊥ n_s
```

where `u_E = b × ∇⊥ φ / B` is the `E×B` drift.

Operationally:

- continuity is the zeroth velocity moment,
- parallel momentum is the field-aligned projection of the first moment,
- pressure/energy is the reduced second-moment balance with collisional
  closure,
- the electrostatic/vorticity closure replaces full perpendicular ion momentum
  with a drift-ordered polarization/vorticity equation.

### Continuity

For an evolved species density `n_s`:

```text
∂t n_s + ∇·Γ_s = S_{n,s}
```

where `Γ_s` is the resolved advective/diffusive flux and `S_{n,s}` collects the
ionisation, recombination, and charge-exchange sources of the active model
(see the FCI neutral component below).

### Parallel Momentum

For the evolved parallel momentum density `n_s V_{∥,s}`:

```text
∂t (n_s V_{∥,s}) + ∇·(Γ_s V_{∥,s})
  = -∇_∥ p_s + F_{coll,s} + F_{thermal,s} + F_{sheath,s} + ∇_∥·Π_{∥,s}
```

The exact active terms depend on the model: the reduced 2-field and 4-field FCI
lanes carry the benchmark-consistent parallel-momentum structure, while the
full drift-reduced Braginskii FCI right-hand side adds the Braginskii friction,
thermal-force, and sheath target closures.

### Pressure / Energy

For an evolved scalar temperature/pressure variable:

```text
∂t p_s + ∇·(p_s u_s) + γ p_s ∇·u_s = Q_{cond,s} + Q_{coll,s}
```

with the dominant parallel conductive closure written in the standard reduced
form

```text
q_{∥,s} ≈ -κ_{∥,s} ∇_{∥} T_s
```

### Potential / Vorticity Closure

The reduced lanes close the system with a perpendicular polarization/vorticity
relation between the potential `φ`, the vorticity `ω`, and the density/current
state. At the operator level this is the familiar reduced electrostatic
structure

```text
ω = ∇⊥·(C ∇⊥ φ)
```

with model-dependent coefficients `C` and metric terms. The corresponding
vorticity transport equation is represented schematically as

```text
∂t ω + ∇·(ω u_E) = ∇∥ J∥ + S_ω
```

with `S_ω` collecting curvature and sheath source terms.

## Reduced-Fluid Operator Structure

Across the drift-reduced FCI lanes, the discrete operators are built from the
same small set of physical ingredients:

- parallel derivatives `Grad_par(f)` and parallel Laplacians on the
  flux-coordinate-independent (FCI) forward/backward field-line maps;
- perpendicular transport/divergence operators on the staged metric payload;
- electrostatic `E×B` transport, represented through an advection bracket or
  equivalent face-flux reconstruction;
- sheath target closures at open-field-line map exits;
- neutral reaction-diffusion source operators.

The exact equation set differs by model, but the implementation reuses these
operator families rather than encoding each case as an unrelated solver. The
shared FCI parallel/perpendicular gradient and Laplacian stencils live in
[native/fci_operators.py](../src/drbx/native/fci_operators.py), boundary and
halo handling in
[native/fci_boundaries.py](../src/drbx/native/fci_boundaries.py) and
[native/fci_halo.py](../src/drbx/native/fci_halo.py), and the geometry maps
in [geometry/fci_geometry.py](../src/drbx/geometry/fci_geometry.py).

### 3-D FCI Sheath Closure

The non-axisymmetric traced-field-line lane has a Bohm-sheath target closure.
[native/fci_sheath_recycling.py](../src/drbx/native/fci_sheath_recycling.py)
derives endpoint masks from the forward and backward map exits and evaluates

```text
c_s = sqrt((T_e + T_i) / m_i)
Gamma_i,target = N_endpoint n c_s
Gamma_e,target = Gamma_i,target
q_e,target = gamma_e Gamma_i,target T_e
q_i,target = gamma_i Gamma_i,target T_i
Gamma_n,recycle = R_recycle Gamma_i,target
Q_n,recycle = E_recycle Gamma_n,recycle
```

so a fixed fraction `R_recycle` of the ion target flux is returned as a neutral
source. The associated validation campaign checks exact particle recycling,
neutral-energy source accounting, and zero-current particle balance on a 3-D
non-axisymmetric map.

### FCI Neutral Reaction-Diffusion

The compact neutral component in
[native/fci_neutral.py](../src/drbx/native/fci_neutral.py) evaluates neutral
diffusion plus the ionisation, recombination, and charge-exchange sources

```text
S_ion = k_ion n_n n_e sqrt(T_e)
S_rec = k_rec n_i n_e / sqrt(T_e)
S_cx  = k_cx  n_n n_i sqrt(T_n + T_i)
```

and verifies that ionisation/recombination conserve plasma-plus-neutral
particles while charge exchange conserves particles and exchanges ion/neutral
momentum.

### Perpendicular Vorticity Inversion

The vorticity component in
[native/fci_vorticity.py](../src/drbx/native/fci_vorticity.py) applies and
inverts the perpendicular polarization relation

```text
Omega = - div_perp(K_pol grad_perp phi)
K_pol = <n / B^2>      (Boussinesq)
K_pol = n / B^2        (non-Boussinesq)
```

with the metric-weighted perpendicular operator, solved by conjugate gradient.
The Boussinesq and non-Boussinesq operators differ on a nonuniform density
field and become identical to roundoff when `n/B^2` is constant.

### Combined Differentiable RHS

The compact combined state in
[native/fci_drb_rhs.py](../src/drbx/native/fci_drb_rhs.py) is a PyTree RHS
that threads the sheath, neutral, and vorticity components into a single
differentiable right-hand side. It carries the Boussinesq/non-Boussinesq
polarization switch through the potential solve and exposes an opt-in
potential-fed `E×B` advection path for the charged-fluid density, pressure, ion
parallel momentum, and vorticity. Neutral gas density, pressure, and momentum
are deliberately not `E×B`-advected in this closure; they remain controlled by
neutral diffusion and reaction terms.

## Numerical Algorithms

The models above are not solved with one monolithic algorithm. The current
native runtime uses a few distinct numerical patterns.

### Structured Finite-Volume / Flux-Form Updates

The compact 1-D/2-D native deck lanes use explicit flux-form field updates on
the structured mesh and metric payload: face reconstruction, metric-aware
transport operators, and explicit source assembly. In the implementation these
transport kernels live in:

- [native/fluid_1d.py](../src/drbx/native/fluid_1d.py) (periodic fluid MMS)
- [native/vorticity.py](../src/drbx/native/vorticity.py) (electrostatic
  vorticity)
- [native/transport.py](../src/drbx/native/transport.py) (anomalous
  diffusion)

### Exact And Explicit Time Integration

- The anomalous-diffusion lane forms the radial diffusion operator and advances
  it with a matrix exponential (`jax.scipy.linalg.expm`), i.e. an exact linear
  propagator over the step.
- The electrostatic vorticity lane integrates the interior state with an
  adaptive Dormand-Prince solver (`jax.experimental.ode.odeint`).
- The FCI models advance with classical fourth-order Runge-Kutta in
  [native/fci_time_integrator.py](../src/drbx/native/fci_time_integrator.py).
- The Hasegawa-Wakatani flagship is a pseudo-spectral solver in the
  perpendicular plane.

### Elliptic Solves

Potential and related closures are handled through the spectral
Fourier--Helmholtz elliptic solve in
[`solvax.elliptic`](https://github.com/uwplasma/SOLVAX), the reusable
structured-solver library. The electrostatic vorticity lane
([native/vorticity.py](../src/drbx/native/vorticity.py)) inverts its
potential with that operator; the compact FCI vorticity component inverts the
metric-weighted perpendicular operator with conjugate gradient; and the
4-field/DRB lanes invert the conservative perpendicular Laplacian with the
solvax GMRES `PerpLaplacianInverseSolver` (optionally multigrid-preconditioned
with a prefactored LU coarse solve) — see
[Solvers and Design Decisions](solvers_and_design.md).

## Linear Stability And Dispersion

`drbx.linear` is the linear solver of the drift-reduced Braginskii
equations. It linearizes a model about an equilibrium and returns the growth
rates and frequencies of its eigenmodes (`delta ~ exp(lambda t)`,
`gamma = Re lambda`, `Omega = Im lambda`) through:

- `jacobian_operator` + `eigenmodes`: the dense Jacobian of an arbitrary
  JAX right-hand side about an equilibrium, then its spectrum -- the general
  engine for any model on a small grid or single-mode reduction;
- three reduced dispersion operators assembled directly from the model
  equations, so diagonalizing them reproduces the analytic dispersion:
  `resistive_drift_wave_operator` (Hasegawa-Wakatani, benchmark B2),
  `shear_alfven_operator` (electron-inertia shear Alfven, B3), and
  `interchange_operator` (curvature-driven Rayleigh-Taylor).

Source: [src/drbx/linear/](../src/drbx/linear/). Verification:
[Linear Dispersion Benchmark](linear_dispersion_benchmark.md) and
`tests/test_linear_dispersion.py`.

## Hasegawa-Wakatani Drift-Wave Turbulence

The closed-field-line drift-wave turbulence flagship is the JAX-native
Hasegawa-Wakatani model in
[native/hasegawa_wakatani.py](../src/drbx/native/hasegawa_wakatani.py): a
pseudo-spectral two-field solver for the potential and density fluctuations in
the perpendicular plane,

```text
∂t ζ  = -{φ, ζ} + α (φ - n) - ν ∇⊥^4 ζ - μ ζ
∂t n  = -{φ, n} + α (φ - n) - κ ∂y φ - ν ∇⊥^4 n - μ n
ζ = ∇⊥^2 φ
```

with an optional scale-independent friction μ
(`HasegawaWakataniParameters.friction`, default 0) that absorbs the 2-D inverse
cascade so fixed-step runs saturate. Besides `hw_run`, the module ships
`hw_run_flux_history` (a jitted rollout that also returns the sampled
particle-flux history, differentiable end to end).

Its single-mode linear growth reproduces the B2 eigenvalue of the linear
dispersion solver to machine precision, and it is differentiable end-to-end,
enabling gradient-based inverse design through turbulence. It is documented in
[Drift-Wave Turbulence](drift_wave_turbulence.md).

## FCI Reduced And Drift-Reduced Braginskii Models

The FCI stack provides several reduced models on the same operator and geometry
payload:

- **2-field** reduced model (density and parallel velocity) in
  [native/fci_2_field_rhs.py](../src/drbx/native/fci_2_field_rhs.py);
- **4-field** model (density, vorticity, ion and electron parallel velocity),
  with free-decay and blob variants, in
  [native/fci_4_field_rhs.py](../src/drbx/native/fci_4_field_rhs.py);
- the full **electrostatic/electromagnetic drift-reduced Braginskii**
  right-hand side (density, potential, `Te`, `Ti`, ion and electron parallel
  velocity, vorticity) in
  [native/fci_drb_EB_rhs.py](../src/drbx/native/fci_drb_EB_rhs.py).

These are assembled from the shared FCI operators, boundary/halo handling, and
geometry maps described above, and are validated on tokamak and non-axisymmetric
stellarator geometry (see
[Stellarator FCI Validation](stellarator_fci_validation.md) and
[Differentiable FCI Flux Tube](stellarator_fci_differentiable.md)).

Primary source files:

- reduced models:
  [native/fci_2_field_rhs.py](../src/drbx/native/fci_2_field_rhs.py),
  [native/fci_4_field_rhs.py](../src/drbx/native/fci_4_field_rhs.py)
- full drift-reduced Braginskii RHS:
  [native/fci_drb_EB_rhs.py](../src/drbx/native/fci_drb_EB_rhs.py),
  [native/fci_drb_rhs.py](../src/drbx/native/fci_drb_rhs.py)
- sheath / neutral / vorticity closures:
  [native/fci_sheath_recycling.py](../src/drbx/native/fci_sheath_recycling.py),
  [native/fci_neutral.py](../src/drbx/native/fci_neutral.py),
  [native/fci_vorticity.py](../src/drbx/native/fci_vorticity.py)
- operators, boundaries, geometry:
  [native/fci_operators.py](../src/drbx/native/fci_operators.py),
  [native/fci_boundaries.py](../src/drbx/native/fci_boundaries.py),
  [native/fci_halo.py](../src/drbx/native/fci_halo.py),
  [geometry/fci_geometry.py](../src/drbx/geometry/fci_geometry.py),
  [geometry/shifted_torus.py](../src/drbx/geometry/shifted_torus.py)

## Compact Electrostatic Vorticity And Diffusion Deck Models

The `drbx run` command backs four compact, accuracy-tested deck models:

- single-component `evolve_density` (one-rhs);
- anomalous diffusion,
  [native/transport.py](../src/drbx/native/transport.py);
- periodic fluid MMS,
  [native/fluid_1d.py](../src/drbx/native/fluid_1d.py);
- electrostatic vorticity,
  [native/vorticity.py](../src/drbx/native/vorticity.py).

These share the structured mesh and metric handling in
[native/mesh.py](../src/drbx/native/mesh.py) and
[native/metrics.py](../src/drbx/native/metrics.py) and are dispatched by
[native/deck_runner.py](../src/drbx/native/deck_runner.py).

## Electromagnetic Reduced Surfaces

Where electron-parallel dynamics is retained explicitly, the compact
electromagnetic operators in
[native/electromagnetic.py](../src/drbx/native/electromagnetic.py) evolve the
parallel current variable

```text
Ajpar = Σ_s Z_s n_s V_{∥,s}
```

together with the reduced parallel-force balance

```text
0 = -e n_e E_∥ - ∇∥ p_e - η_∥ J_∥
```

The electron-inertia shear-Alfven branch of this system is verified analytically
by the `shear_alfven_operator` in the linear dispersion solver, and the full
electromagnetic drift-reduced Braginskii RHS is provided by
[native/fci_drb_EB_rhs.py](../src/drbx/native/fci_drb_EB_rhs.py).

## Differentiable Analysis Surface

The compact differentiable lanes use the standard JAX gradient map

```text
g(θ) = ∇_θ J(θ)
```

and local Gaussian uncertainty propagation through the linearized pushforward

```text
Σ_Q ≈ G Σ_θ G^T ,  G = ∂Q/∂θ
```

These are the surfaces used by the published sensitivity, uncertainty, and
inverse-design examples (see
[Autodiff And Scaling Examples](autodiff_and_scaling_examples.md)). The
end-to-end differentiable lanes today are the compact native-exact diffusion and
vorticity kernels, the Hasegawa-Wakatani turbulence flagship, and the
differentiable FCI drift-reduced RHS.

## JAX Implementation Boundary

`drbx` deliberately separates:

- fully JAX-native compact kernels used for differentiable reduced lanes,
  profiling, and selected-field 3-D reductions;
- NumPy/SciPy boundary code used for CLI orchestration, file I/O, and output
  serialization.

In practice, the current JAX-native building blocks are:

- `jax.numpy` array kernels
- `@jax.jit`
- `jax.vmap`
- `jax.grad` / `jax.value_and_grad`
- `solvax` (restarted flexible GMRES over matrix-free operators for the FCI
  perpendicular-Laplacian inversion, plus tridiagonal and Fourier–Helmholtz
  solves)

`equinox` and `diffrax` are not used by any promoted kernel; they remain
ecosystem options rather than dependencies. The full solver inventory, with
parameters and design rationale, is on
[Solvers and Design Decisions](solvers_and_design.md).

## Output And Restart

User-facing runs produce:

- summary JSON
- arrays NPZ
- restart NPZ
- verbose event log JSON

Primary source files:

- CLI and argument model:
  [src/drbx/cli.py](../src/drbx/cli.py)
- deck dispatch and portable payload / restart writing:
  [src/drbx/native/deck_runner.py](../src/drbx/native/deck_runner.py),
  [src/drbx/runtime/output.py](../src/drbx/runtime/output.py)

## Validation Rules

Before a capability is treated as accuracy-tested, the working rule is:

- one-RHS agreement on the smallest exercising case;
- one-step agreement on the same case;
- short-window agreement when transient behavior matters;
- operator or boundary unit tests for every new branch;
- at least one physics-facing diagnostic;
- restart equivalence when the workflow is user-facing;
- artifact checks for CLI/example surfaces.

The summary of that contract, with figures, is in
[validation_gallery.md](validation_gallery.md), and the layered test taxonomy is
in [testing_strategy.md](testing_strategy.md).
