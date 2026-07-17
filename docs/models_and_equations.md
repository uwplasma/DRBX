# Models and Governing Equations

This page states the governing equations of every model that ships in
`jax_drb`, exactly as they are discretized in the source code: the evolved
unknowns, the normalization, the boundary conditions, the module that
implements each equation set, and the tests that gate it. The equations below
are transcribed from the module docstrings and RHS assembly code, not from a
separate derivation — if this page and the code ever disagree, the code wins
and this page should be fixed.

For the numerical methods used to *solve* these systems (GMRES, multigrid,
tridiagonal solves, spectral inversion, RK4, sharding), see
[Solvers and Design Decisions](solvers_and_design.md).

## Hasegawa-Wakatani drift-wave turbulence

**Module:** [`native/hasegawa_wakatani.py`](../src/jax_drb/native/hasegawa_wakatani.py) —
the closed-field-line (periodic flux-tube) turbulence flagship.

Two fields in the perpendicular plane, the vorticity \(\zeta\) and the density
fluctuation \(n\):

$$
\begin{aligned}
\partial_t \zeta &= -\{\phi, \zeta\} + \alpha\,(\phi - n)
  - \nu\, \nabla_\perp^4 \zeta - \mu\, \zeta,\\
\partial_t n &= -\{\phi, n\} - \kappa\, \partial_y \phi + \alpha\,(\phi - n)
  - \nu\, \nabla_\perp^4 n - \mu\, n,\\
\zeta &= \nabla_\perp^2 \phi
  \quad\Longleftrightarrow\quad \hat\phi_k = -\hat\zeta_k / k^2 ,
\end{aligned}
$$

where \(\{a,b\} = \partial_x a\, \partial_y b - \partial_y a\, \partial_x b\)
is the \(E\times B\) Poisson bracket, \(\alpha\) the adiabaticity (parallel
electron response), \(\kappa\) the background density-gradient drive, \(\nu\)
a \(\nabla_\perp^4\) hyperviscosity absorbing the grid scale, and \(\mu\) an
optional scale-independent friction (default 0) modeling large-scale drag
(sheath or neutral damping); by absorbing the 2-D inverse cascade at the box
scale it lets a fixed-step run reach a statistically steady saturated state.

- **Unknowns:** \(\hat\zeta_k, \hat n_k\) (spectral); \(\phi\) is diagnostic.
- **Normalization:** standard Hasegawa-Wakatani units (lengths in \(\rho_s\),
  times in \(L_n / c_s\) absorbed into \(\alpha, \kappa\)).
- **Boundary conditions:** doubly periodic; solved pseudo-spectrally with
  2/3-rule dealiasing of the bracket.
- **Discretization:** dealiased pseudo-spectral RHS, classical RK4 stepping
  (`hw_step` / `hw_run`), all pure JAX (`jit`/`grad`/`vmap`).
- **Gates:** `tests/test_hasegawa_wakatani.py` — single-mode growth reproduces
  the `resistive_drift_wave_operator` eigenvalue (benchmark B2) to machine
  precision, ideal-invariant conservation, outward-transport sign, and a
  finite-difference-verified gradient of the final fluctuation energy with
  respect to the adiabaticity.
- References: Hasegawa & Wakatani, *PRL* 50, 682 (1983); Numata et al.,
  *Phys. Plasmas* 14, 102312 (2007). Tutorial:
  [From zero to a turbulence movie](tutorial_hasegawa_wakatani.md).

## FCI 2-field reduced model

**Module:** [`native/fci_2_field_rhs.py`](../src/jax_drb/native/fci_2_field_rhs.py) —
the smallest 3-D model on the flux-coordinate-independent (FCI) operator
stack; also the model used by the multi-device `shard_map` lane.

Evolved fields are the density \(n\) and the parallel velocity
\(V_\parallel\), with the potential given by the Boltzmann-like closure
\(\phi = \ln(n / n_0)\) against a stored background \(n_0\). As assembled in
`compute_2field_rhs`:

$$
\begin{aligned}
\partial_t n &= -\frac{\{\phi, n\}}{\rho_\star B}
  + \frac{2}{B}\, C(n) - \frac{2 n}{B}\, C(\phi)
  - n\, \nabla_\parallel V_\parallel + S_n,\\
\partial_t V_\parallel &= -\frac{\{\phi, V_\parallel\}}{\rho_\star B}
  + S_{V},
\end{aligned}
$$

where \(\{\cdot,\cdot\}\) is the metric-aware perpendicular Poisson bracket
(`poisson_bracket_op`), \(C(\cdot)\) the curvature operator built from
precomputed curvature coefficients (`curvature_op` /
`build_curvature_coefficients`), and \(\nabla_\parallel\) the direct
field-aligned gradient \(b^i \partial_i\) (`grad_parallel_op_direct`).

- **Unknowns:** \(n, V_\parallel\) (plus the frozen background \(n_0\)).
- **Normalization:** drift-normalized with the single scale
  \(\rho_\star\); \(B\) is the local field magnitude from the geometry payload.
- **Boundary conditions:** per-field face-BC payloads
  (`BoundaryFaceBC3D`, Dirichlet/Neumann masks per face) and optional
  cut-wall payloads; periodic axes default `(False, True, True)`; physical
  sides close with one-sided stencils.
- **Gates:** `tests/test_mms_slab_2_field.py`,
  `tests/test_mms_shifted_torus_2_field.py` (manufactured solutions),
  `tests/test_fci_sharded_2field.py` (sharded step bit-exact vs single
  device).

## FCI 4-field interchange model

**Module:** [`native/fci_4_field_rhs.py`](../src/jax_drb/native/fci_4_field_rhs.py) —
density, vorticity, and both parallel velocities, with the potential obtained
by inverting the perpendicular Laplacian every RK4 stage.

As assembled in `_assemble_4field_non_diffusive_rhs` (Poisson bracket +
curvature + parallel coupling contributions):

$$
\begin{aligned}
\partial_t n &= -\frac{\{\phi, n\}}{\rho_\star B}
 + \frac{2 T_e}{B} C(n) - \frac{2 n}{B} C(\phi)
 - n \nabla_\parallel V_{\parallel e},\\
\partial_t \omega &= -\frac{\{\phi, \omega\}}{\rho_\star B}
 + \frac{2 B T_e}{n} C(n)
 + \frac{B^2}{n}\left(\nabla_\parallel V_{\parallel i}
   - \nabla_\parallel V_{\parallel e}\right),\\
\partial_t V_{\parallel i} &= -\frac{\{\phi, V_{\parallel i}\}}{\rho_\star B}
 - \frac{T_e}{n} \nabla_\parallel n,\\
\partial_t V_{\parallel e} &= -\frac{\{\phi, V_{\parallel e}\}}{\rho_\star B}
 + \frac{m_i}{m_e} \nabla_\parallel \phi
 - \frac{m_i}{m_e} \frac{T_e}{n} \nabla_\parallel n,
\end{aligned}
$$

closed each stage by the conservative perpendicular polarization inversion

$$
-\nabla_\perp \cdot \left( \nabla_\perp \phi \right) = -\,\omega ,
$$

solved with the lineax GMRES `PerpLaplacianInverseSolver` (tolerance,
iteration, and restart limits are model parameters:
`phi_inversion_tol/maxiter/restart` in `Fci4FieldRhsParameters`). The
free-decay and blob variants (`Fci4FieldFreeDecayParameters`,
`Fci4FieldBlobParameters`) add conservative perpendicular diffusion
\(D_f \nabla_\perp\!\cdot\!\nabla_\perp f\) on each field
(`_assemble_4field_diffusion_rhs`).

- **Unknowns:** \(n, \omega, V_{\parallel i}, V_{\parallel e}\); \(\phi\) is
  reconstructed per stage (warm-started from the previous stage's solution).
- **Normalization:** drift-normalized (\(\rho_\star\)), constant \(T_e\),
  mass ratio \(m_i/m_e\) (default 1836).
- **Boundary conditions:** per-field face/cut-wall BC payloads; free-decay
  (homogeneous Neumann) closures in the shipped stellarator turbulence
  examples; on limiter-open geometry the Bohm sheath sink is applied on the
  target endpoint cells (see the open-SOL closure below).
- **Gates:** `tests/test_mms_shifted_torus_4_field.py`,
  `tests/test_shifted_torus_4_field_free_decay.py`,
  `tests/test_shifted_torus_4_field_blob.py`,
  `tests/test_stellarator_turbulence.py`,
  `tests/test_multigrid_preconditioner.py` (phi-inversion preconditioning).
- Tutorial: [Stellarator FCI turbulence](tutorial_stellarator_fci.md).

## Electrostatic/electromagnetic drift-reduced Braginskii RHS

**Modules:**
[`native/fci_drb_EB_rhs.py`](../src/jax_drb/native/fci_drb_EB_rhs.py)
(electrostatic Boussinesq scaffold with state
\(n, \phi, T_e, T_i, V_i, V_e, \omega\)) and
[`native/fci_drb_rhs.py`](../src/jax_drb/native/fci_drb_rhs.py)
(the compact combined differentiable RHS threading the sheath, neutral, and
vorticity closures into one PyTree). The general drift-reduced Braginskii
moment structure (continuity, parallel momentum, pressure with
\(q_\parallel \approx -\kappa_\parallel \nabla_\parallel T\), and the
polarization/vorticity closure) is documented with literature anchors in
[Physics Models](physics_models.md). Gates:
`tests/test_mms_shifted_torus_EB.py`, `tests/test_shifted_torus_EB_blob.py`,
`tests/test_fci_differentiable.py`.

## SOL flux tube with Bohm sheath boundaries

**Module:** [`native/sol_flux_tube.py`](../src/jax_drb/native/sol_flux_tube.py)
on the open slab geometry
([`geometry/open_slab.py`](../src/jax_drb/geometry/open_slab.py)).

The open-field-line counterpart to the closed flux tubes: an isothermal Euler
system for the density \(n\) and parallel momentum \(m = n v\) along the
field coordinate \(z \in [0, L]\),

$$
\begin{aligned}
\partial_t n + \partial_z (n v) &= S_n,\\
\partial_t (n v) + \partial_z\!\left(n v^2 + n c_s^2\right) &= 0,
\end{aligned}
$$

with a Gaussian upstream particle source \(S_n\) at the parallel midplane and
**Bohm sheath outflow** \(|v| \ge c_s\) enforced at both targets
(\(z = 0\) and \(z = L\)). Faces use a Rusanov (local Lax–Friedrichs) flux;
stepping is RK4. The steady state is the classic two-point SOL solution:
stagnation at the midplane, Mach 1 at each target, target density half the
upstream density.

The sheath/recycling closure
[`native/fci_sheath_recycling.py`](../src/jax_drb/native/fci_sheath_recycling.py)
evaluates, on the FCI endpoint masks (the cells whose forward/backward maps
exit the domain),

$$
\begin{aligned}
c_s &= \sqrt{(T_e + T_i)/m_i}, \qquad
\Gamma_{i,\mathrm{target}} = \Gamma_{e,\mathrm{target}} = n\, c_s,\\
q_{e,\mathrm{target}} &= \gamma_e\, \Gamma_i T_e, \qquad
q_{i,\mathrm{target}} = \gamma_i\, \Gamma_i T_i,\\
\Gamma_{n,\mathrm{recycle}} &= R\, \Gamma_{i,\mathrm{target}}, \qquad
Q_{n,\mathrm{recycle}} = E_{\mathrm{recycle}}\, \Gamma_{n,\mathrm{recycle}},
\end{aligned}
$$

so a fraction \(R\) of the ion target flux returns as a neutral source, with
exact particle-recycling, zero-current, and neutral-energy accounting
identities.

- **Unknowns:** \(n\), \(m = n v\) per flux tube; each \((x, y)\) column is an
  independent tube sharing the target plates.
- **Normalization:** isothermal, normalized to the sound speed and upstream
  density.
- **Gates:** `tests/test_open_field_line_sol.py` (endpoint masks on exactly
  the two target planes, roundoff-closed sheath accounting, two-point steady
  state). Page: [Open-Field-Line SOL](open_field_line_sol.md); tutorial:
  [Building an open SOL](tutorial_open_sol.md).

## hermes-3 neutral system: recycling and detachment

**Package:** [`native/neutrals/`](../src/jax_drb/native/neutrals) — a
self-contained, pure-JAX implementation of the hermes-3 hydrogenic
plasma–neutral closure.

### Atomic rates and reaction sources

[`atomic_rates.py`](../src/jax_drb/native/neutrals/atomic_rates.py) packages
the AMJUEL double-polynomial fits for ionization and recombination
\(\langle\sigma v\rangle(T_e, n_e)\) and the AMJUEL H.2 3.1.8 polynomial for
charge exchange \(\langle\sigma v\rangle(T_{\mathrm{eff}})\) (clamped to the
fitted range \(T \in [0.1, 10^4]\,\mathrm{eV}\),
\(n \in [10^{14}, 10^{22}]\,\mathrm{m^{-3}}\); tables ship with the package).
[`reactions.py`](../src/jax_drb/native/neutrals/reactions.py) assembles the
source channels

$$
S_{\mathrm{iz}} = \langle\sigma v\rangle_{\mathrm{iz}}\, n_n n_e, \qquad
S_{\mathrm{rec}} = \langle\sigma v\rangle_{\mathrm{rec}}\, n_i n_e, \qquad
S_{\mathrm{cx}} = \langle\sigma v\rangle_{\mathrm{cx}}\, n_n n_i ,
$$

in the **Galilean-invariant** form: every particle transfer carries the source
species' momentum \(m V\) and thermal energy \(\tfrac{3}{2} T\), charge
exchange adds a frictional heating \(\tfrac{1}{2} m R\, \Delta V^2\) from the
ion–atom velocity difference, and the electron channel carries the AMJUEL
ionization cost / recombination radiation. Ion and neutral particle and
momentum sources cancel exactly.

### 1D recycling SOL

[`recycling_sol_model.py`](../src/jax_drb/native/neutrals/recycling_sol_model.py)
couples the plasma (ion density, parallel momentum on a *prescribed*
hot-upstream/cold-target temperature profile) to a recycled neutral density:

- neutrals recycled from the Bohm target flux (fraction \(R\)),
- neutral parallel diffusion solved **implicitly** (solvax tridiagonal solve —
  unconditionally stable),
- ionization/recombination applied as an operator-split, per-cell implicit
  update (stable against the stiff ionization source),
- charge-exchange + recombination friction dragging the ion flow.

### Self-consistent detachment

[`detachment_sol_model.py`](../src/jax_drb/native/neutrals/detachment_sol_model.py)
also evolves the plasma pressure \(P\), adding

$$
\partial_t P \supset
\partial_z\!\left(\kappa_0\, T^{5/2}\, \partial_z T\right)
\;-\; n_e\, n_Z\, L(T)\; -\; \gamma\, n\, c_s\, T\big|_{\mathrm{target}}
\;+\; S_P ,
$$

with **Spitzer parallel conduction** \(\kappa \sim T^{5/2}\) solved implicitly
(solvax tridiagonal), the **self-limiting** radiative/ionization energy sink
applied semi-implicitly as \(P \leftarrow P / (1 + \Delta t\,
\mathrm{loss\ rate})\) (cannot drive \(P < 0\), switches off as the plasma
cools), a Bohm sheath heat sink \(\gamma\, n c_s T\) at the target, and an
upstream power source. Scanning upstream density at fixed power reproduces the
SD1D detachment signature: the target cools through 1 eV into the recombining
regime and the target ion flux **rolls over**.

- **Unknowns:** \(n_i\), \(m_i = A\, n_i v\), \(n_n\) (recycling model);
  plus \(P\) (detachment model).
- **Normalization:** hermes-3 units — density\(/N_{\mathrm{norm}}\),
  temperature\(/T_{\mathrm{norm}}\) (eV), velocity over the reference sound
  speed; the atomic rates use physical units internally
  (`PlasmaNormalization`).
- **Boundary conditions:** upstream density Dirichlet-pinned at the
  stagnation point, Bohm outflow at the target, recycled-neutral influx at the
  target.
- **Gates:** `tests/test_native_atomic_rates.py`,
  `tests/test_native_reactions.py`, `tests/test_native_recycling_sol.py`,
  `tests/test_native_detachment_sol.py`, `tests/test_fci_neutrals_3d.py`
  (3-D FCI coupling), `tests/test_detachment_control.py` (autodiff through
  the stiff solve). Page: [Neutrals and Recycling](neutrals_recycling.md).

The compact 3-D neutral reaction-diffusion component used by the combined FCI
RHS lives in [`native/fci_neutral.py`](../src/jax_drb/native/fci_neutral.py)
(sources \(S_{\mathrm{iz}} = k_{\mathrm{iz}} n_n n_e \sqrt{T_e}\),
\(S_{\mathrm{rec}} = k_{\mathrm{rec}} n_i n_e / \sqrt{T_e}\),
\(S_{\mathrm{cx}} = k_{\mathrm{cx}} n_n n_i \sqrt{T_n + T_i}\) plus neutral
diffusion), with conservation gated in `tests/test_fci_neutrals_3d.py`.

## The linearized-DRB engine

**Package:** [`src/jax_drb/linear/`](../src/jax_drb/linear) — growth rates
and frequencies of the drift-reduced Braginskii family.

Linearizing any model about an equilibrium \(u_0\),

$$
\partial_t\, \delta u = A\, \delta u, \qquad
A = \left.\frac{\partial\, \mathrm{rhs}}{\partial u}\right|_{u_0}, \qquad
\delta \sim e^{\lambda t},\quad
\gamma = \mathrm{Re}\,\lambda,\quad \Omega = \mathrm{Im}\,\lambda .
$$

Two entry points, one engine:

- `jacobian_operator(rhs, equilibrium)` builds \(A\) from **any** JAX RHS with
  `jax.jacfwd` (no hand derivation) and `eigenmodes(A)` returns the sorted
  spectrum ([`linear/eigen.py`](../src/jax_drb/linear/eigen.py));
- three analytic single-mode operators assembled directly from the model
  equations ([`linear/dispersion.py`](../src/jax_drb/linear/dispersion.py)):
  `resistive_drift_wave_operator` (Hasegawa-Wakatani, benchmark B2),
  `shear_alfven_operator` (electron-inertia shear Alfvén wave,
  \(\omega = k_\parallel v_A / \sqrt{1 + k_\perp^2 d_e^2}\), benchmark B3),
  and `interchange_operator` (curvature-driven flute mode,
  \(\gamma = \sqrt{g \kappa}\, k_y / k_\perp\)).

Every analytic limit is reproduced to machine precision in
`tests/test_linear_dispersion.py`; the regime survey figure and literature
anchors are on
[The Linearized Drift-Reduced Braginskii Solver](linear_dispersion_benchmark.md),
with the parameter survey in `examples/benchmarks/linear_drb_survey.py`.

## Compact deck models (CLI lanes)

The `jax_drb run` TOML lanes solve smaller accuracy-tested systems documented
in [Physics Models](physics_models.md): anomalous radial diffusion (matrix
exponential propagator, [`native/transport.py`](../src/jax_drb/native/transport.py)),
the periodic 1-D fluid MMS system ([`native/fluid_1d.py`](../src/jax_drb/native/fluid_1d.py)),
and the electrostatic vorticity model
([`native/vorticity.py`](../src/jax_drb/native/vorticity.py)) whose potential
solve is the solvax spectral Fourier–Helmholtz inversion. Gates:
`tests/test_native_transport.py`, `tests/test_native_fluid_1d.py`,
`tests/test_mms_convergence.py`, `tests/test_native_vorticity.py`.
