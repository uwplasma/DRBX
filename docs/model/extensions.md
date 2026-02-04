# Model extensions and toggles

This page documents additional physics options and model variants implemented in `jaxdrb`,
alongside the baseline cold-ion electrostatic model described in `model/equations.md`.

## Polarization: Boussinesq vs non-Boussinesq

The baseline closure uses a Boussinesq (constant-density) polarization relation:

$$
\Omega = -k_\perp^2(l)\,\phi.
$$

`jaxdrb` also supports a **linearized non-Boussinesq** polarization option, in which the closure is
linearized about an equilibrium density profile $n_0(l)$:

$$
\Omega = -k_\perp^2(l)\,n_0(l)\,\phi
\quad\Rightarrow\quad
\phi(l) = -\frac{\Omega(l)}{k_\perp^2(l)\,n_0(l)}.
$$

In the current examples, the equilibrium density is $n_0=1$ by default, so Boussinesq and
non-Boussinesq are identical in linear scans unless you provide a nontrivial $n_0(l)$.

**Code mapping**

- Parameter toggle: `DRBParams(boussinesq=True/False)`.
- Implementation: `jaxdrb.models.cold_ion_drb.phi_from_omega(...)` and the `eq.n0(l)` profile.

## Hot ions (adds Ti)

The hot-ion electrostatic model adds an ion-temperature field $T_i$ and a minimal set of couplings:

$$
Y_{\mathrm{hot}} = \bigl(n,\ \Omega,\ v_{\parallel e},\ v_{\parallel i},\ T_e,\ T_i\bigr).
$$

Two key additions are:

1. **Ion pressure in the ion parallel momentum**
   $$
   \frac{\partial v_{\parallel i}}{\partial t}
   = -\nabla_\parallel\left(\phi + \tau_i(n+T_i)\right),
   \qquad \tau_i \equiv T_{i0}/T_{e0}.
   $$
2. **Ion pressure contribution to curvature forcing** via the total pressure perturbation
   $$
   p_{\mathrm{tot}} = (1+\tau_i)\,n + T_e + \tau_i\,T_i,
   \qquad
   C_p = C(p_{\mathrm{tot}}).
   $$

The ion temperature evolution is implemented in a minimal Braginskii-like form:

$$
\frac{\partial T_i}{\partial t}
= \mathcal{D}_{T_i}(\phi) - \frac{2}{3}\nabla_\parallel v_{\parallel i} + D_{T_i}\,\Delta_\perp T_i,
\qquad
\mathcal{D}_{T_i}(\phi)=-ik_y\omega_{T_i}\phi.
$$

**Code mapping**

- Model implementation: `jaxdrb.models.hot_ion_drb`.
- Parameters: `DRBParams(tau_i=..., omega_Ti=..., DTi=...)`.

## Electromagnetic extension (adds psi ~ A_parallel)

The electromagnetic model adds an inductive field `psi` and uses an Ampère closure for the
parallel current:

$$
Y_{\mathrm{em}} = \bigl(n,\ \Omega,\ \psi,\ v_{\parallel i},\ T_e\bigr),
$$

with

$$
j_\parallel = -\nabla_\perp^2\psi \ \to\ k_\perp^2(l)\,\psi.
$$

The induction equation is implemented in a reduced MHD / generalized-Ohm-like form:

$$
\left(\frac{\beta}{2} + \hat{m}_e\,k_\perp^2(l)\right)\frac{\partial\psi}{\partial t}
= -\nabla_\parallel(\phi - n - T_e) - \eta\,j_\parallel + D_\psi\,\Delta_\perp\psi.
$$

This model eliminates $v_{\parallel e}$ by reconstructing it from $v_{\parallel i}$ and $j_\parallel$:

$$
v_{\parallel e} = v_{\parallel i} - j_\parallel.
$$

**Code mapping**

- Model implementation: `jaxdrb.models.em_drb`.
- Parameters: `DRBParams(beta=..., Dpsi=...)`.

## CLI usage

The CLI can select among built-in model variants via `--model`:

```bash
jaxdrb-scan --model cold-ion-es --geom slab --ky-min 0.05 --ky-max 1.0 --nky 32 --out out/
jaxdrb-scan --model hot-ion-es  --geom slab --tau-i 1.0 --omega-Ti 0.8 --out out_hot/
jaxdrb-scan --model em          --geom slab --beta 0.1 --out out_em/
```

See `cli.md` for the full set of flags and defaults.

## MPSE / sheath boundary conditions (Loizu 2012)

For *open* field lines (SOL/limiter), `jaxdrb` can apply magnetic-pre-sheath entrance (MPSE)
boundary conditions inspired by:

- J. Loizu et al., Phys. Plasmas 19, 122307 (2012).

Two MPSE enforcement modes are available (both are **weak** SAT/penalty relaxations applied at the
two field-line endpoints):

- `sheath_bc_model=0` (**simple**): velocity-only BCs for $v_{\parallel i}$ and $v_{\parallel e}$.
- `sheath_bc_model=1` (**loizu2012**): a *linearized*, model-aligned “full set” that also enforces
  endpoint constraints involving $\partial_\parallel \phi$, $\partial_\parallel n$, $\omega$,
  and $\partial_\parallel T_e=0$.

The Loizu2012 option omits transverse-gradient corrections (terms involving $\partial_x$) in this
initial implementation, to remain consistent with `jaxdrb`’s 1D field-line + Fourier-perp closure.

Example:

```bash
python examples/scripts/03_sheath_mpse/loizu2012_full_mpse_bc.py
```

## Sheath heat transmission and secondary electron emission (SEE)

`jaxdrb` includes an **optional** sheath energy-loss closure intended as a stepping stone toward
quantitative scrape-off-layer (SOL) modeling with sources/sinks, recycling, and full sheath models.

### Heat transmission (energy losses)

Fluid sheath theory is often summarized by a sheath heat-flux condition of the form

$$
q_{\parallel e} \big|_{\mathrm{sheath}} \;\approx\; \gamma_e\,n\,T_e\,c_s,
$$

with an analogous ion term $q_{\parallel i}\approx \gamma_i\,n\,T_i\,c_s$. In a reduced 1D field-line
setting (Fourier in $(x,y)$), enforcing this exactly requires a consistent model for parallel
conduction/convection and a Robin-type boundary condition on temperature.

In the current `jaxdrb` linear workflows, this is implemented as a **lightweight end-loss closure**:
when enabled, the RHS receives an additional term localized at the MPSE nodes,

$$
\partial_t T_e \;\leftarrow\; \partial_t T_e \;-\; \nu_{\mathrm{bc}}\,\chi_{\partial\Omega}\,\gamma_e\,T_e,
$$

and similarly for $T_i$ in the hot-ion model. Here $\nu_{\mathrm{bc}}\sim 2/L_\parallel$ is the same
rate scale used by the MPSE SAT/penalty enforcement, and $\chi_{\partial\Omega}$ is a mask that is
nonzero only at the two field-line endpoints.

**Toggles**

- `DRBParams(sheath_heat_on=True)` enables these energy-loss terms.
- `DRBParams(sheath_gamma_auto=True)` uses an automatic estimate for $\gamma_e$ (see below).
- `DRBParams(sheath_gamma_i=3.5)` sets the ion energy transmission factor used when $T_i$ is evolved.

### SEE (secondary electron emission)

Secondary electron emission tends to reduce the floating potential drop. `jaxdrb` represents this
with a simple, constant-yield correction to the sheath parameter $\Lambda$:

$$
\Lambda_{\mathrm{eff}} \;=\; \Lambda + \ln(1-\delta),
\qquad 0 \le \delta < 1,
$$

where $\delta$ is a constant SEE yield. This affects:

- the *automatic* electron energy transmission factor, via
  $$\gamma_e \approx 2 + \Lambda_{\mathrm{eff}},$$
- and the *nonlinear* (nonlinearized) MPSE electron-flow expression, which uses $\Lambda_{\mathrm{eff}}$
  in the floating-potential shift.

**Toggles**

- `DRBParams(sheath_see_on=True, sheath_see_yield=...)`

**Example**

```bash
python examples/scripts/03_sheath_mpse/sheath_heat_see_effects.py --out out_sheath_heat
```

### Notes and limitations

- This closure is intentionally simple and does **not** replace a full sheath model with energy
  balance, recycling, and self-consistent current closure; it is meant as an incremental step that
  remains matrix-free, robust, and differentiable.
- The Loizu (2012) “full set” MPSE option includes $\partial_\parallel T_e=0$ as part of its boundary
  constraints; combining that option with sheath energy-loss terms is not recommended unless you are
  explicitly studying sensitivity to mixed end conditions.
- By default, `jaxdrb` also applies a small additional damping term localized at the sheath nodes
  (`DRBParams.sheath_end_damp_on=True`) to prevent spurious boundary-driven growth in no-drive limits
  for the reduced 1D linear system. Disable this only if you are explicitly benchmarking sensitivity
  to end-loss modeling details.

## User-defined end conditions (benchmarking / development)

In addition to MPSE/sheath closures, `jaxdrb` provides a **user-defined** boundary-condition hook
for the field-line coordinate $l$:

- periodic
- Dirichlet
- Neumann

These are enforced weakly as RHS relaxation terms at the two endpoints and can be useful for
benchmarking sensitivity to end conditions or for nonlinear-preparation experiments.

See `docs/model/boundary-conditions.md` for details and the API mapping.

## Parallel closures and sinks (optional)

Several additional *linear* terms can be enabled to mimic common Braginskii-like closures and
SOL modeling ingredients while keeping the system matrix-free and robust:

- Parallel electron heat conduction:
  $$\partial_t T_e \;\leftarrow\; \partial_t T_e + \chi_{\parallel,T_e}\,\nabla_\parallel^2 T_e.$$
- Parallel flow diffusion/viscosity:
  $$\partial_t v_{\parallel s} \;\leftarrow\; \partial_t v_{\parallel s} + \nu_{\parallel,s}\,\nabla_\parallel^2 v_{\parallel s},
  \qquad s\in\{e,i\}.$$
- Volumetric sinks (simple damping proxies):
  $$\partial_t f \;\leftarrow\; \partial_t f - \nu_{\mathrm{sink},f}\,f.$$

These are controlled by:

- `chi_par_Te`, `nu_par_e`, `nu_par_i`,
- `nu_sink_n`, `nu_sink_Te`, `nu_sink_vpar`.

`jaxdrb` also supports equilibrium-based **Braginskii/Spitzer temperature scalings** for these
coefficients; see `docs/model/braginskii.md`.

Example workflow:

```bash
python examples/scripts/04_closures_transport/parallel_closures_effects.py
```
