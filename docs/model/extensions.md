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
python examples/3_advanced/11_loizu2012_full_mpse_bc.py
```

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

Example workflow:

```bash
python examples/2_intermediate/06_parallel_closures_effects.py
```
