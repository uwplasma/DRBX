# Model equations (cold-ion DRB)

This page documents the *implemented* system in `jaxdrb`.

> The model here is intentionally a “workhorse” drift-reduced Braginskii-like system used for
> qualitative edge/SOL linear stability exploration. It is not a full SOL model (no full
> sheath boundary-condition *set*, no neutral physics, no realistic sources/sinks, no gyroviscosity,
> etc.). For open-field-line geometries, `jaxdrb` includes a simplified MPSE Bohm-sheath boundary
> enforcement option and a lightweight volumetric end-loss proxy (see below).

## Fields and closure

We evolve the 5-field state:

$$
Y(l,t) = \bigl(n,\ \Omega,\ v_{\parallel e},\ v_{\parallel i},\ T_e\bigr).
$$

The electrostatic potential is obtained from a Boussinesq polarization closure in Fourier form:

$$
\Omega = \nabla_\perp^2 \phi
\quad\Rightarrow\quad
\Omega = -k_\perp^2(l)\,\phi,
$$

so

$$
\phi(l) = -\frac{\Omega(l)}{k_\perp^2(l)}.
$$

In the implementation we apply a small floor to avoid division by very small $k_\perp^2$:

$$
k_\perp^2 \leftarrow \max\{k_\perp^2,\ k_{\perp,\min}^2\}.
$$

## Operators

All physics is expressed through three geometry-provided operators:

1. **Parallel derivative**
   $$\nabla_\parallel f \equiv b\cdot\nabla f,$$
   discretized as a periodic finite difference along `l`.
2. **Curvature operator** $C(f)$, implemented as a linear operator acting on the scalar `f`.
3. **Perpendicular Laplacian** in Fourier form:
   $$\nabla_\perp^2 f \to -k_\perp^2(l)\,f.$$

The curvature operator is deliberately abstract: different geometries can implement different
definitions of $C(\cdot)$ appropriate for the chosen normalization and coordinate conventions.

## The implemented RHS

The implemented RHS is in [`src/jaxdrb/models/cold_ion_drb.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/models/cold_ion_drb.py).

Define the parallel current:

$$
j_\parallel = v_{\parallel i} - v_{\parallel e}.
$$

Define background-gradient drives (local approximation):

$$
\mathcal{D}_n(\phi) = -i k_y\,\omega_n\,\phi,
\qquad
\mathcal{D}_{T_e}(\phi) = -i k_y\,\omega_{T_e}\,\phi.
$$

Define curvature contributions:

$$
C_\phi = C(\phi),
\qquad
C_p = C(n + T_e).
$$

Define the electron-temperature curvature compressibility term (cold-ion Braginskii-style):

$$
C_{T_e} = \frac{2}{3}\,C\!\left(\frac{7}{2}T_e + n - \phi\right).
$$

Define perpendicular diffusion (Fourier Laplacian):

$$
\Delta_\perp f = -k_\perp^2(l)\,f.
$$

Then the model is:

### Continuity

$$
\frac{\partial n}{\partial t}
= \mathcal{D}_n(\phi)
- \nabla_\parallel v_{\parallel e}
 + C_p - C_\phi
 + D_n\,\Delta_\perp n.
$$

### Vorticity

$$
\frac{\partial \Omega}{\partial t}
= \nabla_\parallel j_\parallel
 + C_p
 + D_\Omega\,\Delta_\perp \Omega.
$$

### Electron parallel momentum (Ohm’s law + inertia)

$$
\hat{m}_e\,\frac{\partial v_{\parallel e}}{\partial t}
= \nabla_\parallel(\phi - n - \alpha_{T_e}T_e)
 - \eta\,(v_{\parallel e} - v_{\parallel i}).
$$

`eta` here is a resistive coupling coefficient and `me_hat` is the electron inertia knob. The
coefficient $\alpha_{T_e}$ defaults to $1.71$ (Braginskii electron thermal force).

### Ion parallel momentum (cold ions)

$$
\frac{\partial v_{\parallel i}}{\partial t}
= -\nabla_\parallel \phi.
$$

### Electron temperature

$$
\frac{\partial T_e}{\partial t}
= \mathcal{D}_{T_e}(\phi)
 + C_{T_e}
 - \frac{2}{3}\nabla_\parallel v_{\parallel e}
 + D_{T_e}\,\Delta_\perp T_e.
$$

## Open-field-line sheath closures (MPSE BCs + loss proxy)

For open-field-line geometries (e.g. `OpenSlabGeometry`), sheath physics enters through
boundary conditions at the **magnetic pre-sheath entrance (MPSE)**. In the reduced Braginskii/SOL
literature (e.g. Loizu-style treatments), the simplest cold-ion Bohm-sheath relations are:

$$
v_{\parallel i} = \pm(1-\delta)\,c_s,\qquad
v_{\parallel e} = \pm c_s\,\exp\!\left(\Lambda - \frac{e\phi}{T_e}\right),
$$

with:

- $c_s = \sqrt{T_e}$ (cold-ion limit),
- $\Lambda \approx \tfrac12\ln\!\left(m_i/(2\pi m_e)\right)$,
- $\delta$ an optional transmission correction (we default to $\delta=0$).

### MPSE boundary conditions (implemented)

`jaxdrb` implements a **linearized MPSE** closure for *perturbations* about a Bohm-matched equilibrium
(see Loizu et al. (2012) in [References](../references.md) for a full treatment). With $\phi$ interpreted as a floating-potential-shifted
perturbation (so the equilibrium satisfies ambipolarity), and with the evolved fields $(\phi, T_e)$
representing perturbations, the linearized boundary targets used by `jaxdrb` are:

$$
\delta v_{\parallel i} = \pm(1-\delta)\,\frac{T_e}{2},\qquad
\delta v_{\parallel e} = \pm\left(\frac{T_e}{2} - \phi\right).
$$

Numerically, these are enforced weakly at the two ends of the open field line using a relaxation
(SAT/penalty-style) term with a rate scaled as:

$$
\nu_\mathrm{bc} \approx \texttt{sheath\_bc\_nu\_factor}\,\frac{2}{L_\parallel}.
$$

See: [`src/jaxdrb/models/sheath.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/models/sheath.py).

### Volumetric end-loss proxy (optional)

As a lightweight alternative (useful for quick scans), `jaxdrb` also provides an optional **volumetric**
end-loss proxy controlled by `DRBParams.sheath_loss_on`:

$$
\nu_\mathrm{sh} \;\approx\; \texttt{sheath\_loss\_nu\_factor}\,\frac{2}{L_\parallel},
$$

where $L_\parallel$ is the parallel domain length (the span of the `l` grid). When enabled, the
RHS receives additional stabilizing loss terms:

$$
\partial_t n \to \partial_t n - \nu_\mathrm{sh}\,n,\quad
\partial_t T_e \to \partial_t T_e - \nu_\mathrm{sh}\,T_e,\quad
\partial_t \Omega \to \partial_t \Omega - \nu_\mathrm{sh}\,\Omega,\quad
\partial_t v_{\parallel e} \to \partial_t v_{\parallel e} - \nu_\mathrm{sh}\,v_{\parallel e},\quad
\partial_t v_{\parallel i} \to \partial_t v_{\parallel i} - \nu_\mathrm{sh}\,v_{\parallel i}.
$$

This closure is intended as a lightweight proxy for end-plate losses at Bohm sheaths in reduced
SOL models. It is **not** a substitute for full sheath boundary conditions at the magnetic
pre-sheath entrance.

## What is not implemented (yet)

The following are intentionally deferred:

- Full nonlinear $E\times B$ brackets `[\phi, f]` (single-mode self-nonlinearity is zero).
- Full Loizu-style MPSE boundary conditions (including incidence-angle terms and additional boundary
  closures for other fields beyond $(v_{\parallel e}, v_{\parallel i})$).
- Two-dimensional Poisson solves (we rely on the Fourier closure).
- Full Braginskii closures for viscosity, heat flux, etc.
