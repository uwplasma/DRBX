# Model equations (cold-ion DRB, v1)

This page documents the *implemented* system in `jaxdrb` v1.

> The model here is intentionally a “workhorse” drift-reduced Braginskii-like system used for
> qualitative edge/SOL linear stability exploration. It is not a full SOL model (no sheath BCs,
> no neutral physics, no realistic sources/sinks, no gyroviscosity, etc.).

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

## The v1 RHS

The implemented RHS is in `src/jaxdrb/models/cold_ion_drb.py`.

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

Define perpendicular diffusion (Fourier Laplacian):

$$
\Delta_\perp f = -k_\perp^2(l)\,f.
$$

Then the v1 model is:

### Continuity

$$
\frac{\partial n}{\partial t}
= \mathcal{D}_n(\phi)
- \nabla_\parallel v_{\parallel e}
 C_\phi - C_p
 D_n\,\Delta_\perp n.
$$

### Vorticity

$$
\frac{\partial \Omega}{\partial t}
= -\nabla_\parallel j_\parallel
 C_p
 D_\Omega\,\Delta_\perp \Omega.
$$

### Electron parallel momentum (Ohm’s law + inertia)

$$
\hat{m}_e\,\frac{\partial v_{\parallel e}}{\partial t}
= -\nabla_\parallel(\phi - n - T_e)
 \eta\,(v_{\parallel i} - v_{\parallel e}).
$$

`eta` here is a resistive coupling coefficient and `me_hat` is the electron inertia knob.

### Ion parallel momentum (cold ions)

$$
\frac{\partial v_{\parallel i}}{\partial t}
= -\nabla_\parallel \phi.
$$

### Electron temperature

$$
\frac{\partial T_e}{\partial t}
= \mathcal{D}_{T_e}(\phi)
- \frac{2}{3}\nabla_\parallel v_{\parallel e}
 D_{T_e}\,\Delta_\perp T_e.
$$

## What is *not* in v1

The following are intentionally deferred:

- Full nonlinear $E\times B$ brackets `[\phi, f]` (single-mode self-nonlinearity is zero).
- Non-Boussinesq polarization: $\Omega = \nabla_\perp\cdot(n\nabla_\perp\phi)$.
- Sheath boundary conditions and realistic SOL closures.
- Two-dimensional Poisson solves (we rely on the Fourier closure for v1).
- Full Braginskii closures for viscosity, heat flux, etc.

