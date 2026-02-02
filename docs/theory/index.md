# Theory overview

This section collects the high-level theoretical context behind `jaxdrb`:

- what model we solve (cold-ion drift-reduced Braginskii-like 5-field system),
- what approximations are made to enable a fast, geometry-pluggable, matrix-free solver,
- what “known limits” and qualitative trends we expect to recover.

## What `jaxdrb` is (and is not)

`jaxdrb` is a **linear stability tool** for local (field-line / flux-tube) edge/SOL modes.
It targets workflows common in the SOL literature:

- compute leading eigenvalues $\lambda = \gamma + i\omega$ as functions of $(k_x, k_y)$,
- separate “branches” by varying resistivity and inertia parameters,
- compute the transport proxy $\max(\gamma, 0)/k_y$ and estimate $L_p$ via fixed-point rules
  used in gradient-removal saturation models.

It is **not** a full SOL turbulence code:

- no sheath / line-tied boundary conditions,
- no sources/sinks, no open-field-line connection to divertor plates,
- the default model is electrostatic (an electromagnetic extension model exists, but is still
  intentionally simplified),
- 1D along the field line (perpendicular dependence is Fourier).

Those features are common in SOL codes (GBS/BOUT++/TOKAM3X/etc.) but are intentionally outside
the current scope so we can iterate quickly and keep the solver matrix-free.

## Coordinate and spectral representation

We represent perturbations as:

$$
\tilde{f}(\psi,\alpha,l,t) = \hat{f}(l,t)\,\exp\{ i k_x \psi + i k_y \alpha \},
$$

where:

- $\psi$ is a “radial-like” flux coordinate,
- $\alpha$ is a field-line label (Clebsch-like),
- $l$ is the parallel coordinate along the field line.

Perpendicular derivatives reduce to algebraic factors:

$$
\nabla_\perp^2 \hat{f} = -k_\perp^2(l)\,\hat{f},
\qquad
k_\perp^2(l) = k_x^2 g^{xx}(l) + 2 k_x k_y g^{xy}(l) + k_y^2 g^{yy}(l),
$$

with $(g^{xx}, g^{xy}, g^{yy})$ supplied by the geometry provider.

## Polarization closure and vorticity variable

By default we use the Boussinesq (constant-density) polarization closure:

$$
\Omega(l,t) \equiv -k_\perp^2(l)\,\phi(l,t),
$$

so that:

$$
\phi(l,t) = -\frac{\Omega(l,t)}{k_\perp^2(l)}.
$$

This definition of $\Omega$ is convenient in a Fourier-perp representation because it avoids
solving a Poisson equation while preserving the structure of drift-reduced models.

## Geometry abstraction

Geometry enters only through:

- $k_\perp^2(l)$,
- the parallel derivative $\nabla_\parallel$ (implemented by a 1D FD operator along $l$),
- the curvature operator $C(\cdot)$.

This makes it straightforward to swap between:

- analytic slab and tokamak toy models,
- a tabulated field-line geometry loaded from a `.npz`,
- (future) equilibria / field-line tracing outputs.

## Solvers

`jaxdrb` provides two complementary linear solvers:

1. **Eigenvalue**: matrix-free Arnoldi to compute leading eigenvalues.
2. **Initial-value**: integrate $\dot{v} = Jv$ and estimate $\gamma$ from the late-time log-norm slope
   (with a renormalized “continuous-time power method” formulation).

Both rely on the same primitive: a matrix-free Jacobian–vector product $v \mapsto Jv$
computed using `jax.linearize`.
