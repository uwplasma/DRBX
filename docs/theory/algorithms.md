# Algorithms and numerical methods

This page describes how `jaxdrb` computes growth rates and eigenvalues **without** forming large
matrices.

## 1) Matrix-free Jacobian–vector products

Let the nonlinear model be written as:

$$
\frac{dY}{dt} = F(Y),
$$

and choose an equilibrium $Y_0$ (by default, the equilibrium is the zero-perturbation state).
The linearized operator is the Jacobian:

$$
J = \left.\frac{\partial F}{\partial Y}\right|_{Y=Y_0}.
$$

Instead of forming $J$ explicitly, `jaxdrb` builds the linear map

$$
v \mapsto Jv
$$

using `jax.linearize`. This is the core primitive used by both the eigenvalue and initial-value
solvers.

Implementation:

- `jaxdrb.linear.matvec.linear_matvec`

## 2) Arnoldi (Krylov) eigenvalue solver

Arnoldi builds an orthonormal basis for the Krylov subspace:

$$
\mathcal{K}_m(J, v_0) = \operatorname{span}\{v_0, Jv_0, J^2 v_0, \ldots, J^{m-1} v_0\}.
$$

It produces:

$$
J Q_m \approx Q_m H_m,
$$

where:

- $Q_m$ contains the Krylov basis vectors,
- $H_m$ is an $m\times m$ upper-Hessenberg matrix.

Eigenvalues of $H_m$ (“Ritz values”) approximate eigenvalues of $J$. `jaxdrb` sorts by largest real
part (largest $\gamma$).

Practical details:

- For cases that converge slowly (near-marginal growth, nearly-degenerate modes, or non-normal operators),
  `scan_ky` adapts the Krylov dimension `m` until a relative residual is below
  a tolerance, capped at `arnoldi_max_m`.
- `arnoldi_leading_ritz_vector` constructs a Ritz vector $v = Q_m y$ for plotting eigenfunctions.

Implementation:

- `jaxdrb.linear.arnoldi`
- scan logic: `jaxdrb.analysis.scan`

## 3) Initial-value growth-rate estimation

The initial-value method integrates:

$$
\frac{dv}{dt} = Jv,
$$

and estimates $\gamma$ from the slope of $\log\|v(t)\|$ over a late-time window.

### Renormalized system (continuous-time power method)

To avoid overflow/underflow, `jaxdrb` integrates a renormalized system:

$$
v(t) = \exp(a(t)) u(t),
$$

where $u(t)$ is continuously renormalized. Choosing $a'(t)$ as the instantaneous Rayleigh quotient
removes the dominant exponential growth from $u$:

$$
a'(t) = \gamma(t),\qquad
u'(t) = Ju - \gamma(t)\,u,
$$

with:

$$
\gamma(t) = \Re\left[\frac{u^\dagger Ju}{u^\dagger u}\right],\qquad
\omega(t) = \Im\left[\frac{u^\dagger Ju}{u^\dagger u}\right].
$$

We then fit $a(t)$ and the accumulated phase to obtain $(\gamma,\omega)$.

Implementation:

- `jaxdrb.linear.growthrate.estimate_growth_rate`
- Differentiable variant (useful for optimization workflows): `estimate_growth_rate_jax`

## 4) Perpendicular operators in a flux-tube representation

The key simplification is to represent perpendicular structure as a single Fourier mode
$\exp(i k_x \psi + i k_y \alpha)$, so:

- $\nabla_\perp^2 f \to -k_\perp^2(l)\,f$,
- polarization closure is algebraic: $\Omega = -k_\perp^2 \phi$.

This keeps the operator application cheap and makes matrix-free eigenvalue methods practical.
