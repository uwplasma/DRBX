# Solvers overview

`jaxdrb` provides two complementary ways to compute linear stability:

1. **Initial-value (time domain)**:
   - build `matvec(v) = J·v` at an equilibrium `Y0`,
   - integrate `dv/dt = J v`,
   - estimate the growth rate from the log-norm slope.
2. **Eigenvalue (frequency domain)**:
   - use a matrix-free Arnoldi method to approximate leading eigenvalues of `J`.

Both methods use the same primitive: a matrix-free Jacobian-vector product.

## Matrix-free Jacobian-vector products

Given the RHS function `F(Y) = dY/dt`, and an equilibrium `Y0`, the Jacobian is:

$$
J = \\left.\\frac{\\partial F}{\\partial Y}\\right|_{Y=Y_0}.
$$

Instead of forming `J` explicitly, we use JAX to construct a linear operator:

$$
v \\mapsto J v
$$

with `jax.linearize`. See `src/jaxdrb/linear/matvec.py`.

This is critical for:

- keeping memory use small,
- enabling geometry-agnostic operators,
- supporting large `nl` without building a dense matrix.

