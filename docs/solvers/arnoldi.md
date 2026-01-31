# Arnoldi eigenvalue method (matrix-free)

`jaxdrb` uses a basic matrix-free Arnoldi method to compute Ritz approximations to eigenvalues of
the Jacobian.

## Krylov subspace

Given an initial vector `v0`, Arnoldi builds an orthonormal basis for the Krylov subspace:

$$
\\mathcal{K}_m(A, v_0) = \\operatorname{span}\\{v_0, A v_0, A^2 v_0, \\ldots, A^{m-1} v_0\\}.
$$

It produces an upper Hessenberg matrix `H_m` such that:

$$
A Q_m \\approx Q_m H_m,
$$

where columns of `Q_m` are the orthonormal Krylov basis vectors.

Eigenvalues of `H_m` (“Ritz values”) approximate eigenvalues of `A`.

## Residual estimate

The Ritz pair residual can be estimated from the last subdiagonal element of the Hessenberg matrix.
The implementation returns a residual norm estimate per Ritz value.

## Implementation notes

- `jaxdrb` flattens the `State` pytree into a complex vector for the linear algebra (NumPy).
- The only expensive operation is repeated application of `matvec(v)`.
- For difficult cases (e.g., small `ky`) convergence may require large `m`.

The CLI implements a simple adaptive strategy:

1. run Arnoldi with `m = --arnoldi-m`,
2. compute the relative residual for the leading (largest real part) Ritz value,
3. if it is above `--arnoldi-tol`, increase `m` (up to `--arnoldi-max-m` or the full dimension).

See `src/jaxdrb/linear/arnoldi.py` and the logic in `src/jaxdrb/cli/main.py`.

