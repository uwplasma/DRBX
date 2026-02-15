from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D


def _pad_x(u: jnp.ndarray, dx: float, bc: BC2D) -> jnp.ndarray:
    # Return u padded with one ghost cell in x at both ends: shape (nx+2, ny).
    if bc.kind_x == 0:
        gl = u[-1:, :]
        gr = u[0:1, :]
    elif bc.kind_x == 1:
        gl = 2.0 * bc.x_value - u[1:2, :]
        gr = 2.0 * bc.x_value - u[-2:-1, :]
    else:
        gl = u[1:2, :] - 2.0 * dx * bc.x_grad
        gr = u[-2:-1, :] + 2.0 * dx * bc.x_grad
    return jnp.concatenate([gl, u, gr], axis=0)


def _pad_y(u: jnp.ndarray, dy: float, bc: BC2D) -> jnp.ndarray:
    # Return u padded with one ghost cell in y at both ends: shape (nx, ny+2).
    if bc.kind_y == 0:
        gl = u[:, -1:]
        gr = u[:, 0:1]
    elif bc.kind_y == 1:
        gl = 2.0 * bc.y_value - u[:, 1:2]
        gr = 2.0 * bc.y_value - u[:, -2:-1]
    else:
        gl = u[:, 1:2] - 2.0 * dy * bc.y_grad
        gr = u[:, -2:-1] + 2.0 * dy * bc.y_grad
    return jnp.concatenate([gl, u, gr], axis=1)


def _pad_coeff_x(n: jnp.ndarray, bc: BC2D) -> jnp.ndarray:
    """Pad a variable coefficient field with a single ghost cell in x.

    For non-periodic boundaries we extend by constant edge values to preserve
    positivity of n and maintain an SPD discrete operator.
    """

    if bc.kind_x == 0:
        gl = n[-1:, :]
        gr = n[0:1, :]
    else:
        gl = n[0:1, :]
        gr = n[-1:, :]
    return jnp.concatenate([gl, n, gr], axis=0)


def _pad_coeff_y(n: jnp.ndarray, bc: BC2D) -> jnp.ndarray:
    """Pad a variable coefficient field with a single ghost cell in y."""

    if bc.kind_y == 0:
        gl = n[:, -1:]
        gr = n[:, 0:1]
    else:
        gl = n[:, 0:1]
        gr = n[:, -1:]
    return jnp.concatenate([gl, n, gr], axis=1)


def ddx(u: jnp.ndarray, dx: float, bc: BC2D) -> jnp.ndarray:
    if bc.kind_x == 0:
        return (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2.0 * dx)
    up = _pad_x(u, dx, bc)
    return (up[2:, :] - up[:-2, :]) / (2.0 * dx)


def ddy(u: jnp.ndarray, dy: float, bc: BC2D) -> jnp.ndarray:
    if bc.kind_y == 0:
        return (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2.0 * dy)
    up = _pad_y(u, dy, bc)
    return (up[:, 2:] - up[:, :-2]) / (2.0 * dy)


def laplacian(u: jnp.ndarray, dx: float, dy: float, bc: BC2D) -> jnp.ndarray:
    if bc.kind_x == 0 and bc.kind_y == 0:
        return (jnp.roll(u, -1, axis=0) - 2.0 * u + jnp.roll(u, 1, axis=0)) / dx**2 + (
            jnp.roll(u, -1, axis=1) - 2.0 * u + jnp.roll(u, 1, axis=1)
        ) / dy**2

    upx = _pad_x(u, dx, bc)
    d2x = (upx[2:, :] - 2.0 * upx[1:-1, :] + upx[:-2, :]) / dx**2
    upy = _pad_y(u, dy, bc)
    d2y = (upy[:, 2:] - 2.0 * upy[:, 1:-1] + upy[:, :-2]) / dy**2
    return d2x + d2y


def biharmonic(u: jnp.ndarray, dx: float, dy: float, bc: BC2D) -> jnp.ndarray:
    """Return ∇⁴(u) using two applications of the FD Laplacian."""

    return laplacian(laplacian(u, dx, dy, bc), dx, dy, bc)


def div_n_grad(u: jnp.ndarray, n: jnp.ndarray, dx: float, dy: float, bc: BC2D) -> jnp.ndarray:
    """Return ∇·(n ∇u) using a symmetric flux form.

    This is the variable-coefficient operator needed for non-Boussinesq polarization:
        -∇·(n ∇phi) = Omega.
    """

    upx = _pad_x(u, dx, bc)
    npx = _pad_coeff_x(n, bc)
    n_xp = 0.5 * (npx[1:-1, :] + npx[2:, :])
    n_xm = 0.5 * (npx[1:-1, :] + npx[:-2, :])
    du_xp = (upx[2:, :] - upx[1:-1, :]) / dx
    du_xm = (upx[1:-1, :] - upx[:-2, :]) / dx
    div_x = (n_xp * du_xp - n_xm * du_xm) / dx

    upy = _pad_y(u, dy, bc)
    npy = _pad_coeff_y(n, bc)
    n_yp = 0.5 * (npy[:, 1:-1] + npy[:, 2:])
    n_ym = 0.5 * (npy[:, 1:-1] + npy[:, :-2])
    du_yp = (upy[:, 2:] - upy[:, 1:-1]) / dy
    du_ym = (upy[:, 1:-1] - upy[:, :-2]) / dy
    div_y = (n_yp * du_yp - n_ym * du_ym) / dy

    return div_x + div_y


def boundary_mask(nx: int, ny: int, *, bc: BC2D) -> jnp.ndarray:
    """Mask for boundary nodes relevant to non-periodic BCs."""

    x_b = (bc.kind_x != 0) * (jnp.arange(nx) == 0) + (bc.kind_x != 0) * (jnp.arange(nx) == nx - 1)
    y_b = (bc.kind_y != 0) * (jnp.arange(ny) == 0) + (bc.kind_y != 0) * (jnp.arange(ny) == ny - 1)
    mx = x_b.astype(bool)[:, None]
    my = y_b.astype(bool)[None, :]
    return mx | my


def enforce_bc_relaxation(
    u: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    bc: BC2D,
    nu: float,
) -> jnp.ndarray:
    """Return an RHS term that relaxes boundary values toward the BC targets.

    - Dirichlet: u(boundary) -> value
    - Neumann:   u(boundary) -> u(neighbor) ± h*grad  (1st-order implied value)
    - Periodic:  no enforcement term
    """

    if nu == 0.0 or (bc.kind_x == 0 and bc.kind_y == 0):
        return jnp.zeros_like(u)

    nx, ny = u.shape
    mask = boundary_mask(nx, ny, bc=bc).astype(u.dtype)

    # Default target = current (no forcing) then override edges.
    target = u

    # X boundaries
    if bc.kind_x == 1:
        target = target.at[0, :].set(bc.x_value)
        target = target.at[-1, :].set(bc.x_value)
    elif bc.kind_x == 2:
        target = target.at[0, :].set(u[1, :] - dx * bc.x_grad)
        target = target.at[-1, :].set(u[-2, :] + dx * bc.x_grad)

    # Y boundaries
    if bc.kind_y == 1:
        target = target.at[:, 0].set(bc.y_value)
        target = target.at[:, -1].set(bc.y_value)
    elif bc.kind_y == 2:
        target = target.at[:, 0].set(u[:, 1] - dy * bc.y_grad)
        target = target.at[:, -1].set(u[:, -2] + dy * bc.y_grad)

    return -nu * mask * (u - target)


def _laplacian_homogeneous(u: jnp.ndarray, dx: float, dy: float, bc: BC2D) -> jnp.ndarray:
    """Linear Laplacian with homogeneous BCs.

    - periodic: periodic wrapping
    - dirichlet: homogeneous (value=0) via zero padding ghosts
    - neumann: homogeneous (grad=0) via reflection ghosts
    """

    if bc.kind_x == 0 and bc.kind_y == 0:
        return laplacian(u, dx, dy, bc)

    def pad_x_h(u_):
        if bc.kind_x == 1:
            return jnp.pad(u_, ((1, 1), (0, 0)), mode="constant", constant_values=0.0)
        if bc.kind_x == 2:
            gl = u_[1:2, :]
            gr = u_[-2:-1, :]
            return jnp.concatenate([gl, u_, gr], axis=0)
        # periodic
        gl = u_[-1:, :]
        gr = u_[0:1, :]
        return jnp.concatenate([gl, u_, gr], axis=0)

    def pad_y_h(u_):
        if bc.kind_y == 1:
            return jnp.pad(u_, ((0, 0), (1, 1)), mode="constant", constant_values=0.0)
        if bc.kind_y == 2:
            gl = u_[:, 1:2]
            gr = u_[:, -2:-1]
            return jnp.concatenate([gl, u_, gr], axis=1)
        gl = u_[:, -1:]
        gr = u_[:, 0:1]
        return jnp.concatenate([gl, u_, gr], axis=1)

    upx = pad_x_h(u)
    d2x = (upx[2:, :] - 2.0 * upx[1:-1, :] + upx[:-2, :]) / dx**2
    upy = pad_y_h(u)
    d2y = (upy[:, 2:] - 2.0 * upy[:, 1:-1] + upy[:, :-2]) / dy**2
    return d2x + d2y


def _dct1_even(x: jnp.ndarray) -> jnp.ndarray:
    """DCT-I along axis=0 via an even extension + FFT (unnormalized)."""

    n = x.shape[0]
    if n == 1:
        return x
    ext = jnp.concatenate([x, x[-2:0:-1, :]], axis=0)
    coeffs = jnp.fft.fft(ext, axis=0)
    return coeffs[:n, :].real


def _idct1_even(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse DCT-I along axis=0 for the unnormalized convention in _dct1_even."""

    n = x.shape[0]
    if n == 1:
        return x
    return _dct1_even(x) / (2.0 * (n - 1))


def inv_laplacian_mixed_fft(
    rhs: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    bc: BC2D,
    gauge_epsilon: float | None = None,
    nan_guard: bool = True,
) -> jnp.ndarray:
    """Fast mixed-BC Poisson solve using DCT-I (Neumann) in x and FFT in y.

    Supported BCs:
      - Neumann in x (zero gradient), periodic in y.

    This matches the second-order FD Laplacian spectrum for these BCs.
    """

    if bc.kind_x != 2 or bc.kind_y != 0:
        raise ValueError("inv_laplacian_mixed_fft supports Neumann x + periodic y only.")
    if bc.x_grad != 0.0 or bc.y_grad != 0.0:
        raise ValueError("inv_laplacian_mixed_fft assumes homogeneous Neumann/periodic BCs.")

    if gauge_epsilon is None:
        gauge_epsilon = 1e-12

    nx, ny = rhs.shape
    rhs0 = rhs - jnp.mean(rhs)

    rhs_x = _dct1_even(rhs0)
    rhs_hat = jnp.fft.fft(rhs_x, axis=1)

    kx = jnp.arange(nx, dtype=rhs.dtype)
    lam_x = 4.0 * jnp.sin(0.5 * jnp.pi * kx / (nx - 1)) ** 2 / dx**2
    ky = jnp.arange(ny, dtype=rhs.dtype)
    lam_y = 4.0 * jnp.sin(jnp.pi * ky / ny) ** 2 / dy**2
    lam = lam_x[:, None] + lam_y[None, :]
    lam = jnp.where(lam > 0.0, lam, float(gauge_epsilon))

    u_hat = -rhs_hat / lam
    u_hat = u_hat.at[0, 0].set(0.0)
    u_x = jnp.fft.ifft(u_hat, axis=1)
    if not jnp.iscomplexobj(rhs):
        u_x = u_x.real
    u = _idct1_even(u_x)
    if nan_guard:
        u = jnp.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    return u


def inv_laplacian_cg(
    rhs: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    bc: BC2D,
    maxiter: int = 200,
    tol: float = 1e-10,
    atol: float = 0.0,
    preconditioner: str = "jacobi",
    k2_precond: jnp.ndarray | None = None,
    gauge_epsilon: float | None = None,
    nan_guard: bool = True,
) -> jnp.ndarray:
    """Solve ``∇² u = rhs`` with an SPD(-ish) FD Laplacian using (P)CG.

    Notes
    -----
    - We solve the symmetric system ``(-∇² + eps*P0) u = -rhs`` where ``P0`` projects
      onto the constant mode (mean). For periodic/Neumann problems this removes the
      singular nullspace and selects the zero-mean gauge.
    - ``preconditioner="jacobi"`` applies a simple diagonal preconditioner.
    - This routine is end-to-end differentiable through JAX's ``cg`` implementation.
    """

    nx, ny = rhs.shape
    diag = 2.0 / dx**2 + 2.0 / dy**2

    if gauge_epsilon is None:
        # Gauge-lifting term scale: small compared to the FD Laplacian diagonal.
        gauge_epsilon = 1e-12 * float(diag)

    def _spectral_M(*, shape: tuple[int, int], dx_eff: float, dy_eff: float):
        """FFT-based circulant preconditioner for the FD Laplacian.

        This approximates the inverse of ``(-∇² + eps*P0)`` on a periodic domain of the same
        shape. It is exact for the periodic gauge-fixed case and often reduces CG iterations
        substantially for Dirichlet/Neumann problems.
        """

        if k2_precond is None:
            npx, npy = shape
            kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(npx, d=dx_eff)
            ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(npy, d=dy_eff)
            kx, ky = jnp.meshgrid(kx_1d, ky_1d, indexing="ij")
            k2 = kx**2 + ky**2
        else:
            k2 = k2_precond
        denom = jnp.where(k2 > 0.0, k2, float(gauge_epsilon))

        def M(v_flat):
            v = v_flat.reshape(shape)
            v_hat = jnp.fft.fft2(v)
            u_hat = v_hat / denom
            u = jnp.fft.ifft2(u_hat)
            if not jnp.iscomplexobj(v):
                u = u.real
            return u.reshape((-1,))

        return M

    def make_M(*, size: int, shape: tuple[int, int], dx_eff: float, dy_eff: float):
        if preconditioner == "jacobi":
            inv_diag = 1.0 / jnp.asarray(diag, dtype=rhs.dtype)

            def M(v_flat):
                _ = (size, shape, dx_eff, dy_eff)
                return inv_diag * v_flat

            return M
        if preconditioner == "spectral":
            _ = (size,)
            return _spectral_M(shape=shape, dx_eff=dx_eff, dy_eff=dy_eff)
        if preconditioner in ("none", "", None):
            return None
        raise ValueError(f"Unknown preconditioner: {preconditioner}")

    if bc.kind_x == 0 and bc.kind_y == 0:
        # Periodic: solve full system (nullspace fixed by zero-mean gauge).
        rhs0 = rhs - jnp.mean(rhs)

        def mv(v_flat):
            v = v_flat.reshape((nx, ny))
            # SPD lift: add eps * mean(v) to eliminate the constant nullspace.
            out = -laplacian(v, dx, dy, bc) + float(gauge_epsilon) * jnp.mean(v)
            return out.reshape((-1,))

        b = (-rhs0).reshape((-1,))
        x0 = jnp.zeros_like(b)
        M = make_M(size=b.size, shape=(nx, ny), dx_eff=dx, dy_eff=dy)
        x, _ = jax.scipy.sparse.linalg.cg(mv, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)
        if nan_guard:
            x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        u = x.reshape((nx, ny))
        return u - jnp.mean(u)

    if (bc.kind_x == 0 and bc.kind_y != 0) or (bc.kind_x != 0 and bc.kind_y == 0):
        # Mixed periodic/non-periodic: apply gauge-fixing on the full grid.
        rhs0 = rhs - jnp.mean(rhs)

        def mv(v_flat):
            v = v_flat.reshape((nx, ny))
            out = -laplacian(v, dx, dy, bc) + float(gauge_epsilon) * jnp.mean(v)
            return out.reshape((-1,))

        b = (-rhs0).reshape((-1,))
        x0 = jnp.zeros_like(b)
        M = make_M(size=b.size, shape=(nx, ny), dx_eff=dx, dy_eff=dy)
        x, _ = jax.scipy.sparse.linalg.cg(mv, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)
        if nan_guard:
            x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        u = x.reshape((nx, ny))
        return u - jnp.mean(u)

    if bc.kind_x == 1 and bc.kind_y == 1:
        # Dirichlet: solve for interior unknowns with boundary fixed.
        b_int = (-rhs[1:-1, 1:-1]).reshape((-1,))

        def mv(v_flat):
            v = v_flat.reshape((nx - 2, ny - 2))
            # Set constant Dirichlet values on each pair of boundaries. Corners are set to the
            # average value to avoid an arbitrary overwrite if x_value != y_value.
            u = jnp.zeros((nx, ny), dtype=rhs.dtype)
            u = u.at[0, :].set(float(bc.x_value))
            u = u.at[-1, :].set(float(bc.x_value))
            u = u.at[:, 0].set(float(bc.y_value))
            u = u.at[:, -1].set(float(bc.y_value))
            corner = 0.5 * (float(bc.x_value) + float(bc.y_value))
            u = u.at[0, 0].set(corner)
            u = u.at[0, -1].set(corner)
            u = u.at[-1, 0].set(corner)
            u = u.at[-1, -1].set(corner)
            u = u.at[1:-1, 1:-1].set(v)
            Lu = (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 + (
                u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]
            ) / dy**2
            return (-Lu).reshape((-1,))

        x0 = jnp.zeros_like(b_int)
        M = make_M(size=b_int.size, shape=(nx - 2, ny - 2), dx_eff=dx, dy_eff=dy)
        x, _ = jax.scipy.sparse.linalg.cg(
            mv, b_int, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M
        )
        if nan_guard:
            x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        u = jnp.zeros((nx, ny), dtype=rhs.dtype)
        u = u.at[0, :].set(float(bc.x_value))
        u = u.at[-1, :].set(float(bc.x_value))
        u = u.at[:, 0].set(float(bc.y_value))
        u = u.at[:, -1].set(float(bc.y_value))
        corner = 0.5 * (float(bc.x_value) + float(bc.y_value))
        u = u.at[0, 0].set(corner)
        u = u.at[0, -1].set(corner)
        u = u.at[-1, 0].set(corner)
        u = u.at[-1, -1].set(corner)
        u = u.at[1:-1, 1:-1].set(x.reshape((nx - 2, ny - 2)))
        return u

    if bc.kind_x == 2 and bc.kind_y == 2:
        # Neumann: project rhs to the range of the Laplacian and solve full system,
        # choosing the zero-mean representative for the solution.
        rhs0 = rhs - jnp.mean(rhs)

        # Build a particular solution for constant boundary gradients.
        x = jnp.linspace(0.0, dx * (nx - 1), nx)[:, None]
        y = jnp.linspace(0.0, dy * (ny - 1), ny)[None, :]
        u_bc = bc.x_grad * x + bc.y_grad * y
        rhs_eff = rhs0 - _laplacian_homogeneous(u_bc, dx, dy, bc)

        def mv(v_flat):
            v = v_flat.reshape((nx, ny))
            out = -_laplacian_homogeneous(v, dx, dy, bc) + float(gauge_epsilon) * jnp.mean(v)
            return out.reshape((-1,))

        b = (-rhs_eff).reshape((-1,))
        x0 = jnp.zeros_like(b)
        M = make_M(size=b.size, shape=(nx, ny), dx_eff=dx, dy_eff=dy)
        x, _ = jax.scipy.sparse.linalg.cg(mv, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)
        if nan_guard:
            x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        u = x.reshape((nx, ny)) + u_bc
        return u - jnp.mean(u)

    raise ValueError(
        "inv_laplacian_cg supports periodic, dirichlet/dirichlet, or neumann/neumann BCs."
    )


def inv_div_n_grad_cg(
    rhs: jnp.ndarray,
    *,
    n_coeff: jnp.ndarray,
    dx: float,
    dy: float,
    bc: BC2D,
    maxiter: int = 200,
    tol: float = 1e-10,
    atol: float = 0.0,
    preconditioner: str = "jacobi",
    preconditioner_shift: float = 1e-12,
    gauge_epsilon: float | None = None,
    nan_guard: bool = True,
    n_floor: float = 1e-12,
) -> jnp.ndarray:
    """Solve ``-∇·(n ∇u) = rhs`` with variable coefficient n using (P)CG.

    This is the non-Boussinesq polarization solve needed in 2D DRB models.
    """

    nx, ny = rhs.shape

    n_eff = jnp.asarray(n_coeff)
    n_eff = jnp.maximum(n_eff, jnp.asarray(float(n_floor), dtype=rhs.dtype))

    if gauge_epsilon is None:
        gauge_epsilon = 1e-12

    def diag_from_coeff(nc: jnp.ndarray) -> jnp.ndarray:
        npx = _pad_coeff_x(nc, bc)
        npy = _pad_coeff_y(nc, bc)
        n_xp = 0.5 * (npx[1:-1, :] + npx[2:, :])
        n_xm = 0.5 * (npx[1:-1, :] + npx[:-2, :])
        n_yp = 0.5 * (npy[:, 1:-1] + npy[:, 2:])
        n_ym = 0.5 * (npy[:, 1:-1] + npy[:, :-2])
        return (n_xp + n_xm) / dx**2 + (n_yp + n_ym) / dy**2

    diag = diag_from_coeff(n_eff)

    def _spectral_M(*, shape: tuple[int, int], nbar: float, dx_eff: float, dy_eff: float):
        npx, npy = shape
        kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(npx, d=dx_eff)
        ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(npy, d=dy_eff)
        kx, ky = jnp.meshgrid(kx_1d, ky_1d, indexing="ij")
        k2 = kx**2 + ky**2
        denom = jnp.where(k2 > 0.0, nbar * k2, float(gauge_epsilon))

        def M(v_flat):
            v = v_flat.reshape(shape)
            v_hat = jnp.fft.fft2(v)
            u_hat = v_hat / denom
            u = jnp.fft.ifft2(u_hat)
            if not jnp.iscomplexobj(v):
                u = u.real
            return u.reshape((-1,))

        return M

    def make_M(*, shape: tuple[int, int], nbar: float):
        if preconditioner == "jacobi":
            inv_diag = 1.0 / jnp.maximum(diag, 1e-14)

            def M(v_flat):
                return inv_diag.reshape((-1,)) * v_flat

            return M
        if preconditioner == "spectral":
            return _spectral_M(shape=shape, nbar=nbar, dx_eff=dx, dy_eff=dy)
        if preconditioner == "spectral_jacobi":
            # Symmetric SPD preconditioner:
            #     M ≈ D^{-1/2} (nbar * -Δ)^{-1} D^{-1/2}
            # where D is the diagonal of the variable-coefficient operator.
            inv_sqrt_diag = 1.0 / jnp.sqrt(jnp.maximum(diag + float(preconditioner_shift), 1e-14))
            spectral = _spectral_M(shape=shape, nbar=nbar, dx_eff=dx, dy_eff=dy)

            def M(v_flat):
                v = inv_sqrt_diag.reshape((-1,)) * v_flat
                u = spectral(v)
                return inv_sqrt_diag.reshape((-1,)) * u

            return M
        if preconditioner in ("none", "", None):
            return None
        raise ValueError(f"Unknown preconditioner: {preconditioner}")

    if bc.kind_x == 0 and bc.kind_y == 0:
        rhs0 = rhs - jnp.mean(rhs)

        def mv(v_flat):
            v = v_flat.reshape((nx, ny))
            out = -div_n_grad(v, n_eff, dx, dy, bc) + float(gauge_epsilon) * jnp.mean(v)
            return out.reshape((-1,))

        b = rhs0.reshape((-1,))
        x0 = jnp.zeros_like(b)
        nbar = jnp.mean(n_eff)
        M = make_M(shape=(nx, ny), nbar=nbar)
        x, _ = jax.scipy.sparse.linalg.cg(mv, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)
        if nan_guard:
            x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        u = x.reshape((nx, ny))
        return u - jnp.mean(u)

    if bc.kind_x == 1 and bc.kind_y == 1:
        b_int = (-rhs[1:-1, 1:-1]).reshape((-1,))

        def mv(v_flat):
            v = v_flat.reshape((nx - 2, ny - 2))
            u = jnp.zeros((nx, ny), dtype=rhs.dtype)
            u = u.at[0, :].set(float(bc.x_value))
            u = u.at[-1, :].set(float(bc.x_value))
            u = u.at[:, 0].set(float(bc.y_value))
            u = u.at[:, -1].set(float(bc.y_value))
            corner = 0.5 * (float(bc.x_value) + float(bc.y_value))
            u = u.at[0, 0].set(corner)
            u = u.at[0, -1].set(corner)
            u = u.at[-1, 0].set(corner)
            u = u.at[-1, -1].set(corner)
            u = u.at[1:-1, 1:-1].set(v)
            Lu = div_n_grad(u, n_eff, dx, dy, bc)
            return (-Lu[1:-1, 1:-1]).reshape((-1,))

        x0 = jnp.zeros_like(b_int)
        nbar = jnp.mean(n_eff)
        M = make_M(shape=(nx - 2, ny - 2), nbar=nbar)
        x, _ = jax.scipy.sparse.linalg.cg(
            mv, b_int, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M
        )
        if nan_guard:
            x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        u = jnp.zeros((nx, ny), dtype=rhs.dtype)
        u = u.at[0, :].set(float(bc.x_value))
        u = u.at[-1, :].set(float(bc.x_value))
        u = u.at[:, 0].set(float(bc.y_value))
        u = u.at[:, -1].set(float(bc.y_value))
        corner = 0.5 * (float(bc.x_value) + float(bc.y_value))
        u = u.at[0, 0].set(corner)
        u = u.at[0, -1].set(corner)
        u = u.at[-1, 0].set(corner)
        u = u.at[-1, -1].set(corner)
        u = u.at[1:-1, 1:-1].set(x.reshape((nx - 2, ny - 2)))
        return u

    if bc.kind_x == 2 and bc.kind_y == 2:
        if float(bc.x_grad) != 0.0 or float(bc.y_grad) != 0.0:
            raise ValueError("Non-Boussinesq solve only supports homogeneous Neumann BCs.")
        rhs0 = rhs - jnp.mean(rhs)

        def mv(v_flat):
            v = v_flat.reshape((nx, ny))
            out = -div_n_grad(v, n_eff, dx, dy, bc) + float(gauge_epsilon) * jnp.mean(v)
            return out.reshape((-1,))

        b = (-rhs0).reshape((-1,))
        x0 = jnp.zeros_like(b)
        nbar = jnp.mean(n_eff)
        M = make_M(shape=(nx, ny), nbar=nbar)
        x, _ = jax.scipy.sparse.linalg.cg(mv, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)
        if nan_guard:
            x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        u = x.reshape((nx, ny))
        return u - jnp.mean(u)

    raise ValueError(
        "inv_div_n_grad_cg supports periodic, dirichlet/dirichlet, or homogeneous neumann BCs."
    )
