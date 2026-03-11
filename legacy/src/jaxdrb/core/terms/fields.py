from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.operators.fd2d import (
    div_n_grad,
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    inv_laplacian_fd_fft,
    inv_laplacian_mixed_fft,
    metric_laplacian,
    inv_metric_laplacian_cg,
)
from jaxdrb.operators.spectral2d import inv_laplacian as inv_laplacian_spec

from .ops import ddx, ddy, is_periodic_bc, laplacian


def _full_grid_of(geom: GeometryAdapter):
    return getattr(geom, "grid", None)


def _perp_grid_of(geom: GeometryAdapter):
    grid = _full_grid_of(geom)
    if grid is None:
        return None
    perp = getattr(grid, "perp", None)
    if perp is not None:
        return perp
    if all(hasattr(grid, name) for name in ("nx", "ny", "dx", "dy")):
        return grid
    return None


def _broadcast_to_shape(arr: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
    if arr.shape == shape:
        return arr
    if arr.ndim == 1:
        if len(shape) == 3 and arr.shape[0] == shape[0]:
            arr = arr[:, None, None]
        elif len(shape) == 2 and arr.shape[0] == shape[0]:
            arr = arr[:, None]
        elif len(shape) == 2 and arr.shape[0] == shape[1]:
            arr = arr[None, :]
    elif arr.ndim == 2 and len(shape) == 3:
        if arr.shape == shape[1:]:
            arr = arr[None, :, :]
        elif arr.shape == shape[:2]:
            arr = arr[:, :, None]
        elif arr.shape == (shape[0], shape[2]):
            arr = arr[:, None, :]
    return jnp.broadcast_to(arr, shape)


def _metric_to_3d(arr: jnp.ndarray, *, nz: int, nx: int, ny: int, name: str) -> jnp.ndarray:
    """Broadcast metric/Jacobian data to (nz, nx, ny)."""

    arr = jnp.asarray(arr)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (nz, nx, ny))
    if arr.ndim == 1:
        if arr.shape[0] == nz:
            return jnp.broadcast_to(arr[:, None, None], (nz, nx, ny))
        if arr.shape[0] == nx:
            return jnp.broadcast_to(arr[None, :, None], (nz, nx, ny))
        if arr.shape[0] == ny:
            return jnp.broadcast_to(arr[None, None, :], (nz, nx, ny))
        raise ValueError(f"{name} 1D shape {arr.shape} is incompatible with (nz, nx, ny).")
    if arr.ndim == 2:
        if arr.shape == (nx, ny):
            return jnp.broadcast_to(arr[None, :, :], (nz, nx, ny))
        if arr.shape == (nz, nx):
            return jnp.broadcast_to(arr[:, :, None], (nz, nx, ny))
        if arr.shape == (nx, nz):
            return jnp.broadcast_to(jnp.swapaxes(arr, 0, 1)[:, :, None], (nz, nx, ny))
        if arr.shape == (nz, ny):
            return jnp.broadcast_to(arr[:, None, :], (nz, nx, ny))
        if arr.shape == (ny, nx):
            return jnp.broadcast_to(arr.T[None, :, :], (nz, nx, ny))
        raise ValueError(f"{name} 2D shape {arr.shape} is incompatible with (nz, nx, ny).")
    if arr.ndim == 3 and arr.shape == (nz, nx, ny):
        return arr
    raise ValueError(f"{name} has unsupported shape {arr.shape}; expected (nz, nx, ny).")


def _metric_to_xy(arr: jnp.ndarray, *, nx: int, nz: int, name: str) -> jnp.ndarray:
    """Coerce metric arrays to (nx, nz) for LaplaceXY (x-parallel) solves."""

    arr = jnp.asarray(arr)
    if arr.ndim == 1:
        if arr.shape[0] == nx:
            return jnp.broadcast_to(arr[:, None], (nx, nz))
        if arr.shape[0] == nz:
            return jnp.broadcast_to(arr[None, :], (nx, nz))
    if arr.ndim == 2:
        if arr.shape == (nx, nz):
            return arr
        if arr.shape == (nz, nx):
            return arr.T
    if arr.ndim == 3:
        if arr.shape[0] == nz and arr.shape[1] == nx:
            return arr.mean(axis=2).T
        if arr.shape[0] == nx and arr.shape[1] == nz:
            return arr.mean(axis=2)
    raise ValueError(f"{name} has unsupported shape {arr.shape}; expected (nx, nz).")


def _bc_with_dirichlet_profile(bc: BC2D, ref: jnp.ndarray | None) -> BC2D:
    """Create a BC object with per-side Dirichlet values from ``ref`` when available."""

    if ref is None or ref.ndim != 2:
        return bc
    x_value = bc.x_value
    y_value = bc.y_value
    if bc.kind_x == 1:
        x_value = (ref[0, :], ref[-1, :])
    if bc.kind_y == 1:
        y_value = (ref[:, 0], ref[:, -1])
    return BC2D(
        kind_x=bc.kind_x,
        kind_y=bc.kind_y,
        x_value=x_value,
        y_value=y_value,
        x_grad=bc.x_grad,
        y_grad=bc.y_grad,
    )


def _poisson_bc_eval(
    params: DRBSystemParams,
    bc: BC2D,
    *,
    ref: jnp.ndarray | None,
) -> BC2D:
    bc_eff = bc
    # Hermes INVERT_SET uses field-derived Dirichlet values at radial guards.
    # Only switch to Dirichlet when a reference field/guess is available.
    # Without a reference, keep the configured BC (typically Neumann) instead
    # of forcing zero-Dirichlet boundaries.
    if bool(getattr(params, "poisson_invert_set", False)) and bc.kind_x != 0 and ref is not None:
        bc_eff = BC2D(
            kind_x=1,
            kind_y=bc.kind_y,
            x_value=bc.x_value,
            y_value=bc.y_value,
            x_grad=bc.x_grad,
            y_grad=bc.y_grad,
        )
    return _bc_with_dirichlet_profile(bc_eff, ref)


def _metric_gyy_effective_3d(
    geom: GeometryAdapter,
    *,
    nz: int,
    nx: int,
    ny: int,
    fallback_gyy: jnp.ndarray,
) -> jnp.ndarray:
    """Perpendicular-binormal metric for field-aligned Laplace operators.

    For shifted field-aligned systems, BOUT++/Hermes perpendicular operators use
    a reduced-binormal metric coefficient based on g23/g_23/g_22. When these
    coefficients are unavailable we fall back to the provided gyy metric.
    """

    g23 = getattr(geom, "g23", None)
    g_22 = getattr(geom, "g_22", None)
    g_23 = getattr(geom, "g_23", None)
    if g23 is None or g_22 is None or g_23 is None:
        return _metric_to_3d(fallback_gyy, nz=nz, nx=nx, ny=ny, name="gyy")

    g23_3d = _metric_to_3d(jnp.asarray(g23), nz=nz, nx=nx, ny=ny, name="g23")
    g_23_3d = _metric_to_3d(jnp.asarray(g_23), nz=nz, nx=nx, ny=ny, name="g_23")
    g_22_3d = _metric_to_3d(jnp.asarray(g_22), nz=nz, nx=nx, ny=ny, name="g_22")
    return -(g23_3d * g_23_3d) / jnp.maximum(g_22_3d, 1e-30)


def _metric_solve_xy(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    rhs_xy: jnp.ndarray,
    coeff_xy: jnp.ndarray,
    bc_xy: BC2D,
    *,
    x0: jnp.ndarray | None = None,
    return_iters: bool = False,
) -> jnp.ndarray:
    """Solve LaplaceXY-style div(coeff * G_xy ∇phi) = rhs for k_y=0 mode."""

    grid = _full_grid_of(geom)
    if grid is None or not all(hasattr(grid, name) for name in ("perp", "z", "dz")):
        return geom.inv_laplacian(rhs_xy, x0=x0)

    gxx = getattr(geom, "gxx", None)
    g23 = getattr(geom, "g23", None)
    g_23 = getattr(geom, "g_23", None)
    g_22 = getattr(geom, "g_22", None)
    jac = getattr(geom, "jacobian", None)
    if gxx is None or g23 is None or g_23 is None or g_22 is None:
        return geom.inv_laplacian(rhs_xy, x0=x0)

    nz = int(grid.z.size)
    nx = int(grid.perp.nx)
    gxx_xy = _metric_to_xy(jnp.asarray(gxx), nx=nx, nz=nz, name="gxx_xy")
    g23_xy = _metric_to_xy(jnp.asarray(g23), nx=nx, nz=nz, name="g23_xy")
    g_23_xy = _metric_to_xy(jnp.asarray(g_23), nx=nx, nz=nz, name="g_23_xy")
    g_22_xy = _metric_to_xy(jnp.asarray(g_22), nx=nx, nz=nz, name="g_22_xy")
    jac_xy = None
    if jac is not None:
        jac_xy = _metric_to_xy(jnp.asarray(jac), nx=nx, nz=nz, name="J_xy")

    coeff_y = -(g23_xy * g_23_xy) / jnp.maximum(g_22_xy, 1e-30)
    gxx_eff = gxx_xy * coeff_xy
    gyy_eff = coeff_y * coeff_xy

    precond = params.poisson_preconditioner
    if precond == "auto":
        precond = "jacobi"
    return inv_metric_laplacian_cg(
        rhs_xy,
        gxx=gxx_eff,
        gxy=jnp.zeros_like(gxx_eff),
        gyy=gyy_eff,
        jacobian=jac_xy,
        dx=grid.perp.dx,
        dy=grid.dz,
        bc=bc_xy,
        maxiter=int(params.poisson_maxiter),
        tol=float(params.poisson_tol),
        atol=float(params.poisson_cg_atol),
        preconditioner=str(precond),
        k2_precond=None,
        gauge_epsilon=params.poisson_gauge_epsilon,
        preconditioner_fn=None,
        x0=x0,
        return_iters=return_iters,
    )


def _metric_div_xy(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    field_xy: jnp.ndarray,
    coeff_xy: jnp.ndarray,
    bc_xy: BC2D,
) -> jnp.ndarray:
    """Forward LaplaceXY operator: div(coeff * G_xy ∇field)."""

    grid = _full_grid_of(geom)
    if grid is None or not all(hasattr(grid, name) for name in ("perp", "z", "dz")):
        return geom.laplacian(field_xy)

    gxx = getattr(geom, "gxx", None)
    g23 = getattr(geom, "g23", None)
    g_23 = getattr(geom, "g_23", None)
    g_22 = getattr(geom, "g_22", None)
    jac = getattr(geom, "jacobian", None)
    if gxx is None or g23 is None or g_23 is None or g_22 is None:
        return geom.laplacian(field_xy)

    nz = int(grid.z.size)
    nx = int(grid.perp.nx)
    gxx_xy = _metric_to_xy(jnp.asarray(gxx), nx=nx, nz=nz, name="gxx_xy")
    g23_xy = _metric_to_xy(jnp.asarray(g23), nx=nx, nz=nz, name="g23_xy")
    g_23_xy = _metric_to_xy(jnp.asarray(g_23), nx=nx, nz=nz, name="g_23_xy")
    g_22_xy = _metric_to_xy(jnp.asarray(g_22), nx=nx, nz=nz, name="g_22_xy")
    jac_xy = None
    if jac is not None:
        jac_xy = _metric_to_xy(jnp.asarray(jac), nx=nx, nz=nz, name="J_xy")

    coeff_y = -(g23_xy * g_23_xy) / jnp.maximum(g_22_xy, 1e-30)
    gxx_eff = gxx_xy * coeff_xy
    gyy_eff = coeff_y * coeff_xy
    return metric_laplacian(
        field_xy,
        gxx_eff,
        jnp.zeros_like(gxx_eff),
        gyy_eff,
        grid.perp.dx,
        grid.dz,
        bc_xy,
        jacobian=jac_xy,
    )


def _poisson_coeff_b(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    n_phys: jnp.ndarray,
) -> jnp.ndarray:
    """Return coeff = n_eff / B^2 for B-weighted vorticity definitions."""

    n_eff = _n_eff(params, n_phys)
    Abar = float(getattr(params, "average_atomic_mass", 1.0))
    B = getattr(geom, "B", None)
    if B is None:
        invB2 = jnp.asarray(1.0)
    else:
        invB2 = 1.0 / jnp.maximum(jnp.asarray(B), 1e-12) ** 2
    invB2 = _broadcast_to_shape(jnp.asarray(invB2), n_phys.shape)
    coeff = n_eff * invB2
    if str(getattr(params, "poisson_b_weighted_mode", "scaled")).lower() == "hermes":
        coeff = coeff * Abar
    return jnp.maximum(coeff, jnp.asarray(float(params.n0_min), dtype=coeff.dtype) * invB2)


def _metric_div_coeff(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    field: jnp.ndarray,
    coeff: jnp.ndarray,
    bc_phi: BC2D,
) -> jnp.ndarray:
    """Compute div(coeff * G ∇field) using metric coefficients when available."""

    grid = _perp_grid_of(geom)
    if grid is None:
        return geom.laplacian(field)

    use_shift = bool(
        field.ndim == 3
        and str(getattr(params, "parallel_transform", "none")).lower() == "shifted"
        and hasattr(geom, "shift_idx")
        and getattr(geom, "shift_idx") is not None
        and hasattr(geom, "to_field_aligned")
        and hasattr(geom, "from_field_aligned")
    )
    field_eval = geom.to_field_aligned(field) if use_shift else field
    coeff_eval = geom.to_field_aligned(coeff) if use_shift else coeff

    if not getattr(params, "poisson_metric_on", False):
        if field_eval.ndim == 2:
            out = div_n_grad(field_eval, coeff_eval, dx=grid.dx, dy=grid.dy, bc=bc_phi)
        else:
            out = jax.vmap(lambda f, c: div_n_grad(f, c, dx=grid.dx, dy=grid.dy, bc=bc_phi))(
                field_eval, coeff_eval
            )
        return geom.from_field_aligned(out) if use_shift else out

    if not (hasattr(geom, "metric_available") and geom.metric_available()):
        if field_eval.ndim == 2:
            out = div_n_grad(field_eval, coeff_eval, dx=grid.dx, dy=grid.dy, bc=bc_phi)
        else:
            out = jax.vmap(lambda f, c: div_n_grad(f, c, dx=grid.dx, dy=grid.dy, bc=bc_phi))(
                field_eval, coeff_eval
            )
        return geom.from_field_aligned(out) if use_shift else out

    gxx = jnp.asarray(getattr(geom, "gxx"))
    gxy = jnp.asarray(getattr(geom, "gxy"))
    gyy = jnp.asarray(getattr(geom, "gyy"))
    jac = None if getattr(geom, "jacobian", None) is None else jnp.asarray(geom.jacobian)
    nx = grid.nx
    ny = grid.ny

    def _plane(field_plane, coeff_plane, gxx_plane, gxy_plane, gyy_plane, j_plane):
        # Forward vorticity operator should use the configured Poisson BC policy.
        # INVERT_SET-style boundary values are applied in the inverse solve path
        # (via the `guess` field), not by forcing self-referential Dirichlet values
        # from the field being differentiated.
        bc_plane = _poisson_bc_eval(params, bc_phi, ref=None)
        gxx_plane = gxx_plane * coeff_plane
        gxy_plane = gxy_plane * coeff_plane
        gyy_plane = gyy_plane * coeff_plane
        return metric_laplacian(
            field_plane,
            gxx_plane,
            gxy_plane,
            gyy_plane,
            grid.dx,
            grid.dy,
            bc_plane,
            jacobian=j_plane,
        )

    if field_eval.ndim == 2:
        gxx2 = _metric_to_3d(gxx, nz=1, nx=nx, ny=ny, name="gxx")[0]
        gxy2 = _metric_to_3d(gxy, nz=1, nx=nx, ny=ny, name="gxy")[0]
        gyy2 = _metric_to_3d(gyy, nz=1, nx=nx, ny=ny, name="gyy")[0]
        j2 = None if jac is None else _metric_to_3d(jac, nz=1, nx=nx, ny=ny, name="J")[0]
        out = _plane(field_eval, coeff_eval, gxx2, gxy2, gyy2, j2)
        return geom.from_field_aligned(out) if use_shift else out

    nz = int(field_eval.shape[0])
    gxx3 = _metric_to_3d(gxx, nz=nz, nx=nx, ny=ny, name="gxx")
    gxy3 = _metric_to_3d(gxy, nz=nz, nx=nx, ny=ny, name="gxy")
    # Full 3D metric Poisson follows the X-Z operator (Hermes/BOUT Laplacian),
    # so use the provided gyy coefficient directly. The g23-based reduced
    # coefficient is only used in the dedicated LaplaceXY split path.
    gyy3 = _metric_to_3d(gyy, nz=nz, nx=nx, ny=ny, name="gyy")
    if jac is None:
        out = jax.vmap(lambda f, c, gx, gxy_loc, gy: _plane(f, c, gx, gxy_loc, gy, None))(
            field_eval, coeff_eval, gxx3, gxy3, gyy3
        )
        return geom.from_field_aligned(out) if use_shift else out
    jac3 = _metric_to_3d(jac, nz=nz, nx=nx, ny=ny, name="J")
    out = jax.vmap(_plane)(field_eval, coeff_eval, gxx3, gxy3, gyy3, jac3)
    return geom.from_field_aligned(out) if use_shift else out


def _metric_solve_coeff(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    rhs: jnp.ndarray,
    coeff: jnp.ndarray,
    bc_phi: BC2D,
    *,
    x0: jnp.ndarray | None = None,
    return_iters: bool = False,
) -> jnp.ndarray:
    """Solve div(coeff * G ∇phi) = rhs using metric coefficients."""

    grid = _perp_grid_of(geom)
    if grid is None:
        return geom.inv_laplacian(rhs, x0=x0)

    use_shift = bool(
        rhs.ndim == 3
        and str(getattr(params, "parallel_transform", "none")).lower() == "shifted"
        and hasattr(geom, "shift_idx")
        and getattr(geom, "shift_idx") is not None
        and hasattr(geom, "to_field_aligned")
        and hasattr(geom, "from_field_aligned")
    )
    rhs_eval = geom.to_field_aligned(rhs) if use_shift else rhs
    coeff_eval = geom.to_field_aligned(coeff) if use_shift else coeff
    x0_eval = geom.to_field_aligned(x0) if (use_shift and x0 is not None) else x0

    gxx = jnp.asarray(getattr(geom, "gxx"))
    gxy = jnp.asarray(getattr(geom, "gxy"))
    gyy = jnp.asarray(getattr(geom, "gyy"))
    jac = None if getattr(geom, "jacobian", None) is None else jnp.asarray(geom.jacobian)
    nx = grid.nx
    ny = grid.ny

    def _plane(rhs_plane, coeff_plane, gxx_plane, gxy_plane, gyy_plane, j_plane, guess=None):
        bc_plane = _poisson_bc_eval(params, bc_phi, ref=guess)
        gxx_plane = gxx_plane * coeff_plane
        gxy_plane = gxy_plane * coeff_plane
        gyy_plane = gyy_plane * coeff_plane
        precond = params.poisson_preconditioner
        if precond == "auto":
            precond = "spectral"
        k2_precond = None
        ops = getattr(geom, "perp_ops", None)
        if ops is not None and str(precond) == "spectral":
            k2_precond = ops.k2
        return inv_metric_laplacian_cg(
            rhs_plane,
            gxx=gxx_plane,
            gxy=gxy_plane,
            gyy=gyy_plane,
            jacobian=j_plane,
            dx=grid.dx,
            dy=grid.dy,
            bc=bc_plane,
            maxiter=int(params.poisson_maxiter),
            tol=float(params.poisson_tol),
            atol=float(params.poisson_cg_atol),
            preconditioner=str(precond),
            k2_precond=k2_precond,
            gauge_epsilon=params.poisson_gauge_epsilon,
            preconditioner_fn=getattr(geom, "poisson_preconditioner_fn", None),
            x0=guess,
            return_iters=return_iters,
        )

    if rhs_eval.ndim == 2:
        gxx2 = _metric_to_3d(gxx, nz=1, nx=nx, ny=ny, name="gxx")[0]
        gxy2 = _metric_to_3d(gxy, nz=1, nx=nx, ny=ny, name="gxy")[0]
        gyy2 = _metric_to_3d(gyy, nz=1, nx=nx, ny=ny, name="gyy")[0]
        j2 = None if jac is None else _metric_to_3d(jac, nz=1, nx=nx, ny=ny, name="J")[0]
        return _plane(rhs_eval, coeff_eval, gxx2, gxy2, gyy2, j2, x0_eval)

    nz = int(rhs_eval.shape[0])
    gxx3 = _metric_to_3d(gxx, nz=nz, nx=nx, ny=ny, name="gxx")
    gxy3 = _metric_to_3d(gxy, nz=nz, nx=nx, ny=ny, name="gxy")
    # Full 3D metric Poisson follows the X-Z operator (Hermes/BOUT Laplacian),
    # so use the provided gyy coefficient directly. The g23-based reduced
    # coefficient is only used in the dedicated LaplaceXY split path.
    gyy3 = _metric_to_3d(gyy, nz=nz, nx=nx, ny=ny, name="gyy")
    out = None
    if jac is None:
        if x0_eval is None:
            out = jax.vmap(lambda r, c, gx, gxy_loc, gy: _plane(r, c, gx, gxy_loc, gy, None, None))(
                rhs_eval, coeff_eval, gxx3, gxy3, gyy3
            )
        else:
            out = jax.vmap(
                lambda r, c, gx, gxy_loc, gy, guess: _plane(r, c, gx, gxy_loc, gy, None, guess)
            )(rhs_eval, coeff_eval, gxx3, gxy3, gyy3, x0_eval)
    else:
        j3 = _metric_to_3d(jac, nz=nz, nx=nx, ny=ny, name="J")
        if x0_eval is None:
            out = jax.vmap(_plane)(rhs_eval, coeff_eval, gxx3, gxy3, gyy3, j3)
        else:
            out = jax.vmap(_plane)(rhs_eval, coeff_eval, gxx3, gxy3, gyy3, j3, x0_eval)

    if use_shift and return_iters:
        phi_eval, iters = out
        return geom.from_field_aligned(phi_eval), iters
    if use_shift:
        return geom.from_field_aligned(out)
    return out


def _diamagnetic_polarisation_term(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    n_phys: jnp.ndarray,
    Ti: jnp.ndarray | None,
    bc_phi: BC2D,
) -> jnp.ndarray:
    if not bool(getattr(params, "diamagnetic_polarisation_on", False)):
        return jnp.zeros_like(n_phys)

    tau_i = float(getattr(params, "tau_i", 0.0))
    Ti_eff = jnp.zeros_like(n_phys) if Ti is None else Ti
    p_i = tau_i * (n_phys * Ti_eff)

    B = getattr(geom, "B", None)
    if B is None:
        invB2 = jnp.asarray(1.0)
    else:
        invB2 = 1.0 / jnp.maximum(jnp.asarray(B), 1e-12) ** 2
    invB2 = _broadcast_to_shape(jnp.asarray(invB2), n_phys.shape)

    scale = float(getattr(params, "diamagnetic_polarisation_scale", 1.0))
    grid = _perp_grid_of(geom)

    if grid is None:
        if isinstance(invB2, jnp.ndarray):
            term = ddx(params, geom, invB2 * ddx(params, geom, p_i, bc_phi), bc_phi) + ddy(
                params, geom, invB2 * ddy(params, geom, p_i, bc_phi), bc_phi
            )
        else:
            term = geom.laplacian(p_i)
        return term * scale

    if p_i.ndim == 2:
        term = div_n_grad(p_i, invB2, dx=grid.dx, dy=grid.dy, bc=bc_phi)
        return term * scale

    term = jax.vmap(lambda p, coeff: div_n_grad(p, coeff, dx=grid.dx, dy=grid.dy, bc=bc_phi))(
        p_i, invB2
    )
    return term * scale


def phys_n(params: DRBSystemParams, n: jnp.ndarray) -> jnp.ndarray:
    if not params.log_n:
        return n
    clip = params.log_n_clip
    if clip is None:
        return jnp.exp(n)
    clip_val = float(clip)
    return jnp.exp(jnp.clip(n, a_min=-clip_val, a_max=clip_val))


def phys_Te(params: DRBSystemParams, Te: jnp.ndarray) -> jnp.ndarray:
    if not params.log_Te:
        Te_phys = Te
        if float(params.temperature_floor) > 0.0:
            floor_val = float(params.temperature_floor)
            Te_phys = 0.5 * (Te_phys + jnp.sqrt(Te_phys * Te_phys + floor_val * floor_val))
        return Te_phys
    clip = params.log_Te_clip
    if clip is None:
        Te_phys = jnp.exp(Te)
    clip_val = float(clip)
    Te_phys = jnp.exp(jnp.clip(Te, a_min=-clip_val, a_max=clip_val))
    if float(params.temperature_floor) > 0.0:
        floor_val = float(params.temperature_floor)
        Te_phys = 0.5 * (Te_phys + jnp.sqrt(Te_phys * Te_phys + floor_val * floor_val))
    return Te_phys


def log_rhs(
    params: DRBSystemParams,
    rhs: jnp.ndarray,
    phys: jnp.ndarray,
    floor: float,
    log_on: bool,
) -> jnp.ndarray:
    if not log_on:
        return rhs
    denom = jnp.maximum(phys, float(floor))
    return rhs / denom


def _n_eff(params: DRBSystemParams, n: jnp.ndarray) -> jnp.ndarray:
    n_eff = float(params.n0)
    if params.non_boussinesq_perturbed_density_on:
        n_eff = n_eff + jnp.real(jnp.asarray(n))
    n_eff = jnp.maximum(jnp.asarray(n_eff), float(params.n0_min))
    if params.n0_max is not None:
        n_eff = jnp.minimum(n_eff, float(params.n0_max))
    return n_eff


def _electron_pressure(
    params: DRBSystemParams,
    n_phys: jnp.ndarray,
    Te: jnp.ndarray,
) -> jnp.ndarray:
    """Return electron pressure used by Hermes-style polarization closures."""
    model = str(getattr(params, "electron_pressure_model", "nTe")).lower()
    if model == "te":
        return Te
    return n_phys * Te


def phi_from_omega(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    omega: jnp.ndarray,
    n_phys: jnp.ndarray,
    bc_phi: BC2D,
    Ti: jnp.ndarray | None = None,
    Te: jnp.ndarray | None = None,
    phi_guess: jnp.ndarray | None = None,
    return_iters: bool = False,
) -> jnp.ndarray:
    grid = _perp_grid_of(geom)
    full_grid = _full_grid_of(geom)
    scale = float(params.poisson_scale)
    omega = omega / scale if scale != 1.0 else omega
    if params.poisson_b_weighted:
        mode = str(getattr(params, "poisson_b_weighted_mode", "scaled")).lower()
        if mode == "hermes":
            Abar = float(getattr(params, "average_atomic_mass", 1.0))
            B = getattr(geom, "B", None)
            if B is None:
                invB2 = jnp.asarray(1.0)
            else:
                invB2 = 1.0 / jnp.maximum(jnp.asarray(B), 1e-12) ** 2
            coeff = _broadcast_to_shape(jnp.asarray(Abar) * invB2, n_phys.shape)
            rhs = omega
            if bool(getattr(params, "diamagnetic_polarisation_on", False)):
                Ti_eff = jnp.zeros_like(n_phys) if Ti is None else Ti
                Te_eff = jnp.zeros_like(n_phys) if Te is None else Te
                me_hat = float(getattr(params, "me_hat", 0.0))
                p_hat = Ti_eff * n_phys - me_hat * _electron_pressure(params, n_phys, Te_eff)
                rhs = rhs - _metric_div_coeff(params, geom, p_hat, coeff, bc_phi)
            if grid is None:
                phi = geom.inv_laplacian(rhs, x0=phi_guess)
                return (phi, jnp.asarray(0)) if return_iters else phi
            if (
                params.poisson_metric_on
                and hasattr(geom, "metric_available")
                and geom.metric_available()
            ):
                split_n0 = bool(getattr(params, "poisson_split_n0", False))
                if split_n0 and rhs.ndim == 3:
                    rhs0 = rhs.mean(axis=2)
                    rhs1 = rhs - rhs0[..., None]
                    coeff0 = coeff.mean(axis=2)
                    phi_guess0 = None
                    phi_guess1 = None
                    if phi_guess is not None:
                        phi_guess0 = phi_guess.mean(axis=2)
                        phi_guess1 = phi_guess - phi_guess0[..., None]
                    phi1 = _metric_solve_coeff(
                        params,
                        geom,
                        rhs1,
                        coeff,
                        bc_phi,
                        x0=phi_guess1,
                        return_iters=return_iters,
                    )
                    if return_iters:
                        phi1, iters1 = phi1
                    phi1 = phi1 - phi1.mean(axis=2, keepdims=True)
                    bc_xy = BC2D(
                        kind_x=bc_phi.kind_x,
                        kind_y=2 if getattr(full_grid, "open_field_line", False) else bc_phi.kind_y,
                        x_value=bc_phi.x_value,
                        y_value=0.0,
                        x_grad=bc_phi.x_grad,
                        y_grad=0.0,
                    )
                    phi0_xy = _metric_solve_xy(
                        params,
                        geom,
                        jnp.swapaxes(rhs0, 0, 1),
                        jnp.swapaxes(coeff0, 0, 1),
                        bc_xy,
                        x0=None if phi_guess0 is None else jnp.swapaxes(phi_guess0, 0, 1),
                        return_iters=return_iters,
                    )
                    if return_iters:
                        phi0_xy, iters0 = phi0_xy
                    phi0 = jnp.swapaxes(phi0_xy, 0, 1)
                    phi = phi0[..., None] + phi1
                    if return_iters:
                        iters = jnp.maximum(jnp.asarray(iters0), jnp.asarray(iters1))
                        return phi, iters
                    return phi

                return _metric_solve_coeff(
                    params,
                    geom,
                    rhs,
                    coeff,
                    bc_phi,
                    x0=phi_guess,
                    return_iters=return_iters,
                )
            precond = params.polarization_preconditioner
            if precond == "auto":
                precond = "spectral_jacobi"
            return inv_div_n_grad_cg(
                rhs,
                n_coeff=coeff,
                dx=grid.dx,
                dy=grid.dy,
                bc=bc_phi,
                maxiter=int(params.polarization_cg_maxiter),
                tol=float(params.polarization_cg_tol),
                atol=float(params.polarization_cg_atol),
                preconditioner=precond,
                preconditioner_shift=float(params.polarization_precond_shift),
                preconditioner_fn=getattr(geom, "polarization_preconditioner_fn", None),
                x0=phi_guess,
                return_iters=return_iters,
            )

        coeff = _poisson_coeff_b(params, geom, n_phys)
        rhs = omega * coeff if mode == "scaled" else omega
        if bool(getattr(params, "diamagnetic_polarisation_on", False)):
            tau_i = float(getattr(params, "tau_i", 0.0))
            Ti_eff = jnp.zeros_like(n_phys) if Ti is None else Ti
            p_i = tau_i * (n_phys * Ti_eff)
            rhs = rhs - _metric_div_coeff(params, geom, p_i, coeff, bc_phi)
        if grid is None:
            phi = geom.inv_laplacian(rhs, x0=phi_guess)
            return (phi, jnp.asarray(0)) if return_iters else phi
        if (
            params.poisson_metric_on
            and hasattr(geom, "metric_available")
            and geom.metric_available()
        ):
            return _metric_solve_coeff(
                params,
                geom,
                rhs,
                coeff,
                bc_phi,
                x0=phi_guess,
                return_iters=return_iters,
            )
        precond = params.polarization_preconditioner
        if precond == "auto":
            precond = "spectral_jacobi"
        phi_guess_eff = None if params.non_boussinesq_perturbed_density_on else phi_guess
        return inv_div_n_grad_cg(
            rhs,
            n_coeff=coeff,
            dx=grid.dx,
            dy=grid.dy,
            bc=bc_phi,
            maxiter=int(params.polarization_cg_maxiter),
            tol=float(params.polarization_cg_tol),
            atol=float(params.polarization_cg_atol),
            preconditioner=precond,
            preconditioner_shift=float(params.polarization_precond_shift),
            preconditioner_fn=getattr(geom, "polarization_preconditioner_fn", None),
            x0=phi_guess_eff,
            return_iters=return_iters,
        )

    omega = omega - _diamagnetic_polarisation_term(params, geom, n_phys, Ti, bc_phi)
    if grid is None:
        if params.boussinesq:
            phi = geom.inv_laplacian(omega, x0=phi_guess)
            return (phi, jnp.asarray(0)) if return_iters else phi
        n_eff = _n_eff(params, n_phys)
        phi_guess_eff = phi_guess
        if params.non_boussinesq_perturbed_density_on:
            phi_guess_eff = None
        phi = geom.inv_div_n_grad(n_eff, omega, x0=phi_guess_eff)
        return (phi, jnp.asarray(0)) if return_iters else phi

    with jax.named_scope("poisson_solve"):
        if params.boussinesq:
            if params.poisson_metric_on and hasattr(geom, "inv_laplacian_metric"):
                metric_ok = True
                if hasattr(geom, "metric_available"):
                    metric_ok = bool(geom.metric_available())
                if metric_ok:
                    phi = geom.inv_laplacian_metric(omega, x0=phi_guess)
                    return (phi, jnp.asarray(0)) if return_iters else phi
            poisson = params.poisson
            if params.poisson_force_spectral_when_periodic and is_periodic_bc(bc_phi, geom):
                poisson = "spectral"
            if (
                params.poisson_force_fd_fft_when_nonperiodic
                and not is_periodic_bc(bc_phi, geom)
                and poisson == "spectral"
            ):
                poisson = "cg_fd"
            if poisson == "spectral":
                if not is_periodic_bc(bc_phi, geom):
                    raise ValueError("Spectral Poisson solve requires periodic BCs in x and y.")
                phi = inv_laplacian_spec(omega, grid.k2, k2_min=params.k2_min)
                return (phi, jnp.asarray(0)) if return_iters else phi
            if poisson == "mixed_fft":
                phi = inv_laplacian_mixed_fft(
                    omega,
                    dx=grid.dx,
                    dy=grid.dy,
                    bc=bc_phi,
                    gauge_epsilon=params.poisson_gauge_epsilon,
                )
                return (phi, jnp.asarray(0)) if return_iters else phi
            precond = params.poisson_preconditioner
            if precond == "auto":
                precond = "spectral"
            if precond == "spectral" and not is_periodic_bc(bc_phi, geom):
                precond = "jacobi"
            if poisson == "cg_fd":
                try:
                    eigs = getattr(geom, "poisson_fd_fft_eigs", None)
                    lam_x, lam_y = eigs if eigs is not None else (None, None)
                    phi = inv_laplacian_fd_fft(
                        omega,
                        dx=grid.dx,
                        dy=grid.dy,
                        bc=bc_phi,
                        gauge_epsilon=params.poisson_gauge_epsilon,
                        lam_x=lam_x,
                        lam_y=lam_y,
                    )
                    return (phi, jnp.asarray(0)) if return_iters else phi
                except ValueError:
                    pass
            precond_fn = getattr(geom, "poisson_preconditioner_fn", None)
            phi = inv_laplacian_cg(
                omega,
                dx=grid.dx,
                dy=grid.dy,
                bc=bc_phi,
                maxiter=int(params.poisson_cg_maxiter),
                tol=float(params.poisson_cg_tol),
                atol=float(params.poisson_cg_atol),
                preconditioner=str(precond),
                k2_precond=grid.k2 if str(precond) == "spectral" else None,
                gauge_epsilon=params.poisson_gauge_epsilon,
                preconditioner_fn=precond_fn,
                x0=phi_guess,
                return_iters=return_iters,
            )
            return phi if return_iters else phi

        n_eff = _n_eff(params, n_phys)
        precond = params.polarization_preconditioner
        if precond == "auto":
            precond = "spectral_jacobi"
        phi_guess_eff = phi_guess
        if params.non_boussinesq_perturbed_density_on:
            phi_guess_eff = None
        phi = inv_div_n_grad_cg(
            omega,
            n_coeff=n_eff,
            dx=grid.dx,
            dy=grid.dy,
            bc=bc_phi,
            maxiter=int(params.polarization_cg_maxiter),
            tol=float(params.polarization_cg_tol),
            atol=float(params.polarization_cg_atol),
            preconditioner=precond,
            preconditioner_shift=float(params.polarization_precond_shift),
            preconditioner_fn=getattr(geom, "polarization_preconditioner_fn", None),
            x0=phi_guess_eff,
            return_iters=return_iters,
        )
        return phi if return_iters else phi


def omega_from_phi(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    phi: jnp.ndarray,
    n_phys: jnp.ndarray,
    bc_phi: BC2D,
    Ti: jnp.ndarray | None = None,
    Te: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Forward operator for Poisson/polarization: omega = ∇² phi (or div(n ∇ phi))."""

    scale = float(params.poisson_scale)
    grid = _perp_grid_of(geom)
    full_grid = _full_grid_of(geom)

    if params.boussinesq:
        if params.poisson_b_weighted:
            coeff = _poisson_coeff_b(params, geom, n_phys)
            mode = str(getattr(params, "poisson_b_weighted_mode", "scaled")).lower()
            if mode == "hermes":
                Abar = float(getattr(params, "average_atomic_mass", 1.0))
                B = getattr(geom, "B", None)
                if B is None:
                    invB2 = jnp.asarray(1.0)
                else:
                    invB2 = 1.0 / jnp.maximum(jnp.asarray(B), 1e-12) ** 2
                coeff = _broadcast_to_shape(jnp.asarray(Abar) * invB2, n_phys.shape)
            omega_base = _metric_div_coeff(params, geom, phi, coeff, bc_phi)
            if mode == "hermes":
                split_n0 = bool(getattr(params, "poisson_split_n0", False))
                if (
                    split_n0
                    and phi.ndim == 3
                    and params.poisson_metric_on
                    and hasattr(geom, "metric_available")
                    and geom.metric_available()
                ):
                    phi0 = phi.mean(axis=2)
                    phi1 = phi - phi0[..., None]
                    coeff0 = coeff.mean(axis=2)
                    omega1 = _metric_div_coeff(params, geom, phi1, coeff, bc_phi)
                    bc_xy = BC2D(
                        kind_x=bc_phi.kind_x,
                        kind_y=2 if getattr(full_grid, "open_field_line", False) else bc_phi.kind_y,
                        x_value=bc_phi.x_value,
                        y_value=0.0,
                        x_grad=bc_phi.x_grad,
                        y_grad=0.0,
                    )
                    omega0_xy = _metric_div_xy(
                        params,
                        geom,
                        jnp.swapaxes(phi0, 0, 1),
                        jnp.swapaxes(coeff0, 0, 1),
                        bc_xy,
                    )
                    omega0 = jnp.swapaxes(omega0_xy, 0, 1)
                    omega_base = omega1 + omega0[..., None]
            if bool(getattr(params, "diamagnetic_polarisation_on", False)):
                if mode == "hermes":
                    Ti_eff = jnp.zeros_like(n_phys) if Ti is None else Ti
                    Te_eff = jnp.zeros_like(n_phys) if Te is None else Te
                    me_hat = float(getattr(params, "me_hat", 0.0))
                    p_hat = Ti_eff * n_phys - me_hat * _electron_pressure(params, n_phys, Te_eff)
                    # Hermes polarization solve is ∇·[(Abar/B^2)∇(phi + Pi_hat)] = Vort.
                    # Pi_hat enters with the same metric coefficient as phi (no n_eff scaling).
                    coeff_pi = coeff
                    split_n0 = bool(getattr(params, "poisson_split_n0", False))
                    if (
                        split_n0
                        and p_hat.ndim == 3
                        and params.poisson_metric_on
                        and hasattr(geom, "metric_available")
                        and geom.metric_available()
                    ):
                        p0 = p_hat.mean(axis=2)
                        p1 = p_hat - p0[..., None]
                        coeff0 = coeff_pi.mean(axis=2)
                        term1 = _metric_div_coeff(params, geom, p1, coeff_pi, bc_phi)
                        bc_xy = BC2D(
                            kind_x=bc_phi.kind_x,
                            kind_y=(
                                2 if getattr(full_grid, "open_field_line", False) else bc_phi.kind_y
                            ),
                            x_value=bc_phi.x_value,
                            y_value=0.0,
                            x_grad=bc_phi.x_grad,
                            y_grad=0.0,
                        )
                        term0_xy = _metric_div_xy(
                            params,
                            geom,
                            jnp.swapaxes(p0, 0, 1),
                            jnp.swapaxes(coeff0, 0, 1),
                            bc_xy,
                        )
                        term0 = jnp.swapaxes(term0_xy, 0, 1)
                        omega_base = omega_base + term1 + term0[..., None]
                    else:
                        omega_base = omega_base + _metric_div_coeff(
                            params, geom, p_hat, coeff_pi, bc_phi
                        )
                else:
                    tau_i = float(getattr(params, "tau_i", 0.0))
                    Ti_eff = jnp.zeros_like(n_phys) if Ti is None else Ti
                    p_i = tau_i * (n_phys * Ti_eff)
                    omega_base = omega_base + _metric_div_coeff(params, geom, p_i, coeff, bc_phi)
            omega = omega_base if mode == "hermes" else omega_base / coeff
            return omega * scale if scale != 1.0 else omega

        if params.poisson_metric_on and hasattr(geom, "laplacian_metric"):
            metric_ok = True
            if hasattr(geom, "metric_available"):
                metric_ok = bool(geom.metric_available())
            if metric_ok:
                omega_metric = geom.laplacian_metric(phi)
                omega_metric = omega_metric + _diamagnetic_polarisation_term(
                    params, geom, n_phys, Ti, bc_phi
                )
                return omega_metric * scale if scale != 1.0 else omega_metric

        omega = laplacian(params, geom, phi, bc_phi)
        omega = omega + _diamagnetic_polarisation_term(params, geom, n_phys, Ti, bc_phi)
        return omega * scale if scale != 1.0 else omega

    n_eff = _n_eff(params, n_phys)
    if grid is None:
        omega = geom.laplacian(phi)
        return omega * scale if scale != 1.0 else omega

    if params.poisson_b_weighted:
        coeff = _poisson_coeff_b(params, geom, n_phys)
        mode = str(getattr(params, "poisson_b_weighted_mode", "scaled")).lower()
        omega_base = _metric_div_coeff(params, geom, phi, coeff, bc_phi)
        if bool(getattr(params, "diamagnetic_polarisation_on", False)):
            if mode == "hermes":
                Ti_eff = jnp.zeros_like(n_phys) if Ti is None else Ti
                Te_eff = jnp.zeros_like(n_phys) if Te is None else Te
                me_hat = float(getattr(params, "me_hat", 0.0))
                p_hat = Ti_eff * n_phys - me_hat * _electron_pressure(params, n_phys, Te_eff)
                coeff_pi = coeff
                omega_base = omega_base + _metric_div_coeff(params, geom, p_hat, coeff_pi, bc_phi)
            else:
                tau_i = float(getattr(params, "tau_i", 0.0))
                Ti_eff = jnp.zeros_like(n_phys) if Ti is None else Ti
                p_i = tau_i * (n_phys * Ti_eff)
                omega_base = omega_base + _metric_div_coeff(params, geom, p_i, coeff, bc_phi)
        omega = omega_base if mode == "hermes" else omega_base / coeff
        return omega * scale if scale != 1.0 else omega

    if phi.ndim == 2:
        omega = div_n_grad(phi, n_eff, dx=grid.dx, dy=grid.dy, bc=bc_phi)
        omega = omega + _diamagnetic_polarisation_term(params, geom, n_phys, Ti, bc_phi)
        return omega * scale if scale != 1.0 else omega

    omega = jax.vmap(lambda p, nloc: div_n_grad(p, nloc, dx=grid.dx, dy=grid.dy, bc=bc_phi))(
        phi, n_eff
    )
    omega = omega + _diamagnetic_polarisation_term(params, geom, n_phys, Ti, bc_phi)
    return omega * scale if scale != 1.0 else omega
