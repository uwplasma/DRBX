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
)
from jaxdrb.operators.spectral2d import inv_laplacian as inv_laplacian_spec

from .ops import ddx, ddy, grid_of, is_periodic_bc, laplacian


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
    elif arr.ndim == 2 and len(shape) == 3 and arr.shape == shape[1:]:
        arr = arr[None, :, :]
    return jnp.broadcast_to(arr, shape)


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
    p_i = tau_i * (n_phys + Ti_eff)

    B = getattr(geom, "B", None)
    if B is None:
        invB2 = 1.0
    else:
        invB2 = 1.0 / jnp.maximum(jnp.asarray(B), 1e-12) ** 2
        invB2 = _broadcast_to_shape(invB2, n_phys.shape)

    scale = float(getattr(params, "diamagnetic_polarisation_scale", 1.0))
    grid = grid_of(geom)

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
        return Te
    clip = params.log_Te_clip
    if clip is None:
        return jnp.exp(Te)
    clip_val = float(clip)
    return jnp.exp(jnp.clip(Te, a_min=-clip_val, a_max=clip_val))


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


def phi_from_omega(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    omega: jnp.ndarray,
    n_phys: jnp.ndarray,
    bc_phi: BC2D,
    Ti: jnp.ndarray | None = None,
    phi_guess: jnp.ndarray | None = None,
    return_iters: bool = False,
) -> jnp.ndarray:
    grid = grid_of(geom)
    scale = float(params.poisson_scale)
    omega = omega / scale if scale != 1.0 else omega
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
) -> jnp.ndarray:
    """Forward operator for Poisson/polarization: omega = ∇² phi (or div(n ∇ phi))."""

    scale = float(params.poisson_scale)

    if params.boussinesq:
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
    grid = grid_of(geom)
    if grid is None:
        omega = geom.laplacian(phi)
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
