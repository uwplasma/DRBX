from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.operators.fd2d import (
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    inv_laplacian_fd_fft,
    inv_laplacian_mixed_fft,
)
from jaxdrb.operators.spectral2d import inv_laplacian as inv_laplacian_spec

from .ops import grid_of, is_periodic_bc


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
    phi_guess: jnp.ndarray | None = None,
) -> jnp.ndarray:
    grid = grid_of(geom)
    if grid is None:
        if params.boussinesq:
            return geom.inv_laplacian(omega, x0=phi_guess)
        n_eff = _n_eff(params, n_phys)
        return geom.inv_div_n_grad(n_eff, omega, x0=phi_guess)

    if params.boussinesq:
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
            return inv_laplacian_spec(omega, grid.k2, k2_min=params.k2_min)
        if poisson == "mixed_fft":
            return inv_laplacian_mixed_fft(
                omega,
                dx=grid.dx,
                dy=grid.dy,
                bc=bc_phi,
                gauge_epsilon=params.poisson_gauge_epsilon,
            )
        precond = params.poisson_preconditioner
        if precond == "auto":
            precond = "spectral"
        if precond == "spectral" and not is_periodic_bc(bc_phi, geom):
            precond = "jacobi"
        if poisson == "cg_fd":
            try:
                eigs = getattr(geom, "poisson_fd_fft_eigs", None)
                lam_x, lam_y = eigs if eigs is not None else (None, None)
                return inv_laplacian_fd_fft(
                    omega,
                    dx=grid.dx,
                    dy=grid.dy,
                    bc=bc_phi,
                    gauge_epsilon=params.poisson_gauge_epsilon,
                    lam_x=lam_x,
                    lam_y=lam_y,
                )
            except ValueError:
                pass
        precond_fn = getattr(geom, "poisson_preconditioner_fn", None)
        return inv_laplacian_cg(
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
        )

    n_eff = _n_eff(params, n_phys)
    precond = params.polarization_preconditioner
    if precond == "auto":
        precond = "spectral_jacobi"
    return inv_div_n_grad_cg(
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
        x0=phi_guess,
    )
