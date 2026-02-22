from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.state import DRBSystemState
from jaxdrb.operators.fd2d import enforce_bc_relaxation

from .ops import grid_of, is_2d, region_mask


def _perp_grid(geom: GeometryAdapter):
    grid = grid_of(geom)
    if grid is not None:
        return grid
    grid = getattr(geom, "grid", None)
    if grid is None:
        return None
    return getattr(grid, "perp", grid)


def _sol_y_taper(geom: GeometryAdapter, width: float) -> jnp.ndarray | None:
    grid = _perp_grid(geom)
    if grid is None or width <= 0.0:
        return None
    y = grid.y[None, :]
    Ly = float(grid.Ly)
    y0 = max(float(width), 1e-8)
    taper = jnp.tanh(y / y0) ** 4 * jnp.tanh((Ly - y) / y0) ** 4
    return taper


def _sol_source_profile(
    geom: GeometryAdapter,
    *,
    xs: float,
    width: float,
    src_mask: jnp.ndarray,
    y_taper: jnp.ndarray | None,
) -> jnp.ndarray:
    grid = _perp_grid(geom)
    if grid is None:
        return jnp.zeros_like(src_mask)
    width = max(width, 1e-8)
    x = grid.x[:, None]
    profile = jnp.exp(-0.5 * ((x - xs) / width) ** 2)
    if y_taper is not None:
        profile = profile * y_taper
    return profile * src_mask


def sol_masks(
    params: DRBSystemParams, geom: GeometryAdapter
) -> tuple[jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray | float]:
    shape = geom.shape()

    # Prefer region masks if provided by the geometry.
    mask_open = None
    for name in ("open", "sol", "divertor", "leg"):
        mask_open = region_mask(geom, name, shape)
        if mask_open is not None:
            break

    mask_closed = None
    for name in ("closed", "core"):
        mask_closed = region_mask(geom, name, shape)
        if mask_closed is not None:
            break

    if mask_open is not None or mask_closed is not None:
        if mask_open is None and mask_closed is not None:
            mask_open = 1.0 - mask_closed
        if mask_closed is None and mask_open is not None:
            mask_closed = 1.0 - mask_open
        nonlinear_scale = mask_closed + float(params.sol_nonlinear_open_scale) * mask_open
        return mask_closed, mask_open, nonlinear_scale

    # Fall back to 2D SOL masks if no region masks are available.
    grid = _perp_grid(geom)
    if grid is None:
        return None, None, 1.0
    xs = float(params.sol_xs)
    width = max(float(params.sol_width), 1e-8)
    x = grid.x[:, None]
    if params.sol_open_left:
        mask_open = 0.5 * (1.0 - jnp.tanh((x - xs) / width))
    else:
        mask_open = 0.5 * (1.0 + jnp.tanh((x - xs) / width))
    mask_closed = 1.0 - mask_open
    y_taper = _sol_y_taper(geom, float(params.sol_mask_y_taper))
    if y_taper is not None:
        mask_open = mask_open * y_taper
        mask_closed = mask_closed * y_taper
    nonlinear_scale = mask_closed + float(params.sol_nonlinear_open_scale) * mask_open
    return mask_closed, mask_open, nonlinear_scale


def apply_sol_phi_bc(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    phi: jnp.ndarray,
    Te_phys: jnp.ndarray,
    bc_phi: BC2D,
) -> jnp.ndarray:
    if not params.sol_on or not params.sol_phi_bc_on:
        return phi
    if not is_2d(geom) or bc_phi.kind_x == 0:
        return phi
    phi_bc = float(params.sol_phi_bc_lambda) * Te_phys
    if params.sol_open_left:
        return phi.at[0, :].set(phi_bc[0, :])
    return phi.at[-1, :].set(phi_bc[-1, :])


def sol_sources(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    *,
    n_phys: jnp.ndarray,
    Te_phys: jnp.ndarray,
    mask_closed: jnp.ndarray | None,
    mask_open: jnp.ndarray | None,
) -> DRBSystemState:
    if not params.sol_on or mask_open is None:
        z = jnp.zeros_like(n_phys)
        return DRBSystemState(
            n=z,
            omega=jnp.zeros_like(n_phys),
            vpar_e=jnp.zeros_like(n_phys),
            vpar_i=jnp.zeros_like(n_phys),
            Te=jnp.zeros_like(n_phys),
            Ti=None,
            psi=None,
            N=None,
        )

    n_eq = (
        float(params.sol_n_sol) + (float(params.sol_n_core) - float(params.sol_n_sol)) * mask_closed
    )
    Te_eq = (
        float(params.sol_Te_sol)
        + (float(params.sol_Te_core) - float(params.sol_Te_sol)) * mask_closed
    )
    relax = float(params.sol_relax_core) * mask_closed + float(params.sol_relax_open) * mask_open
    sol_source_n = relax * (n_eq - n_phys)
    sol_source_Te = relax * (Te_eq - Te_phys)

    if (params.sol_source_n0 != 0.0) or (params.sol_source_Te0 != 0.0):
        src_mask = 1.0
        mode = str(params.sol_source_mask).lower()
        if mode == "closed":
            src_mask = mask_closed
        elif mode == "open":
            src_mask = mask_open
        y_taper = _sol_y_taper(geom, float(params.sol_source_y_taper))
        profile = _sol_source_profile(
            geom,
            xs=float(params.sol_source_xs),
            width=float(params.sol_source_width),
            src_mask=src_mask,
            y_taper=y_taper,
        )
        sol_source_n = sol_source_n + float(params.sol_source_n0) * profile
        sol_source_Te = sol_source_Te + float(params.sol_source_Te0) * profile

        if (params.sol_source2_n0 != 0.0) or (params.sol_source2_Te0 != 0.0):
            profile2 = _sol_source_profile(
                geom,
                xs=float(params.sol_source2_xs),
                width=float(params.sol_source2_width),
                src_mask=src_mask,
                y_taper=y_taper,
            )
            sol_source_n = sol_source_n + float(params.sol_source2_n0) * profile2
            sol_source_Te = sol_source_Te + float(params.sol_source2_Te0) * profile2

    return DRBSystemState(
        n=sol_source_n,
        omega=jnp.zeros_like(sol_source_n),
        vpar_e=jnp.zeros_like(sol_source_n),
        vpar_i=jnp.zeros_like(sol_source_n),
        Te=sol_source_Te,
        Ti=None,
        psi=None,
        N=None,
    )


def sol_sinks(
    params: DRBSystemParams,
    *,
    n_phys: jnp.ndarray,
    Te_phys: jnp.ndarray,
    omega: jnp.ndarray,
    mask_open: jnp.ndarray | None,
) -> DRBSystemState:
    if not params.sol_on or mask_open is None:
        z = jnp.zeros_like(n_phys)
        return DRBSystemState(
            n=z,
            omega=jnp.zeros_like(n_phys),
            vpar_e=jnp.zeros_like(n_phys),
            vpar_i=jnp.zeros_like(n_phys),
            Te=jnp.zeros_like(n_phys),
            Ti=None,
            psi=None,
            N=None,
        )

    n_pos = jnp.maximum(n_phys, float(params.sol_n_floor))
    Te_pos = jnp.maximum(Te_phys, float(params.sol_Te_floor))
    sol_sink_n = -float(params.sol_sink_open_n) * mask_open * n_pos
    sol_sink_Te = -float(params.sol_sink_open_Te) * mask_open * Te_pos
    sol_sink_omega = sol_sink_open_omega(params, omega, mask_open)
    sol_sink_vpar = -float(params.sol_sink_open_vpar) * mask_open

    return DRBSystemState(
        n=sol_sink_n,
        omega=sol_sink_omega,
        vpar_e=sol_sink_vpar,
        vpar_i=sol_sink_vpar,
        Te=sol_sink_Te,
        Ti=None,
        psi=None,
        N=None,
    )


def sol_sink_open_omega(
    params: DRBSystemParams, omega: jnp.ndarray, mask_open: jnp.ndarray | None
) -> jnp.ndarray:
    nu = float(params.sol_sink_open_omega)
    if nu == 0.0 or mask_open is None:
        return jnp.zeros_like(omega)
    mode = str(params.sol_sink_open_omega_mode).lower()
    if mode in ("global", "avg", "mean"):
        denom = jnp.sum(mask_open, axis=1, keepdims=True)
        denom = jnp.where(denom > 0.0, denom, 1.0)
        omega_avg = jnp.sum(omega * mask_open, axis=1, keepdims=True) / denom
        return -nu * mask_open * (omega - omega_avg)
    return -nu * mask_open * omega


def sol_parallel_loss(
    params: DRBSystemParams,
    y: DRBSystemState,
    phi: jnp.ndarray,
    *,
    n_phys: jnp.ndarray,
    Te_phys: jnp.ndarray,
    mask_open: jnp.ndarray | None,
) -> DRBSystemState:
    if not params.sol_parallel_loss_on or mask_open is None:
        return DRBSystemState(
            n=jnp.zeros_like(y.n),
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )
    q = float(params.sol_parallel_loss_q)
    if q <= 0.0:
        return DRBSystemState(
            n=jnp.zeros_like(y.n),
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    coeff = float(params.sol_parallel_loss_coeff) / (2.0 * jnp.pi * q)
    model = str(params.sol_parallel_loss_model).lower()
    Te_floor = max(float(params.sol_parallel_loss_Te_floor), float(params.sol_Te_floor))
    Te_eff = jnp.maximum(Te_phys, Te_floor)
    n_pos = jnp.maximum(n_phys, float(params.sol_n_floor))
    if model in ("linear", "lin", "bohm_linear"):
        delta = phi / Te_eff - float(params.sol_parallel_loss_lambda)
        gamma = coeff * delta
    elif model in ("exp", "bohm_exp"):
        exp_arg = float(params.sol_parallel_loss_lambda) - phi / Te_eff
        exp_arg = jnp.clip(exp_arg, a_min=-10.0, a_max=10.0)
        gamma = coeff * (1.0 - jnp.exp(exp_arg))
    else:
        gamma = coeff
    cs = jnp.sqrt(Te_eff)
    loss_n = -gamma * mask_open * n_pos * cs
    loss_Te = -gamma * mask_open * 3.0 * n_pos * Te_eff * cs
    loss_vpar = jnp.zeros_like(y.vpar_e)
    if params.sol_parallel_loss_vpar_on:
        loss_vpar = -gamma * mask_open
    loss_omega = jnp.zeros_like(y.omega)
    if params.sol_parallel_loss_omega_on:
        loss_omega = -gamma * mask_open * y.omega
    return DRBSystemState(
        n=loss_n,
        omega=loss_omega,
        vpar_e=loss_vpar * y.vpar_e,
        vpar_i=loss_vpar * y.vpar_i,
        Te=loss_Te,
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def sol_sheath_phi_term(
    params: DRBSystemParams,
    y: DRBSystemState,
    phi: jnp.ndarray,
    *,
    n_phys: jnp.ndarray,
    Te_phys: jnp.ndarray,
    mask_open: jnp.ndarray | None,
) -> DRBSystemState:
    if (
        not params.sol_sheath_phi_on
        or bool(params.sol_sheath_phi_implicit)
        or mask_open is None
        or float(params.sol_parallel_loss_q) <= 0.0
    ):
        return DRBSystemState(
            n=jnp.zeros_like(y.n),
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    Te_floor = max(float(params.sol_sheath_phi_Te_floor), float(params.sol_Te_floor))
    Te_eff = jnp.maximum(Te_phys, Te_floor)
    n_pos = jnp.maximum(n_phys, float(params.sol_n_floor))
    cs = jnp.sqrt(Te_eff)
    gamma = float(params.sol_sheath_phi_coeff) / (2.0 * jnp.pi * float(params.sol_parallel_loss_q))
    model = str(params.sol_sheath_phi_model).lower()
    if model in ("linear", "lin"):
        delta = phi / Te_eff - float(params.sol_sheath_phi_lambda)
        clip = float(params.sol_sheath_phi_clip)
        if clip > 0.0:
            delta = jnp.clip(delta, a_min=-clip, a_max=clip)
        sheath_current = n_pos * cs * delta
    else:
        exp_arg = float(params.sol_sheath_phi_lambda) - phi / Te_eff
        clip = float(params.sol_sheath_phi_clip)
        exp_arg = jnp.clip(exp_arg, a_min=-clip, a_max=clip)
        sheath_current = n_pos * cs * (1.0 - jnp.exp(exp_arg))
    domega = -gamma * mask_open * sheath_current
    return DRBSystemState(
        n=jnp.zeros_like(y.n),
        omega=domega,
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=jnp.zeros_like(y.Te),
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def sol_sheath_omega_sink(
    params: DRBSystemParams, omega: jnp.ndarray, mask_open: jnp.ndarray | None
) -> jnp.ndarray:
    if (
        not params.sol_sheath_omega_on
        or mask_open is None
        or float(params.sol_parallel_loss_q) <= 0.0
    ):
        return jnp.zeros_like(omega)
    gamma = float(params.sol_sheath_omega_coeff) / (
        2.0 * jnp.pi * float(params.sol_parallel_loss_q)
    )
    denom = jnp.sum(mask_open, axis=1, keepdims=True)
    denom = jnp.where(denom > 0.0, denom, 1.0)
    omega_avg = jnp.sum(omega * mask_open, axis=1, keepdims=True) / denom
    return -gamma * mask_open * (omega - omega_avg)


def sol_omega_bc_dirichlet(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    y: DRBSystemState,
    *,
    bc_omega: BC2D,
) -> DRBSystemState:
    if not params.sol_on or not params.sol_omega_bc_dirichlet_on or not is_2d(geom):
        return DRBSystemState(
            n=jnp.zeros_like(y.n),
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    grid = grid_of(geom)
    if grid is None:
        return DRBSystemState(
            n=jnp.zeros_like(y.n),
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    bc = BC2D(
        kind_x=1,
        kind_y=1 if params.sol_omega_bc_apply_y else bc_omega.kind_y,
        x_value=float(params.sol_omega_bc_value),
        y_value=float(bc_omega.y_value),
    )
    omega_bc = enforce_bc_relaxation(
        y.omega,
        dx=grid.dx,
        dy=grid.dy,
        bc=bc,
        nu=float(params.sol_omega_bc_nu),
    )
    return DRBSystemState(
        n=jnp.zeros_like(y.n),
        omega=omega_bc,
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=jnp.zeros_like(y.Te),
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def sol_vpar_bc_dirichlet(
    params: DRBSystemParams, geom: GeometryAdapter, y: DRBSystemState
) -> DRBSystemState:
    if not params.sol_on or not params.sol_vpar_bc_dirichlet_on or not is_2d(geom):
        return DRBSystemState(
            n=jnp.zeros_like(y.n),
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    nu_bc = float(params.sol_vpar_bc_nu)
    vpar_val = float(params.sol_vpar_bc_value)
    ny = y.vpar_e.shape[1]
    mask_bottom = (jnp.arange(ny) == 0).astype(y.vpar_e.dtype)[None, :]
    mask_top = (jnp.arange(ny) == (ny - 1)).astype(y.vpar_e.dtype)[None, :]
    vpar_e_bc = -nu_bc * (mask_bottom * (y.vpar_e - (-vpar_val)) + mask_top * (y.vpar_e - vpar_val))
    vpar_i_bc = -nu_bc * (mask_bottom * (y.vpar_i - (-vpar_val)) + mask_top * (y.vpar_i - vpar_val))
    return DRBSystemState(
        n=jnp.zeros_like(y.n),
        omega=jnp.zeros_like(y.omega),
        vpar_e=vpar_e_bc,
        vpar_i=vpar_i_bc,
        Te=jnp.zeros_like(y.Te),
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def sol_edge_relaxation(
    params: DRBSystemParams, geom: GeometryAdapter, y: DRBSystemState
) -> DRBSystemState:
    if not params.sol_on or not params.sol_edge_relax_on or not is_2d(geom):
        return DRBSystemState(
            n=jnp.zeros_like(y.n),
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    nu_bc = float(params.sol_edge_relax_nu)
    n_floor = float(params.sol_n_floor)
    Te_floor = float(params.sol_Te_floor)
    n_right = float(params.sol_edge_n_right)
    Te_right = float(params.sol_edge_Te_right)
    if params.log_n:
        n_right = jnp.log(jnp.maximum(n_right, n_floor))
    if params.log_Te:
        Te_right = jnp.log(jnp.maximum(Te_right, Te_floor))

    nx, ny = y.n.shape
    mask_left = (jnp.arange(nx) == 0).astype(y.n.dtype)[:, None]
    mask_right = (jnp.arange(nx) == (nx - 1)).astype(y.n.dtype)[:, None]
    mask_bottom = (jnp.arange(ny) == 0).astype(y.n.dtype)[None, :]
    mask_top = (jnp.arange(ny) == (ny - 1)).astype(y.n.dtype)[None, :]

    n_left_target = y.n[1, :]
    Te_left_target = y.Te[1, :]
    n_right_target = jnp.full_like(y.n[0, :], n_right)
    Te_right_target = jnp.full_like(y.Te[0, :], Te_right)

    n_bc = -nu_bc * (mask_left * (y.n - n_left_target) + mask_right * (y.n - n_right_target))
    Te_bc = -nu_bc * (mask_left * (y.Te - Te_left_target) + mask_right * (y.Te - Te_right_target))
    if params.sol_edge_relax_apply_y:
        n_bc = n_bc - nu_bc * (mask_bottom * (y.n - y.n[:, [1]]) + mask_top * (y.n - y.n[:, [-2]]))
        Te_bc = Te_bc - nu_bc * (
            mask_bottom * (y.Te - y.Te[:, [1]]) + mask_top * (y.Te - y.Te[:, [-2]])
        )

    return DRBSystemState(
        n=n_bc,
        omega=jnp.zeros_like(y.omega),
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=Te_bc,
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
