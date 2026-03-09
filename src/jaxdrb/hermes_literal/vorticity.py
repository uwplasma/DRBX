"""Literal Hermes vorticity pieces for strict parity work."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from jaxdrb.bc import BC2D

from .delp2 import delp2_runtime
from .exb import div_n_bxgrad_f_b_xppm
from .fv import div_a_grad_perp

if TYPE_CHECKING:
    from jaxdrb.core.state import DRBSystemState
    from jaxdrb.core.terms.context import TermContext


def pi_hat(
    params,
    *,
    n_phys: jnp.ndarray,
    Te_phys: jnp.ndarray,
    Ti: jnp.ndarray,
) -> jnp.ndarray:
    """Mirror Hermes `Pi_hat` contribution used in vorticity advection."""

    if not bool(getattr(params, "diamagnetic_polarisation_on", False)):
        return jnp.zeros_like(n_phys)
    abar = max(float(getattr(params, "average_atomic_mass", 1.0)), 1e-12)
    electron_coeff = float(getattr(params, "me_hat", 0.0)) / abar
    electron_pressure = n_phys * Te_phys
    ion_pressure = n_phys * Ti
    return ion_pressure - electron_coeff * electron_pressure


def _mirror_zlength(geom) -> float:
    grid = getattr(geom, "grid", None)
    perp = getattr(grid, "perp", None)
    if perp is None:
        raise ValueError("Hermes mirror vorticity path requires a field-aligned grid.")
    return float(getattr(perp, "dy", 1.0)) * float(perp.ny)


def _phi_bcx_for_delp(params, bc_phi: BC2D) -> int:
    kind = int(getattr(bc_phi, "kind_x", 0))
    if bool(getattr(params, "poisson_invert_set", False)) and kind != 0:
        return 1
    return kind


def _runtime_exb_term(
    adv: jnp.ndarray,
    f: jnp.ndarray,
    *,
    geom,
    params,
    bc_phi: BC2D,
    bc_adv: BC2D,
    adv_free_o2: bool,
) -> jnp.ndarray:
    adv_arr = jnp.asarray(adv, dtype=jnp.float64)
    f_arr = jnp.asarray(f, dtype=jnp.float64)
    if adv_arr.shape != f_arr.shape or adv_arr.ndim != 3:
        raise ValueError(
            f"Mirror vorticity ExB runtime expects matching `(nz, nx, ny)` fields, got {adv_arr.shape} and {f_arr.shape}."
        )
    if any(
        getattr(geom, name, None) is None
        for name in (
            "jacobian",
            "metric_dx",
            "metric_dy",
            "metric_dz",
            "gxx",
            "g23",
            "B",
            "z_shift",
        )
    ):
        raise ValueError(
            "Hermes mirror vorticity ExB path requires Jacobian, metrics, Bxy, and z_shift."
        )
    periodic_parallel = not bool(getattr(getattr(geom, "grid", None), "open_field_line", False))
    lower_boundary_open = bool(getattr(getattr(geom, "grid", None), "open_field_line", False))
    upper_boundary_open = lower_boundary_open

    return div_n_bxgrad_f_b_xppm(
        adv_arr,
        f_arr,
        jacobian=geom.jacobian,
        dx=geom.metric_dx,
        dy=geom.metric_dy,
        dz=geom.metric_dz,
        g11=geom.gxx,
        g23=geom.g23,
        bxy=geom.B,
        z_shift=geom.z_shift,
        zlength=_mirror_zlength(geom),
        bc_phi=bc_phi,
        bc_adv=bc_adv,
        bndry_flux=bool(getattr(params, "exb_bndry_flux", True)),
        poloidal=bool(getattr(params, "exb_poloidal_flows", False)),
        positive=False,
        interp=str(getattr(params, "parallel_shift_interp", "spectral")),
        neumann_boundary_average_z=bool(getattr(params, "neumann_boundary_average_y", False)),
        use_mc=True,
        periodic_parallel=periodic_parallel,
        periodic_binormal=int(getattr(bc_adv, "kind_y", 0)) == 0,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
        poisson_invert_set=False,
        parallel_edge_block=int(getattr(params, "hermes_mirror_parallel_edge_block", 0)),
        apply_free_o2_adv=adv_free_o2,
        poloidal_scale=float(getattr(params, "exb_poloidal_scale", 1.0)),
        poloidal_x_scale=float(getattr(params, "exb_poloidal_x_scale", 1.0)),
        poloidal_y_scale=float(getattr(params, "exb_poloidal_y_scale", 1.0)),
    )


def full_omega_exb_advection(
    ctx: "TermContext",
    y: "DRBSystemState",
    *,
    phi: jnp.ndarray,
    scale: jnp.ndarray | float,
) -> jnp.ndarray:
    """Literal Stage 1 translation of the Hermes full vorticity ExB branch.

    Source of truth:
    `/Users/rogerio/local/hermes-3/src/vorticity.cxx`
    """

    abar = float(getattr(ctx.params, "average_atomic_mass", 1.0))
    b = getattr(ctx.geom, "B", None)
    if b is None:
        inv_bsq = jnp.asarray(1.0, dtype=phi.dtype)
    else:
        inv_bsq = 1.0 / jnp.maximum(jnp.asarray(b, dtype=phi.dtype), 1e-12) ** 2
    inv_bsq = jnp.broadcast_to(inv_bsq, phi.shape)
    bc = ctx.bcs

    pihat = pi_hat(ctx.params, n_phys=ctx.n_phys, Te_phys=ctx.Te_phys, Ti=ctx.Ti)
    term = -_runtime_exb_term(
        0.5 * y.omega,
        phi,
        geom=ctx.geom,
        params=ctx.params,
        bc_phi=bc.phi,
        bc_adv=bc.omega,
        adv_free_o2=False,
    )

    vedotgradpi = ctx.geom.bracket(phi, pihat, bc_phi=bc.phi, bc_f=bc.Te)
    coeff = 0.5 * abar * inv_bsq
    term = term - div_a_grad_perp(
        coeff,
        vedotgradpi,
        jacobian=ctx.geom.jacobian,
        dx=ctx.geom.metric_dx,
        dy=ctx.geom.metric_dy,
        dz=ctx.geom.metric_dz,
        g11=ctx.geom.gxx,
        g23=ctx.geom.g23,
        g_22=ctx.geom.g_22,
        g_23=ctx.geom.g_23,
        g33=ctx.geom.gyy,
        bxy=ctx.geom.B,
        z_shift=ctx.geom.z_shift,
        zlength=_mirror_zlength(ctx.geom),
        bc_kind_x=_phi_bcx_for_delp(ctx.params, bc.phi),
        bc_value_x=getattr(bc.phi, "x_value", 0.0),
        bc_grad_x=getattr(bc.phi, "x_grad", 0.0),
        coeff_bc_kind_x=2,
        coeff_bc_value_x=0.0,
        coeff_bc_grad_x=0.0,
        interp=str(getattr(ctx.params, "parallel_shift_interp", "spectral")),
        periodic_parallel=not bool(
            getattr(getattr(ctx.geom, "grid", None), "open_field_line", False)
        ),
        periodic_binormal=int(getattr(bc.phi, "kind_y", 0)) == 0,
        lower_boundary_open=bool(
            getattr(getattr(ctx.geom, "grid", None), "open_field_line", False)
        ),
        upper_boundary_open=bool(
            getattr(getattr(ctx.geom, "grid", None), "open_field_line", False)
        ),
        apply_free_o2=True,
    )

    bc_delp = bc.phi
    if bool(getattr(ctx.params, "poisson_invert_set", False)) and bc.phi.kind_x != 0:
        bc_delp = BC2D(
            kind_x=1,
            kind_y=bc.phi.kind_y,
            x_value=0.0,
            y_value=bc.phi.y_value,
            x_grad=0.0,
            y_grad=bc.phi.y_grad,
        )
    delp_phi = delp2_runtime(phi, geom=ctx.geom, bc_field=bc_delp)
    delp_phi_2b2 = 0.5 * abar * delp_phi * inv_bsq
    term = term - _runtime_exb_term(
        delp_phi_2b2,
        phi + pihat,
        geom=ctx.geom,
        params=ctx.params,
        bc_phi=bc.phi,
        bc_adv=bc.omega,
        adv_free_o2=True,
    )
    return term * scale
