"""Literal Hermes vorticity pieces for strict parity work."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from jaxdrb.bc import BC2D

from .boundary import apply_free_o2_field3d
from .exb import (
    _as_runtime_metric2d,
    _pad_runtime_field,
    _pad_runtime_metric,
    div_n_bxgrad_f_b_xppm_local,
)
from .fv import div_a_grad_perp
from .types import FieldAlignedLocalLayout

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
    metric_dz = getattr(geom, "metric_dz", None)
    grid = getattr(geom, "grid", None)
    perp = getattr(grid, "perp", None)
    if metric_dz is None or perp is None:
        raise ValueError("Hermes mirror vorticity path requires metric_dz and field-aligned grid.")
    return float(jnp.asarray(metric_dz, dtype=jnp.float64).mean()) * float(perp.ny)


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
    nz, nx, ny = (int(v) for v in adv_arr.shape)

    jacobian = getattr(geom, "jacobian", None)
    dx = getattr(geom, "metric_dx", None)
    dy = getattr(geom, "metric_dy", None)
    dz = getattr(geom, "metric_dz", None)
    g11 = getattr(geom, "gxx", None)
    g23 = getattr(geom, "g23", None)
    bxy = getattr(geom, "B", None)
    z_shift = getattr(geom, "z_shift", None)
    if any(val is None for val in (jacobian, dx, dy, dz, g11, g23, bxy, z_shift)):
        raise ValueError(
            "Hermes mirror vorticity ExB path requires Jacobian, metrics, Bxy, and z_shift."
        )

    layout = FieldAlignedLocalLayout(
        pstart=2,
        pend=nz + 1,
        xstart=2,
        xend=nx + 1,
        open_field_line=bool(getattr(getattr(geom, "grid", None), "open_field_line", False)),
    )
    interior = (
        slice(layout.pstart, layout.pend + 1),
        slice(layout.xstart, layout.xend + 1),
        slice(None),
    )

    dx2d = _as_runtime_metric2d(dx, nz=nz, nx=nx, ny=ny, name="dx")
    dy2d = _as_runtime_metric2d(dy, nz=nz, nx=nx, ny=ny, name="dy")
    dz2d = _as_runtime_metric2d(dz, nz=nz, nx=nx, ny=ny, name="dz")
    J2d = _as_runtime_metric2d(jacobian, nz=nz, nx=nx, ny=ny, name="jacobian")
    g11_2d = _as_runtime_metric2d(g11, nz=nz, nx=nx, ny=ny, name="g11")
    g23_2d = _as_runtime_metric2d(g23, nz=nz, nx=nx, ny=ny, name="g23")
    bxy_2d = _as_runtime_metric2d(bxy, nz=nz, nx=nx, ny=ny, name="bxy")
    zshift_2d = _as_runtime_metric2d(z_shift, nz=nz, nx=nx, ny=ny, name="z_shift")

    periodic_parallel = not bool(getattr(getattr(geom, "grid", None), "open_field_line", False))
    lower_boundary_open = bool(getattr(getattr(geom, "grid", None), "open_field_line", False))
    upper_boundary_open = lower_boundary_open

    phi_kind_x = _phi_bcx_for_delp(params, bc_phi)
    adv_local = _pad_runtime_field(
        adv_arr,
        dx=dx2d,
        dy=dy2d,
        bc_kind_x=int(getattr(bc_adv, "kind_x", 0)),
        bc_value_x=getattr(bc_adv, "x_value", 0.0),
        bc_grad_x=getattr(bc_adv, "x_grad", 0.0),
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    if adv_free_o2:
        adv_local = apply_free_o2_field3d(
            adv_local,
            axis=1,
            interior_start=layout.xstart,
            interior_end=layout.xend,
            guard_width=layout.x_guards,
        )
        if not periodic_parallel:
            adv_local = apply_free_o2_field3d(
                adv_local,
                axis=0,
                interior_start=layout.pstart,
                interior_end=layout.pend,
                guard_width=layout.p_guards,
                apply_lower=lower_boundary_open,
                apply_upper=upper_boundary_open,
            )

    f_local = _pad_runtime_field(
        f_arr,
        dx=dx2d,
        dy=dy2d,
        bc_kind_x=phi_kind_x,
        bc_value_x=getattr(bc_phi, "x_value", 0.0),
        bc_grad_x=getattr(bc_phi, "x_grad", 0.0),
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    J_local = _pad_runtime_metric(
        J2d,
        periodic_x=int(getattr(bc_adv, "kind_x", 0)) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    dx_local = _pad_runtime_metric(
        dx2d,
        periodic_x=int(getattr(bc_adv, "kind_x", 0)) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    dy_local = _pad_runtime_metric(
        dy2d,
        periodic_x=int(getattr(bc_adv, "kind_x", 0)) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    dz_local = _pad_runtime_metric(
        dz2d,
        periodic_x=int(getattr(bc_adv, "kind_x", 0)) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g11_local = _pad_runtime_metric(
        g11_2d,
        periodic_x=int(getattr(bc_adv, "kind_x", 0)) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g23_local = _pad_runtime_metric(
        g23_2d,
        periodic_x=int(getattr(bc_adv, "kind_x", 0)) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    bxy_local = _pad_runtime_metric(
        bxy_2d,
        periodic_x=int(getattr(bc_adv, "kind_x", 0)) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    zshift_local = _pad_runtime_metric(
        zshift_2d,
        periodic_x=int(getattr(bc_adv, "kind_x", 0)) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )

    result_local = div_n_bxgrad_f_b_xppm_local(
        adv_local,
        f_local,
        jacobian=J_local,
        dx=dx_local,
        dy=dy_local,
        dz=dz_local,
        g11=g11_local,
        g23=g23_local,
        bxy=bxy_local,
        z_shift=zshift_local,
        zlength=_mirror_zlength(geom),
        layout=layout,
        bndry_flux=bool(getattr(params, "exb_bndry_flux", True)),
        poloidal=bool(getattr(params, "exb_poloidal_flows", False)),
        positive=False,
        interp=str(getattr(params, "parallel_shift_interp", "spectral")),
        bc_kind_x=int(getattr(bc_adv, "kind_x", 0)),
        bc_value_x=float(getattr(bc_adv, "x_value", 0.0)),
        bc_grad_x=float(getattr(bc_adv, "x_grad", 0.0)),
        neumann_boundary_average_z=bool(getattr(params, "neumann_boundary_average_y", False)),
        use_mc=True,
        periodic_parallel=periodic_parallel,
        periodic_binormal=int(getattr(bc_adv, "kind_y", 0)) == 0,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
        poloidal_scale=float(getattr(params, "exb_poloidal_scale", 1.0)),
        poloidal_x_scale=float(getattr(params, "exb_poloidal_x_scale", 1.0)),
        poloidal_y_scale=float(getattr(params, "exb_poloidal_y_scale", 1.0)),
    )
    return result_local[interior]


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

    from jaxdrb.core.terms.fields import _metric_div_coeff

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
    delp_phi = _metric_div_coeff(
        ctx.params,
        ctx.geom,
        phi,
        jnp.ones_like(phi),
        bc_delp,
    )
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
