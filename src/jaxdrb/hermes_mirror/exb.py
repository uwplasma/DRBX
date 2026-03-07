"""Hermes ExB mirror operators.

Planned in Phase 3 of `/Users/rogerio/local/jax_drb/plan.md`.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.bc import BC2D

from .boundary import apply_neumann_field3d
from .primitives import Stencil1D, mc_limiter
from .species import (
    prepare_poloidal_x_dfdy_local,
    prepare_poloidal_x_dfdy_local_ref,
    prepare_poloidal_y_dfdx_local,
    prepare_poloidal_y_dfdx_local_ref,
)
from .transform import (
    build_shifted_metric_fft_phases,
    build_shifted_metric_weights,
    from_field_aligned_all,
    from_field_aligned_all_fft,
    to_field_aligned_all,
    to_field_aligned_all_fft,
)
from .types import FieldAlignedLocalLayout


def _shift(arr: jnp.ndarray, offset: int, axis: int, *, periodic: bool) -> jnp.ndarray:
    if periodic:
        return jnp.roll(arr, int(offset), axis=axis)
    n = arr.shape[axis]
    idx = jnp.clip(jnp.arange(n) + int(offset), 0, n - 1)
    return jnp.take(arr, idx, axis=axis)


def _x_ghost(
    arr: jnp.ndarray,
    *,
    side: str,
    kind: int,
    value: float,
    grad: float,
    dx: jnp.ndarray,
    neumann_boundary_average_z: bool,
    order: int = 1,
) -> jnp.ndarray:
    if side == "left":
        edge = arr[:, 0, :]
        near = arr[:, 1, :] if arr.shape[1] > 1 else edge
        dx_edge = dx[:, 0, :]
        if neumann_boundary_average_z and kind == 2:
            edge = jnp.mean(edge, axis=-1, keepdims=True)
            edge = jnp.broadcast_to(edge, arr[:, 0, :].shape)
        if kind == 1:
            return (2.0 * float(value) - edge) if order == 1 else (2.0 * float(value) - near)
        if kind == 2:
            return (
                (edge - float(grad) * dx_edge)
                if order == 1
                else (near - 3.0 * float(grad) * dx_edge)
            )
        return arr[:, -1, :]
    edge = arr[:, -1, :]
    near = arr[:, -2, :] if arr.shape[1] > 1 else edge
    dx_edge = dx[:, -1, :]
    if neumann_boundary_average_z and kind == 2:
        edge = jnp.mean(edge, axis=-1, keepdims=True)
        edge = jnp.broadcast_to(edge, arr[:, -1, :].shape)
    if kind == 1:
        return (2.0 * float(value) - edge) if order == 1 else (2.0 * float(value) - near)
    if kind == 2:
        return (
            (edge + float(grad) * dx_edge) if order == 1 else (near + 3.0 * float(grad) * dx_edge)
        )
    return arr[:, 0, :]


def _fromm_face(
    arr: jnp.ndarray,
    vel: jnp.ndarray,
    *,
    axis: int,
    periodic: bool,
    positive: bool = False,
) -> jnp.ndarray:
    a_i = arr
    a_ip1 = _shift(arr, +1, axis, periodic=periodic)
    a_im1 = _shift(arr, -1, axis, periodic=periodic)
    a_ip2 = _shift(arr, +2, axis, periodic=periodic)
    upwind_pos = a_i + 0.25 * (a_ip1 - a_im1)
    upwind_neg = a_ip1 - 0.25 * (a_ip2 - a_i)
    face_val = jnp.where(vel > 0.0, upwind_pos, upwind_neg)
    return jnp.maximum(face_val, 0.0) if positive else face_val


def _mc_lr(
    arr: jnp.ndarray,
    *,
    axis: int,
    periodic: bool,
    periodic_x: bool,
    bc_kind_x: int,
    bc_value_x: float,
    bc_grad_x: float,
    dx: jnp.ndarray,
    neumann_boundary_average_z: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if axis == 1 and (not periodic):
        left_ghost = _x_ghost(
            arr,
            side="left",
            kind=bc_kind_x,
            value=bc_value_x,
            grad=bc_grad_x,
            dx=dx,
            neumann_boundary_average_z=neumann_boundary_average_z,
            order=1,
        )
        right_ghost = _x_ghost(
            arr,
            side="right",
            kind=bc_kind_x,
            value=bc_value_x,
            grad=bc_grad_x,
            dx=dx,
            neumann_boundary_average_z=neumann_boundary_average_z,
            order=1,
        )
        a_c = arr
        a_m = jnp.concatenate([left_ghost[:, None, :], arr[:, :-1, :]], axis=1)
        a_p = jnp.concatenate([arr[:, 1:, :], right_ghost[:, None, :]], axis=1)
    else:
        a_c = arr
        a_m = _shift(arr, -1, axis=axis, periodic=periodic)
        a_p = _shift(arr, +1, axis=axis, periodic=periodic)
    stencil = mc_limiter(Stencil1D(c=a_c, m=a_m, p=a_p))
    return stencil.L, stencil.R


def _fromm_x_boundary_flux(
    adv_arr: jnp.ndarray,
    vel: jnp.ndarray,
    *,
    side: str,
    kind: int,
    value: float,
    grad: float,
    dx: jnp.ndarray,
    positive: bool,
    neumann_boundary_average_z: bool,
) -> jnp.ndarray:
    if side == "left":
        a0 = adv_arr[:, 0, :]
        a1 = adv_arr[:, 1, :] if adv_arr.shape[1] > 1 else a0
        ag1 = _x_ghost(
            adv_arr,
            side="left",
            kind=kind,
            value=value,
            grad=grad,
            dx=dx,
            neumann_boundary_average_z=neumann_boundary_average_z,
            order=1,
        )
        ag2 = _x_ghost(
            adv_arr,
            side="left",
            kind=kind,
            value=value,
            grad=grad,
            dx=dx,
            neumann_boundary_average_z=neumann_boundary_average_z,
            order=2,
        )
        outflow = a0 - 0.25 * (a1 - ag1)
        inflow = ag1 + 0.25 * (a0 - ag2)
        face_val = jnp.where(vel < 0.0, outflow, inflow)
    else:
        a0 = adv_arr[:, -1, :]
        am1 = adv_arr[:, -2, :] if adv_arr.shape[1] > 1 else a0
        ag1 = _x_ghost(
            adv_arr,
            side="right",
            kind=kind,
            value=value,
            grad=grad,
            dx=dx,
            neumann_boundary_average_z=neumann_boundary_average_z,
            order=1,
        )
        ag2 = _x_ghost(
            adv_arr,
            side="right",
            kind=kind,
            value=value,
            grad=grad,
            dx=dx,
            neumann_boundary_average_z=neumann_boundary_average_z,
            order=2,
        )
        outflow = a0 + 0.25 * (ag1 - am1)
        inflow = ag1 - 0.25 * (ag2 - a0)
        face_val = jnp.where(vel > 0.0, outflow, inflow)
    face_val = jnp.maximum(face_val, 0.0) if positive else face_val
    return vel * face_val


def _as_field_aligned_metric2d(
    arr: jnp.ndarray | float,
    *,
    npar: int,
    nx: int,
    nbinorm: int,
    name: str,
) -> jnp.ndarray:
    out = jnp.asarray(arr, dtype=jnp.float64)
    if out.ndim == 0:
        return jnp.full((npar, nx), out, dtype=jnp.float64)
    if out.ndim == 1:
        if out.shape[0] == npar:
            return jnp.broadcast_to(out[:, None], (npar, nx))
        if out.shape[0] == nx:
            return jnp.broadcast_to(out[None, :], (npar, nx))
    if out.ndim == 2 and out.shape == (npar, nx):
        return out
    if out.ndim == 2 and out.shape == (npar, 1):
        return jnp.broadcast_to(out, (npar, nx))
    if out.ndim == 2 and out.shape == (1, nx):
        return jnp.broadcast_to(out, (npar, nx))
    if out.ndim == 3 and out.shape == (npar, nx, nbinorm):
        return out[..., 0]
    raise ValueError(
        f"{name} must be scalar, shape ({npar}, {nx}), or shape ({npar}, {nx}, {nbinorm}); got {out.shape}."
    )


def _to_field_aligned_local(
    field: jnp.ndarray,
    *,
    z_shift: jnp.ndarray | float,
    zlength: float,
    open_field_line: bool,
    interp: str,
) -> jnp.ndarray:
    field_arr = jnp.asarray(field, dtype=jnp.float64)
    npar, nx, nbinorm = (int(v) for v in field_arr.shape)
    interp_name = str(interp).lower()
    if interp_name == "spectral":
        phases = build_shifted_metric_fft_phases(
            z_shift,
            nx=nx,
            npar=npar,
            nbinorm=nbinorm,
            zlength=float(zlength),
            open_field_line=open_field_line,
        )
        return to_field_aligned_all_fft(field_arr, phases)
    if interp_name == "linear":
        dz = float(zlength) / max(nbinorm, 1)
        if dz <= 0.0:
            raise ValueError(f"zlength={zlength} gives invalid binormal spacing {dz}.")
        shift_idx = jnp.asarray(z_shift, dtype=jnp.float64) / dz
        weights = build_shifted_metric_weights(
            shift_idx,
            nx=nx,
            npar=npar,
            nbinorm=nbinorm,
            open_field_line=open_field_line,
        )
        return to_field_aligned_all(field_arr, weights)
    raise ValueError(f"Unsupported interp={interp!r}; expected 'spectral' or 'linear'.")


def _from_field_aligned_local(
    field: jnp.ndarray,
    *,
    z_shift: jnp.ndarray | float,
    zlength: float,
    open_field_line: bool,
    interp: str,
) -> jnp.ndarray:
    field_arr = jnp.asarray(field, dtype=jnp.float64)
    npar, nx, nbinorm = (int(v) for v in field_arr.shape)
    interp_name = str(interp).lower()
    if interp_name == "spectral":
        phases = build_shifted_metric_fft_phases(
            z_shift,
            nx=nx,
            npar=npar,
            nbinorm=nbinorm,
            zlength=float(zlength),
            open_field_line=open_field_line,
        )
        return from_field_aligned_all_fft(field_arr, phases)
    if interp_name == "linear":
        dz = float(zlength) / max(nbinorm, 1)
        if dz <= 0.0:
            raise ValueError(f"zlength={zlength} gives invalid binormal spacing {dz}.")
        shift_idx = jnp.asarray(z_shift, dtype=jnp.float64) / dz
        weights = build_shifted_metric_weights(
            shift_idx,
            nx=nx,
            npar=npar,
            nbinorm=nbinorm,
            open_field_line=open_field_line,
        )
        return from_field_aligned_all(field_arr, weights)
    raise ValueError(f"Unsupported interp={interp!r}; expected 'spectral' or 'linear'.")


def div_n_bxgrad_f_b_xppm_xz(
    n: jnp.ndarray,
    f: jnp.ndarray,
    *,
    jacobian: jnp.ndarray,
    dx: jnp.ndarray,
    dz: jnp.ndarray,
    periodic_x: bool,
    periodic_z: bool,
    bndry_flux: bool,
    use_mc: bool,
    bc_kind_x: int = 0,
    bc_value_x: float = 0.0,
    bc_grad_x: float = 0.0,
    positive: bool = False,
    neumann_boundary_average_z: bool = False,
) -> jnp.ndarray:
    """Fused X-Z branch of Hermes `Div_n_bxGrad_f_B_XPPM`."""

    n_arr = jnp.asarray(n, dtype=jnp.float64)
    f_arr = jnp.asarray(f, dtype=jnp.float64)
    npar, nx, nbinorm = (int(v) for v in n_arr.shape)
    J = jnp.broadcast_to(
        _as_field_aligned_metric2d(jacobian, npar=npar, nx=nx, nbinorm=nbinorm, name="jacobian")[
            :, :, None
        ],
        n_arr.shape,
    )
    dx_arr = jnp.broadcast_to(
        _as_field_aligned_metric2d(dx, npar=npar, nx=nx, nbinorm=nbinorm, name="dx")[:, :, None],
        n_arr.shape,
    )
    dz_arr = jnp.broadcast_to(
        _as_field_aligned_metric2d(dz, npar=npar, nx=nx, nbinorm=nbinorm, name="dz")[:, :, None],
        n_arr.shape,
    )

    if periodic_x:
        f_xm = _shift(f_arr, -1, axis=1, periodic=True)
        f_xp = _shift(f_arr, +1, axis=1, periodic=True)
        J_xm = _shift(J, -1, axis=1, periodic=True)
        J_xp = _shift(J, +1, axis=1, periodic=True)
    else:
        phi_left_ghost = _x_ghost(
            f_arr,
            side="left",
            kind=bc_kind_x,
            value=bc_value_x,
            grad=bc_grad_x,
            dx=dx_arr,
            neumann_boundary_average_z=neumann_boundary_average_z,
            order=1,
        )
        phi_right_ghost = _x_ghost(
            f_arr,
            side="right",
            kind=bc_kind_x,
            value=bc_value_x,
            grad=bc_grad_x,
            dx=dx_arr,
            neumann_boundary_average_z=neumann_boundary_average_z,
            order=1,
        )
        f_xm = jnp.concatenate([phi_left_ghost[:, None, :], f_arr[:, :-1, :]], axis=1)
        f_xp = jnp.concatenate([f_arr[:, 1:, :], phi_right_ghost[:, None, :]], axis=1)
        if J.shape[1] > 1:
            J_left_ghost = 2.0 * J[:, 0, :] - J[:, 1, :]
            J_right_ghost = 2.0 * J[:, -1, :] - J[:, -2, :]
        else:
            J_left_ghost = J[:, 0, :]
            J_right_ghost = J[:, -1, :]
        J_xm = jnp.concatenate([J_left_ghost[:, None, :], J[:, :-1, :]], axis=1)
        J_xp = jnp.concatenate([J[:, 1:, :], J_right_ghost[:, None, :]], axis=1)

    f_zm = _shift(f_arr, -1, axis=2, periodic=periodic_z)
    f_zp = _shift(f_arr, +1, axis=2, periodic=periodic_z)
    f_xm_zm = _shift(f_xm, -1, axis=2, periodic=periodic_z)
    f_xm_zp = _shift(f_xm, +1, axis=2, periodic=periodic_z)
    f_xp_zm = _shift(f_xp, -1, axis=2, periodic=periodic_z)
    f_xp_zp = _shift(f_xp, +1, axis=2, periodic=periodic_z)

    fmm = 0.25 * (f_arr + f_xm + f_zm + f_xm_zm)
    fmp = 0.25 * (f_arr + f_xm + f_zp + f_xm_zp)
    fpp = 0.25 * (f_arr + f_xp + f_zp + f_xp_zp)
    fpm = 0.25 * (f_arr + f_xp + f_zm + f_xp_zm)

    v_u = J * (fmp - fpp) / jnp.maximum(dx_arr, 1e-30)
    v_r = 0.5 * (J + J_xp) * (fpp - fpm) / jnp.maximum(dz_arr, 1e-30)

    if use_mc:
        left_x, right_x = _mc_lr(
            n_arr,
            axis=1,
            periodic=periodic_x,
            periodic_x=periodic_x,
            bc_kind_x=bc_kind_x,
            bc_value_x=bc_value_x,
            bc_grad_x=bc_grad_x,
            dx=dx_arr,
            neumann_boundary_average_z=neumann_boundary_average_z,
        )
        right_state = jnp.maximum(right_x, 0.0) if positive else right_x
        left_state_next = _shift(left_x, +1, axis=1, periodic=periodic_x)
        left_state_next = jnp.maximum(left_state_next, 0.0) if positive else left_state_next
        flux_r = jnp.where(v_r > 0.0, v_r * right_state, v_r * left_state_next)

        if periodic_x:
            flux_l = _shift(flux_r, -1, axis=1, periodic=True)
        else:
            v_l = 0.5 * (J + J_xm) * (fmp - fmm) / jnp.maximum(dz_arr, 1e-30)
            left_ghost = _x_ghost(
                n_arr,
                side="left",
                kind=bc_kind_x,
                value=bc_value_x,
                grad=bc_grad_x,
                dx=dx_arr,
                neumann_boundary_average_z=neumann_boundary_average_z,
                order=1,
            )
            right_ghost = _x_ghost(
                n_arr,
                side="right",
                kind=bc_kind_x,
                value=bc_value_x,
                grad=bc_grad_x,
                dx=dx_arr,
                neumann_boundary_average_z=neumann_boundary_average_z,
                order=1,
            )
            left_in = 0.5 * (left_ghost + n_arr[:, 0, :])
            right_in = 0.5 * (right_ghost + n_arr[:, -1, :])
            left_out = left_x[:, 0, :]
            right_out = right_x[:, -1, :]
            if positive:
                left_in = jnp.maximum(left_in, 0.0)
                right_in = jnp.maximum(right_in, 0.0)
                left_out = jnp.maximum(left_out, 0.0)
                right_out = jnp.maximum(right_out, 0.0)
            flux_left_b = jnp.where(
                v_l[:, 0, :] < 0.0, v_l[:, 0, :] * left_out, v_l[:, 0, :] * left_in
            )
            flux_right_b = jnp.where(
                v_r[:, -1, :] > 0.0, v_r[:, -1, :] * right_out, v_r[:, -1, :] * right_in
            )
            if not bndry_flux:
                flux_left_b = jnp.where(v_l[:, 0, :] < 0.0, v_l[:, 0, :] * left_out, 0.0)
                flux_right_b = jnp.where(v_r[:, -1, :] > 0.0, v_r[:, -1, :] * right_out, 0.0)
            flux_r = flux_r.at[:, -1, :].set(flux_right_b)
            flux_l = jnp.concatenate([flux_left_b[:, None, :], flux_r[:, :-1, :]], axis=1)

        left_z, right_z = _mc_lr(
            n_arr,
            axis=2,
            periodic=periodic_z,
            periodic_x=True,
            bc_kind_x=0,
            bc_value_x=0.0,
            bc_grad_x=0.0,
            dx=dx_arr,
            neumann_boundary_average_z=False,
        )
        up_state = right_z
        down_state = _shift(left_z, +1, axis=2, periodic=periodic_z)
        if positive:
            up_state = jnp.maximum(up_state, 0.0)
            down_state = jnp.maximum(down_state, 0.0)
        flux_u = jnp.where(v_u > 0.0, v_u * up_state, v_u * down_state)
        flux_d = _shift(flux_u, -1, axis=2, periodic=periodic_z)
    else:
        n_face_r = _fromm_face(n_arr, v_r, axis=1, periodic=periodic_x, positive=positive)
        flux_r = v_r * n_face_r
        if periodic_x:
            flux_l = _shift(flux_r, -1, axis=1, periodic=True)
        else:
            v_l = 0.5 * (J + J_xm) * (fmp - fmm) / jnp.maximum(dz_arr, 1e-30)
            flux_left_b = _fromm_x_boundary_flux(
                n_arr,
                v_l[:, 0, :],
                side="left",
                kind=bc_kind_x,
                value=bc_value_x,
                grad=bc_grad_x,
                dx=dx_arr,
                positive=positive,
                neumann_boundary_average_z=neumann_boundary_average_z,
            )
            flux_right_b = _fromm_x_boundary_flux(
                n_arr,
                v_r[:, -1, :],
                side="right",
                kind=bc_kind_x,
                value=bc_value_x,
                grad=bc_grad_x,
                dx=dx_arr,
                positive=positive,
                neumann_boundary_average_z=neumann_boundary_average_z,
            )
            if not bndry_flux:
                right_out = n_arr[:, -1, :] + 0.25 * (
                    _x_ghost(
                        n_arr,
                        side="right",
                        kind=bc_kind_x,
                        value=bc_value_x,
                        grad=bc_grad_x,
                        dx=dx_arr,
                        neumann_boundary_average_z=neumann_boundary_average_z,
                        order=1,
                    )
                    - n_arr[:, -2, :]
                )
                left_out = n_arr[:, 0, :] - 0.25 * (
                    n_arr[:, 1, :]
                    - _x_ghost(
                        n_arr,
                        side="left",
                        kind=bc_kind_x,
                        value=bc_value_x,
                        grad=bc_grad_x,
                        dx=dx_arr,
                        neumann_boundary_average_z=neumann_boundary_average_z,
                        order=1,
                    )
                )
                if positive:
                    right_out = jnp.maximum(right_out, 0.0)
                    left_out = jnp.maximum(left_out, 0.0)
                flux_left_b = jnp.where(v_l[:, 0, :] < 0.0, v_l[:, 0, :] * left_out, 0.0)
                flux_right_b = jnp.where(v_r[:, -1, :] > 0.0, v_r[:, -1, :] * right_out, 0.0)
            flux_r = flux_r.at[:, -1, :].set(flux_right_b)
            flux_l = jnp.concatenate([flux_left_b[:, None, :], flux_r[:, :-1, :]], axis=1)

        n_face_u = _fromm_face(n_arr, v_u, axis=2, periodic=periodic_z, positive=positive)
        flux_u = v_u * n_face_u
        flux_d = _shift(flux_u, -1, axis=2, periodic=periodic_z)

    return (flux_r - flux_l) / (jnp.maximum(J, 1e-30) * jnp.maximum(dx_arr, 1e-30)) + (
        flux_u - flux_d
    ) / (jnp.maximum(J, 1e-30) * jnp.maximum(dz_arr, 1e-30))


def div_n_bxgrad_f_b_xppm_xz_ref(
    n: jnp.ndarray,
    f: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    """Reference X-Z mirror operator.

    The current reference implementation intentionally shares the same
    differentiable helper logic as the fused path but remains a stable,
    separately named oracle for Phase 3 tests before the full ExB operator is
    assembled.
    """

    return div_n_bxgrad_f_b_xppm_xz(n, f, **kwargs)


def div_n_bxgrad_f_b_xppm_xy_x_local_ref(
    n: jnp.ndarray,
    dfdy: jnp.ndarray,
    *,
    jacobian: jnp.ndarray | float,
    dx: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    layout: FieldAlignedLocalLayout,
    bndry_flux: bool,
    positive: bool = False,
) -> jnp.ndarray:
    """Reference local X-flux branch of Hermes `Div_n_bxGrad_f_B_XPPM`."""

    n_arr = np.asarray(n, dtype=np.float64)
    dfdy_arr = np.asarray(dfdy, dtype=np.float64)
    if n_arr.shape != dfdy_arr.shape:
        raise ValueError(f"n shape {n_arr.shape} != dfdy shape {dfdy_arr.shape}.")
    layout.validate(tuple(int(v) for v in n_arr.shape))
    npar, nx, nbinorm = (int(v) for v in n_arr.shape)

    J = np.asarray(
        _as_field_aligned_metric2d(
            jnp.asarray(jacobian), npar=npar, nx=nx, nbinorm=nbinorm, name="jacobian"
        )
    )
    dx_arr = np.asarray(
        _as_field_aligned_metric2d(jnp.asarray(dx), npar=npar, nx=nx, nbinorm=nbinorm, name="dx")
    )
    g11_arr = np.asarray(
        _as_field_aligned_metric2d(jnp.asarray(g11), npar=npar, nx=nx, nbinorm=nbinorm, name="g11")
    )
    g23_arr = np.asarray(
        _as_field_aligned_metric2d(jnp.asarray(g23), npar=npar, nx=nx, nbinorm=nbinorm, name="g23")
    )
    bxy_arr = np.asarray(
        _as_field_aligned_metric2d(jnp.asarray(bxy), npar=npar, nx=nx, nbinorm=nbinorm, name="bxy")
    )
    coeff = g11_arr * g23_arr / np.maximum(bxy_arr * bxy_arr, 1e-30)

    xs = layout.xstart - 1
    xe = layout.xend
    if not bndry_flux:
        xs = layout.xstart
        xe = layout.xend - 1

    out = np.zeros_like(n_arr)
    for i in range(xs, xe + 1):
        for j in range(layout.pstart - 1, layout.pend + 1):
            for k in range(nbinorm):
                f_r = 0.5 * (
                    (coeff[j, i + 1] * dfdy_arr[j, i + 1, k]) + (coeff[j, i] * dfdy_arr[j, i, k])
                )
                vx = 0.5 * (J[j, i + 1] + J[j, i]) * f_r

                if vx > 0.0:
                    nval = n_arr[j, i, k] + 0.25 * (n_arr[j, i + 1, k] - n_arr[j, i - 1, k])
                else:
                    nval = n_arr[j, i + 1, k] - 0.25 * (n_arr[j, i + 2, k] - n_arr[j, i, k])
                if positive and (nval < 0.0):
                    nval = 0.0
                flux = vx * nval
                out[j, i, k] += flux / (dx_arr[j, i] * J[j, i])
                out[j, i + 1, k] -= flux / (dx_arr[j, i + 1] * J[j, i + 1])

    return jnp.asarray(out, dtype=jnp.float64)


def div_n_bxgrad_f_b_xppm_xy_x_local(
    n: jnp.ndarray,
    dfdy: jnp.ndarray,
    *,
    jacobian: jnp.ndarray | float,
    dx: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    layout: FieldAlignedLocalLayout,
    bndry_flux: bool,
    positive: bool = False,
) -> jnp.ndarray:
    """Fused local X-flux branch of Hermes `Div_n_bxGrad_f_B_XPPM`."""

    n_arr = jnp.asarray(n, dtype=jnp.float64)
    dfdy_arr = jnp.asarray(dfdy, dtype=jnp.float64)
    if n_arr.shape != dfdy_arr.shape:
        raise ValueError(f"n shape {n_arr.shape} != dfdy shape {dfdy_arr.shape}.")
    layout.validate(tuple(int(v) for v in n_arr.shape))
    npar, nx, nbinorm = (int(v) for v in n_arr.shape)

    J2d = _as_field_aligned_metric2d(jacobian, npar=npar, nx=nx, nbinorm=nbinorm, name="jacobian")
    dx2d = _as_field_aligned_metric2d(dx, npar=npar, nx=nx, nbinorm=nbinorm, name="dx")
    g11_2d = _as_field_aligned_metric2d(g11, npar=npar, nx=nx, nbinorm=nbinorm, name="g11")
    g23_2d = _as_field_aligned_metric2d(g23, npar=npar, nx=nx, nbinorm=nbinorm, name="g23")
    bxy_2d = _as_field_aligned_metric2d(bxy, npar=npar, nx=nx, nbinorm=nbinorm, name="bxy")

    coeff = g11_2d * g23_2d / jnp.maximum(bxy_2d * bxy_2d, 1e-30)
    coeff_dfdy = coeff[:, :, None] * dfdy_arr
    vx = 0.25 * (J2d[:, 1:] + J2d[:, :-1])[:, :, None] * (coeff_dfdy[:, 1:] + coeff_dfdy[:, :-1])

    face_vx = vx[:, 1:-1, :]
    n_im1 = n_arr[:, :-3, :]
    n_i = n_arr[:, 1:-2, :]
    n_ip1 = n_arr[:, 2:-1, :]
    n_ip2 = n_arr[:, 3:, :]
    pos_state = n_i + 0.25 * (n_ip1 - n_im1)
    neg_state = n_ip1 - 0.25 * (n_ip2 - n_i)
    face_state = jnp.where(face_vx > 0.0, pos_state, neg_state)
    if positive:
        face_state = jnp.maximum(face_state, 0.0)
    flux = face_vx * face_state

    xs = layout.xstart - 1
    xe = layout.xend
    if not bndry_flux:
        xs = layout.xstart
        xe = layout.xend - 1
    face_idx = jnp.arange(1, nx - 2)
    face_mask = (face_idx >= int(xs)) & (face_idx <= int(xe))
    p_idx = jnp.arange(npar)
    p_mask = (p_idx >= int(layout.pstart - 1)) & (p_idx <= int(layout.pend))
    flux = jnp.where(p_mask[:, None, None] & face_mask[None, :, None], flux, 0.0)

    out = jnp.zeros_like(n_arr)
    out = out.at[:, 1:-2, :].add(
        flux / jnp.maximum((dx2d[:, 1:-2] * J2d[:, 1:-2])[:, :, None], 1e-30)
    )
    out = out.at[:, 2:-1, :].add(
        -flux / jnp.maximum((dx2d[:, 2:-1] * J2d[:, 2:-1])[:, :, None], 1e-30)
    )
    return out


def div_n_bxgrad_f_b_xppm_xy_x_local_from_fields_ref(
    n: jnp.ndarray,
    f: jnp.ndarray,
    *,
    dy: jnp.ndarray | float,
    dx: jnp.ndarray | float,
    z_shift: jnp.ndarray | float,
    zlength: float,
    jacobian: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    layout: FieldAlignedLocalLayout,
    bndry_flux: bool,
    interp: str = "spectral",
    positive: bool = False,
    lower_boundary_open: bool = True,
    upper_boundary_open: bool = False,
    periodic_parallel: bool = False,
) -> jnp.ndarray:
    """Reference local X-flux branch starting from unaligned local fields."""

    dfdy = prepare_poloidal_x_dfdy_local_ref(
        f,
        dy=dy,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        layout=layout,
        interp=interp,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
        periodic_parallel=periodic_parallel,
    )
    return div_n_bxgrad_f_b_xppm_xy_x_local_ref(
        n,
        dfdy,
        jacobian=jacobian,
        dx=dx,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=bndry_flux,
        positive=positive,
    )


def div_n_bxgrad_f_b_xppm_xy_x_local_from_fields(
    n: jnp.ndarray,
    f: jnp.ndarray,
    *,
    dy: jnp.ndarray | float,
    dx: jnp.ndarray | float,
    z_shift: jnp.ndarray | float,
    zlength: float,
    jacobian: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    layout: FieldAlignedLocalLayout,
    bndry_flux: bool,
    interp: str = "spectral",
    positive: bool = False,
    lower_boundary_open: bool = True,
    upper_boundary_open: bool = False,
    periodic_parallel: bool = False,
) -> jnp.ndarray:
    """Fused local X-flux branch starting from unaligned local fields."""

    dfdy = prepare_poloidal_x_dfdy_local(
        f,
        dy=dy,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        layout=layout,
        interp=interp,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
        periodic_parallel=periodic_parallel,
    )
    return div_n_bxgrad_f_b_xppm_xy_x_local(
        n,
        dfdy,
        jacobian=jacobian,
        dx=dx,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=bndry_flux,
        positive=positive,
    )


def div_n_bxgrad_f_b_xppm_local_ref(
    n: jnp.ndarray,
    f: jnp.ndarray,
    *,
    jacobian: jnp.ndarray | float,
    dx: jnp.ndarray | float,
    dy: jnp.ndarray | float,
    dz: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    z_shift: jnp.ndarray | float,
    zlength: float,
    layout: FieldAlignedLocalLayout,
    bndry_flux: bool,
    poloidal: bool,
    positive: bool = False,
    interp: str = "spectral",
    bc_kind_x: int = 2,
    bc_value_x: float = 0.0,
    bc_grad_x: float = 0.0,
    neumann_boundary_average_z: bool = True,
    use_mc: bool = True,
    periodic_parallel: bool = False,
    periodic_binormal: bool = True,
    lower_boundary_open: bool = True,
    upper_boundary_open: bool = False,
    poloidal_scale: float = 1.0,
    poloidal_x_scale: float = 1.0,
    poloidal_y_scale: float = 1.0,
) -> jnp.ndarray:
    """Reference local full mirror of Hermes `Div_n_bxGrad_f_B_XPPM`."""

    xz = div_n_bxgrad_f_b_xppm_xz_ref(
        n,
        f,
        jacobian=jacobian,
        dx=dx,
        dz=dz,
        periodic_x=False,
        periodic_z=periodic_binormal,
        bndry_flux=bndry_flux,
        use_mc=use_mc,
        bc_kind_x=bc_kind_x,
        bc_value_x=bc_value_x,
        bc_grad_x=bc_grad_x,
        positive=positive,
        neumann_boundary_average_z=neumann_boundary_average_z,
    )
    if not poloidal:
        return xz

    x_flux = div_n_bxgrad_f_b_xppm_xy_x_local_from_fields_ref(
        n,
        f,
        dy=dy,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        jacobian=jacobian,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=bndry_flux,
        interp=interp,
        positive=positive,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
        periodic_parallel=periodic_parallel,
    )
    y_fa = div_n_bxgrad_f_b_xppm_xy_y_local_from_fields_ref(
        n,
        f,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        jacobian=jacobian,
        dy=dy,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        interp=interp,
        bndry_flux=bndry_flux,
        positive=positive,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
        periodic_parallel=periodic_parallel,
    )
    y_flux = _from_field_aligned_local(
        y_fa,
        z_shift=z_shift,
        zlength=zlength,
        open_field_line=layout.open_field_line,
        interp=interp,
    )
    return xz + float(poloidal_scale) * (
        float(poloidal_x_scale) * x_flux + float(poloidal_y_scale) * y_flux
    )


def div_n_bxgrad_f_b_xppm_local(
    n: jnp.ndarray,
    f: jnp.ndarray,
    *,
    jacobian: jnp.ndarray | float,
    dx: jnp.ndarray | float,
    dy: jnp.ndarray | float,
    dz: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    z_shift: jnp.ndarray | float,
    zlength: float,
    layout: FieldAlignedLocalLayout,
    bndry_flux: bool,
    poloidal: bool,
    positive: bool = False,
    interp: str = "spectral",
    bc_kind_x: int = 2,
    bc_value_x: float = 0.0,
    bc_grad_x: float = 0.0,
    neumann_boundary_average_z: bool = True,
    use_mc: bool = True,
    periodic_parallel: bool = False,
    periodic_binormal: bool = True,
    lower_boundary_open: bool = True,
    upper_boundary_open: bool = False,
    poloidal_scale: float = 1.0,
    poloidal_x_scale: float = 1.0,
    poloidal_y_scale: float = 1.0,
) -> jnp.ndarray:
    """Fused local full mirror of Hermes `Div_n_bxGrad_f_B_XPPM`."""

    xz = div_n_bxgrad_f_b_xppm_xz(
        n,
        f,
        jacobian=jacobian,
        dx=dx,
        dz=dz,
        periodic_x=False,
        periodic_z=periodic_binormal,
        bndry_flux=bndry_flux,
        use_mc=use_mc,
        bc_kind_x=bc_kind_x,
        bc_value_x=bc_value_x,
        bc_grad_x=bc_grad_x,
        positive=positive,
        neumann_boundary_average_z=neumann_boundary_average_z,
    )
    if not poloidal:
        return xz

    x_flux = div_n_bxgrad_f_b_xppm_xy_x_local_from_fields(
        n,
        f,
        dy=dy,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        jacobian=jacobian,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=bndry_flux,
        interp=interp,
        positive=positive,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
        periodic_parallel=periodic_parallel,
    )
    y_fa = div_n_bxgrad_f_b_xppm_xy_y_local_from_fields(
        n,
        f,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        jacobian=jacobian,
        dy=dy,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        interp=interp,
        bndry_flux=bndry_flux,
        positive=positive,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
        periodic_parallel=periodic_parallel,
    )
    y_flux = _from_field_aligned_local(
        y_fa,
        z_shift=z_shift,
        zlength=zlength,
        open_field_line=layout.open_field_line,
        interp=interp,
    )
    return xz + float(poloidal_scale) * (
        float(poloidal_x_scale) * x_flux + float(poloidal_y_scale) * y_flux
    )


def _boundary_side_values(value: object) -> tuple[jnp.ndarray, jnp.ndarray]:
    if isinstance(value, tuple | list):
        if len(value) != 2:
            raise ValueError(f"Boundary side value must have length 2, got {len(value)}.")
        lower, upper = value
    else:
        lower = upper = value
    return jnp.asarray(lower, dtype=jnp.float64), jnp.asarray(upper, dtype=jnp.float64)


def _as_runtime_metric2d(
    arr: jnp.ndarray | float,
    *,
    nz: int,
    nx: int,
    ny: int,
    name: str,
) -> jnp.ndarray:
    out = jnp.asarray(arr, dtype=jnp.float64)
    if out.ndim == 0:
        return jnp.full((nz, nx), out, dtype=jnp.float64)
    if out.ndim == 1:
        if out.shape[0] == nz:
            return jnp.broadcast_to(out[:, None], (nz, nx))
        if out.shape[0] == nx:
            return jnp.broadcast_to(out[None, :], (nz, nx))
    if out.ndim == 2:
        if out.shape == (nz, nx):
            return out
        if out.shape == (nx, nz):
            return jnp.swapaxes(out, 0, 1)
        if out.shape == (nz, 1):
            return jnp.broadcast_to(out, (nz, nx))
        if out.shape == (1, nx):
            return jnp.broadcast_to(out, (nz, nx))
        if out.shape == (nx, ny):
            return jnp.broadcast_to(out[None, :, 0], (nz, nx))
    if out.ndim == 3 and out.shape == (nz, nx, ny):
        return out[:, :, 0]
    raise ValueError(
        f"{name} must be scalar, ({nz}, {nx}), ({nz}, {nx}, {ny}), or broadcastable to ({nz}, {nx}); got {out.shape}."
    )


def _pad_x_runtime(
    field: jnp.ndarray,
    *,
    bc_kind_x: int,
    bc_value_x: object,
    bc_grad_x: object,
    dx: jnp.ndarray,
) -> jnp.ndarray:
    field_arr = jnp.asarray(field, dtype=jnp.float64)
    edge_l = field_arr[:, :1, :]
    edge_r = field_arr[:, -1:, :]
    if int(bc_kind_x) == 0:
        return jnp.concatenate(
            [field_arr[:, -2:, :], field_arr, field_arr[:, :2, :]],
            axis=1,
        )

    out = jnp.concatenate([edge_l, edge_l, field_arr, edge_r, edge_r], axis=1)
    lower_val, upper_val = _boundary_side_values(bc_value_x)
    lower_grad, upper_grad = _boundary_side_values(bc_grad_x)

    if int(bc_kind_x) == 1:
        left_edge = out[:, 2, :]
        left_near = out[:, 3, :]
        right_edge = out[:, -3, :]
        right_near = out[:, -4, :]
        out = out.at[:, 1, :].set((2.0 * lower_val) - left_edge)
        out = out.at[:, 0, :].set((2.0 * lower_val) - left_near)
        out = out.at[:, -2, :].set((2.0 * upper_val) - right_edge)
        out = out.at[:, -1, :].set((2.0 * upper_val) - right_near)
        return out

    dx_pad = jnp.concatenate([dx[:, :1], dx[:, :1], dx, dx[:, -1:], dx[:, -1:]], axis=1)
    return apply_neumann_field3d(
        out,
        axis=1,
        interior_start=2,
        interior_end=field_arr.shape[1] + 1,
        spacing=dx_pad[:, :, None],
        lower_gradient=lower_grad,
        upper_gradient=upper_grad,
        guard_width=2,
    )


def _pad_parallel_runtime(
    field: jnp.ndarray,
    *,
    dy: jnp.ndarray,
    periodic_parallel: bool,
    lower_boundary_open: bool,
    upper_boundary_open: bool,
) -> jnp.ndarray:
    field_arr = jnp.asarray(field, dtype=jnp.float64)
    if periodic_parallel:
        return jnp.concatenate([field_arr[-2:, :, :], field_arr, field_arr[:2, :, :]], axis=0)
    lower = field_arr[:1, :, :]
    upper = field_arr[-1:, :, :]
    out = jnp.concatenate([lower, lower, field_arr, upper, upper], axis=0)
    dy_pad = jnp.concatenate([dy[:1, :], dy[:1, :], dy, dy[-1:, :], dy[-1:, :]], axis=0)
    return apply_neumann_field3d(
        out,
        axis=0,
        interior_start=2,
        interior_end=field_arr.shape[0] + 1,
        spacing=dy_pad[:, :, None],
        lower_gradient=0.0,
        upper_gradient=0.0,
        guard_width=2,
        apply_lower=lower_boundary_open,
        apply_upper=upper_boundary_open,
    )


def _pad_runtime_field(
    field: jnp.ndarray,
    *,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    bc_kind_x: int,
    bc_value_x: object,
    bc_grad_x: object,
    periodic_parallel: bool,
    lower_boundary_open: bool,
    upper_boundary_open: bool,
) -> jnp.ndarray:
    out = _pad_x_runtime(
        field,
        bc_kind_x=bc_kind_x,
        bc_value_x=bc_value_x,
        bc_grad_x=bc_grad_x,
        dx=dx,
    )
    return _pad_parallel_runtime(
        out,
        dy=jnp.concatenate([dy[:, :1], dy[:, :1], dy, dy[:, -1:], dy[:, -1:]], axis=1),
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )


def _pad_runtime_metric(
    metric: jnp.ndarray,
    *,
    periodic_x: bool,
    periodic_parallel: bool,
    lower_boundary_open: bool,
    upper_boundary_open: bool,
) -> jnp.ndarray:
    metric_arr = jnp.asarray(metric, dtype=jnp.float64)
    if periodic_x:
        x_padded = jnp.concatenate([metric_arr[:, -2:], metric_arr, metric_arr[:, :2]], axis=1)
    else:
        edge_l = metric_arr[:, :1]
        edge_r = metric_arr[:, -1:]
        x_padded = jnp.concatenate([edge_l, edge_l, metric_arr, edge_r, edge_r], axis=1)
        x_padded = apply_neumann_field3d(
            x_padded[:, :, None],
            axis=1,
            interior_start=2,
            interior_end=metric_arr.shape[1] + 1,
            spacing=1.0,
            lower_gradient=0.0,
            upper_gradient=0.0,
            guard_width=2,
        )[:, :, 0]
    if periodic_parallel:
        return jnp.concatenate([x_padded[-2:, :], x_padded, x_padded[:2, :]], axis=0)
    lower = x_padded[:1, :]
    upper = x_padded[-1:, :]
    out = jnp.concatenate([lower, lower, x_padded, upper, upper], axis=0)
    return apply_neumann_field3d(
        out[:, :, None],
        axis=0,
        interior_start=2,
        interior_end=metric_arr.shape[0] + 1,
        spacing=1.0,
        lower_gradient=0.0,
        upper_gradient=0.0,
        guard_width=2,
        apply_lower=lower_boundary_open,
        apply_upper=upper_boundary_open,
    )[:, :, 0]


def _runtime_local_edge_block(
    field: jnp.ndarray,
    phi: jnp.ndarray,
    *,
    start: int,
    block: int,
    dx2d: jnp.ndarray,
    dy2d: jnp.ndarray,
    dz2d: jnp.ndarray,
    J2d: jnp.ndarray,
    g11_2d: jnp.ndarray,
    g23_2d: jnp.ndarray,
    bxy_2d: jnp.ndarray,
    zshift_2d: jnp.ndarray,
    zlength: float,
    bc_kind_x: int,
    bc_value_x: object,
    bc_grad_x: object,
    phi_kind_x: int,
    phi_value_x: object,
    phi_grad_x: object,
    bndry_flux: bool,
    poloidal: bool,
    positive: bool,
    interp: str,
    neumann_boundary_average_z: bool,
    use_mc: bool,
    periodic_parallel: bool,
    periodic_binormal: bool,
    lower_boundary_open: bool,
    upper_boundary_open: bool,
    poloidal_scale: float,
    poloidal_x_scale: float,
    poloidal_y_scale: float,
) -> jnp.ndarray:
    stop = int(start + block)
    field_block = jnp.asarray(field[start:stop], dtype=jnp.float64)
    phi_block = jnp.asarray(phi[start:stop], dtype=jnp.float64)
    dx_block = jnp.asarray(dx2d[start:stop], dtype=jnp.float64)
    dy_block = jnp.asarray(dy2d[start:stop], dtype=jnp.float64)
    dz_block = jnp.asarray(dz2d[start:stop], dtype=jnp.float64)
    J_block = jnp.asarray(J2d[start:stop], dtype=jnp.float64)
    g11_block = jnp.asarray(g11_2d[start:stop], dtype=jnp.float64)
    g23_block = jnp.asarray(g23_2d[start:stop], dtype=jnp.float64)
    bxy_block = jnp.asarray(bxy_2d[start:stop], dtype=jnp.float64)
    zshift_block = jnp.asarray(zshift_2d[start:stop], dtype=jnp.float64)

    apply_lower = bool(lower_boundary_open and start == 0)
    apply_upper = bool(upper_boundary_open and stop == field.shape[0])

    field_local = _pad_runtime_field(
        field_block,
        dx=dx_block,
        dy=dy_block,
        bc_kind_x=bc_kind_x,
        bc_value_x=bc_value_x,
        bc_grad_x=bc_grad_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=apply_lower,
        upper_boundary_open=apply_upper,
    )
    phi_local = _pad_runtime_field(
        phi_block,
        dx=dx_block,
        dy=dy_block,
        bc_kind_x=phi_kind_x,
        bc_value_x=phi_value_x,
        bc_grad_x=phi_grad_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=apply_lower,
        upper_boundary_open=apply_upper,
    )
    J_local = _pad_runtime_metric(
        J_block,
        periodic_x=bc_kind_x == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=apply_lower,
        upper_boundary_open=apply_upper,
    )
    dx_local = _pad_runtime_metric(
        dx_block,
        periodic_x=bc_kind_x == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=apply_lower,
        upper_boundary_open=apply_upper,
    )
    dy_local = _pad_runtime_metric(
        dy_block,
        periodic_x=bc_kind_x == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=apply_lower,
        upper_boundary_open=apply_upper,
    )
    dz_local = _pad_runtime_metric(
        dz_block,
        periodic_x=bc_kind_x == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=apply_lower,
        upper_boundary_open=apply_upper,
    )
    g11_local = _pad_runtime_metric(
        g11_block,
        periodic_x=bc_kind_x == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=apply_lower,
        upper_boundary_open=apply_upper,
    )
    g23_local = _pad_runtime_metric(
        g23_block,
        periodic_x=bc_kind_x == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=apply_lower,
        upper_boundary_open=apply_upper,
    )
    bxy_local = _pad_runtime_metric(
        bxy_block,
        periodic_x=bc_kind_x == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=apply_lower,
        upper_boundary_open=apply_upper,
    )
    zshift_local = _pad_runtime_metric(
        zshift_block,
        periodic_x=bc_kind_x == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=apply_lower,
        upper_boundary_open=apply_upper,
    )
    layout = FieldAlignedLocalLayout(
        pstart=2,
        pend=block + 1,
        xstart=2,
        xend=field.shape[1] + 1,
        open_field_line=not periodic_parallel,
    )
    result_local = div_n_bxgrad_f_b_xppm_local(
        field_local,
        phi_local,
        jacobian=J_local,
        dx=dx_local,
        dy=dy_local,
        dz=dz_local,
        g11=g11_local,
        g23=g23_local,
        bxy=bxy_local,
        z_shift=zshift_local,
        zlength=zlength,
        layout=layout,
        bndry_flux=bndry_flux,
        poloidal=poloidal,
        positive=positive,
        interp=interp,
        bc_kind_x=bc_kind_x,
        bc_value_x=float(jnp.asarray(bc_value_x if np.isscalar(bc_value_x) else 0.0)),
        bc_grad_x=float(jnp.asarray(bc_grad_x if np.isscalar(bc_grad_x) else 0.0)),
        neumann_boundary_average_z=neumann_boundary_average_z,
        use_mc=use_mc,
        periodic_parallel=periodic_parallel,
        periodic_binormal=periodic_binormal,
        lower_boundary_open=apply_lower,
        upper_boundary_open=apply_upper,
        poloidal_scale=poloidal_scale,
        poloidal_x_scale=poloidal_x_scale,
        poloidal_y_scale=poloidal_y_scale,
    )
    return result_local[2 : block + 2, 2 : field.shape[1] + 2, :]


def div_n_bxgrad_f_b_xppm(
    n: jnp.ndarray,
    f: jnp.ndarray,
    *,
    jacobian: jnp.ndarray | float,
    dx: jnp.ndarray | float,
    dy: jnp.ndarray | float,
    dz: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    z_shift: jnp.ndarray | float,
    zlength: float,
    bc_phi: BC2D,
    bc_adv: BC2D,
    bndry_flux: bool,
    poloidal: bool,
    positive: bool = False,
    interp: str = "spectral",
    neumann_boundary_average_z: bool = True,
    use_mc: bool = True,
    periodic_parallel: bool = False,
    periodic_binormal: bool = True,
    lower_boundary_open: bool = True,
    upper_boundary_open: bool = True,
    poisson_invert_set: bool = False,
    parallel_edge_block: int = 0,
    poloidal_scale: float = 1.0,
    poloidal_x_scale: float = 1.0,
    poloidal_y_scale: float = 1.0,
) -> jnp.ndarray:
    """Runtime mirror wrapper from global `(nz, nx, ny)` arrays.

    This reconstructs the guard-inclusive local BOUT/Hermes storage contract,
    runs the validated local mirror operator, and slices the physical interior
    cells back out for the active JAX field-aligned geometry path.
    """

    n_arr = jnp.asarray(n, dtype=jnp.float64)
    f_arr = jnp.asarray(f, dtype=jnp.float64)
    if n_arr.shape != f_arr.shape or n_arr.ndim != 3:
        raise ValueError(
            f"Runtime mirror ExB expects matching `(nz, nx, ny)` fields, got {n_arr.shape} and {f_arr.shape}."
        )
    nz, nx, ny = (int(v) for v in n_arr.shape)
    layout = FieldAlignedLocalLayout(
        pstart=2,
        pend=nz + 1,
        xstart=2,
        xend=nx + 1,
        open_field_line=not periodic_parallel,
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

    periodic_x = int(getattr(bc_adv, "kind_x", 0)) == 0
    phi_kind_x = int(getattr(bc_phi, "kind_x", 0))
    if poisson_invert_set and phi_kind_x != 0:
        phi_kind_x = 1

    n_local = _pad_runtime_field(
        n_arr,
        dx=dx2d,
        dy=dy2d,
        bc_kind_x=int(getattr(bc_adv, "kind_x", 0)),
        bc_value_x=getattr(bc_adv, "x_value", 0.0),
        bc_grad_x=getattr(bc_adv, "x_grad", 0.0),
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
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
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    dx_local = _pad_runtime_metric(
        dx2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    dy_local = _pad_runtime_metric(
        dy2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    dz_local = _pad_runtime_metric(
        dz2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g11_local = _pad_runtime_metric(
        g11_2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g23_local = _pad_runtime_metric(
        g23_2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    bxy_local = _pad_runtime_metric(
        bxy_2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    zshift_local = _pad_runtime_metric(
        zshift_2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )

    result_local = div_n_bxgrad_f_b_xppm_local(
        n_local,
        f_local,
        jacobian=J_local,
        dx=dx_local,
        dy=dy_local,
        dz=dz_local,
        g11=g11_local,
        g23=g23_local,
        bxy=bxy_local,
        z_shift=zshift_local,
        zlength=zlength,
        layout=layout,
        bndry_flux=bndry_flux,
        poloidal=poloidal,
        positive=positive,
        interp=interp,
        bc_kind_x=int(getattr(bc_adv, "kind_x", 0)),
        bc_value_x=float(getattr(bc_adv, "x_value", 0.0)),
        bc_grad_x=float(getattr(bc_adv, "x_grad", 0.0)),
        neumann_boundary_average_z=neumann_boundary_average_z,
        use_mc=use_mc,
        periodic_parallel=periodic_parallel,
        periodic_binormal=periodic_binormal,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
        poloidal_scale=poloidal_scale,
        poloidal_x_scale=poloidal_x_scale,
        poloidal_y_scale=poloidal_y_scale,
    )
    result = result_local[interior]

    edge_block = int(parallel_edge_block)
    if edge_block > 0 and (not periodic_parallel) and (2 * edge_block) < nz and edge_block <= nz:
        result = result.at[:edge_block].set(
            _runtime_local_edge_block(
                n_arr,
                f_arr,
                start=0,
                block=edge_block,
                dx2d=dx2d,
                dy2d=dy2d,
                dz2d=dz2d,
                J2d=J2d,
                g11_2d=g11_2d,
                g23_2d=g23_2d,
                bxy_2d=bxy_2d,
                zshift_2d=zshift_2d,
                zlength=zlength,
                bc_kind_x=int(getattr(bc_adv, "kind_x", 0)),
                bc_value_x=getattr(bc_adv, "x_value", 0.0),
                bc_grad_x=getattr(bc_adv, "x_grad", 0.0),
                phi_kind_x=phi_kind_x,
                phi_value_x=getattr(bc_phi, "x_value", 0.0),
                phi_grad_x=getattr(bc_phi, "x_grad", 0.0),
                bndry_flux=bndry_flux,
                poloidal=poloidal,
                positive=positive,
                interp=interp,
                neumann_boundary_average_z=neumann_boundary_average_z,
                use_mc=use_mc,
                periodic_parallel=periodic_parallel,
                periodic_binormal=periodic_binormal,
                lower_boundary_open=lower_boundary_open,
                upper_boundary_open=upper_boundary_open,
                poloidal_scale=poloidal_scale,
                poloidal_x_scale=poloidal_x_scale,
                poloidal_y_scale=poloidal_y_scale,
            )
        )
        result = result.at[nz - edge_block :].set(
            _runtime_local_edge_block(
                n_arr,
                f_arr,
                start=nz - edge_block,
                block=edge_block,
                dx2d=dx2d,
                dy2d=dy2d,
                dz2d=dz2d,
                J2d=J2d,
                g11_2d=g11_2d,
                g23_2d=g23_2d,
                bxy_2d=bxy_2d,
                zshift_2d=zshift_2d,
                zlength=zlength,
                bc_kind_x=int(getattr(bc_adv, "kind_x", 0)),
                bc_value_x=getattr(bc_adv, "x_value", 0.0),
                bc_grad_x=getattr(bc_adv, "x_grad", 0.0),
                phi_kind_x=phi_kind_x,
                phi_value_x=getattr(bc_phi, "x_value", 0.0),
                phi_grad_x=getattr(bc_phi, "x_grad", 0.0),
                bndry_flux=bndry_flux,
                poloidal=poloidal,
                positive=positive,
                interp=interp,
                neumann_boundary_average_z=neumann_boundary_average_z,
                use_mc=use_mc,
                periodic_parallel=periodic_parallel,
                periodic_binormal=periodic_binormal,
                lower_boundary_open=lower_boundary_open,
                upper_boundary_open=upper_boundary_open,
                poloidal_scale=poloidal_scale,
                poloidal_x_scale=poloidal_x_scale,
                poloidal_y_scale=poloidal_y_scale,
            )
        )
    return result


def div_n_bxgrad_f_b_xppm_xy_y_local_ref(
    n_fa: jnp.ndarray,
    dfdx_fa: jnp.ndarray,
    *,
    jacobian: jnp.ndarray | float,
    dy: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    layout: FieldAlignedLocalLayout,
    bndry_flux: bool,
    positive: bool = False,
    lower_boundary_open: bool = True,
    upper_boundary_open: bool = False,
    periodic_parallel: bool = False,
) -> jnp.ndarray:
    """Reference local Y-flux branch of Hermes `Div_n_bxGrad_f_B_XPPM`."""

    n_arr = np.asarray(n_fa, dtype=np.float64)
    dfdx_arr = np.asarray(dfdx_fa, dtype=np.float64)
    if n_arr.shape != dfdx_arr.shape:
        raise ValueError(f"n_fa shape {n_arr.shape} != dfdx_fa shape {dfdx_arr.shape}.")
    layout.validate(tuple(int(v) for v in n_arr.shape))
    npar, nx, nbinorm = (int(v) for v in n_arr.shape)

    J = np.asarray(
        _as_field_aligned_metric2d(
            jnp.asarray(jacobian), npar=npar, nx=nx, nbinorm=nbinorm, name="jacobian"
        )
    )
    dy_arr = np.asarray(
        _as_field_aligned_metric2d(jnp.asarray(dy), npar=npar, nx=nx, nbinorm=nbinorm, name="dy")
    )
    g11_arr = np.asarray(
        _as_field_aligned_metric2d(jnp.asarray(g11), npar=npar, nx=nx, nbinorm=nbinorm, name="g11")
    )
    g23_arr = np.asarray(
        _as_field_aligned_metric2d(jnp.asarray(g23), npar=npar, nx=nx, nbinorm=nbinorm, name="g23")
    )
    bxy_arr = np.asarray(
        _as_field_aligned_metric2d(jnp.asarray(bxy), npar=npar, nx=nx, nbinorm=nbinorm, name="bxy")
    )
    coeff = g11_arr * g23_arr / np.maximum(bxy_arr * bxy_arr, 1e-30)

    ys = layout.pstart - 1
    ye = layout.pend
    if (not bndry_flux) and (not periodic_parallel):
        if lower_boundary_open:
            ys = layout.pstart
        if upper_boundary_open:
            ye = layout.pend - 1

    out = np.zeros_like(n_arr)
    for i in range(layout.xstart, layout.xend + 1):
        for j in range(ys, ye + 1):
            for k in range(nbinorm):
                f_u = 0.5 * (
                    (coeff[j + 1, i] * dfdx_arr[j + 1, i, k]) + (coeff[j, i] * dfdx_arr[j, i, k])
                )
                vy = -0.5 * (J[j + 1, i] + J[j, i]) * f_u

                if lower_boundary_open and (not periodic_parallel) and (j == layout.pstart - 1):
                    vy = min(vy, 0.0)
                if upper_boundary_open and (not periodic_parallel) and (j == layout.pend):
                    vy = max(vy, 0.0)

                if vy > 0.0:
                    nval = n_arr[j, i, k] + 0.25 * (n_arr[j + 1, i, k] - n_arr[j - 1, i, k])
                else:
                    nval = n_arr[j + 1, i, k] - 0.25 * (n_arr[j + 2, i, k] - n_arr[j, i, k])
                if positive and (nval < 0.0):
                    nval = 0.0
                flux = vy * nval
                out[j, i, k] += flux / (dy_arr[j, i] * J[j, i])
                out[j + 1, i, k] -= flux / (dy_arr[j + 1, i] * J[j + 1, i])

    return jnp.asarray(out, dtype=jnp.float64)


def div_n_bxgrad_f_b_xppm_xy_y_local(
    n_fa: jnp.ndarray,
    dfdx_fa: jnp.ndarray,
    *,
    jacobian: jnp.ndarray | float,
    dy: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    layout: FieldAlignedLocalLayout,
    bndry_flux: bool,
    positive: bool = False,
    lower_boundary_open: bool = True,
    upper_boundary_open: bool = False,
    periodic_parallel: bool = False,
) -> jnp.ndarray:
    """Fused local Y-flux branch of Hermes `Div_n_bxGrad_f_B_XPPM`."""

    n_arr = jnp.asarray(n_fa, dtype=jnp.float64)
    dfdx_arr = jnp.asarray(dfdx_fa, dtype=jnp.float64)
    if n_arr.shape != dfdx_arr.shape:
        raise ValueError(f"n_fa shape {n_arr.shape} != dfdx_fa shape {dfdx_arr.shape}.")
    layout.validate(tuple(int(v) for v in n_arr.shape))
    npar, nx, nbinorm = (int(v) for v in n_arr.shape)

    J2d = _as_field_aligned_metric2d(jacobian, npar=npar, nx=nx, nbinorm=nbinorm, name="jacobian")
    dy2d = _as_field_aligned_metric2d(dy, npar=npar, nx=nx, nbinorm=nbinorm, name="dy")
    g11_2d = _as_field_aligned_metric2d(g11, npar=npar, nx=nx, nbinorm=nbinorm, name="g11")
    g23_2d = _as_field_aligned_metric2d(g23, npar=npar, nx=nx, nbinorm=nbinorm, name="g23")
    bxy_2d = _as_field_aligned_metric2d(bxy, npar=npar, nx=nx, nbinorm=nbinorm, name="bxy")

    J = jnp.broadcast_to(J2d[:, :, None], n_arr.shape)
    dy_arr = jnp.broadcast_to(dy2d[:, :, None], n_arr.shape)
    coeff = jnp.broadcast_to(
        (g11_2d * g23_2d / jnp.maximum(bxy_2d * bxy_2d, 1e-30))[:, :, None],
        n_arr.shape,
    )
    coeff_dfdx = coeff * dfdx_arr
    vy = -0.25 * (J[1:] + J[:-1]) * (coeff_dfdx[1:] + coeff_dfdx[:-1])

    if lower_boundary_open and (not periodic_parallel):
        vy = vy.at[layout.pstart - 1].set(jnp.minimum(vy[layout.pstart - 1], 0.0))
    if upper_boundary_open and (not periodic_parallel):
        vy = vy.at[layout.pend].set(jnp.maximum(vy[layout.pend], 0.0))

    n_m = _shift(n_arr, -1, axis=0, periodic=periodic_parallel)[:-1]
    n_c = n_arr[:-1]
    n_p = n_arr[1:]
    n_pp = _shift(n_arr, +2, axis=0, periodic=periodic_parallel)[:-1]
    pos_state = n_c + 0.25 * (n_p - n_m)
    neg_state = n_p - 0.25 * (n_pp - n_c)
    face_state = jnp.where(vy > 0.0, pos_state, neg_state)
    if positive:
        face_state = jnp.maximum(face_state, 0.0)
    flux = vy * face_state

    ys = layout.pstart - 1
    ye = layout.pend
    if (not bndry_flux) and (not periodic_parallel):
        if lower_boundary_open:
            ys = layout.pstart
        if upper_boundary_open:
            ye = layout.pend - 1
    face_idx = jnp.arange(npar - 1)
    face_mask = (face_idx >= int(ys)) & (face_idx <= int(ye))
    x_idx = jnp.arange(nx)
    x_mask = (x_idx >= int(layout.xstart)) & (x_idx <= int(layout.xend))
    flux = jnp.where(face_mask[:, None, None] & x_mask[None, :, None], flux, 0.0)

    out = jnp.zeros_like(n_arr)
    out = out.at[:-1].add(flux / jnp.maximum(dy_arr[:-1] * J[:-1], 1e-30))
    out = out.at[1:].add(-flux / jnp.maximum(dy_arr[1:] * J[1:], 1e-30))
    return out


def div_n_bxgrad_f_b_xppm_xy_y_local_from_fields_ref(
    n: jnp.ndarray,
    f: jnp.ndarray,
    *,
    dx: jnp.ndarray | float,
    z_shift: jnp.ndarray | float,
    zlength: float,
    jacobian: jnp.ndarray | float,
    dy: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    layout: FieldAlignedLocalLayout,
    interp: str = "spectral",
    bndry_flux: bool = True,
    positive: bool = False,
    lower_boundary_open: bool = True,
    upper_boundary_open: bool = False,
    periodic_parallel: bool = False,
) -> jnp.ndarray:
    """Reference local Y-flux branch starting from unaligned local fields."""

    dfdx_fa = prepare_poloidal_y_dfdx_local_ref(
        f,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        layout=layout,
        interp=interp,
    )
    n_fa = _to_field_aligned_local(
        n,
        z_shift=z_shift,
        zlength=zlength,
        open_field_line=layout.open_field_line,
        interp=interp,
    )
    return div_n_bxgrad_f_b_xppm_xy_y_local_ref(
        n_fa,
        dfdx_fa,
        jacobian=jacobian,
        dy=dy,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=bndry_flux,
        positive=positive,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
        periodic_parallel=periodic_parallel,
    )


def div_n_bxgrad_f_b_xppm_xy_y_local_from_fields(
    n: jnp.ndarray,
    f: jnp.ndarray,
    *,
    dx: jnp.ndarray | float,
    z_shift: jnp.ndarray | float,
    zlength: float,
    jacobian: jnp.ndarray | float,
    dy: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    layout: FieldAlignedLocalLayout,
    interp: str = "spectral",
    bndry_flux: bool = True,
    positive: bool = False,
    lower_boundary_open: bool = True,
    upper_boundary_open: bool = False,
    periodic_parallel: bool = False,
) -> jnp.ndarray:
    """Fused local Y-flux branch starting from unaligned local fields."""

    dfdx_fa = prepare_poloidal_y_dfdx_local(
        f,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        layout=layout,
        interp=interp,
    )
    n_fa = _to_field_aligned_local(
        n,
        z_shift=z_shift,
        zlength=zlength,
        open_field_line=layout.open_field_line,
        interp=interp,
    )
    return div_n_bxgrad_f_b_xppm_xy_y_local(
        n_fa,
        dfdx_fa,
        jacobian=jacobian,
        dy=dy,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=bndry_flux,
        positive=positive,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
        periodic_parallel=periodic_parallel,
    )
