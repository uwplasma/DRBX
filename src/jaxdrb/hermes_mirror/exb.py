"""Hermes ExB mirror operators.

Planned in Phase 3 of `/Users/rogerio/local/jax_drb/plan.md`.
"""

from __future__ import annotations

import jax.numpy as jnp

from .primitives import Stencil1D, mc_limiter


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
    J = jnp.broadcast_to(jnp.asarray(jacobian, dtype=jnp.float64), n_arr.shape)
    dx_arr = jnp.broadcast_to(jnp.asarray(dx, dtype=jnp.float64), n_arr.shape)
    dz_arr = jnp.broadcast_to(jnp.asarray(dz, dtype=jnp.float64), n_arr.shape)

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


def div_n_bxgrad_f_b_xppm(*args, **kwargs):
    raise NotImplementedError("Phase 3: mirror ExB transport is not landed yet.")
