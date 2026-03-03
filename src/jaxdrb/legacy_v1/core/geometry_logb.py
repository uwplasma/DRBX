from __future__ import annotations

import jax.numpy as jnp


def salpha_logb_coefficients(
    theta: jnp.ndarray,
    *,
    epsilon: float,
    q: float,
    shat: float,
    R0: float,
    r0: float | None,
    theta_scale: float,
    B0: float | None = None,
    epsilon_x_grad: float | None = None,
    theta_ballooning_on: bool = False,
    theta_ballooning_r: float | None = None,
    linear_shear_on: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute log-B curvature coefficients for s-alpha geometry.

    Returns (curv_x, curv_y, dpar_factor, B).
    """
    B_const = 1.0 if B0 is None else float(B0)
    B = jnp.ones_like(theta) * B_const

    r0_val = float(r0) if r0 is not None else float(epsilon) * float(R0)
    r0_val = max(r0_val, 1e-8)

    q_eff = float(q)
    if theta_ballooning_on:
        r_ref = r0_val if theta_ballooning_r is None else float(theta_ballooning_r)
        q_eff = float(q) + float(shat) * (r_ref - r0_val) / r0_val
        q_eff = max(q_eff, 1e-8)

    theta_eff = theta * (float(q) / max(q_eff, 1e-8))

    dlogB_dtheta = float(epsilon) * jnp.sin(theta_eff)
    scale = max(float(theta_scale), 1e-8)
    dtheta_dz = (float(q) / max(q_eff, 1e-8)) / scale
    dlogB_dz = dlogB_dtheta * dtheta_dz

    if epsilon_x_grad is None:
        eps_grad = 1.0 / max(float(R0), 1e-8)
    else:
        eps_grad = float(epsilon_x_grad)

    dtheta_dx = 0.0
    if linear_shear_on:
        dq_dx = float(shat) / r0_val
        dtheta_dx = -(theta_eff / max(q_eff, 1e-8)) * dq_dx
    dlogB_dx = -eps_grad * jnp.cos(theta_eff) + dlogB_dtheta * dtheta_dx

    curv_x = -B * dlogB_dz
    curv_y = B * dlogB_dx

    dpar_factor = jnp.ones_like(theta) * (scale / max(float(q) * float(R0), 1e-8))
    return curv_x, curv_y, dpar_factor, B
