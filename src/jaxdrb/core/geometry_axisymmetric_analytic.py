from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import jax
import jax.numpy as jnp

from jaxdrb.core.geometry_logb import salpha_logb_coefficients

from jaxdrb.core.geometry_field_aligned import FieldAlignedGeometryAdapter, FieldAlignedGrid
from jaxdrb.core.region_bcs import parse_region_bcs
from jaxdrb.core.params import DRBSystemParams


@dataclass(frozen=True)
class AxisymmetricAnalyticSpec:
    model: str
    nz: int
    Lz: float
    open_field_line: bool
    theta_scale: float | None = None


def _theta_grid(spec: AxisymmetricAnalyticSpec) -> tuple[jnp.ndarray, float, float]:
    if spec.open_field_line:
        z = jnp.linspace(-0.5 * spec.Lz, 0.5 * spec.Lz, spec.nz, endpoint=True)
        dz = float(spec.Lz / max(spec.nz - 1, 1))
    else:
        z = jnp.linspace(-0.5 * spec.Lz, 0.5 * spec.Lz, spec.nz, endpoint=False)
        dz = float(spec.Lz / max(spec.nz, 1))
    theta_scale = spec.theta_scale
    if theta_scale is None or float(theta_scale) <= 0.0:
        theta_scale = float(z[-1] - z[0]) / (2.0 * jnp.pi)
        theta_scale = max(theta_scale, 1e-8)
    theta = z / float(theta_scale)
    return theta, dz, float(theta_scale)


def _dd_theta(f: jnp.ndarray, dtheta: float, *, periodic: bool) -> jnp.ndarray:
    if periodic:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dtheta)
    df = jnp.zeros_like(f)
    df = df.at[1:-1].set((f[2:] - f[:-2]) / (2.0 * dtheta))
    df = df.at[0].set((f[1] - f[0]) / dtheta)
    df = df.at[-1].set((f[-1] - f[-2]) / dtheta)
    return df


def _cross(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return jnp.stack([cx, cy, cz], axis=-1)


def _apply_sheath_windows(
    theta: jnp.ndarray,
    *,
    windows: Iterable[tuple[float, float]] | None,
    sign: Iterable[float] | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if not windows:
        return jnp.zeros_like(theta), jnp.zeros_like(theta)
    mask = jnp.zeros_like(theta)
    sign_arr = jnp.zeros_like(theta)
    signs = list(sign) if sign is not None else [1.0] * len(list(windows))
    for (theta_min, theta_max), sgn in zip(windows, signs):
        in_window = (theta >= theta_min) & (theta <= theta_max)
        mask = jnp.where(in_window, 1.0, mask)
        sign_arr = jnp.where(in_window, float(sgn), sign_arr)
    return mask, sign_arr


def _region_masks_from_policy(theta: jnp.ndarray, policy: dict[str, Any]) -> dict[str, jnp.ndarray]:
    regions = policy.get("regions", None)
    if not regions:
        return {}
    masks: dict[str, jnp.ndarray] = {}
    for region in regions:
        name = str(region.get("name", "")).strip()
        if not name:
            continue
        windows = None
        if "theta" in region:
            windows = [region["theta"]]
        elif "theta_window" in region:
            windows = [region["theta_window"]]
        elif "theta_windows" in region:
            windows = region["theta_windows"]
        if not windows:
            continue
        mask = jnp.zeros_like(theta, dtype=jnp.bool_)
        for theta_min, theta_max in windows:
            theta_min = float(theta_min)
            theta_max = float(theta_max)
            mask = mask | ((theta >= theta_min) & (theta <= theta_max))
        masks[name] = mask.astype(jnp.float64)
    return masks


def analytic_salpha_coefficients(
    *,
    spec: AxisymmetricAnalyticSpec,
    shat: float,
    alpha: float,
    q: float,
    R0: float,
    epsilon: float,
    r0: float | None,
    curvature0: float,
    b_min: float,
    curvature_model: str,
    B0: float | None = None,
    epsilon_x_grad: float | None = None,
    theta_ballooning_on: bool = False,
    theta_ballooning_r: float | None = None,
    linear_shear_on: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    theta, _, theta_scale = _theta_grid(spec)
    model = str(curvature_model).lower()
    if model in ("logb", "logb_curvature", "logb_bracket"):
        curv_x, curv_y, dpar_factor, B = salpha_logb_coefficients(
            theta,
            epsilon=float(epsilon),
            q=float(q),
            shat=float(shat),
            R0=float(R0),
            r0=r0,
            theta_scale=float(theta_scale),
            B0=B0,
            epsilon_x_grad=epsilon_x_grad,
            theta_ballooning_on=theta_ballooning_on,
            theta_ballooning_r=theta_ballooning_r,
            linear_shear_on=linear_shear_on,
        )
    else:
        B = 1.0 / jnp.maximum(1.0 + epsilon * jnp.cos(theta), b_min)
        if model == "ky_only":
            curv_x = jnp.zeros_like(theta)
            curv_y = curvature0 * jnp.cos(theta) * B
        else:
            curv_x = curvature0 * jnp.sin(theta) * B
            curv_y = curvature0 * jnp.cos(theta) * B
        scale = max(theta_scale, 1e-8)
        dpar_factor = jnp.ones_like(theta) * (scale / max(q * R0, 1e-8))
    return theta, curv_x, curv_y, dpar_factor, B


def analytic_miller_coefficients(
    *,
    spec: AxisymmetricAnalyticSpec,
    R0: float,
    r_minor: float,
    q: float,
    kappa: float,
    delta: float,
    B0: float,
    b_min: float,
    curvature_model: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    theta, _, theta_scale = _theta_grid(spec)
    periodic = not spec.open_field_line
    dtheta = float(theta[1] - theta[0]) if theta.size > 1 else 1.0

    # Miller surface parameterization.
    theta_m = theta + delta * jnp.sin(theta)
    R = R0 + r_minor * jnp.cos(theta_m)
    dR_dtheta = -r_minor * jnp.sin(theta_m) * (1.0 + delta * jnp.cos(theta))
    dZ_dtheta = kappa * r_minor * jnp.cos(theta)
    ds_p = jnp.maximum(jnp.sqrt(dR_dtheta**2 + dZ_dtheta**2), 1e-8)
    t_hat_R = dR_dtheta / ds_p
    t_hat_Z = dZ_dtheta / ds_p

    # Magnetic field components.
    Bphi = B0 * R0 / jnp.maximum(R, 1e-8)
    Bp = (r_minor * Bphi) / max(q * R0, 1e-8)
    BR = Bp * t_hat_R
    BZ = Bp * t_hat_Z

    Bmag = jnp.maximum(jnp.sqrt(BR**2 + BZ**2 + Bphi**2), b_min)
    b = jnp.stack([BR / Bmag, Bphi / Bmag, BZ / Bmag], axis=-1)

    # Compute curvature from b(s).
    dphi_dtheta = 1.0 / max(q, 1e-8)
    dl_dtheta = jnp.sqrt(ds_p**2 + (R * dphi_dtheta) ** 2)
    db_dtheta = _dd_theta(b, dtheta, periodic=periodic)
    kappa_vec = db_dtheta / dl_dtheta[:, None]

    # Build perpendicular basis.
    n_hat_R = -t_hat_Z
    n_hat_Z = t_hat_R
    e_x = jnp.stack([n_hat_R, jnp.zeros_like(n_hat_R), n_hat_Z], axis=-1)
    e_x = e_x / jnp.maximum(jnp.linalg.norm(e_x, axis=-1, keepdims=True), 1e-8)
    e_y = _cross(b, e_x)
    e_y = e_y / jnp.maximum(jnp.linalg.norm(e_y, axis=-1, keepdims=True), 1e-8)

    curv_x = jnp.sum(kappa_vec * e_x, axis=-1)
    curv_y = jnp.sum(kappa_vec * e_y, axis=-1)

    scale = max(theta_scale, 1e-8)
    dpar_factor = scale / jnp.maximum(dl_dtheta, 1e-8)
    return theta, curv_x, curv_y, dpar_factor, Bmag


def _psi76_and_grad(
    R: jnp.ndarray,
    Z: jnp.ndarray,
    *,
    I0: float,
    sigma0: float,
    R1: float,
    Z1: float,
    Z2: float,
    rho_s0: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r1 = (R - R1) ** 2 + (Z - Z1) ** 2
    r2 = (R - R1) ** 2 + (Z - Z2) ** 2
    r1 = jnp.maximum(r1, 1e-12)
    r2 = jnp.maximum(r2, 1e-12)
    exp1 = jnp.exp(r1 / (sigma0**2))
    dpsi_dR = I0 * (R - R1) * (1.0 / r1 + exp1 / r1 + 1.0 / r2)
    dpsi_dZ = I0 * ((Z - Z1) * (1.0 / r1 + exp1 / r1) + (Z - Z2) * (1.0 / r2))
    # The exponential integral term is omitted from psi here; the gradients are kept analytic.
    psi = 0.5 * I0 * (jnp.log(r1 / (rho_s0**2)) + jnp.log(r2 / (rho_s0**2)))
    return psi, dpsi_dR, dpsi_dZ


def _trace_field_line_psi76(
    *,
    R_start: float,
    Z_start: float,
    dz: float,
    nz: int,
    params: dict[str, float],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Trace forward/backward in arc length using RK4.
    def field_dir(R, Z):
        _, dpsi_dR, dpsi_dZ = _psi76_and_grad(
            R,
            Z,
            I0=params["I0"],
            sigma0=params["sigma0"],
            R1=params["R1"],
            Z1=params["Z1"],
            Z2=params["Z2"],
            rho_s0=params["rho_s0"],
        )
        BR = dpsi_dZ / jnp.maximum(R, 1e-8)
        BZ = -dpsi_dR / jnp.maximum(R, 1e-8)
        Bphi = params["B0"] * params["R0"] / jnp.maximum(R, 1e-8)
        Bmag = jnp.maximum(jnp.sqrt(BR**2 + BZ**2 + Bphi**2), 1e-12)
        return BR / Bmag, BZ / Bmag, Bphi / Bmag

    def rk4_step(state, step):
        R, Z = state
        sign = step
        k1_R, k1_Z, _ = field_dir(R, Z)
        k2_R, k2_Z, _ = field_dir(R + 0.5 * sign * dz * k1_R, Z + 0.5 * sign * dz * k1_Z)
        k3_R, k3_Z, _ = field_dir(R + 0.5 * sign * dz * k2_R, Z + 0.5 * sign * dz * k2_Z)
        k4_R, k4_Z, _ = field_dir(R + sign * dz * k3_R, Z + sign * dz * k3_Z)
        R_next = R + (sign * dz / 6.0) * (k1_R + 2.0 * k2_R + 2.0 * k3_R + k4_R)
        Z_next = Z + (sign * dz / 6.0) * (k1_Z + 2.0 * k2_Z + 2.0 * k3_Z + k4_Z)
        return (R_next, Z_next), (R_next, Z_next)

    n_half = nz // 2
    steps = jnp.ones((n_half,), dtype=jnp.float64)
    (Rf, Zf), forward = jax.lax.scan(lambda s, _: rk4_step(s, 1.0), (R_start, Z_start), steps)
    (Rb, Zb), backward = jax.lax.scan(lambda s, _: rk4_step(s, -1.0), (R_start, Z_start), steps)
    Rf_arr, Zf_arr = forward
    Rb_arr, Zb_arr = backward

    R_full = jnp.concatenate([jnp.flip(Rb_arr, axis=0), jnp.asarray([R_start]), Rf_arr], axis=0)
    Z_full = jnp.concatenate([jnp.flip(Zb_arr, axis=0), jnp.asarray([Z_start]), Zf_arr], axis=0)
    z = dz * (jnp.arange(R_full.size) - n_half)
    return R_full, Z_full, z


def analytic_xpoint_psi76_coefficients(
    *,
    spec: AxisymmetricAnalyticSpec,
    R0: float,
    r_minor: float,
    q: float,
    b_min: float,
    curvature_model: str,
    I0: float,
    sigma0: float,
    R1: float,
    Z1: float,
    Z2: float,
    rho_s0: float,
    B0: float,
    R_start: float,
    Z_start: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    theta, dz, _ = _theta_grid(spec)
    periodic = not spec.open_field_line
    params = dict(
        I0=I0,
        sigma0=sigma0,
        R1=R1,
        Z1=Z1,
        Z2=Z2,
        rho_s0=rho_s0,
        B0=B0,
        R0=R0,
    )

    R, Z, z = _trace_field_line_psi76(
        R_start=R_start,
        Z_start=Z_start,
        dz=float(dz),
        nz=int(spec.nz),
        params=params,
    )

    psi, dpsi_dR, dpsi_dZ = _psi76_and_grad(
        R,
        Z,
        I0=params["I0"],
        sigma0=params["sigma0"],
        R1=params["R1"],
        Z1=params["Z1"],
        Z2=params["Z2"],
        rho_s0=params["rho_s0"],
    )
    BR = dpsi_dZ / jnp.maximum(R, 1e-8)
    BZ = -dpsi_dR / jnp.maximum(R, 1e-8)
    Bphi = B0 * R0 / jnp.maximum(R, 1e-8)
    Bmag = jnp.maximum(jnp.sqrt(BR**2 + BZ**2 + Bphi**2), b_min)
    b = jnp.stack([BR / Bmag, Bphi / Bmag, BZ / Bmag], axis=-1)

    # curvature from b(s)
    db_ds = _dd_theta(b, float(dz), periodic=periodic)
    kappa_vec = db_ds

    # Perpendicular basis from grad psi
    grad_psi = jnp.stack([dpsi_dR, jnp.zeros_like(dpsi_dR), dpsi_dZ], axis=-1)
    e_x = grad_psi / jnp.maximum(jnp.linalg.norm(grad_psi, axis=-1, keepdims=True), 1e-8)
    e_y = _cross(b, e_x)
    e_y = e_y / jnp.maximum(jnp.linalg.norm(e_y, axis=-1, keepdims=True), 1e-8)

    curv_x = jnp.sum(kappa_vec * e_x, axis=-1)
    curv_y = jnp.sum(kappa_vec * e_y, axis=-1)

    dpar_factor = jnp.ones_like(curv_x)
    theta_geom = jnp.arctan2(Z - Z1, R - R1)
    return theta_geom, curv_x, curv_y, dpar_factor, Bmag


def build_axisymmetric_analytic_adapter(
    *,
    params: DRBSystemParams,
    cfg: dict[str, Any],
) -> FieldAlignedGeometryAdapter:
    model = str(cfg.get("model", "salpha")).lower()
    spec = AxisymmetricAnalyticSpec(
        model=model,
        nz=int(cfg.get("nz", 32)),
        Lz=float(cfg.get("Lz", 2 * jnp.pi)),
        open_field_line=bool(cfg.get("open_field_line", False)),
        theta_scale=cfg.get("theta_scale", None),
    )

    curv_model = str(cfg.get("curvature_model", "vector_xy"))
    b_min = float(cfg.get("b_min", 0.05))

    theta_for_sheath: jnp.ndarray | None = None

    if model in ("salpha", "s-alpha", "s_alpha"):
        theta, curv_x, curv_y, dpar_factor, B = analytic_salpha_coefficients(
            spec=spec,
            shat=float(cfg.get("shat", 0.796)),
            alpha=float(cfg.get("alpha", 0.0)),
            q=float(cfg.get("q", 1.4)),
            R0=float(cfg.get("R0", 1.0)),
            epsilon=float(cfg.get("epsilon", 0.18)),
            r0=cfg.get("r0", None),
            curvature0=float(cfg.get("curvature0", float(cfg.get("epsilon", 0.18)))),
            b_min=b_min,
            curvature_model=curv_model,
            B0=cfg.get("B0", None),
            epsilon_x_grad=cfg.get("epsilon_x_grad", None),
            theta_ballooning_on=bool(cfg.get("theta_ballooning_on", False)),
            theta_ballooning_r=cfg.get("theta_ballooning_r", None),
            linear_shear_on=bool(cfg.get("linear_shear_on", False)),
        )
        theta_for_sheath = theta
    elif model in ("miller", "miller_simple"):
        theta, curv_x, curv_y, dpar_factor, B = analytic_miller_coefficients(
            spec=spec,
            R0=float(cfg.get("R0", 1.0)),
            r_minor=float(cfg.get("r_minor", 0.18)),
            q=float(cfg.get("q", 1.4)),
            kappa=float(cfg.get("kappa", 1.0)),
            delta=float(cfg.get("delta", 0.0)),
            B0=float(cfg.get("B0", 1.0)),
            b_min=b_min,
            curvature_model=curv_model,
        )
        theta_for_sheath = theta
    elif model in ("xpoint_psi76", "x-point-76", "xpoint"):
        if int(spec.nz) % 2 == 0:
            raise ValueError("xpoint_psi76 requires an odd nz to center the field line.")
        theta_geom, curv_x, curv_y, dpar_factor, B = analytic_xpoint_psi76_coefficients(
            spec=spec,
            R0=float(cfg.get("R0", 100.0)),
            r_minor=float(cfg.get("r_minor", 1.0)),
            q=float(cfg.get("q", 1.4)),
            b_min=b_min,
            curvature_model=curv_model,
            I0=float(cfg.get("I0", 40.0)),
            sigma0=float(cfg.get("sigma0", 6.25)),
            R1=float(cfg.get("R1", 100.0)),
            Z1=float(cfg.get("Z1", 0.0)),
            Z2=float(cfg.get("Z2", -40.0)),
            rho_s0=float(cfg.get("rho_s0", 1.0)),
            B0=float(cfg.get("B0", 1.0)),
            R_start=float(cfg.get("R_start", float(cfg.get("R0", 100.0)) + float(cfg.get("r_minor", 1.0)))),
            Z_start=float(cfg.get("Z_start", 0.0)),
        )
        theta_for_sheath = theta_geom
    else:
        raise ValueError(f"Unknown axisymmetric analytic model '{model}'.")

    policy = cfg.get("boundary_policy", {}) if isinstance(cfg.get("boundary_policy", {}), dict) else {}
    if "sheath_windows" not in cfg and "sheath_windows" in policy:
        cfg = dict(cfg)
        cfg["sheath_windows"] = policy["sheath_windows"]
    if "sheath_sign" not in cfg and "sheath_sign" in policy:
        cfg = dict(cfg)
        cfg["sheath_sign"] = policy["sheath_sign"]

    region_masks = None
    region_bcs = None
    if theta_for_sheath is not None and policy:
        masks = _region_masks_from_policy(theta_for_sheath, policy)
        region_masks = masks if masks else None
        if region_masks:
            region_bcs = parse_region_bcs(policy, region_masks)

    grid = FieldAlignedGrid.make(
        nx=int(cfg.get("nx", 32)),
        ny=int(cfg.get("ny", 32)),
        nz=int(spec.nz),
        Lx=float(cfg.get("Lx", 2 * jnp.pi)),
        Ly=float(cfg.get("Ly", 2 * jnp.pi)),
        Lz=float(spec.Lz),
        bc_x=str(cfg.get("bc_x", "periodic")),
        bc_y=str(cfg.get("bc_y", "periodic")),
        dealias=bool(cfg.get("dealias", True)),
        open_field_line=spec.open_field_line,
        bc_value_x=float(cfg.get("bc_value_x", 0.0)),
        bc_value_y=float(cfg.get("bc_value_y", 0.0)),
        bc_grad_x=float(cfg.get("bc_grad_x", 0.0)),
        bc_grad_y=float(cfg.get("bc_grad_y", 0.0)),
        region_masks=region_masks,
        region_bcs=region_bcs,
    )

    sheath_windows = cfg.get("sheath_windows", None)
    sheath_sign = cfg.get("sheath_sign", None)
    if sheath_windows and theta_for_sheath is not None:
        windows = [(float(a), float(b)) for a, b in sheath_windows]
        sign = None if sheath_sign is None else [float(s) for s in sheath_sign]
        mask, sign_arr = _apply_sheath_windows(theta_for_sheath, windows=windows, sign=sign)
        grid = FieldAlignedGrid.from_z(
            perp=grid.perp,
            z=grid.z,
            open_field_line=spec.open_field_line,
            sheath_mask=mask,
            sheath_sign=sign_arr,
            region_masks=region_masks,
            region_bcs=region_bcs,
        )

    return FieldAlignedGeometryAdapter.from_coefficients(
        params=params,
        grid=grid,
        curv_x=curv_x,
        curv_y=curv_y,
        dpar_factor=dpar_factor,
        B=B,
    )
