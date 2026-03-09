"""Literal Hermes parallel runtime helpers.

This module owns the live open-field parallel transport contract used by the
literal Stage 1 density/pressure cache, instead of routing that path through
the unified-core helper layer.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.context import TermContext
from jaxdrb.core.terms.ops import laplacian

from .div_ops import div_par_centered
from .fv import div_par_mod
from .sheath import ParallelSheathState, build_parallel_sheath_state
from .shifted_metric import (
    build_shifted_metric_fft_phases,
    build_shifted_metric_weights,
    to_field_aligned_all,
    to_field_aligned_all_fft,
    to_field_aligned_nox,
    to_field_aligned_nox_fft,
)


def _minmod(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    s = 0.5 * (jnp.sign(a) + jnp.sign(b))
    return s * jnp.minimum(jnp.abs(a), jnp.abs(b))


def _limited_slope(f: jnp.ndarray, limiter: str) -> jnp.ndarray:
    df = f[1:] - f[:-1]
    if limiter == "none":
        slope = jnp.zeros_like(f)
        if f.shape[0] > 1:
            if f.shape[0] == 2:
                slope = slope.at[0].set(df[0])
                slope = slope.at[-1].set(df[-1])
            else:
                slope = slope.at[1:-1].set(0.5 * (df[:-1] + df[1:]))
                slope = slope.at[0].set(df[0])
                slope = slope.at[-1].set(df[-1])
        return slope
    df_b = df[:-1]
    df_f = df[1:]
    if limiter == "mc":
        slope = _minmod(_minmod(2.0 * df_b, 2.0 * df_f), 0.5 * (df_b + df_f))
    else:
        slope = _minmod(df_b, df_f)
    slope_full = jnp.zeros_like(f)
    slope_full = slope_full.at[1:-1].set(slope)
    slope_full = slope_full.at[0].set(df[0])
    slope_full = slope_full.at[-1].set(df[-1])
    return slope_full


def _edge_slope(
    prev_val: jnp.ndarray,
    cell_val: jnp.ndarray,
    next_val: jnp.ndarray,
    limiter: str,
) -> jnp.ndarray:
    df_b = cell_val - prev_val
    df_f = next_val - cell_val
    if limiter == "none":
        return 0.5 * (next_val - prev_val)
    if limiter == "mc":
        return _minmod(_minmod(2.0 * df_b, 2.0 * df_f), 0.5 * (df_b + df_f))
    return _minmod(df_b, df_f)


def _apply_open_boundary_reconstruction(
    left: jnp.ndarray,
    right: jnp.ndarray,
    f: jnp.ndarray,
    *,
    limiter: str,
    ghost_low: jnp.ndarray | None,
    ghost_high: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    out_l = left
    out_r = right
    if ghost_low is not None and f.shape[0] >= 2:
        slope_low = _edge_slope(jnp.asarray(ghost_low), f[0], f[1], limiter)
        out_l = out_l.at[0].set(f[0] - 0.5 * slope_low)
        out_r = out_r.at[0].set(f[0] + 0.5 * slope_low)
    if ghost_high is not None and f.shape[0] >= 2:
        slope_high = _edge_slope(f[-2], f[-1], jnp.asarray(ghost_high), limiter)
        out_l = out_l.at[-1].set(f[-1] - 0.5 * slope_high)
        out_r = out_r.at[-1].set(f[-1] + 0.5 * slope_high)
    return out_l, out_r


def _flux_divergence_open(
    f: jnp.ndarray,
    v: jnp.ndarray,
    dz: float,
    limiter: str,
    wave: jnp.ndarray | None = None,
    J: jnp.ndarray | None = None,
    gpar: jnp.ndarray | None = None,
    dpar_factor: jnp.ndarray | None = None,
    sign: float = 1.0,
    scheme: str = "rusanov",
    fixflux: bool = True,
    boundary_flux_low: jnp.ndarray | None = None,
    boundary_flux_high: jnp.ndarray | None = None,
    ghost_low_f: jnp.ndarray | None = None,
    ghost_high_f: jnp.ndarray | None = None,
    ghost_low_v: jnp.ndarray | None = None,
    ghost_high_v: jnp.ndarray | None = None,
) -> jnp.ndarray:
    slope_f = _limited_slope(f, limiter)
    slope_v = _limited_slope(v, limiter)
    f_L = f - 0.5 * slope_f
    f_R = f + 0.5 * slope_f
    v_L = v - 0.5 * slope_v
    v_R = v + 0.5 * slope_v
    if any(val is not None for val in (ghost_low_f, ghost_high_f)):
        f_L, f_R = _apply_open_boundary_reconstruction(
            f_L, f_R, f, limiter=limiter, ghost_low=ghost_low_f, ghost_high=ghost_high_f
        )
    if any(val is not None for val in (ghost_low_v, ghost_high_v)):
        v_L, v_R = _apply_open_boundary_reconstruction(
            v_L, v_R, v, limiter=limiter, ghost_low=ghost_low_v, ghost_high=ghost_high_v
        )

    left_f = f_R[:-1]
    right_f = f_L[1:]
    left_v = v_R[:-1]
    right_v = v_L[1:]

    abs_v = jnp.abs(v)
    amax_pair = abs_v if wave is None else jnp.maximum(abs_v, jnp.abs(wave))
    amax = jnp.maximum(amax_pair[:-1], amax_pair[1:])

    scheme = scheme.lower()
    if scheme == "lax":
        flux = left_f * 0.5 * (left_v + amax) + right_f * 0.5 * (right_v - amax)
    else:
        flux = 0.5 * (left_f * left_v + right_f * right_v) + 0.5 * amax * (left_f - right_f)

    div = jnp.zeros_like(f)
    if boundary_flux_low is not None and boundary_flux_high is not None:
        left_bndry = jnp.asarray(boundary_flux_low)
        right_bndry = jnp.asarray(boundary_flux_high)
    elif fixflux and scheme == "lax":
        left_bndry = 0.5 * (f[0] + f[1]) * 0.5 * (v[0] + v[1])
        right_bndry = 0.5 * (f[-1] + f[-2]) * 0.5 * (v[-1] + v[-2])
    else:
        left_bndry = f[0] * v[0]
        right_bndry = f[-1] * v[-1]

    if J is None:
        div = div.at[1:-1].set((flux[1:] - flux[:-1]) / dz)
        div = div.at[0].set((flux[0] - left_bndry) / dz)
        div = div.at[-1].set((right_bndry - flux[-1]) / dz)
    else:
        Jc = jnp.asarray(J)
        if Jc.ndim == 1:
            Jc = Jc[:, None, None]
        elif Jc.ndim == 2:
            Jc = Jc[None, :, :]
        if gpar is None:
            J_face = 0.5 * (Jc[1:] + Jc[:-1])
            fluxJ = flux * J_face
            div = div.at[1:-1].set((fluxJ[1:] - fluxJ[:-1]) / (dz * jnp.maximum(Jc[1:-1], 1e-30)))
            div = div.at[0].set((fluxJ[0] - Jc[0] * left_bndry) / (dz * jnp.maximum(Jc[0], 1e-30)))
            div = div.at[-1].set(
                (Jc[-1] * right_bndry - fluxJ[-1]) / (dz * jnp.maximum(Jc[-1], 1e-30))
            )
        else:
            gpar_c = jnp.asarray(gpar)
            if gpar_c.ndim == 1:
                gpar_c = gpar_c[:, None, None]
            elif gpar_c.ndim == 2:
                gpar_c = gpar_c[None, :, :]
            sqrt_gpar = jnp.sqrt(jnp.maximum(gpar_c, 1e-30))
            common_r = (Jc[1:] + Jc[:-1]) / (sqrt_gpar[1:] + sqrt_gpar[:-1])
            flux_factor_rc = common_r / (dz * jnp.maximum(Jc[:-1], 1e-30))
            flux_factor_rp = common_r / (dz * jnp.maximum(Jc[1:], 1e-30))
            div = div.at[:-1].add(flux * flux_factor_rc)
            div = div.at[1:].add(-flux * flux_factor_rp)
            if fixflux:
                boundary_factor_low = 1.0 / (dz * jnp.maximum(sqrt_gpar[0], 1e-30))
                boundary_factor_high = 1.0 / (dz * jnp.maximum(sqrt_gpar[-1], 1e-30))
                div = div.at[0].add(-left_bndry * boundary_factor_low)
                div = div.at[-1].add(right_bndry * boundary_factor_high)

    if dpar_factor is not None and gpar is None:
        div = div * jnp.asarray(dpar_factor)
    return float(sign) * div


def _centered_divergence_open(
    f: jnp.ndarray,
    dz: float,
    *,
    J: jnp.ndarray | None = None,
    gpar: jnp.ndarray | None = None,
    dpar_factor: jnp.ndarray | None = None,
    sign: float = 1.0,
    boundary_flux_low: jnp.ndarray | None = None,
    boundary_flux_high: jnp.ndarray | None = None,
) -> jnp.ndarray:
    face = 0.5 * (f[1:] + f[:-1])
    div = jnp.zeros_like(f)
    left_bndry = f[0] if boundary_flux_low is None else jnp.asarray(boundary_flux_low)
    right_bndry = f[-1] if boundary_flux_high is None else jnp.asarray(boundary_flux_high)

    if J is None:
        div = div.at[1:-1].set((face[1:] - face[:-1]) / dz)
        div = div.at[0].set((face[0] - left_bndry) / dz)
        div = div.at[-1].set((right_bndry - face[-1]) / dz)
    else:
        Jc = jnp.asarray(J)
        if Jc.ndim == 1:
            Jc = Jc[:, None, None]
        elif Jc.ndim == 2:
            Jc = Jc[None, :, :]
        if gpar is None:
            J_face = 0.5 * (Jc[1:] + Jc[:-1])
            fluxJ = face * J_face
            div = div.at[1:-1].set((fluxJ[1:] - fluxJ[:-1]) / (dz * jnp.maximum(Jc[1:-1], 1e-30)))
            div = div.at[0].set((fluxJ[0] - Jc[0] * left_bndry) / (dz * jnp.maximum(Jc[0], 1e-30)))
            div = div.at[-1].set(
                (Jc[-1] * right_bndry - fluxJ[-1]) / (dz * jnp.maximum(Jc[-1], 1e-30))
            )
        else:
            gpar_c = jnp.asarray(gpar)
            if gpar_c.ndim == 1:
                gpar_c = gpar_c[:, None, None]
            elif gpar_c.ndim == 2:
                gpar_c = gpar_c[None, :, :]
            sqrt_gpar = jnp.sqrt(jnp.maximum(gpar_c, 1e-30))
            common_r = (Jc[1:] + Jc[:-1]) / (sqrt_gpar[1:] + sqrt_gpar[:-1])
            flux_factor_rc = common_r / (dz * jnp.maximum(Jc[:-1], 1e-30))
            flux_factor_rp = common_r / (dz * jnp.maximum(Jc[1:], 1e-30))
            div = div.at[:-1].add(face * flux_factor_rc)
            div = div.at[1:].add(-face * flux_factor_rp)
            boundary_factor_low = 1.0 / (dz * jnp.maximum(sqrt_gpar[0], 1e-30))
            boundary_factor_high = 1.0 / (dz * jnp.maximum(sqrt_gpar[-1], 1e-30))
            div = div.at[0].add(-left_bndry * boundary_factor_low)
            div = div.at[-1].add(right_bndry * boundary_factor_high)

    if dpar_factor is not None and gpar is None:
        div = div * jnp.asarray(dpar_factor)
    return float(sign) * div


def _shift_boundary_flux_to_field_aligned(
    flux: jnp.ndarray | None,
    *,
    params,
    geom,
    z_index: int,
) -> jnp.ndarray | None:
    if flux is None:
        return None
    if str(getattr(params, "parallel_transform", "none")).lower() != "shifted":
        return flux
    shift_idx = getattr(geom, "shift_idx", None)
    if shift_idx is None:
        return flux

    arr = jnp.asarray(flux)
    if arr.ndim != 2:
        return arr

    shift = jnp.asarray(shift_idx[z_index], dtype=jnp.float64)
    if shift.ndim == 0:
        shift = jnp.full((arr.shape[0],), shift, dtype=jnp.float64)
    if shift.ndim == 2:
        shift = jnp.mean(shift, axis=-1)
    if shift.ndim != 1 or int(shift.shape[0]) != int(arr.shape[0]):
        return arr

    grid = getattr(geom, "grid", None)
    interp = str(getattr(params, "parallel_shift_interp", "spectral")).lower()
    open_field_line = bool(getattr(grid, "open_field_line", False))
    bc = getattr(getattr(grid, "perp", None), "bc", None)
    preserve_x_boundaries = open_field_line and int(getattr(bc, "kind_x", 0)) != 0
    field = arr[None, :, :]
    if interp == "spectral":
        z_shift = getattr(geom, "z_shift", None)
        if z_shift is None:
            return arr
        z_shift_row = jnp.asarray(z_shift[z_index], dtype=jnp.float64)
        if z_shift_row.ndim == 0:
            z_shift_row = jnp.full((arr.shape[0],), z_shift_row, dtype=jnp.float64)
        if z_shift_row.ndim == 2:
            z_shift_row = jnp.mean(z_shift_row, axis=-1)
        if z_shift_row.ndim != 1 or int(z_shift_row.shape[0]) != int(arr.shape[0]):
            return arr
        dy = float(getattr(getattr(grid, "perp", None), "dy", 1.0))
        phases = build_shifted_metric_fft_phases(
            z_shift_row[None, :],
            nx=int(arr.shape[0]),
            npar=1,
            nbinorm=int(arr.shape[1]),
            zlength=dy * float(arr.shape[1]),
            open_field_line=open_field_line,
        )
        shifted = (
            to_field_aligned_nox_fft(field, phases)
            if preserve_x_boundaries
            else to_field_aligned_all_fft(field, phases)
        )
        return shifted[0]

    weights = build_shifted_metric_weights(
        shift[None, :],
        nx=int(arr.shape[0]),
        npar=1,
        nbinorm=int(arr.shape[1]),
        open_field_line=open_field_line,
    )
    shifted = (
        to_field_aligned_nox(field, weights)
        if preserve_x_boundaries
        else to_field_aligned_all(field, weights)
    )
    return shifted[0]


def dpar_flux_conservative(
    ctx: TermContext,
    f: jnp.ndarray,
    v: jnp.ndarray,
    *,
    wave: jnp.ndarray | None = None,
    boundary_flux_low: jnp.ndarray | None = None,
    boundary_flux_high: jnp.ndarray | None = None,
    ghost_low_f: jnp.ndarray | None = None,
    ghost_high_f: jnp.ndarray | None = None,
    ghost_low_v: jnp.ndarray | None = None,
    ghost_high_v: jnp.ndarray | None = None,
) -> jnp.ndarray:
    grid = getattr(ctx.geom, "grid", None)
    limiter = str(ctx.params.parallel_limiter).lower()
    if str(getattr(ctx.params, "parallel_flux_scheme", "rusanov")).lower() == "hermes_mirror":
        if grid is None or f.ndim != 3:
            raise ValueError("Hermes literal parallel flux requires a 3D field-aligned grid.")
        transform = str(getattr(ctx.params, "parallel_transform", "none")).lower()
        use_shift = transform == "shifted"
        scheme = str(getattr(ctx.params, "parallel_scheme", "rusanov")).lower()
        if use_shift:
            f = ctx.geom.to_field_aligned_nox(f)
            v = ctx.geom.to_field_aligned_nox(v)
            if boundary_flux_low is not None:
                boundary_flux_low = _shift_boundary_flux_to_field_aligned(
                    boundary_flux_low, params=ctx.params, geom=ctx.geom, z_index=0
                )
            if boundary_flux_high is not None:
                boundary_flux_high = _shift_boundary_flux_to_field_aligned(
                    boundary_flux_high, params=ctx.params, geom=ctx.geom, z_index=-1
                )
            if ghost_low_f is not None:
                ghost_low_f = _shift_boundary_flux_to_field_aligned(
                    ghost_low_f, params=ctx.params, geom=ctx.geom, z_index=0
                )
            if ghost_high_f is not None:
                ghost_high_f = _shift_boundary_flux_to_field_aligned(
                    ghost_high_f, params=ctx.params, geom=ctx.geom, z_index=-1
                )
            if ghost_low_v is not None:
                ghost_low_v = _shift_boundary_flux_to_field_aligned(
                    ghost_low_v, params=ctx.params, geom=ctx.geom, z_index=0
                )
            if ghost_high_v is not None:
                ghost_high_v = _shift_boundary_flux_to_field_aligned(
                    ghost_high_v, params=ctx.params, geom=ctx.geom, z_index=-1
                )
        J = getattr(ctx.geom, "jacobian", None)
        dy = getattr(ctx.geom, "metric_dy", None)
        dpar_factor = getattr(ctx.geom, "dpar_factor", None)
        gpar = getattr(ctx.geom, "gpar", None) if ctx.params.use_gpar_flux else None
        sign = float(ctx.params.parallel_sign)
        boundary_flux_scale = float(getattr(ctx.params, "parallel_boundary_flux_scale", 1.0))

        if use_shift:
            if J is not None:
                J = ctx.geom.to_field_aligned_nox(J)
            if dy is not None:
                dy = ctx.geom.to_field_aligned_nox(dy)
            if gpar is not None:
                gpar = ctx.geom.to_field_aligned_nox(gpar)

        if grid.open_field_line:
            if wave is None:
                if scheme == "hermes_mirror" and (
                    ghost_low_f is not None or ghost_high_f is not None
                ):
                    div = div_par_centered(
                        f,
                        dz=float(grid.dz),
                        dy=dy,
                        J=J,
                        gpar=gpar,
                        dpar_factor=dpar_factor,
                        sign=sign,
                        ghost_low=ghost_low_f,
                        ghost_high=ghost_high_f,
                        boundary_flux_scale=boundary_flux_scale,
                    )
                else:
                    div = _centered_divergence_open(
                        f,
                        float(grid.dz),
                        J=J,
                        gpar=gpar,
                        dpar_factor=dpar_factor,
                        sign=sign,
                        boundary_flux_low=boundary_flux_low,
                        boundary_flux_high=boundary_flux_high,
                    )
            else:
                fixflux = bool(ctx.params.parallel_fixflux)
                if scheme == "hermes_mirror" and any(
                    val is not None
                    for val in (ghost_low_f, ghost_high_f, ghost_low_v, ghost_high_v)
                ):
                    div = div_par_mod(
                        f,
                        v,
                        wave,
                        dz=float(grid.dz),
                        dy=dy,
                        limiter=limiter,
                        J=J,
                        gpar=gpar,
                        dpar_factor=dpar_factor,
                        sign=sign,
                        fixflux=fixflux,
                        ghost_low_f=ghost_low_f,
                        ghost_high_f=ghost_high_f,
                        ghost_low_v=ghost_low_v,
                        ghost_high_v=ghost_high_v,
                        boundary_flux_scale=boundary_flux_scale,
                    )
                else:
                    div = _flux_divergence_open(
                        f,
                        v,
                        float(grid.dz),
                        limiter,
                        wave=wave,
                        J=J,
                        gpar=gpar,
                        dpar_factor=dpar_factor,
                        sign=sign,
                        scheme=scheme,
                        fixflux=fixflux,
                        boundary_flux_low=boundary_flux_low,
                        boundary_flux_high=boundary_flux_high,
                        ghost_low_f=ghost_low_f,
                        ghost_high_f=ghost_high_f,
                        ghost_low_v=ghost_low_v,
                        ghost_high_v=ghost_high_v,
                    )
        else:
            div = ctx.geom.dpar(f * v, bc_kind="dirichlet")

        if use_shift:
            div = ctx.geom.from_field_aligned_nox(div)
        return div

    return ctx.geom.dpar(f * v, bc_kind="dirichlet")


def fastest_wave(ctx: TermContext) -> jnp.ndarray:
    Te = ctx.Te_prepared
    Ti = ctx.Ti_prepared if ctx.hot_on else None
    aa_e = jnp.maximum(float(ctx.params.me_hat), 1e-12)
    aa_i = jnp.maximum(float(getattr(ctx.params, "average_atomic_mass", 1.0)), 1e-12)

    fast = jnp.sqrt(Te / aa_e)
    total_pressure = ctx.pe_prepared
    total_density = ctx.n_prepared * aa_i
    if Ti is not None:
        fast = jnp.maximum(fast, jnp.sqrt(Ti / aa_i))
        total_pressure = total_pressure + ctx.pi_prepared
    sound_speed = jnp.sqrt(total_pressure / jnp.maximum(total_density, 1e-12))
    fast = jnp.maximum(fast, sound_speed)
    return fast


def pressure_transport_coeffs(ctx: TermContext) -> tuple[float, float]:
    model = str(getattr(ctx.params, "parallel_pressure_model", "custom")).lower()
    if model == "hermes_vgradp":
        return 5.0 / 3.0, 2.0 / 3.0
    if model == "hermes_pdivv":
        return 1.0, 0.0
    return (
        float(getattr(ctx.params, "parallel_pressure_flux_coeff", 1.0)),
        float(getattr(ctx.params, "parallel_pressure_work_coeff", 0.0)),
    )


def _with_boundary_targets(
    v: jnp.ndarray,
    v_target: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    out = jnp.asarray(v)
    target = jnp.asarray(v_target)
    mask = jnp.asarray(mask)
    out = out.at[0].set(jnp.where(mask[0] > 0.0, target[0], out[0]))
    out = out.at[-1].set(jnp.where(mask[-1] > 0.0, target[-1], out[-1]))
    return out


def _sheath_boundary_data(
    ctx: TermContext,
    y: DRBSystemState,
) -> ParallelSheathState | None:
    if not bool(ctx.params.parallel_use_sheath_targets):
        return None
    grid = getattr(ctx.geom, "grid", None)
    if grid is None or not bool(getattr(grid, "open_field_line", False)):
        return None
    if not (bool(ctx.params.sheath_on) or bool(ctx.params.sheath_bc_on)):
        return None
    if not hasattr(ctx.geom, "sheath_mask_sign"):
        return None
    if y.Te.shape[0] < 2:
        return None

    mask, sign = ctx.geom.sheath_mask_sign()
    mask = jnp.broadcast_to(mask, y.Te.shape)
    sign = jnp.broadcast_to(sign, y.Te.shape)

    return build_parallel_sheath_state(
        n_e=ctx.n_prepared,
        Te=ctx.Te_prepared,
        pe=ctx.pe_prepared,
        phi=ctx.phi,
        v_e=y.vpar_e,
        n_i=ctx.n_prepared,
        Ti=ctx.Ti_prepared,
        pi=ctx.pi_prepared,
        v_i=y.vpar_i,
        me_hat=max(float(ctx.params.me_hat), 1e-12),
        ion_mass=max(float(getattr(ctx.params, "average_atomic_mass", 1.0)), 1e-12),
        ion_charge=1.0,
        secondary_electron_coef=float(getattr(ctx.params, "sheath_secondary_electron_coef", 0.0)),
        wall_potential=float(getattr(ctx.params, "sheath_wall_potential", 0.0)),
        floor_potential=bool(getattr(ctx.params, "sheath_floor_potential", True)),
        ion_adiabatic=float(getattr(ctx.params, "sheath_ion_adiabatic", 5.0 / 3.0)),
        mask=mask,
        sign=sign,
    )


def _sheath_velocity_targets(
    ctx: TermContext,
    y: DRBSystemState,
    sheath_data: ParallelSheathState | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if sheath_data is None:
        return y.vpar_e, y.vpar_i
    mode = str(getattr(ctx.params, "parallel_sheath_flux_mode", "boundary_flux")).lower()
    if mode != "replace_boundary":
        return y.vpar_e, y.vpar_i

    assert sheath_data.mask is not None
    mask = sheath_data.mask
    ve_target = jnp.zeros_like(y.vpar_e)
    vi_target = jnp.zeros_like(y.vpar_i)
    ve_target = ve_target.at[0].set(sheath_data.ve_sheath_low)
    ve_target = ve_target.at[-1].set(sheath_data.ve_sheath_high)
    vi_target = vi_target.at[0].set(sheath_data.vi_sheath_low)
    vi_target = vi_target.at[-1].set(sheath_data.vi_sheath_high)
    return _with_boundary_targets(y.vpar_e, ve_target, mask), _with_boundary_targets(
        y.vpar_i, vi_target, mask
    )


class ParallelVars(eqx.Module):
    vpar_e_flux: jnp.ndarray
    vpar_i_flux: jnp.ndarray
    sheath_data: ParallelSheathState | None
    dpar_ve: jnp.ndarray
    dpar_vi: jnp.ndarray
    dpar_Te: jnp.ndarray
    dpar_Ti: jnp.ndarray
    dpar_j: jnp.ndarray
    dpar_psi: jnp.ndarray
    grad_par_phi_pe: jnp.ndarray
    jpar_total: jnp.ndarray


def parallel_vars(ctx: TermContext, y: DRBSystemState) -> ParallelVars:
    sheath_data = _sheath_boundary_data(ctx, y)
    vpar_e_flux, vpar_i_flux = _sheath_velocity_targets(ctx, y, sheath_data)

    with jax.named_scope("parallel_dpar"):
        dpar_ve = ctx.geom.dpar(vpar_e_flux, bc_kind="dirichlet")
        dpar_vi = ctx.geom.dpar(vpar_i_flux, bc_kind="dirichlet")
        dpar_Te = ctx.geom.dpar(ctx.Te_prepared, bc_kind="neumann")
        dpar_Ti = (
            ctx.geom.dpar(ctx.Ti_prepared, bc_kind="neumann")
            if ctx.hot_on
            else jnp.zeros_like(ctx.Ti_prepared)
        )

    jpar_fluid = ctx.n_prepared * (y.vpar_i - y.vpar_e)
    jpar_em = (
        -laplacian(ctx.params, ctx.geom, ctx.psi, ctx.bcs.psi)
        if ctx.em_on
        else jnp.zeros_like(jpar_fluid)
    )
    jpar_total = jpar_fluid + jpar_em
    with jax.named_scope("parallel_current"):
        use_boundary_flux = (
            sheath_data is not None
            and str(getattr(ctx.params, "parallel_sheath_flux_mode", "boundary_flux")).lower()
            == "boundary_flux"
        )
        if use_boundary_flux:
            assert sheath_data is not None
            boundary_flux_scale = float(getattr(ctx.params, "parallel_boundary_flux_scale", 1.0))
            j_low = 0.5 * (jpar_total[0] + sheath_data.j_ghost_low) * boundary_flux_scale
            j_high = 0.5 * (jpar_total[-1] + sheath_data.j_ghost_high) * boundary_flux_scale
            dpar_j = dpar_flux_conservative(
                ctx,
                jpar_total,
                jnp.ones_like(jpar_total),
                wave=None,
                boundary_flux_low=j_low,
                boundary_flux_high=j_high,
                ghost_low_f=sheath_data.j_ghost_low,
                ghost_high_f=sheath_data.j_ghost_high,
            )
        elif hasattr(ctx.geom, "div_par"):
            dpar_j = ctx.geom.div_par(jpar_total, bc_kind="dirichlet")
        else:
            dpar_j = ctx.geom.dpar(jpar_total, bc_kind="dirichlet")

    with jax.named_scope("parallel_grad_phi_pe"):
        grad_par_phi_pe = ctx.geom.dpar(
            ctx.phi
            - ctx.n_prepared
            - float(ctx.params.alpha_Te_ohm) * ctx.Te_prepared
            - float(ctx.params.alpha_Ti_ohm) * ctx.Ti_prepared,
            bc_kind="dirichlet",
        )

    with jax.named_scope("parallel_dpar_psi"):
        dpar_psi = (
            ctx.geom.dpar(ctx.psi, bc_kind="dirichlet") if ctx.em_on else jnp.zeros_like(ctx.psi)
        )

    return ParallelVars(
        vpar_e_flux=vpar_e_flux,
        vpar_i_flux=vpar_i_flux,
        sheath_data=sheath_data,
        dpar_ve=dpar_ve,
        dpar_vi=dpar_vi,
        dpar_Te=dpar_Te,
        dpar_Ti=dpar_Ti,
        dpar_j=dpar_j,
        dpar_psi=dpar_psi,
        grad_par_phi_pe=grad_par_phi_pe,
        jpar_total=jpar_total,
    )
