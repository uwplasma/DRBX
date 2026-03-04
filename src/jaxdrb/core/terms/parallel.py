from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState

from .context import TermContext
from .ops import laplacian


def _minmod(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    s = 0.5 * (jnp.sign(a) + jnp.sign(b))
    return s * jnp.minimum(jnp.abs(a), jnp.abs(b))


def _limited_slope(f: jnp.ndarray, limiter: str) -> jnp.ndarray:
    df = f[1:] - f[:-1]
    if limiter == "none":
        slope = jnp.zeros_like(f)
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
) -> jnp.ndarray:
    slope_f = _limited_slope(f, limiter)
    slope_v = _limited_slope(v, limiter)
    f_L = f - 0.5 * slope_f
    f_R = f + 0.5 * slope_f
    v_L = v - 0.5 * slope_v
    v_R = v + 0.5 * slope_v

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
                div = div.at[0].add(-left_bndry * flux_factor_rc[0])
                div = div.at[-1].add(right_bndry * flux_factor_rp[-1])

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
    if shift.ndim != 1:
        return arr
    if int(shift.shape[0]) != int(arr.shape[0]):
        return arr

    ny = int(arr.shape[-1])
    y = jnp.arange(ny, dtype=jnp.float64)[None, :]
    y_src = (y + shift[:, None]) % float(ny)
    y0 = jnp.floor(y_src).astype(jnp.int32)
    y1 = (y0 + 1) % ny
    frac = y_src - y0
    f0 = jnp.take_along_axis(arr, y0, axis=-1)
    f1 = jnp.take_along_axis(arr, y1, axis=-1)
    return (1.0 - frac) * f0 + frac * f1


def _dpar_flux_conservative(
    ctx: TermContext,
    f: jnp.ndarray,
    v: jnp.ndarray,
    *,
    wave: jnp.ndarray | None = None,
    boundary_flux_low: jnp.ndarray | None = None,
    boundary_flux_high: jnp.ndarray | None = None,
) -> jnp.ndarray:
    grid = getattr(ctx.geom, "grid", None)
    limiter = str(ctx.params.parallel_limiter).lower()
    if grid is not None and getattr(grid, "open_field_line", False):
        use_shift = (
            getattr(ctx.geom, "to_field_aligned", None) is not None
            and str(ctx.params.parallel_transform).lower() == "shifted"
        )
        if use_shift:
            f = ctx.geom.to_field_aligned(f)
            v = ctx.geom.to_field_aligned(v)
            if wave is not None:
                wave = ctx.geom.to_field_aligned(wave)
            boundary_flux_low = _shift_boundary_flux_to_field_aligned(
                boundary_flux_low, params=ctx.params, geom=ctx.geom, z_index=0
            )
            boundary_flux_high = _shift_boundary_flux_to_field_aligned(
                boundary_flux_high, params=ctx.params, geom=ctx.geom, z_index=-1
            )
        J = getattr(ctx.geom, "jacobian", None)
        dpar_factor = getattr(ctx.geom, "dpar_factor", None)
        gpar = getattr(ctx.geom, "gpar", None) if ctx.params.use_gpar_flux else None
        sign = float(ctx.params.parallel_sign)
        scheme = str(ctx.params.parallel_flux_scheme)
        fixflux = bool(ctx.params.parallel_fixflux)
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
        )
        if use_shift:
            div = ctx.geom.from_field_aligned(div)
        return div
    return ctx.geom.dpar(f * v, bc_kind="dirichlet")


def _fastest_wave(ctx: TermContext) -> jnp.ndarray:
    """Hermes-style fastest wave estimate for parallel flux stabilization."""

    Te = ctx.Te_phys
    Ti = ctx.Ti if ctx.hot_on else None
    aa_e = jnp.maximum(float(ctx.params.me_hat), 1e-12)
    aa_i = jnp.maximum(float(getattr(ctx.params, "average_atomic_mass", 1.0)), 1e-12)

    fast = jnp.sqrt(Te / aa_e)
    total_pressure = ctx.n_phys * Te
    total_density = ctx.n_phys * aa_i
    if Ti is not None:
        fast = jnp.maximum(fast, jnp.sqrt(Ti / aa_i))
        total_pressure = total_pressure + ctx.n_phys * Ti
    sound_speed = jnp.sqrt(total_pressure / jnp.maximum(total_density, 1e-12))
    fast = jnp.maximum(fast, sound_speed)
    return fast


def _pressure_transport_coeffs(ctx: TermContext) -> tuple[float, float]:
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
    v = jnp.asarray(v)
    v_target = jnp.asarray(v_target)
    mask = jnp.asarray(mask)
    out = v
    out = out.at[0].set(jnp.where(mask[0] > 0.0, v_target[0], v[0]))
    out = out.at[-1].set(jnp.where(mask[-1] > 0.0, v_target[-1], v[-1]))
    return out


def _sheath_boundary_data(
    ctx: TermContext,
    y: DRBSystemState,
) -> dict[str, jnp.ndarray] | None:
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

    def _limit_free(fm: jnp.ndarray, fc: jnp.ndarray) -> jnp.ndarray:
        # Hermes/BOUT limited free-gradient extrapolation used for sheath ghosts.
        return jnp.where(
            jnp.logical_or(fm < fc, fm < 1e-10),
            fc,
            (fc * fc) / jnp.maximum(fm, 1e-10),
        )

    # Lower-boundary ghost values.
    n_l = ctx.n_phys[0]
    n_lp = ctx.n_phys[1]
    n_lg = _limit_free(n_lp, n_l)
    Te_l = ctx.Te_phys[0]
    Te_lp = ctx.Te_phys[1]
    Te_lg = _limit_free(Te_lp, Te_l)
    Ti_l = ctx.Ti[0]
    Ti_lp = ctx.Ti[1]
    Ti_lg = _limit_free(Ti_lp, Ti_l)
    phi_l = ctx.phi[0]
    phi_lp = ctx.phi[1]
    phi_lg = 2.0 * phi_l - phi_lp

    # Upper-boundary ghost values.
    n_u = ctx.n_phys[-1]
    n_um = ctx.n_phys[-2]
    n_ug = _limit_free(n_um, n_u)
    Te_u = ctx.Te_phys[-1]
    Te_um = ctx.Te_phys[-2]
    Te_ug = _limit_free(Te_um, Te_u)
    Ti_u = ctx.Ti[-1]
    Ti_um = ctx.Ti[-2]
    Ti_ug = _limit_free(Ti_um, Ti_u)
    phi_u = ctx.phi[-1]
    phi_um = ctx.phi[-2]
    phi_ug = 2.0 * phi_u - phi_um

    Me = jnp.maximum(float(ctx.params.me_hat), 1e-12)
    Mi = jnp.maximum(float(getattr(ctx.params, "average_atomic_mass", 1.0)), 1e-12)
    Ge = jnp.clip(float(getattr(ctx.params, "sheath_secondary_electron_coef", 0.0)), 0.0, 1.0)
    phi_wall = float(getattr(ctx.params, "sheath_wall_potential", 0.0))
    floor_potential = bool(getattr(ctx.params, "sheath_floor_potential", True))
    adiabatic_i = float(getattr(ctx.params, "sheath_ion_adiabatic", 5.0 / 3.0))
    Zi = 1.0

    # Electron sheath velocity (Hermes sheath_boundary form).
    ne_sh_l = 0.5 * (n_l + n_lg)
    ne_sh_u = 0.5 * (n_u + n_ug)
    Te_sh_l = jnp.maximum(0.5 * (Te_l + Te_lg), 1e-10)
    Te_sh_u = jnp.maximum(0.5 * (Te_u + Te_ug), 1e-10)
    phi_sh_l = 0.5 * (phi_l + phi_lg)
    phi_sh_u = 0.5 * (phi_u + phi_ug)
    if floor_potential:
        phi_sh_l = jnp.maximum(phi_sh_l, phi_wall)
        phi_sh_u = jnp.maximum(phi_sh_u, phi_wall)
    pref_l = jnp.sqrt(Te_sh_l / (2.0 * jnp.pi * Me))
    pref_u = jnp.sqrt(Te_sh_u / (2.0 * jnp.pi * Me))
    exp_l = jnp.exp(-(phi_sh_l - phi_wall) / jnp.maximum(Te_sh_l, 1e-5))
    exp_u = jnp.exp(-(phi_sh_u - phi_wall) / jnp.maximum(Te_sh_u, 1e-5))
    ve_sh_l = -(1.0 - Ge) * pref_l * exp_l
    ve_sh_u = (1.0 - Ge) * pref_u * exp_u

    # Ion sheath velocity (Bohm/Tskhakaya-like estimate).
    ni_sh_l = 0.5 * (n_l + n_lg)
    ni_sh_u = 0.5 * (n_u + n_ug)
    Ti_sh_l = jnp.maximum(0.5 * (Ti_l + Ti_lg), 1e-8)
    Ti_sh_u = jnp.maximum(0.5 * (Ti_u + Ti_ug), 1e-8)
    s_l = jnp.clip(ni_sh_l / jnp.maximum(ne_sh_l, 1e-10), 0.0, 1.0)
    s_u = jnp.clip(ni_sh_u / jnp.maximum(ne_sh_u, 1e-10), 0.0, 1.0)

    grad_ne_l = n_lp - ne_sh_l
    grad_ni_l = n_lp - ni_sh_l
    grad_ne_u = n_um - ne_sh_u
    grad_ni_u = n_um - ni_sh_u
    small_l = jnp.abs(grad_ni_l) < 1e-3
    small_u = jnp.abs(grad_ni_u) < 1e-3
    grad_ne_l = jnp.where(small_l, 1e-3, grad_ne_l)
    grad_ni_l = jnp.where(small_l, 1e-3, grad_ni_l)
    grad_ne_u = jnp.where(small_u, 1e-3, grad_ne_u)
    grad_ni_u = jnp.where(small_u, 1e-3, grad_ni_u)

    Ci2_l = jnp.clip(
        (adiabatic_i * Ti_sh_l + Zi * s_l * Te_sh_l * grad_ne_l / grad_ni_l) / Mi,
        0.0,
        100.0,
    )
    Ci2_u = jnp.clip(
        (adiabatic_i * Ti_sh_u + Zi * s_u * Te_sh_u * grad_ne_u / grad_ni_u) / Mi,
        0.0,
        100.0,
    )
    vi_sh_l = -jnp.sqrt(Ci2_l)
    vi_sh_u = jnp.sqrt(Ci2_u)

    pe_l = n_l * Te_l
    pe_lp = n_lp * Te_lp
    pe_lg = _limit_free(pe_lp, pe_l)
    pe_u = n_u * Te_u
    pe_um = n_um * Te_um
    pe_ug = _limit_free(pe_um, pe_u)

    pi_l = n_l * Ti_l
    pi_lp = n_lp * Ti_lp
    pi_lg = _limit_free(pi_lp, pi_l)
    pi_u = n_u * Ti_u
    pi_um = n_um * Ti_um
    pi_ug = _limit_free(pi_um, pi_u)

    return {
        "mask": mask,
        "sign": sign,
        "n_lg": n_lg,
        "n_ug": n_ug,
        "pe_lg": pe_lg,
        "pe_ug": pe_ug,
        "pi_lg": pi_lg,
        "pi_ug": pi_ug,
        "ve_sh_l": ve_sh_l,
        "ve_sh_u": ve_sh_u,
        "vi_sh_l": vi_sh_l,
        "vi_sh_u": vi_sh_u,
    }


def _sheath_velocity_targets(
    ctx: TermContext,
    y: DRBSystemState,
    sheath_data: dict[str, jnp.ndarray] | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if sheath_data is None:
        return y.vpar_e, y.vpar_i
    mode = str(getattr(ctx.params, "parallel_sheath_flux_mode", "boundary_flux")).lower()
    if mode != "replace_boundary":
        return y.vpar_e, y.vpar_i

    mask = sheath_data["mask"]
    ve_target = jnp.zeros_like(y.vpar_e)
    vi_target = jnp.zeros_like(y.vpar_i)
    ve_target = ve_target.at[0].set(sheath_data["ve_sh_l"])
    ve_target = ve_target.at[-1].set(sheath_data["ve_sh_u"])
    vi_target = vi_target.at[0].set(sheath_data["vi_sh_l"])
    vi_target = vi_target.at[-1].set(sheath_data["vi_sh_u"])
    return _with_boundary_targets(y.vpar_e, ve_target, mask), _with_boundary_targets(
        y.vpar_i, vi_target, mask
    )


class ParallelVars(eqx.Module):
    vpar_e_flux: jnp.ndarray
    vpar_i_flux: jnp.ndarray
    sheath_data: dict[str, jnp.ndarray] | None
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
        dpar_Te = ctx.geom.dpar(y.Te, bc_kind="neumann")
        dpar_Ti = ctx.geom.dpar(ctx.Ti, bc_kind="neumann") if ctx.hot_on else jnp.zeros_like(ctx.Ti)

    # Keep current from evolved cell-centered velocities. Sheath targets are
    # applied to boundary flux reconstruction, not directly to j_par closure.
    jpar_fluid = ctx.n_phys * (y.vpar_i - y.vpar_e)
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
            j_low = (
                sheath_data["n_lg"]
                * (sheath_data["vi_sh_l"] - sheath_data["ve_sh_l"])
                * boundary_flux_scale
            )
            j_high = (
                sheath_data["n_ug"]
                * (sheath_data["vi_sh_u"] - sheath_data["ve_sh_u"])
                * boundary_flux_scale
            )
            dpar_j = _dpar_flux_conservative(
                ctx,
                jpar_total,
                jnp.ones_like(jpar_total),
                wave=None,
                boundary_flux_low=j_low,
                boundary_flux_high=j_high,
            )
        elif hasattr(ctx.geom, "div_par"):
            dpar_j = ctx.geom.div_par(jpar_total, bc_kind="dirichlet")
        else:
            dpar_j = ctx.geom.dpar(jpar_total, bc_kind="dirichlet")

    with jax.named_scope("parallel_grad_phi_pe"):
        grad_par_phi_pe = ctx.geom.dpar(
            ctx.phi
            - ctx.n_phys
            - float(ctx.params.alpha_Te_ohm) * ctx.Te_phys
            - float(ctx.params.alpha_Ti_ohm) * ctx.Ti,
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


def parallel_conservative_terms(
    ctx: TermContext, y: DRBSystemState, par: ParallelVars
) -> DRBSystemState:
    fastest_wave = _fastest_wave(ctx)
    boundary_flux_scale = float(getattr(ctx.params, "parallel_boundary_flux_scale", 1.0))
    tau_i = float(ctx.params.tau_i) if ctx.hot_on else 0.0
    pressure_flux_coeff, pressure_work_coeff = _pressure_transport_coeffs(ctx)
    vi_par_pressure = ctx.phi + tau_i * (ctx.n_phys + ctx.Ti)
    momentum_model = str(ctx.params.parallel_momentum_model).lower()
    vpar_e_flux = par.vpar_e_flux
    vpar_i_flux = par.vpar_i_flux
    sheath_flux_mode = str(
        getattr(ctx.params, "parallel_sheath_flux_mode", "boundary_flux")
    ).lower()
    use_boundary_flux = par.sheath_data is not None and sheath_flux_mode == "boundary_flux"

    def _boundary_fluxes(
        f: jnp.ndarray, ghost_low: str, ghost_high: str, vel_low: str, vel_high: str
    ) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
        if not use_boundary_flux:
            return None, None
        assert par.sheath_data is not None
        left = (
            0.5
            * (f[0] + par.sheath_data[ghost_low])
            * par.sheath_data[vel_low]
            * boundary_flux_scale
        )
        right = (
            0.5
            * (f[-1] + par.sheath_data[ghost_high])
            * par.sheath_data[vel_high]
            * boundary_flux_scale
        )
        return left, right

    if momentum_model == "conservative":
        n_eff = jnp.maximum(ctx.n_phys, float(ctx.params.n0_min))
        n_blow, n_bhigh = _boundary_fluxes(ctx.n_phys, "n_lg", "n_ug", "ve_sh_l", "ve_sh_u")
        dn = -_dpar_flux_conservative(
            ctx,
            ctx.n_phys,
            vpar_e_flux,
            wave=fastest_wave,
            boundary_flux_low=n_blow,
            boundary_flux_high=n_bhigh,
        )
        pe = ctx.n_phys * ctx.Te_phys
        pe_blow, pe_bhigh = _boundary_fluxes(pe, "pe_lg", "pe_ug", "ve_sh_l", "ve_sh_u")
        dp_e = pressure_flux_coeff * (
            -_dpar_flux_conservative(
                ctx,
                pe,
                vpar_e_flux,
                wave=fastest_wave,
                boundary_flux_low=pe_blow,
                boundary_flux_high=pe_bhigh,
            )
        )
        if pressure_work_coeff != 0.0:
            dp_e = dp_e + pressure_work_coeff * (
                vpar_e_flux * ctx.geom.dpar(pe, bc_kind="dirichlet")
            )
        dTe = (dp_e - ctx.Te_phys * dn) / n_eff
        if ctx.hot_on:
            pi = ctx.n_phys * ctx.Ti
            pi_blow, pi_bhigh = _boundary_fluxes(pi, "pi_lg", "pi_ug", "vi_sh_l", "vi_sh_u")
            dp_i = pressure_flux_coeff * (
                -_dpar_flux_conservative(
                    ctx,
                    pi,
                    vpar_i_flux,
                    wave=fastest_wave,
                    boundary_flux_low=pi_blow,
                    boundary_flux_high=pi_bhigh,
                )
            )
            if pressure_work_coeff != 0.0:
                dp_i = dp_i + pressure_work_coeff * (
                    vpar_i_flux * ctx.geom.dpar(pi, bc_kind="dirichlet")
                )
            dTi = (dp_i - ctx.Ti * dn) / n_eff
        else:
            dTi = jnp.zeros_like(par.dpar_vi)
    elif bool(ctx.params.parallel_flux_conservative):
        n_eff = jnp.maximum(ctx.n_phys, float(ctx.params.n0_min))
        n_blow, n_bhigh = _boundary_fluxes(ctx.n_phys, "n_lg", "n_ug", "ve_sh_l", "ve_sh_u")
        dn = -_dpar_flux_conservative(
            ctx,
            ctx.n_phys,
            vpar_e_flux,
            wave=fastest_wave,
            boundary_flux_low=n_blow,
            boundary_flux_high=n_bhigh,
        )
        pe = ctx.n_phys * ctx.Te_phys
        pe_blow, pe_bhigh = _boundary_fluxes(pe, "pe_lg", "pe_ug", "ve_sh_l", "ve_sh_u")
        dp_e = pressure_flux_coeff * (
            -_dpar_flux_conservative(
                ctx,
                pe,
                vpar_e_flux,
                wave=fastest_wave,
                boundary_flux_low=pe_blow,
                boundary_flux_high=pe_bhigh,
            )
        )
        if pressure_work_coeff != 0.0:
            dp_e = dp_e + pressure_work_coeff * (
                vpar_e_flux * ctx.geom.dpar(pe, bc_kind="dirichlet")
            )
        dTe = (dp_e - ctx.Te_phys * dn) / n_eff
        if ctx.hot_on:
            pi = ctx.n_phys * ctx.Ti
            pi_blow, pi_bhigh = _boundary_fluxes(pi, "pi_lg", "pi_ug", "vi_sh_l", "vi_sh_u")
            dp_i = pressure_flux_coeff * (
                -_dpar_flux_conservative(
                    ctx,
                    pi,
                    vpar_i_flux,
                    wave=fastest_wave,
                    boundary_flux_low=pi_blow,
                    boundary_flux_high=pi_bhigh,
                )
            )
            if pressure_work_coeff != 0.0:
                dp_i = dp_i + pressure_work_coeff * (
                    vpar_i_flux * ctx.geom.dpar(pi, bc_kind="dirichlet")
                )
            dTi = (dp_i - ctx.Ti * dn) / n_eff
        else:
            dTi = jnp.zeros_like(par.dpar_vi)
    else:
        dn = -par.dpar_ve
        dTe = -(2.0 / 3.0) * par.dpar_ve
        dTi = -(2.0 / 3.0) * par.dpar_vi if ctx.hot_on else jnp.zeros_like(par.dpar_vi)

    if bool(getattr(ctx.params, "parallel_temperature_compression_on", False)):
        comp = float(getattr(ctx.params, "parallel_temperature_compression_coeff", 2.0 / 3.0))
        dTe = dTe - comp * ctx.Te_phys * par.dpar_ve
        if ctx.hot_on:
            dTi = dTi - comp * ctx.Ti * par.dpar_vi

    if momentum_model == "conservative":
        n_eff = jnp.maximum(ctx.n_phys, float(ctx.params.n0_min))
        zi = 1.0
        dNV_e = (
            -_dpar_flux_conservative(ctx, ctx.n_phys * vpar_e_flux, vpar_e_flux, wave=fastest_wave)
            - ctx.geom.dpar(pe, bc_kind="dirichlet")
            + ctx.n_phys * ctx.geom.dpar(ctx.phi, bc_kind="dirichlet")
        )
        dNV_i = -_dpar_flux_conservative(
            ctx, ctx.n_phys * vpar_i_flux, vpar_i_flux, wave=fastest_wave
        )
        if ctx.hot_on:
            dNV_i = dNV_i - ctx.geom.dpar(ctx.n_phys * ctx.Ti, bc_kind="dirichlet")
        dNV_i = dNV_i - zi * ctx.n_phys * ctx.geom.dpar(ctx.phi, bc_kind="dirichlet")
        vpar_e = (dNV_e - vpar_e_flux * dn) / n_eff
        vpar_i = (dNV_i - vpar_i_flux * dn) / n_eff
    else:
        vpar_e = par.grad_par_phi_pe / jnp.maximum(float(ctx.params.me_hat), 1e-12) - par.dpar_psi
        vpar_i = -ctx.geom.dpar(vi_par_pressure, bc_kind="dirichlet")

    psi = -par.grad_par_phi_pe if ctx.em_on else jnp.zeros_like(y.omega)

    return DRBSystemState(
        n=dn,
        omega=par.dpar_j,
        vpar_e=vpar_e,
        vpar_i=vpar_i,
        Te=dTe,
        Ti=dTi if ctx.hot_on and y.Ti is not None else None,
        psi=psi if y.psi is not None else None,
        N=None if y.N is None else None,
    )
