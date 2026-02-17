from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.state import CoreSplit, CoreState
from jaxdrb.nonlinear.fd import (
    ddx as ddx_fd,
    ddy as ddy_fd,
    biharmonic as biharmonic_fd,
    enforce_bc_relaxation,
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    inv_laplacian_fd_fft,
    inv_laplacian_mixed_fft,
    laplacian as laplacian_fd,
)
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.neutrals import NeutralParams, rhs_neutral
from jaxdrb.nonlinear.spectral import (
    biharmonic,
    dealias,
    ddy as ddy_spec,
    inv_laplacian,
    laplacian,
    poisson_bracket_spectral,
    rfft2,
    irfft2,
)
from jaxdrb.operators.brackets import (
    poisson_bracket_arakawa,
    poisson_bracket_arakawa_fd,
    poisson_bracket_centered,
)


class Core2DParams(eqx.Module):
    """Unified 2D parameter set covering HW/DRB/hot-ion/EM variants."""

    # Model selection and toggles.
    model_kind: Literal["drb", "hw"] = eqx.field(static=True, default="drb")
    hot_ion_on: bool = eqx.field(static=True, default=False)
    em_on: bool = eqx.field(static=True, default=False)

    # Background-gradient drives (DRB-like).
    omega_n: float = 0.0
    omega_Te: float = 0.0
    omega_Ti: float = 0.0
    omega_drive_mask: Literal["all", "closed", "open"] = "all"

    # HW drives/coupling.
    kappa: float = 1.0
    alpha: float = 1.0
    alpha_nonzonal_only: bool = False

    # Parallel coupling modeled via constant k_par (DRB-like).
    kpar: float = 0.0
    eta: float = 0.0
    me_hat: float = 0.2

    # Electromagnetism (EM branch).
    beta: float = 0.0
    Dpsi: float = 0.0

    # Hot-ion parameters.
    tau_i: float = 1.0
    alpha_Te_ohm: float = 1.71
    alpha_Ti: float = 1.0

    # Curvature drive.
    curvature_on: bool = False
    curvature_coeff: float = 0.0
    curvature_model: str = "slab"
    curvature_theta_scale: float | None = None
    curvature_scale: float | None = None

    # Polarization closure (Boussinesq vs non-Boussinesq).
    boussinesq: bool = True
    n0: float = 1.0
    n0_min: float = 1e-6
    n0_max: float | None = None
    non_boussinesq_perturbed_density_on: bool = False

    # Log-form state variables (GBS-style).
    log_n: bool = False
    log_Te: bool = False
    log_n_clip: float | None = 50.0
    log_Te_clip: float | None = 50.0

    # Dissipation.
    Dn: float = 0.0
    DOmega: float = 0.0
    DTe: float = 0.0
    DTi: float = 0.0

    # Hyperdiffusion (biharmonic).
    Dn4: float = 0.0
    DOmega4: float = 0.0
    DTe4: float = 0.0
    DTi4: float = 0.0
    Dpsi4: float = 0.0

    # HW hyperdiffusion knobs (kept for compatibility).
    nu4_n: float = 0.0
    nu4_omega: float = 0.0

    # Optional drag/damping.
    mu_zonal_omega: float = 0.0
    mu_lin_n: float = 0.0
    mu_lin_omega: float = 0.0
    mu_lin_vpar_e: float = 0.0
    mu_lin_vpar_i: float = 0.0
    mu_lin_Te: float = 0.0

    # Numerical options.
    bracket: Literal["spectral", "arakawa", "centered"] = "arakawa"
    bracket_zero_mean: bool = False
    exb_scale: float = 1.0
    poisson: Literal["spectral", "cg_fd", "mixed_fft"] = "spectral"
    poisson_preconditioner: str = "auto"
    poisson_cg_maxiter: int = 300
    poisson_cg_tol: float = 1e-8
    poisson_cg_atol: float = 0.0
    poisson_gauge_epsilon: float | None = None
    dealias_on: bool = True
    k2_min: float = 1e-12
    bc_enforce_nu: float = 0.0

    # Optional per-field BC overrides (None -> use Grid2D bc).
    bc_n: BC2D | None = None
    bc_omega: BC2D | None = None
    bc_vpar_e: BC2D | None = None
    bc_vpar_i: BC2D | None = None
    bc_Te: BC2D | None = None
    bc_Ti: BC2D | None = None
    bc_psi: BC2D | None = None
    bc_phi: BC2D | None = None

    # Non-Boussinesq variable-coefficient polarization solve settings.
    polarization_cg_maxiter: int = 400
    polarization_cg_tol: float = 1e-8
    polarization_cg_atol: float = 0.0
    polarization_preconditioner: str = "auto"
    polarization_precond_shift: float = 1e-12

    # Operator split toggles.
    operator_split_on: bool = False
    operator_conservative_on: bool = True
    operator_source_on: bool = True
    operator_dissipative_on: bool = True

    # Optional neutral coupling (plasma-neutral exchange).
    neutrals: NeutralParams = NeutralParams()

    # Optional SOL-like closed→open radial setup.
    sol_on: bool = False
    sol_xs: float = 0.0
    sol_width: float = 0.05
    sol_open_left: bool = False
    sol_mask_y_taper: float = 0.0
    sol_n_core: float = 1.0
    sol_n_sol: float = 0.2
    sol_Te_core: float = 1.0
    sol_Te_sol: float = 0.2
    sol_relax_core: float = 0.2
    sol_relax_open: float = 0.6
    sol_sink_open_n: float = 0.0
    sol_sink_open_Te: float = 0.0
    sol_sink_open_omega: float = 0.0
    sol_sink_open_omega_mode: str = "local"
    sol_sink_open_vpar: float = 0.0
    sol_nonlinear_open_scale: float = 1.0
    sol_n_floor: float = 0.0
    sol_Te_floor: float = 0.0
    sol_source_n0: float = 0.0
    sol_source_Te0: float = 0.0
    sol_source_xs: float = 0.0
    sol_source_width: float = 1.0
    sol_source2_n0: float = 0.0
    sol_source2_Te0: float = 0.0
    sol_source2_xs: float = 0.0
    sol_source2_width: float = 1.0
    sol_source_mask: str = "all"
    sol_source_y_taper: float = 0.0
    sol_parallel_loss_on: bool = False
    sol_parallel_loss_model: Literal["bohm", "bohm_exp", "bohm_linear"] = "bohm"
    sol_parallel_loss_q: float = 4.0
    sol_parallel_loss_coeff: float = 1.0
    sol_parallel_loss_lambda: float = 3.0
    sol_parallel_loss_Te_floor: float = 1e-6
    sol_parallel_loss_vpar_on: bool = False
    sol_parallel_loss_omega_on: bool = False
    sol_sheath_omega_on: bool = False
    sol_sheath_omega_coeff: float = 1.0
    sol_sheath_phi_on: bool = False
    sol_sheath_phi_model: str = "exp"
    sol_sheath_phi_lambda: float = 3.0
    sol_sheath_phi_coeff: float = 1.0
    sol_sheath_phi_Te_floor: float = 1e-6
    sol_sheath_phi_clip: float = 10.0
    sol_gbs_bc_on: bool = False
    sol_gbs_bc_nu: float = 0.0
    sol_gbs_n_right: float = 0.1
    sol_gbs_Te_right: float = 0.1
    sol_gbs_apply_y: bool = True
    sol_omega_bc_dirichlet_on: bool = False
    sol_omega_bc_value: float = 0.0
    sol_omega_bc_nu: float = 1.0
    sol_omega_bc_apply_y: bool = False
    sol_vpar_bc_dirichlet_on: bool = False
    sol_vpar_bc_value: float = 0.0
    sol_vpar_bc_nu: float = 1.0
    sol_phi_bc_on: bool = False
    sol_phi_bc_lambda: float = 3.0


def coerce_core2d_params(
    params: Core2DParams | eqx.Module,
    **overrides,
) -> Core2DParams:
    """Return a Core2DParams instance populated from another params object."""

    if isinstance(params, Core2DParams) and not overrides:
        return params
    data: dict[str, object] = {}
    for name in Core2DParams.__dataclass_fields__:
        if hasattr(params, name):
            data[name] = getattr(params, name)
    data.update(overrides)
    return Core2DParams(**data)


class Core2DModel(eqx.Module):
    """Unified 2D RHS for DRB/HW hot-ion/EM variants."""

    params: Core2DParams
    grid: Grid2D

    def _bc_or(self, bc: BC2D | None, fallback: BC2D | None = None) -> BC2D:
        if bc is not None:
            return bc
        if fallback is not None:
            return fallback
        return self.grid.bc

    def _bc_phi(self) -> BC2D:
        return self._bc_or(self.params.bc_phi, self._bc_or(self.params.bc_omega, self.grid.bc))

    def _is_periodic_bc(self, bc: BC2D) -> bool:
        return (
            bc.kind_x == 0
            and bc.kind_y == 0
            and self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
        )

    def _is_periodic_pair(self, bc_a: BC2D, bc_b: BC2D) -> bool:
        return self._is_periodic_bc(bc_a) and self._is_periodic_bc(bc_b)

    def phi_from_omega(self, omega: jnp.ndarray, n: jnp.ndarray | None = None) -> jnp.ndarray:
        bc_phi = self._bc_phi()
        if self.params.boussinesq:
            if self.params.poisson == "spectral":
                if not self._is_periodic_bc(bc_phi):
                    raise ValueError("Spectral Poisson solve requires periodic BCs in x and y.")
                return inv_laplacian(omega, self.grid.k2, k2_min=self.params.k2_min)
            if self.params.poisson == "mixed_fft":
                return inv_laplacian_mixed_fft(
                    omega,
                    dx=self.grid.dx,
                    dy=self.grid.dy,
                    bc=bc_phi,
                    gauge_epsilon=self.params.poisson_gauge_epsilon,
                )
            precond = self.params.poisson_preconditioner
            if precond == "auto":
                precond = "spectral"
            if precond == "spectral" and not self._is_periodic_bc(bc_phi):
                precond = "jacobi"
            if self.params.poisson == "cg_fd":
                try:
                    return inv_laplacian_fd_fft(
                        omega,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=bc_phi,
                        gauge_epsilon=self.params.poisson_gauge_epsilon,
                    )
                except ValueError:
                    pass
            return inv_laplacian_cg(
                omega,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=bc_phi,
                maxiter=int(self.params.poisson_cg_maxiter),
                tol=float(self.params.poisson_cg_tol),
                atol=float(self.params.poisson_cg_atol),
                preconditioner=str(precond),
                k2_precond=self.grid.k2 if str(precond) == "spectral" else None,
                gauge_epsilon=self.params.poisson_gauge_epsilon,
            )

        if n is None:
            raise ValueError("Non-Boussinesq polarization requires density n.")
        n_eff = float(self.params.n0)
        if self.params.non_boussinesq_perturbed_density_on:
            n_eff = n_eff + jnp.real(jnp.asarray(n))
        n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
        if self.params.n0_max is not None:
            n_eff = jnp.minimum(n_eff, float(self.params.n0_max))
        if self.params.polarization_preconditioner == "auto":
            precond = "spectral_jacobi"
        else:
            precond = self.params.polarization_preconditioner
        return inv_div_n_grad_cg(
            omega,
            n_coeff=n_eff,
            dx=self.grid.dx,
            dy=self.grid.dy,
            bc=bc_phi,
            maxiter=int(self.params.polarization_cg_maxiter),
            tol=float(self.params.polarization_cg_tol),
            atol=float(self.params.polarization_cg_atol),
            preconditioner=precond,
            preconditioner_shift=float(self.params.polarization_precond_shift),
        )

    def _bracket(
        self, phi: jnp.ndarray, f: jnp.ndarray, *, bc_phi: BC2D, bc_f: BC2D
    ) -> jnp.ndarray:
        scale = float(self.params.exb_scale)
        periodic_pair = self._is_periodic_pair(bc_phi, bc_f)
        if self.params.bracket == "spectral":
            if not periodic_pair:
                raise ValueError("Spectral bracket requires periodic BCs in x and y.")
            return poisson_bracket_spectral(
                phi,
                f,
                kx=self.grid.kx,
                ky=self.grid.ky,
                dealias_mask=self.grid.dealias_mask if self.params.dealias_on else None,
            )
        if self.params.bracket == "arakawa":
            if periodic_pair:
                return poisson_bracket_arakawa(phi, f, self.grid.dx, self.grid.dy)
            j = poisson_bracket_arakawa_fd(phi, f, self.grid.dx, self.grid.dy, bc_phi, bc_f)
            if self.params.bracket_zero_mean:
                j = j - jnp.mean(j)
            if self.params.dealias_on and periodic_pair:
                return scale * dealias(j, self.grid.dealias_mask)
            return scale * j
        if periodic_pair:
            j = poisson_bracket_centered(phi, f, self.grid.dx, self.grid.dy)
        else:
            dphi_dx = ddx_fd(phi, self.grid.dx, bc_phi)
            dphi_dy = ddy_fd(phi, self.grid.dy, bc_phi)
            df_dx = ddx_fd(f, self.grid.dx, bc_f)
            df_dy = ddy_fd(f, self.grid.dy, bc_f)
            j = dphi_dx * df_dy - dphi_dy * df_dx
        if self.params.bracket_zero_mean and not periodic_pair:
            j = j - jnp.mean(j)
        if self.params.dealias_on and periodic_pair:
            return scale * dealias(j, self.grid.dealias_mask)
        return scale * j

    def _dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.params.kpar == 0.0:
            return jnp.zeros_like(f)
        return 1j * float(self.params.kpar) * f

    def _curvature(self, f: jnp.ndarray, bc_f: BC2D) -> jnp.ndarray:
        if not self.params.curvature_on or self.params.curvature_coeff == 0.0:
            return jnp.zeros_like(f)
        coeff = float(self.params.curvature_coeff)
        model = str(self.params.curvature_model).lower()
        if model in ("tokamak", "salpha", "sin", "sinusoidal"):
            theta_scale = self.params.curvature_theta_scale
            if theta_scale is None or float(theta_scale) <= 0.0:
                theta_scale = float(self.grid.Ly) / (2.0 * jnp.pi)
            theta = self.grid.y[None, :] / float(theta_scale)
            df_dx = ddx_fd(f, self.grid.dx, bc_f)
            df_dy = ddy_fd(f, self.grid.dy, bc_f)
            curv = jnp.sin(theta) * df_dx + jnp.cos(theta) * df_dy
        else:
            if self._is_periodic_bc(bc_f) and self.params.poisson == "spectral":
                df_dy = ddy_spec(f, self.grid.ky)
            else:
                df_dy = ddy_fd(f, self.grid.dy, bc_f)
            curv = df_dy
        curv = -coeff * curv
        scale = self.params.curvature_scale
        if scale is None:
            scale = float(self.params.exb_scale)
        if float(scale) != 1.0:
            curv = curv * float(scale)
        return curv

    def _phys_n(self, n: jnp.ndarray) -> jnp.ndarray:
        if not self.params.log_n:
            return n
        clip = self.params.log_n_clip
        if clip is not None and float(clip) > 0.0:
            n = jnp.clip(n, a_min=-float(clip), a_max=float(clip))
        return jnp.exp(n)

    def _phys_Te(self, Te: jnp.ndarray) -> jnp.ndarray:
        if not self.params.log_Te:
            return Te
        clip = self.params.log_Te_clip
        if clip is not None and float(clip) > 0.0:
            Te = jnp.clip(Te, a_min=-float(clip), a_max=float(clip))
        return jnp.exp(Te)

    def _log_rhs(
        self, rhs: jnp.ndarray, phys: jnp.ndarray, floor: float, log_on: bool
    ) -> jnp.ndarray:
        if not log_on:
            return rhs
        denom = jnp.maximum(phys, float(floor))
        return rhs / denom

    def _sol_y_taper(self, width: float) -> jnp.ndarray | None:
        if width <= 0.0:
            return None
        y = self.grid.y[None, :]
        Ly = float(self.grid.Ly)
        y0 = max(float(width), 1e-8)
        taper = jnp.tanh(y / y0) ** 4 * jnp.tanh((Ly - y) / y0) ** 4
        return taper

    def _sol_source_profile(
        self,
        *,
        xs: float,
        width: float,
        src_mask: jnp.ndarray,
        y_taper: jnp.ndarray | None,
    ) -> jnp.ndarray:
        width = max(width, 1e-8)
        x = self.grid.x[:, None]
        profile = jnp.exp(-0.5 * ((x - xs) / width) ** 2)
        if y_taper is not None:
            profile = profile * y_taper
        return profile * src_mask

    def _sol_masks(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        xs = float(self.params.sol_xs)
        width = max(float(self.params.sol_width), 1e-8)
        x = self.grid.x[:, None]
        if self.params.sol_open_left:
            mask_open = 0.5 * (1.0 - jnp.tanh((x - xs) / width))
        else:
            mask_open = 0.5 * (1.0 + jnp.tanh((x - xs) / width))
        mask_closed = 1.0 - mask_open
        y_taper = self._sol_y_taper(float(self.params.sol_mask_y_taper))
        if y_taper is not None:
            mask_open = mask_open * y_taper
            mask_closed = mask_closed * y_taper
        return mask_closed, mask_open

    def _sol_sink_open_omega(self, omega: jnp.ndarray, mask_open: jnp.ndarray) -> jnp.ndarray:
        mode = str(self.params.sol_sink_open_omega_mode).lower()
        nu = float(self.params.sol_sink_open_omega)
        if nu == 0.0:
            return jnp.zeros_like(omega)
        if mode in ("global", "avg", "mean"):
            denom = jnp.sum(mask_open, axis=1, keepdims=True)
            denom = jnp.where(denom > 0.0, denom, 1.0)
            omega_avg = jnp.sum(omega * mask_open, axis=1, keepdims=True) / denom
            return -nu * mask_open * (omega - omega_avg)
        return -nu * mask_open * omega

    def _sol_parallel_loss(
        self, y: CoreState, phi: jnp.ndarray, mask_open: jnp.ndarray
    ) -> CoreState:
        if not self.params.sol_parallel_loss_on:
            return CoreState.zeros_like(y)
        q = float(self.params.sol_parallel_loss_q)
        if q <= 0.0:
            return CoreState.zeros_like(y)
        coeff = float(self.params.sol_parallel_loss_coeff) / (2.0 * jnp.pi * q)
        model = str(self.params.sol_parallel_loss_model).lower()
        Te_floor = max(
            float(self.params.sol_parallel_loss_Te_floor), float(self.params.sol_Te_floor)
        )
        Te_eff = jnp.maximum(self._phys_Te(y.Te), Te_floor)
        n_floor = float(self.params.sol_n_floor)
        n_pos = jnp.maximum(self._phys_n(y.n), n_floor)
        if model in ("linear", "lin"):
            delta = phi / Te_eff - float(self.params.sol_parallel_loss_lambda)
            gamma = coeff * delta
        elif model in ("exp", "bohm_exp"):
            exp_arg = float(self.params.sol_parallel_loss_lambda) - phi / Te_eff
            exp_arg = jnp.clip(exp_arg, a_min=-10.0, a_max=10.0)
            gamma = coeff * (1.0 - jnp.exp(exp_arg))
        else:
            gamma = coeff
        cs = jnp.sqrt(Te_eff)
        loss_n = -gamma * mask_open * n_pos * cs
        loss_Te = -gamma * mask_open * 3.0 * n_pos * Te_eff * cs
        loss_vpar = jnp.zeros_like(y.vpar_e)
        if self.params.sol_parallel_loss_vpar_on:
            loss_vpar = -gamma * mask_open
        loss_omega = jnp.zeros_like(y.omega)
        if self.params.sol_parallel_loss_omega_on:
            loss_omega = -gamma * mask_open * y.omega
        return CoreState.from_optional(
            n=loss_n,
            omega=loss_omega,
            vpar_e=loss_vpar * y.vpar_e,
            vpar_i=loss_vpar * y.vpar_i,
            Te=loss_Te,
        )

    def _sol_sheath_phi_term(
        self, y: CoreState, phi: jnp.ndarray, mask_open: jnp.ndarray
    ) -> CoreState:
        if not self.params.sol_sheath_phi_on or float(self.params.sol_parallel_loss_q) <= 0.0:
            return CoreState.zeros_like(y)
        Te_floor = max(float(self.params.sol_sheath_phi_Te_floor), float(self.params.sol_Te_floor))
        Te_eff = jnp.maximum(self._phys_Te(y.Te), Te_floor)
        n_floor = float(self.params.sol_n_floor)
        n_pos = jnp.maximum(self._phys_n(y.n), n_floor)
        cs = jnp.sqrt(Te_eff)
        gamma = float(self.params.sol_sheath_phi_coeff) / (
            2.0 * jnp.pi * float(self.params.sol_parallel_loss_q)
        )
        model = str(self.params.sol_sheath_phi_model).lower()
        if model in ("linear", "lin"):
            delta = phi / Te_eff - float(self.params.sol_sheath_phi_lambda)
            clip = float(self.params.sol_sheath_phi_clip)
            if clip > 0.0:
                delta = jnp.clip(delta, a_min=-clip, a_max=clip)
            sheath_current = n_pos * cs * delta
        else:
            exp_arg = float(self.params.sol_sheath_phi_lambda) - phi / Te_eff
            clip = float(self.params.sol_sheath_phi_clip)
            exp_arg = jnp.clip(exp_arg, a_min=-clip, a_max=clip)
            sheath_current = n_pos * cs * (1.0 - jnp.exp(exp_arg))
        domega = -gamma * mask_open * sheath_current
        return CoreState.from_optional(
            n=jnp.zeros_like(y.n),
            omega=domega,
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
        )

    def _sol_sheath_omega_sink(self, omega: jnp.ndarray, mask_open: jnp.ndarray) -> jnp.ndarray:
        if not self.params.sol_sheath_omega_on or float(self.params.sol_parallel_loss_q) <= 0.0:
            return jnp.zeros_like(omega)
        gamma = float(self.params.sol_sheath_omega_coeff) / (
            2.0 * jnp.pi * float(self.params.sol_parallel_loss_q)
        )
        denom = jnp.sum(mask_open, axis=1, keepdims=True)
        denom = jnp.where(denom > 0.0, denom, 1.0)
        omega_avg = jnp.sum(omega * mask_open, axis=1, keepdims=True) / denom
        return -gamma * mask_open * (omega - omega_avg)

    def rhs_decomposed(self, t: float, y: CoreState) -> CoreSplit:
        if self.params.model_kind == "hw":
            return self._rhs_hw_decomposed(t, y)
        return self._rhs_drb_decomposed(t, y)

    def rhs(self, t: float, y: CoreState) -> CoreState:
        split = self.rhs_decomposed(t, y)
        if not self.params.operator_split_on:
            return split.total()
        out = CoreState.zeros_like(y)
        if self.params.operator_conservative_on:
            out = out.add(split.conservative)
        if self.params.operator_source_on:
            out = out.add(split.source)
        if self.params.operator_dissipative_on:
            out = out.add(split.dissipative)
        return out

    def _rhs_hw_decomposed(self, t: float, y: CoreState) -> CoreSplit:
        _ = t
        n = y.n
        omega = y.omega
        bc_n = self._bc_or(self.params.bc_n)
        bc_omega = self._bc_or(self.params.bc_omega)
        bc_phi = self._bc_phi()

        phi = self.phi_from_omega(omega)

        adv_n = self._bracket(phi, n, bc_phi=bc_phi, bc_f=bc_n)
        adv_w = self._bracket(phi, omega, bc_phi=bc_phi, bc_f=bc_omega)

        if (
            self.params.bracket == "spectral"
            and self._is_periodic_bc(bc_phi)
            and self._is_periodic_bc(bc_n)
        ):
            dphi_dy = ddy_spec(phi, self.grid.ky)
            dn_dy = ddy_spec(n, self.grid.ky)
        else:
            dphi_dy = ddy_fd(phi, self.grid.dy, bc_phi)
            dn_dy = ddy_fd(n, self.grid.dy, bc_n)

        drive_n = -float(self.params.kappa) * dphi_dy
        drive_w = -float(self.params.kappa) * dn_dy

        couple = float(self.params.alpha) * (phi - n)
        if self.params.alpha_nonzonal_only:
            couple = couple - jnp.mean(couple, axis=1, keepdims=True)

        if self._is_periodic_bc(bc_n) and self.params.poisson == "spectral":
            lap_n = laplacian(n, self.grid.k2)
        else:
            lap_n = laplacian_fd(n, self.grid.dx, self.grid.dy, bc_n)
        if self._is_periodic_bc(bc_omega) and self.params.poisson == "spectral":
            lap_w = laplacian(omega, self.grid.k2)
        else:
            lap_w = laplacian_fd(omega, self.grid.dx, self.grid.dy, bc_omega)

        dn = -adv_n + drive_n + couple + float(self.params.Dn) * lap_n
        dw = -adv_w + drive_w + couple + float(self.params.DOmega) * lap_w

        if self.params.nu4_n != 0.0 or self.params.nu4_omega != 0.0:
            if self._is_periodic_bc(bc_n) and self._is_periodic_bc(bc_omega):
                dn = dn - float(self.params.nu4_n) * biharmonic(n, self.grid.k2)
                dw = dw - float(self.params.nu4_omega) * biharmonic(omega, self.grid.k2)
            else:
                dn = dn - float(self.params.nu4_n) * biharmonic_fd(
                    n, self.grid.dx, self.grid.dy, bc_n
                )
                dw = dw - float(self.params.nu4_omega) * biharmonic_fd(
                    omega, self.grid.dx, self.grid.dy, bc_omega
                )

        if self.params.bc_enforce_nu != 0.0:
            dn = dn + enforce_bc_relaxation(
                n, dx=self.grid.dx, dy=self.grid.dy, bc=bc_n, nu=self.params.bc_enforce_nu
            )
            dw = dw + enforce_bc_relaxation(
                omega,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=bc_omega,
                nu=self.params.bc_enforce_nu,
            )

        dN = jnp.zeros_like(n)
        if self.params.neutrals.enabled:
            adv_N = self._bracket(phi, y.N, bc_phi=bc_phi, bc_f=bc_n)
            if self._is_periodic_bc(bc_n) and self.params.poisson == "spectral":
                lap_N = laplacian(y.N, self.grid.k2)
            else:
                lap_N = laplacian_fd(y.N, self.grid.dx, self.grid.dy, bc_n)
            dN, dn_from_neutrals, dw_from_neutrals = rhs_neutral(
                N=y.N,
                n=n,
                omega=omega,
                dn0=self.params.neutrals,
                adv_N=adv_N,
                lap_N=lap_N,
            )
            if self.params.bc_enforce_nu != 0.0:
                dN = dN + enforce_bc_relaxation(
                    y.N, dx=self.grid.dx, dy=self.grid.dy, bc=bc_n, nu=self.params.bc_enforce_nu
                )
            dn = dn + dn_from_neutrals
            dw = dw + dw_from_neutrals

        conservative = CoreState.from_optional(
            n=jnp.zeros_like(n),
            omega=jnp.zeros_like(omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=jnp.zeros_like(y.Ti),
            psi=jnp.zeros_like(y.psi),
            N=jnp.zeros_like(y.N),
        )
        source = CoreState.from_optional(
            n=drive_n,
            omega=drive_w,
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=jnp.zeros_like(y.Ti),
            psi=jnp.zeros_like(y.psi),
            N=jnp.zeros_like(y.N),
        )
        dissipative = CoreState.from_optional(
            n=dn - drive_n,
            omega=dw - drive_w,
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=jnp.zeros_like(y.Ti),
            psi=jnp.zeros_like(y.psi),
            N=dN,
        )
        return CoreSplit(conservative=conservative, source=source, dissipative=dissipative)

    def _rhs_drb_decomposed(self, t: float, y: CoreState) -> CoreSplit:
        _ = t
        n = y.n
        omega = y.omega
        vpar_e = y.vpar_e
        vpar_i = y.vpar_i
        Te = y.Te
        Ti = y.Ti
        psi = y.psi

        bc_n = self._bc_or(self.params.bc_n)
        bc_omega = self._bc_or(self.params.bc_omega)
        bc_vpar_e = self._bc_or(self.params.bc_vpar_e)
        bc_vpar_i = self._bc_or(self.params.bc_vpar_i)
        bc_Te = self._bc_or(self.params.bc_Te)
        bc_Ti = self._bc_or(self.params.bc_Ti)
        bc_psi = self._bc_or(self.params.bc_psi)
        bc_phi = self._bc_phi()

        if self.params.em_on:
            if self.params.poisson != "spectral":
                raise ValueError("DRB2D EM branch currently requires spectral Poisson.")
            if not (
                self._is_periodic_bc(bc_n)
                and self._is_periodic_bc(bc_omega)
                and self._is_periodic_bc(bc_psi)
                and self._is_periodic_bc(bc_vpar_i)
                and self._is_periodic_bc(bc_Te)
                and self._is_periodic_bc(bc_phi)
            ):
                raise ValueError("DRB2D EM branch currently requires periodic BCs for all fields.")

        n_phys = self._phys_n(n)
        Te_phys = self._phys_Te(Te)
        n_floor = float(self.params.sol_n_floor)
        Te_floor = float(self.params.sol_Te_floor)

        phi = self.phi_from_omega(omega, n=n_phys)
        if self.params.sol_on and self.params.sol_phi_bc_on and bc_phi.kind_x != 0:
            phi_bc = float(self.params.sol_phi_bc_lambda) * Te_phys
            if self.params.sol_open_left:
                phi = phi.at[0, :].set(phi_bc[0, :])
            else:
                phi = phi.at[-1, :].set(phi_bc[-1, :])

        mask_closed = None
        mask_open = None
        nonlinear_scale = 1.0
        if self.params.sol_on:
            mask_closed, mask_open = self._sol_masks()
            if mask_open is not None:
                nonlinear_scale = (
                    mask_closed + float(self.params.sol_nonlinear_open_scale) * mask_open
                )

        if self.params.em_on:
            psi_hat = rfft2(psi)
            jpar = -laplacian(psi, self.grid.k2)
            vpar_e = vpar_i - jpar

        adv_n_phys = -self._bracket(phi, n_phys, bc_phi=bc_phi, bc_f=bc_n) * nonlinear_scale
        adv_w = -self._bracket(phi, omega, bc_phi=bc_phi, bc_f=bc_omega) * nonlinear_scale
        adv_ve = -self._bracket(phi, vpar_e, bc_phi=bc_phi, bc_f=bc_vpar_e) * nonlinear_scale
        adv_vi = -self._bracket(phi, vpar_i, bc_phi=bc_phi, bc_f=bc_vpar_i) * nonlinear_scale
        adv_Te_phys = -self._bracket(phi, Te_phys, bc_phi=bc_phi, bc_f=bc_Te) * nonlinear_scale
        adv_Ti = -self._bracket(phi, Ti, bc_phi=bc_phi, bc_f=bc_Ti) * nonlinear_scale
        adv_psi = -self._bracket(phi, psi, bc_phi=bc_phi, bc_f=bc_psi) * nonlinear_scale

        grad_par_phi_pe = self._dpar(phi - n_phys - float(self.params.alpha_Te_ohm) * Te_phys)
        jpar = vpar_i - vpar_e

        tau_i = float(self.params.tau_i) if self.params.hot_ion_on else 0.0
        C_phi = self._curvature(phi, bc_phi)
        C_n = self._curvature(n_phys, bc_n)
        C_Te = self._curvature(Te_phys, bc_Te)
        C_Ti = self._curvature(Ti, bc_Ti) if self.params.hot_ion_on else jnp.zeros_like(Ti)
        C_p = (1.0 + tau_i) * C_n + C_Te + tau_i * C_Ti
        C_T = (2.0 / 3.0) * ((7.0 / 2.0) * C_Te + C_n - C_phi)

        dn_cons_phys = adv_n_phys - self._dpar(vpar_e)
        dTe_cons_phys = adv_Te_phys - (2.0 / 3.0) * self._dpar(vpar_e)
        conservative = CoreState.from_optional(
            n=self._log_rhs(dn_cons_phys, n_phys, n_floor, self.params.log_n),
            omega=adv_w + self._dpar(jpar),
            vpar_e=adv_ve + grad_par_phi_pe / jnp.maximum(float(self.params.me_hat), 1e-12),
            vpar_i=adv_vi - self._dpar(phi + tau_i * (n_phys + Ti)),
            Te=self._log_rhs(dTe_cons_phys, Te_phys, Te_floor, self.params.log_Te),
            Ti=adv_Ti - (2.0 / 3.0) * self._dpar(vpar_i)
            if self.params.hot_ion_on
            else jnp.zeros_like(Ti),
            psi=adv_psi,
        )

        drive_n_phys = 0.0
        drive_Te_phys = 0.0
        drive_Ti = 0.0
        if (
            self.params.omega_n != 0.0
            or self.params.omega_Te != 0.0
            or (self.params.hot_ion_on and self.params.omega_Ti != 0.0)
        ):
            if self.grid.bc.kind_y != 0 or bc_phi.kind_y != 0:
                raise ValueError("Drive terms assume periodic y for spectral ky representation.")
            dphi_dy = ddy_fd(phi, self.grid.dy, bc_phi)
            drive_mask = 1.0
            if self.params.sol_on and mask_open is not None:
                mode = str(self.params.omega_drive_mask).lower()
                if mode == "closed":
                    drive_mask = mask_closed
                elif mode == "open":
                    drive_mask = mask_open
            drive_n_phys = -float(self.params.omega_n) * dphi_dy * drive_mask
            drive_Te_phys = -float(self.params.omega_Te) * dphi_dy * drive_mask
            drive_Ti = -float(self.params.omega_Ti) * dphi_dy * drive_mask

        source = CoreState.from_optional(
            n=self._log_rhs(drive_n_phys + (C_p - C_phi), n_phys, n_floor, self.params.log_n),
            omega=C_p,
            vpar_e=jnp.zeros_like(vpar_e),
            vpar_i=jnp.zeros_like(vpar_i),
            Te=self._log_rhs(drive_Te_phys + C_T, Te_phys, Te_floor, self.params.log_Te),
            Ti=drive_Ti + C_Ti if self.params.hot_ion_on else jnp.zeros_like(Ti),
            psi=jnp.zeros_like(psi),
        )

        sol_sink_n = 0.0
        sol_sink_Te = 0.0
        sol_sink_omega = 0.0
        sol_sink_vpar = 0.0
        if self.params.sol_on and mask_open is not None:
            n_eq = (
                float(self.params.sol_n_sol)
                + (float(self.params.sol_n_core) - float(self.params.sol_n_sol)) * mask_closed
            )
            Te_eq = (
                float(self.params.sol_Te_sol)
                + (float(self.params.sol_Te_core) - float(self.params.sol_Te_sol)) * mask_closed
            )
            relax = (
                float(self.params.sol_relax_core) * mask_closed
                + float(self.params.sol_relax_open) * mask_open
            )
            sol_source_n = relax * (n_eq - n_phys)
            sol_source_Te = relax * (Te_eq - Te_phys)
            if (self.params.sol_source_n0 != 0.0) or (self.params.sol_source_Te0 != 0.0):
                xs = float(self.params.sol_source_xs)
                src_mask = 1.0
                mode = str(self.params.sol_source_mask).lower()
                if mode == "closed":
                    src_mask = mask_closed
                elif mode == "open":
                    src_mask = mask_open
                y_taper = self._sol_y_taper(float(self.params.sol_source_y_taper))
                profile = self._sol_source_profile(
                    xs=xs,
                    width=float(self.params.sol_source_width),
                    src_mask=src_mask,
                    y_taper=y_taper,
                )
                sol_source_n = sol_source_n + float(self.params.sol_source_n0) * profile
                sol_source_Te = sol_source_Te + float(self.params.sol_source_Te0) * profile
                if (self.params.sol_source2_n0 != 0.0) or (self.params.sol_source2_Te0 != 0.0):
                    xs2 = float(self.params.sol_source2_xs)
                    profile2 = self._sol_source_profile(
                        xs=xs2,
                        width=float(self.params.sol_source2_width),
                        src_mask=src_mask,
                        y_taper=y_taper,
                    )
                    sol_source_n = sol_source_n + float(self.params.sol_source2_n0) * profile2
                    sol_source_Te = sol_source_Te + float(self.params.sol_source2_Te0) * profile2
            sol_source_n = self._log_rhs(sol_source_n, n_phys, n_floor, self.params.log_n)
            sol_source_Te = self._log_rhs(sol_source_Te, Te_phys, Te_floor, self.params.log_Te)
            source = source.add(
                CoreState.from_optional(
                    n=sol_source_n,
                    omega=jnp.zeros_like(omega),
                    vpar_e=jnp.zeros_like(vpar_e),
                    vpar_i=jnp.zeros_like(vpar_i),
                    Te=sol_source_Te,
                )
            )
            n_pos = jnp.maximum(n_phys, n_floor)
            Te_pos = jnp.maximum(Te_phys, Te_floor)
            sol_sink_n = -float(self.params.sol_sink_open_n) * mask_open * n_pos
            sol_sink_Te = -float(self.params.sol_sink_open_Te) * mask_open * Te_pos
            sol_sink_omega = self._sol_sink_open_omega(omega, mask_open)
            sol_sink_vpar = -float(self.params.sol_sink_open_vpar) * mask_open

        n_diff = n_phys if self.params.log_n else n
        Te_diff = Te_phys if self.params.log_Te else Te
        if self._is_periodic_bc(bc_n) and self.params.poisson == "spectral":
            lap_n = laplacian(n_diff, self.grid.k2)
            bih_n = biharmonic(n_diff, self.grid.k2)
        else:
            lap_n = laplacian_fd(n_diff, self.grid.dx, self.grid.dy, bc_n)
            bih_n = biharmonic_fd(n_diff, self.grid.dx, self.grid.dy, bc_n)
        if self._is_periodic_bc(bc_omega) and self.params.poisson == "spectral":
            lap_w = laplacian(omega, self.grid.k2)
            bih_w = biharmonic(omega, self.grid.k2)
        else:
            lap_w = laplacian_fd(omega, self.grid.dx, self.grid.dy, bc_omega)
            bih_w = biharmonic_fd(omega, self.grid.dx, self.grid.dy, bc_omega)
        if self._is_periodic_bc(bc_Te) and self.params.poisson == "spectral":
            lap_Te = laplacian(Te_diff, self.grid.k2)
            bih_Te = biharmonic(Te_diff, self.grid.k2)
        else:
            lap_Te = laplacian_fd(Te_diff, self.grid.dx, self.grid.dy, bc_Te)
            bih_Te = biharmonic_fd(Te_diff, self.grid.dx, self.grid.dy, bc_Te)
        if self._is_periodic_bc(bc_Ti) and self.params.poisson == "spectral":
            lap_Ti = laplacian(Ti, self.grid.k2)
            bih_Ti = biharmonic(Ti, self.grid.k2)
        else:
            lap_Ti = laplacian_fd(Ti, self.grid.dx, self.grid.dy, bc_Ti)
            bih_Ti = biharmonic_fd(Ti, self.grid.dx, self.grid.dy, bc_Ti)

        omega_zonal = jnp.mean(omega, axis=1, keepdims=True) + jnp.zeros_like(omega)

        diss_n_phys = (
            float(self.params.Dn) * lap_n
            - float(self.params.Dn4) * bih_n
            - float(self.params.mu_lin_n) * n_phys
            + sol_sink_n
        )
        diss_Te_phys = (
            float(self.params.DTe) * lap_Te
            - float(self.params.DTe4) * bih_Te
            - float(self.params.mu_lin_Te) * Te_phys
            + sol_sink_Te
        )

        diss_Ti = float(self.params.DTi) * lap_Ti - float(self.params.DTi4) * bih_Ti

        diss_w = (
            float(self.params.DOmega) * lap_w
            - float(self.params.DOmega4) * bih_w
            - float(self.params.mu_zonal_omega) * omega_zonal
            - float(self.params.mu_lin_omega) * omega
            + sol_sink_omega
        )

        diss_ve = -(float(self.params.eta) / jnp.maximum(float(self.params.me_hat), 1e-12)) * (
            vpar_e - vpar_i
        )
        diss_ve = diss_ve - float(self.params.mu_lin_vpar_e) * vpar_e
        diss_vi = -float(self.params.mu_lin_vpar_i) * vpar_i
        diss_ve = diss_ve + sol_sink_vpar * vpar_e
        diss_vi = diss_vi + sol_sink_vpar * vpar_i

        if self.params.em_on:
            psi_hat = rfft2(psi)
            jpar_hat = self.grid.k2 * psi_hat
            lap_psi_hat = -self.grid.k2 * psi_hat
            diss_psi_hat = (
                -float(self.params.eta) * jpar_hat
                + float(self.params.Dpsi) * lap_psi_hat
                - float(self.params.Dpsi4) * (self.grid.k2**2) * psi_hat
            )
            diss_psi = self._psi_rhs_from_terms(diss_psi_hat)
        else:
            diss_psi = jnp.zeros_like(psi)

        dissipative = CoreState.from_optional(
            n=self._log_rhs(diss_n_phys, jnp.maximum(n_phys, n_floor), n_floor, self.params.log_n),
            omega=diss_w,
            vpar_e=diss_ve,
            vpar_i=diss_vi,
            Te=self._log_rhs(
                diss_Te_phys, jnp.maximum(Te_phys, Te_floor), Te_floor, self.params.log_Te
            ),
            Ti=diss_Ti if self.params.hot_ion_on else jnp.zeros_like(Ti),
            psi=diss_psi,
        )

        sol_par_loss = None
        sol_sheath_phi = None
        sol_sheath_omega = None
        if self.params.sol_on and self.params.sol_parallel_loss_on and mask_open is not None:
            sol_par_loss = self._sol_parallel_loss(y, phi, mask_open)
            dissipative = dissipative.add(sol_par_loss)
        if self.params.sol_on and self.params.sol_sheath_phi_on and mask_open is not None:
            sol_sheath_phi = self._sol_sheath_phi_term(y, phi, mask_open)
            dissipative = dissipative.add(sol_sheath_phi)
        if self.params.sol_on and self.params.sol_sheath_omega_on and mask_open is not None:
            sol_sheath_omega = self._sol_sheath_omega_sink(omega, mask_open)
            dissipative = dissipative.add(
                CoreState.from_optional(
                    n=jnp.zeros_like(n),
                    omega=sol_sheath_omega,
                    vpar_e=jnp.zeros_like(vpar_e),
                    vpar_i=jnp.zeros_like(vpar_i),
                    Te=jnp.zeros_like(Te),
                )
            )

        if self.params.sol_on and self.params.sol_omega_bc_dirichlet_on:
            bc_omega = BC2D(
                kind_x=1,
                kind_y=1 if self.params.sol_omega_bc_apply_y else self.grid.bc.kind_y,
                x_value=float(self.params.sol_omega_bc_value),
                y_value=float(self.grid.bc.y_value),
            )
            omega_bc = enforce_bc_relaxation(
                omega,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=bc_omega,
                nu=float(self.params.sol_omega_bc_nu),
            )
            dissipative = dissipative.add(
                CoreState.from_optional(
                    n=jnp.zeros_like(n),
                    omega=omega_bc,
                    vpar_e=jnp.zeros_like(vpar_e),
                    vpar_i=jnp.zeros_like(vpar_i),
                    Te=jnp.zeros_like(Te),
                )
            )

        if self.params.sol_on and self.params.sol_vpar_bc_dirichlet_on:
            nu_bc = float(self.params.sol_vpar_bc_nu)
            vpar_val = float(self.params.sol_vpar_bc_value)
            ny = vpar_e.shape[1]
            mask_bottom = (jnp.arange(ny) == 0).astype(vpar_e.dtype)[None, :]
            mask_top = (jnp.arange(ny) == (ny - 1)).astype(vpar_e.dtype)[None, :]
            vpar_e_bc = -nu_bc * (
                mask_bottom * (vpar_e - (-vpar_val)) + mask_top * (vpar_e - vpar_val)
            )
            vpar_i_bc = -nu_bc * (
                mask_bottom * (vpar_i - (-vpar_val)) + mask_top * (vpar_i - vpar_val)
            )
            dissipative = dissipative.add(
                CoreState.from_optional(
                    n=jnp.zeros_like(n),
                    omega=jnp.zeros_like(omega),
                    vpar_e=vpar_e_bc,
                    vpar_i=vpar_i_bc,
                    Te=jnp.zeros_like(Te),
                )
            )

        if self.params.bc_enforce_nu != 0.0:
            dissipative = dissipative.add(
                CoreState.from_optional(
                    n=enforce_bc_relaxation(
                        n,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=bc_n,
                        nu=self.params.bc_enforce_nu,
                    ),
                    omega=enforce_bc_relaxation(
                        omega,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=bc_omega,
                        nu=self.params.bc_enforce_nu,
                    ),
                    vpar_e=enforce_bc_relaxation(
                        vpar_e,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=bc_vpar_e,
                        nu=self.params.bc_enforce_nu,
                    ),
                    vpar_i=enforce_bc_relaxation(
                        vpar_i,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=bc_vpar_i,
                        nu=self.params.bc_enforce_nu,
                    ),
                    Te=enforce_bc_relaxation(
                        Te,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=bc_Te,
                        nu=self.params.bc_enforce_nu,
                    ),
                    Ti=enforce_bc_relaxation(
                        Ti,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=bc_Ti,
                        nu=self.params.bc_enforce_nu,
                    )
                    if self.params.hot_ion_on
                    else jnp.zeros_like(Ti),
                )
            )

        if self.params.sol_on and self.params.sol_gbs_bc_on and self.params.sol_gbs_bc_nu != 0.0:
            nu_bc = float(self.params.sol_gbs_bc_nu)
            n_floor = float(self.params.sol_n_floor)
            Te_floor = float(self.params.sol_Te_floor)
            n_right = float(self.params.sol_gbs_n_right)
            Te_right = float(self.params.sol_gbs_Te_right)
            if self.params.log_n:
                n_right = jnp.log(jnp.maximum(n_right, n_floor))
            if self.params.log_Te:
                Te_right = jnp.log(jnp.maximum(Te_right, Te_floor))

            nx, ny = n.shape
            mask_left = (jnp.arange(nx) == 0).astype(n.dtype)[:, None]
            mask_right = (jnp.arange(nx) == (nx - 1)).astype(n.dtype)[:, None]
            mask_bottom = (jnp.arange(ny) == 0).astype(n.dtype)[None, :]
            mask_top = (jnp.arange(ny) == (ny - 1)).astype(n.dtype)[None, :]

            n_left_target = n[1, :]
            Te_left_target = Te[1, :]
            n_right_target = jnp.full_like(n[0, :], n_right)
            Te_right_target = jnp.full_like(Te[0, :], Te_right)

            n_bc = -nu_bc * (mask_left * (n - n_left_target) + mask_right * (n - n_right_target))
            Te_bc = -nu_bc * (
                mask_left * (Te - Te_left_target) + mask_right * (Te - Te_right_target)
            )
            if self.params.sol_gbs_apply_y:
                n_bc = n_bc - nu_bc * (mask_bottom * (n - n[:, [1]]) + mask_top * (n - n[:, [-2]]))
                Te_bc = Te_bc - nu_bc * (
                    mask_bottom * (Te - Te[:, [1]]) + mask_top * (Te - Te[:, [-2]])
                )

            dissipative = dissipative.add(
                CoreState.from_optional(
                    n=n_bc,
                    omega=jnp.zeros_like(omega),
                    vpar_e=jnp.zeros_like(vpar_e),
                    vpar_i=jnp.zeros_like(vpar_i),
                    Te=Te_bc,
                )
            )

        if self.params.neutrals.enabled:
            if self._is_periodic_bc(bc_n) and self.params.poisson == "spectral":
                lap_N = laplacian(y.N, self.grid.k2)
            else:
                lap_N = laplacian_fd(y.N, self.grid.dx, self.grid.dy, bc_n)
            adv_N = -self._bracket(phi, y.N, bc_phi=bc_phi, bc_f=bc_n)
            dN, dn_from_neutrals, dw_from_neutrals = rhs_neutral(
                N=y.N,
                n=n,
                omega=omega,
                dn0=self.params.neutrals,
                adv_N=adv_N,
                lap_N=lap_N,
            )
            if self.params.bc_enforce_nu != 0.0:
                dN = dN + enforce_bc_relaxation(
                    y.N,
                    dx=self.grid.dx,
                    dy=self.grid.dy,
                    bc=bc_n,
                    nu=self.params.bc_enforce_nu,
                )
            source = source.add(
                CoreState.from_optional(
                    n=dn_from_neutrals,
                    omega=dw_from_neutrals,
                    vpar_e=jnp.zeros_like(vpar_e),
                    vpar_i=jnp.zeros_like(vpar_i),
                    Te=jnp.zeros_like(Te),
                    N=dN,
                )
            )

        return CoreSplit(conservative=conservative, source=source, dissipative=dissipative)

    def _psi_rhs_from_terms(self, term_hat: jnp.ndarray) -> jnp.ndarray:
        coef = 0.5 * float(self.params.beta) + float(self.params.me_hat) * jnp.maximum(
            self.grid.k2, self.params.k2_min
        )
        coef = jnp.maximum(coef, 1e-12)
        psi_rhs_hat = term_hat / coef
        return irfft2(psi_rhs_hat, real_output=(self.params.kpar == 0.0))
