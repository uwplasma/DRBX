from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.core.nonlinear2d import Core2DModel, coerce_core2d_params
from jaxdrb.core.state import CoreState
from jaxdrb.operators.brackets import (
    poisson_bracket_arakawa,
    poisson_bracket_arakawa_fd,
    poisson_bracket_centered,
)
from jaxdrb.bc import BC2D

from .integrate import DiffraxSolverName, diffeqsolve as diffeqsolve_ode, diffeqsolve_fixed_steps
from .grid import Grid2D
from .fd import ddx as ddx_fd
from .fd import ddy as ddy_fd
from .fd import (
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    inv_laplacian_fd_fft,
    inv_laplacian_mixed_fft,
)
from .spectral import (
    dealias,
    ddy as ddy_spec,
    inv_laplacian,
    poisson_bracket_spectral,
)
from .neutrals import NeutralParams


class DRB2DParams(eqx.Module):
    """Conservative 2D nonlinear DRB testbed (periodic in x/y).

    This is a minimal nonlinear DRB model used to validate conservative operators
    and prepare the full nonlinear field-line system.
    """

    # Background-gradient drives (optional).
    omega_n: float = 0.0
    omega_Te: float = 0.0
    omega_drive_mask: Literal["all", "closed", "open"] = "all"

    # Parallel coupling modeled via constant k_par (optional).
    kpar: float = 0.0
    eta: float = 0.0
    me_hat: float = 0.2

    # Curvature drive (simple slab or tokamak-like interchange model).
    curvature_on: bool = False
    curvature_coeff: float = 0.0
    curvature_model: str = "slab"
    curvature_theta_scale: float | None = None
    # Optional extra scaling for curvature terms. When None, curvature uses exb_scale
    # for backward compatibility with older runs.
    curvature_scale: float | None = None

    # Polarization closure (Boussinesq vs non-Boussinesq).
    boussinesq: bool = True
    n0: float = 1.0
    n0_min: float = 1e-6
    n0_max: float | None = None
    non_boussinesq_perturbed_density_on: bool = False
    # Log-form state variables (GBS-style): evolve theta=ln n and chi=ln Te.
    log_n: bool = False
    log_Te: bool = False
    log_n_clip: float | None = 50.0
    log_Te_clip: float | None = 50.0

    # Dissipation.
    Dn: float = 0.0
    DOmega: float = 0.0
    DTe: float = 0.0

    # Hyperdiffusion (biharmonic), implemented as -D4 * ∇⁴. Useful for keeping coarse-grid
    # nonlinear movies visually turbulent without excessive Laplacian diffusion.
    Dn4: float = 0.0
    DOmega4: float = 0.0
    DTe4: float = 0.0

    # Optional drag on zonal (ky=0) vorticity, implemented as -mu * <omega>_y.
    # This is a common numerical control knob in 2D drift-wave/interchange turbulence to
    # prevent long-time condensation into a purely zonal/banded state.
    mu_zonal_omega: float = 0.0

    # Optional linear damping terms, implemented as -mu * f.
    #
    # In 2D perpendicular-box testbeds there is no explicit parallel coordinate; these
    # terms act as a simple surrogate for parallel losses / large-scale friction to help
    # reach statistically steady turbulence in long-time runs.
    mu_lin_n: float = 0.0
    mu_lin_omega: float = 0.0
    mu_lin_vpar_e: float = 0.0
    mu_lin_vpar_i: float = 0.0
    mu_lin_Te: float = 0.0

    # Numerical options.
    bracket: Literal["spectral", "arakawa", "centered"] = "arakawa"
    # For non-periodic grids, enforce zero-mean on the Poisson bracket to
    # suppress unphysical domain-mean drift from boundary discretization.
    bracket_zero_mean: bool = False
    # Optional scaling for ExB advection terms (e.g., to match alternative normalizations).
    exb_scale: float = 1.0
    poisson: Literal["spectral", "cg_fd", "mixed_fft"] = "spectral"
    poisson_preconditioner: str = "auto"
    poisson_cg_maxiter: int = 300
    poisson_cg_tol: float = 1e-8
    poisson_cg_atol: float = 0.0
    # Gauge-lifting epsilon for the Poisson solve (Neumann/mixed BCs). None uses a small default.
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
    bc_phi: BC2D | None = None
    # Non-Boussinesq variable-coefficient polarization solve settings.
    polarization_cg_maxiter: int = 400
    polarization_cg_tol: float = 1e-8
    polarization_cg_atol: float = 0.0
    polarization_preconditioner: str = "auto"
    polarization_precond_shift: float = 1e-12

    # Thermal-force coefficient in Ohm's law.
    alpha_Te_ohm: float = 1.71

    # Operator split toggles (shared pattern).
    operator_split_on: bool = False
    operator_conservative_on: bool = True
    operator_source_on: bool = True
    operator_dissipative_on: bool = True

    # Optional neutral coupling (plasma-neutral exchange).
    neutrals: NeutralParams = NeutralParams()

    # Optional SOL-like closed→open radial setup (LCFS at fixed x = x_s).
    sol_on: bool = False
    sol_xs: float = 0.0
    sol_width: float = 0.05
    sol_open_left: bool = False
    # Optional y-taper for open/closed masks (GBS-style limiter shaping).
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
    # Optional scale factor for nonlinear ExB terms in the open-field-line region.
    sol_nonlinear_open_scale: float = 1.0
    # Positivity floors for SOL closures (used in loss terms).
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
    # Optional y-taper for SOL sources (GBS iy_startsource behavior).
    sol_source_y_taper: float = 0.0
    # Parallel loss closure (field-line averaged Bohm sheath).
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
    # Optional GBS-style boundary relaxation for n, Te (left Neumann, right Dirichlet).
    sol_gbs_bc_on: bool = False
    sol_gbs_bc_nu: float = 0.0
    sol_gbs_n_right: float = 0.1
    sol_gbs_Te_right: float = 0.1
    sol_gbs_apply_y: bool = True
    # Radial boundary conditions (GBS-style): omega Dirichlet at x-boundaries.
    sol_omega_bc_dirichlet_on: bool = False
    sol_omega_bc_value: float = 0.0
    sol_omega_bc_nu: float = 1.0
    sol_omega_bc_apply_y: bool = False
    sol_vpar_bc_dirichlet_on: bool = False
    sol_vpar_bc_value: float = 0.0
    sol_vpar_bc_nu: float = 1.0
    # Optional sheath potential boundary for phi (approximate clamp).
    sol_phi_bc_on: bool = False
    sol_phi_bc_lambda: float = 3.0


class DRB2DState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
    vpar_e: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray
    N: jnp.ndarray | None = None


class DRB2DDecomposition(eqx.Module):
    conservative: DRB2DState
    source: DRB2DState
    dissipative: DRB2DState

    def total(self) -> DRB2DState:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: DRB2DState, b: DRB2DState) -> DRB2DState:
    if a.N is None and b.N is None:
        N = None
    elif a.N is None:
        N = b.N
    elif b.N is None:
        N = a.N
    else:
        N = a.N + b.N
    return DRB2DState(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        vpar_e=a.vpar_e + b.vpar_e,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
        N=N,
    )


def _state_zeros_like(y: DRB2DState) -> DRB2DState:
    z = jnp.zeros_like(y.n)
    N = None if y.N is None else jnp.zeros_like(y.N)
    return DRB2DState(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z, N=N)


def _to_core_state(y: DRB2DState) -> CoreState:
    return CoreState.from_optional(
        n=y.n,
        omega=y.omega,
        vpar_e=y.vpar_e,
        vpar_i=y.vpar_i,
        Te=y.Te,
        N=y.N,
    )


def _from_core_state(y: CoreState, *, template: DRB2DState | None = None) -> DRB2DState:
    N = y.N
    if template is not None and template.N is None:
        N = None
    return DRB2DState(n=y.n, omega=y.omega, vpar_e=y.vpar_e, vpar_i=y.vpar_i, Te=y.Te, N=N)


class DRB2DModel(eqx.Module):
    params: DRB2DParams
    grid: Grid2D
    _core: Core2DModel = eqx.field(init=False, repr=False)

    def __post_init__(self):
        core_params = coerce_core2d_params(
            self.params,
            model_kind="drb",
            hot_ion_on=False,
            em_on=False,
        )
        object.__setattr__(self, "_core", Core2DModel(params=core_params, grid=self.grid))

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
        x = self.grid.x[:, None]
        profile = jnp.exp(-0.5 * ((x - xs) / max(width, 1e-8)) ** 2)
        if y_taper is not None:
            profile = profile * y_taper
        return profile * src_mask

    def _sol_masks(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        if not self.params.sol_on:
            z = jnp.zeros((self.grid.nx, self.grid.ny), dtype=self.grid.x.dtype)
            return z, 1.0 - z
        xs = float(self.params.sol_xs)
        width = max(float(self.params.sol_width), 1e-6)
        x = self.grid.x[:, None]
        if self.params.sol_open_left:
            mask_open = 0.5 * (1.0 - jnp.tanh((x - xs) / width))
        else:
            mask_open = 0.5 * (1.0 + jnp.tanh((x - xs) / width))
        y_taper = self._sol_y_taper(float(self.params.sol_mask_y_taper))
        if y_taper is not None:
            mask_open = mask_open * y_taper
        mask_closed = 1.0 - mask_open
        return mask_closed, mask_open

    def _sol_sink_open_omega(self, omega: jnp.ndarray, mask_open: jnp.ndarray) -> jnp.ndarray:
        nu = float(self.params.sol_sink_open_omega)
        if nu == 0.0:
            return jnp.zeros_like(omega)
        mode = str(self.params.sol_sink_open_omega_mode).lower()
        if mode in {"avg", "avg_y", "fieldline", "field_line", "fieldline_avg", "field_line_avg"}:
            denom = jnp.sum(mask_open, axis=1, keepdims=True)
            denom = jnp.where(denom > 0, denom, 1.0)
            omega_avg = jnp.sum(omega * mask_open, axis=1, keepdims=True) / denom
            return -nu * mask_open * (omega - omega_avg)
        return -nu * mask_open * omega

    def _sol_parallel_loss(
        self, y: DRB2DState, phi: jnp.ndarray, mask_open: jnp.ndarray
    ) -> DRB2DState:
        if not self.params.sol_parallel_loss_on or float(self.params.sol_parallel_loss_q) <= 0.0:
            return _state_zeros_like(y)

        Te_floor = max(
            float(self.params.sol_parallel_loss_Te_floor), float(self.params.sol_Te_floor)
        )
        Te_eff = jnp.maximum(self._phys_Te(y.Te), Te_floor)
        n_floor = float(self.params.sol_n_floor)
        n_pos = jnp.maximum(self._phys_n(y.n), n_floor)
        cs = jnp.sqrt(Te_eff)

        gamma = float(self.params.sol_parallel_loss_coeff) / (
            2.0 * jnp.pi * float(self.params.sol_parallel_loss_q)
        )
        model = str(self.params.sol_parallel_loss_model).lower()
        if model == "bohm_exp":
            exp_arg = float(self.params.sol_parallel_loss_lambda) - phi / Te_eff
            exp_arg = jnp.clip(exp_arg, a_min=-10.0, a_max=10.0)
            flux = cs * jnp.exp(exp_arg)
            vpar_i_target = cs
            vpar_e_target = flux
        elif model == "bohm_linear":
            exp_arg = float(self.params.sol_parallel_loss_lambda) - phi / Te_eff
            exp_arg = jnp.clip(exp_arg, a_min=-10.0, a_max=10.0)
            flux = cs * jnp.maximum(1.0 + exp_arg, 0.0)
            vpar_i_target = cs
            vpar_e_target = flux
        else:
            flux = cs
            vpar_i_target = cs
            vpar_e_target = cs

        loss_n = -gamma * mask_open * n_pos * flux
        loss_Te = -(2.0 / 3.0) * gamma * mask_open * Te_eff * flux
        loss_omega = jnp.zeros_like(y.omega)
        loss_vpar_e = jnp.zeros_like(y.vpar_e)
        loss_vpar_i = jnp.zeros_like(y.vpar_i)

        if self.params.sol_parallel_loss_omega_on:
            loss_omega = -gamma * mask_open * (y.vpar_i - y.vpar_e)
        if self.params.sol_parallel_loss_vpar_on:
            loss_vpar_e = -gamma * mask_open * (y.vpar_e - vpar_e_target)
            loss_vpar_i = -gamma * mask_open * (y.vpar_i - vpar_i_target)

        loss_n = self._log_rhs(loss_n, n_pos, n_floor, self.params.log_n)
        loss_Te = self._log_rhs(loss_Te, Te_eff, Te_floor, self.params.log_Te)

        return DRB2DState(
            n=loss_n,
            omega=loss_omega,
            vpar_e=loss_vpar_e,
            vpar_i=loss_vpar_i,
            Te=loss_Te,
            N=None,
        )

    def omega_rhs_terms(self, y: DRB2DState) -> dict[str, jnp.ndarray]:
        return self._core.omega_rhs_terms(_to_core_state(y))

    def _sol_sheath_phi_term(
        self, y: DRB2DState, phi: jnp.ndarray, mask_open: jnp.ndarray
    ) -> DRB2DState:
        if not self.params.sol_sheath_phi_on or float(self.params.sol_parallel_loss_q) <= 0.0:
            return _state_zeros_like(y)

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

        return DRB2DState(
            n=jnp.zeros_like(y.n),
            omega=domega,
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            N=None,
        )

    def _sol_sheath_omega_sink(self, omega: jnp.ndarray, mask_open: jnp.ndarray) -> jnp.ndarray:
        """Field-line-averaged sheath dissipation for 2D SOL models.

        Applies a stable linear sink on vorticity in the open-field-line region:
          dω/dt = -γ * (ω - ⟨ω⟩_y),  γ = coeff / (2π q).
        """

        if not self.params.sol_sheath_omega_on or float(self.params.sol_parallel_loss_q) <= 0.0:
            return jnp.zeros_like(omega)
        gamma = float(self.params.sol_sheath_omega_coeff) / (
            2.0 * jnp.pi * float(self.params.sol_parallel_loss_q)
        )
        denom = jnp.sum(mask_open, axis=1, keepdims=True)
        denom = jnp.where(denom > 0.0, denom, 1.0)
        omega_avg = jnp.sum(omega * mask_open, axis=1, keepdims=True) / denom
        return -gamma * mask_open * (omega - omega_avg)

    def rhs_decomposed(self, t: float, y: DRB2DState) -> DRB2DDecomposition:
        core_state = _to_core_state(y)
        split = self._core.rhs_decomposed(t, core_state)
        return DRB2DDecomposition(
            conservative=_from_core_state(split.conservative, template=y),
            source=_from_core_state(split.source, template=y),
            dissipative=_from_core_state(split.dissipative, template=y),
        )

    def rhs(self, t: float, y: DRB2DState) -> DRB2DState:
        split = self.rhs_decomposed(t, y)
        if not self.params.operator_split_on:
            return split.total()

        out = _state_zeros_like(y)
        if self.params.operator_conservative_on:
            out = _state_add(out, split.conservative)
        if self.params.operator_source_on:
            out = _state_add(out, split.source)
        if self.params.operator_dissipative_on:
            out = _state_add(out, split.dissipative)
        return out

    def energy(self, y: DRB2DState) -> jnp.ndarray:
        return self._core.energy(_to_core_state(y))

    def energy_rate(self, y: DRB2DState, dy: DRB2DState) -> jnp.ndarray:
        return self._core.energy_rate(_to_core_state(y), _to_core_state(dy))

    def energy_budget(self, y: DRB2DState) -> dict[str, jnp.ndarray]:
        return self._core.energy_budget(_to_core_state(y))

    def diffeqsolve(
        self,
        *,
        y0: DRB2DState,
        t0: float,
        t1: float,
        dt0: float,
        save_ts: jnp.ndarray | None = None,
        solver: DiffraxSolverName = "tsit5",
        adaptive: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        max_steps: int = 200_000,
        progress: bool | None = None,
    ):
        return diffeqsolve_ode(
            self.rhs,
            y0=y0,
            t0=t0,
            t1=t1,
            dt0=dt0,
            save_ts=save_ts,
            solver=solver,
            adaptive=adaptive,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            progress=progress,
        )

    def diffeqsolve_fixed_steps(
        self,
        *,
        y0: DRB2DState,
        t0: float,
        dt: float,
        nsteps: int,
        solver: DiffraxSolverName = "dopri5",
        save_every: int = 1,
        max_steps: int | None = None,
        progress: bool | None = None,
    ):
        return diffeqsolve_fixed_steps(
            self.rhs,
            y0=y0,
            t0=t0,
            dt=dt,
            nsteps=nsteps,
            solver=solver,
            save_every=save_every,
            max_steps=max_steps,
            progress=progress,
        )
