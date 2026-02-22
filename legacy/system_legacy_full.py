from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.state import DRBSystemSplit, DRBSystemState, _state_add, _state_zeros_like
from jaxdrb.core.terms import (
    build_context,
    exb_advection_terms,
    parallel_vars,
    parallel_conservative_terms,
    curvature_terms,
    drive_terms,
    diffusion_terms,
    sol_sources,
    sol_sinks,
    sol_parallel_loss,
    sol_sheath_phi_term,
    sol_sheath_omega_sink,
    sol_omega_bc_dirichlet,
    sol_vpar_bc_dirichlet,
    sol_gbs_bc_relaxation,
    neutrals_terms,
    sheath_terms,
    line_bc_terms,
    perp_bc_relaxation,
    field_bc_relaxation,
    log_rhs,
)
from jaxdrb.operators.fd2d import (
    ddx as ddx_fd,
    ddy as ddy_fd,
    enforce_bc_relaxation,
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    inv_laplacian_fd_fft,
    inv_laplacian_mixed_fft,
    laplacian as laplacian_fd,
    biharmonic as biharmonic_fd,
)
from jaxdrb.operators.spectral2d import (
    ddx as ddx_spec,
    ddy as ddy_spec,
    inv_laplacian as inv_laplacian_spec,
    laplacian,
    biharmonic,
    dealias,
)


class DRBSystem(eqx.Module):
    """Unified DRB system model.

    The geometry adapter owns discretization details; this class owns physics.
    """

    params: DRBSystemParams
    geom: GeometryAdapter
    scheduler: object = eqx.field(init=False, static=True)

    def __post_init__(self):
        from jaxdrb.core.terms.registry import build_scheduler

        object.__setattr__(self, "scheduler", build_scheduler(self.params))

    def _grid(self):
        return getattr(self.geom, "grid", None)

    def _is_2d(self) -> bool:
        grid = self._grid()
        return grid is not None and getattr(self.geom, "ndim", None) == 2

    def _bc_or(self, bc: BC2D | None, fallback: BC2D | None = None) -> BC2D:
        if bc is not None:
            return bc
        if fallback is not None:
            return fallback
        grid = self._grid()
        if grid is not None:
            return grid.bc
        return BC2D.periodic()

    def _bc_phi(self) -> BC2D:
        return self._bc_or(self.params.bc_phi, self._bc_or(self.params.bc_omega, None))

    def _is_periodic_bc(self, bc: BC2D) -> bool:
        if bc.kind_x != 0 or bc.kind_y != 0:
            return False
        grid = self._grid()
        if grid is None:
            return True
        return grid.bc.kind_x == 0 and grid.bc.kind_y == 0

    def _is_periodic_pair(self, bc_a: BC2D, bc_b: BC2D) -> bool:
        return self._is_periodic_bc(bc_a) and self._is_periodic_bc(bc_b)

    def _phys_n(self, n: jnp.ndarray) -> jnp.ndarray:
        if not self.params.log_n:
            return n
        clip = self.params.log_n_clip
        if clip is None:
            return jnp.exp(n)
        clip_val = float(clip)
        return jnp.exp(jnp.clip(n, a_min=-clip_val, a_max=clip_val))

    def _phys_Te(self, Te: jnp.ndarray) -> jnp.ndarray:
        if not self.params.log_Te:
            return Te
        clip = self.params.log_Te_clip
        if clip is None:
            return jnp.exp(Te)
        clip_val = float(clip)
        return jnp.exp(jnp.clip(Te, a_min=-clip_val, a_max=clip_val))

    def _log_rhs(
        self, rhs: jnp.ndarray, phys: jnp.ndarray, floor: float, log_on: bool
    ) -> jnp.ndarray:
        if not log_on:
            return rhs
        denom = jnp.maximum(phys, float(floor))
        return rhs / denom

    def _ddx(self, f: jnp.ndarray, bc: BC2D) -> jnp.ndarray:
        grid = self._grid()
        if grid is None:
            return self.geom.ddx(f)
        if self._is_periodic_bc(bc) and self.params.poisson == "spectral":
            return ddx_spec(f, grid.kx)
        return ddx_fd(f, grid.dx, bc)

    def _ddy(self, f: jnp.ndarray, bc: BC2D) -> jnp.ndarray:
        grid = self._grid()
        if grid is None:
            return self.geom.ddy(f)
        if self._is_periodic_bc(bc) and self.params.poisson == "spectral":
            return ddy_spec(f, grid.ky)
        return ddy_fd(f, grid.dy, bc)

    def _laplacian(self, f: jnp.ndarray, bc: BC2D) -> jnp.ndarray:
        grid = self._grid()
        if grid is None:
            return self.geom.laplacian(f)
        if self._is_periodic_bc(bc) and self.params.poisson == "spectral":
            return laplacian(f, grid.k2)
        return laplacian_fd(f, grid.dx, grid.dy, bc)

    def _biharmonic(self, f: jnp.ndarray, bc: BC2D) -> jnp.ndarray:
        grid = self._grid()
        if grid is None:
            return self.geom.biharmonic(f)
        if self._is_periodic_bc(bc) and self.params.poisson == "spectral":
            return biharmonic(f, grid.k2)
        return biharmonic_fd(f, grid.dx, grid.dy, bc)

    def _sol_y_taper(self, width: float) -> jnp.ndarray | None:
        grid = self._grid()
        if grid is None or width <= 0.0:
            return None
        y = grid.y[None, :]
        Ly = float(grid.Ly)
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
        grid = self._grid()
        if grid is None:
            return jnp.zeros_like(src_mask)
        width = max(width, 1e-8)
        x = grid.x[:, None]
        profile = jnp.exp(-0.5 * ((x - xs) / width) ** 2)
        if y_taper is not None:
            profile = profile * y_taper
        return profile * src_mask

    def _sol_masks(self) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[None, None]:
        grid = self._grid()
        if grid is None:
            return None, None
        xs = float(self.params.sol_xs)
        width = max(float(self.params.sol_width), 1e-8)
        x = grid.x[:, None]
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

    def _sol_parallel_loss(self, y: DRBSystemState, phi: jnp.ndarray, mask_open: jnp.ndarray):
        if not self.params.sol_parallel_loss_on:
            return _state_zeros_like(y)
        q = float(self.params.sol_parallel_loss_q)
        if q <= 0.0:
            return _state_zeros_like(y)
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
        return DRBSystemState(
            n=loss_n,
            omega=loss_omega,
            vpar_e=loss_vpar * y.vpar_e,
            vpar_i=loss_vpar * y.vpar_i,
            Te=loss_Te,
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    def _sol_sheath_phi_term(self, y: DRBSystemState, phi: jnp.ndarray, mask_open: jnp.ndarray):
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
        return DRBSystemState(
            n=jnp.zeros_like(y.n),
            omega=domega,
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
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

    def _rhs_drb_split(self, t: float, y: DRBSystemState) -> DRBSystemSplit:
        _ = t
        ctx = build_context(self.params, self.geom, y)
        return self.scheduler.run(ctx, y)

    def rhs_split(self, t: float, y: DRBSystemState) -> DRBSystemSplit:
        return self._rhs_drb_split(t, y)

    def _rhs_drb_split_legacy(self, t: float, y: DRBSystemState) -> DRBSystemSplit:
        _ = t
        hot_on = bool(self.params.hot_ion_on) and (y.Ti is not None)
        em_on = bool(self.params.em_on) and (y.psi is not None)
        neut_on = (
            bool(self.params.neutrals_on) and (y.N is not None) and self.params.neutrals.enabled
        )

        Ti = y.Ti if hot_on and y.Ti is not None else jnp.zeros_like(y.Te)
        psi = y.psi if em_on and y.psi is not None else jnp.zeros_like(y.Te)

        bc_n = self._bc_or(self.params.bc_n)
        bc_omega = self._bc_or(self.params.bc_omega)
        bc_vpar_e = self._bc_or(self.params.bc_vpar_e)
        bc_vpar_i = self._bc_or(self.params.bc_vpar_i)
        bc_Te = self._bc_or(self.params.bc_Te)
        bc_Ti = self._bc_or(self.params.bc_Ti)
        bc_psi = self._bc_or(self.params.bc_psi)
        bc_phi = self._bc_phi()

        n_phys = self._phys_n(y.n)
        Te_phys = self._phys_Te(y.Te)
        n_floor = float(self.params.sol_n_floor)
        Te_floor = float(self.params.sol_Te_floor)

        phi = self._phi_from_omega(y.omega, n_phys)

        if (
            self.params.sol_on
            and self.params.sol_phi_bc_on
            and bc_phi.kind_x != 0
            and self._is_2d()
        ):
            phi_bc = float(self.params.sol_phi_bc_lambda) * Te_phys
            if self.params.sol_open_left:
                phi = phi.at[0, :].set(phi_bc[0, :])
            else:
                phi = phi.at[-1, :].set(phi_bc[-1, :])

        mask_closed = None
        mask_open = None
        nonlinear_scale = 1.0
        if self.params.sol_on and self._is_2d():
            mask_closed, mask_open = self._sol_masks()
            if mask_open is not None:
                nonlinear_scale = (
                    mask_closed + float(self.params.sol_nonlinear_open_scale) * mask_open
                )

        adv_n_phys = -self.geom.bracket(phi, n_phys, bc_phi=bc_phi, bc_f=bc_n) * nonlinear_scale
        adv_w = -self.geom.bracket(phi, y.omega, bc_phi=bc_phi, bc_f=bc_omega) * nonlinear_scale
        adv_ve = -self.geom.bracket(phi, y.vpar_e, bc_phi=bc_phi, bc_f=bc_vpar_e) * nonlinear_scale
        adv_vi = -self.geom.bracket(phi, y.vpar_i, bc_phi=bc_phi, bc_f=bc_vpar_i) * nonlinear_scale
        adv_Te_phys = -self.geom.bracket(phi, Te_phys, bc_phi=bc_phi, bc_f=bc_Te) * nonlinear_scale
        adv_Ti = -self.geom.bracket(phi, Ti, bc_phi=bc_phi, bc_f=bc_Ti) * nonlinear_scale
        adv_psi = -self.geom.bracket(phi, psi, bc_phi=bc_phi, bc_f=bc_psi) * nonlinear_scale

        dpar_ve = self.geom.dpar(y.vpar_e, bc_kind="dirichlet")
        dpar_vi = self.geom.dpar(y.vpar_i, bc_kind="dirichlet")
        dpar_Te = self.geom.dpar(y.Te, bc_kind="neumann")
        dpar_Ti = self.geom.dpar(Ti, bc_kind="neumann") if hot_on else jnp.zeros_like(Ti)

        jpar_fluid = y.vpar_i - y.vpar_e
        jpar_em = -self._laplacian(psi, bc_psi) if em_on else jnp.zeros_like(jpar_fluid)
        jpar_total = jpar_fluid + jpar_em
        dpar_j = self.geom.dpar(jpar_total, bc_kind="dirichlet")

        grad_par_phi_pe = self.geom.dpar(
            phi
            - n_phys
            - float(self.params.alpha_Te_ohm) * Te_phys
            - float(self.params.alpha_Ti_ohm) * Ti,
            bc_kind="dirichlet",
        )
        dpar_psi = self.geom.dpar(psi, bc_kind="dirichlet") if em_on else jnp.zeros_like(psi)

        tau_i = float(self.params.tau_i) if hot_on else 0.0
        vi_par_pressure = phi + tau_i * (n_phys + Ti)

        dn_cons_phys = adv_n_phys - dpar_ve
        dTe_cons_phys = adv_Te_phys - (2.0 / 3.0) * dpar_ve
        Ti_conservative = adv_Ti - (2.0 / 3.0) * dpar_vi
        psi_conservative = adv_psi - grad_par_phi_pe

        conservative = DRBSystemState(
            n=self._log_rhs(dn_cons_phys, n_phys, n_floor, self.params.log_n),
            omega=adv_w + dpar_j,
            vpar_e=adv_ve
            + grad_par_phi_pe / jnp.maximum(float(self.params.me_hat), 1e-12)
            - dpar_psi,
            vpar_i=adv_vi - self.geom.dpar(vi_par_pressure, bc_kind="dirichlet"),
            Te=self._log_rhs(dTe_cons_phys, Te_phys, Te_floor, self.params.log_Te),
            Ti=(Ti_conservative if hot_on else jnp.zeros_like(y.Ti)) if y.Ti is not None else None,
            psi=(psi_conservative if em_on else jnp.zeros_like(y.psi))
            if y.psi is not None
            else None,
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

        C_phi = self.geom.curvature(phi)
        C_n = self.geom.curvature(n_phys)
        C_Te = self.geom.curvature(Te_phys)
        C_Ti = self.geom.curvature(Ti) if hot_on else jnp.zeros_like(Ti)
        C_p = (1.0 + tau_i) * C_n + C_Te + tau_i * C_Ti
        C_T = (2.0 / 3.0) * ((7.0 / 2.0) * C_Te + C_n - C_phi)

        drive_n_phys = 0.0
        drive_Te_phys = 0.0
        drive_Ti = 0.0
        if (
            self.params.omega_n != 0.0
            or self.params.omega_Te != 0.0
            or (self.params.hot_ion_on and self.params.omega_Ti != 0.0)
        ):
            dphi_dy = self._ddy(phi, bc_phi) if self._is_2d() else self.geom.ddy(phi)
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

        source = DRBSystemState(
            n=self._log_rhs(drive_n_phys + (C_p - C_phi), n_phys, n_floor, self.params.log_n),
            omega=C_p,
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=self._log_rhs(drive_Te_phys + C_T, Te_phys, Te_floor, self.params.log_Te),
            Ti=(drive_Ti + C_Ti) if hot_on and y.Ti is not None else None,
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
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
            source = _state_add(
                source,
                DRBSystemState(
                    n=sol_source_n,
                    omega=jnp.zeros_like(y.omega),
                    vpar_e=jnp.zeros_like(y.vpar_e),
                    vpar_i=jnp.zeros_like(y.vpar_i),
                    Te=sol_source_Te,
                    Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
                    psi=None if y.psi is None else jnp.zeros_like(y.psi),
                    N=None if y.N is None else jnp.zeros_like(y.N),
                ),
            )
            n_pos = jnp.maximum(n_phys, n_floor)
            Te_pos = jnp.maximum(Te_phys, Te_floor)
            sol_sink_n = -float(self.params.sol_sink_open_n) * mask_open * n_pos
            sol_sink_Te = -float(self.params.sol_sink_open_Te) * mask_open * Te_pos
            sol_sink_omega = self._sol_sink_open_omega(y.omega, mask_open)
            sol_sink_vpar = -float(self.params.sol_sink_open_vpar) * mask_open

        n_diff = n_phys if self.params.log_n else y.n
        Te_diff = Te_phys if self.params.log_Te else y.Te
        lap_n = self._laplacian(n_diff, bc_n)
        bih_n = self._biharmonic(n_diff, bc_n)
        lap_w = self._laplacian(y.omega, bc_omega)
        bih_w = self._biharmonic(y.omega, bc_omega)
        lap_Te = self._laplacian(Te_diff, bc_Te)
        bih_Te = self._biharmonic(Te_diff, bc_Te)
        lap_Ti = self._laplacian(Ti, bc_Ti)
        bih_Ti = self._biharmonic(Ti, bc_Ti)

        omega_zonal = (
            jnp.mean(y.omega, axis=1, keepdims=True) + jnp.zeros_like(y.omega)
            if self._is_2d()
            else jnp.zeros_like(y.omega)
        )

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
            - float(self.params.mu_lin_omega) * y.omega
            + sol_sink_omega
        )

        eta_eff = self.params.eta_par if self.params.eta_par != 0.0 else self.params.eta
        me = jnp.maximum(float(self.params.me_hat), 1e-8)
        diss_ve = -(float(eta_eff) / me) * (y.vpar_e - y.vpar_i)
        diss_ve = diss_ve - float(self.params.mu_lin_vpar_e) * y.vpar_e
        diss_vi = -float(self.params.mu_lin_vpar_i) * y.vpar_i
        diss_ve = diss_ve + sol_sink_vpar * y.vpar_e
        diss_vi = diss_vi + sol_sink_vpar * y.vpar_i

        Ti_diss = (
            diss_Ti + self.params.chi_par * self.geom.dpar(dpar_Ti, bc_kind="neumann")
            if hot_on
            else None
        )
        psi_diss = (
            -float(eta_eff) * jpar_total
            + float(self.params.Dpsi) * self._laplacian(psi, bc_psi)
            - float(self.params.Dpsi4) * self._biharmonic(psi, bc_psi)
            + float(self.params.chi_par) * self.geom.dpar(dpar_psi, bc_kind="dirichlet")
            if em_on
            else None
        )

        dissipative = DRBSystemState(
            n=self._log_rhs(diss_n_phys, jnp.maximum(n_phys, n_floor), n_floor, self.params.log_n),
            omega=diss_w,
            vpar_e=diss_ve,
            vpar_i=diss_vi,
            Te=self._log_rhs(
                diss_Te_phys, jnp.maximum(Te_phys, Te_floor), Te_floor, self.params.log_Te
            ),
            Ti=(Ti_diss if hot_on else jnp.zeros_like(y.Ti)) if y.Ti is not None else None,
            psi=(psi_diss if em_on else jnp.zeros_like(y.psi)) if y.psi is not None else None,
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

        if self.params.sol_on and mask_open is not None:
            dissipative = _state_add(dissipative, self._sol_parallel_loss(y, phi, mask_open))
        if self.params.sol_on and self.params.sol_sheath_phi_on and mask_open is not None:
            dissipative = _state_add(dissipative, self._sol_sheath_phi_term(y, phi, mask_open))
        if self.params.sol_on and self.params.sol_sheath_omega_on and mask_open is not None:
            sol_sheath_omega = self._sol_sheath_omega_sink(y.omega, mask_open)
            dissipative = _state_add(
                dissipative,
                DRBSystemState(
                    n=jnp.zeros_like(y.n),
                    omega=sol_sheath_omega,
                    vpar_e=jnp.zeros_like(y.vpar_e),
                    vpar_i=jnp.zeros_like(y.vpar_i),
                    Te=jnp.zeros_like(y.Te),
                    Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
                    psi=None if y.psi is None else jnp.zeros_like(y.psi),
                    N=None if y.N is None else jnp.zeros_like(y.N),
                ),
            )

        if self.params.sol_on and self.params.sol_omega_bc_dirichlet_on and self._is_2d():
            bc_omega = BC2D(
                kind_x=1,
                kind_y=1 if self.params.sol_omega_bc_apply_y else self._grid().bc.kind_y,
                x_value=float(self.params.sol_omega_bc_value),
                y_value=float(self._grid().bc.y_value),
            )
            omega_bc = enforce_bc_relaxation(
                y.omega,
                dx=self._grid().dx,
                dy=self._grid().dy,
                bc=bc_omega,
                nu=float(self.params.sol_omega_bc_nu),
            )
            dissipative = _state_add(
                dissipative,
                DRBSystemState(
                    n=jnp.zeros_like(y.n),
                    omega=omega_bc,
                    vpar_e=jnp.zeros_like(y.vpar_e),
                    vpar_i=jnp.zeros_like(y.vpar_i),
                    Te=jnp.zeros_like(y.Te),
                    Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
                    psi=None if y.psi is None else jnp.zeros_like(y.psi),
                    N=None if y.N is None else jnp.zeros_like(y.N),
                ),
            )

        if self.params.sol_on and self.params.sol_vpar_bc_dirichlet_on and self._is_2d():
            nu_bc = float(self.params.sol_vpar_bc_nu)
            vpar_val = float(self.params.sol_vpar_bc_value)
            ny = y.vpar_e.shape[1]
            mask_bottom = (jnp.arange(ny) == 0).astype(y.vpar_e.dtype)[None, :]
            mask_top = (jnp.arange(ny) == (ny - 1)).astype(y.vpar_e.dtype)[None, :]
            vpar_e_bc = -nu_bc * (
                mask_bottom * (y.vpar_e - (-vpar_val)) + mask_top * (y.vpar_e - vpar_val)
            )
            vpar_i_bc = -nu_bc * (
                mask_bottom * (y.vpar_i - (-vpar_val)) + mask_top * (y.vpar_i - vpar_val)
            )
            dissipative = _state_add(
                dissipative,
                DRBSystemState(
                    n=jnp.zeros_like(y.n),
                    omega=jnp.zeros_like(y.omega),
                    vpar_e=vpar_e_bc,
                    vpar_i=vpar_i_bc,
                    Te=jnp.zeros_like(y.Te),
                    Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
                    psi=None if y.psi is None else jnp.zeros_like(y.psi),
                    N=None if y.N is None else jnp.zeros_like(y.N),
                ),
            )

        if self.params.bc_enforce_nu != 0.0 and self._is_2d():
            dissipative = _state_add(
                dissipative,
                DRBSystemState(
                    n=enforce_bc_relaxation(
                        y.n,
                        dx=self._grid().dx,
                        dy=self._grid().dy,
                        bc=bc_n,
                        nu=self.params.bc_enforce_nu,
                    ),
                    omega=enforce_bc_relaxation(
                        y.omega,
                        dx=self._grid().dx,
                        dy=self._grid().dy,
                        bc=bc_omega,
                        nu=self.params.bc_enforce_nu,
                    ),
                    vpar_e=enforce_bc_relaxation(
                        y.vpar_e,
                        dx=self._grid().dx,
                        dy=self._grid().dy,
                        bc=bc_vpar_e,
                        nu=self.params.bc_enforce_nu,
                    ),
                    vpar_i=enforce_bc_relaxation(
                        y.vpar_i,
                        dx=self._grid().dx,
                        dy=self._grid().dy,
                        bc=bc_vpar_i,
                        nu=self.params.bc_enforce_nu,
                    ),
                    Te=enforce_bc_relaxation(
                        y.Te,
                        dx=self._grid().dx,
                        dy=self._grid().dy,
                        bc=bc_Te,
                        nu=self.params.bc_enforce_nu,
                    ),
                    Ti=enforce_bc_relaxation(
                        Ti,
                        dx=self._grid().dx,
                        dy=self._grid().dy,
                        bc=bc_Ti,
                        nu=self.params.bc_enforce_nu,
                    )
                    if hot_on
                    else jnp.zeros_like(Ti),
                    psi=None if y.psi is None else jnp.zeros_like(y.psi),
                    N=None if y.N is None else jnp.zeros_like(y.N),
                ),
            )

        if (
            self.params.sol_on
            and self.params.sol_gbs_bc_on
            and self.params.sol_gbs_bc_nu != 0.0
            and self._is_2d()
        ):
            nu_bc = float(self.params.sol_gbs_bc_nu)
            n_floor = float(self.params.sol_n_floor)
            Te_floor = float(self.params.sol_Te_floor)
            n_right = float(self.params.sol_gbs_n_right)
            Te_right = float(self.params.sol_gbs_Te_right)
            if self.params.log_n:
                n_right = jnp.log(jnp.maximum(n_right, n_floor))
            if self.params.log_Te:
                Te_right = jnp.log(jnp.maximum(Te_right, Te_floor))

            nx, ny = y.n.shape
            mask_left = (jnp.arange(nx) == 0).astype(y.n.dtype)[:, None]
            mask_right = (jnp.arange(nx) == (nx - 1)).astype(y.n.dtype)[:, None]
            mask_bottom = (jnp.arange(ny) == 0).astype(y.n.dtype)[None, :]
            mask_top = (jnp.arange(ny) == (ny - 1)).astype(y.n.dtype)[None, :]

            n_left_target = y.n[1, :]
            Te_left_target = y.Te[1, :]
            n_right_target = jnp.full_like(y.n[0, :], n_right)
            Te_right_target = jnp.full_like(y.Te[0, :], Te_right)

            n_bc = -nu_bc * (
                mask_left * (y.n - n_left_target) + mask_right * (y.n - n_right_target)
            )
            Te_bc = -nu_bc * (
                mask_left * (y.Te - Te_left_target) + mask_right * (y.Te - Te_right_target)
            )
            if self.params.sol_gbs_apply_y:
                n_bc = n_bc - nu_bc * (
                    mask_bottom * (y.n - y.n[:, [1]]) + mask_top * (y.n - y.n[:, [-2]])
                )
                Te_bc = Te_bc - nu_bc * (
                    mask_bottom * (y.Te - y.Te[:, [1]]) + mask_top * (y.Te - y.Te[:, [-2]])
                )

            dissipative = _state_add(
                dissipative,
                DRBSystemState(
                    n=n_bc,
                    omega=jnp.zeros_like(y.omega),
                    vpar_e=jnp.zeros_like(y.vpar_e),
                    vpar_i=jnp.zeros_like(y.vpar_i),
                    Te=Te_bc,
                    Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
                    psi=None if y.psi is None else jnp.zeros_like(y.psi),
                    N=None if y.N is None else jnp.zeros_like(y.N),
                ),
            )

        if neut_on and y.N is not None:
            lap_N = self._laplacian(y.N, bc_n)
            adv_N = -self.geom.bracket(phi, y.N, bc_phi=bc_phi, bc_f=bc_n)
            dN, dn_neut, dw_neut = rhs_neutral(
                N=y.N,
                n=y.n,
                omega=y.omega,
                dn0=self.params.neutrals,
                adv_N=adv_N,
                lap_N=lap_N,
            )
            if self.params.bc_enforce_nu != 0.0 and self._is_2d():
                dN = dN + enforce_bc_relaxation(
                    y.N,
                    dx=self._grid().dx,
                    dy=self._grid().dy,
                    bc=bc_n,
                    nu=self.params.bc_enforce_nu,
                )
            source = _state_add(
                source,
                DRBSystemState(
                    n=dn_neut,
                    omega=dw_neut,
                    vpar_e=jnp.zeros_like(y.vpar_e),
                    vpar_i=jnp.zeros_like(y.vpar_i),
                    Te=jnp.zeros_like(y.Te),
                    Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
                    psi=None if y.psi is None else jnp.zeros_like(y.psi),
                    N=dN,
                ),
            )

        if self.params.perp_bc_nu != 0.0 and hasattr(self.geom, "enforce_bc_relaxation"):

            def relax(field: jnp.ndarray) -> jnp.ndarray:
                return self.geom.enforce_bc_relaxation(field, nu=float(self.params.perp_bc_nu))

            dissipative = _state_add(
                dissipative,
                DRBSystemState(
                    n=relax(y.n),
                    omega=relax(y.omega),
                    vpar_e=relax(y.vpar_e),
                    vpar_i=relax(y.vpar_i),
                    Te=relax(y.Te),
                    Ti=None if y.Ti is None else relax(y.Ti),
                    psi=None if y.psi is None else relax(y.psi),
                    N=None if y.N is None else relax(y.N),
                ),
            )

        dissipative = _state_add(dissipative, self._sheath_split(y, phi))
        dissipative = _state_add(dissipative, self._apply_line_bcs(y))

        return DRBSystemSplit(conservative=conservative, source=source, dissipative=dissipative)

    def rhs(self, t: float, y: DRBSystemState) -> DRBSystemState:
        split = self.rhs_split(t, y)
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

    def _state_from(self, y: DRBSystemState, **kwargs) -> DRBSystemState:
        return DRBSystemState(
            n=kwargs.get("n", jnp.zeros_like(y.n)),
            omega=kwargs.get("omega", jnp.zeros_like(y.omega)),
            vpar_e=kwargs.get("vpar_e", jnp.zeros_like(y.vpar_e)),
            vpar_i=kwargs.get("vpar_i", jnp.zeros_like(y.vpar_i)),
            Te=kwargs.get("Te", jnp.zeros_like(y.Te)),
            Ti=None if y.Ti is None else kwargs.get("Ti", jnp.zeros_like(y.Ti)),
            psi=None if y.psi is None else kwargs.get("psi", jnp.zeros_like(y.psi)),
            N=None if y.N is None else kwargs.get("N", jnp.zeros_like(y.N)),
        )

    def energy(self, y: DRBSystemState) -> jnp.ndarray:
        if self.params.em_on and y.psi is not None:
            return self._energy_em(y)
        if self.params.hot_ion_on and y.Ti is not None:
            return self._energy_hot(y)
        return self._energy_drb(y)

    def enstrophy(self, y: DRBSystemState) -> jnp.ndarray:
        return 0.5 * jnp.mean(jnp.real(jnp.conj(y.omega) * y.omega))

    def _energy_drb(self, y: DRBSystemState) -> jnp.ndarray:
        n = self._phys_n(y.n)
        Te = self._phys_Te(y.Te)
        phi = self._phi_from_omega(y.omega, n=n)
        bc_phi = self._bc_phi()
        c_T = 1.5 * float(self.params.alpha_Te_ohm)
        gradphi_x = self._ddx(phi, bc_phi)
        gradphi_y = self._ddy(phi, bc_phi)
        if not self.params.boussinesq:
            n_eff = float(self.params.n0)
            if self.params.non_boussinesq_perturbed_density_on:
                n_eff = n_eff + jnp.real(jnp.asarray(n))
            n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
            if self.params.n0_max is not None:
                n_eff = jnp.minimum(n_eff, float(self.params.n0_max))
            phi_term = jnp.real(n_eff) * (
                jnp.real(jnp.conj(gradphi_x) * gradphi_x)
                + jnp.real(jnp.conj(gradphi_y) * gradphi_y)
            )
        else:
            phi_term = jnp.real(jnp.conj(gradphi_x) * gradphi_x) + jnp.real(
                jnp.conj(gradphi_y) * gradphi_y
            )
        return 0.5 * jnp.mean(
            jnp.real(jnp.conj(n) * n)
            + phi_term
            + float(self.params.me_hat) * jnp.real(jnp.conj(y.vpar_e) * y.vpar_e)
            + jnp.real(jnp.conj(y.vpar_i) * y.vpar_i)
            + c_T * jnp.real(jnp.conj(Te) * Te)
        )

    def _energy_hot(self, y: DRBSystemState) -> jnp.ndarray:
        phi = self._phi_from_omega(y.omega, n=y.n)
        bc_phi = self._bc_phi()
        c_Te = 1.5 * float(self.params.alpha_Te_ohm)
        c_Ti = 1.5 * float(self.params.alpha_Ti)
        gradphi_x = self._ddx(phi, bc_phi)
        gradphi_y = self._ddy(phi, bc_phi)
        n_eff = 1.0
        if not self.params.boussinesq:
            n_eff = float(self.params.n0)
            if self.params.non_boussinesq_perturbed_density_on:
                n_eff = n_eff + jnp.real(jnp.asarray(y.n))
            n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
            if self.params.n0_max is not None:
                n_eff = jnp.minimum(n_eff, float(self.params.n0_max))
        phi_term = jnp.real(n_eff) * (
            jnp.real(jnp.conj(gradphi_x) * gradphi_x) + jnp.real(jnp.conj(gradphi_y) * gradphi_y)
        )
        return 0.5 * jnp.mean(
            jnp.real(jnp.conj(y.n) * y.n)
            + phi_term
            + float(self.params.me_hat) * jnp.real(jnp.conj(y.vpar_e) * y.vpar_e)
            + jnp.real(jnp.conj(y.vpar_i) * y.vpar_i)
            + c_Te * jnp.real(jnp.conj(y.Te) * y.Te)
            + c_Ti * jnp.real(jnp.conj(y.Ti) * y.Ti)
        )

    def _energy_em(self, y: DRBSystemState) -> jnp.ndarray:
        phi = self._phi_from_omega(y.omega, n=y.n)
        bc_phi = self._bc_phi()
        c_T = 1.5 * float(self.params.alpha_Te_ohm)
        jpar = -self._laplacian(y.psi, self._bc_or(self.params.bc_psi))
        gradphi_x = self._ddx(phi, bc_phi)
        gradphi_y = self._ddy(phi, bc_phi)
        n_eff = 1.0
        if not self.params.boussinesq:
            n_eff = float(self.params.n0)
            if self.params.non_boussinesq_perturbed_density_on:
                n_eff = n_eff + jnp.real(jnp.asarray(y.n))
            n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
            if self.params.n0_max is not None:
                n_eff = jnp.minimum(n_eff, float(self.params.n0_max))
        phi_term = jnp.real(n_eff) * (
            jnp.real(jnp.conj(gradphi_x) * gradphi_x) + jnp.real(jnp.conj(gradphi_y) * gradphi_y)
        )
        psi_term = float(self.params.beta) * jnp.real(jnp.conj(y.psi) * jpar)
        return 0.5 * jnp.mean(
            jnp.real(jnp.conj(y.n) * y.n)
            + phi_term
            + float(self.params.me_hat) * jnp.real(jnp.conj(y.vpar_e) * y.vpar_e)
            + jnp.real(jnp.conj(y.vpar_i) * y.vpar_i)
            + c_T * jnp.real(jnp.conj(y.Te) * y.Te)
            + psi_term
        )

    def energy_rate(self, y: DRBSystemState, dy: DRBSystemState) -> jnp.ndarray:
        if self.params.em_on and y.psi is not None:
            return self._energy_rate_em(y, dy)
        if self.params.hot_ion_on and y.Ti is not None:
            return self._energy_rate_hot(y, dy)
        return self._energy_rate_drb(y, dy)

    def _energy_rate_drb(self, y: DRBSystemState, dy: DRBSystemState) -> jnp.ndarray:
        if self.params.boussinesq and not (self.params.log_n or self.params.log_Te):
            n = self._phys_n(y.n)
            Te = self._phys_Te(y.Te)
            phi = self._phi_from_omega(y.omega, n=n)
            c_T = 1.5 * float(self.params.alpha_Te_ohm)
            return jnp.mean(
                jnp.real(jnp.conj(n) * dy.n)
                - jnp.real(jnp.conj(phi) * dy.omega)
                + float(self.params.me_hat) * jnp.real(jnp.conj(y.vpar_e) * dy.vpar_e)
                + jnp.real(jnp.conj(y.vpar_i) * dy.vpar_i)
                + c_T * jnp.real(jnp.conj(Te) * dy.Te)
            )

        _, edot = jax.jvp(self._energy_drb, (y,), (dy,))
        return edot

    def _energy_rate_hot(self, y: DRBSystemState, dy: DRBSystemState) -> jnp.ndarray:
        if self.params.boussinesq and not (self.params.log_n or self.params.log_Te):
            phi = self._phi_from_omega(y.omega, n=y.n)
            c_Te = 1.5 * float(self.params.alpha_Te_ohm)
            c_Ti = 1.5 * float(self.params.alpha_Ti)
            return jnp.mean(
                jnp.real(jnp.conj(y.n) * dy.n)
                - jnp.real(jnp.conj(phi) * dy.omega)
                + float(self.params.me_hat) * jnp.real(jnp.conj(y.vpar_e) * dy.vpar_e)
                + jnp.real(jnp.conj(y.vpar_i) * dy.vpar_i)
                + c_Te * jnp.real(jnp.conj(y.Te) * dy.Te)
                + c_Ti * jnp.real(jnp.conj(y.Ti) * dy.Ti)
            )

        _, edot = jax.jvp(self._energy_hot, (y,), (dy,))
        return edot

    def _energy_rate_em(self, y: DRBSystemState, dy: DRBSystemState) -> jnp.ndarray:
        if self.params.boussinesq and not (self.params.log_n or self.params.log_Te):
            phi = self._phi_from_omega(y.omega, n=y.n)
            c_T = 1.5 * float(self.params.alpha_Te_ohm)
            jpar = -self._laplacian(y.psi, self._bc_or(self.params.bc_psi))
            return jnp.mean(
                jnp.real(jnp.conj(y.n) * dy.n)
                - jnp.real(jnp.conj(phi) * dy.omega)
                + float(self.params.me_hat) * jnp.real(jnp.conj(y.vpar_e) * dy.vpar_e)
                + jnp.real(jnp.conj(y.vpar_i) * dy.vpar_i)
                + c_T * jnp.real(jnp.conj(y.Te) * dy.Te)
                + float(self.params.beta) * jnp.real(jnp.conj(jpar) * dy.psi)
            )

        _, edot = jax.jvp(self._energy_em, (y,), (dy,))
        return edot

    def energy_budget(self, y: DRBSystemState) -> dict[str, jnp.ndarray]:
        if self._is_2d():
            if self.params.em_on and y.psi is not None:
                return self._energy_budget_em(y)
            if self.params.hot_ion_on and y.Ti is not None:
                return self._energy_budget_hot(y)
            return self._energy_budget_drb(y)

        split = self.rhs_split(0.0, y)
        phi = self._phi_from_omega(y.omega, y.n)
        dy_sh = self._sheath_split(y, phi)
        total = self.energy_rate(y, split.total())
        conservative = self.energy_rate(y, split.conservative)
        source = self.energy_rate(y, split.source)
        dissipative = self.energy_rate(y, split.dissipative)
        sheath = self.energy_rate(y, dy_sh)
        dissipative_other = dissipative - sheath
        residual = total - (conservative + source + dissipative_other + sheath)
        return {
            "total": total,
            "conservative": conservative,
            "source": source,
            "dissipative_other": dissipative_other,
            "sheath": sheath,
            "residual": residual,
        }

    def _energy_budget_drb(self, y: DRBSystemState) -> dict[str, jnp.ndarray]:
        n = self._phys_n(y.n)
        omega = y.omega
        vpar_e = y.vpar_e
        vpar_i = y.vpar_i
        Te = self._phys_Te(y.Te)
        bc_n = self._bc_or(self.params.bc_n)
        bc_omega = self._bc_or(self.params.bc_omega)
        bc_vpar_e = self._bc_or(self.params.bc_vpar_e)
        bc_vpar_i = self._bc_or(self.params.bc_vpar_i)
        bc_Te = self._bc_or(self.params.bc_Te)
        bc_phi = self._bc_phi()
        n_floor = float(self.params.sol_n_floor)
        Te_floor = float(self.params.sol_Te_floor)

        phi = self._phi_from_omega(omega, n=n)

        mask_open = None
        nonlinear_scale = 1.0
        if self.params.sol_on:
            mask_closed, mask_open = self._sol_masks()
            if mask_open is not None:
                nonlinear_scale = (
                    mask_closed + float(self.params.sol_nonlinear_open_scale) * mask_open
                )

        adv_n = self._log_rhs(
            -self.geom.bracket(phi, n, bc_phi=bc_phi, bc_f=bc_n) * nonlinear_scale,
            n,
            n_floor,
            self.params.log_n,
        )
        adv_w = -self.geom.bracket(phi, omega, bc_phi=bc_phi, bc_f=bc_omega) * nonlinear_scale
        adv_ve = -self.geom.bracket(phi, vpar_e, bc_phi=bc_phi, bc_f=bc_vpar_e) * nonlinear_scale
        adv_vi = -self.geom.bracket(phi, vpar_i, bc_phi=bc_phi, bc_f=bc_vpar_i) * nonlinear_scale
        adv_Te = self._log_rhs(
            -self.geom.bracket(phi, Te, bc_phi=bc_phi, bc_f=bc_Te) * nonlinear_scale,
            Te,
            Te_floor,
            self.params.log_Te,
        )

        grad_par_phi_pe = self.geom.dpar(
            phi - n - float(self.params.alpha_Te_ohm) * Te, bc_kind="dirichlet"
        )
        jpar = vpar_i - vpar_e
        par_n = self._log_rhs(
            -self.geom.dpar(vpar_e, bc_kind="dirichlet"),
            n,
            n_floor,
            self.params.log_n,
        )
        par_w = self.geom.dpar(jpar, bc_kind="dirichlet")
        par_ve = grad_par_phi_pe / jnp.maximum(float(self.params.me_hat), 1e-12)
        par_vi = -self.geom.dpar(phi, bc_kind="dirichlet")
        par_Te = self._log_rhs(
            -(2.0 / 3.0) * self.geom.dpar(vpar_e, bc_kind="dirichlet"),
            Te,
            Te_floor,
            self.params.log_Te,
        )

        C_phi = self.geom.curvature(phi)
        C_n = self.geom.curvature(n)
        C_Te = self.geom.curvature(Te)
        C_p = C_n + C_Te
        C_T = (2.0 / 3.0) * ((7.0 / 2.0) * C_Te + C_n - C_phi)

        drive_n = 0.0
        drive_Te = 0.0
        if self.params.omega_n != 0.0 or self.params.omega_Te != 0.0:
            dphi_dy = self._ddy(phi, bc_phi)
            drive_n = self._log_rhs(
                -float(self.params.omega_n) * dphi_dy, n, n_floor, self.params.log_n
            )
            drive_Te = self._log_rhs(
                -float(self.params.omega_Te) * dphi_dy, Te, Te_floor, self.params.log_Te
            )

        sol_source_n = 0.0
        sol_source_Te = 0.0
        sol_sink_n = 0.0
        sol_sink_Te = 0.0
        sol_sink_omega = 0.0
        sol_sink_vpar = 0.0
        if self.params.sol_on and mask_open is not None:
            mask_closed, mask_open = self._sol_masks()
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
            sol_source_n = relax * (n_eq - n)
            sol_source_Te = relax * (Te_eq - Te)
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
            sol_source_n = self._log_rhs(sol_source_n, n, n_floor, self.params.log_n)
            sol_source_Te = self._log_rhs(sol_source_Te, Te, Te_floor, self.params.log_Te)
            n_pos = jnp.maximum(n, n_floor)
            Te_pos = jnp.maximum(Te, Te_floor)
            sol_sink_n = self._log_rhs(
                -float(self.params.sol_sink_open_n) * mask_open * n_pos,
                n_pos,
                n_floor,
                self.params.log_n,
            )
            sol_sink_Te = self._log_rhs(
                -float(self.params.sol_sink_open_Te) * mask_open * Te_pos,
                Te_pos,
                Te_floor,
                self.params.log_Te,
            )
            sol_sink_omega = self._sol_sink_open_omega(omega, mask_open)
            sol_sink_vpar = -float(self.params.sol_sink_open_vpar) * mask_open

        sol_par_loss = None
        sol_sheath_phi = None
        sol_sheath_omega = None
        if self.params.sol_on and self.params.sol_parallel_loss_on and mask_open is not None:
            sol_par_loss = self._sol_parallel_loss(y, phi, mask_open)
        if self.params.sol_on and self.params.sol_sheath_phi_on and mask_open is not None:
            sol_sheath_phi = self._sol_sheath_phi_term(y, phi, mask_open)
        if self.params.sol_on and self.params.sol_sheath_omega_on and mask_open is not None:
            sol_sheath_omega = self._sol_sheath_omega_sink(omega, mask_open)

        lap_n = self._laplacian(n, bc_n)
        bih_n = self._biharmonic(n, bc_n)
        lap_w = self._laplacian(omega, bc_omega)
        bih_w = self._biharmonic(omega, bc_omega)
        lap_Te = self._laplacian(Te, bc_Te)
        bih_Te = self._biharmonic(Te, bc_Te)

        omega_zonal = jnp.mean(omega, axis=1, keepdims=True) + jnp.zeros_like(omega)

        diss_n = (
            self._log_rhs(
                float(self.params.Dn) * lap_n - float(self.params.Dn4) * bih_n,
                n,
                n_floor,
                self.params.log_n,
            )
            + sol_sink_n
        )
        diss_w = (
            float(self.params.DOmega) * lap_w
            - float(self.params.DOmega4) * bih_w
            - float(self.params.mu_zonal_omega) * omega_zonal
            + sol_sink_omega
        )
        diss_ve = -(float(self.params.eta) / jnp.maximum(float(self.params.me_hat), 1e-12)) * (
            vpar_e - vpar_i
        )
        diss_ve = diss_ve + sol_sink_vpar * vpar_e
        diss_vi = sol_sink_vpar * vpar_i
        diss_Te = (
            self._log_rhs(
                float(self.params.DTe) * lap_Te - float(self.params.DTe4) * bih_Te,
                Te,
                Te_floor,
                self.params.log_Te,
            )
            + sol_sink_Te
        )

        if self.params.bc_enforce_nu != 0.0 and self._grid() is not None:
            diss_n = diss_n + enforce_bc_relaxation(
                n, dx=self._grid().dx, dy=self._grid().dy, bc=bc_n, nu=self.params.bc_enforce_nu
            )
            diss_w = diss_w + enforce_bc_relaxation(
                omega,
                dx=self._grid().dx,
                dy=self._grid().dy,
                bc=bc_omega,
                nu=self.params.bc_enforce_nu,
            )
            diss_ve = diss_ve + enforce_bc_relaxation(
                vpar_e,
                dx=self._grid().dx,
                dy=self._grid().dy,
                bc=bc_vpar_e,
                nu=self.params.bc_enforce_nu,
            )
            diss_vi = diss_vi + enforce_bc_relaxation(
                vpar_i,
                dx=self._grid().dx,
                dy=self._grid().dy,
                bc=bc_vpar_i,
                nu=self.params.bc_enforce_nu,
            )
            diss_Te = diss_Te + enforce_bc_relaxation(
                Te, dx=self._grid().dx, dy=self._grid().dy, bc=bc_Te, nu=self.params.bc_enforce_nu
            )

        edot_adv = self.energy_rate(
            y,
            self._state_from(y, n=adv_n, omega=adv_w, vpar_e=adv_ve, vpar_i=adv_vi, Te=adv_Te),
        )
        edot_par = self.energy_rate(
            y,
            self._state_from(y, n=par_n, omega=par_w, vpar_e=par_ve, vpar_i=par_vi, Te=par_Te),
        )
        edot_curv = self.energy_rate(
            y,
            self._state_from(
                y,
                n=self._log_rhs(C_p - C_phi, n, n_floor, self.params.log_n),
                omega=C_p,
                Te=self._log_rhs(C_T, Te, Te_floor, self.params.log_Te),
            ),
        )
        edot_drive = self.energy_rate(
            y,
            self._state_from(y, n=drive_n, omega=jnp.zeros_like(omega), Te=drive_Te),
        )
        if self.params.sol_on:
            edot_sol_relax = self.energy_rate(
                y,
                self._state_from(
                    y,
                    n=sol_source_n,
                    omega=jnp.zeros_like(omega),
                    Te=sol_source_Te,
                ),
            )
            edot_sol_sink = self.energy_rate(
                y,
                self._state_from(
                    y,
                    n=sol_sink_n,
                    omega=sol_sink_omega,
                    vpar_e=sol_sink_vpar * vpar_e,
                    vpar_i=sol_sink_vpar * vpar_i,
                    Te=sol_sink_Te,
                ),
            )
            edot_sol_par_loss = None
            if sol_par_loss is not None:
                edot_sol_par_loss = self.energy_rate(y, sol_par_loss)
            edot_sol_sheath_phi = None
            if sol_sheath_phi is not None:
                edot_sol_sheath_phi = self.energy_rate(y, sol_sheath_phi)
            edot_sol_sheath_omega = None
            if sol_sheath_omega is not None:
                edot_sol_sheath_omega = self.energy_rate(
                    y,
                    self._state_from(y, omega=sol_sheath_omega),
                )
            edot_sol_omega_bc = None
            if self.params.sol_omega_bc_dirichlet_on and self._grid() is not None:
                bc_omega_dir = BC2D(
                    kind_x=1,
                    kind_y=self._grid().bc.kind_y,
                    x_value=float(self.params.sol_omega_bc_value),
                    y_value=float(self._grid().bc.y_value),
                )
                omega_bc = enforce_bc_relaxation(
                    omega,
                    dx=self._grid().dx,
                    dy=self._grid().dy,
                    bc=bc_omega_dir,
                    nu=float(self.params.sol_omega_bc_nu),
                )
                edot_sol_omega_bc = self.energy_rate(
                    y,
                    self._state_from(y, omega=omega_bc),
                )
        edot_diss = self.energy_rate(
            y,
            self._state_from(y, n=diss_n, omega=diss_w, vpar_e=diss_ve, vpar_i=diss_vi, Te=diss_Te),
        )

        out = {
            "E_dot_adv": edot_adv,
            "E_dot_parallel": edot_par,
            "E_dot_curvature": edot_curv,
            "E_dot_drive": edot_drive,
            "E_dot_diss": edot_diss,
        }
        if self.params.sol_on:
            out["E_dot_sol_relax"] = edot_sol_relax
            out["E_dot_sol_sink"] = edot_sol_sink
            if edot_sol_par_loss is not None:
                out["E_dot_sol_par_loss"] = edot_sol_par_loss
            if edot_sol_sheath_phi is not None:
                out["E_dot_sol_sheath_phi"] = edot_sol_sheath_phi
            if edot_sol_sheath_omega is not None:
                out["E_dot_sol_sheath_omega"] = edot_sol_sheath_omega
            if edot_sol_omega_bc is not None:
                out["E_dot_sol_omega_bc"] = edot_sol_omega_bc
            if self.params.sol_vpar_bc_dirichlet_on and self._grid() is not None:
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
                edot_sol_vpar_bc = self.energy_rate(
                    y,
                    self._state_from(y, vpar_e=vpar_e_bc, vpar_i=vpar_i_bc),
                )
                out["E_dot_sol_vpar_bc"] = edot_sol_vpar_bc
        if y.N is not None and self.params.neutrals.enabled and self._grid() is not None:
            lap_N = self._laplacian(y.N, bc_n)
            adv_N = -self.geom.bracket(phi, y.N, bc_phi=bc_phi, bc_f=bc_n)
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
                    dx=self._grid().dx,
                    dy=self._grid().dy,
                    bc=bc_n,
                    nu=self.params.bc_enforce_nu,
                )
            edot_neutrals = self.energy_rate(
                y,
                self._state_from(y, n=dn_from_neutrals, omega=dw_from_neutrals, N=dN),
            )
            out["E_dot_neutrals"] = edot_neutrals

        out["E_dot_total"] = (
            out["E_dot_adv"]
            + out["E_dot_parallel"]
            + out["E_dot_curvature"]
            + out["E_dot_drive"]
            + out["E_dot_diss"]
            + out.get("E_dot_sol_relax", 0.0)
            + out.get("E_dot_sol_sink", 0.0)
            + out.get("E_dot_sol_par_loss", 0.0)
            + out.get("E_dot_sol_omega_bc", 0.0)
            + out.get("E_dot_sol_sheath_phi", 0.0)
            + out.get("E_dot_sol_sheath_omega", 0.0)
            + out.get("E_dot_sol_vpar_bc", 0.0)
            + out.get("E_dot_neutrals", 0.0)
        )
        return out

    def _energy_budget_hot(self, y: DRBSystemState) -> dict[str, jnp.ndarray]:
        n = y.n
        omega = y.omega
        vpar_e = y.vpar_e
        vpar_i = y.vpar_i
        Te = y.Te
        Ti = y.Ti
        phi = self._phi_from_omega(omega, n=n)
        bc_n = self._bc_or(self.params.bc_n)
        bc_omega = self._bc_or(self.params.bc_omega)
        bc_vpar_e = self._bc_or(self.params.bc_vpar_e)
        bc_vpar_i = self._bc_or(self.params.bc_vpar_i)
        bc_Te = self._bc_or(self.params.bc_Te)
        bc_Ti = self._bc_or(self.params.bc_Ti)
        bc_phi = self._bc_phi()

        adv_n = -self.geom.bracket(phi, n, bc_phi=bc_phi, bc_f=bc_n)
        adv_w = -self.geom.bracket(phi, omega, bc_phi=bc_phi, bc_f=bc_omega)
        adv_ve = -self.geom.bracket(phi, vpar_e, bc_phi=bc_phi, bc_f=bc_vpar_e)
        adv_vi = -self.geom.bracket(phi, vpar_i, bc_phi=bc_phi, bc_f=bc_vpar_i)
        adv_Te = -self.geom.bracket(phi, Te, bc_phi=bc_phi, bc_f=bc_Te)
        adv_Ti = -self.geom.bracket(phi, Ti, bc_phi=bc_phi, bc_f=bc_Ti)

        grad_par_phi_pe = self.geom.dpar(
            phi - n - float(self.params.alpha_Te_ohm) * Te, bc_kind="dirichlet"
        )
        jpar = vpar_i - vpar_e
        tau_i = float(self.params.tau_i)
        par_n = -self.geom.dpar(vpar_e, bc_kind="dirichlet")
        par_w = self.geom.dpar(jpar, bc_kind="dirichlet")
        par_ve = grad_par_phi_pe / jnp.maximum(float(self.params.me_hat), 1e-12)
        par_vi = -self.geom.dpar(phi + tau_i * (n + Ti), bc_kind="dirichlet")
        par_Te = -(2.0 / 3.0) * self.geom.dpar(vpar_e, bc_kind="dirichlet")
        par_Ti = -(2.0 / 3.0) * self.geom.dpar(vpar_i, bc_kind="dirichlet")

        C_phi = self.geom.curvature(phi)
        C_n = self.geom.curvature(n)
        C_Te = self.geom.curvature(Te)
        C_Ti = self.geom.curvature(Ti)
        C_p = (1.0 + tau_i) * C_n + C_Te + tau_i * C_Ti
        C_Te_term = (2.0 / 3.0) * ((7.0 / 2.0) * C_Te + C_n - C_phi)

        drive_n = 0.0
        drive_Te = 0.0
        drive_Ti = 0.0
        if self.params.omega_n != 0.0 or self.params.omega_Te != 0.0 or self.params.omega_Ti != 0.0:
            dphi_dy = self._ddy(phi, bc_phi)
            drive_n = -float(self.params.omega_n) * dphi_dy
            drive_Te = -float(self.params.omega_Te) * dphi_dy
            drive_Ti = -float(self.params.omega_Ti) * dphi_dy

        lap_n = self._laplacian(n, bc_n)
        bih_n = self._biharmonic(n, bc_n)
        lap_w = self._laplacian(omega, bc_omega)
        bih_w = self._biharmonic(omega, bc_omega)
        lap_Te = self._laplacian(Te, bc_Te)
        bih_Te = self._biharmonic(Te, bc_Te)
        lap_Ti = self._laplacian(Ti, bc_Ti)
        bih_Ti = self._biharmonic(Ti, bc_Ti)

        omega_zonal = jnp.mean(omega, axis=1, keepdims=True) + jnp.zeros_like(omega)

        diss_n = float(self.params.Dn) * lap_n - float(self.params.Dn4) * bih_n
        diss_w = (
            float(self.params.DOmega) * lap_w
            - float(self.params.DOmega4) * bih_w
            - float(self.params.mu_zonal_omega) * omega_zonal
        )
        diss_ve = -(float(self.params.eta) / jnp.maximum(float(self.params.me_hat), 1e-12)) * (
            vpar_e - vpar_i
        )
        diss_vi = jnp.zeros_like(vpar_i)
        diss_Te = float(self.params.DTe) * lap_Te - float(self.params.DTe4) * bih_Te
        diss_Ti = float(self.params.DTi) * lap_Ti - float(self.params.DTi4) * bih_Ti

        if self.params.bc_enforce_nu != 0.0 and self._grid() is not None:
            diss_n = diss_n + enforce_bc_relaxation(
                n, dx=self._grid().dx, dy=self._grid().dy, bc=bc_n, nu=self.params.bc_enforce_nu
            )
            diss_w = diss_w + enforce_bc_relaxation(
                omega,
                dx=self._grid().dx,
                dy=self._grid().dy,
                bc=bc_omega,
                nu=self.params.bc_enforce_nu,
            )
            diss_ve = diss_ve + enforce_bc_relaxation(
                vpar_e,
                dx=self._grid().dx,
                dy=self._grid().dy,
                bc=bc_vpar_e,
                nu=self.params.bc_enforce_nu,
            )
            diss_vi = diss_vi + enforce_bc_relaxation(
                vpar_i,
                dx=self._grid().dx,
                dy=self._grid().dy,
                bc=bc_vpar_i,
                nu=self.params.bc_enforce_nu,
            )
            diss_Te = diss_Te + enforce_bc_relaxation(
                Te, dx=self._grid().dx, dy=self._grid().dy, bc=bc_Te, nu=self.params.bc_enforce_nu
            )
            diss_Ti = diss_Ti + enforce_bc_relaxation(
                Ti, dx=self._grid().dx, dy=self._grid().dy, bc=bc_Ti, nu=self.params.bc_enforce_nu
            )

        E_dot_adv = self.energy_rate(
            y,
            self._state_from(
                y,
                n=adv_n,
                omega=adv_w,
                vpar_e=adv_ve,
                vpar_i=adv_vi,
                Te=adv_Te,
                Ti=adv_Ti,
            ),
        )
        E_dot_parallel = self.energy_rate(
            y,
            self._state_from(
                y,
                n=par_n,
                omega=par_w,
                vpar_e=par_ve,
                vpar_i=par_vi,
                Te=par_Te,
                Ti=par_Ti,
            ),
        )
        E_dot_curvature = self.energy_rate(
            y,
            self._state_from(y, n=C_p - C_phi, omega=C_p, Te=C_Te_term),
        )
        E_dot_drive = self.energy_rate(
            y,
            self._state_from(y, n=drive_n, omega=jnp.zeros_like(omega), Te=drive_Te, Ti=drive_Ti),
        )
        E_dot_diss = self.energy_rate(
            y,
            self._state_from(
                y, n=diss_n, omega=diss_w, vpar_e=diss_ve, vpar_i=diss_vi, Te=diss_Te, Ti=diss_Ti
            ),
        )

        E_dot_total = E_dot_adv + E_dot_parallel + E_dot_curvature + E_dot_drive + E_dot_diss

        return {
            "E_dot_adv": E_dot_adv,
            "E_dot_parallel": E_dot_parallel,
            "E_dot_curvature": E_dot_curvature,
            "E_dot_drive": E_dot_drive,
            "E_dot_diss": E_dot_diss,
            "E_dot_total": E_dot_total,
        }

    def _energy_budget_em(self, y: DRBSystemState) -> dict[str, jnp.ndarray]:
        n = y.n
        omega = y.omega
        vpar_e = y.vpar_e
        vpar_i = y.vpar_i
        Te = y.Te
        psi = y.psi
        phi = self._phi_from_omega(omega, n=n)
        bc_n = self._bc_or(self.params.bc_n)
        bc_omega = self._bc_or(self.params.bc_omega)
        bc_vpar_e = self._bc_or(self.params.bc_vpar_e)
        bc_vpar_i = self._bc_or(self.params.bc_vpar_i)
        bc_Te = self._bc_or(self.params.bc_Te)
        bc_psi = self._bc_or(self.params.bc_psi)
        bc_phi = self._bc_phi()

        adv_n = -self.geom.bracket(phi, n, bc_phi=bc_phi, bc_f=bc_n)
        adv_w = -self.geom.bracket(phi, omega, bc_phi=bc_phi, bc_f=bc_omega)
        adv_ve = -self.geom.bracket(phi, vpar_e, bc_phi=bc_phi, bc_f=bc_vpar_e)
        adv_vi = -self.geom.bracket(phi, vpar_i, bc_phi=bc_phi, bc_f=bc_vpar_i)
        adv_Te = -self.geom.bracket(phi, Te, bc_phi=bc_phi, bc_f=bc_Te)
        adv_psi = -self.geom.bracket(phi, psi, bc_phi=bc_phi, bc_f=bc_psi)

        grad_par_phi_pe = self.geom.dpar(
            phi - n - float(self.params.alpha_Te_ohm) * Te, bc_kind="dirichlet"
        )
        jpar_em = -self._laplacian(psi, bc_psi)
        jpar_total = vpar_i - vpar_e + jpar_em
        par_n = -self.geom.dpar(vpar_e, bc_kind="dirichlet")
        par_w = self.geom.dpar(jpar_total, bc_kind="dirichlet")
        par_ve = grad_par_phi_pe / jnp.maximum(float(self.params.me_hat), 1e-12) - self.geom.dpar(
            psi, bc_kind="dirichlet"
        )
        par_vi = -self.geom.dpar(phi, bc_kind="dirichlet")
        par_Te = -(2.0 / 3.0) * self.geom.dpar(vpar_e, bc_kind="dirichlet")
        par_psi = -grad_par_phi_pe

        C_phi = self.geom.curvature(phi)
        C_n = self.geom.curvature(n)
        C_Te = self.geom.curvature(Te)
        C_p = C_n + C_Te
        C_Te_term = (2.0 / 3.0) * ((7.0 / 2.0) * C_Te + C_n - C_phi)

        drive_n = 0.0
        drive_Te = 0.0
        if self.params.omega_n != 0.0 or self.params.omega_Te != 0.0:
            dphi_dy = self._ddy(phi, bc_phi)
            drive_n = -float(self.params.omega_n) * dphi_dy
            drive_Te = -float(self.params.omega_Te) * dphi_dy

        lap_n = self._laplacian(n, bc_n)
        bih_n = self._biharmonic(n, bc_n)
        lap_w = self._laplacian(omega, bc_omega)
        bih_w = self._biharmonic(omega, bc_omega)
        lap_Te = self._laplacian(Te, bc_Te)
        bih_Te = self._biharmonic(Te, bc_Te)
        lap_psi = self._laplacian(psi, bc_psi)
        bih_psi = self._biharmonic(psi, bc_psi)

        omega_zonal = jnp.mean(omega, axis=1, keepdims=True) + jnp.zeros_like(omega)

        diss_n = float(self.params.Dn) * lap_n - float(self.params.Dn4) * bih_n
        diss_w = (
            float(self.params.DOmega) * lap_w
            - float(self.params.DOmega4) * bih_w
            - float(self.params.mu_zonal_omega) * omega_zonal
        )
        diss_ve = -(float(self.params.eta) / jnp.maximum(float(self.params.me_hat), 1e-12)) * (
            vpar_e - vpar_i
        )
        diss_vi = jnp.zeros_like(vpar_i)
        diss_Te = float(self.params.DTe) * lap_Te - float(self.params.DTe4) * bih_Te
        diss_psi = float(self.params.Dpsi) * lap_psi - float(self.params.Dpsi4) * bih_psi

        if self.params.bc_enforce_nu != 0.0 and self._grid() is not None:
            diss_n = diss_n + enforce_bc_relaxation(
                n, dx=self._grid().dx, dy=self._grid().dy, bc=bc_n, nu=self.params.bc_enforce_nu
            )
            diss_w = diss_w + enforce_bc_relaxation(
                omega,
                dx=self._grid().dx,
                dy=self._grid().dy,
                bc=bc_omega,
                nu=self.params.bc_enforce_nu,
            )
            diss_ve = diss_ve + enforce_bc_relaxation(
                vpar_e,
                dx=self._grid().dx,
                dy=self._grid().dy,
                bc=bc_vpar_e,
                nu=self.params.bc_enforce_nu,
            )
            diss_vi = diss_vi + enforce_bc_relaxation(
                vpar_i,
                dx=self._grid().dx,
                dy=self._grid().dy,
                bc=bc_vpar_i,
                nu=self.params.bc_enforce_nu,
            )
            diss_Te = diss_Te + enforce_bc_relaxation(
                Te, dx=self._grid().dx, dy=self._grid().dy, bc=bc_Te, nu=self.params.bc_enforce_nu
            )
            diss_psi = diss_psi + enforce_bc_relaxation(
                psi, dx=self._grid().dx, dy=self._grid().dy, bc=bc_psi, nu=self.params.bc_enforce_nu
            )

        E_dot_adv = self.energy_rate(
            y,
            self._state_from(
                y,
                n=adv_n,
                omega=adv_w,
                vpar_e=adv_ve,
                vpar_i=adv_vi,
                Te=adv_Te,
                psi=adv_psi,
            ),
        )
        E_dot_parallel = self.energy_rate(
            y,
            self._state_from(
                y,
                n=par_n,
                omega=par_w,
                vpar_e=par_ve,
                vpar_i=par_vi,
                Te=par_Te,
                psi=par_psi,
            ),
        )
        E_dot_curvature = self.energy_rate(
            y,
            self._state_from(y, n=C_p - C_phi, omega=C_p, Te=C_Te_term),
        )
        E_dot_drive = self.energy_rate(
            y,
            self._state_from(y, n=drive_n, omega=jnp.zeros_like(omega), Te=drive_Te),
        )
        E_dot_diss = self.energy_rate(
            y,
            self._state_from(
                y,
                n=diss_n,
                omega=diss_w,
                vpar_e=diss_ve,
                vpar_i=diss_vi,
                Te=diss_Te,
                psi=diss_psi,
            ),
        )

        E_dot_total = E_dot_adv + E_dot_parallel + E_dot_curvature + E_dot_drive + E_dot_diss

        return {
            "E_dot_adv": E_dot_adv,
            "E_dot_parallel": E_dot_parallel,
            "E_dot_curvature": E_dot_curvature,
            "E_dot_drive": E_dot_drive,
            "E_dot_diss": E_dot_diss,
            "E_dot_total": E_dot_total,
        }

    def _phi_from_omega(self, omega: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
        grid = self._grid()
        bc_phi = self._bc_phi()
        if grid is None:
            if self.params.boussinesq:
                return self.geom.inv_laplacian(omega)
            n_eff = float(self.params.n0)
            if self.params.non_boussinesq_perturbed_density_on:
                n_eff = n_eff + jnp.real(jnp.asarray(n))
            n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
            if self.params.n0_max is not None:
                n_eff = jnp.minimum(n_eff, float(self.params.n0_max))
            return self.geom.inv_div_n_grad(n_eff, omega)

        if self.params.boussinesq:
            if self.params.poisson == "spectral":
                if not self._is_periodic_bc(bc_phi):
                    raise ValueError("Spectral Poisson solve requires periodic BCs in x and y.")
                return inv_laplacian(omega, grid.k2, k2_min=self.params.k2_min)
            if self.params.poisson == "mixed_fft":
                return inv_laplacian_mixed_fft(
                    omega,
                    dx=grid.dx,
                    dy=grid.dy,
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
                        dx=grid.dx,
                        dy=grid.dy,
                        bc=bc_phi,
                        gauge_epsilon=self.params.poisson_gauge_epsilon,
                    )
                except ValueError:
                    pass
            return inv_laplacian_cg(
                omega,
                dx=grid.dx,
                dy=grid.dy,
                bc=bc_phi,
                maxiter=int(self.params.poisson_cg_maxiter),
                tol=float(self.params.poisson_cg_tol),
                atol=float(self.params.poisson_cg_atol),
                preconditioner=str(precond),
                k2_precond=grid.k2 if str(precond) == "spectral" else None,
                gauge_epsilon=self.params.poisson_gauge_epsilon,
            )

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
            dx=grid.dx,
            dy=grid.dy,
            bc=bc_phi,
            maxiter=int(self.params.polarization_cg_maxiter),
            tol=float(self.params.polarization_cg_tol),
            atol=float(self.params.polarization_cg_atol),
            preconditioner=precond,
            preconditioner_shift=float(self.params.polarization_precond_shift),
        )

    def _psi_from_current(self, jpar: jnp.ndarray) -> jnp.ndarray:
        rhs = -jpar
        return self.geom.inv_laplacian(rhs)

    def _kappa_profile(self) -> jnp.ndarray | float:
        try:
            return self.geom.kappa_profile()
        except NotImplementedError:
            return float(self.params.kappa)

    def _sheath_nu_base(self) -> float:
        geom = getattr(self.geom, "geom", None)
        grid = getattr(self.geom, "grid", None)
        l = None
        if grid is not None and hasattr(grid, "l"):
            l = jnp.asarray(grid.l)
        elif geom is not None and hasattr(geom, "l"):
            l = jnp.asarray(geom.l)
        elif hasattr(self.geom, "l"):
            l = jnp.asarray(getattr(self.geom, "l"))
        if l is None or l.size < 2:
            Lpar = 1.0
        else:
            Lpar = jnp.abs(l[-1] - l[0])
        factor = float(self.params.sheath_nu_factor)
        if not bool(self.params.sheath_on) and bool(self.params.sheath_bc_on):
            factor = float(self.params.sheath_bc_nu_factor)
        return float(factor) * (2.0 / (float(Lpar) + 1e-30))

    def _sheath_nu(self) -> tuple[float, float, float]:
        nu_base = self._sheath_nu_base()
        nu_m = float(self.params.sheath_nu_mom)
        nu_p = float(self.params.sheath_nu_particle)
        nu_e = float(self.params.sheath_nu_energy)
        if nu_m == 0.0 and (self.params.sheath_on or self.params.sheath_bc_on):
            nu_m = nu_base
        if nu_p == 0.0 and self.params.sheath_bc_on:
            nu_p = nu_base
        if nu_e == 0.0 and self.params.sheath_bc_on:
            nu_e = nu_base
        return nu_m, nu_p, nu_e

    def _sheath_split(self, y: DRBSystemState, phi: jnp.ndarray) -> DRBSystemState:
        if not (self.params.sheath_on or self.params.sheath_bc_on):
            return _state_zeros_like(y)
        model = self.params.sheath_bc_model_fci
        if isinstance(self.params.sheath_bc_model, str):
            model = self.params.sheath_bc_model
        if isinstance(model, int):
            model = "loizu_linear" if int(model) == 1 else "simple"
        if model in {"loizu_linear", "loizu2012", "loizu"}:
            return self._sheath_split_loizu_linear(y, phi)
        return self._sheath_split_simple(y, phi)

    def _sheath_split_simple(self, y: DRBSystemState, phi: jnp.ndarray) -> DRBSystemState:
        mask, sign = self.geom.sheath_mask_sign()
        dve = jnp.zeros_like(y.vpar_e)
        dvi = jnp.zeros_like(y.vpar_i)
        dn = jnp.zeros_like(y.n)
        domega = jnp.zeros_like(y.omega)
        dTe = jnp.zeros_like(y.Te)
        dTi = None if y.Ti is None else jnp.zeros_like(y.Ti)
        dpsi = None if y.psi is None else jnp.zeros_like(y.psi)

        nu_m, nu_p, nu_e = self._sheath_nu()
        if nu_m != 0.0:
            hot_on = bool(self.params.hot_ion_on) and (y.Ti is not None)
            tau_i = float(self.params.tau_i) if hot_on else 0.0
            cs0 = jnp.sqrt(1.0 + tau_i)
            dcs = (
                0.5
                * (y.Te + (y.Ti if hot_on and y.Ti is not None else 0.0))
                / jnp.maximum(cs0, 1e-12)
            )
            vpar_i_target = sign * (1.0 - float(self.params.sheath_delta)) * dcs
            vpar_e_target = sign * (dcs - phi)
            dvi = dvi - nu_m * mask * (y.vpar_i - vpar_i_target)
            dve = dve - nu_m * mask * (y.vpar_e - vpar_e_target)

        if nu_p != 0.0:
            dn = dn - nu_p * mask * y.n
            domega = domega - nu_p * mask * y.omega

        if nu_e != 0.0:
            dTe = dTe - nu_e * self.params.sheath_gamma_e * mask * y.Te
            if dTi is not None:
                dTi = dTi - nu_e * self.params.sheath_gamma_i * mask * y.Ti

        if dpsi is not None and self.params.em_on:
            dj_sh = dvi - dve
            dpsi = self._psi_from_current(dj_sh)

        return DRBSystemState(
            n=dn,
            omega=domega,
            vpar_e=dve,
            vpar_i=dvi,
            Te=dTe,
            Ti=dTi,
            psi=dpsi if y.psi is not None else None,
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    def _sheath_split_loizu_linear(self, y: DRBSystemState, phi: jnp.ndarray) -> DRBSystemState:
        grid = getattr(self.geom, "grid", None)
        geom_line = getattr(self.geom, "geom", None)
        line = None
        if grid is not None and hasattr(grid, "l"):
            line = jnp.asarray(grid.l)
        elif geom_line is not None and hasattr(geom_line, "l"):
            line = jnp.asarray(geom_line.l)
        elif hasattr(self.geom, "l"):
            line = jnp.asarray(getattr(self.geom, "l"))
        if line is None or line.size < 5:
            return self._sheath_split_simple(y, phi)
        nz = int(getattr(grid, "nz", int(line.size)))

        mask, sign = self.geom.sheath_mask_sign()
        nu_m, nu_p, nu_e = self._sheath_nu()

        dn = jnp.zeros_like(y.n)
        domega = jnp.zeros_like(y.omega)
        dve = jnp.zeros_like(y.vpar_e)
        dvi = jnp.zeros_like(y.vpar_i)
        dTe = jnp.zeros_like(y.Te)
        dTi = None if y.Ti is None else jnp.zeros_like(y.Ti)
        dpsi = None if y.psi is None else jnp.zeros_like(y.psi)

        hot_on = bool(self.params.hot_ion_on) and (y.Ti is not None)
        tau_i = float(self.params.tau_i) if hot_on else 0.0
        cs0 = jnp.sqrt(1.0 + tau_i)
        inv_cs0 = 1.0 / jnp.maximum(cs0, 1e-12)
        delta = float(self.params.sheath_delta)
        cos2 = jnp.maximum(float(self.params.sheath_cos2), 1e-8)

        left = 0
        right = nz - 1
        mask_l = jnp.asarray(mask[left], dtype=y.n.dtype)
        mask_r = jnp.asarray(mask[right], dtype=y.n.dtype)
        sign_l = jnp.where(sign[left] != 0.0, sign[left], -1.0)
        sign_r = jnp.where(sign[right] != 0.0, sign[right], +1.0)

        Ti_arr = y.Ti if hot_on and y.Ti is not None else jnp.zeros_like(y.Te)
        dcs_l = 0.5 * inv_cs0 * (y.Te[left] + Ti_arr[left])
        dcs_r = 0.5 * inv_cs0 * (y.Te[right] + Ti_arr[right])

        vi_bc_l = sign_l * (1.0 - delta) * dcs_l
        vi_bc_r = sign_r * (1.0 - delta) * dcs_r

        phi_bc_l = phi[1] + cs0 * (y.vpar_i[1] - vi_bc_l)
        phi_bc_r = phi[-2] + cs0 * (y.vpar_i[-2] - vi_bc_r)

        n_bc_l = y.n[1] + inv_cs0 * (y.vpar_i[1] - vi_bc_l)
        n_bc_r = y.n[-2] + inv_cs0 * (y.vpar_i[-2] - vi_bc_r)

        dcs_bc_l = 0.5 * inv_cs0 * (y.Te[1] + Ti_arr[1])
        dcs_bc_r = 0.5 * inv_cs0 * (y.Te[-2] + Ti_arr[-2])
        ve_bc_l = sign_l * (dcs_bc_l - phi_bc_l)
        ve_bc_r = sign_r * (dcs_bc_r - phi_bc_r)

        phi_target = phi
        phi_target = phi_target.at[left].set(phi_bc_l)
        phi_target = phi_target.at[right].set(phi_bc_r)
        omega_from_phi = self.geom.laplacian(phi_target)
        omega_bc_l = omega_from_phi[left]
        omega_bc_r = omega_from_phi[right]

        dl = jnp.maximum(jnp.asarray(jnp.mean(jnp.diff(line)), dtype=y.n.dtype), 1e-8)
        dl2 = dl * dl
        v2_target_l = -omega_bc_l / (cos2 * cs0)
        v2_target_r = -omega_bc_r / (cos2 * cs0)
        vi_adj_l = (2.0 * vi_bc_l + 4.0 * y.vpar_i[2] - y.vpar_i[3] - dl2 * v2_target_l) / 5.0
        vi_adj_r = (2.0 * vi_bc_r + 4.0 * y.vpar_i[-3] - y.vpar_i[-4] - dl2 * v2_target_r) / 5.0

        if nu_m != 0.0:
            dvi = dvi.at[left].add(-nu_m * mask_l * (y.vpar_i[left] - vi_bc_l))
            dvi = dvi.at[right].add(-nu_m * mask_r * (y.vpar_i[right] - vi_bc_r))
            dvi = dvi.at[1].add(-nu_m * mask_l * (y.vpar_i[1] - vi_adj_l))
            dvi = dvi.at[-2].add(-nu_m * mask_r * (y.vpar_i[-2] - vi_adj_r))

            dve = dve.at[left].add(-nu_m * mask_l * (y.vpar_e[left] - ve_bc_l))
            dve = dve.at[right].add(-nu_m * mask_r * (y.vpar_e[right] - ve_bc_r))

        if nu_p != 0.0:
            dn = dn.at[left].add(-nu_p * mask_l * (y.n[left] - n_bc_l))
            dn = dn.at[right].add(-nu_p * mask_r * (y.n[right] - n_bc_r))
            domega = domega.at[left].add(-nu_p * mask_l * (y.omega[left] - omega_bc_l))
            domega = domega.at[right].add(-nu_p * mask_r * (y.omega[right] - omega_bc_r))

        if nu_e != 0.0:
            Te_bc_l = y.Te[1]
            Te_bc_r = y.Te[-2]
            dTe = dTe.at[left].add(
                -nu_e * mask_l * (y.Te[left] - Te_bc_l)
                - nu_e * self.params.sheath_gamma_e * mask_l * y.Te[left]
            )
            dTe = dTe.at[right].add(
                -nu_e * mask_r * (y.Te[right] - Te_bc_r)
                - nu_e * self.params.sheath_gamma_e * mask_r * y.Te[right]
            )
            if dTi is not None and y.Ti is not None:
                Ti_bc_l = y.Ti[1]
                Ti_bc_r = y.Ti[-2]
                dTi = dTi.at[left].add(
                    -nu_e * mask_l * (y.Ti[left] - Ti_bc_l)
                    - nu_e * self.params.sheath_gamma_i * mask_l * y.Ti[left]
                )
                dTi = dTi.at[right].add(
                    -nu_e * mask_r * (y.Ti[right] - Ti_bc_r)
                    - nu_e * self.params.sheath_gamma_i * mask_r * y.Ti[right]
                )

        if dpsi is not None and self.params.em_on:
            dj_sh = dvi - dve
            dpsi = self._psi_from_current(dj_sh)

        return DRBSystemState(
            n=dn,
            omega=domega,
            vpar_e=dve,
            vpar_i=dvi,
            Te=dTe,
            Ti=dTi,
            psi=dpsi if y.psi is not None else None,
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    def _apply_line_bcs(self, y: DRBSystemState) -> DRBSystemState:
        if not self.params.line_bcs.enabled:
            return _state_zeros_like(y)
        dl = float(getattr(self.geom, "geom", self.geom).dl) if hasattr(self.geom, "geom") else None
        if dl is None:
            return _state_zeros_like(y)
        from jaxdrb.core.bcs import bc_relaxation_1d

        bc = self.params.line_bcs
        return DRBSystemState(
            n=bc_relaxation_1d(y.n, bc=bc.n, dl=dl),
            omega=bc_relaxation_1d(y.omega, bc=bc.omega, dl=dl),
            vpar_e=bc_relaxation_1d(y.vpar_e, bc=bc.vpar_e, dl=dl),
            vpar_i=bc_relaxation_1d(y.vpar_i, bc=bc.vpar_i, dl=dl),
            Te=bc_relaxation_1d(y.Te, bc=bc.Te, dl=dl),
            Ti=None if y.Ti is None else bc_relaxation_1d(y.Ti, bc=bc.Ti, dl=dl),
            psi=None if y.psi is None else bc_relaxation_1d(y.psi, bc=bc.psi, dl=dl),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )
