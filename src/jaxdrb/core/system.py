from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.state import DRBSystemSplit, DRBSystemState, _state_add, _state_zeros_like
from jaxdrb.core.terms import build_context
from jaxdrb.operators.fd2d import (
    ddx as ddx_fd,
    ddy as ddy_fd,
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
        grid = getattr(self.geom, "grid", None)
        if grid is None:
            return None
        if getattr(self.geom, "ndim", None) == 2:
            return grid
        return None

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

    def _rhs_drb_split(self, t: float, y: DRBSystemState) -> DRBSystemSplit:
        _ = t
        ctx = build_context(self.params, self.geom, y)
        return self.scheduler.run(ctx, y)

    def rhs_split(self, t: float, y: DRBSystemState) -> DRBSystemSplit:
        return self._rhs_drb_split(t, y)

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
            Ti=None
            if y.Ti is None
            else kwargs.get("Ti", jnp.zeros_like(y.Ti)),
            psi=None
            if y.psi is None
            else kwargs.get("psi", jnp.zeros_like(y.psi)),
            N=None
            if y.N is None
            else kwargs.get("N", jnp.zeros_like(y.N)),
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
            jnp.real(jnp.conj(gradphi_x) * gradphi_x)
            + jnp.real(jnp.conj(gradphi_y) * gradphi_y)
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
            jnp.real(jnp.conj(gradphi_x) * gradphi_x)
            + jnp.real(jnp.conj(gradphi_y) * gradphi_y)
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
        ctx = build_context(self.params, self.geom, y)
        split, term_map = self.scheduler.run_with_terms(ctx, y)
        total = self.energy_rate(y, split.total())
        conservative = self.energy_rate(y, split.conservative)
        source = self.energy_rate(y, split.source)
        dissipative = self.energy_rate(y, split.dissipative)
        residual = total - (conservative + source + dissipative)
        out = {
            "total": total,
            "conservative": conservative,
            "source": source,
            "dissipative": dissipative,
            "residual": residual,
        }
        for name, term in term_map.items():
            out[f"E_dot_{name}"] = self.energy_rate(y, term)
        return out

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
            poisson = self.params.poisson
            if self.params.poisson_force_spectral_when_periodic and self._is_periodic_bc(bc_phi):
                poisson = "spectral"
            if poisson == "spectral":
                if not self._is_periodic_bc(bc_phi):
                    raise ValueError("Spectral Poisson solve requires periodic BCs in x and y.")
                return inv_laplacian(omega, grid.k2, k2_min=self.params.k2_min)
            if poisson == "mixed_fft":
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
            if poisson == "cg_fd":
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
