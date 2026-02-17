from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.core.nonlinear2d import Core2DModel, coerce_core2d_params
from jaxdrb.core.state import CoreState

from jaxdrb.bc import BC2D
from jaxdrb.operators.brackets import (
    poisson_bracket_arakawa,
    poisson_bracket_arakawa_fd,
    poisson_bracket_centered,
)

from .integrate import DiffraxSolverName, diffeqsolve as diffeqsolve_ode
from .grid import Grid2D
from .neutrals import NeutralParams
from .fd import ddx as ddx_fd
from .fd import ddy as ddy_fd
from .fd import inv_laplacian_cg, inv_laplacian_fd_fft
from .spectral import dealias, inv_laplacian, poisson_bracket_spectral


class HW2DParams(eqx.Module):
    """Hasegawa–Wakatani-like 2D nonlinear drift-wave testbed."""

    # Background-gradient drive (proxy for R/Ln).
    kappa: float = 1.0

    # Parallel coupling (adiabaticity / resistive coupling).
    alpha: float = 1.0

    # Dissipation.
    Dn: float = 1e-3
    DOmega: float = 1e-3
    nu4_n: float = 0.0  # hyperdiffusion coefficient for n: adds -nu4_n ∇⁴ n
    nu4_omega: float = 0.0  # hyperdiffusion coefficient for omega: adds -nu4_omega ∇⁴ omega

    # Numerical options.
    bracket: Literal["spectral", "arakawa", "centered"] = "arakawa"
    poisson: Literal["spectral", "cg_fd"] = "spectral"
    dealias_on: bool = True
    k2_min: float = 1e-12
    bc_enforce_nu: float = 0.0  # boundary relaxation rate for non-periodic BCs
    # Optional per-field BC overrides (None -> use Grid2D bc).
    bc_n: BC2D | None = None
    bc_omega: BC2D | None = None
    bc_phi: BC2D | None = None

    # Optional "modified HW" coupling: apply α(φ-n) only to non-zonal components (ky≠0),
    # avoiding unphysical damping of zonal flows.
    alpha_nonzonal_only: bool = False

    # Optional neutral coupling.
    neutrals: NeutralParams = NeutralParams()


class HW2DState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
    N: jnp.ndarray | None = None


def _to_core_state(y: HW2DState) -> CoreState:
    return CoreState.from_optional(
        n=y.n,
        omega=y.omega,
        vpar_e=jnp.zeros_like(y.n),
        vpar_i=jnp.zeros_like(y.n),
        Te=jnp.zeros_like(y.n),
        N=y.N,
    )


def _from_core_state(y: CoreState, *, template: HW2DState | None = None) -> HW2DState:
    N = y.N
    if template is not None and template.N is None:
        N = None
    return HW2DState(n=y.n, omega=y.omega, N=N)


class HW2DModel(eqx.Module):
    params: HW2DParams
    grid: Grid2D
    _core: Core2DModel = eqx.field(init=False, repr=False)

    def __post_init__(self):
        core_params = coerce_core2d_params(
            self.params,
            model_kind="hw",
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

    def phi_from_omega(self, omega: jnp.ndarray) -> jnp.ndarray:
        bc_phi = self._bc_phi()
        if self.params.poisson == "spectral":
            if not self._is_periodic_bc(bc_phi):
                raise ValueError("Spectral Poisson solve requires periodic BCs in x and y.")
            return inv_laplacian(omega, self.grid.k2, k2_min=self.params.k2_min)
        if self.params.poisson == "cg_fd":
            try:
                return inv_laplacian_fd_fft(
                    omega,
                    dx=self.grid.dx,
                    dy=self.grid.dy,
                    bc=bc_phi,
                )
            except ValueError:
                pass
        return inv_laplacian_cg(
            omega,
            dx=self.grid.dx,
            dy=self.grid.dy,
            bc=bc_phi,
            maxiter=300,
            preconditioner="spectral" if self._is_periodic_bc(bc_phi) else "jacobi",
        )

    def _bracket(
        self, phi: jnp.ndarray, f: jnp.ndarray, *, bc_phi: BC2D, bc_f: BC2D
    ) -> jnp.ndarray:
        periodic_pair = self._is_periodic_bc(bc_phi) and self._is_periodic_bc(bc_f)
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
                # Arakawa's Jacobian is designed to conserve quadratic invariants on periodic grids.
                # Applying an FFT filter to it can break these conservation properties, so we return it
                # as-is.
                return poisson_bracket_arakawa(phi, f, self.grid.dx, self.grid.dy)
            return poisson_bracket_arakawa_fd(phi, f, self.grid.dx, self.grid.dy, bc_phi, bc_f)
        if periodic_pair:
            j = poisson_bracket_centered(phi, f, self.grid.dx, self.grid.dy)
        else:
            dphi_dx = ddx_fd(phi, self.grid.dx, bc_phi)
            dphi_dy = ddy_fd(phi, self.grid.dy, bc_phi)
            df_dx = ddx_fd(f, self.grid.dx, bc_f)
            df_dy = ddy_fd(f, self.grid.dy, bc_f)
            j = dphi_dx * df_dy - dphi_dy * df_dx
        if self.params.dealias_on and periodic_pair:
            return dealias(j, self.grid.dealias_mask)
        return j

    def rhs(self, t: float, y: HW2DState) -> HW2DState:
        core_state = _to_core_state(y)
        core_rhs = self._core.rhs(t, core_state)
        return _from_core_state(core_rhs, template=y)

    def diffeqsolve(
        self,
        *,
        y0: HW2DState,
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

    def diagnostics(self, y: HW2DState) -> dict[str, jnp.ndarray]:
        return self._core.diagnostics(_to_core_state(y))

    def energy_budget(self, y: HW2DState) -> dict[str, jnp.ndarray]:
        return self._core.energy_budget(_to_core_state(y))
