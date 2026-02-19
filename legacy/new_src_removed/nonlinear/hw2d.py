from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.core import DRBSystem, DRBSystemState, Geometry2DAdapter, coerce_system_params

from jaxdrb.bc import BC2D
from jaxdrb.operators.brackets import (
    poisson_bracket_arakawa,
    poisson_bracket_arakawa_fd,
    poisson_bracket_centered,
)

from .integrate import DiffraxSolverName, diffeqsolve as diffeqsolve_ode, diffeqsolve_fixed_steps
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

    # Experimental: route through unified core system (default on).
    # Compatibility flag; unified core is now the only supported path.
    use_unified_core: bool = True


class HW2DState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
    N: jnp.ndarray | None = None


def _to_system_state(y: HW2DState) -> DRBSystemState:
    z = jnp.zeros_like(y.n)
    return DRBSystemState(
        n=y.n,
        omega=y.omega,
        vpar_e=z,
        vpar_i=z,
        Te=z,
        Ti=None,
        psi=None,
        N=y.N,
    )


def _from_system_state(y: DRBSystemState, *, template: HW2DState | None = None) -> HW2DState:
    N = y.N
    if template is not None and template.N is None:
        N = None
    return HW2DState(n=y.n, omega=y.omega, N=N)


def _from_system_split(split, *, template: HW2DState | None = None) -> HW2DDecomposition:
    return HW2DDecomposition(
        conservative=_from_system_state(split.conservative, template=template),
        source=_from_system_state(split.source, template=template),
        dissipative=_from_system_state(split.dissipative, template=template),
    )

class HW2DModel(eqx.Module):
    params: HW2DParams
    grid: Grid2D
    _system: DRBSystem = eqx.field(init=False, repr=False)

    def __post_init__(self):
        if not self.params.use_unified_core:
            raise ValueError("HW2DModel now requires use_unified_core=True (unified DRBSystem).")
        sys_params = coerce_system_params(self.params, model_kind="hw")
        geom = Geometry2DAdapter(grid=self.grid, params=sys_params)
        system = DRBSystem(params=sys_params, geom=geom)
        object.__setattr__(self, "_system", system)

    def phi_from_omega(self, omega: jnp.ndarray, n: jnp.ndarray | None = None) -> jnp.ndarray:
        return self._system._phi_from_omega(omega, n=n)

    def rhs_decomposed(self, t: float, y: HW2DState) -> HW2DDecomposition:
        split = self._system.rhs_split(t, _to_system_state(y))
        return _from_system_split(split, template=y)

    def rhs(self, t: float, y: HW2DState) -> HW2DState:
        out = self._system.rhs(t, _to_system_state(y))
        return _from_system_state(out, template=y)

    def diagnostics(self, y: HW2DState) -> dict[str, jnp.ndarray]:
        diag = {
            "E": self._system.energy(_to_system_state(y)),
            "Z": self._system.enstrophy(_to_system_state(y)),
        }
        if self.params.nbar_on:
            diag["Nbar"] = jnp.mean(y.n, axis=(0, 1))
        return diag

    def energy_budget(self, y: HW2DState) -> dict[str, jnp.ndarray]:
        return self._system.energy_budget(_to_system_state(y))

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

    def diffeqsolve_fixed_steps(
        self,
        *,
        y0: HW2DState,
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
