from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.core import DRBSystem, DRBSystemState, Geometry2DAdapter, coerce_system_params

from jaxdrb.bc import BC2D
from .integrate import DiffraxSolverName, diffeqsolve as diffeqsolve_ode
from .grid import Grid2D


class DRB2DEMParams(eqx.Module):
    """Electromagnetic extension of the conservative 2D nonlinear DRB testbed."""

    # Background-gradient drives.
    omega_n: float = 0.0
    omega_Te: float = 0.0

    # Parallel coupling modeled via constant k_par (optional).
    kpar: float = 0.0
    eta: float = 0.0
    me_hat: float = 0.2
    beta: float = 0.0
    Dpsi: float = 0.0

    # Curvature drive (simple slab interchange model).
    curvature_on: bool = False
    curvature_coeff: float = 0.0

    # Polarization closure (Boussinesq vs non-Boussinesq).
    boussinesq: bool = True
    n0: float = 1.0
    n0_min: float = 1e-6
    non_boussinesq_perturbed_density_on: bool = False

    # Dissipation.
    Dn: float = 0.0
    DOmega: float = 0.0
    DTe: float = 0.0

    # Hyperdiffusion (biharmonic), implemented as -D4 * ∇⁴.
    Dn4: float = 0.0
    DOmega4: float = 0.0
    DTe4: float = 0.0
    Dpsi4: float = 0.0

    # Optional drag on zonal (ky=0) vorticity: -mu * <omega>_y.
    mu_zonal_omega: float = 0.0

    # Numerical options.
    bracket: Literal["spectral", "arakawa", "centered"] = "arakawa"
    poisson: Literal["spectral", "cg_fd"] = "spectral"
    dealias_on: bool = True
    k2_min: float = 1e-12
    bc_enforce_nu: float = 0.0
    # Optional per-field BC overrides (None -> use Grid2D bc).
    bc_n: BC2D | None = None
    bc_omega: BC2D | None = None
    bc_vpar_i: BC2D | None = None
    bc_Te: BC2D | None = None
    bc_psi: BC2D | None = None
    bc_phi: BC2D | None = None
    # Non-Boussinesq variable-coefficient polarization solve settings.
    polarization_cg_maxiter: int = 400
    polarization_cg_tol: float = 1e-8
    polarization_cg_atol: float = 0.0

    # Thermal-force coefficient in Ohm's law.
    alpha_Te_ohm: float = 1.71

    # Operator split toggles.
    operator_split_on: bool = False
    operator_conservative_on: bool = True
    operator_source_on: bool = True
    operator_dissipative_on: bool = True

    # Compatibility flag; unified core is now the only supported path.
    use_unified_core: bool = True


class DRB2DEMState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
    vpar_e: jnp.ndarray
    psi: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray


class DRB2DEMDecomposition(eqx.Module):
    conservative: DRB2DEMState
    source: DRB2DEMState
    dissipative: DRB2DEMState

    def total(self) -> DRB2DEMState:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: DRB2DEMState, b: DRB2DEMState) -> DRB2DEMState:
    return DRB2DEMState(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        vpar_e=a.vpar_e + b.vpar_e,
        psi=a.psi + b.psi,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
    )


def _state_zeros_like(y: DRB2DEMState) -> DRB2DEMState:
    z = jnp.zeros_like(y.n)
    return DRB2DEMState(n=z, omega=z, vpar_e=z, psi=z, vpar_i=z, Te=z)


def _to_system_state(y: DRB2DEMState) -> DRBSystemState:
    return DRBSystemState(
        n=y.n,
        omega=y.omega,
        vpar_e=y.vpar_e,
        vpar_i=y.vpar_i,
        Te=y.Te,
        Ti=None,
        psi=y.psi,
        N=None,
    )


def _from_system_state(
    y: DRBSystemState, *, template: DRB2DEMState | None = None
) -> DRB2DEMState:
    _ = template
    return DRB2DEMState(
        n=y.n,
        omega=y.omega,
        vpar_e=y.vpar_e,
        psi=y.psi if y.psi is not None else jnp.zeros_like(y.n),
        vpar_i=y.vpar_i,
        Te=y.Te,
    )


def _from_system_split(
    split, *, template: DRB2DEMState | None = None
) -> DRB2DEMDecomposition:
    return DRB2DEMDecomposition(
        conservative=_from_system_state(split.conservative, template=template),
        source=_from_system_state(split.source, template=template),
        dissipative=_from_system_state(split.dissipative, template=template),
    )

class DRB2DEMModel(eqx.Module):
    params: DRB2DEMParams
    grid: Grid2D
    _system: DRBSystem = eqx.field(init=False, repr=False)

    def __post_init__(self):
        if not self.params.use_unified_core:
            raise ValueError("DRB2DEMModel now requires use_unified_core=True (unified DRBSystem).")
        sys_params = coerce_system_params(self.params, em_on=True)
        geom = Geometry2DAdapter(grid=self.grid, params=sys_params)
        system = DRBSystem(params=sys_params, geom=geom)
        object.__setattr__(self, "_system", system)

    def phi_from_omega(self, omega: jnp.ndarray, n: jnp.ndarray | None = None) -> jnp.ndarray:
        return self._system._phi_from_omega(omega, n=n)

    def rhs_decomposed(self, t: float, y: DRB2DEMState) -> DRB2DEMDecomposition:
        split = self._system.rhs_split(t, _to_system_state(y))
        return _from_system_split(split, template=y)

    def rhs(self, t: float, y: DRB2DEMState) -> DRB2DEMState:
        out = self._system.rhs(t, _to_system_state(y))
        return _from_system_state(out, template=y)

    def energy(self, y: DRB2DEMState) -> jnp.ndarray:
        return self._system.energy(_to_system_state(y))

    def energy_rate(self, y: DRB2DEMState, dy: DRB2DEMState) -> jnp.ndarray:
        return self._system.energy_rate(_to_system_state(y), _to_system_state(dy))

    def energy_budget(self, y: DRB2DEMState) -> dict[str, jnp.ndarray]:
        return self._system.energy_budget(_to_system_state(y))

    def diffeqsolve(
        self,
        *,
        y0: DRB2DEMState,
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
