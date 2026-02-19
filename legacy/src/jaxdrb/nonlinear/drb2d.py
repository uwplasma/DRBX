from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.core import DRBSystem, DRBSystemState, Geometry2DAdapter, coerce_system_params
from jaxdrb.bc import BC2D

from .integrate import DiffraxSolverName, diffeqsolve as diffeqsolve_ode, diffeqsolve_fixed_steps
from .grid import Grid2D
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

    # Compatibility flag; unified core is now the only supported path.
    use_unified_core: bool = True


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


def _to_system_state(y: DRB2DState) -> DRBSystemState:
    return DRBSystemState(
        n=y.n,
        omega=y.omega,
        vpar_e=y.vpar_e,
        vpar_i=y.vpar_i,
        Te=y.Te,
        Ti=None,
        psi=None,
        N=y.N,
    )


def _from_system_state(y: DRBSystemState, *, template: DRB2DState | None = None) -> DRB2DState:
    N = y.N
    if template is not None and template.N is None:
        N = None
    return DRB2DState(n=y.n, omega=y.omega, vpar_e=y.vpar_e, vpar_i=y.vpar_i, Te=y.Te, N=N)


def _from_system_split(split, *, template: DRB2DState | None = None) -> DRB2DDecomposition:
    return DRB2DDecomposition(
        conservative=_from_system_state(split.conservative, template=template),
        source=_from_system_state(split.source, template=template),
        dissipative=_from_system_state(split.dissipative, template=template),
    )

class DRB2DModel(eqx.Module):
    params: DRB2DParams
    grid: Grid2D
    _system: DRBSystem = eqx.field(init=False, repr=False)

    def __post_init__(self):
        if not self.params.use_unified_core:
            raise ValueError("DRB2DModel now requires use_unified_core=True (unified DRBSystem).")
        sys_params = coerce_system_params(self.params)
        geom = Geometry2DAdapter(grid=self.grid, params=sys_params)
        system = DRBSystem(params=sys_params, geom=geom)
        object.__setattr__(self, "_system", system)

    def phi_from_omega(self, omega: jnp.ndarray, n: jnp.ndarray | None = None) -> jnp.ndarray:
        return self._system._phi_from_omega(omega, n=n)

    def rhs_decomposed(self, t: float, y: DRB2DState) -> DRB2DDecomposition:
        split = self._system.rhs_split(t, _to_system_state(y))
        return _from_system_split(split, template=y)

    def rhs(self, t: float, y: DRB2DState) -> DRB2DState:
        out = self._system.rhs(t, _to_system_state(y))
        return _from_system_state(out, template=y)

    def energy(self, y: DRB2DState) -> jnp.ndarray:
        return self._system.energy(_to_system_state(y))

    def energy_rate(self, y: DRB2DState, dy: DRB2DState) -> jnp.ndarray:
        return self._system.energy_rate(_to_system_state(y), _to_system_state(dy))

    def energy_budget(self, y: DRB2DState) -> dict[str, jnp.ndarray]:
        return self._system.energy_budget(_to_system_state(y))

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
