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
from .fd import ddx as ddx_fd
from .fd import ddy as ddy_fd
from .fd import (
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    inv_laplacian_fd_fft,
)
from .grid import Grid2D
from .spectral import (
    dealias,
    ddy as ddy_spec,
    inv_laplacian,
    poisson_bracket_spectral,
    irfft2,
)


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


class DRB2DEMState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
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
        psi=a.psi + b.psi,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
    )


def _state_zeros_like(y: DRB2DEMState) -> DRB2DEMState:
    z = jnp.zeros_like(y.n)
    return DRB2DEMState(n=z, omega=z, psi=z, vpar_i=z, Te=z)


def _to_core_state(y: DRB2DEMState) -> CoreState:
    return CoreState.from_optional(
        n=y.n,
        omega=y.omega,
        vpar_e=jnp.zeros_like(y.n),
        vpar_i=y.vpar_i,
        Te=y.Te,
        psi=y.psi,
    )


def _from_core_state(y: CoreState) -> DRB2DEMState:
    return DRB2DEMState(
        n=y.n,
        omega=y.omega,
        psi=y.psi,
        vpar_i=y.vpar_i,
        Te=y.Te,
    )


class DRB2DEMModel(eqx.Module):
    params: DRB2DEMParams
    grid: Grid2D
    _core: Core2DModel = eqx.field(init=False, repr=False)

    def __post_init__(self):
        core_params = coerce_core2d_params(
            self.params,
            model_kind="drb",
            hot_ion_on=False,
            em_on=True,
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

        if n is None:
            raise ValueError("Non-Boussinesq polarization requires density n.")
        n_eff = float(self.params.n0)
        if self.params.non_boussinesq_perturbed_density_on:
            n_eff = n_eff + jnp.real(jnp.asarray(n))
        n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
        precond = "spectral_jacobi" if self._is_periodic_bc(bc_phi) else "jacobi"
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
        )

    def _bracket(
        self, phi: jnp.ndarray, f: jnp.ndarray, *, bc_phi: BC2D, bc_f: BC2D
    ) -> jnp.ndarray:
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

    def _dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.params.kpar == 0.0:
            return jnp.zeros_like(f)
        return 1j * float(self.params.kpar) * f

    def _curvature(self, f: jnp.ndarray, bc_f: BC2D) -> jnp.ndarray:
        if not self.params.curvature_on or self.params.curvature_coeff == 0.0:
            return jnp.zeros_like(f)
        if self._is_periodic_bc(bc_f) and self.params.poisson == "spectral":
            df_dy = ddy_spec(f, self.grid.ky)
        else:
            df_dy = ddy_fd(f, self.grid.dy, bc_f)
        return -float(self.params.curvature_coeff) * df_dy

    def _psi_rhs_from_terms(self, term_hat: jnp.ndarray) -> jnp.ndarray:
        coef = 0.5 * float(self.params.beta) + float(self.params.me_hat) * jnp.maximum(
            self.grid.k2, self.params.k2_min
        )
        coef = jnp.maximum(coef, 1e-12)
        psi_rhs_hat = term_hat / coef
        # For kpar>0 runs we evolve complex Fourier-mode amplitudes; preserve the complex
        # inverse FFT in that case. For kpar=0 the state is expected to be real-valued.
        return irfft2(psi_rhs_hat, real_output=(self.params.kpar == 0.0))

    def rhs_decomposed(self, t: float, y: DRB2DEMState) -> DRB2DEMDecomposition:
        core_state = _to_core_state(y)
        split = self._core.rhs_decomposed(t, core_state)
        return DRB2DEMDecomposition(
            conservative=_from_core_state(split.conservative),
            source=_from_core_state(split.source),
            dissipative=_from_core_state(split.dissipative),
        )

    def rhs(self, t: float, y: DRB2DEMState) -> DRB2DEMState:
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

    def energy(self, y: DRB2DEMState) -> jnp.ndarray:
        return self._core.energy(_to_core_state(y))

    def energy_rate(self, y: DRB2DEMState, dy: DRB2DEMState) -> jnp.ndarray:
        return self._core.energy_rate(_to_core_state(y), _to_core_state(dy))

    def energy_budget(self, y: DRB2DEMState) -> dict[str, jnp.ndarray]:
        return self._core.energy_budget(_to_core_state(y))

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
