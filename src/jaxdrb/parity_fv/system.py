from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState

from .geometry import ParityFVGeometry
from .params import ParityFVParams
from .terms_density import density_parallel_tendency
from .terms_pressure import pressure_parallel_tendencies
from .terms_vorticity import vorticity_curvature_tendency, vorticity_parallel_tendency


def _zeros_like_state(y: DRBSystemState) -> DRBSystemState:
    return DRBSystemState(
        n=jnp.zeros_like(y.n),
        omega=jnp.zeros_like(y.omega),
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=jnp.zeros_like(y.Te),
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def _state_add(a: DRBSystemState, b: DRBSystemState) -> DRBSystemState:
    def _opt_add(x, y):
        if x is None and y is None:
            return None
        if x is None:
            return y
        if y is None:
            return x
        return x + y

    return DRBSystemState(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        vpar_e=a.vpar_e + b.vpar_e,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
        Ti=_opt_add(a.Ti, b.Ti),
        psi=_opt_add(a.psi, b.psi),
        N=_opt_add(a.N, b.N),
    )


@dataclass(frozen=True)
class ParityFVSplit:
    total_state: DRBSystemState

    def total(self) -> DRBSystemState:
        return self.total_state


class ParityFVScheduler:
    def __init__(self, system: "ParityFVSystem") -> None:
        self._system = system

    def run_with_terms(self, ctx, y: DRBSystemState):
        phi_override = None if ctx is None else getattr(ctx, "phi", None)
        split, term_map, _, _ = self._system.rhs_terms(0.0, y, phi_override=phi_override)
        return split, term_map


class ParityFVSystem:
    """Minimal parity-FV engine with driver/audit compatible interfaces."""

    engine: str = "parity_fv"

    def __init__(
        self,
        *,
        params: ParityFVParams,
        geom: ParityFVGeometry,
        limiter: str = "mc",
        poisson_scale: float = 1.0,
        parallel_on: bool = True,
        curvature_on: bool = True,
        source_n0: float = 0.0,
        parallel_pressure_flux_coeff: float = 5.0 / 3.0,
        parallel_pressure_work_coeff: float = 2.0 / 3.0,
        vorticity_parallel_coeff: float = 1.0,
        curvature_coeff: float = 1.0,
    ) -> None:
        self.params = params
        self.geom = geom
        self.limiter = str(limiter)
        self.poisson_scale = float(poisson_scale)
        self.parallel_on = bool(parallel_on)
        self.curvature_on = bool(curvature_on)
        self.source_n0 = float(source_n0)
        self.parallel_pressure_flux_coeff = float(parallel_pressure_flux_coeff)
        self.parallel_pressure_work_coeff = float(parallel_pressure_work_coeff)
        self.vorticity_parallel_coeff = float(vorticity_parallel_coeff)
        self.curvature_coeff = float(curvature_coeff)
        self.scheduler = ParityFVScheduler(self)

    def _phys_n(self, n: jnp.ndarray) -> jnp.ndarray:
        return n

    def _phys_Te(self, Te: jnp.ndarray) -> jnp.ndarray:
        return Te

    def _phi_from_omega(
        self,
        omega: jnp.ndarray,
        *,
        n: jnp.ndarray | None = None,
        Ti: jnp.ndarray | None = None,
        Te: jnp.ndarray | None = None,
        phi_guess: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        _ = (n, Ti, Te, phi_guess)
        return self.poisson_scale * omega

    def _omega_from_phi(
        self,
        phi: jnp.ndarray,
        *,
        n: jnp.ndarray | None = None,
        Ti: jnp.ndarray | None = None,
        Te: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        _ = (n, Ti, Te)
        scale = self.poisson_scale if abs(self.poisson_scale) > 1e-30 else 1.0
        return phi / scale

    def _parallel_term(self, y: DRBSystemState) -> DRBSystemState:
        if (not self.parallel_on) or int(y.n.shape[0]) <= 1:
            return _zeros_like_state(y)

        dn = density_parallel_tendency(
            y.n,
            y.vpar_e,
            dz=float(self.params.dz),
            limiter=self.limiter,
            n_floor=float(self.params.n_floor),
        )
        pe, dTe = pressure_parallel_tendencies(
            y.n,
            y.Te,
            y.vpar_e,
            dn_parallel=dn,
            dz=float(self.params.dz),
            limiter=self.limiter,
            n_floor=float(self.params.n_floor),
            Te_floor=float(self.params.te_floor),
            flux_coeff=float(self.parallel_pressure_flux_coeff),
            work_coeff=float(self.parallel_pressure_work_coeff),
        )
        _ = pe
        domega = vorticity_parallel_tendency(
            y.vpar_e,
            y.vpar_i,
            dz=float(self.params.dz),
            coeff=float(self.vorticity_parallel_coeff),
        )
        return DRBSystemState(
            n=dn,
            omega=domega,
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=dTe,
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    def _curvature_term(self, y: DRBSystemState) -> DRBSystemState:
        if not self.curvature_on:
            return _zeros_like_state(y)
        n_eff = jnp.maximum(y.n, float(self.params.n_floor))
        Te_eff = jnp.maximum(y.Te, float(self.params.te_floor))
        pe = n_eff * Te_eff
        domega = vorticity_curvature_tendency(
            pe,
            self.geom.bxcv,
            dx=float(self.params.dx),
            coeff=float(self.curvature_coeff),
        )
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

    def _source_term(self, y: DRBSystemState) -> DRBSystemState:
        if self.source_n0 == 0.0:
            return _zeros_like_state(y)
        s = jnp.full_like(y.n, self.source_n0)
        return DRBSystemState(
            n=s,
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    def rhs_terms(
        self,
        t: float,
        y: DRBSystemState,
        *,
        phi_override: jnp.ndarray | None = None,
    ) -> tuple[ParityFVSplit, dict[str, DRBSystemState], jnp.ndarray, jnp.ndarray]:
        _ = t
        term_map: dict[str, DRBSystemState] = {}
        term_map["parallel"] = self._parallel_term(y)
        term_map["curvature"] = self._curvature_term(y)
        term_map["volume_source"] = self._source_term(y)
        total = _zeros_like_state(y)
        for term in term_map.values():
            total = _state_add(total, term)
        phi = self._phi_from_omega(y.omega) if phi_override is None else phi_override
        phi_iters = jnp.asarray(0.0, dtype=y.n.dtype)
        return ParityFVSplit(total), term_map, phi, phi_iters

    def rhs(self, t: float, y: DRBSystemState) -> DRBSystemState:
        split, _, _, _ = self.rhs_terms(t, y)
        return split.total()

    def rhs_explicit(self, t: float, y: DRBSystemState) -> DRBSystemState:
        return self.rhs(t, y)

    def rhs_stiff(self, t: float, y: DRBSystemState) -> DRBSystemState:
        return _zeros_like_state(y)

    def rhs_with_phi(
        self, t: float, y: DRBSystemState, phi_guess: jnp.ndarray | None
    ) -> tuple[DRBSystemState, jnp.ndarray]:
        _ = phi_guess
        split, _, phi, _ = self.rhs_terms(t, y)
        return split.total(), phi

    def rhs_with_phi_iters(
        self, t: float, y: DRBSystemState, phi_guess: jnp.ndarray | None
    ) -> tuple[DRBSystemState, jnp.ndarray, jnp.ndarray]:
        _ = phi_guess
        split, _, phi, phi_iters = self.rhs_terms(t, y)
        return split.total(), phi, phi_iters
