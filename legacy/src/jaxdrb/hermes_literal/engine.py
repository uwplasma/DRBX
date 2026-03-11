"""Executable literal Hermes engine.

This engine replaces the unified scheduler with a Hermes-literal runtime path
for the active Stage 1 baseline. Geometry ingestion, normalization, Poisson
inversion, and state initialization are reused from the current JAX runtime,
but RHS assembly is owned here.
"""

from __future__ import annotations

import copy
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.state import DRBSystemSplit, DRBSystemState, _state_add, _state_zeros_like
from jaxdrb.operators.fd2d import (
    biharmonic as biharmonic_fd,
    build_div_n_grad_preconditioner,
    build_fd_fft_eigs,
    build_laplacian_preconditioner,
    ddx as ddx_fd,
    ddy as ddy_fd,
    laplacian as laplacian_fd,
)
from jaxdrb.operators.spectral2d import (
    biharmonic,
    ddx as ddx_spec,
    ddy as ddy_spec,
    laplacian,
)

from .context import build_context
from .fields import (
    _diamagnetic_polarisation_term,
    omega_from_phi as omega_from_phi_fn,
    phi_from_omega as phi_from_omega_fn,
)
from .registry import DEFAULT_TERM_SCHEDULE, STIFF_TERM_SCHEDULE, TERM_REGISTRY


def _explicit_term_schedule(term_schedule: tuple[str, ...]) -> tuple[str, ...]:
    stiff = set(STIFF_TERM_SCHEDULE)
    return tuple(name for name in term_schedule if name not in stiff)


class LiteralTermScheduler(eqx.Module):
    """Minimal scheduler owned by the literal engine."""

    term_names: tuple[str, ...] = eqx.field(static=True)

    def run(self, ctx, y: DRBSystemState) -> DRBSystemSplit:
        split, _ = self.run_with_terms(ctx, y)
        return split

    def run_with_terms(
        self, ctx, y: DRBSystemState
    ) -> tuple[DRBSystemSplit, dict[str, DRBSystemState]]:
        work: dict[str, object] = {}
        conservative = _state_zeros_like(y)
        source = _state_zeros_like(y)
        dissipative = _state_zeros_like(y)
        term_map: dict[str, DRBSystemState] = {}

        for name in self.term_names:
            spec = TERM_REGISTRY[name]
            term = spec.fn(ctx, y, work)
            term_map[name] = term
            if spec.category == "conservative":
                conservative = _state_add(conservative, term)
            elif spec.category == "source":
                source = _state_add(source, term)
            else:
                dissipative = _state_add(dissipative, term)

        return (
            DRBSystemSplit(conservative=conservative, source=source, dissipative=dissipative),
            term_map,
        )


class HermesLiteralSystem(eqx.Module):
    """Literal Hermes baseline runtime system.

    Geometry ingestion and initial-state construction still reuse the existing
    driver path, but the executable runtime no longer holds a live unified
    `DRBSystem`. RHS assembly, Poisson helpers, physical/log conversion, and
    diagnostics are owned directly here.
    """

    params: DRBSystemParams
    geom: GeometryAdapter
    engine: str = eqx.field(static=True, default="hermes_literal")
    scheduler: LiteralTermScheduler = eqx.field(static=True)
    scheduler_explicit: LiteralTermScheduler = eqx.field(static=True)
    scheduler_stiff: LiteralTermScheduler = eqx.field(static=True)
    poisson_precond_cache: dict = eqx.field(static=True, default_factory=dict)
    polarization_precond_cache: dict = eqx.field(static=True, default_factory=dict)
    poisson_fdfft_cache: dict = eqx.field(static=True, default_factory=dict)

    def __init__(
        self,
        *,
        params: DRBSystemParams,
        geom: GeometryAdapter,
        term_schedule: Sequence[str] | None = None,
    ):
        if term_schedule is None:
            params_schedule = getattr(params, "term_schedule", None)
            active_schedule = tuple(params_schedule) if params_schedule else DEFAULT_TERM_SCHEDULE
        else:
            active_schedule = tuple(term_schedule)
        explicit_schedule = _explicit_term_schedule(active_schedule)
        stiff_schedule = tuple(name for name in active_schedule if name in set(STIFF_TERM_SCHEDULE))

        object.__setattr__(self, "params", params)
        object.__setattr__(self, "geom", geom)
        object.__setattr__(self, "scheduler", LiteralTermScheduler(active_schedule))
        object.__setattr__(self, "scheduler_explicit", LiteralTermScheduler(explicit_schedule))
        object.__setattr__(self, "scheduler_stiff", LiteralTermScheduler(stiff_schedule))
        object.__setattr__(self, "poisson_precond_cache", {})
        object.__setattr__(self, "polarization_precond_cache", {})
        object.__setattr__(self, "poisson_fdfft_cache", {})

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

    def _poisson_precond_cached(self, *, bc: BC2D, nx: int, ny: int, dx: float, dy: float):
        precond = self.params.poisson_preconditioner
        if precond == "auto":
            precond = "spectral" if self._is_periodic_bc(bc) else "jacobi"
        if precond in ("none", "", None):
            return None
        if precond == "spectral" and not self._is_periodic_bc(bc):
            precond = "jacobi"
        shape = (nx, ny)
        if bc.kind_x == 1 and bc.kind_y == 1:
            shape = (nx - 2, ny - 2)
        key = (
            shape,
            bc.kind_x,
            bc.kind_y,
            float(dx),
            float(dy),
            str(precond),
            self.params.poisson_gauge_epsilon,
        )
        cached = self.poisson_precond_cache.get(key)
        if cached is not None:
            return cached
        k2_precond = None
        if str(precond) == "spectral" and self._is_periodic_bc(bc):
            grid = self._grid()
            if grid is not None:
                k2_precond = grid.k2
        precond_fn = build_laplacian_preconditioner(
            shape=shape,
            dx=dx,
            dy=dy,
            bc=bc,
            preconditioner=str(precond),
            k2_precond=k2_precond,
            gauge_epsilon=self.params.poisson_gauge_epsilon,
        )
        self.poisson_precond_cache[key] = precond_fn
        return precond_fn

    def _polarization_precond_cached(self, *, bc: BC2D, nx: int, ny: int, dx: float, dy: float):
        if self.params.polarization_preconditioner == "auto":
            precond = "spectral_jacobi" if self._is_periodic_bc(bc) else "jacobi"
        else:
            precond = self.params.polarization_preconditioner
        if precond in ("none", "", None):
            return None
        n_eff = max(float(self.params.n0), float(self.params.n0_min))
        key = (nx, ny, bc.kind_x, bc.kind_y, float(dx), float(dy), str(precond), float(n_eff))
        cached = self.polarization_precond_cache.get(key)
        if cached is not None:
            return cached
        n_coeff = jnp.full((nx, ny), n_eff, dtype=jnp.float64)
        precond_fn = build_div_n_grad_preconditioner(
            n_coeff=n_coeff,
            dx=dx,
            dy=dy,
            bc=bc,
            preconditioner=str(precond),
            preconditioner_shift=float(self.params.polarization_precond_shift),
            gauge_epsilon=self.params.poisson_gauge_epsilon,
            n_floor=float(self.params.n0_min),
        )
        self.polarization_precond_cache[key] = precond_fn
        return precond_fn

    def _poisson_fdfft_eigs_cached(self, *, bc: BC2D, nx: int, ny: int, dx: float, dy: float):
        key = (nx, ny, bc.kind_x, bc.kind_y, float(dx), float(dy))
        cached = self.poisson_fdfft_cache.get(key)
        if cached is not None:
            return cached
        eigs = build_fd_fft_eigs(nx=nx, ny=ny, dx=dx, dy=dy, bc=bc, dtype=jnp.float64)
        self.poisson_fdfft_cache[key] = eigs
        return eigs

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
            Te_phys = Te
            if float(self.params.temperature_floor) > 0.0:
                floor_val = float(self.params.temperature_floor)
                Te_phys = 0.5 * (Te_phys + jnp.sqrt(Te_phys * Te_phys + floor_val * floor_val))
            return Te_phys
        clip = self.params.log_Te_clip
        if clip is None:
            Te_phys = jnp.exp(Te)
            if float(self.params.temperature_floor) > 0.0:
                floor_val = float(self.params.temperature_floor)
                Te_phys = 0.5 * (Te_phys + jnp.sqrt(Te_phys * Te_phys + floor_val * floor_val))
            return Te_phys
        clip_val = float(clip)
        Te_phys = jnp.exp(jnp.clip(Te, a_min=-clip_val, a_max=clip_val))
        if float(self.params.temperature_floor) > 0.0:
            floor_val = float(self.params.temperature_floor)
            Te_phys = 0.5 * (Te_phys + jnp.sqrt(Te_phys * Te_phys + floor_val * floor_val))
        return Te_phys

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

    def _apply_split(self, split: DRBSystemSplit, y: DRBSystemState) -> DRBSystemState:
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

    def _phi_from_omega(
        self,
        omega: jnp.ndarray,
        n: jnp.ndarray,
        Ti: jnp.ndarray | None = None,
        Te: jnp.ndarray | None = None,
        phi_guess: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        bc_phi = self._bc_phi()
        return phi_from_omega_fn(
            self.params,
            self.geom,
            omega,
            n,
            bc_phi,
            Ti=Ti,
            Te=Te,
            phi_guess=phi_guess,
        )

    def _omega_from_phi(
        self,
        phi: jnp.ndarray,
        n: jnp.ndarray,
        Ti: jnp.ndarray | None = None,
        Te: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        bc_phi = self._bc_phi()
        return omega_from_phi_fn(self.params, self.geom, phi, n, bc_phi, Ti=Ti, Te=Te)

    def rhs_split(self, t: float, y: DRBSystemState) -> DRBSystemSplit:
        _ = t
        ctx = build_context(self.params, self.geom, y)
        return self.scheduler.run(ctx, y)

    def rhs(self, t: float, y: DRBSystemState) -> DRBSystemState:
        split = self.rhs_split(t, y)
        return self._apply_split(split, y)

    def rhs_explicit(self, t: float, y: DRBSystemState) -> DRBSystemState:
        _ = t
        ctx = build_context(self.params, self.geom, y)
        split = self.scheduler_explicit.run(ctx, y)
        return self._apply_split(split, y)

    def rhs_stiff(self, t: float, y: DRBSystemState) -> DRBSystemState:
        _ = t
        need_phi = (
            bool(self.params.phi_relax_in_rhs) or float(self.params.phi_par_dissipation) != 0.0
        )
        ctx = build_context(self.params, self.geom, y, skip_phi=not need_phi)
        split = self.scheduler_stiff.run(ctx, y)
        return self._apply_split(split, y)

    def rhs_with_phi(
        self, t: float, y: DRBSystemState, phi_guess: jnp.ndarray | None = None
    ) -> tuple[DRBSystemState, jnp.ndarray]:
        _ = t
        ctx = build_context(self.params, self.geom, y, phi_guess=phi_guess)
        split = self.scheduler.run(ctx, y)
        return self._apply_split(split, y), ctx.phi

    def rhs_with_phi_iters(
        self, t: float, y: DRBSystemState, phi_guess: jnp.ndarray | None = None
    ) -> tuple[DRBSystemState, jnp.ndarray, jnp.ndarray]:
        _ = t
        ctx = build_context(self.params, self.geom, y, phi_guess=phi_guess, return_phi_iters=True)
        split = self.scheduler.run(ctx, y)
        iters = ctx.phi_iters if ctx.phi_iters is not None else jnp.asarray(0)
        return self._apply_split(split, y), ctx.phi, iters

    def rhs_explicit_with_phi(
        self, t: float, y: DRBSystemState, phi_guess: jnp.ndarray | None = None
    ) -> tuple[DRBSystemState, jnp.ndarray]:
        _ = t
        ctx = build_context(self.params, self.geom, y, phi_guess=phi_guess)
        split = self.scheduler_explicit.run(ctx, y)
        return self._apply_split(split, y), ctx.phi

    def rhs_explicit_with_phi_iters(
        self, t: float, y: DRBSystemState, phi_guess: jnp.ndarray | None = None
    ) -> tuple[DRBSystemState, jnp.ndarray, jnp.ndarray]:
        _ = t
        ctx = build_context(self.params, self.geom, y, phi_guess=phi_guess, return_phi_iters=True)
        split = self.scheduler_explicit.run(ctx, y)
        iters = ctx.phi_iters if ctx.phi_iters is not None else jnp.asarray(0)
        return self._apply_split(split, y), ctx.phi, iters

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
        phi = self._phi_from_omega(y.omega, n=n, Ti=y.Ti, Te=Te)
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
        phi = self._phi_from_omega(y.omega, n=y.n, Ti=y.Ti, Te=y.Te)
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
        phi = self._phi_from_omega(y.omega, n=y.n, Ti=y.Ti, Te=y.Te)
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
            phi = self._phi_from_omega(y.omega, n=n, Ti=y.Ti, Te=Te)
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
            phi = self._phi_from_omega(y.omega, n=y.n, Ti=y.Ti, Te=y.Te)
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
            phi = self._phi_from_omega(y.omega, n=y.n, Ti=y.Ti, Te=y.Te)
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
        if getattr(self.params, "diamagnetic_polarisation_on", False):
            omega_pol = _diamagnetic_polarisation_term(
                self.params, self.geom, ctx.n_phys, ctx.Ti, self._bc_phi()
            )
            pol_term = DRBSystemState(
                n=jnp.zeros_like(y.n),
                omega=omega_pol,
                vpar_e=jnp.zeros_like(y.vpar_e),
                vpar_i=jnp.zeros_like(y.vpar_i),
                Te=jnp.zeros_like(y.Te),
                Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
                psi=None if y.psi is None else jnp.zeros_like(y.psi),
                N=None if y.N is None else jnp.zeros_like(y.N),
            )
            out["E_dot_diamagnetic_polarisation"] = self.energy_rate(y, pol_term)
        return out


def build_system(cfg, norm_info):
    """Build the executable literal Hermes system from the active config.

    This reuses the current config/geometry/state initialization path but forces
    the runtime engine to this literal scheduler, and forces Hermes-mirror
    transport schemes for the strict Stage 1 baseline.
    """

    from jaxdrb.driver import BuiltSystem, build_system_from_config

    cfg_base = copy.deepcopy(cfg)
    cfg_base["engine"] = "unified"
    norm_cfg = dict(cfg_base.get("normalization", {}))
    norm_cfg["enabled"] = False
    cfg_base["normalization"] = norm_cfg

    numerics = dict(cfg_base.get("numerics", {}))
    numerics.setdefault("exb_flux_scheme", "hermes_mirror")
    numerics.setdefault("parallel_flux_scheme", "hermes_mirror")
    cfg_base["numerics"] = numerics

    built = build_system_from_config(cfg_base)
    system = HermesLiteralSystem(params=built.system.params, geom=built.system.geom)
    return BuiltSystem(system=system, state=built.state, normalization=norm_info)
