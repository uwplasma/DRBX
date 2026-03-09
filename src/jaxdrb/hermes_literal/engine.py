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
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemSplit, DRBSystemState, _state_add, _state_zeros_like
from jaxdrb.core.system import DRBSystem
from jaxdrb.core.terms import build_context
from jaxdrb.core.terms.registry import (
    DEFAULT_TERM_SCHEDULE,
    STIFF_TERM_SCHEDULE,
    TERM_REGISTRY,
)


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

    The base DRB system is retained only for shared geometry-dependent helpers
    such as Poisson inversion, physical/log variable conversion, and diagnostics.
    RHS assembly and term ordering are owned by the literal scheduler here.
    """

    params: object
    geom: object
    base_system: DRBSystem
    engine: str = eqx.field(static=True, default="hermes_literal")
    scheduler: LiteralTermScheduler = eqx.field(static=True)
    scheduler_explicit: LiteralTermScheduler = eqx.field(static=True)
    scheduler_stiff: LiteralTermScheduler = eqx.field(static=True)

    def __init__(self, *, base_system: DRBSystem, term_schedule: Sequence[str] | None = None):
        if term_schedule is None:
            params_schedule = getattr(base_system.params, "term_schedule", None)
            active_schedule = tuple(params_schedule) if params_schedule else DEFAULT_TERM_SCHEDULE
        else:
            active_schedule = tuple(term_schedule)
        explicit_schedule = _explicit_term_schedule(active_schedule)
        stiff_schedule = tuple(name for name in active_schedule if name in set(STIFF_TERM_SCHEDULE))

        object.__setattr__(self, "params", base_system.params)
        object.__setattr__(self, "geom", base_system.geom)
        object.__setattr__(self, "base_system", base_system)
        object.__setattr__(self, "scheduler", LiteralTermScheduler(active_schedule))
        object.__setattr__(self, "scheduler_explicit", LiteralTermScheduler(explicit_schedule))
        object.__setattr__(self, "scheduler_stiff", LiteralTermScheduler(stiff_schedule))

    def _apply_split(self, split: DRBSystemSplit, y: DRBSystemState) -> DRBSystemState:
        return self.base_system._apply_split(split, y)

    def _phys_n(self, n: jnp.ndarray) -> jnp.ndarray:
        return self.base_system._phys_n(n)

    def _phys_Te(self, Te: jnp.ndarray) -> jnp.ndarray:
        return self.base_system._phys_Te(Te)

    def _phi_from_omega(
        self,
        omega: jnp.ndarray,
        n: jnp.ndarray,
        Ti: jnp.ndarray | None = None,
        Te: jnp.ndarray | None = None,
        phi_guess: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        return self.base_system._phi_from_omega(omega, n=n, Ti=Ti, Te=Te, phi_guess=phi_guess)

    def _omega_from_phi(
        self,
        phi: jnp.ndarray,
        n: jnp.ndarray,
        Ti: jnp.ndarray | None = None,
        Te: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        return self.base_system._omega_from_phi(phi, n=n, Ti=Ti, Te=Te)

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
        return self.base_system._state_from(y, **kwargs)

    def energy(self, y: DRBSystemState) -> jnp.ndarray:
        return self.base_system.energy(y)

    def enstrophy(self, y: DRBSystemState) -> jnp.ndarray:
        return self.base_system.enstrophy(y)


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
    if not isinstance(built.system, DRBSystem):
        raise TypeError(
            f"Expected unified DRBSystem as literal engine base, got {type(built.system)}."
        )

    system = HermesLiteralSystem(base_system=built.system)
    return BuiltSystem(system=system, state=built.state, normalization=norm_info)
