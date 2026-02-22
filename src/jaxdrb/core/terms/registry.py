from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.state import DRBSystemSplit, DRBSystemState, _state_add, _state_zeros_like

from .advection import exb_advection_terms
from .bc_relaxation import field_bc_relaxation
from .context import TermContext
from .curvature import curvature_terms
from .diffusion import diffusion_terms
from .drive import drive_terms
from .extra_dissipation import extra_dissipation_terms
from .fields import log_rhs
from .line_bcs import line_bc_terms
from .neutrals import neutrals_terms
from .parallel import parallel_conservative_terms, parallel_vars
from .perp_bc import perp_bc_relaxation
from .region_bc import region_bc_relaxation
from .sheath import sheath_terms
from .sol import (
    sol_edge_relaxation,
    sol_omega_bc_dirichlet,
    sol_parallel_loss,
    sol_sheath_omega_sink,
    sol_sheath_phi_term,
    sol_sinks,
    sol_sources,
    sol_vpar_bc_dirichlet,
)
from .volume_source import volume_source_terms

Category = Literal["conservative", "source", "dissipative"]
TermFn = Callable[[TermContext, DRBSystemState, dict[str, object]], DRBSystemState]


@dataclass(frozen=True)
class TermSpec:
    name: str
    category: Category
    fn: TermFn


class TermScheduler(eqx.Module):
    terms: tuple[TermSpec, ...] = eqx.field(static=True)

    def run(self, ctx: TermContext, y: DRBSystemState) -> DRBSystemSplit:
        split, _ = self.run_with_terms(ctx, y)
        return split

    def run_with_terms(
        self, ctx: TermContext, y: DRBSystemState
    ) -> tuple[DRBSystemSplit, dict[str, DRBSystemState]]:
        work: dict[str, object] = {}
        conservative = _state_zeros_like(y)
        source = _state_zeros_like(y)
        dissipative = _state_zeros_like(y)
        term_map: dict[str, DRBSystemState] = {}

        for spec in self.terms:
            term = spec.fn(ctx, y, work)
            term_map[spec.name] = term
            if spec.category == "conservative":
                conservative = _state_add(conservative, term)
            elif spec.category == "source":
                source = _state_add(source, term)
            else:
                dissipative = _state_add(dissipative, term)

        return DRBSystemSplit(conservative=conservative, source=source, dissipative=dissipative), term_map


def _get_par(work: dict[str, object], ctx: TermContext, y: DRBSystemState):
    par = work.get("par_vars")
    if par is None:
        par = parallel_vars(ctx, y)
        work["par_vars"] = par
    return par


def _log_term_nTe(ctx: TermContext, term: DRBSystemState) -> DRBSystemState:
    return DRBSystemState(
        n=log_rhs(ctx.params, term.n, ctx.n_phys, ctx.n_floor, ctx.params.log_n),
        omega=term.omega,
        vpar_e=term.vpar_e,
        vpar_i=term.vpar_i,
        Te=log_rhs(ctx.params, term.Te, ctx.Te_phys, ctx.Te_floor, ctx.params.log_Te),
        Ti=None if term.Ti is None else term.Ti,
        psi=None if term.psi is None else term.psi,
        N=None if term.N is None else term.N,
    )


def _term_advection(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    term = exb_advection_terms(ctx, y)
    term = _log_term_nTe(ctx, term)
    return DRBSystemState(
        n=term.n,
        omega=term.omega,
        vpar_e=term.vpar_e,
        vpar_i=term.vpar_i,
        Te=term.Te,
        Ti=term.Ti if y.Ti is not None else None,
        psi=term.psi if y.psi is not None else None,
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def _term_parallel(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    par = _get_par(work, ctx, y)
    term = parallel_conservative_terms(ctx, y, par)
    term = _log_term_nTe(ctx, term)
    return DRBSystemState(
        n=term.n,
        omega=term.omega,
        vpar_e=term.vpar_e,
        vpar_i=term.vpar_i,
        Te=term.Te,
        Ti=term.Ti if y.Ti is not None else None,
        psi=term.psi if y.psi is not None else None,
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def _term_curvature(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    term = curvature_terms(ctx, y)
    term = _log_term_nTe(ctx, term)
    return DRBSystemState(
        n=term.n,
        omega=term.omega,
        vpar_e=term.vpar_e,
        vpar_i=term.vpar_i,
        Te=term.Te,
        Ti=term.Ti if y.Ti is not None else None,
        psi=term.psi if y.psi is not None else None,
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def _term_drive(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    term = drive_terms(ctx, y)
    term = _log_term_nTe(ctx, term)
    return DRBSystemState(
        n=term.n,
        omega=term.omega,
        vpar_e=term.vpar_e,
        vpar_i=term.vpar_i,
        Te=term.Te,
        Ti=term.Ti if y.Ti is not None else None,
        psi=term.psi if y.psi is not None else None,
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def _term_volume_source(
    ctx: TermContext, y: DRBSystemState, work: dict[str, object]
) -> DRBSystemState:
    _ = work
    term = volume_source_terms(ctx, y)
    term = _log_term_nTe(ctx, term)
    return DRBSystemState(
        n=term.n,
        omega=term.omega,
        vpar_e=term.vpar_e,
        vpar_i=term.vpar_i,
        Te=term.Te,
        Ti=term.Ti if y.Ti is not None else None,
        psi=term.psi if y.psi is not None else None,
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def _term_sol_sources(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    term = sol_sources(
        ctx.params,
        ctx.geom,
        n_phys=ctx.n_phys,
        Te_phys=ctx.Te_phys,
        mask_closed=ctx.mask_closed,
        mask_open=ctx.mask_open,
    )
    term = _log_term_nTe(ctx, term)
    return DRBSystemState(
        n=term.n,
        omega=term.omega,
        vpar_e=term.vpar_e,
        vpar_i=term.vpar_i,
        Te=term.Te,
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def _term_neutrals(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    return neutrals_terms(ctx, y)


def _term_diffusion(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    par = _get_par(work, ctx, y)
    term = diffusion_terms(ctx, y, par)
    term = _log_term_nTe(ctx, term)
    return DRBSystemState(
        n=term.n,
        omega=term.omega,
        vpar_e=term.vpar_e,
        vpar_i=term.vpar_i,
        Te=term.Te,
        Ti=term.Ti if y.Ti is not None else None,
        psi=term.psi if y.psi is not None else None,
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def _term_extra_dissipation(
    ctx: TermContext, y: DRBSystemState, work: dict[str, object]
) -> DRBSystemState:
    _ = work
    return extra_dissipation_terms(ctx, y)


def _term_sol_sinks(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    term = sol_sinks(
        ctx.params,
        n_phys=ctx.n_phys,
        Te_phys=ctx.Te_phys,
        omega=y.omega,
        mask_open=ctx.mask_open,
    )
    term = DRBSystemState(
        n=log_rhs(ctx.params, term.n, ctx.n_phys, ctx.n_floor, ctx.params.log_n),
        omega=term.omega,
        vpar_e=term.vpar_e * y.vpar_e,
        vpar_i=term.vpar_i * y.vpar_i,
        Te=log_rhs(ctx.params, term.Te, ctx.Te_phys, ctx.Te_floor, ctx.params.log_Te),
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
    return term


def _term_sol_parallel_loss(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    term = sol_parallel_loss(
        ctx.params,
        y,
        ctx.phi,
        n_phys=ctx.n_phys,
        Te_phys=ctx.Te_phys,
        mask_open=ctx.mask_open,
    )
    term = DRBSystemState(
        n=log_rhs(ctx.params, term.n, ctx.n_phys, ctx.n_floor, ctx.params.log_n),
        omega=term.omega,
        vpar_e=term.vpar_e,
        vpar_i=term.vpar_i,
        Te=log_rhs(ctx.params, term.Te, ctx.Te_phys, ctx.Te_floor, ctx.params.log_Te),
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
    return term


def _term_sol_sheath_phi(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    return sol_sheath_phi_term(
        ctx.params,
        y,
        ctx.phi,
        n_phys=ctx.n_phys,
        Te_phys=ctx.Te_phys,
        mask_open=ctx.mask_open,
    )


def _term_sol_sheath_omega(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    if not ctx.params.sol_sheath_omega_on or ctx.mask_open is None:
        return _state_zeros_like(y)
    sol_omega = sol_sheath_omega_sink(ctx.params, y.omega, ctx.mask_open)
    return DRBSystemState(
        n=jnp.zeros_like(y.n),
        omega=sol_omega,
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=jnp.zeros_like(y.Te),
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def _term_sol_omega_bc(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    return sol_omega_bc_dirichlet(ctx.params, ctx.geom, y, bc_omega=ctx.bcs.omega)


def _term_sol_vpar_bc(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    return sol_vpar_bc_dirichlet(ctx.params, ctx.geom, y)


def _term_sol_edge_relax(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    return sol_edge_relaxation(ctx.params, ctx.geom, y)


def _term_field_bc(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    return field_bc_relaxation(ctx.params, ctx.geom, y, ctx.bcs)


def _term_region_bc(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    return region_bc_relaxation(ctx, y)


def _term_perp_bc(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    return perp_bc_relaxation(ctx.params, ctx.geom, y)


def _term_sheath(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    return sheath_terms(ctx.params, ctx.geom, y, ctx.phi)


def _term_line_bcs(ctx: TermContext, y: DRBSystemState, work: dict[str, object]) -> DRBSystemState:
    return line_bc_terms(ctx.params, ctx.geom, y)


TERM_REGISTRY: dict[str, TermSpec] = {
    "advection": TermSpec("advection", "conservative", _term_advection),
    "parallel": TermSpec("parallel", "conservative", _term_parallel),
    "curvature": TermSpec("curvature", "source", _term_curvature),
    "drive": TermSpec("drive", "source", _term_drive),
    "volume_source": TermSpec("volume_source", "source", _term_volume_source),
    "sol_sources": TermSpec("sol_sources", "source", _term_sol_sources),
    "neutrals": TermSpec("neutrals", "source", _term_neutrals),
    "diffusion": TermSpec("diffusion", "dissipative", _term_diffusion),
    "extra_dissipation": TermSpec("extra_dissipation", "dissipative", _term_extra_dissipation),
    "sol_sinks": TermSpec("sol_sinks", "dissipative", _term_sol_sinks),
    "sol_parallel_loss": TermSpec("sol_parallel_loss", "dissipative", _term_sol_parallel_loss),
    "sol_sheath_phi": TermSpec("sol_sheath_phi", "dissipative", _term_sol_sheath_phi),
    "sol_sheath_omega": TermSpec("sol_sheath_omega", "dissipative", _term_sol_sheath_omega),
    "sol_omega_bc": TermSpec("sol_omega_bc", "dissipative", _term_sol_omega_bc),
    "sol_vpar_bc": TermSpec("sol_vpar_bc", "dissipative", _term_sol_vpar_bc),
    "sol_edge_relax": TermSpec("sol_edge_relax", "dissipative", _term_sol_edge_relax),
    "region_bc_relax": TermSpec("region_bc_relax", "dissipative", _term_region_bc),
    "field_bc_relax": TermSpec("field_bc_relax", "dissipative", _term_field_bc),
    "perp_bc_relax": TermSpec("perp_bc_relax", "dissipative", _term_perp_bc),
    "sheath": TermSpec("sheath", "dissipative", _term_sheath),
    "line_bcs": TermSpec("line_bcs", "dissipative", _term_line_bcs),
}

DEFAULT_TERM_SCHEDULE: tuple[str, ...] = (
    "advection",
    "parallel",
    "curvature",
    "drive",
    "volume_source",
    "sol_sources",
    "neutrals",
    "diffusion",
    "extra_dissipation",
    "sol_sinks",
    "sol_parallel_loss",
    "sol_sheath_phi",
    "sol_sheath_omega",
    "sol_omega_bc",
    "sol_vpar_bc",
    "sol_edge_relax",
    "region_bc_relax",
    "field_bc_relax",
    "perp_bc_relax",
    "sheath",
    "line_bcs",
)

STIFF_TERM_SCHEDULE: tuple[str, ...] = (
    "diffusion",
    "extra_dissipation",
    "sol_sinks",
    "sol_omega_bc",
    "sol_vpar_bc",
    "sol_edge_relax",
    "region_bc_relax",
    "field_bc_relax",
    "perp_bc_relax",
    "line_bcs",
)

PRESET_TERM_SCHEDULES: dict[str, tuple[str, ...]] = {
    # Minimal linear physics for fast benchmarks (no nonlinear advection).
    "benchmark_linear": (
        "parallel",
        "curvature",
        "drive",
        "diffusion",
    ),
    # Minimal nonlinear set for fast benchmarks (adds ExB advection).
    "benchmark_nonlinear": (
        "advection",
        "parallel",
        "curvature",
        "drive",
        "diffusion",
    ),
    # Very small quick-check set (no drive; useful for stability/perf).
    "benchmark_min": (
        "advection",
        "parallel",
        "curvature",
        "diffusion",
    ),
}


def _resolve_term_schedule(params: DRBSystemParams) -> tuple[str, ...]:
    if params.term_schedule is not None:
        return tuple(params.term_schedule)
    preset = getattr(params, "term_schedule_preset", None)
    if preset:
        key = str(preset)
        if key not in PRESET_TERM_SCHEDULES:
            raise ValueError(f"Unknown term_schedule_preset: {key}")
        return PRESET_TERM_SCHEDULES[key]
    return DEFAULT_TERM_SCHEDULE


def build_scheduler(params: DRBSystemParams) -> TermScheduler:
    names = _resolve_term_schedule(params)
    terms = tuple(TERM_REGISTRY[name] for name in names)
    return TermScheduler(terms=terms)


def build_scheduler_from_names(names: Sequence[str]) -> TermScheduler:
    terms = tuple(TERM_REGISTRY[name] for name in names)
    return TermScheduler(terms=terms)


def split_term_schedule(params: DRBSystemParams) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = _resolve_term_schedule(params)
    stiff = tuple(name for name in names if name in STIFF_TERM_SCHEDULE)
    explicit = tuple(name for name in names if name not in STIFF_TERM_SCHEDULE)
    return explicit, stiff


def available_terms() -> tuple[str, ...]:
    return tuple(TERM_REGISTRY.keys())
