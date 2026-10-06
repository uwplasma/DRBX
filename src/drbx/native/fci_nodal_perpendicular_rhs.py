"""Single-device nodal perpendicular right-hand side and potential solve (chunk C1).

Composes the nodal SBP perpendicular operators on one device for a state ``(E, P, F)`` of the fields ``opts.fields``
(a subset of :data:`FIELD_NAMES`, in that order) with the terms of ``opts.terms``:

* ``"bracket"``: ``rho_star * sbp_bracket(plan, phi, state, SatBoundaryData((wall.value,)), 1.0, bracket_c_kappa)`` on every
  field (``opts.rho_star_convention == "single-length"``; the legacy convention passes ``rho_star`` to ``sbp_bracket`` as its
  divisor);
* ``"curvature"``: ``rho_star * sbp_curvature`` on ``(n, Te, Ti, omega)`` with ``phi`` as the fifth column, scattered into
  the field slots (the other fields get zero; the legacy convention has no ``rho_star`` factor);
* ``"diffusion"``: ``D_f * laplacian_action(...)`` per field, Dirichlet rows reading ``wall.value`` and Neumann rows
  ``wall.normal`` (``opts.neumann_mode``: the physical-normal derivative or the conormal flux).

The potential is either solved (``phi_mode="solve"``) from the polarization equation ``L psi = (Omega - sigma) / rho_star**2``
(single-length; legacy: ``L psi = Omega - sigma``) with Dirichlet ``psi`` data, ``psi = phi + tau p`` (``p = n Ti`` for ``psi="phi_plus_tau_pi"``, ``p = Ti`` for the legacy
``"phi_plus_tau_ti"``), ``phi = psi - tau p``, or prescribed (``phi_mode="prescribed"``).

Everything is jittable with the :class:`NodalPerpendicularContext` (built once per geometry on the host), ``params``,
``state``, ``wall``, ``phi``, ``sigma``, ``source`` as traced arguments and the :class:`NodalPerpendicularOptions` static;
:data:`nodal_perpendicular_rhs_jit` and :data:`solve_potential_jit` are the jitted entry points. Wall data are the caller's
(manufactured values for MMS; production applies :func:`bracket_rule_inflow`). Single device only: the ``eta`` operators
wrap periodically and the layout is family A (one wall).

``rho_star`` is the physical ``rho_s0 / L_ref`` (``NodalPerpendicularOptions.rho_star_convention``). ``"single-length"``
(default) is the one-length normalization: the E x B bracket and the whole curvature term carry ``rho_star``, the
polarization is ``rho_star**2 lap_perp(phi + tau p_i) = Omega`` (``Omega`` is the vorticity field in ``state``), and the
diffusion is unchanged. ``"legacy-bracket-only"`` is the previous form (bracket divided by ``rho_star``, curvature and
polarization free of it), which the frozen campaigns pin at ``rho_star = 0.05``. The scaling acts on the outputs and the
solve right-hand side, never on the operators (``sbp_bracket`` and ``sbp_curvature`` keep their meaning), so the operators'
dissipation and the fixed solver regularization are not rescaled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry.sbp_laplacian import LaplacianPlan
from drbx.native.fci_perpendicular_face_corrections import ABSOLUTE_METHODS
from drbx.native.fci_perpendicular_plane_preconditioner import CoreSchurPreconditioner, PlanePreconditioner
from drbx.native.fci_perpendicular_reconstruction_state import DIRICHLET, NEUMANN
from drbx.native.fci_perpendicular_sbp_bracket import sbp_bracket
from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData
from drbx.native.fci_perpendicular_sbp_curvature import PSI_VARIANTS, curvature_flux, sbp_curvature
from drbx.native.fci_perpendicular_sbp_laplacian import LaplacianBoundaryData, laplacian_action
from drbx.native.fci_perpendicular_sbp_ops import trace
from drbx.native.fci_perpendicular_sbp_laplacian_solve import build_dirichlet_preconditioner, solve_dirichlet
from drbx.stencils.nodal_plan import NodalPlan

__all__ = [
    "FIELD_NAMES", "TERM_NAMES", "CURVATURE_FIELDS", "RHO_STAR_CONVENTIONS", "NodalPerpendicularContext", "NodalPerpendicularOptions",
    "NodalPerpendicularParams", "NodalWallData", "NodalPerpendicularTerms", "build_nodal_perpendicular_context",
    "pressure_variable", "psi_wall_data", "solve_potential", "bracket_rule_inflow", "nodal_perpendicular_rhs",
    "nodal_perpendicular_rhs_jit", "solve_potential_jit",
]

FIELD_NAMES = ("density", "Te", "Ti", "Vi", "Ve", "vorticity")          # Q09 order
TERM_NAMES = ("bracket", "curvature", "diffusion")
CURVATURE_FIELDS = ("density", "Te", "Ti", "vorticity")                  # the columns of the P06 curvature state
RHO_STAR_CONVENTIONS = ("single-length", "legacy-bracket-only")
_NEUMANN_MODES = ("physical", "conormal")
_PHI_MODES = ("solve", "prescribed")


class NodalPerpendicularContext(NamedTuple):
    """Per-geometry pytree (a jit argument): the nodal plan, the Laplacian plan, the preconditioner and ``F = |J| K / B``."""

    plan: NodalPlan
    lplan: LaplacianPlan
    prec: CoreSchurPreconditioner | PlanePreconditioner | None
    curvature_flux: jax.Array


def build_nodal_perpendicular_context(plan: NodalPlan, lplan: LaplacianPlan, *, prec=None, build_preconditioner: bool = True,
                                      **prec_kw) -> NodalPerpendicularContext:
    """Host builder. Checks that both plans share the norm ``Hp`` bitwise (build them from one jacobian) and the grid.

    The preconditioner is ``prec`` if given, otherwise ``build_dirichlet_preconditioner(lplan, **prec_kw)`` when
    ``build_preconditioner`` is true (``prec_kw`` can set ``coeff``/``c_kappa`` of the polarization matrix; the default is the
    unit coefficient and ``c_kappa = 1``, matching ``laplacian_c_kappa = 1``), else ``None`` (no potential solve).
    """
    st, lst = plan.structure, lplan.structure
    if (st.n_eta, st.P) != (lst.n_eta, lst.P):
        raise ValueError(f"nodal plan (E, P) = {(st.n_eta, st.P)} differs from the Laplacian plan {(lst.n_eta, lst.P)}")
    if np.shape(plan.Hp) != np.shape(lplan.Hp) or not np.array_equal(np.asarray(plan.Hp), np.asarray(lplan.Hp)):
        raise ValueError("plan.Hp and lplan.Hp differ: build the NodalPlan and the LaplacianPlan from the same jacobian")
    if prec is not None and prec_kw:
        raise ValueError("pass either a prebuilt prec or preconditioner options, not both")
    if prec is None and build_preconditioner:
        prec = build_dirichlet_preconditioner(lplan, **prec_kw)
    return NodalPerpendicularContext(plan, lplan, prec, jnp.asarray(curvature_flux(plan)))


@dataclass(frozen=True)
class NodalPerpendicularOptions:
    """Static (hashable) options; validated on construction. ``rho_star_convention`` is one of :data:`RHO_STAR_CONVENTIONS`
    (see the module docstring)."""

    fields: tuple[str, ...] = ("density", "Te", "Ti", "vorticity")
    terms: tuple[str, ...] = ("bracket", "curvature", "diffusion")
    diffusion_kinds: tuple[str, ...] = ()
    neumann_mode: str = "physical"
    phi_mode: str = "solve"
    psi: str = "phi_plus_tau_pi"
    bracket_c_kappa: float = 1.0
    curvature_jump_dissipation: bool = False
    curvature_c_kappa: float = 0.0
    absolute_method: str = "closed_form"
    laplacian_c_kappa: float = 1.0
    phi_rtol: float = 1e-10
    phi_maxit: int = 200
    rho_star_convention: str = "single-length"

    def __post_init__(self):
        for name in ("fields", "terms", "diffusion_kinds"):
            value = getattr(self, name)
            if isinstance(value, str):
                raise ValueError(f"{name} must be a sequence of strings, got the string {value!r}")
            object.__setattr__(self, name, tuple(value))
        fields, terms, kinds = self.fields, self.terms, self.diffusion_kinds
        if not fields:
            raise ValueError("fields must not be empty")
        bad = [f for f in fields if f not in FIELD_NAMES]
        if bad:
            raise ValueError(f"unknown fields {bad}; fields must be a subset of {FIELD_NAMES}")
        order = [FIELD_NAMES.index(f) for f in fields]
        if order != sorted(set(order)):
            raise ValueError(f"fields must be distinct and in the order {FIELD_NAMES}, got {fields}")
        bad = [t for t in terms if t not in TERM_NAMES]
        if bad or len(set(terms)) != len(terms):
            raise ValueError(f"terms must be distinct entries of {TERM_NAMES}, got {terms}")
        if "curvature" in terms:
            missing = [f for f in CURVATURE_FIELDS if f not in fields]
            if missing:
                raise ValueError(f"the curvature term needs the fields {CURVATURE_FIELDS}; missing {missing}")
        if "diffusion" in terms:
            if len(kinds) != len(fields):
                raise ValueError(f"diffusion_kinds needs one entry per field ({len(fields)}), got {len(kinds)}")
        if any(k not in (DIRICHLET, NEUMANN) for k in kinds):
            raise ValueError(f"diffusion_kinds entries must be 'dirichlet' or 'neumann', got {kinds}")
        if kinds and len(kinds) != len(fields):
            raise ValueError(f"diffusion_kinds needs one entry per field ({len(fields)}), got {len(kinds)}")
        if self.neumann_mode not in _NEUMANN_MODES:
            raise ValueError(f"neumann_mode must be one of {_NEUMANN_MODES}, got {self.neumann_mode!r}")
        if self.phi_mode not in _PHI_MODES:
            raise ValueError(f"phi_mode must be one of {_PHI_MODES}, got {self.phi_mode!r}")
        if self.psi not in PSI_VARIANTS:
            raise ValueError(f"psi must be one of {PSI_VARIANTS}, got {self.psi!r}")
        if self.absolute_method not in ABSOLUTE_METHODS:
            raise ValueError(f"absolute_method must be one of {ABSOLUTE_METHODS}, got {self.absolute_method!r}")
        if self.rho_star_convention not in RHO_STAR_CONVENTIONS:
            raise ValueError(f"rho_star_convention must be one of {RHO_STAR_CONVENTIONS}, got {self.rho_star_convention!r}")
        if not self.phi_maxit >= 1:
            raise ValueError(f"phi_maxit must be >= 1, got {self.phi_maxit}")

    @property
    def n_fields(self) -> int:
        return len(self.fields)

    @property
    def curvature_index(self) -> tuple[int, ...]:
        """Positions of ``(n, Te, Ti, omega)`` in ``fields``."""
        return tuple(self.fields.index(f) for f in CURVATURE_FIELDS)


class NodalPerpendicularParams(NamedTuple):
    rho_star: jax.Array          # scalar; rho_s0 / L_ref (single-length), the bracket divisor (legacy-bracket-only)
    tau: jax.Array               # scalar
    D: jax.Array                 # (F,) perpendicular diffusion per field in ``opts.fields`` order


class NodalWallData(NamedTuple):
    """Traced data of the (single, family A) wall: ``value (E, N, F)`` inflow (bracket, curvature) and Dirichlet (diffusion)
    data, ``normal (E, N, F)`` the Neumann datum (read for Neumann-kind fields), ``psi (E, N)`` the Dirichlet ``psi`` datum."""

    value: jax.Array
    normal: jax.Array | None = None
    psi: jax.Array | None = None


class NodalPerpendicularTerms(NamedTuple):
    total: jax.Array
    bracket: jax.Array
    curvature: jax.Array
    diffusion: jax.Array         # ``(E, P, F)``, zeros if unselected
    phi: jax.Array               # (E, P)
    psi: jax.Array               # (E, P)
    solve_info: dict


# ---------------------------------------------------------------------------
# Potential
# ---------------------------------------------------------------------------
def pressure_variable(opts: NodalPerpendicularOptions, n, Ti):
    """``n Ti`` for ``psi = "phi_plus_tau_pi"``, ``Ti`` for the legacy ``"phi_plus_tau_ti"``."""
    return n * Ti if opts.psi == "phi_plus_tau_pi" else Ti


def psi_wall_data(opts: NodalPerpendicularOptions, tau, phi_w, n_w, Ti_w):
    """Dirichlet ``psi`` data on the wall: ``phi_w + tau * pressure_variable(n_w, Ti_w)``."""
    return phi_w + tau * pressure_variable(opts, n_w, Ti_w)


def _no_solve_info():
    return {"iterations": jnp.asarray(0), "relative_residual": jnp.asarray(0.0), "converged": jnp.asarray(True)}


def polarization_rhs(opts: NodalPerpendicularOptions, params: NodalPerpendicularParams, omega, sigma=None):
    """Right-hand side ``(Omega - sigma) / rho_star**2`` of the polarization solve (no ``rho_star`` for the legacy convention)."""
    omega = jnp.asarray(omega)
    rhs = omega if sigma is None else omega - jnp.asarray(sigma)
    if opts.rho_star_convention == "single-length":
        rhs = rhs / params.rho_star ** 2
    return rhs


def solve_potential(ctx: NodalPerpendicularContext, opts: NodalPerpendicularOptions, params: NodalPerpendicularParams, omega, n,
                    Ti, psi_wall, *, sigma=None, x0=None):
    """Solve ``L psi = (Omega - sigma) / rho_star**2`` (``omega`` is the vorticity ``Omega``; ``sigma=None`` is zero) with Dirichlet
    ``psi_wall (E, N)`` by preconditioned CG with the context's preconditioner; ``phi = psi - tau * pressure_variable(n, Ti)``.
    ``opts.rho_star_convention == "legacy-bracket-only"`` solves ``L psi = Omega - sigma`` (no ``rho_star``). The right-hand side
    is divided, not the operator, so the fixed regularization of the solve keeps its relative weight. Returns
    ``(psi, phi, info)``."""
    if ctx.prec is None:
        raise ValueError("the context has no preconditioner (build it with build_preconditioner=True)")
    rhs = polarization_rhs(opts, params, omega, sigma)
    psi, info = solve_dirichlet(ctx.lplan, rhs, LaplacianBoundaryData(value=(jnp.asarray(psi_wall),)), ctx.prec,
                                c_kappa=opts.laplacian_c_kappa, x0=x0, rtol=opts.phi_rtol, maxit=opts.phi_maxit)
    phi = psi - params.tau * pressure_variable(opts, n, Ti)
    return psi, phi, info


# ---------------------------------------------------------------------------
# Wall data rule
# ---------------------------------------------------------------------------
def _plan_wall_trace(plan: NodalPlan, f):
    """Wall trace ``(E, N, F)`` of the nodal ``f`` by the bracket's own SAT trace (``a`` of ``wall_inflow``)."""
    blk, side = plan.structure.walls[0][:2]
    return trace(plan, blk, side, f)


def bracket_rule_inflow(ctx: NodalPerpendicularContext, opts: NodalPerpendicularOptions, state, value):
    """Wall inflow data of the bracket (and curvature) rule: Neumann-kind fields take their own wall trace, Dirichlet-kind
    fields keep ``value``. Returns ``(E, N, F)``; needs ``opts.diffusion_kinds``.

    The own trace is the nodal plan's three-point trace that ``sbp_bracket`` and ``sbp_curvature`` penalise, so ``a - data``
    and the inflow SAT vanish exactly for the Neumann fields (the cubic Laplacian trace ``lplan.t_out`` would leave a
    discretisation-level penalty).
    """
    kinds = opts.diffusion_kinds
    if len(kinds) != opts.n_fields:
        raise ValueError("bracket_rule_inflow needs opts.diffusion_kinds (one kind per field)")
    state, value = jnp.asarray(state), jnp.asarray(value)
    dirichlet = np.array([k == DIRICHLET for k in kinds])
    if dirichlet.all():
        return value
    own = _plan_wall_trace(ctx.plan, state)
    return jnp.where(dirichlet, value, own)


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------
def _check_inputs(ctx, opts, params, state, wall):
    check_input_shapes(ctx.plan.structure.n_eta, ctx.plan.structure.P, ctx.lplan.structure.N, opts, params, state, wall)


def check_input_shapes(E, P, N, opts, params, state, wall):
    """Shape and presence checks of ``state (E, P, F)``, the wall data ``(E, N, F)`` and ``params.D`` (shared with the sharded RHS)."""
    F = opts.n_fields
    if state.shape != (E, P, F):
        raise ValueError(f"state must have shape {(E, P, F)}, got {state.shape}")
    if wall.value is None or wall.value.shape != (E, N, F):
        raise ValueError(f"wall.value must have shape {(E, N, F)}, got {None if wall.value is None else wall.value.shape}")
    if wall.normal is not None and wall.normal.shape != (E, N, F):
        raise ValueError(f"wall.normal must have shape {(E, N, F)}, got {wall.normal.shape}")
    if "diffusion" in opts.terms:
        if jnp.shape(params.D) not in ((F,), ()):
            raise ValueError(f"params.D must have shape ({F},), got {jnp.shape(params.D)}")
        if NEUMANN in opts.diffusion_kinds and wall.normal is None:
            raise ValueError("wall.normal is required for Neumann-kind diffusion fields")


def _bracket_term(ctx, opts, params, state, wall, phi):
    bcd = SatBoundaryData((wall.value,))
    if opts.rho_star_convention == "single-length":
        return params.rho_star * sbp_bracket(ctx.plan, phi, state, bcd, 1.0, opts.bracket_c_kappa)
    return sbp_bracket(ctx.plan, phi, state, bcd, params.rho_star, opts.bracket_c_kappa)


def curvature_operands(opts, state, wall, phi):
    """``(q, bcd)`` of the curvature operator: ``q = (n, Te, Ti, omega, phi)`` and its wall inflow data."""
    idx = list(opts.curvature_index)
    q = jnp.concatenate([state[..., idx], phi[..., None]], axis=-1)
    return q, SatBoundaryData((wall.value[..., idx],))


def scatter_curvature(opts, params, state, rhs):
    """Scale the curvature operator output by ``rho_star`` (single-length) and scatter it into the field slots of ``state``."""
    if opts.rho_star_convention == "single-length":
        rhs = params.rho_star * rhs
    return jnp.zeros_like(state).at[..., list(opts.curvature_index)].set(rhs)


def _curvature_term(ctx, opts, params, state, wall, phi):
    q, bcd = curvature_operands(opts, state, wall, phi)
    rhs = sbp_curvature(ctx.plan, q, bcd, tau=params.tau, psi=opts.psi,
                        absolute_method=opts.absolute_method, jump_dissipation=opts.curvature_jump_dissipation,
                        c_kappa=opts.curvature_c_kappa, F=ctx.curvature_flux)
    return scatter_curvature(opts, params, state, rhs)


def diffusion_boundary_data(opts, wall):
    """Laplacian wall data of the diffusion term: Dirichlet ``wall.value``, Neumann ``wall.normal`` (per ``opts.neumann_mode``)."""
    if opts.neumann_mode == "physical":
        return LaplacianBoundaryData(value=(wall.value,), normal_derivative=None if wall.normal is None else (wall.normal,))
    return LaplacianBoundaryData(value=(wall.value,), conormal=None if wall.normal is None else (wall.normal,))


def _diffusion_term(ctx, opts, params, state, wall):
    lap = laplacian_action(ctx.lplan, state, diffusion_boundary_data(opts, wall), opts.diffusion_kinds, None,
                           opts.laplacian_c_kappa, neumann_mode=opts.neumann_mode)
    return jnp.asarray(params.D) * lap


def nodal_perpendicular_rhs(ctx: NodalPerpendicularContext, opts: NodalPerpendicularOptions, params: NodalPerpendicularParams,
                            state, wall: NodalWallData, *, phi=None, psi_x0=None, sigma=None, source=None
                            ) -> NodalPerpendicularTerms:
    """The perpendicular right-hand side of ``state (E, P, F)`` (see the module docstring).

    ``phi_mode="solve"`` needs ``wall.psi`` and the vorticity field ``Omega`` in ``state`` (``psi_x0`` warm-starts the CG,
    ``sigma`` is the extra polarization source of ``L psi = (Omega - sigma) / rho_star**2``); ``phi_mode="prescribed"`` needs ``phi (E, P)``
    and reports ``psi = phi + tau p`` (``psi = phi`` when the pressure fields are not in ``opts.fields``: a bracket or
    diffusion subset that never uses ``psi``).
    """
    state = jnp.asarray(state)
    _check_inputs(ctx, opts, params, state, wall)
    n, Ti = check_potential_arguments(opts, state, wall, phi)
    if opts.phi_mode == "solve":
        psi, phi, info = solve_potential(ctx, opts, params, state[..., opts.fields.index("vorticity")], n, Ti, wall.psi,
                                         sigma=sigma, x0=psi_x0)
    else:
        phi, psi, info = prescribed_potential(opts, params, n, Ti, phi)
    zeros = jnp.zeros_like(state)
    bracket = _bracket_term(ctx, opts, params, state, wall, phi) if "bracket" in opts.terms else zeros
    curvature = _curvature_term(ctx, opts, params, state, wall, phi) if "curvature" in opts.terms else zeros
    diffusion = _diffusion_term(ctx, opts, params, state, wall) if "diffusion" in opts.terms else zeros
    return assemble_terms(bracket, curvature, diffusion, phi, psi, info, source)


def check_potential_arguments(opts, state, wall, phi):
    """Validate ``phi`` / ``wall.psi`` / the required fields against ``opts.phi_mode``; returns ``(n, Ti)`` (``None`` if absent)."""
    fields = opts.fields
    n = state[..., fields.index("density")] if "density" in fields else None
    Ti = state[..., fields.index("Ti")] if "Ti" in fields else None
    if opts.phi_mode == "solve":
        if phi is not None:
            raise ValueError("phi is given but opts.phi_mode == 'solve'; use phi_mode='prescribed' or psi_x0")
        if wall.psi is None:
            raise ValueError("wall.psi is required for phi_mode == 'solve'")
        if n is None or Ti is None or "vorticity" not in fields:
            raise ValueError("the potential solve needs the fields density, Ti and vorticity")
    elif phi is None:
        raise ValueError("phi is required for phi_mode == 'prescribed'")
    return n, Ti


def prescribed_potential(opts, params, n, Ti, phi):
    """``(phi, psi, info)`` of a prescribed ``phi``: ``psi = phi + tau p`` (``psi = phi`` without the pressure fields)."""
    phi = jnp.asarray(phi)
    have_p = Ti is not None and (opts.psi != "phi_plus_tau_pi" or n is not None)
    psi = phi + params.tau * pressure_variable(opts, n, Ti) if have_p else phi          # psi = phi without the pressure fields
    return phi, psi, _no_solve_info()


def assemble_terms(bracket, curvature, diffusion, phi, psi, info, source) -> NodalPerpendicularTerms:
    """Sum the terms (plus the optional ``source``) into a :class:`NodalPerpendicularTerms`."""
    total = bracket + curvature + diffusion
    if source is not None:
        total = total + jnp.asarray(source)
    return NodalPerpendicularTerms(total, bracket, curvature, diffusion, phi, psi, info)


nodal_perpendicular_rhs_jit = jax.jit(nodal_perpendicular_rhs, static_argnames=("opts",))
solve_potential_jit = jax.jit(solve_potential, static_argnames=("opts",))
