"""Model adapter of the nodal SBP perpendicular stack: state, stage information, accepted defaults and the stage RHS factory.

The runtime operators live in :mod:`drbx.native.fci_nodal_perpendicular_rhs` (``nodal_perpendicular_rhs``, ``solve_potential``);
this module composes them with a source, a potential and wall data into the ``rhs_fn(state, t, carry) -> (rhs_state, carry,
StageInfo)`` contract of :class:`~drbx.native.fci_time_integrator.Rk4Stepper`.

* :class:`NodalState` ``(n, Te, Ti, omega)``, each ``(E, P)``, an :class:`~drbx.native.fci_model.FciModelState`.
* :class:`StageInfo` / :func:`make_stage_info`: traced scalars of one stage (CG iterations, relative residual, convergence,
  minima of the stage state, finiteness).
* :data:`NODAL_PERPENDICULAR_DEFAULTS`: the selector values accepted at the P10 evolved MMS (frozen record).
* :func:`nodal_stage_rhs` (pure) and :func:`make_nodal_rhs` (closure factory): the stage RHS.

Stage RHS: ``rhs = R_h(state; wall(t)) + S(t)`` of the selected terms, with the potential either prescribed (``phi_fn``) or
solved (``phi_mode="solve"``, ``sigma_fn`` the optional polarization source).

======================  ==============================================  =========================================
``options.phi_mode``    potential                                       carry
======================  ==============================================  =========================================
``"prescribed"``        ``phi = phi_fn(t, q)``                          passed through unchanged
``"solve"``             CG solve of ``L psi = (Omega - sigma) / rho*^2``  ``psi`` of the solve (the next warm start)
======================  ==============================================  =========================================

Callbacks (all pure, jittable; ``t`` is the traced stage time, ``q`` the ``(E, P, 4)`` stage state): ``wall_fn(t, q) ->
NodalWallData``; ``source_fn(t, q, wall) -> (E, P, 4)`` (``None``: no source); ``phi_fn(t, q) -> (E, P)`` (prescribed mode only);
``sigma_fn(t, q) -> (E, P)`` (solve mode only; ``None``: ``sigma = 0``). A callback that needs geometry or parameters should
receive them as *jit arguments* of the enclosing program (build the callbacks inside it), so that no geometry constant enters
the HLO: :func:`nodal_stage_rhs` is a pure function of ``(context, params, state, t, carry)`` with ``t`` traced, and one
compilation serves every stage time. :func:`make_nodal_rhs` closes over ``context`` and ``params`` for convenience (tests, small
geometries) and exposes the pure form as ``rhs_fn.unbound``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from drbx.native.fci_model import FciModelState
from drbx.native.fci_nodal_perpendicular_rhs import (CURVATURE_FIELDS, NodalPerpendicularContext, NodalPerpendicularOptions,
                                                     NodalPerpendicularParams, NodalWallData, nodal_perpendicular_rhs)

__all__ = ["NODAL_STATE_FIELDS", "NodalPerpendicularDefaults", "NODAL_PERPENDICULAR_DEFAULTS", "NodalState", "StageInfo",
           "make_stage_info", "zero_carry", "nodal_stage_rhs", "make_nodal_rhs"]

#: the fields of :class:`NodalState`, in the state order ``(n, Te, Ti, omega)`` (:data:`CURVATURE_FIELDS`)
NODAL_STATE_FIELDS = CURVATURE_FIELDS


# ---------------------------------------------------------------------------------------------------------------------
# accepted defaults
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NodalPerpendicularDefaults:
    """The selector values of the nodal SBP perpendicular stack accepted at the P10 evolved MMS (commit 555c1b76,
    ``scripts/p10_evolved_mms/configuration.json`` and ``source.nodal_options``).

    ``layout_family`` is the nodal layout of the geometry (family A: one wall); ``curvature_variant`` is the descriptive name of
    the curvature selector (centred split form, no jump dissipation, no core damping: ``curvature_jump_dissipation = False``,
    ``curvature_c_kappa = 0``); ``bracket_c_kappa = 1`` is the bracket's upwinding / dissipation weight; ``tau`` the ion
    temperature ratio of ``psi = phi + tau p_i`` (a parameter, not an option); ``phi_rtol`` / ``phi_maxit`` the budget of the
    polarization CG; ``preconditioner_*`` the preconditioner of the geometry bundle (``core_schur``, float32 factors).
    Run-dependent entries (``diffusion_kinds`` per boundary pattern, ``phi_mode``, ``rho_star``, ``D``) are not defaults.
    """

    layout_family: str = "A"
    fields: tuple = NODAL_STATE_FIELDS
    terms: tuple = ("bracket", "curvature", "diffusion")
    neumann_mode: str = "physical"
    psi: str = "phi_plus_tau_pi"
    tau: float = 1.0
    bracket_c_kappa: float = 1.0
    curvature_variant: str = "centred"
    curvature_jump_dissipation: bool = False
    curvature_c_kappa: float = 0.0
    absolute_method: str = "closed_form"
    laplacian_c_kappa: float = 1.0
    phi_rtol: float = 1e-11
    phi_maxit: int = 50
    preconditioner_method: str = "core_schur"
    preconditioner_factor_dtype: str = "float32"

    @property
    def preconditioner(self) -> dict:
        """The keyword arguments of the polarization preconditioner (``build_dirichlet_preconditioner``)."""
        return {"method": self.preconditioner_method, "factor_dtype": self.preconditioner_factor_dtype}

    def options(self, *, diffusion_kinds=(), phi_mode: str = "solve", **override) -> NodalPerpendicularOptions:
        """The :class:`NodalPerpendicularOptions` of these defaults for a boundary pattern ``diffusion_kinds`` (one kind per
        field) and a potential mode; ``override`` replaces any option (``terms``, ``fields``, ...)."""
        kw = dict(fields=self.fields, terms=self.terms, diffusion_kinds=diffusion_kinds, neumann_mode=self.neumann_mode,
                  phi_mode=phi_mode, psi=self.psi, bracket_c_kappa=self.bracket_c_kappa,
                  curvature_jump_dissipation=self.curvature_jump_dissipation, curvature_c_kappa=self.curvature_c_kappa,
                  absolute_method=self.absolute_method, laplacian_c_kappa=self.laplacian_c_kappa, phi_rtol=self.phi_rtol,
                  phi_maxit=self.phi_maxit)
        kw.update(override)
        return NodalPerpendicularOptions(**kw)


NODAL_PERPENDICULAR_DEFAULTS = NodalPerpendicularDefaults()


# ---------------------------------------------------------------------------------------------------------------------
# state and stage information
# ---------------------------------------------------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NodalState(FciModelState):
    """``(n, Te, Ti, omega)``, each ``(E, P)`` (the order of :data:`NODAL_STATE_FIELDS`)."""

    n: object
    Te: object
    Ti: object
    omega: object

    @classmethod
    def from_array(cls, array):
        if jnp.ndim(array) != 3 or jnp.shape(array)[-1] != 4:
            raise ValueError(f"expected an (E, P, 4) array, got shape {jnp.shape(array)}")
        return cls(*(array[..., i] for i in range(4)))

    def array(self):
        """``(E, P, 4)``."""
        return jnp.stack(self.field_values(), axis=-1)


class StageInfo(NamedTuple):
    """Traced scalars of one stage evaluation (``cg_*`` are ``0, 0.0, True`` when the stage has no potential solve)."""

    t: jax.Array
    cg_iterations: jax.Array
    cg_relative_residual: jax.Array
    cg_converged: jax.Array
    min_n: jax.Array                  # minima of the stage *state*
    min_Te: jax.Array
    min_Ti: jax.Array
    finite: jax.Array                 # the stage state and the stage RHS are all finite


def make_stage_info(t, q, rhs, solve_info=None) -> StageInfo:
    """:class:`StageInfo` of a stage with state ``q (E, P, 4)`` and RHS ``rhs (E, P, 4)`` (``solve_info``: dict of
    ``iterations``, ``relative_residual``, ``converged``; ``None``: no solve). Also for toy RHS functions in tests."""
    if solve_info is None:
        solve_info = {"iterations": 0, "relative_residual": 0.0, "converged": True}
    return StageInfo(t=jnp.asarray(t, dtype=jnp.float64),
                     cg_iterations=jnp.asarray(solve_info["iterations"], dtype=jnp.int64),
                     cg_relative_residual=jnp.asarray(solve_info["relative_residual"], dtype=jnp.float64),
                     cg_converged=jnp.asarray(solve_info["converged"], dtype=bool),
                     min_n=jnp.min(q[..., 0]), min_Te=jnp.min(q[..., 1]), min_Ti=jnp.min(q[..., 2]),
                     finite=jnp.all(jnp.isfinite(q)) & jnp.all(jnp.isfinite(rhs)))


def zero_carry(context: NodalPerpendicularContext):
    """A zero ``psi`` carry ``(E, P)`` (a cold CG start; any estimate of ``psi`` is a better warm start)."""
    st = context.plan.structure
    return jnp.zeros((st.n_eta, st.P), dtype=jnp.float64)


# ---------------------------------------------------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------------------------------------------------
def _check_callbacks(options: NodalPerpendicularOptions, phi_fn, sigma_fn) -> None:
    if options.fields != NODAL_STATE_FIELDS:
        raise ValueError(f"the nodal stage needs options.fields == {NODAL_STATE_FIELDS}, got {options.fields}")
    if options.phi_mode == "solve":
        if phi_fn is not None:
            raise ValueError("phi_fn is given but options.phi_mode == 'solve' (the potential is solved)")
    else:
        if phi_fn is None:
            raise ValueError("phi_fn is required for phi_mode == 'prescribed'")
        if sigma_fn is not None:
            raise ValueError("sigma_fn is given but options.phi_mode == 'prescribed' (there is no polarization solve)")


def nodal_stage_rhs(context: NodalPerpendicularContext, params: NodalPerpendicularParams, state: NodalState, t, carry, *,
                    options: NodalPerpendicularOptions, wall_fn, source_fn=None, phi_fn=None, sigma_fn=None):
    """``rhs = R_h(state; wall(t)) + S(t)`` and the new carry; pure and jittable with ``context``, ``params``, ``state``, ``t`` and
    ``carry`` traced and ``options`` (hence the callbacks' structure) static. Returns ``(NodalState, carry, StageInfo)``.

    ``carry`` is the ``psi`` warm start of the CG for ``options.phi_mode == "solve"`` (the new carry is the ``psi`` of the
    solve) and is passed through unchanged for ``"prescribed"``. See the module docstring for the callbacks.
    """
    if not isinstance(state, NodalState):
        raise TypeError(f"state must be a NodalState, got {type(state).__name__}")
    _check_callbacks(options, phi_fn, sigma_fn)
    q = state.array()
    wall = wall_fn(t, q)
    source = None if source_fn is None else source_fn(t, q, wall)
    if options.phi_mode == "solve":
        sigma = None if sigma_fn is None else sigma_fn(t, q)
        out = nodal_perpendicular_rhs(context, options, params, q, wall, psi_x0=carry, sigma=sigma, source=source)
        new_carry = out.psi
    else:
        out = nodal_perpendicular_rhs(context, options, params, q, wall, phi=phi_fn(t, q), source=source)
        new_carry = carry
    return NodalState.from_array(out.total), new_carry, make_stage_info(t, q, out.total, out.solve_info)


def make_nodal_rhs(context: NodalPerpendicularContext, params: NodalPerpendicularParams, options: NodalPerpendicularOptions, *,
                   terms, phi_mode: str, wall_fn, source_fn=None, phi_fn=None, sigma_fn=None):
    """The :class:`~drbx.native.fci_time_integrator.Rk4Stepper`-compatible ``rhs_fn(state, t, carry) -> (NodalState, carry,
    StageInfo)`` of the stage of ``terms`` and ``phi_mode`` (``options`` with these two replaced; it must carry the
    ``diffusion_kinds`` if ``"diffusion"`` is selected).

    ``context`` and ``params`` are closed over; ``t`` stays traced, so ``jax.jit(rhs_fn)`` compiles once for every stage time
    (the closed-over geometry is then an HLO constant: for large geometries use ``rhs_fn.unbound``, the same stage as a pure
    function ``f(context, params, state, t, carry)``, and pass ``context`` and ``params`` as jit arguments). ``rhs_fn.options``
    is the resolved :class:`NodalPerpendicularOptions`.
    """
    options = dataclasses.replace(options, terms=tuple(terms), phi_mode=phi_mode)
    _check_callbacks(options, phi_fn, sigma_fn)
    unbound = partial(nodal_stage_rhs, options=options, wall_fn=wall_fn, source_fn=source_fn, phi_fn=phi_fn, sigma_fn=sigma_fn)

    def rhs_fn(state, t, carry):
        return unbound(context, params, state, t, carry)

    rhs_fn.options = options
    rhs_fn.unbound = unbound
    return rhs_fn
