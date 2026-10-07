"""Stage RHS of the P10 evolved MMS (chunk C3): the nodal perpendicular RHS plus the manufactured source, as an
:class:`~drbx.native.fci_time_integrator.Rk4Stepper`-compatible ``rhs_fn(state, t, carry) -> (rhs_state, carry, StageInfo)``.

Modes (contract ``p10_c3_c6_contract.md``), state order ``(n, Te, Ti, Omega)`` = :class:`NodalState`:

====================  ===================================  =========================================  =====================
mode                  terms                                phi                                        carry
====================  ===================================  =========================================  =====================
``"diffusion"`` (E1)  ``("diffusion",)``                   prescribed exact ``phi(t)`` (unused)       zeros, passed through
``"hyperbolic"`` (E2) ``("bracket", "curvature")``         prescribed exact ``phi(t)``                zeros, passed through
``"coupled"`` (E3)    ``("bracket", "curvature",           ``phi_mode="solve"`` with ``sigma(t)``     ``psi`` of the last solve
                      "diffusion")``                       (CG warm start = carry)                    (the warm start)
====================  ===================================  =========================================  =====================

``source="continuum"`` (the "N" run) uses ``S = d_t q - sum_{selected terms} R_cont`` and (coupled) the continuum
``sigma = Omega - rho*^2 L_cont psi``; ``source="discrete"`` (the "O" run) uses ``S_h = d_t q - R_h(q)`` and ``sigma_h`` of
:func:`source.discrete` with options restricted to the mode's terms, so the exact nodal state is an exact solution of the
semi-discrete system. Wall data at the stage time: :func:`source.wall_data` (manufactured values, no bracket rule).

The stage evaluates, once per call: the manufactured fields at the nodes (continuum: values + gradients + Hessians + ``d_t``,
one pass of :meth:`MmsFields.all`; discrete: values + ``d_t`` only), the wall data, ``R_h`` of the *exact* state (discrete
source only: one RHS evaluation without solve and ``L_h psi``), and the actual RHS (with the ``psi`` CG solve for the coupled
mode). The source arrays enter the RHS through ``source=`` of :func:`nodal_perpendicular_rhs`, so no field is recomputed.

Compile strategy: :func:`stage_rhs` is a pure function of ``(bundle, p, state, t, carry)`` with static
:class:`StageConfig` (mode, source, pattern, options, fields): the driver passes the (large) bundle and the parameters as jit
*arguments* (no geometry constants in the HLO) and ``t`` traced, so one compilation serves every stage time. :func:`make_stage_rhs`
closes over a bundle and parameters for convenience (tests, small bundles).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native.fci_model import FciModelState                                                    # noqa: E402
from drbx.native.fci_nodal_perpendicular_rhs import (NodalPerpendicularOptions, nodal_perpendicular_rhs,   # noqa: E402
                                                     solve_potential)
from p10_evolved_mms import source as S                                                            # noqa: E402
from p10_evolved_mms.fields import FIELDS, MmsParams                                               # noqa: E402

#: stage mode -> the selected RHS terms
MODES = {"diffusion": ("diffusion",), "hyperbolic": ("bracket", "curvature"),
         "coupled": ("bracket", "curvature", "diffusion")}
SOURCES = ("continuum", "discrete")


# ---------------------------------------------------------------------------------------------------------------------
# state and stage information
# ---------------------------------------------------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NodalState(FciModelState):
    """``(n, Te, Ti, Omega)``, each ``(E, P)`` (pattern of ``scripts/q09_evolved_mms/evolution.py:SixState``)."""

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


# ---------------------------------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StageConfig:
    """Static (hashable) configuration of a stage RHS."""

    mode: str
    source: str
    pattern: str
    opts: NodalPerpendicularOptions
    fields: object = FIELDS

    @property
    def terms(self) -> tuple:
        return MODES[self.mode]

    @property
    def coupled(self) -> bool:
        return self.mode == "coupled"


def stage_config(mode: str, source: str, pattern: str, *, opts_override: dict | None = None, fields=FIELDS) -> StageConfig:
    """Validate and build the :class:`StageConfig`; ``opts_override`` replaces any option of
    :func:`source.nodal_options` except ``terms`` and ``phi_mode`` (fixed by the mode)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}, got {mode!r}")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    override = dict(opts_override or {})
    for key in ("terms", "phi_mode"):
        if key in override:
            raise ValueError(f"{key!r} is fixed by the mode and cannot be overridden")
    opts = S.nodal_options(pattern, phi_mode="solve" if mode == "coupled" else "prescribed", terms=MODES[mode], **override)
    assert opts.psi == "phi_plus_tau_pi", opts.psi
    return StageConfig(mode, source, pattern, opts, fields)


def as_jax_params(p: MmsParams) -> MmsParams:
    """The parameters as float64 jax scalars (``D`` an array): the form passed to jitted programs as an argument."""
    return MmsParams(*(jnp.asarray(x, dtype=jnp.float64) for x in p))


# ---------------------------------------------------------------------------------------------------------------------
# exact data
# ---------------------------------------------------------------------------------------------------------------------
def exact_state(bundle, p: MmsParams, t, fields=FIELDS):
    """The manufactured ``q(t) = (n, Te, Ti, Omega)`` at the nodes ``(E, P, 4)``."""
    return fields.values(bundle.ref.points, t, p)[..., :4]


def initial_carry(bundle, mode: str, p: MmsParams, t0, fields=FIELDS):
    """The stage carry at ``t0``: the exact ``psi(t0) = phi + tau n Ti`` ``(E, P)`` for ``"coupled"``, zeros otherwise."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}, got {mode!r}")
    V = fields.values(bundle.ref.points, t0, p)
    if mode != "coupled":
        return jnp.zeros(V.shape[:-1], dtype=jnp.float64)
    return V[..., 4] + p.tau * S.pressure(S.PSI, V[..., 0], V[..., 2])


# ---------------------------------------------------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------------------------------------------------
def _source_terms(bundle, cfg: StageConfig, params, p, t, wall):
    """``(S, sigma, phi, Continuum | Discrete)`` of the stage: the source of the mode's term set, the polarization source
    (``None`` unless coupled) and the exact prescribed potential ``phi(t)``."""
    if cfg.source == "continuum":
        c = S.continuum(bundle, t, p, cfg.opts.psi, cfg.fields)
        Rsel = sum(getattr(c, term) for term in cfg.terms)
        return c.dq - Rsel, (c.sigma if cfg.coupled else None), c.phi, c
    d = S.discrete(bundle, bundle.ctx, cfg.opts, params, t, p, wall=wall, fields=cfg.fields)
    return d.S, (d.sigma if cfg.coupled else None), d.phi, d


def stage_rhs(bundle, p: MmsParams, state: NodalState, t, carry, *, cfg: StageConfig):
    """``rhs = R_h(state; wall(t)) + S(t)`` for the mode of ``cfg`` and the new carry; pure and jittable with ``bundle``, ``p``,
    ``state``, ``t``, ``carry`` traced. Returns ``(NodalState, carry, StageInfo)``."""
    opts = cfg.opts
    q = state.array()
    params = p.nodal()
    wall = S.wall_data(bundle, t, p, cfg.pattern, opts.psi, cfg.fields)
    src, sigma, phi, _ = _source_terms(bundle, cfg, params, p, t, wall)
    if cfg.coupled:
        out = nodal_perpendicular_rhs(bundle.ctx, opts, params, q, wall, psi_x0=carry, sigma=sigma, source=src)
        new_carry = out.psi
    else:
        out = nodal_perpendicular_rhs(bundle.ctx, opts, params, q, wall, phi=phi, source=src)
        new_carry = carry
    return NodalState.from_array(out.total), new_carry, make_stage_info(t, q, out.total, out.solve_info)


@lru_cache(maxsize=64)
def stage_rhs_unbound(cfg: StageConfig):
    """``f(bundle, p, state, t, carry)`` with the configuration bound (cached: one object per configuration, so the drivers'
    jitted programs are shared)."""
    return partial(stage_rhs, cfg=cfg)


def make_stage_rhs(bundle, mode: str, source: str, p: MmsParams, pattern: str, *, opts_override: dict | None = None,
                   fields=FIELDS):
    """The :class:`Rk4Stepper`-compatible ``rhs_fn(state, t, carry) -> (NodalState, carry, StageInfo)`` of a mode (bundle and
    parameters closed over; ``fields`` is a test hook: any :class:`MmsFields`)."""
    cfg = stage_config(mode, source, pattern, opts_override=opts_override, fields=fields)

    def rhs_fn(state, t, carry):
        return stage_rhs(bundle, p, state, t, carry, cfg=cfg)

    rhs_fn.config = cfg
    return rhs_fn


# ---------------------------------------------------------------------------------------------------------------------
# snapshot diagnostics
# ---------------------------------------------------------------------------------------------------------------------
def snapshot_diagnostics(bundle, p: MmsParams, t, q, *, cfg: StageConfig) -> dict:
    """Diagnostics at the state ``q (E, P, 4)`` at time ``t`` (jittable):

    * ``q_exact``; ``tau = R_h(q_exact) - R_cont(q_exact)`` of the mode's terms (prescribed exact ``phi``, wall data of ``t``);
    * coupled: ``psi, phi`` of a fresh *cold-start* solve at ``q`` with the run's ``sigma`` (continuum or discrete),
      ``psi_exact, phi_exact`` and the elliptic control ``psi_ctrl, phi_ctrl``: the same solve with the exact ``Omega, n, Ti``
      and the continuum ``sigma(t)``; ``psi_iterations, psi_ctrl_iterations`` (CG) and their relative residuals.
    """
    opts = cfg.opts
    params = p.nodal()
    wall = S.wall_data(bundle, t, p, cfg.pattern, opts.psi, cfg.fields)
    c = S.continuum(bundle, t, p, opts.psi, cfg.fields)
    d = S.discrete(bundle, bundle.ctx, opts, params, t, p, wall=wall, fields=cfg.fields)
    out = {"q_exact": d.q, "tau": d.R - sum(getattr(c, term) for term in cfg.terms)}
    if cfg.coupled:
        sigma = c.sigma if cfg.source == "continuum" else d.sigma
        psi, phi, info = solve_potential(bundle.ctx, opts, params, q[..., 3], q[..., 0], q[..., 2], wall.psi, sigma=sigma)
        qe = d.q
        psi_c, phi_c, info_c = solve_potential(bundle.ctx, opts, params, qe[..., 3], qe[..., 0], qe[..., 2], wall.psi,
                                               sigma=c.sigma)
        out.update(psi=psi, phi=phi, psi_exact=d.psi, phi_exact=d.phi, psi_ctrl=psi_c, phi_ctrl=phi_c,
                   psi_iterations=info["iterations"], psi_relative_residual=info["relative_residual"],
                   psi_ctrl_iterations=info_c["iterations"], psi_ctrl_relative_residual=info_c["relative_residual"])
    return out


@lru_cache(maxsize=64)
def snapshot_diagnostics_jit(cfg: StageConfig):
    """The jitted :func:`snapshot_diagnostics` of a configuration: ``f(bundle, p, t, q)``."""
    return jax.jit(partial(snapshot_diagnostics, cfg=cfg))
