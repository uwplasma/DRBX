"""Continuum and discrete manufactured sources of the P10 evolved MMS (chunk C2): per stage time ``t``, all in ``jnp``.

Conventions (``NodalPerpendicularOptions.rho_star_convention == "single-length"``): the E x B bracket and the curvature term
carry ``rho_star``, the polarization is ``Omega = rho_star**2 lap_perp(phi + tau p_i)`` (solved as
``L psi = (Omega - sigma) / rho_star**2``, ``psi = phi + tau p_i``, ``p_i = n Ti`` for ``psi="phi_plus_tau_pi"``), the
diffusion ``D_f lap_perp f`` is unchanged. With the manufactured ``q = (n, Te, Ti, Omega)`` and ``phi`` (:mod:`fields`):

* :func:`continuum`: ``R_cont`` per term at the nodes from the exact fields and the extracted metric, the source
  ``S = d_t q - R_cont`` and the polarization source ``sigma = Omega - rho_star**2 L_cont psi``;

  - bracket   ``rho_star * (-h . (grad phi x grad f) / |J|)``;
  - curvature ``rho_star * M_psi(q) C(q)[0:4] + coeff C(psi)`` (:func:`continuum_curvature_rhs`, the ``jnp`` port of
    ``fci_perpendicular_sbp_curvature.continuum_rhs``, ``C = (K / B) . grad``);
  - diffusion ``D_f (divA . grad f + A : grad grad f) / |J|`` (``A = |J| P_perp``, ``divA = d_i A^{ij}``);

* :func:`discrete`: the discrete twin, ``S_h = d_t q - R_h(q, phi, wall)`` (:func:`nodal_perpendicular_rhs`,
  ``phi_mode="prescribed"``) and ``sigma_h = Omega - rho_star**2 L_h(psi; psi_w)``, so that the exact nodal state solves the
  semi-discrete system (the "O" run of the design);
* :func:`wall_data`: :class:`NodalWallData` at stage time ``t``: the manufactured values (bracket / curvature inflow data and the
  Dirichlet diffusion data), the physical-normal derivative ``n . grad f`` (Neumann data, pattern ``"NNN-D"``) and the Dirichlet
  ``psi_w = phi_w + tau p_w``.

Every function takes the :class:`~bundle.Bundle` (geometry) and :class:`~fields.MmsParams` and is traceable (``jit``-able with
``t`` traced; :data:`continuum_jit`, :data:`discrete_jit`, :data:`wall_data_jit`): no host work per call.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import dataclasses
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native.fci_curvature_production_flux import curvature_principal_matrix                    # noqa: E402
from drbx.native.fci_nodal_perpendicular_rhs import (NodalPerpendicularOptions, NodalWallData,       # noqa: E402
                                                     nodal_perpendicular_rhs)
from drbx.native.fci_perpendicular_sbp_curvature import PSI_VARIANTS                                 # noqa: E402
from drbx.native.fci_perpendicular_sbp_laplacian import LaplacianBoundaryData, laplacian_action      # noqa: E402
from p10_evolved_mms.fields import CONFIG, FIELDS, MmsParams                                         # noqa: E402

PSI = CONFIG["psi"]
#: diffusion boundary kinds per field ``(n, Te, Ti, Omega)`` of the BC patterns
PATTERNS = {name: tuple(kinds) for name, kinds in CONFIG["patterns"].items()}


def pattern_kinds(pattern: str) -> tuple:
    if pattern not in PATTERNS:
        raise ValueError(f"unknown BC pattern {pattern!r}; choose from {tuple(PATTERNS)}")
    return PATTERNS[pattern]


def nodal_options(pattern: str, *, phi_mode: str = "prescribed", **override) -> NodalPerpendicularOptions:
    """The RHS options of the harness for a BC pattern (``configuration.json``: centred curvature, ``c_kappa`` of the bracket and
    the Laplacian, physical-normal Neumann data, the solve budget); ``override`` replaces any option."""
    c = CONFIG
    kw = dict(fields=tuple(c["fields"]), terms=("bracket", "curvature", "diffusion"), diffusion_kinds=pattern_kinds(pattern),
              neumann_mode=c["neumann_mode"], phi_mode=phi_mode, psi=c["psi"], bracket_c_kappa=float(c["bracket_c_kappa"]),
              curvature_jump_dissipation=bool(c["curvature"]["jump_dissipation"]),
              curvature_c_kappa=float(c["curvature"]["c_kappa"]), absolute_method=c["curvature"]["absolute_method"],
              laplacian_c_kappa=float(c["laplacian_c_kappa"]), phi_rtol=float(c["solve"]["rtol"]),
              phi_maxit=int(c["solve"]["maxit"]), rho_star_convention=c["rho_star_convention"])
    kw.update(override)
    return NodalPerpendicularOptions(**kw)


# ---------------------------------------------------------------------------------------------------------------------
# pointwise continuum kernels (field axis last; logical frame)
# ---------------------------------------------------------------------------------------------------------------------
def pressure(psi: str, n, Ti):
    """``n Ti`` for ``psi="phi_plus_tau_pi"``, ``Ti`` for ``"phi_plus_tau_ti"``."""
    return n * Ti if psi == "phi_plus_tau_pi" else Ti


def point_bracket(h, jac, ga, gb):
    """``-h . (grad a x grad b) / |J|`` (no ``rho_star``): ``h (..., 3)``, ``jac (...)``, ``ga (..., 3)``, ``gb (..., F, 3)`` ->
    ``(..., F)``. The frozen references write ``-(h x grad a) . grad b / (|J| rho*)`` (the same triple product)."""
    cross = jnp.cross(h[..., None, :], ga[..., None, :])
    return -jnp.sum(cross * gb, axis=-1) / jac[..., None]


def laplacian_cont(A, divA, jac, G, H):
    """``(divA . grad f + A : grad grad f) / |J|``: ``G (..., F, 3)``, ``H (..., F, 3, 3)`` -> ``(..., F)``."""
    num = jnp.einsum("...j,...fj->...f", divA, G) + jnp.einsum("...ij,...fij->...f", A, H)
    return num / jac[..., None]


def continuum_curvature_rhs(values, gradients, K, B, tau=1.0, psi="phi_plus_tau_pi"):
    """``jnp`` port of ``fci_perpendicular_sbp_curvature.continuum_rhs``: the continuum curvature RHS ``(..., 4)`` at nodes from
    ``values (5, ...)``, logical ``gradients (5, ..., 3)``, logical ``K (..., 3)`` and ``B (...)`` (no ``rho_star``)."""
    if psi not in PSI_VARIANTS:
        raise ValueError(f"psi must be one of {PSI_VARIANTS}, got {psi!r}")
    values, gradients, K, B = (jnp.asarray(x, dtype=jnp.float64) for x in (values, gradients, K, B))
    C = jnp.einsum("...d,f...d->f...", K, gradients) / B
    n, te, ti = values[0], values[1], values[2]
    M = curvature_principal_matrix(n, te, ti, B, tau, psi=psi)
    material = jnp.einsum("...ij,j...->...i", M, C[:4])
    if psi == "phi_plus_tau_pi":
        cpsi = C[4] + tau * (ti * C[0] + n * C[2])
    else:
        cpsi = C[4] + tau * C[2]
    coeff = jnp.stack((-2.0 * n, -4.0 * te / 3.0, -4.0 * ti / 3.0, jnp.zeros_like(n)), axis=-1)
    return material + coeff * cpsi[..., None]


def psi_derivatives(psi: str, tau, V, G, H):
    """``psi = phi + tau p`` with its logical gradient and Hessian from the stacks of ``(n, Te, Ti, Omega, phi)``: ``V (..., 5)``,
    ``G (..., 5, 3)``, ``H (..., 5, 3, 3)`` -> ``(psi (...), grad (..., 3), hess (..., 3, 3))``."""
    n, Ti, ph = V[..., 0], V[..., 2], V[..., 4]
    gn, gT, gp = G[..., 0, :], G[..., 2, :], G[..., 4, :]
    if psi == "phi_plus_tau_pi":
        p = n * Ti
        gradp = Ti[..., None] * gn + n[..., None] * gT
        hessp = (Ti[..., None, None] * H[..., 0, :, :] + n[..., None, None] * H[..., 2, :, :]
                 + gn[..., :, None] * gT[..., None, :] + gT[..., :, None] * gn[..., None, :])
    else:
        p, gradp, hessp = Ti, gT, H[..., 2, :, :]
    return ph + tau * p, gp + tau * gradp, H[..., 4, :, :] + tau * hessp


# ---------------------------------------------------------------------------------------------------------------------
# continuum source
# ---------------------------------------------------------------------------------------------------------------------
class Continuum(NamedTuple):
    """All arrays on the nodes: ``q, dq, bracket, curvature, diffusion, R, S (E, P, 4)``; ``phi, psi, sigma, lap_psi (E, P)``."""

    q: jax.Array              # (n, Te, Ti, Omega) exact values
    dq: jax.Array             # d_t q
    phi: jax.Array
    psi: jax.Array
    bracket: jax.Array        # rho_star * point bracket
    curvature: jax.Array      # rho_star * continuum curvature
    diffusion: jax.Array      # D_f L_cont f
    R: jax.Array              # R_cont = bracket + curvature + diffusion
    S: jax.Array              # d_t q - R_cont
    lap_psi: jax.Array        # L_cont psi
    sigma: jax.Array          # Omega - rho_star^2 L_cont psi


def continuum(bundle, t, p: MmsParams, psi: str = PSI, fields=FIELDS) -> Continuum:
    """The continuum RHS terms, source ``S`` and polarization source ``sigma`` at the nodes at time ``t`` (see the module docstring)."""
    ref = bundle.ref
    V, G, H, dV = fields.all(ref.points, t, p)
    q, dq, phi = V[..., :4], dV[..., :4], V[..., 4]
    gq = G[..., :4, :]
    bracket = p.rho_star * point_bracket(ref.h, ref.jac, G[..., 4, :], gq)
    curvature = p.rho_star * continuum_curvature_rhs(jnp.moveaxis(V, -1, 0), jnp.moveaxis(G, -2, 0), ref.K, ref.B, p.tau, psi)
    diffusion = jnp.asarray(p.D) * laplacian_cont(ref.A, ref.divA, ref.jac, gq, H[..., :4, :, :])
    R = bracket + curvature + diffusion
    psi_v, gpsi, hpsi = psi_derivatives(psi, p.tau, V, G, H)
    lap_psi = laplacian_cont(ref.A, ref.divA, ref.jac, gpsi[..., None, :], hpsi[..., None, :, :])[..., 0]
    sigma = q[..., 3] - p.rho_star ** 2 * lap_psi
    return Continuum(q, dq, phi, psi_v, bracket, curvature, diffusion, R, dq - R, lap_psi, sigma)


# ---------------------------------------------------------------------------------------------------------------------
# wall data
# ---------------------------------------------------------------------------------------------------------------------
def _wall_data(bundle, t, p: MmsParams, psi: str, with_normal: bool, fields=FIELDS) -> NodalWallData:
    ref = bundle.ref
    V, G = fields.values_grad(ref.wall_points, t, p)
    value = V[..., :4]
    normal = None
    if with_normal:
        a = ref.wall_ginv_u / jnp.sqrt(ref.wall_ginv_u[..., 0:1])                  # g^{u j} / sqrt(g^{uu}): n . grad f = a . grad f
        normal = jnp.einsum("enj,enfj->enf", a, G[..., :4, :])
    psi_w = V[..., 4] + p.tau * pressure(psi, V[..., 0], V[..., 2])
    return NodalWallData(value, normal, psi_w)


def wall_data(bundle, t, p: MmsParams, pattern: str, psi: str = PSI, fields=FIELDS) -> NodalWallData:
    """Traced wall data at stage time ``t`` for a BC pattern: ``value (E, N, 4)`` the manufactured wall values of
    ``(n, Te, Ti, Omega)`` (inflow data of the bracket and the curvature, Dirichlet diffusion data), ``normal (E, N, 4)`` the
    physical-normal derivative ``n . grad f = g^{u j} d_j f / sqrt(g^{uu})`` (Neumann diffusion data; ``None`` for ``"DDDD"``)
    and ``psi (E, N)`` the Dirichlet ``psi_w = phi_w + tau p_w`` of the polarization solve / ``sigma_h``."""
    kinds = pattern_kinds(pattern)
    return _wall_data(bundle, t, p, psi, any(k == "neumann" for k in kinds), fields)


# ---------------------------------------------------------------------------------------------------------------------
# discrete source
# ---------------------------------------------------------------------------------------------------------------------
class Discrete(NamedTuple):
    """Same layout as :class:`Continuum` with the discrete operators: ``bracket, curvature, diffusion, R`` are ``R_h`` of the
    exact nodal state, ``S = d_t q - R_h``, ``lap_psi = L_h(psi; psi_w)``, ``sigma = Omega - rho_star^2 L_h psi``."""

    q: jax.Array
    dq: jax.Array
    phi: jax.Array
    psi: jax.Array
    bracket: jax.Array
    curvature: jax.Array
    diffusion: jax.Array
    R: jax.Array
    S: jax.Array
    lap_psi: jax.Array
    sigma: jax.Array


def discrete(bundle, ctx, opts: NodalPerpendicularOptions, params, t, p: MmsParams, *, wall: NodalWallData | None = None,
             fields=FIELDS) -> Discrete:
    """The discrete source ``S_h`` and ``sigma_h`` at time ``t`` (``ctx=None``: ``bundle.ctx``; ``params=None``: ``p.nodal()``).

    ``R_h`` is :func:`nodal_perpendicular_rhs` of the exact nodal state with the prescribed exact ``phi`` (``opts.phi_mode`` is
    replaced by ``"prescribed"``) and the wall data of :func:`wall_data` (or ``wall``); ``sigma_h = Omega - rho_star^2
    L_h(psi; psi_w)`` (``rho_star`` factor only for the single-length convention), so ``solve_potential(..., sigma=sigma_h)``
    returns the exact ``psi`` up to the solver tolerance."""
    ctx = bundle.ctx if ctx is None else ctx
    params = p.nodal() if params is None else params
    opts = dataclasses.replace(opts, phi_mode="prescribed")
    ref = bundle.ref
    V, dV = fields.values_dt(ref.points, t, p)
    q, dq, phi = V[..., :4], dV[..., :4], V[..., 4]
    if wall is None:
        wall = _wall_data(bundle, t, p, opts.psi, "neumann" in opts.diffusion_kinds, fields)
    out = nodal_perpendicular_rhs(ctx, opts, params, q, wall, phi=phi)
    psi_v = out.psi
    lap_psi = laplacian_action(ctx.lplan, psi_v, LaplacianBoundaryData(value=(wall.psi,)), "dirichlet", None, opts.laplacian_c_kappa)
    r2 = params.rho_star ** 2 if opts.rho_star_convention == "single-length" else 1.0
    sigma = q[..., 3] - r2 * lap_psi
    return Discrete(q, dq, phi, psi_v, out.bracket, out.curvature, out.diffusion, out.total, dq - out.total, lap_psi, sigma)


continuum_jit = jax.jit(continuum, static_argnames=("psi", "fields"))
wall_data_jit = jax.jit(wall_data, static_argnames=("pattern", "psi", "fields"))
discrete_jit = jax.jit(discrete, static_argnames=("opts", "fields"))
