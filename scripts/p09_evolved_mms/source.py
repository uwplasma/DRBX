"""Pure-JAX per-time evaluator of the P09 evolved MMS: owner-level sources and boundary data from ``geometry.Geometry``.

``Source(om).evaluate(geo, t, params)`` (jitted; ``t`` traced, geometry arrays are device inputs, no host work per call)
returns the owner arrays (``N`` owners; fields in the order ``density, Te, Ti, vorticity`` unless noted):

* ``qbar (N, 5)`` / ``dqbar_dt (N, 5)``: raw-volume owner averages of the five columns ``n, Te, Ti, omega, phi`` and of
  their ``d/dt`` at the raw midpoints (``p08_step6_global.fields.owner_average_chunk``);
* ``bracket`` (``-[phi, g] / rho_star``: ``point_bracket`` at the raw midpoints, raw-volume owner mean), ``curv_material`` /
  ``curv_remainder`` / ``curv_total`` (``_continuum_terms``, evolution-weighted owner means), ``diffusion``
  (``D_f div(P_perp grad f) = -D_f O_q3``, q3 face integrand scattered lower ``-`` / upper ``+`` over the owner volume):
  exactly the definitions of ``p_shared.perpendicular_reference_rhs.reference_rhs``; ``Rbar = bracket + curv_total +
  diffusion``;
* ``S = dqbar_dt[:, :4] - Rbar`` (the source of ``d_t q = R + S``);
* ``Dbar_psi (N,)`` the same face functional (``D = 1``) on the polarization variable ``psi = phi + tau n Ti``
  (hot-ion Boussinesq ``omega = L_perp psi``; gradient ``grad phi + tau (Ti grad n + n grad Ti)``), and
  ``sigma = omegabar - Dbar_psi``: the campaign solves ``A psi_h = -(omega_h - sigma) - B g_psi``.
  ``Source(polarization_variable="phi_plus_tau_ti")`` restores the legacy ``psi = phi + tau Ti``. The curvature remainder
  ``cpsi = C(phi) + tau C(Ti)`` in ``continuum_terms`` is a curvature split, independent of this selector.

``Source.boundary(geo, t, params)`` -> ``(BoundaryData of the five columns, psi BoundaryData)`` at the plan's Dirichlet /
Neumann points (value and tangential ``(theta, eta)`` gradient; ``g_N = a . grad f``; ``psi`` has Dirichlet data only);
``boundary_dt`` returns ``(data, d data / dt)`` (forward-mode tangent of the same function).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.dont_write_bytecode = True

from drbx.native.fci_curvature_production_flux import curvature_principal_matrix          # noqa: E402
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData                # noqa: E402

from p09_evolved_mms.fields import TimeState                                                # noqa: E402

N_SLOT, TI_SLOT, PHI_SLOT = 0, 2, 4


class Params(NamedTuple):
    rho_star: object
    tau: object
    D: object            # (4,) diffusion coefficients of density, Te, Ti, vorticity


def default_params(rho_star: float = 0.05, tau: float = 1.0, D=0.01) -> Params:
    return Params(jnp.asarray(rho_star, jnp.float64), jnp.asarray(tau, jnp.float64),
                  jnp.broadcast_to(jnp.asarray(D, jnp.float64), (4,)))


def point_bracket(h, jacobian, grad_a, grad_b):
    """``p05_direct_midpoint_global.direct_operator.point_bracket`` in ``jnp``: ``-h . (grad a x grad b) / |J|``."""
    return -jnp.einsum("...i,...i->...", jnp.cross(h, grad_a), grad_b) / jnp.abs(jacobian)


def continuum_terms(values, gradients, bmag, kvec, tau):
    """``p06_structured_global.numerics._continuum_terms`` in ``jnp`` (material, remainder, total); ``values (5, Q)``,
    ``gradients (5, Q, 3)``, ``bmag (Q,)``, ``kvec (Q, 3)``."""
    n, te, ti = values[0], values[1], values[2]
    curvature = jnp.einsum("pd,fpd->fp", kvec, gradients)
    matrix = curvature_principal_matrix(n, te, ti, bmag, tau)
    material = jnp.einsum("pij,jp->pi", matrix, curvature[:4]) / jnp.maximum(bmag[:, None], 1.0e-30)
    cpsi = curvature[4] + tau * curvature[2]
    coeff = jnp.column_stack((-2 * n / bmag, -4 * te / (3 * bmag), -4 * ti / (3 * bmag), jnp.zeros_like(n)))
    remainder = coeff * cpsi[:, None]
    return material, remainder, material + remainder


def owner_sum(x, weight, owner, count: int):
    """``sum_{rows of the owner} weight * x`` (``segment_sum``; ``x (R, ...)``)."""
    return jax.ops.segment_sum(weight.reshape((-1,) + (1,) * (x.ndim - 1)) * x, owner, num_segments=count)


def face_scatter(face_value, lower, upper, count: int):
    """Lower ``-`` / upper ``+`` scatter of ``face_value (F, ...)`` over ``count`` owners (``-1`` owners dropped)."""
    lo, hi = jnp.where(lower >= 0, lower, count), jnp.where(upper >= 0, upper, count)
    return (jax.ops.segment_sum(-face_value, lo, num_segments=count + 1)[:count]
            + jax.ops.segment_sum(face_value, hi, num_segments=count + 1)[:count])


class Source:
    """Jitted evaluators built from the time-dependent fields (``fields.TimeState``). ``block`` is the point-block size
    of the ``lax.map`` over face / boundary points (bounds the transient memory of the gradient kernels)."""

    def __init__(self, om: float = 1.0, specs=None, block: int = 65536,
                 polarization_variable: str = "phi_plus_tau_pi"):
        if polarization_variable not in ("phi_plus_tau_pi", "phi_plus_tau_ti"):
            raise ValueError("polarization_variable must be 'phi_plus_tau_pi' or 'phi_plus_tau_ti', "
                             f"got {polarization_variable!r}")
        self.polarization_variable = polarization_variable
        self.state = TimeState(specs, om)
        self.block = int(block)
        self.evaluate = jax.jit(self._evaluate)
        self.boundary = jax.jit(self._boundary)
        self.boundary_dt = jax.jit(self._boundary_dt)

    def _grad_blocks(self, points, t):
        """``(value (P, 5), grad (P, 5, 3))`` of ``points (P, 3)`` in fixed blocks."""
        count = points.shape[0]
        nb = -(-count // self.block)
        padded = jnp.pad(points, ((0, nb * self.block - count), (0, 0)), mode="edge").reshape(nb, self.block, 3)
        v, g = jax.lax.map(lambda p: self.state.batch_value_grad(p, t), padded)
        return v.reshape(-1, v.shape[-1])[:count], g.reshape((-1,) + g.shape[2:])[:count]

    def _psi(self, value, grad, tau):
        """``(psi, grad psi)`` from ``value (..., 5)`` and ``grad (..., 5, k)`` (``k``: 3 or the tangential pair)."""
        if self.polarization_variable == "phi_plus_tau_ti":
            return (value[..., PHI_SLOT] + tau * value[..., TI_SLOT],
                    grad[..., PHI_SLOT, :] + tau * grad[..., TI_SLOT, :])
        n, ti = value[..., N_SLOT], value[..., TI_SLOT]
        return (value[..., PHI_SLOT] + tau * n * ti,
                grad[..., PHI_SLOT, :] + tau * (ti[..., None] * grad[..., N_SLOT, :] + n[..., None] * grad[..., TI_SLOT, :]))

    def _evaluate(self, geo, t, params):
        n_own = geo.owner_volume.shape[0]
        tau, vol = params.tau, geo.owner_volume
        v, g, dt = self.state.batch_value_grad_dt(geo.points, t)                    # (Q, 5), (Q, 5, 3), (Q, 5)
        qbar = owner_sum(v, geo.raw_volume, geo.raw_owner, n_own) / vol[:, None]
        dqbar = owner_sum(dt, geo.raw_volume, geo.raw_owner, n_own) / vol[:, None]
        action = point_bracket(geo.h[:, None], geo.jac[:, None], g[:, PHI_SLOT][:, None], g[:, :4])
        bracket = owner_sum(action, geo.raw_volume, geo.raw_owner, n_own) / vol[:, None] / params.rho_star
        material, remainder, total = continuum_terms(v.T, jnp.moveaxis(g, 0, 1), geo.B, geo.K, tau)
        w = geo.evolution_weight
        denom = jnp.maximum(jax.ops.segment_sum(w, geo.raw_owner, num_segments=n_own), 1e-300)[:, None]
        curv = [owner_sum(a, w, geo.raw_owner, n_own) / denom for a in (material, remainder, total)]
        vf, gf = self._grad_blocks(geo.face_points.reshape(-1, 3), t)
        vf = vf.reshape(geo.face_points.shape[0], geo.face_points.shape[1], 5)
        gf = gf.reshape(geo.face_points.shape[0], geo.face_points.shape[1], 5, 3)
        gf = jnp.concatenate([gf[:, :, :4], self._psi(vf, gf, tau)[1][:, :, None]], axis=2)
        face_o = jnp.einsum("fqa,fqka->fk", geo.face_integrand, gf)                  # (F, 5): 4 fields, psi
        o_q3 = face_scatter(face_o, geo.face_lower, geo.face_upper, n_own) / vol[:, None]
        diffusion, dbar_psi = -o_q3[:, :4] * params.D, -o_q3[:, 4]
        rbar = bracket + curv[2] + diffusion
        return {"qbar": qbar, "dqbar_dt": dqbar, "bracket": bracket, "curv_material": curv[0],
                "curv_remainder": curv[1], "curv_total": curv[2], "diffusion": diffusion, "Rbar": rbar,
                "S": dqbar[:, :4] - rbar, "Dbar_psi": dbar_psi, "sigma": qbar[:, 3] - dbar_psi}

    def _boundary(self, geo, t, params):
        v, g = self._grad_blocks(geo.dirichlet_points, t)                           # (Qd, 5), (Qd, 5, 3)
        tangential = jnp.moveaxis(g[:, :, 1:], 1, 2)                                # (Qd, 2, 5)
        _v, gn = self._grad_blocks(geo.neumann_points, t)
        normal = jnp.einsum("qa,qfa->qf", geo.neumann_a, gn)
        psi_value, psi_tangential = self._psi(v, jnp.moveaxis(tangential, 1, 2), params.tau)
        psi = BoundaryData(psi_value[:, None], psi_tangential[:, :, None], None)
        return BoundaryData(v, tangential, normal), psi

    def _boundary_dt(self, geo, t, params):
        t = jnp.asarray(t, jnp.float64)
        return jax.jvp(lambda s: self._boundary(geo, s, params), (t,), (jnp.ones_like(t),))
