"""Reduced isothermal scrape-off-layer (SOL) flux-tube parallel transport.

An open-field-line model: on the open slab geometry
(:func:`dkx.geometry.build_open_slab_geometry`), plasma is transported along
the field to the two target plates, where a Bohm sheath drains it at the sound
speed. The model evolves the parallel density ``n`` and parallel momentum
``m = n v`` as an isothermal Euler system in the field-parallel coordinate ``z``,

    d n / dt + d (n v) / dz = S_n
    d m / dt + d (n v^2 + n c_s^2) / dz = 0

with an upstream particle source ``S_n`` and Bohm sheath outflow (``|v| >= c_s``)
at ``z = 0`` and ``z = L``. Faces use a Rusanov (local Lax--Friedrichs) flux; the
whole update is pure JAX (``jit``/``grad``/``vmap`` transparent). The steady
state is the classic two-point SOL solution: the flow accelerates from a
stagnation point to the sound speed at each target, and the target density is
half the upstream density.

The transport acts along the parallel (``z``) axis of the ``(nx, ny, nz)`` field
arrays, so each ``(x, y)`` column is an independent flux tube sharing the open
target plates.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..geometry import FciGeometry3D

__all__ = [
    "SolFluxTubeParameters",
    "sol_flux_tube_source",
    "sol_flux_tube_rhs",
    "sol_flux_tube_step",
    "sol_flux_tube_run",
]


@dataclass(frozen=True)
class SolFluxTubeParameters:
    """Parameters of the reduced isothermal SOL flux tube."""

    sound_speed: float = 1.0            # normalized isothermal sound speed c_s
    source_amplitude: float = 0.02      # peak upstream particle source
    source_width: float = 4.0           # parallel width of the upstream source
    density_floor: float = 1.0e-6


def sol_flux_tube_source(geometry: FciGeometry3D, params: SolFluxTubeParameters) -> jnp.ndarray:
    """Upstream particle source: a Gaussian centered at the parallel midplane."""

    z = jnp.asarray(geometry.grid.z.centers, dtype=jnp.float64)
    midplane = 0.5 * (float(z[0]) + float(z[-1]))
    profile = params.source_amplitude * jnp.exp(-((z - midplane) ** 2) / params.source_width**2)
    return jnp.broadcast_to(profile[None, None, :], geometry.shape)


def _flux(density: jnp.ndarray, momentum: jnp.ndarray, c_s: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    velocity = momentum / density
    return momentum, momentum * velocity + density * c_s**2


def _rusanov(nL, mL, nR, mR, c_s):
    fL_n, fL_m = _flux(nL, mL, c_s)
    fR_n, fR_m = _flux(nR, mR, c_s)
    speed = jnp.maximum(jnp.abs(mL / nL), jnp.abs(mR / nR)) + c_s
    flux_n = 0.5 * (fL_n + fR_n) - 0.5 * speed * (nR - nL)
    flux_m = 0.5 * (fL_m + fR_m) - 0.5 * speed * (mR - mL)
    return flux_n, flux_m


def sol_flux_tube_rhs(
    density: jnp.ndarray,
    momentum: jnp.ndarray,
    geometry: FciGeometry3D,
    params: SolFluxTubeParameters,
    source: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Parallel-transport RHS ``(dn/dt, dm/dt)`` with Bohm sheath targets."""

    c_s = float(params.sound_speed)
    dz = jnp.asarray(geometry.spacing.dz, dtype=jnp.float64)
    n = jnp.maximum(density, float(params.density_floor))
    v = momentum / n

    # Interior faces along z (between adjacent cells).
    interior_n, interior_m = _rusanov(n[:, :, :-1], momentum[:, :, :-1], n[:, :, 1:], momentum[:, :, 1:], c_s)

    # Bohm sheath outflow at the two targets: |v| >= c_s pointing into the plate.
    v_left = jnp.minimum(v[:, :, 0], -c_s)
    v_right = jnp.maximum(v[:, :, -1], c_s)
    left_n, left_m = _flux(n[:, :, 0], n[:, :, 0] * v_left, c_s)
    right_n, right_m = _flux(n[:, :, -1], n[:, :, -1] * v_right, c_s)

    face_n = jnp.concatenate([left_n[:, :, None], interior_n, right_n[:, :, None]], axis=2)
    face_m = jnp.concatenate([left_m[:, :, None], interior_m, right_m[:, :, None]], axis=2)

    d_density = -(face_n[:, :, 1:] - face_n[:, :, :-1]) / dz + source
    d_momentum = -(face_m[:, :, 1:] - face_m[:, :, :-1]) / dz
    return d_density, d_momentum


def sol_flux_tube_step(density, momentum, geometry, params, source, dt):
    """One RK4 step of the reduced SOL flux tube (density floored to stay positive)."""

    def rhs(n, m):
        return sol_flux_tube_rhs(n, m, geometry, params, source)

    k1n, k1m = rhs(density, momentum)
    k2n, k2m = rhs(density + 0.5 * dt * k1n, momentum + 0.5 * dt * k1m)
    k3n, k3m = rhs(density + 0.5 * dt * k2n, momentum + 0.5 * dt * k2m)
    k4n, k4m = rhs(density + dt * k3n, momentum + dt * k3m)
    density = density + dt / 6.0 * (k1n + 2 * k2n + 2 * k3n + k4n)
    momentum = momentum + dt / 6.0 * (k1m + 2 * k2m + 2 * k3m + k4m)
    return jnp.maximum(density, float(params.density_floor)), momentum


def sol_flux_tube_run(density, momentum, geometry, params, source, *, dt, steps):
    """Advance ``steps`` RK4 steps with a jitted ``lax.scan``; return the final state."""

    @jax.jit
    def _run(n0, m0):
        def body(carry, _):
            n, m = carry
            return sol_flux_tube_step(n, m, geometry, params, source, dt), None

        (n_final, m_final), _ = jax.lax.scan(body, (n0, m0), None, length=steps)
        return n_final, m_final

    return _run(density, momentum)
