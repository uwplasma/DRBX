"""Physical-wall closures and the wall-face flux contract.

A wall closure maps the owner state and the primitive wall trace of every
parallel leg that ends on a physical wall to a complete wall state
``(n, Te, Ti, Vi, Ve)``.  That state is consumed by the existing wall path:
the selected-wall backward-Euler block, the explicit residual and the wall
current exported to the vorticity equation.  For the sheath, the (n, Te, Ti,
Vi) rows of each wall leg are the conservative flux difference
:func:`sheath_wall_leg_residual`, handed to the same block as its coupled
residual, so the wall removes exactly the contract fluxes below.

:func:`wall_face_fluxes` is the flux contract: from one wall state it returns
the outward number, parallel-momentum, electron/ion energy fluxes and current
per unit cross-section normal to ``b``.  Multiplying by the leg cross-section
``A_perp = V_cell / dx_leg`` (the measure the discrete wall-leg update divides
by) gives the per-leg wall loss; the wall surface area is
``A_perp / sin(alpha)`` with ``sin(alpha) = |b.n_w|``, so totals are
independent of ``alpha`` and only surface densities depend on it.

Normalization: n in n0, Te in Te0, Ti in Ti0 with ``tau = Ti0/Te0``,
velocities in ``c_s0 = sqrt(Te0/m_i)``.  The local sound speed is
``c_s = sqrt(Te + tau*Ti)`` (isothermal electrons and ions, gamma = 1).
Energy fluxes are in ``n0 Te0 c_s0``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax.numpy as jnp

WALL_CLOSURES = ("bohm", "zero-flow")
_FLOOR = 1.0e-30


@dataclass(frozen=True)
class SheathClosure:
    """Bohm sheath closure options (hashable static model configuration).

    ``electron_heat_transmission`` and ``ion_heat_transmission`` are the
    *total* Stangeby transmission coefficients at Mach 1 (q = gamma n T c_s).
    ``None`` for electrons selects the floating-wall value
    ``2/(1-d) - 0.5 ln[2 pi (me/mi)(1 + Ti/Te)/(1-d)^2]`` with secondary
    emission ``d``.  The wall energy flux is ``gamma n T V`` in total: the
    wall-leg flux difference replaces the advected enthalpy ``(5/2) n T V``
    of the owner centre by it, so no part is counted twice.  The ion kinetic
    energy of the sonic outflow is carried by the Vi row and is not removed
    from the temperatures again.
    """

    electron_heat_transmission: float | None = None
    ion_heat_transmission: float = 2.5
    secondary_electron_emission: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.secondary_electron_emission < 1.0:
            raise ValueError("secondary_electron_emission must be in [0, 1)")
        if self.ion_heat_transmission < 2.5:
            raise ValueError("ion_heat_transmission must be >= 5/2 (advected enthalpy)")
        if self.electron_heat_transmission is not None and self.electron_heat_transmission < 2.5:
            raise ValueError("electron_heat_transmission must be >= 5/2 (advected enthalpy)")

    def gamma_e(self, ti_over_te, mi_over_me):
        """Total electron transmission (Stangeby floating wall when unset)."""
        if self.electron_heat_transmission is not None:
            return float(self.electron_heat_transmission)
        d = float(self.secondary_electron_emission)
        return 2.0 / (1.0 - d) - 0.5 * jnp.log(
            2.0 * math.pi * (1.0 + ti_over_te) / (mi_over_me * (1.0 - d) ** 2)
        )


class WallFaceFluxes(NamedTuple):
    """Outward fluxes through a wall leg, per unit cross-section normal to b."""

    particle: jnp.ndarray
    parallel_momentum: jnp.ndarray
    electron_energy: jnp.ndarray
    ion_energy: jnp.ndarray
    kinetic_energy: jnp.ndarray
    current: jnp.ndarray


def sound_speed(Te, Ti, tau) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(Te + tau * Ti, _FLOOR))


def bohm_wall_state(trace, center, direction, tau) -> jnp.ndarray:
    """Return the Bohm sheath-entrance state for wall legs.

    ``trace`` holds the physical wall trace (Te, Ti from the Neumann wall
    law); ``direction`` is +1 for a forward (+b) leg and -1 for a backward leg,
    i.e. ``sign(b.n_w)`` of the wall it hits.  The flow leaves at
    ``V_w = s*max(s*Vi_owner, c_s)`` (Bohm inequality) and the electrons follow
    at the same velocity (local ambipolar floating wall: zero current per
    leg).  The entrance density conserves the total parallel momentum flux
    over the half leg, ``n_w (V_w^2 + c_s^2) = n (Vi^2 + Te + tau Ti)``, the
    isothermal presheath relation ``n_w = n (1 + M^2)/2`` (two-point
    ``n_t = n_u/2`` for a stagnant owner).  The Neumann density trace is
    not used: with it the conservative momentum row is unstable.
    """

    trace = jnp.asarray(trace, dtype=jnp.float64)
    center = jnp.asarray(center, dtype=jnp.float64)
    s = jnp.asarray(direction, dtype=jnp.float64)
    cs = sound_speed(trace[..., 1], trace[..., 2], tau)
    velocity = s * jnp.maximum(s * center[..., 3], cs)
    n, Te, Ti, Vi, Ve = (center[..., i] for i in range(5))
    momentum = n * Vi * Vi + n * (Te + tau * Ti)
    density = momentum / (velocity * velocity + cs * cs)
    return trace.at[..., 0].set(density).at[..., 3].set(velocity).at[..., 4].set(velocity)


def wall_face_fluxes(wall_state, direction, tau, gamma_e, gamma_i, owner_velocity=0.0) -> WallFaceFluxes:
    """Flux contract: outward fluxes carried by a complete wall state.

    ``particle = n V s``, ``parallel_momentum = n V^2 + n (Te + tau Ti)``
    (outward normal momentum flux), energies use the total transmission
    coefficients at the sheath entrance, ``q_e = gamma_e n Te |V|`` and
    ``q_i = tau gamma_i n Ti |V|``, the ion flow kinetic energy
    ``(1/2) Vi_owner^2`` per lost ion (the momentum flux is continuous over
    the half leg, so the flow is not re-accelerated at the expense of the
    temperatures) and ``current = n (Vi - Ve) s``.
    """

    w = jnp.asarray(wall_state, dtype=jnp.float64)
    s = jnp.asarray(direction, dtype=jnp.float64)
    n, Te, Ti, Vi, Ve = (w[..., i] for i in range(5))
    outflow = n * Vi * s
    return WallFaceFluxes(
        particle=outflow,
        parallel_momentum=n * Vi * Vi + n * (Te + tau * Ti),
        electron_energy=gamma_e * Te * outflow,
        ion_energy=tau * gamma_i * Ti * outflow,
        kinetic_energy=0.5 * jnp.square(jnp.asarray(owner_velocity, dtype=jnp.float64)) * outflow,
        current=n * (Vi - Ve) * s,
    )


def sheath_wall_leg_residual(wall_state, center, dx, direction, tau, gamma_e, gamma_i) -> jnp.ndarray:
    """Conservative wall-leg rates for the (n, Te, Ti, Vi) rows.

    The wall leg of the owner cell is the flux difference between the wall
    face and the owner centre, ``-s (F_w - F_c) / dx`` with ``F`` along +b and
    ``s = sign(b.n_w)``, for particles ``n Ve``, parallel momentum
    ``n Vi^2 + n (Te + tau Ti)`` and thermal energies.  The wall energy flux
    is the total sheath transmission ``gamma n T V``; the centre flux is the
    advected enthalpy ``(5/2) n T V``.  The outward losses are therefore
    exactly :func:`wall_face_fluxes` of the same wall state, with no part of
    the enthalpy counted twice.  Rates are written for the primitive
    variables (the Ve row is zero: it keeps the characteristic wall law).
    """

    w = jnp.asarray(wall_state, dtype=jnp.float64)
    c = jnp.asarray(center, dtype=jnp.float64)
    s = jnp.asarray(direction, dtype=jnp.float64)
    factor = -s / jnp.maximum(jnp.abs(dx), _FLOOR)
    n_c = jnp.maximum(c[..., 0], _FLOOR)

    def fluxes(u, ge, gi):
        n, Te, Ti, Vi, Ve = (u[..., i] for i in range(5))
        return (
            n * Ve,
            n * Vi * Vi + n * (Te + tau * Ti),
            ge * n * Te * Ve,
            gi * n * Ti * Ve,
        )

    Fw = fluxes(w, gamma_e, gamma_i)
    Fc = fluxes(c, 2.5, 2.5)
    d_particle = Fw[0] - Fc[0]
    rate_n = factor * d_particle
    rate_vi = factor * ((Fw[1] - Fc[1]) - c[..., 3] * d_particle) / n_c
    rate_te = factor * ((Fw[2] - Fc[2]) - 1.5 * c[..., 1] * d_particle) / (1.5 * n_c)
    rate_ti = factor * ((Fw[3] - Fc[3]) - 1.5 * c[..., 2] * d_particle) / (1.5 * n_c)
    return jnp.stack((rate_n, rate_te, rate_ti, rate_vi, jnp.zeros_like(rate_n)), axis=-1)


__all__ = [
    "WALL_CLOSURES",
    "SheathClosure",
    "WallFaceFluxes",
    "bohm_wall_state",
    "sheath_wall_leg_residual",
    "sound_speed",
    "wall_face_fluxes",
]
