from __future__ import annotations

import equinox as eqx


class DRBParams(eqx.Module):
    # Background-gradient drives (dimensionless)
    omega_n: float = 0.5
    omega_Te: float = 0.0

    # Parallel physics
    eta: float = 1.0  # resistivity-like coefficient
    me_hat: float = 1e-3  # electron inertia knob (set small for resistive branch)

    # Curvature drive
    curvature_on: bool = True

    # Perpendicular diffusion (stabilizing)
    Dn: float = 0.02
    DOmega: float = 0.02
    DTe: float = 0.02

    # Polarization closure safety
    kperp2_min: float = 1e-6
