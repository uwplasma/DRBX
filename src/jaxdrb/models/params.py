from __future__ import annotations

import equinox as eqx


class DRBParams(eqx.Module):
    # Background-gradient drives (dimensionless)
    omega_n: float = 0.5
    omega_Te: float = 0.0
    omega_Ti: float = 0.0

    # Parallel physics
    eta: float = 1.0  # resistivity-like coefficient
    me_hat: float = 1e-3  # electron inertia knob (set small for resistive branch)

    # Electromagnetism (used by electromagnetic model variants)
    beta: float = 0.0  # normalized beta (sets inductive coupling strength)
    Dpsi: float = 0.0  # optional perpendicular diffusion on psi

    # Ion thermodynamics (used by hot-ion model variants)
    tau_i: float = 0.0  # Ti0/Te0 (tau_i=0 recovers cold ions)
    DTi: float = 0.02

    # Curvature drive
    curvature_on: bool = True

    # Perpendicular diffusion (stabilizing)
    Dn: float = 0.02
    DOmega: float = 0.02
    DTe: float = 0.02

    # Polarization closure safety
    kperp2_min: float = 1e-6

    # Polarization closure model
    #
    # Boussinesq: Omega = -k_perp^2 * phi
    # Non-Boussinesq (linearized about equilibrium n0): Omega = -k_perp^2 * n0 * phi
    boussinesq: bool = True
    n0_min: float = 1e-6
