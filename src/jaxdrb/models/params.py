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

    # Braginskii coefficients (dimensionless, in the same normalization as the model).
    #
    # Many drift-reduced Braginskii implementations include an electron "thermal force" term
    # in Ohm's law, often written as ∇_||(phi - n - 1.71 Te) in (Te/e) potential units.
    alpha_Te_ohm: float = 1.71

    # Optional open-field-line / sheath closure knobs.
    #
    # When `sheath_on=True` and the geometry provides `sheath_mask` and `sheath_sign`, we
    # apply Bohm-sheath entrance conditions through a penalty relaxation at the two ends.
    # This enables SOL-like studies without changing the matrix-free workflows.
    sheath_on: bool = False
    sheath_nu_factor: float = 1.0
    sheath_Te_floor: float = 1e-6
