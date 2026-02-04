from __future__ import annotations

import equinox as eqx

from jaxdrb.models.bcs import LineBCs


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

    # Parallel closures (optional; act along the field line)
    #
    # These are simple, robust placeholders for Braginskii-like transport:
    #   χ_|| ∂_||^2 Te     (parallel electron heat conduction)
    #   ν_|| ∂_||^2 v_||   (parallel viscosity / diffusion of parallel flow)
    #
    # These are most relevant for open-field-line (SOL) studies, and can help
    # regularize small-scale parallel structure in linear problems.
    chi_par_Te: float = 0.0
    nu_par_e: float = 0.0
    nu_par_i: float = 0.0

    # Simple volumetric sinks (optional)
    #
    # These are linear damping terms used as crude proxies for sources/sinks that
    # appear in full SOL models (ionization, radiation, cross-field losses, etc.).
    nu_sink_n: float = 0.0
    nu_sink_Te: float = 0.0
    nu_sink_vpar: float = 0.0

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
    # There are two related but distinct mechanisms:
    #
    # - `sheath_bc_on`: Loizu-style magnetic-pre-sheath entrance boundary conditions (MPSE),
    #   enforced weakly at the two ends of an *open* field line.
    # - `sheath_loss_on`: a lightweight volumetric end-loss proxy nu_sh ~ 2/L_parallel, useful
    #   for quick stabilization studies (not a substitute for MPSE BCs).
    #
    # Backwards compatibility: older configs used `sheath_on` + `sheath_nu_factor` as the
    # volumetric loss proxy.
    sheath_bc_on: bool = False
    sheath_bc_nu_factor: float = 1.0
    # MPSE model selector:
    #   0 = simple (velocity-only, linearized or nonlinear)
    #   1 = Loizu 2012 "full set" (linearized: includes density/potential/vorticity/Te-gradient constraints)
    sheath_bc_model: int = 0
    sheath_cos2: float = 1.0  # proxy for cos^2(incidence angle) in Loizu 2012 vorticity BC
    sheath_bc_linearized: bool = True
    sheath_lambda: float = 3.28  # ~ 0.5 ln(mi/(2π me)) for hydrogen
    sheath_delta: float = 0.0  # ion transmission correction (cold ions -> 0)
    sheath_Te_floor: float = 1e-6

    sheath_loss_on: bool = False
    sheath_loss_nu_factor: float = 1.0

    # Deprecated (kept for compatibility; treated as sheath_loss_*):
    sheath_on: bool = False
    sheath_nu_factor: float = 1.0

    # Optional user-defined line boundary conditions (Dirichlet/Neumann/Periodic).
    #
    # These are enforced weakly as additional RHS terms and are primarily intended
    # for benchmarking and for nonlinear-transition preparation work. In most SOL
    # studies you will use MPSE/sheath BCs instead.
    line_bcs: LineBCs = eqx.field(static=True, default=LineBCs.disabled())
