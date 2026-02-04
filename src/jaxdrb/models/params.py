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
    chi_par_Ti: float = 0.0  # hot-ion model only
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

    # Braginskii-like transport scalings (optional, equilibrium-based).
    #
    # When enabled, a subset of Spitzer/Braginskii temperature scalings are applied using the
    # equilibrium temperature profile(s) along the field line:
    #   - Spitzer resistivity:          η ∝ T_e^{-3/2}
    #   - Spitzer-Härm conduction:      χ_|| ∝ T^{5/2}
    #   - Parallel viscosity proxy:     ν_|| ∝ T^{5/2}
    #
    # This is implemented in a way that keeps the linear RHS matrix-free and differentiable:
    # coefficients are evaluated on the equilibrium profile (Te0, Ti0) and treated as spatially
    # varying multipliers in the linear operators.
    braginskii_on: bool = False
    braginskii_eta_on: bool = True
    braginskii_kappa_e_on: bool = True
    braginskii_kappa_i_on: bool = True
    braginskii_visc_e_on: bool = True
    braginskii_visc_i_on: bool = True
    braginskii_Tref: float = 1.0
    braginskii_T_floor: float = 1e-3
    braginskii_T_smooth: float = 1e-3

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
    sheath_bc_on: bool = True
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

    # Optional sheath heat transmission / energy loss closures (open field lines).
    #
    # These are simple, physically-motivated end-loss terms controlled by sheath heat
    # transmission factors γ. They are intended as a next-step bridge toward fully
    # quantitative SOL modeling with sources, recycling, and state-dependent closures.
    sheath_heat_on: bool = False
    sheath_gamma_auto: bool = True
    sheath_gamma_e: float = 0.0  # used if sheath_gamma_auto=False
    sheath_gamma_i: float = 3.5

    # Optional secondary electron emission (SEE) model (simple constant yield).
    #
    # In an ambipolar, Maxwellian sheath model, SEE reduces the floating potential drop.
    # We represent this by modifying the effective Λ used in the floating-potential shift:
    #   Λ_eff = Λ + ln(1 - δ_SEE),  0 <= δ_SEE < 1.
    sheath_see_on: bool = False
    sheath_see_yield: float = 0.0

    # Boundary-localized end losses at the sheath nodes.
    #
    # In the reduced 1D field-line + Fourier-perp model, MPSE velocity-only boundary conditions can
    # otherwise excite spurious boundary-driven growth in "no-drive" limits. This option adds a
    # lightweight damping term localized at the MPSE nodes for (n, omega, Te, and psi where present).
    #
    # This is separate from `sheath_loss_on` (volumetric) and `sheath_heat_on` (energy transmission),
    # and is on by default for robustness. Disable only if you are explicitly benchmarking sensitivity
    # to end-loss modeling details.
    sheath_end_damp_on: bool = True

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
