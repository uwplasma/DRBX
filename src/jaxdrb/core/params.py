from __future__ import annotations

from dataclasses import replace
from typing import Literal

import equinox as eqx

from jaxdrb.bc import BC2D
from jaxdrb.core.bcs import LineBCs
from jaxdrb.core.closures.neutrals import NeutralParams


class PhysicsParams(eqx.Module):
    """Core physics toggles and couplings for the unified DRB system."""

    nonlinear_on: bool = eqx.field(static=True, default=True)
    hot_ion_on: bool = eqx.field(static=True, default=False)
    em_on: bool = eqx.field(static=True, default=False)
    neutrals_on: bool = eqx.field(static=True, default=False)
    boussinesq: bool = eqx.field(static=True, default=True)
    non_boussinesq_perturbed_density_on: bool = eqx.field(static=True, default=False)
    log_n: bool = eqx.field(static=True, default=False)
    log_Te: bool = eqx.field(static=True, default=False)

    # Background-gradient drives / coupling.
    omega_n: float = 0.0
    omega_Te: float = 0.0
    omega_Ti: float = 0.0
    omega_drive_mask: Literal["all", "closed", "open"] = "all"
    drive_from_equilibrium_on: bool = False
    drive_equilibrium_mode: Literal["auto", "sol", "constant"] = "auto"
    drive_equilibrium_n0: float | None = None
    drive_equilibrium_Te0: float | None = None
    drive_equilibrium_Ti0: float | None = None

    # HW-like coupling (retained for compatibility).
    kappa: float = 0.0
    kappa_profile: Literal["constant", "cosine"] = "constant"
    kappa_theta0: float = 0.0
    alpha: float = 0.0
    alpha_nonzonal_only: bool = False

    # Parallel coupling / Ohm's law.
    kpar: float = 0.0
    eta: float = 0.0
    eta_par: float = 0.0
    me_hat: float = 1.0
    alpha_Te_ohm: float = 1.71
    alpha_Ti: float = 1.0
    alpha_Ti_ohm: float = 0.0
    # Conservative parallel pressure transport coefficients:
    # dp = c_flux * (-Div_par(P v)) + c_work * v * Grad_par(P).
    parallel_pressure_model: Literal["custom", "hermes_vgradp", "hermes_pdivv"] = "custom"
    parallel_pressure_flux_coeff: float = 1.0
    parallel_pressure_work_coeff: float = 0.0
    # Optional compressional heating/cooling in temperature equations:
    # dT/dt ... -(gamma-1) T div(v_parallel), matching Hermes p_div_v split.
    parallel_temperature_compression_on: bool = False
    parallel_temperature_compression_coeff: float = 2.0 / 3.0

    # Electromagnetism.
    beta: float = 0.0
    Dpsi: float = 0.0

    # Hot-ion parameters.
    tau_i: float = 1.0
    # Average ion atomic mass (Hermes-style vorticity coefficient).
    average_atomic_mass: float = 1.0

    # Curvature drive.
    curvature_on: bool = False
    curvature_coeff: float = 0.0
    curvature_model: str = "slab"
    curvature_theta_scale: float | None = None
    curvature_scale: float | None = None
    curvature_Te_coeff: float | None = None
    curvature_n_coeff: float = 1.0

    # Diamagnetic drift (Hermes-style) using Curl(b/B) from bxcv/curvature vectors.
    diamagnetic_on: bool = False
    diamag_form: float = 1.0
    diamag_form_profile: str | None = None
    diamag_density_model: Literal["electron", "ion", "none"] = "electron"
    diamag_bndry_flux: bool = True
    diamagnetic_flux_scheme: Literal["fd", "fv"] = "fd"
    diamagnetic_use_jacobian: bool = False

    # Diamagnetic current (DivJdia) contribution in vorticity equation.
    diamagnetic_current_on: bool = False
    diamagnetic_current_scale: float = 1.0
    diamagnetic_current_bndry_flux: bool = True
    diamagnetic_current_energy_on: bool = True

    # Diamagnetic polarisation current (adds div((1/B^2) grad p_i) to omega operator).
    diamagnetic_polarisation_on: bool = False
    diamagnetic_polarisation_scale: float = 1.0

    # Polarization closure.
    n0: float = 1.0
    n0_min: float = 1e-6
    n0_max: float | None = None
    k2_min: float = 1e-12
    kperp2_min: float = 1e-6

    # Floors (Hermes-style temperature floor used in sound speed / low-n diffusion).
    temperature_floor: float = 0.0

    # Log-form state variables.
    log_n_clip: float | None = 50.0
    log_Te_clip: float | None = 50.0

    # Generic volumetric sources (Gaussian sources).
    source_on: bool = False
    source_profile: Literal["gaussian_x", "gaussian_xy"] = "gaussian_x"
    source_x_mode: Literal["grid", "bout"] = "grid"
    source_only_in_core: bool = False
    source_n0: float = 0.0
    source_Te0: float = 0.0
    source_Te_is_pressure: bool = False
    source_x0: float = 0.0
    source_y0: float = 0.0
    source_width_x: float = 1.0
    source_width_y: float = 1.0
    # Electron pressure closure used in Hermes-style polarization:
    # "nTe" -> p_e = n * Te (default, physically consistent for Te state variable)
    # "Te"  -> p_e = Te (for legacy pressure-state compatibility)
    electron_pressure_model: Literal["nTe", "Te"] = "nTe"


class TransportParams(eqx.Module):
    """Transport/dissipation controls (diffusion, hyperdiffusion, damping)."""

    # Diffusion and dissipation.
    Dn: float = 0.0
    DOmega: float = 0.0
    Dvpar: float = 0.0
    DTe: float = 0.0
    DTi: float = 0.0
    Dpsi: float = 0.0
    chi_par: float = 0.0
    chi_par_Te: float = 0.0
    chi_par_Ti: float = 0.0
    nu_par_e: float = 0.0
    nu_par_i: float = 0.0
    nu_sink_n: float = 0.0
    nu_sink_Te: float = 0.0
    nu_sink_vpar: float = 0.0

    # Hyperdiffusion.
    Dn4: float = 0.0
    DOmega4: float = 0.0
    DTe4: float = 0.0
    DTi4: float = 0.0
    Dpsi4: float = 0.0
    nu4_n: float = 0.0
    nu4_omega: float = 0.0

    # Optional drag/damping.
    mu_zonal_omega: float = 0.0
    mu_lin_n: float = 0.0
    mu_lin_omega: float = 0.0
    mu_lin_vpar_e: float = 0.0
    mu_lin_vpar_i: float = 0.0
    mu_lin_Te: float = 0.0

    # Low-density perpendicular diffusion (Hermes-style).
    low_n_diffuse_perp_on: bool = False
    low_n_diffuse_perp_coeff: float = 1.0

    # Parallel dissipation toggles (Div_par terms).
    vort_par_dissipation: float = 0.0
    phi_par_dissipation: float = 0.0
    phi_par_dissipation_model: Literal["laplacian", "lax_fv"] = "laplacian"
    phi_dissipation_on: bool = True
    phi_sheath_dissipation_on: bool = False
    core_vorticity_damping_on: bool = True
    core_vorticity_damping_coeff: float = 0.0

    # Braginskii coefficient scalings.
    braginskii_on: bool = False
    braginskii_state_dependent_on: bool = False
    braginskii_eta_on: bool = True
    braginskii_kappa_e_on: bool = True
    braginskii_kappa_i_on: bool = True
    braginskii_visc_e_on: bool = True
    braginskii_visc_i_on: bool = True
    braginskii_Tref: float = 1.0
    braginskii_T_floor: float = 1e-3
    braginskii_T_smooth: float = 1e-3

    # Braginskii collision-driven closures.
    braginskii_heat_exchange_on: bool = False
    braginskii_friction_on: bool = False
    braginskii_frictional_heating_on: bool = True
    classical_diffusion_on: bool = False

    braginskii_nu_ei: float = 0.0
    braginskii_nu_ii: float = 0.0
    braginskii_nu_floor: float = 1e-12
    braginskii_friction_coeff: float = 0.51

    classical_diffusion_custom_D: float = -1.0
    classical_diffusion_custom_kappa_e: float = -1.0
    classical_diffusion_custom_kappa_i: float = -1.0


class SheathParams(eqx.Module):
    """Sheath / open-field-line closures."""

    sheath_on: bool = False
    sheath_nu_factor: float = 1.0
    sheath_bc_on: bool = True
    sheath_bc_nu_factor: float = 1.0
    sheath_bc_model: int | str = 0
    sheath_bc_linearized: bool = True
    sheath_lambda: float = 3.28
    sheath_Te_floor: float = 1e-6
    sheath_heat_on: bool = False
    sheath_gamma_auto: bool = True
    sheath_gamma_e: float = 0.0
    sheath_gamma_i: float = 3.5
    sheath_energy_model: Literal["relaxation", "hermes_flux"] = "relaxation"
    sheath_energy_flux_scale: float = 1.0
    sheath_secondary_electron_coef: float = 0.0
    sheath_wall_potential: float = 0.0
    sheath_floor_potential: bool = True
    sheath_electron_adiabatic: float = 5.0 / 3.0
    sheath_ion_adiabatic: float = 5.0 / 3.0
    sheath_see_on: bool = False
    sheath_see_yield: float = 0.0
    sheath_end_damp_on: bool = True
    sheath_loss_on: bool = False
    sheath_loss_nu_factor: float = 1.0
    sheath_nu_mom: float = 0.0
    sheath_nu_particle: float = 0.0
    sheath_nu_energy: float = 0.0
    sheath_delta: float = 0.0
    sheath_cos2: float = 1.0
    sheath_bc_model_fci: Literal["simple", "loizu_linear"] = "simple"


class SOLParams(eqx.Module):
    """SOL-like closed→open radial setup (2D drivers/closures)."""

    sol_on: bool = False
    sol_xs: float = 0.0
    sol_width: float = 0.05
    sol_open_left: bool = False
    sol_mask_y_taper: float = 0.0
    sol_n_core: float = 1.0
    sol_n_sol: float = 0.2
    sol_Te_core: float = 1.0
    sol_Te_sol: float = 0.2
    sol_relax_core: float = 0.2
    sol_relax_open: float = 0.6
    sol_sink_open_n: float = 0.0
    sol_sink_open_Te: float = 0.0
    sol_sink_open_omega: float = 0.0
    sol_sink_open_omega_mode: str = "local"
    sol_sink_open_vpar: float = 0.0
    sol_nonlinear_open_scale: float = 1.0
    sol_n_floor: float = 0.0
    sol_Te_floor: float = 0.0
    sol_source_n0: float = 0.0
    sol_source_Te0: float = 0.0
    sol_source_xs: float = 0.0
    sol_source_width: float = 1.0
    sol_source2_n0: float = 0.0
    sol_source2_Te0: float = 0.0
    sol_source2_xs: float = 0.0
    sol_source2_width: float = 1.0
    sol_source_mask: str = "all"
    sol_source_y_taper: float = 0.0
    sol_parallel_loss_on: bool = False
    sol_parallel_loss_model: Literal["bohm", "bohm_exp", "bohm_linear"] = "bohm"
    sol_parallel_loss_q: float = 4.0
    sol_parallel_loss_coeff: float = 1.0
    sol_parallel_loss_lambda: float = 3.0
    sol_parallel_loss_Te_floor: float = 1e-6
    sol_parallel_loss_vpar_on: bool = False
    sol_parallel_loss_omega_on: bool = False
    sol_sheath_omega_on: bool = False
    sol_sheath_omega_coeff: float = 1.0
    sol_sheath_phi_on: bool = False
    sol_sheath_phi_model: str = "exp"
    sol_sheath_phi_lambda: float = 3.0
    sol_sheath_phi_coeff: float = 1.0
    sol_sheath_phi_dissipation_on: bool = True
    sol_sheath_phi_Te_floor: float = 1e-6
    sol_sheath_phi_clip: float = 10.0
    sol_sheath_phi_implicit: bool = False
    sol_sheath_phi_implicit_solver: str = "gmres"
    sol_sheath_phi_implicit_rtol: float = 1e-6
    sol_sheath_phi_implicit_atol: float = 1e-8
    sol_sheath_phi_implicit_maxiter: int = 100
    sol_sheath_phi_implicit_restart: int = 30
    sol_edge_relax_on: bool = False
    sol_edge_relax_nu: float = 0.0
    sol_edge_n_right: float = 0.1
    sol_edge_Te_right: float = 0.1
    sol_edge_relax_apply_y: bool = True
    sol_omega_bc_dirichlet_on: bool = False
    sol_omega_bc_value: float = 0.0
    sol_omega_bc_nu: float = 1.0
    sol_omega_bc_apply_y: bool = False
    sol_vpar_bc_dirichlet_on: bool = False
    sol_vpar_bc_value: float = 0.0
    sol_vpar_bc_nu: float = 1.0
    sol_phi_bc_on: bool = False
    sol_phi_bc_lambda: float = 3.0


class ClosureParams(eqx.Module):
    """Closures (sheath, SOL, neutrals, line BCs)."""

    sheath: SheathParams = eqx.field(default_factory=SheathParams)
    sol: SOLParams = eqx.field(default_factory=SOLParams)
    neutrals: NeutralParams = eqx.field(default_factory=NeutralParams)
    line_bcs: LineBCs = eqx.field(static=True, default=LineBCs.disabled())


class NumericsParams(eqx.Module):
    """Numerical and solver knobs."""

    bracket: Literal["spectral", "arakawa", "centered"] = "arakawa"
    bracket_zero_mean: bool = False
    exb_scale: float = 1.0
    exb_y_scale: float = 1.0
    # ExB advection form: "bracket" (default) or "flux" (Hermes-style conservative form).
    exb_advection_form: Literal["bracket", "flux"] = "bracket"
    # Flux-form ExB stencil in field-aligned geometry:
    # - "centered": centered flux divergence
    # - "hermes_fromm": Hermes/BOUT-style Fromm-upwind X-Z transport
    exb_flux_scheme: Literal["centered", "hermes_fromm"] = "centered"
    # When using flux-form ExB advection, advect conservative variables (n, n*v, p).
    exb_advect_conservative: bool = False
    # Include the metric-coupled X-Y ExB advection contribution present in
    # field-aligned BOUT/Hermes coordinates (poloidal flow term).
    exb_poloidal_flows: bool = False
    # Optional scale for the metric-coupled X-Y ExB advection contribution.
    exb_poloidal_scale: float = 1.0
    # Hermes/BOUT option equivalent to `neumann_boundary_average_z=true` for
    # perpendicular operators: when x-boundary BC is Neumann, average boundary
    # values over the perpendicular-y (z-like) index before ghost padding.
    neumann_boundary_average_y: bool = False
    perp_operator: Literal["spectral", "fd", "fv"] = "spectral"
    parallel_z_mode: Literal["vmap", "scan"] = "vmap"
    parallel_limiter: Literal["none", "minmod", "mc"] = "none"
    parallel_flux_scheme: Literal["rusanov", "lax"] = "rusanov"
    parallel_fixflux: bool = True
    parallel_flux_conservative: bool = False
    parallel_momentum_model: Literal["reduced", "conservative"] = "reduced"
    parallel_transform: Literal["none", "shifted"] = "none"
    # Use Bohm-target sheath velocities on parallel boundary faces for open-field
    # conservative fluxes (Hermes-like boundary flux treatment).
    parallel_use_sheath_targets: bool = False
    # Sheath target usage in conservative parallel fluxes:
    # "boundary_flux" keeps interior v|| and applies Bohm targets only on sheath faces;
    # "replace_boundary" overwrites boundary-cell v|| with Bohm targets (legacy mode).
    parallel_sheath_flux_mode: Literal["boundary_flux", "replace_boundary"] = "replace_boundary"
    # Scale applied to sheath boundary face fluxes in conservative parallel
    # transport (useful for cross-code parity calibration).
    parallel_boundary_flux_scale: float = 1.0
    dpar_factor_scale: float = 1.0
    use_gpar_flux: bool = False
    poisson_scale: float = 1.0
    # Use B^2-weighted vorticity definition: omega = (B^2/n) div((n/B^2) grad phi)
    poisson_b_weighted: bool = False
    # For B-weighted vorticity, choose omega definition:
    # "scaled" => omega = (B^2/n) div((n/B^2) grad phi)
    # "hermes" => omega = div((Abar*n0/B^2) grad phi) (+ diamag pol)
    poisson_b_weighted_mode: Literal["scaled", "hermes"] = "scaled"
    # Hermes-style split of k_y=0 component (LaplaceXY) vs k_y!=0 (Laplacian).
    poisson_split_n0: bool = False
    # BOUT/Hermes INVERT_SET-style Poisson boundary handling: use field/guess
    # boundary values as Dirichlet data in non-periodic x.
    poisson_invert_set: bool = False
    # Hermes INVERT_SET applies boundary values at the cell face (midpoint
    # between guard and first interior cell). When enabled, approximate this
    # half-cell shift by using the average of boundary and adjacent interior
    # values as Dirichlet data in Poisson/polarization operators.
    poisson_invert_set_midpoint: bool = True
    poisson_metric_on: bool = False
    poisson: Literal["spectral", "cg_fd", "mixed_fft"] = "spectral"
    poisson_force_spectral_when_periodic: bool = True
    poisson_force_fd_fft_when_nonperiodic: bool = True
    poisson_preconditioner: str = "auto"
    poisson_cg_maxiter: int = 300
    poisson_cg_tol: float = 1e-8
    poisson_cg_atol: float = 0.0
    poisson_gauge_epsilon: float | None = None
    poisson_maxiter: int = 400
    poisson_tol: float = 1e-10
    dealias_on: bool = True

    # Non-Boussinesq variable-coefficient polarization solve settings.
    polarization_cg_maxiter: int = 400
    polarization_cg_tol: float = 1e-8
    polarization_cg_atol: float = 0.0
    polarization_preconditioner: str = "auto"
    polarization_precond_shift: float = 1e-12

    # FCI parallel-operator knobs.
    use_target_aware_dpar: bool = True
    parallel_sign: float = 1.0
    target_scheme: str = "appendix_b"

    # Operator split toggles (conservative/source/dissipative).
    operator_split_on: bool = False
    operator_conservative_on: bool = True
    operator_source_on: bool = True
    operator_dissipative_on: bool = True

    # Optional term schedule override (list of term names).
    term_schedule: tuple[str, ...] | None = eqx.field(static=True, default=None)
    term_schedule_preset: str | None = eqx.field(static=True, default=None)

    # Include phi boundary relaxation as an RHS term (for implicit solvers).
    phi_relax_in_rhs: bool = False


class BCParams(eqx.Module):
    """Boundary condition overrides for fields."""

    # Region-policy BC enforcement (from boundary_policy regions).
    region_bc_on: bool = True

    # Global BC relaxation rate (applies when per-field overrides are None).
    bc_enforce_nu: float = 0.0
    bc_enforce_nu_n: float | None = None
    bc_enforce_nu_omega: float | None = None
    bc_enforce_nu_vpar_e: float | None = None
    bc_enforce_nu_vpar_i: float | None = None
    bc_enforce_nu_Te: float | None = None
    bc_enforce_nu_Ti: float | None = None
    bc_enforce_nu_psi: float | None = None
    bc_enforce_nu_phi: float | None = None

    # BC overrides for 2D fields (None -> use geometry default).
    bc_n: BC2D | None = None
    bc_omega: BC2D | None = None
    bc_vpar_e: BC2D | None = None
    bc_vpar_i: BC2D | None = None
    bc_Te: BC2D | None = None
    bc_Ti: BC2D | None = None
    bc_psi: BC2D | None = None
    bc_phi: BC2D | None = None

    # Perpendicular BCs for 3D FCI planes.
    perp_bc: BC2D = eqx.field(default_factory=BC2D.periodic)
    perp_bc_nu: float = 0.0


class DRBSystemParams(eqx.Module):
    """Unified parameter set for the full DRB system.

    Groups parameters into physics, transport, closure, numerics, and BC bundles.
    Backwards-compatible attribute access is provided via __getattr__.
    """

    physics: PhysicsParams = eqx.field(default_factory=PhysicsParams)
    transport: TransportParams = eqx.field(default_factory=TransportParams)
    closure: ClosureParams = eqx.field(default_factory=ClosureParams)
    numerics: NumericsParams = eqx.field(default_factory=NumericsParams)
    bcs: BCParams = eqx.field(default_factory=BCParams)

    def __getattr__(self, name: str):
        for group in (self.physics, self.transport, self.closure, self.numerics, self.bcs):
            if hasattr(group, name):
                return getattr(group, name)
            if isinstance(group, eqx.Module):
                for field in getattr(group, "__dataclass_fields__", {}):
                    sub = getattr(group, field)
                    if isinstance(sub, eqx.Module) and hasattr(sub, name):
                        return getattr(sub, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name}")

    def update(self, **kwargs):
        """Return a new params object with updated attributes (supports nested groups)."""

        params = self
        for key, val in kwargs.items():
            if key in getattr(params, "__dataclass_fields__", {}):
                params = replace(params, **{key: val})
                continue
            updated = False
            for group_name in ("physics", "transport", "closure", "numerics", "bcs"):
                group = getattr(params, group_name)
                if hasattr(group, key):
                    new_group = replace(group, **{key: val})
                    params = replace(params, **{group_name: new_group})
                    updated = True
                    break
                if isinstance(group, eqx.Module):
                    for field in getattr(group, "__dataclass_fields__", {}):
                        sub = getattr(group, field)
                        if isinstance(sub, eqx.Module) and hasattr(sub, key):
                            new_sub = replace(sub, **{key: val})
                            new_group = replace(group, **{field: new_sub})
                            params = replace(params, **{group_name: new_group})
                            updated = True
                            break
                    if updated:
                        break
            if not updated:
                raise AttributeError(f"Unknown parameter: {key}")
        return params


def update_params_from_dict(params: DRBSystemParams, data: dict) -> DRBSystemParams:
    """Recursively update DRBSystemParams from a nested dictionary."""

    def _update_group(group, values: dict):
        out = group
        for k, v in values.items():
            if hasattr(out, k):
                attr = getattr(out, k)
                if isinstance(attr, eqx.Module) and isinstance(v, dict):
                    new_attr = _update_group(attr, v)
                    out = replace(out, **{k: new_attr})
                else:
                    out = replace(out, **{k: v})
            else:
                raise AttributeError(f"Unknown parameter: {k}")
        return out

    out = params
    for k, v in data.items():
        if k in getattr(out, "__dataclass_fields__", {}):
            attr = getattr(out, k)
            if isinstance(attr, eqx.Module) and isinstance(v, dict):
                out = replace(out, **{k: _update_group(attr, v)})
            else:
                out = replace(out, **{k: v})
            continue
        # Try nested groups.
        updated = False
        for group_name in ("physics", "transport", "closure", "numerics", "bcs"):
            group = getattr(out, group_name)
            if hasattr(group, k):
                new_group = _update_group(group, {k: v})
                out = replace(out, **{group_name: new_group})
                updated = True
                break
        if not updated:
            closure = getattr(out, "closure")
            for sub_name in ("sol", "sheath", "neutrals", "line_bcs"):
                sub_group = getattr(closure, sub_name)
                if hasattr(sub_group, k):
                    new_sub = _update_group(sub_group, {k: v})
                    new_closure = replace(closure, **{sub_name: new_sub})
                    out = replace(out, closure=new_closure)
                    updated = True
                    break
        if not updated:
            raise AttributeError(f"Unknown parameter: {k}")
    return out
