from __future__ import annotations

from dataclasses import dataclass, replace
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

    # Electromagnetism.
    beta: float = 0.0
    Dpsi: float = 0.0

    # Hot-ion parameters.
    tau_i: float = 1.0

    # Curvature drive.
    curvature_on: bool = False
    curvature_coeff: float = 0.0
    curvature_model: str = "slab"
    curvature_theta_scale: float | None = None
    curvature_scale: float | None = None

    # Polarization closure.
    n0: float = 1.0
    n0_min: float = 1e-6
    n0_max: float | None = None
    k2_min: float = 1e-12
    kperp2_min: float = 1e-6

    # Log-form state variables.
    log_n_clip: float | None = 50.0
    log_Te_clip: float | None = 50.0


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
    sol_sheath_phi_Te_floor: float = 1e-6
    sol_sheath_phi_clip: float = 10.0
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
    perp_operator: Literal["spectral", "fd", "fv"] = "spectral"
    poisson: Literal["spectral", "cg_fd", "mixed_fft"] = "spectral"
    poisson_force_spectral_when_periodic: bool = True
    poisson_preconditioner: str = "auto"
    poisson_cg_maxiter: int = 300
    poisson_cg_tol: float = 1e-8
    poisson_cg_atol: float = 0.0
    poisson_gauge_epsilon: float | None = None
    poisson_maxiter: int = 400
    poisson_tol: float = 1e-10
    dealias_on: bool = True
    bc_enforce_nu: float = 0.0

    # Non-Boussinesq variable-coefficient polarization solve settings.
    polarization_cg_maxiter: int = 400
    polarization_cg_tol: float = 1e-8
    polarization_cg_atol: float = 0.0
    polarization_preconditioner: str = "auto"
    polarization_precond_shift: float = 1e-12

    # FCI parallel-operator knobs.
    use_target_aware_dpar: bool = True
    target_scheme: str = "appendix_b"

    # Operator split toggles (conservative/source/dissipative).
    operator_split_on: bool = False
    operator_conservative_on: bool = True
    operator_source_on: bool = True
    operator_dissipative_on: bool = True

    # Optional term schedule override (list of term names).
    term_schedule: tuple[str, ...] | None = eqx.field(static=True, default=None)


class BCParams(eqx.Module):
    """Boundary condition overrides for fields."""

    # Region-policy BC enforcement (from boundary_policy regions).
    region_bc_on: bool = True

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
            raise AttributeError(f"Unknown parameter: {k}")
    return out
