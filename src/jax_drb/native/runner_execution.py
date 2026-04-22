from __future__ import annotations

from typing import Any

from ..parity.reference import make_default_overrides, merge_overrides


def effective_overrides(parity_mode: str, *, reference_case: Any | None) -> tuple[str, ...]:
    case_overrides = reference_case.extra_overrides if reference_case is not None else ()
    return merge_overrides(make_default_overrides(parity_mode), case_overrides)


def effective_output_steps(parity_mode: str, *, configured_nout: int) -> int:
    if parity_mode == "one_rhs":
        return 0
    if parity_mode == "one_step":
        return 1
    return configured_nout


def restart_variable_names(run_config: Any) -> tuple[str, ...]:
    if _is_supported_diffusion_case(run_config):
        section = run_config.components[0].section
        return (f"N{section}", f"P{section}")
    if _is_supported_periodic_fluid_mms_case_placeholder(run_config):
        section = run_config.components[0].section
        return (f"N{section}", f"P{section}", f"NV{section}")
    if _is_supported_electrostatic_vorticity_case_placeholder(run_config):
        return ("Vort",)
    if _is_supported_blob2d_case_placeholder(run_config):
        return ("Ne", "Vort")
    if _is_supported_drift_wave_case_placeholder(run_config):
        return ("Ni", "NVe", "Vort")
    return ()


def _is_supported_diffusion_case(run_config: Any) -> bool:
    components = getattr(run_config, "components", ())
    if len(components) != 1:
        return False
    component = components[0]
    component_types = tuple(component.types)
    return component_types == ("evolve_density", "evolve_pressure", "anomalous_diffusion")


def _is_supported_periodic_fluid_mms_case_placeholder(run_config: Any) -> bool:
    components = getattr(run_config, "components", ())
    if len(components) != 1:
        return False
    component = components[0]
    component_types = tuple(component.types)
    return component_types == ("evolve_density", "evolve_pressure", "evolve_momentum")


def _is_supported_electrostatic_vorticity_case_placeholder(run_config: Any) -> bool:
    components = getattr(run_config, "components", ())
    return len(components) == 1 and tuple(components[0].types) == ("vorticity",)


def _is_supported_blob2d_case_placeholder(run_config: Any) -> bool:
    components = getattr(run_config, "components", ())
    return len(components) == 1 and tuple(components[0].types) == ("blob2d",)


def _is_supported_drift_wave_case_placeholder(run_config: Any) -> bool:
    components = getattr(run_config, "components", ())
    return len(components) == 1 and tuple(components[0].types) == ("drift_wave",)
