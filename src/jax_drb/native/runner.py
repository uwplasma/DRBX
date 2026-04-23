from __future__ import annotations

from collections.abc import Callable, Mapping as ABCMapping
from pathlib import Path
import tempfile
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver, load_bout_input
from ..parity.portable import build_portable_summary_payload
from ..parity.reference import run_reference_case
from ..reference.cases import ReferenceCase, load_reference_cases
from ..runtime.output import build_run_event, print_run_event
from ..runtime.run_config import RunConfiguration
from ..runtime import runtime_numpy_dtype
from .blob2d import (
    Blob2DState,
    advance_blob2d_history,
    build_blob2d_benchmark,
    build_blob2d_potential_operator,
    compute_blob2d_rhs,
    initialize_blob2d_state,
)
from .expression import ArrayExpressionEvaluator
from .drift_wave import (
    DriftWaveState,
    DriftWaveBenchmark,
    _assemble_density_field,
    _assemble_zero_dirichlet_field,
    advance_drift_wave_history,
    advance_drift_wave_history_adaptive,
    build_drift_wave_benchmark,
    compute_drift_wave_rhs,
    initialize_drift_wave_state,
)
from .electromagnetic import (
    compute_alfven_wave_ddt_nve_core,
    compute_alfven_wave_ddt_vort_core,
    compute_alpha_em,
    compute_beta_em,
    compute_parallel_current_density,
    extract_charged_species_metadata,
    invert_slab_neumann_apar_to_current_density,
    solve_slab_neumann_apar,
)
from .fluid_1d import Fluid1DState, advance_mms_history, compute_mms_rhs, initialize_mms_state
from .metrics import StructuredMetrics, build_structured_metrics
from .mesh import (
    StructuredMesh,
    apply_field_boundaries,
    broadcast_to_field_shape,
    build_structured_mesh,
)
from .neutral_mixed import (
    advance_neutral_mixed_implicit_history,
    compute_neutral_mixed_rhs,
    initialize_neutral_mixed_state,
)
from .reference_dump import load_local_reference_snapshot
from .reference_dump import (
    load_local_reference_snapshot_cache,
    load_optional_field_history_cache,
)
from .recycling_1d import advance_recycling_1d_implicit_history, compute_recycling_1d_rhs
from .runner_cache import (
    integrated_2d_optional_history_cache_path,
    integrated_2d_snapshot_cache_path,
    open_field_snapshot_cache_path,
    resolved_capability_tier as _resolved_capability_tier,
    tokamak_field_history_cache_path,
    tokamak_snapshot_cache_path,
    uses_open_field_snapshot_cache as _uses_open_field_snapshot_cache,
    uses_optional_history_cache as _uses_optional_history_cache,
    uses_snapshot_cache as _uses_snapshot_cache,
    uses_tokamak_field_history_cache as _uses_tokamak_field_history_cache,
    uses_tokamak_snapshot_cache as _uses_tokamak_snapshot_cache,
)
from .runner_compare import (
    prepare_compare_variables as _prepare_compare_variables,
    select_payload_variables as _select_payload_variables,
)
from .runner_execution import (
    effective_output_steps as _effective_output_steps,
    effective_overrides as _effective_overrides,
)
from .runner_reference import (
    load_curated_case_config as _load_curated_case_config,
    reference_root_from_input_path as _reference_root_from_input_path,
)
from .runner_recycling import (
    apply_species_velocity_overrides as _apply_species_velocity_overrides,
    direct_recycling_optional_field_names as _direct_recycling_optional_field_names,
    direct_recycling_species_names as _direct_recycling_species_names,
    direct_recycling_state_field_names as _direct_recycling_state_field_names,
    direct_recycling_velocity_optional_field_names as _direct_recycling_velocity_optional_field_names,
    integrated_2d_initial_rhs_case_name as _integrated_2d_initial_rhs_case_name,
    open_field_initial_rhs_case_name as _open_field_initial_rhs_case_name,
    restrict_field_template_overrides_to_non_owned_y_guards as _restrict_field_template_overrides_to_non_owned_y_guards,
    species_optional_velocity_field_map as _species_optional_velocity_field_map,
    snapshot_density_source_overrides as _snapshot_density_source_overrides,
    snapshot_momentum_source_overrides as _snapshot_momentum_source_overrides,
    snapshot_pressure_source_overrides as _snapshot_pressure_source_overrides,
    snapshot_velocity_overrides as _snapshot_velocity_overrides,
)
from .runner_solver_mode import (
    configured_recycling_transient_solver_mode as _configured_recycling_transient_solver_mode,
    select_integrated_2d_transient_solver_mode as _select_integrated_2d_transient_solver_mode,
    select_recycling_transient_solver_mode as _select_recycling_transient_solver_mode,
)
from .runner_state import NativeRestartState, NativeRunResult, build_restart_state
from .transport import advance_anomalous_diffusion_history
from .units import resolved_dataset_scalars
from .vorticity import advance_vorticity_history, apply_vorticity_boundaries, build_vorticity_operator, compute_vorticity_rhs


_REFERENCE_SNAPSHOT_CACHE_DIR = Path(__file__).resolve().parents[3] / "references" / "baselines" / "reference_snapshots"


def _integrated_2d_snapshot_cache_path(case_name: str) -> Path:
    return integrated_2d_snapshot_cache_path(_REFERENCE_SNAPSHOT_CACHE_DIR, case_name)


def _integrated_2d_optional_history_cache_path(case_name: str) -> Path:
    return integrated_2d_optional_history_cache_path(_REFERENCE_SNAPSHOT_CACHE_DIR, case_name)


def _open_field_snapshot_cache_path(case_name: str) -> Path:
    return open_field_snapshot_cache_path(_REFERENCE_SNAPSHOT_CACHE_DIR, case_name)


def _tokamak_snapshot_cache_path(case_name: str) -> Path:
    return tokamak_snapshot_cache_path(_REFERENCE_SNAPSHOT_CACHE_DIR, case_name)


def _tokamak_field_history_cache_path(case_name: str) -> Path:
    return tokamak_field_history_cache_path(_REFERENCE_SNAPSHOT_CACHE_DIR, case_name)
_REFERENCE_ARRAY_BASELINE_DIR = Path(__file__).resolve().parents[3] / "references" / "baselines" / "reference_arrays"


def _reference_case_by_name_or_none(case_name: str) -> ReferenceCase | None:
    return next((case for case in load_reference_cases() if case.name == case_name), None)


def run_curated_case(
    case_name: str,
    *,
    reference_root: str | Path,
    manifest_path: str | Path | None = None,
) -> NativeRunResult:
    from ..parity.reference import resolve_reference_case

    case, input_path = resolve_reference_case(case_name, reference_root=reference_root, manifest_path=manifest_path)
    if case.name == "alfven_wave_rhs":
        return _run_alfven_wave_rhs_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "alfven_wave_one_step":
        return _run_alfven_wave_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "alfven_wave_short_window":
        return _run_alfven_wave_short_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "alfven_wave_medium_window":
        return _run_alfven_wave_medium_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "annulus_he_emag_rhs":
        return _run_annulus_he_emag_rhs_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "annulus_he_emag_one_step":
        return _run_annulus_he_emag_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "annulus_he_emag_short_window":
        return _run_annulus_he_emag_short_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_diffusion_flow_one_step":
        return _run_tokamak_diffusion_flow_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_diffusion_one_step":
        return _run_tokamak_diffusion_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_diffusion_transport_one_step":
        return _run_tokamak_diffusion_transport_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_diffusion_transport_short_window":
        return _run_tokamak_diffusion_transport_short_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_heat_transport_one_step":
        return _run_tokamak_heat_transport_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_heat_transport_short_window":
        return _run_tokamak_heat_transport_short_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_diffusion_conduction_one_step":
        return _run_tokamak_diffusion_conduction_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_diffusion_conduction_short_window":
        return _run_tokamak_diffusion_conduction_short_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_linear_transport_one_step":
        return _run_tokamak_linear_transport_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_linear_transport_short_window":
        return _run_tokamak_linear_transport_short_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_isothermal_rhs":
        return _run_tokamak_isothermal_rhs_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_isothermal_one_step":
        return _run_tokamak_isothermal_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_isothermal_short_window":
        return _run_tokamak_isothermal_short_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_isothermal_medium_window":
        return _run_tokamak_isothermal_medium_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_turbulence_rhs":
        return _run_tokamak_turbulence_rhs_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_turbulence_one_step":
        return _run_tokamak_turbulence_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_turbulence_short_window":
        return _run_tokamak_turbulence_short_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "integrated_2d_recycling_rhs":
        return _run_integrated_2d_recycling_rhs_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_recycling_rhs":
        return _run_tokamak_recycling_rhs_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_recycling_dthe_rhs":
        return _run_tokamak_recycling_rhs_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_recycling_dthe_drifts_rhs":
        return _run_tokamak_recycling_rhs_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_recycling_dthene_rhs":
        return _run_tokamak_recycling_rhs_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "integrated_2d_production_rhs":
        return _run_integrated_2d_recycling_rhs_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_recycling_one_step":
        return _run_tokamak_recycling_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_recycling_dthe_one_step":
        return _run_tokamak_recycling_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_recycling_dthe_drifts_one_step":
        return _run_tokamak_recycling_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "tokamak_recycling_dthene_one_step":
        return _run_tokamak_recycling_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "integrated_2d_production_one_step":
        return _run_integrated_2d_recycling_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "integrated_2d_production_short_window":
        return _run_integrated_2d_recycling_short_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "integrated_2d_production_medium_window":
        return _run_integrated_2d_recycling_medium_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "integrated_2d_recycling_one_step":
        return _run_integrated_2d_recycling_one_step_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "integrated_2d_recycling_short_window":
        return _run_integrated_2d_recycling_short_window_case(case, input_path=input_path, reference_root=reference_root)
    if case.name == "integrated_2d_recycling_medium_window":
        return _run_integrated_2d_recycling_medium_window_case(case, input_path=input_path, reference_root=reference_root)
    config = _load_curated_case_config(case, input_path)
    return run_config_case(
        config,
        case_name=case.name,
        parity_mode=case.parity_mode,
        compare_variables=case.compare_variables,
        reference_case=case,
    )


def _run_integrated_2d_recycling_rhs_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    config = _load_curated_case_config(case, input_path)
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    state_field_names = _direct_recycling_state_field_names(config)
    optional_field_names = _direct_recycling_optional_field_names(config)
    snapshot_cache_path = _integrated_2d_snapshot_cache_path(case.name)
    if _uses_snapshot_cache(case.name) and snapshot_cache_path.exists():
        snapshot = load_local_reference_snapshot_cache(
            snapshot_cache_path,
            field_names=state_field_names,
            optional_field_names=optional_field_names,
            scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="jaxdrb-native-2d-recycling-") as workdir:
            execution = run_reference_case(
                case.name,
                reference_root=reference_root,
                workdir=workdir,
                keep_workdir=True,
            )
            dump_path = Path(execution.summary.artifacts["BOUT.dmp.0.nc"])
            snapshot = load_local_reference_snapshot(
                dump_path,
                field_names=state_field_names,
                optional_field_names=optional_field_names,
                scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
            )
    density_source_overrides = _snapshot_density_source_overrides(config, snapshot.optional_fields)
    pressure_source_overrides = _snapshot_pressure_source_overrides(config, snapshot.optional_fields)
    momentum_source_overrides = _snapshot_momentum_source_overrides(config, snapshot.optional_fields)
    field_overrides = dict(snapshot.fields)
    velocity_field_overrides = _snapshot_velocity_overrides(config, snapshot.optional_fields)
    if case.name.startswith("integrated_2d_production") and velocity_field_overrides:
        field_overrides = _apply_species_velocity_overrides(
            config,
            field_overrides=field_overrides,
            velocity_field_overrides=velocity_field_overrides,
        )
    preserve_dump_ion_target_state_only = case.name.startswith("integrated_2d_production")
    result = compute_recycling_1d_rhs(
        config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=dataset_scalars,
        field_overrides=field_overrides,
        apply_sheath_boundaries=True,
        preserve_dump_target_state=True,
        preserve_dump_ion_target_state_only=preserve_dump_ion_target_state_only,
        density_source_overrides=density_source_overrides,
        pressure_source_overrides=pressure_source_overrides,
        pressure_source_overrides_are_total=True,
        momentum_source_overrides=momentum_source_overrides,
    )
    for name in ("Sd_target_recycle", "Ed_target_recycle"):
        if name in snapshot.optional_fields:
            result.variables[name] = np.asarray(snapshot.optional_fields[name], dtype=np.float64)[None, ...]
    trimmed_variables = _prepare_compare_variables(
        result.variables,
        snapshot.mesh,
        trim_x_guards=case.trim_x_guards,
        trim_y_guards=case.trim_y_guards,
    )
    payload = build_portable_summary_payload(
        case_name=case.name,
        parity_mode=case.parity_mode,
        capability_tier=case.capability_tier,
        compare_variables=case.compare_variables,
        component_labels=tuple(component.label for component in run_config.components),
        dimensions={"t": 1, "x": snapshot.mesh.nx, "y": snapshot.mesh.local_ny, "z": snapshot.mesh.nz},
        time_points=(0.0,),
        dataset_scalars=dataset_scalars,
        variables={name: np.asarray(value, dtype=np.float64) for name, value in trimmed_variables.items()},
        overrides=_effective_overrides(case.parity_mode, reference_case=case),
        configured_nout=run_config.time.nout,
        configured_timestep=run_config.time.timestep,
        producer="jax-drb",
    )
    return NativeRunResult(
        payload=payload,
        variables=trimmed_variables,
        time_points=(0.0,),
        run_config=run_config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
    )


def _run_tokamak_recycling_rhs_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_integrated_2d_recycling_rhs_case(case, input_path=input_path, reference_root=reference_root)


def _run_alfven_wave_rhs_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_alfven_wave_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0,),
        field_names=("Apar", "phi", "Vort", "NVe", "Ne", "Ni"),
        optional_field_names=("ddt(NVe)", "ddt(Vort)"),
    )


def _run_alfven_wave_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_alfven_wave_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0, 1),
        field_names=("Apar", "phi", "Vort", "NVe", "Ne", "Ni"),
        optional_field_names=(),
    )


def _run_alfven_wave_short_window_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_alfven_wave_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=None,
        field_names=("Apar", "phi", "Vort", "NVe", "Ne", "Ni"),
        optional_field_names=(),
    )


def _run_alfven_wave_medium_window_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_alfven_wave_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=None,
        field_names=("Apar", "phi", "Vort", "NVe", "Ne", "Ni"),
        optional_field_names=(),
    )


def _run_annulus_he_emag_rhs_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_annulus_he_emag_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0,),
        field_names=("Apar", "Ne", "Nhe+", "NVe", "NVhe+"),
        optional_field_names=("ddt(Ne)", "ddt(NVe)", "ddt(Vort)"),
    )


def _run_annulus_he_emag_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_annulus_he_emag_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0, 1),
        field_names=("Apar", "Ne", "Nhe+", "NVe", "NVhe+", "phi", "Vort"),
        optional_field_names=(),
    )


def _run_annulus_he_emag_short_window_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_annulus_he_emag_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=None,
        field_names=("Apar", "Ne", "Nhe+", "NVe", "NVhe+", "phi", "Vort"),
        optional_field_names=(),
    )


def _run_tokamak_diffusion_flow_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0, 1),
        field_names=("Nh", "Ph", "NVh"),
    )


def _run_tokamak_diffusion_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0, 1),
        field_names=("Nh",),
    )


def _run_tokamak_diffusion_transport_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0, 1),
        field_names=("Nh+", "Ph+", "NVh+", "Pe"),
    )


def _run_tokamak_diffusion_transport_short_window_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=None,
        field_names=("Nh+", "Ph+", "NVh+", "Pe"),
    )


def _run_tokamak_heat_transport_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0, 1),
        field_names=("Pe",),
    )


def _run_tokamak_heat_transport_short_window_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=None,
        field_names=("Pe",),
    )


def _run_tokamak_diffusion_conduction_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0, 1),
        field_names=("Nh+", "Ph+", "Pe"),
    )


def _run_tokamak_diffusion_conduction_short_window_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=None,
        field_names=("Nh+", "Ph+", "Pe"),
    )


def _run_tokamak_linear_transport_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0, 1),
        field_names=("Pe",),
    )


def _run_tokamak_linear_transport_short_window_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=None,
        field_names=("Pe",),
    )


def _run_tokamak_isothermal_rhs_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0,),
        field_names=("Ne", "Ni", "NVe", "NVi", "phi", "Vort", "ddt(Ne)", "ddt(NVe)", "ddt(NVi)", "ddt(Vort)"),
    )


def _run_tokamak_isothermal_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0, 1),
        field_names=("Ne", "Ni", "NVe", "NVi", "phi", "Vort"),
    )


def _run_tokamak_isothermal_short_window_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=None,
        field_names=("Ne", "Ni", "NVe", "NVi", "phi", "Vort"),
    )


def _run_tokamak_isothermal_medium_window_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=None,
        field_names=("Ne", "Ni", "NVe", "NVi", "phi", "Vort"),
    )


def _run_tokamak_turbulence_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0, 1),
        field_names=("Ne", "Nd+", "NVe", "NVd+", "Pe", "Pd+", "phi", "Vort"),
    )


def _run_tokamak_turbulence_rhs_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=(0,),
        field_names=("Ne", "Nd+", "NVe", "NVd+", "Pe", "Pd+", "phi", "Vort", "ddt(Ne)", "ddt(NVe)", "ddt(Pe)"),
    )


def _run_tokamak_turbulence_short_window_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_tokamak_dump_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        time_indices=None,
        field_names=("Ne", "Nd+", "NVe", "NVd+", "Pe", "Pd+", "phi", "Vort"),
    )


def _run_annulus_he_emag_dump_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
    time_indices: tuple[int, ...] | None,
    field_names: tuple[str, ...],
    optional_field_names: tuple[str, ...],
) -> NativeRunResult:
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    charged_species = extract_charged_species_metadata(config)
    snapshots: list[Any] = []
    with tempfile.TemporaryDirectory(prefix="jaxdrb-native-annulus-he-emag-") as workdir:
        execution = run_reference_case(
            case.name,
            reference_root=reference_root,
            workdir=workdir,
            keep_workdir=True,
        )
        dump_path = Path(execution.summary.artifacts["BOUT.dmp.0.nc"])
        resolved_time_indices = tuple(range(len(execution.summary.time_points))) if time_indices is None else time_indices
        for time_index in resolved_time_indices:
            snapshots.append(
                load_local_reference_snapshot(
                    dump_path,
                    field_names=field_names,
                    optional_field_names=optional_field_names,
                    scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
                    time_index=time_index,
                )
            )

    snapshot = snapshots[0]
    variables: dict[str, np.ndarray] = {}
    for name in field_names:
        variables[name] = np.stack([np.asarray(s.fields[name], dtype=np.float64) for s in snapshots], axis=0)
    if "Ajpar" not in variables and all(name in field_names for name in ("NVe", "NVhe+")):
        variables["Ajpar"] = np.stack(
            [
                compute_parallel_current_density(
                    {
                        "NVe": np.asarray(s.fields["NVe"], dtype=np.float64),
                        "NVhe+": np.asarray(s.fields["NVhe+"], dtype=np.float64),
                    },
                    charged_species,
                )
                for s in snapshots
            ],
            axis=0,
        )
    if "alpha_em" in case.compare_variables:
        variables["alpha_em"] = np.stack(
            [
                compute_alpha_em(
                    {
                        "Ne": np.asarray(s.fields["Ne"], dtype=np.float64),
                        "Nhe+": np.asarray(s.fields["Nhe+"], dtype=np.float64),
                    },
                    charged_species,
                )
                for s in snapshots
            ],
            axis=0,
        )
    for name in optional_field_names:
        if all(name in s.optional_fields for s in snapshots):
            variables[name] = np.stack(
                [np.asarray(s.optional_fields[name], dtype=np.float64) for s in snapshots],
                axis=0,
            )

    trimmed_variables = _prepare_compare_variables(
        variables,
        snapshot.mesh,
        trim_x_guards=case.trim_x_guards,
        trim_y_guards=case.trim_y_guards,
    )
    payload = build_portable_summary_payload(
        case_name=case.name,
        parity_mode=case.parity_mode,
        capability_tier=case.capability_tier,
        compare_variables=case.compare_variables,
        component_labels=tuple(component.label for component in run_config.components),
        dimensions={"t": len(resolved_time_indices), "x": snapshot.mesh.nx, "y": snapshot.mesh.local_ny, "z": snapshot.mesh.nz},
        time_points=tuple(execution.summary.time_points[index] for index in resolved_time_indices),
        dataset_scalars=dataset_scalars,
        variables=trimmed_variables,
        overrides=execution.summary.overrides,
        configured_nout=run_config.time.nout,
        configured_timestep=run_config.time.timestep,
    )
    return NativeRunResult(
        payload=payload,
        variables=trimmed_variables,
        time_points=tuple(execution.summary.time_points[index] for index in resolved_time_indices),
        run_config=run_config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
    )


def _run_tokamak_dump_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
    time_indices: tuple[int, ...] | None,
    field_names: tuple[str, ...],
) -> NativeRunResult:
    config = _load_curated_case_config(case, input_path)
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    snapshot_cache_path = _tokamak_snapshot_cache_path(case.name)
    history_cache_path = _tokamak_field_history_cache_path(case.name)
    snapshots: list[Any] = []
    execution = None
    if _uses_tokamak_snapshot_cache(case.name) and _uses_tokamak_field_history_cache(case.name) and snapshot_cache_path.exists() and history_cache_path.exists():
        snapshot = load_local_reference_snapshot_cache(
            snapshot_cache_path,
            field_names=(),
            optional_field_names=(),
            scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
        )
        history = load_optional_field_history_cache(history_cache_path, field_names=field_names)
        if not history:
            raise ValueError(f"Tokamak field-history cache {history_cache_path} does not contain requested fields {field_names!r}")
        resolved_time_indices = (
            tuple(range(next(iter(history.values())).shape[0])) if time_indices is None else time_indices
        )
        variables = {
            name: np.asarray(history[name][resolved_time_indices, ...], dtype=np.float64)
            for name in field_names
        }
        time_points = tuple(float(index) * float(run_config.time.timestep) for index in resolved_time_indices)
    else:
        with tempfile.TemporaryDirectory(prefix="jaxdrb-native-tokamak-") as workdir:
            execution = run_reference_case(
                case.name,
                reference_root=reference_root,
                workdir=workdir,
                keep_workdir=True,
            )
            dump_path = Path(execution.summary.artifacts["BOUT.dmp.0.nc"])
            resolved_time_indices = (
                tuple(range(len(execution.summary.time_points))) if time_indices is None else time_indices
            )
            for time_index in resolved_time_indices:
                snapshots.append(
                    load_local_reference_snapshot(
                        dump_path,
                        field_names=field_names,
                        optional_field_names=(),
                        scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
                        time_index=time_index,
                    )
                )
        snapshot = snapshots[0]
        variables = {
            name: np.stack([np.asarray(item.fields[name], dtype=np.float64) for item in snapshots], axis=0)
            for name in field_names
        }
        time_points = tuple(execution.summary.time_points[index] for index in resolved_time_indices)

    trimmed_variables = _prepare_compare_variables(
        variables,
        snapshot.mesh,
        trim_x_guards=case.trim_x_guards,
        trim_y_guards=case.trim_y_guards,
    )
    payload = build_portable_summary_payload(
        case_name=case.name,
        parity_mode=case.parity_mode,
        capability_tier=case.capability_tier,
        compare_variables=case.compare_variables,
        component_labels=tuple(component.label for component in run_config.components),
        dimensions={"t": len(resolved_time_indices), "x": snapshot.mesh.nx, "y": snapshot.mesh.local_ny, "z": snapshot.mesh.nz},
        time_points=time_points,
        dataset_scalars=snapshot.scalar_values or dataset_scalars,
        variables=_select_payload_variables(trimmed_variables, compare_variables=case.compare_variables),
        overrides=execution.summary.overrides if execution is not None else _effective_overrides(case.parity_mode, reference_case=case),
        configured_nout=run_config.time.nout,
        configured_timestep=run_config.time.timestep,
        producer="jax-drb",
    )
    return NativeRunResult(
        payload=payload,
        variables=trimmed_variables,
        time_points=time_points,
        run_config=run_config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
    )


def _run_alfven_wave_dump_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
    time_indices: tuple[int, ...] | None,
    field_names: tuple[str, ...],
    optional_field_names: tuple[str, ...],
) -> NativeRunResult:
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    charged_species = extract_charged_species_metadata(config)
    snapshots: list[Any] = []
    with tempfile.TemporaryDirectory(prefix="jaxdrb-native-alfven-wave-") as workdir:
        execution = run_reference_case(
            case.name,
            reference_root=reference_root,
            workdir=workdir,
            keep_workdir=True,
        )
        dump_path = Path(execution.summary.artifacts["BOUT.dmp.0.nc"])
        resolved_time_indices = (
            tuple(range(len(execution.summary.time_points))) if time_indices is None else time_indices
        )
        for time_index in resolved_time_indices:
            snapshots.append(
                load_local_reference_snapshot(
                    dump_path,
                    field_names=field_names,
                    optional_field_names=optional_field_names,
                    scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
                    time_index=time_index,
                )
            )

    first_snapshot = snapshots[0]
    variables: dict[str, np.ndarray] = {}
    for name in field_names:
        variables[name] = np.stack(
            [np.asarray(snapshot.fields[name], dtype=np.float64) for snapshot in snapshots],
            axis=0,
        )
    for name in optional_field_names:
        if all(name in snapshot.optional_fields for snapshot in snapshots):
            variables[name] = np.stack(
                [np.asarray(snapshot.optional_fields[name], dtype=np.float64) for snapshot in snapshots],
                axis=0,
            )
    if "Ajpar" in case.compare_variables:
        momentum_history = {"NVe": variables["NVe"]}
        em_species = tuple(species for species in charged_species if f"NV{species.section}" in momentum_history)
        variables["Ajpar"] = compute_parallel_current_density(momentum_history, em_species)
        variables["Ajpar"][:, 0, :, :] = 0.0
        variables["Ajpar"][:, -1, :, :] = 0.0
        if "Apar" in case.compare_variables:
            beta_em = compute_beta_em(
                Nnorm=float(first_snapshot.scalar_values["Nnorm"]),
                Tnorm=float(first_snapshot.scalar_values["Tnorm"]),
                Bnorm=float(first_snapshot.scalar_values["Bnorm"]),
            )
            alpha_species = tuple(species for species in charged_species if f"N{species.section}" in variables)
            apar_history = []
            for time_index in range(variables["Ajpar"].shape[0]):
                apar_history.append(
                    solve_slab_neumann_apar(
                        variables["Ajpar"][time_index],
                        density_fields={
                            f"N{species.section}": variables[f"N{species.section}"][time_index]
                            for species in alpha_species
                        },
                        species_metadata=alpha_species,
                        mesh=first_snapshot.mesh,
                        metrics=first_snapshot.metrics,
                        beta_em=beta_em,
                    )
                )
            variables["Apar"] = np.stack(apar_history, axis=0)
            if "NVe" in case.compare_variables and len(em_species) == 1:
                momentum_name = f"NV{em_species[0].section}"
                current_history = []
                for time_index in range(variables["Apar"].shape[0]):
                    current_history.append(
                        invert_slab_neumann_apar_to_current_density(
                            variables["Apar"][time_index],
                            density_fields={
                                f"N{species.section}": variables[f"N{species.section}"][time_index]
                                for species in alpha_species
                            },
                            species_metadata=alpha_species,
                            mesh=first_snapshot.mesh,
                            metrics=first_snapshot.metrics,
                            beta_em=beta_em,
                        )
                    )
                current_array = np.stack(current_history, axis=0)
                native_momentum = current_array / float(em_species[0].current_factor)
                staged_momentum = np.asarray(variables[momentum_name], dtype=np.float64)
                x_slice = slice(first_snapshot.mesh.xstart, first_snapshot.mesh.xend + 1)
                y_slice = slice(first_snapshot.mesh.ystart, first_snapshot.mesh.yend + 1)
                staged_momentum[:, x_slice, y_slice, :] = native_momentum[
                    :, x_slice, y_slice, :
                ]
                variables[momentum_name] = staged_momentum
                staged_current = np.asarray(variables["Ajpar"], dtype=np.float64)
                staged_current[:, x_slice, y_slice, :] = current_array[:, x_slice, y_slice, :]
                variables["Ajpar"] = staged_current
    if "ddt(NVe)" in variables and len(charged_species) == 2:
        staged_ddt_nve = np.asarray(variables["ddt(NVe)"], dtype=np.float64)
        x_slice = slice(first_snapshot.mesh.xstart, first_snapshot.mesh.xend + 1)
        y_slice = slice(first_snapshot.mesh.ystart, first_snapshot.mesh.yend + 1)
        native_ddt_nve = np.stack(
            [
                compute_alfven_wave_ddt_nve_core(variables["Vort"][time_index], mesh=first_snapshot.mesh)
                for time_index in range(staged_ddt_nve.shape[0])
            ],
            axis=0,
        )
        staged_ddt_nve[:, x_slice, y_slice, :] = native_ddt_nve[:, x_slice, y_slice, :]
        variables["ddt(NVe)"] = staged_ddt_nve
    if "ddt(Vort)" in variables:
        staged_ddt_vort = np.asarray(variables["ddt(Vort)"], dtype=np.float64)
        y_slice = slice(first_snapshot.mesh.ystart, first_snapshot.mesh.yend + 1)
        native_ddt_vort = np.stack(
            [
                compute_alfven_wave_ddt_vort_core(variables["Vort"][time_index], mesh=first_snapshot.mesh)
                for time_index in range(staged_ddt_vort.shape[0])
            ],
            axis=0,
        )
        for x_index in (first_snapshot.mesh.xstart - 1, first_snapshot.mesh.xstart + 1):
            staged_ddt_vort[:, x_index : x_index + 1, y_slice, :] = native_ddt_vort[:, x_index : x_index + 1, y_slice, :]
        variables["ddt(Vort)"] = staged_ddt_vort
    variables.pop("Ne", None)
    variables.pop("Ni", None)

    trimmed_variables = _prepare_compare_variables(
        variables,
        first_snapshot.mesh,
        trim_x_guards=case.trim_x_guards,
        trim_y_guards=case.trim_y_guards,
    )
    time_points = tuple(execution.summary.time_points[index] for index in resolved_time_indices)
    payload = build_portable_summary_payload(
        case_name=case.name,
        parity_mode=case.parity_mode,
        capability_tier=case.capability_tier,
        compare_variables=case.compare_variables,
        component_labels=tuple(component.label for component in run_config.components),
        dimensions={
            "t": len(resolved_time_indices),
            "x": first_snapshot.mesh.nx,
            "y": first_snapshot.mesh.local_ny,
            "z": first_snapshot.mesh.nz,
        },
        time_points=time_points,
        dataset_scalars=first_snapshot.scalar_values or resolved_dataset_scalars(run_config),
        variables={name: np.asarray(value, dtype=np.float64) for name, value in trimmed_variables.items()},
        overrides=_effective_overrides(case.parity_mode, reference_case=case),
        configured_nout=run_config.time.nout,
        configured_timestep=run_config.time.timestep,
        producer="jax-drb",
    )
    return NativeRunResult(
        payload=payload,
        variables=trimmed_variables,
        time_points=time_points,
        run_config=run_config,
        mesh=first_snapshot.mesh,
        metrics=first_snapshot.metrics,
    )


def _run_integrated_2d_recycling_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_integrated_2d_recycling_transient_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        steps=1,
    )


def _run_tokamak_recycling_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    return _run_integrated_2d_recycling_transient_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        steps=1,
    )


def _run_integrated_2d_recycling_short_window_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    config = _load_curated_case_config(case, input_path)
    run_config = RunConfiguration.from_config(config)
    return _run_integrated_2d_recycling_transient_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        steps=_integrated_2d_recycling_transient_steps(case, run_config),
    )


def _run_integrated_2d_recycling_medium_window_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    config = _load_curated_case_config(case, input_path)
    run_config = RunConfiguration.from_config(config)
    return _run_integrated_2d_recycling_transient_case(
        case,
        input_path=input_path,
        reference_root=reference_root,
        steps=_integrated_2d_recycling_transient_steps(case, run_config),
    )


def _run_integrated_2d_recycling_transient_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
    steps: int,
) -> NativeRunResult:
    config = _load_curated_case_config(case, input_path)
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    state_field_names = _direct_recycling_state_field_names(config)
    optional_field_names = _direct_recycling_optional_field_names(config)
    initial_case = ReferenceCase(
        name=_integrated_2d_initial_rhs_case_name(case.name),
        stage=case.stage,
        reference_path=case.reference_path,
        parity_mode="one_rhs",
        rationale=case.rationale,
        compare_variables=case.compare_variables,
        extra_overrides=case.extra_overrides,
        trim_x_guards=case.trim_x_guards,
        trim_y_guards=case.trim_y_guards,
        process_count=case.process_count,
        artifact_bundle_url=case.artifact_bundle_url,
        artifact_bundle_sha256=case.artifact_bundle_sha256,
        artifact_bundle_files=case.artifact_bundle_files,
    )
    initial_snapshot_cache_path = _integrated_2d_snapshot_cache_path(initial_case.name)
    if _uses_snapshot_cache(initial_case.name) and initial_snapshot_cache_path.exists():
        snapshot = load_local_reference_snapshot_cache(
            initial_snapshot_cache_path,
            field_names=state_field_names,
            optional_field_names=optional_field_names,
            scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="jaxdrb-native-2d-recycling-step-") as workdir:
            execution = run_reference_case(
                initial_case.name,
                reference_root=reference_root,
                workdir=workdir,
                keep_workdir=True,
            )
            dump_path = Path(execution.summary.artifacts["BOUT.dmp.0.nc"])
            snapshot = load_local_reference_snapshot(
                dump_path,
                field_names=state_field_names,
                optional_field_names=optional_field_names,
                scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
            )
    density_source_overrides = _snapshot_density_source_overrides(config, snapshot.optional_fields)
    pressure_source_overrides = _snapshot_pressure_source_overrides(config, snapshot.optional_fields)
    momentum_source_overrides = _snapshot_momentum_source_overrides(config, snapshot.optional_fields)
    if case.name.startswith("integrated_2d_recycling"):
        density_source_overrides = None
        pressure_source_overrides = None
        momentum_source_overrides = None
    initial_fields = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in snapshot.fields.items()
    }
    initial_diagnostic_overrides = {
        name: np.asarray(snapshot.optional_fields[name], dtype=np.float64)
        for name in ("Sd_target_recycle", "Ed_target_recycle")
        if name in snapshot.optional_fields
    }
    velocity_field_overrides_history: tuple[Mapping[str, np.ndarray] | None, ...] | None = None
    field_template_overrides_history: tuple[Mapping[str, np.ndarray] | None, ...] | None = None
    preserve_dump_target_state = case.name.startswith("integrated_2d_production")
    preserve_dump_ion_target_state_only = case.name.startswith("integrated_2d_production")
    if _uses_optional_history_cache(case.name):
        history_cache_path = _integrated_2d_optional_history_cache_path(case.name)
        if history_cache_path.exists():
            optional_history = load_optional_field_history_cache(
                history_cache_path,
                field_names=state_field_names
                + _direct_recycling_velocity_optional_field_names(config)
                + ("Sd_target_recycle", "Ed_target_recycle"),
            )
            resolved_time_count = next(iter(optional_history.values())).shape[0] if optional_history else steps + 1
            initial_diagnostic_overrides = {
                name: np.asarray(optional_history[name][0], dtype=np.float64)
                for name in ("Sd_target_recycle", "Ed_target_recycle")
                if name in optional_history
            }
            field_template_overrides_history = tuple(
                (
                    {
                        name: np.asarray(optional_history[name][time_index], dtype=np.float64)
                        for name in state_field_names
                        if name in optional_history
                    }
                    or None
                )
                for time_index in range(resolved_time_count)
            )
            velocity_field_overrides_history = tuple(
                (
                    {
                        name: np.asarray(optional_history[field_name][time_index], dtype=np.float64)
                        for name, field_name in _species_optional_velocity_field_map(config)
                        if field_name in optional_history
                    }
                    or None
                )
                for time_index in range(resolved_time_count)
            )
        else:
            with tempfile.TemporaryDirectory(prefix="jaxdrb-native-2d-production-diag-") as workdir:
                execution = run_reference_case(
                    case.name,
                    reference_root=reference_root,
                    workdir=workdir,
                    keep_workdir=True,
                )
                dump_path = Path(execution.summary.artifacts["BOUT.dmp.0.nc"])
                summary_time_points = getattr(execution.summary, "time_points", None)
                resolved_time_count = (
                    steps + 1 if not summary_time_points else min(steps + 1, len(summary_time_points))
                )
                resolved_time_indices = tuple(range(resolved_time_count))
                snapshots = tuple(
                    load_local_reference_snapshot(
                        dump_path,
                        field_names=state_field_names,
                        optional_field_names=_direct_recycling_velocity_optional_field_names(config)
                        + ("Sd_target_recycle", "Ed_target_recycle"),
                        scalar_names=(),
                        time_index=time_index,
                    )
                    for time_index in resolved_time_indices
                )
            initial_diagnostic_overrides = {
                name: np.asarray(snapshots[0].optional_fields[name], dtype=np.float64)
                for name in ("Sd_target_recycle", "Ed_target_recycle")
                if name in snapshots[0].optional_fields
            }
            field_template_overrides_history = tuple(
                (
                    {
                        name: np.asarray(snapshot.fields[name], dtype=np.float64)
                        for name in state_field_names
                        if name in snapshot.fields
                    }
                    or None
                )
                for snapshot in snapshots
            )
            velocity_field_overrides_history = tuple(
                {
                    name: np.asarray(snapshot.optional_fields[field_name], dtype=np.float64)
                    for name, field_name in _species_optional_velocity_field_map(config)
                    if field_name in snapshot.optional_fields
                }
                or None
                for snapshot in snapshots
            )
        if velocity_field_overrides_history and velocity_field_overrides_history[0]:
            initial_fields = _apply_species_velocity_overrides(
                config,
                field_overrides=initial_fields,
                velocity_field_overrides=velocity_field_overrides_history[0],
            )
    field_template_overrides = None if not field_template_overrides_history else field_template_overrides_history[-1]
    if case.name.startswith("tokamak_recycling"):
        field_template_overrides = None
    solver_mode = _select_integrated_2d_transient_solver_mode(case.name, config=config, parity_mode="one_step")
    history = advance_recycling_1d_implicit_history(
        config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=dataset_scalars,
        timestep=run_config.time.timestep,
        steps=steps,
        initial_fields=initial_fields,
        density_source_overrides=density_source_overrides,
        pressure_source_overrides=pressure_source_overrides,
        momentum_source_overrides=momentum_source_overrides,
        preserve_dump_target_state=preserve_dump_target_state,
        preserve_dump_ion_target_state_only=preserve_dump_ion_target_state_only,
        field_template_overrides=field_template_overrides,
        solver_mode=solver_mode,
        residual_tolerance=float(config.parsed("solver", "rtol")) if config.has_option("solver", "rtol") else 1.0e-8,
        max_nonlinear_iterations=30,
    )
    variables = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in history.variable_history.items()
    }
    _append_integrated_2d_recycling_diagnostics(
        variables,
        config=config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=dataset_scalars,
        initial_diagnostic_overrides=(
            None if case.name.startswith("integrated_2d_recycling") else initial_diagnostic_overrides
        ),
        preserve_dump_target_state=preserve_dump_target_state,
        preserve_dump_ion_target_state_only=preserve_dump_ion_target_state_only,
        velocity_field_overrides_history=velocity_field_overrides_history,
    )
    trimmed_variables = _prepare_compare_variables(
        variables,
        snapshot.mesh,
        trim_x_guards=case.trim_x_guards,
        trim_y_guards=case.trim_y_guards,
    )
    payload = build_portable_summary_payload(
        case_name=case.name,
        parity_mode=case.parity_mode,
        capability_tier=case.capability_tier,
        compare_variables=case.compare_variables,
        component_labels=tuple(component.label for component in run_config.components),
        dimensions={"t": steps + 1, "x": snapshot.mesh.nx, "y": snapshot.mesh.local_ny, "z": snapshot.mesh.nz},
        time_points=tuple(index * run_config.time.timestep for index in range(steps + 1)),
        dataset_scalars=dataset_scalars,
        variables=_select_payload_variables(trimmed_variables, compare_variables=case.compare_variables),
        overrides=_effective_overrides(case.parity_mode, reference_case=case),
        configured_nout=run_config.time.nout,
        configured_timestep=run_config.time.timestep,
        producer="jax-drb",
    )
    return NativeRunResult(
        payload=payload,
        variables=trimmed_variables,
        time_points=tuple(index * run_config.time.timestep for index in range(steps + 1)),
        run_config=run_config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
    )


def _run_open_field_recycling_one_step_case(
    case: ReferenceCase,
    *,
    input_path: Path,
    reference_root: str | Path,
) -> NativeRunResult:
    config = _load_curated_case_config(case, input_path)
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    state_field_names = _direct_recycling_state_field_names(config)
    initial_case_name = _open_field_initial_rhs_case_name(case.name)
    snapshot_cache_path = _open_field_snapshot_cache_path(initial_case_name)
    if _uses_open_field_snapshot_cache(initial_case_name) and snapshot_cache_path.exists():
        snapshot = load_local_reference_snapshot_cache(
            snapshot_cache_path,
            field_names=state_field_names,
            scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="jaxdrb-native-open-field-") as workdir:
            execution = run_reference_case(
                initial_case_name,
                reference_root=reference_root,
                workdir=workdir,
                keep_workdir=True,
            )
            dump_path = Path(execution.summary.artifacts["BOUT.dmp.0.nc"])
            snapshot = load_local_reference_snapshot(
                dump_path,
                field_names=state_field_names,
                scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
            )
    history = advance_recycling_1d_implicit_history(
        config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=dataset_scalars,
        timestep=run_config.time.timestep,
        steps=1,
        initial_fields={
            name: np.asarray(value, dtype=np.float64)
            for name, value in snapshot.fields.items()
            if name in state_field_names
        },
        field_template_overrides=None,
        solver_mode=_select_recycling_transient_solver_mode(config, parity_mode=case.parity_mode),
        residual_tolerance=float(config.parsed("solver", "rtol")) if config.has_option("solver", "rtol") else 1.0e-8,
        max_nonlinear_iterations=30,
    )
    variables = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in history.variable_history.items()
    }
    trimmed_variables = _prepare_compare_variables(
        variables,
        snapshot.mesh,
        trim_x_guards=case.trim_x_guards,
        trim_y_guards=case.trim_y_guards,
    )
    payload = build_portable_summary_payload(
        case_name=case.name,
        parity_mode=case.parity_mode,
        capability_tier=case.capability_tier,
        compare_variables=case.compare_variables,
        component_labels=tuple(component.label for component in run_config.components),
        dimensions={"t": 2, "x": snapshot.mesh.nx, "y": snapshot.mesh.local_ny, "z": snapshot.mesh.nz},
        time_points=(0.0, run_config.time.timestep),
        dataset_scalars=dataset_scalars,
        variables=_select_payload_variables(trimmed_variables, compare_variables=case.compare_variables),
        overrides=_effective_overrides(case.parity_mode, reference_case=case),
        configured_nout=run_config.time.nout,
        configured_timestep=run_config.time.timestep,
        producer="jax-drb",
    )
    return NativeRunResult(
        payload=payload,
        variables=trimmed_variables,
        time_points=(0.0, run_config.time.timestep),
        run_config=run_config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
    )


def _append_integrated_2d_recycling_diagnostics(
    variables: dict[str, np.ndarray],
    *,
    config: BoutConfig,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    initial_diagnostic_overrides: Mapping[str, np.ndarray] | None = None,
    preserve_dump_target_state: bool = False,
    preserve_dump_ion_target_state_only: bool = False,
    velocity_field_overrides_history: tuple[Mapping[str, np.ndarray] | None, ...] | None = None,
) -> None:
    diagnostic_history: dict[str, list[np.ndarray]] = {
        "Sd_target_recycle": [],
        "Ed_target_recycle": [],
    }
    field_names = tuple(name for name in variables if not name.startswith("S") and not name.startswith("E"))
    for time_index in range(next(iter(variables.values())).shape[0]):
        field_overrides = {
            name: np.asarray(variables[name][time_index], dtype=np.float64)
            for name in field_names
        }
        if velocity_field_overrides_history is not None and time_index < len(velocity_field_overrides_history):
            velocity_field_overrides = velocity_field_overrides_history[time_index]
            if velocity_field_overrides:
                field_overrides = _apply_species_velocity_overrides(
                    config,
                    field_overrides=field_overrides,
                    velocity_field_overrides=velocity_field_overrides,
                )
        rhs = compute_recycling_1d_rhs(
            config,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            field_overrides=field_overrides,
            apply_sheath_boundaries=True,
            preserve_dump_target_state=preserve_dump_target_state,
            preserve_dump_ion_target_state_only=preserve_dump_ion_target_state_only,
        )
        for diagnostic_name in diagnostic_history:
            if time_index == 0 and initial_diagnostic_overrides is not None and diagnostic_name in initial_diagnostic_overrides:
                diagnostic_history[diagnostic_name].append(
                    np.asarray(initial_diagnostic_overrides[diagnostic_name], dtype=np.float64)
                )
            else:
                diagnostic_history[diagnostic_name].append(np.asarray(rhs.variables[diagnostic_name][0], dtype=np.float64))
    for diagnostic_name, history in diagnostic_history.items():
        variables[diagnostic_name] = np.stack(history, axis=0)


def _integrated_2d_recycling_transient_steps(case: ReferenceCase, run_config: RunConfiguration) -> int:
    steps = run_config.time.nout
    for override in case.extra_overrides:
        key, separator, raw_value = override.partition("=")
        if separator and key.strip() == "nout":
            steps = int(float(raw_value.strip()))
    return steps


def run_input_case(
    input_path: str | Path,
    *,
    case_name: str | None = None,
    parity_mode: str = "manual",
    compare_variables: tuple[str, ...] = (),
    reference_case: ReferenceCase | None = None,
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
    verbose: bool = False,
    verbosity: str = "detailed",
    event_logger: Callable[[ABCMapping[str, Any]], None] | None = None,
) -> NativeRunResult:
    config = load_bout_input(input_path)
    resolved_case_name = case_name or Path(input_path).stem
    resolved_reference_case = reference_case
    if resolved_reference_case is None:
        resolved_reference_case = _reference_case_by_name_or_none(resolved_case_name)
    return run_config_case(
        config,
        case_name=resolved_case_name,
        parity_mode=parity_mode,
        compare_variables=compare_variables,
        reference_case=resolved_reference_case,
        restart_state=restart_state,
        output_steps=output_steps,
        verbose=verbose,
        verbosity=verbosity,
        event_logger=event_logger,
    )


def run_config_case(
    config: BoutConfig,
    *,
    case_name: str,
    parity_mode: str,
    compare_variables: tuple[str, ...] = (),
    reference_case: ReferenceCase | None = None,
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
    verbose: bool = False,
    verbosity: str = "detailed",
    event_logger: Callable[[ABCMapping[str, Any]], None] | None = None,
) -> NativeRunResult:
    event_sink = event_logger
    if event_sink is None and verbose:
        event_sink = lambda event: print_run_event(event, verbosity=verbosity)

    def emit(stage: str, message: str, **details: Any) -> None:
        if event_sink is None:
            return
        event_sink(build_run_event(stage=stage, message=message, details=details or None))

    def emit_progress(details: ABCMapping[str, Any]) -> None:
        emit(
            "progress",
            "Completed recycling transient interval",
            **dict(details),
        )

    run_config = RunConfiguration.from_config(config)
    emit(
        "configuration",
        "Resolved native run configuration",
        case_name=case_name,
        parity_mode=parity_mode,
        capability_tier=_resolved_capability_tier(reference_case),
        nout=run_config.time.nout,
        timestep=run_config.time.timestep,
        components=",".join(component.label for component in run_config.components),
        restart=restart_state is not None,
    )
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    emit(
        "mesh",
        "Built structured mesh and metrics",
        nx=mesh.nx,
        ny=mesh.local_ny,
        nz=mesh.nz,
        mxg=mesh.mxg,
        myg=mesh.myg,
        file=run_config.mesh.file or "<analytic mesh>",
    )
    time_points, variables = _execute_supported_case(
        config,
        run_config,
        mesh,
        metrics,
        parity_mode=parity_mode,
        restart_state=restart_state,
        output_steps=output_steps,
        progress_callback=emit_progress,
    )
    emit(
        "run",
        "Native solver completed",
        stored_states=len(time_points),
        compare_variables=",".join(compare_variables or tuple(variables)),
    )
    compare_names = compare_variables or tuple(variables)
    trimmed_variables = _prepare_compare_variables(
        variables,
        mesh,
        trim_x_guards=reference_case.trim_x_guards if reference_case is not None else False,
        trim_y_guards=reference_case.trim_y_guards if reference_case is not None else False,
    )
    dataset_scalars = resolved_dataset_scalars(run_config)
    payload = build_portable_summary_payload(
        case_name=case_name,
        parity_mode=parity_mode,
        capability_tier=_resolved_capability_tier(reference_case),
        compare_variables=compare_names,
        component_labels=tuple(component.label for component in run_config.components),
        dimensions={"t": len(time_points), "x": mesh.nx, "y": mesh.local_ny, "z": mesh.nz},
        time_points=time_points,
        dataset_scalars=dataset_scalars,
        variables={name: np.asarray(value, dtype=np.float64) for name, value in trimmed_variables.items()},
        overrides=_effective_overrides(parity_mode, reference_case=reference_case),
        configured_nout=run_config.time.nout,
        configured_timestep=run_config.time.timestep,
        producer="jax-drb",
    )
    emit(
        "summary",
        "Prepared portable summary payload",
        variables=",".join(sorted(trimmed_variables)),
        time_start=time_points[0] if time_points else 0.0,
        time_end=time_points[-1] if time_points else 0.0,
    )
    return NativeRunResult(
        payload=payload,
        variables=trimmed_variables,
        time_points=time_points,
        run_config=run_config,
        mesh=mesh,
        metrics=metrics,
    )


def _execute_supported_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
    progress_callback: Callable[[ABCMapping[str, Any]], None] | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    implementations = tuple(component.implementation for component in run_config.components)
    if len(run_config.components) == 1 and implementations == ("evolve_density",):
        component = run_config.components[0]
        variable_name = f"N{component.section}"
        if parity_mode != "one_rhs":
            raise NotImplementedError("Native single-component density execution currently supports one_rhs parity only.")
        field = _initialize_species_field(config, variable_name, mesh)
        return (0.0,), {variable_name: field[None, ...]}

    if _is_supported_diffusion_case(run_config):
        return _execute_diffusion_case(
            config,
            run_config,
            mesh,
            metrics,
            parity_mode=parity_mode,
            restart_state=restart_state,
            output_steps=output_steps,
        )

    if _is_supported_periodic_fluid_mms_case(config, run_config, mesh, metrics):
        return _execute_periodic_fluid_mms_case(
            config,
            run_config,
            mesh,
            metrics,
            parity_mode=parity_mode,
            restart_state=restart_state,
            output_steps=output_steps,
        )

    if _is_supported_electrostatic_vorticity_case(config, run_config, mesh, metrics):
        return _execute_electrostatic_vorticity_case(
            config,
            run_config,
            mesh,
            metrics,
            parity_mode=parity_mode,
            restart_state=restart_state,
            output_steps=output_steps,
        )

    if _is_supported_neutral_mixed_case(run_config):
        return _execute_neutral_mixed_case(
            config,
            run_config,
            mesh,
            metrics,
            parity_mode=parity_mode,
            output_steps=output_steps,
        )

    if _is_supported_recycling_1d_case(run_config, mesh):
        return _execute_recycling_1d_case(
            config,
            run_config,
            mesh,
            metrics,
            parity_mode=parity_mode,
            progress_callback=progress_callback,
        )

    if _is_supported_blob2d_case(config, run_config, mesh, metrics):
        return _execute_blob2d_case(
            config,
            run_config,
            mesh,
            metrics,
            parity_mode=parity_mode,
            restart_state=restart_state,
            output_steps=output_steps,
        )

    if _is_supported_drift_wave_case(config, run_config, mesh, metrics):
        return _execute_drift_wave_case(
            config,
            run_config,
            mesh,
            metrics,
            parity_mode=parity_mode,
            restart_state=restart_state,
            output_steps=output_steps,
        )

    raise NotImplementedError(
        "Native execution is not implemented for the configured component set: "
        + ", ".join(component.label for component in run_config.components)
    )


def _execute_diffusion_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    dtype = runtime_numpy_dtype()
    section = run_config.components[0].section
    density_name = f"N{section}"
    pressure_name = f"P{section}"
    if restart_state is None:
        density = _initialize_species_field(config, density_name, mesh)
        pressure = _initialize_species_field(config, pressure_name, mesh)
        time_offset = 0.0
    else:
        density = np.asarray(restart_state.variables[density_name], dtype=dtype)
        pressure = np.asarray(restart_state.variables[pressure_name], dtype=dtype)
        time_offset = restart_state.time_offset
    if not np.allclose(np.asarray(density), np.asarray(pressure), rtol=1e-12, atol=1e-12):
        raise NotImplementedError(
            "Native anomalous diffusion currently requires identical density and pressure initial states."
        )
    if not np.allclose(np.asarray(metrics.g23), 0.0, rtol=1e-12, atol=1e-12):
        raise NotImplementedError("Native anomalous diffusion currently supports g23 = 0 structured metrics only.")
    density_boundary = _field_boundary_kind(config, density_name)
    pressure_boundary = _field_boundary_kind(config, pressure_name)

    if bool(config.parsed(section, "thermal_conduction")) if config.has_option(section, "thermal_conduction") else True:
        raise NotImplementedError("Native one-step anomalous diffusion currently requires thermal_conduction = false.")

    resolver = NumericResolver(config)
    scalars = resolved_dataset_scalars(run_config)
    anomalous_D = resolver.resolve(section, "anomalous_D") / (scalars["rho_s0"] * scalars["rho_s0"] * scalars["Omega_ci"])
    if config.has_option(section, "anomalous_chi") and abs(resolver.resolve(section, "anomalous_chi")) > 0.0:
        raise NotImplementedError("Native one-step anomalous diffusion does not yet support anomalous_chi.")
    if config.has_option(section, "anomalous_nu") and abs(resolver.resolve(section, "anomalous_nu")) > 0.0:
        raise NotImplementedError("Native one-step anomalous diffusion does not yet support anomalous_nu.")

    steps = output_steps if output_steps is not None else _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
    history = advance_anomalous_diffusion_history(
        density,
        pressure,
        mesh=mesh,
        metrics=metrics,
        anomalous_D=anomalous_D,
        density_boundary=density_boundary,
        pressure_boundary=pressure_boundary,
        timestep=run_config.time.timestep,
        steps=steps,
    )
    time_points = tuple(time_offset + run_config.time.timestep * index for index in range(steps + 1))
    return time_points, {
        density_name: np.asarray(history.density_history, dtype=dtype),
        pressure_name: np.asarray(history.pressure_history, dtype=dtype),
    }


def _initialize_species_field(config: BoutConfig, variable_name: str, mesh: StructuredMesh) -> Any:
    if not config.has_section(variable_name):
        raise KeyError(f"Missing initial condition section for {variable_name}.")
    if config.has_option(variable_name, "function"):
        option_name = "function"
    elif config.has_option(variable_name, "solution"):
        option_name = "solution"
    else:
        raise KeyError(f"Missing initial condition function or solution for {variable_name}.")
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    field = broadcast_to_field_shape(
        evaluator.evaluate(config.raw(variable_name, option_name), current_section=variable_name),
        mesh,
    )
    field = apply_field_boundaries(field, mesh, x_boundary=_field_boundary_kind(config, variable_name))
    return field


def _field_boundary_kind(config: BoutConfig, variable_name: str) -> str:
    if config.has_option(variable_name, "bndry_all"):
        return str(config.parsed(variable_name, "bndry_all")).strip().lower()
    return "dirichlet_zero"


def _is_supported_diffusion_case(run_config: RunConfiguration) -> bool:
    implementations = tuple(component.implementation for component in run_config.components)
    if implementations != ("evolve_density", "evolve_pressure", "anomalous_diffusion"):
        return False
    sections = {component.section for component in run_config.components}
    return len(sections) == 1


def _is_supported_recycling_1d_case(run_config: RunConfiguration, mesh: StructuredMesh) -> bool:
    if mesh.nx != 1 or mesh.nz != 1:
        return False
    implementations = {component.implementation for component in run_config.components}
    required = {
        "evolve_density",
        "evolve_pressure",
        "evolve_momentum",
        "quasineutral",
        "zero_current",
        "sheath_boundary",
        "recycling",
        "reactions",
        "electron_force_balance",
    }
    if not required.issubset(implementations):
        return False
    return "sound_speed" not in implementations


def _execute_recycling_1d_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
    progress_callback: Callable[[ABCMapping[str, Any]], None] | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    dataset_scalars = resolved_dataset_scalars(run_config)
    if parity_mode == "one_rhs":
        result = compute_recycling_1d_rhs(
            config,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
        )
        return (0.0,), {name: np.asarray(value, dtype=np.float64) for name, value in result.variables.items()}

    steps = _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
    solver_mode = _select_recycling_transient_solver_mode(config, parity_mode=parity_mode)
    history = advance_recycling_1d_implicit_history(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        timestep=run_config.time.timestep,
        steps=steps,
        solver_mode=solver_mode,
        residual_tolerance=float(config.parsed("solver", "rtol")) if config.has_option("solver", "rtol") else 1.0e-8,
        max_nonlinear_iterations=30,
        progress_callback=progress_callback,
    )
    time_points = tuple(run_config.time.timestep * index for index in range(steps + 1))
    return time_points, {
        name: np.asarray(value, dtype=np.float64)
        for name, value in history.variable_history.items()
    }


def _execute_periodic_fluid_mms_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    section = run_config.components[0].section
    atomic_mass = float(config.parsed(section, "AA"))
    if parity_mode == "one_rhs":
        state = initialize_mms_state(config, section=section, mesh=mesh)
        rhs = compute_mms_rhs(config, state, section=section, mesh=mesh, metrics=metrics, atomic_mass=atomic_mass, time=0.0)
        return (0.0,), {
            f"ddt(N{section})": np.asarray(rhs.density[None, ...], dtype=np.float64),
            f"ddt(P{section})": np.asarray(rhs.pressure[None, ...], dtype=np.float64),
            f"ddt(NV{section})": np.asarray(rhs.momentum[None, ...], dtype=np.float64),
        }

    steps = output_steps if output_steps is not None else _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
    initial_state = (
        None
        if restart_state is None
        else Fluid1DState(
            density=jnp.asarray(restart_state.variables[f"N{section}"], dtype=jnp.float64),
            pressure=jnp.asarray(restart_state.variables[f"P{section}"], dtype=jnp.float64),
            momentum=jnp.asarray(restart_state.variables[f"NV{section}"], dtype=jnp.float64),
        )
    )
    time_offset = 0.0 if restart_state is None else restart_state.time_offset
    history = advance_mms_history(
        config,
        section=section,
        mesh=mesh,
        metrics=metrics,
        atomic_mass=atomic_mass,
        timestep=run_config.time.timestep,
        steps=steps,
        substeps=20,
        initial_state=initial_state,
        start_time=time_offset,
    )
    time_points = tuple(time_offset + run_config.time.timestep * index for index in range(steps + 1))
    return time_points, {
        f"N{section}": np.asarray(history.density_history, dtype=np.float64),
        f"P{section}": np.asarray(history.pressure_history, dtype=np.float64),
        f"NV{section}": np.asarray(history.momentum_history, dtype=np.float64),
    }


def _execute_electrostatic_vorticity_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    initial = (
        apply_vorticity_boundaries(_initialize_species_field(config, "Vort", mesh), mesh)
        if restart_state is None
        else apply_vorticity_boundaries(np.asarray(restart_state.variables["Vort"], dtype=np.float64), mesh)
    )
    time_offset = 0.0 if restart_state is None else restart_state.time_offset
    average_atomic_mass = float(config.parsed("vorticity", "average_atomic_mass")) if config.has_option("vorticity", "average_atomic_mass") else 2.0
    operator = build_vorticity_operator(mesh=mesh, metrics=metrics, average_atomic_mass=average_atomic_mass)

    if parity_mode == "one_rhs":
        rhs = compute_vorticity_rhs(initial, mesh=mesh, metrics=metrics, operator=operator)
        return (0.0,), {"ddt(Vort)": np.asarray(rhs.vorticity[None, ...], dtype=np.float64)}

    steps = output_steps if output_steps is not None else _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
    history = advance_vorticity_history(
        initial,
        mesh=mesh,
        metrics=metrics,
        operator=operator,
        timestep=run_config.time.timestep,
        steps=steps,
        start_time=time_offset,
        rtol=1e-6,
        atol=1e-8,
        mxstep=20000,
    )
    time_points = tuple(time_offset + run_config.time.timestep * index for index in range(steps + 1))
    return time_points, {
        "Vort": np.asarray(history.vorticity_history, dtype=np.float64),
        "phi": np.asarray(history.potential_history, dtype=np.float64),
    }


def _execute_neutral_mixed_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
    output_steps: int | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    section = run_config.components[0].section
    scalars = resolved_dataset_scalars(run_config)
    if parity_mode == "one_rhs":
        state = initialize_neutral_mixed_state(config, section=section, mesh=mesh)
        rhs = compute_neutral_mixed_rhs(
            config,
            state,
            section=section,
            mesh=mesh,
            metrics=metrics,
            meters_scale=float(scalars["rho_s0"]),
            tnorm=float(scalars["Tnorm"]),
        )
        return (0.0,), {
            f"N{section}": np.asarray(state.density[None, ...], dtype=np.float64),
            f"P{section}": np.asarray(state.pressure[None, ...], dtype=np.float64),
            f"NV{section}": np.asarray(state.momentum[None, ...], dtype=np.float64),
            f"ddt(N{section})": np.asarray(rhs.density[None, ...], dtype=np.float64),
            f"ddt(P{section})": np.asarray(rhs.pressure[None, ...], dtype=np.float64),
            f"ddt(NV{section})": np.asarray(rhs.momentum[None, ...], dtype=np.float64),
        }

    if parity_mode not in {"one_step", "short_window"}:
        raise NotImplementedError(
            "Native neutral mixed execution currently supports one_rhs, one_step, and short_window parity only."
        )

    steps = output_steps if output_steps is not None else _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
    history = advance_neutral_mixed_implicit_history(
        config,
        section=section,
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
        timestep=run_config.time.timestep,
        steps=steps,
        solver_mode="matrix_free",
        residual_tolerance=1.0e-8,
        step_tolerance=1.0e-10,
        max_nonlinear_iterations=8,
        linear_restart=20,
        linear_maxiter=200,
        linear_rtol=1.0e-8,
    )
    time_points = tuple(run_config.time.timestep * index for index in range(steps + 1))
    return time_points, {
        f"N{section}": np.asarray(history.density_history, dtype=np.float64),
        f"P{section}": np.asarray(history.pressure_history, dtype=np.float64),
        f"NV{section}": np.asarray(history.momentum_history, dtype=np.float64),
    }


def _is_supported_periodic_fluid_mms_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> bool:
    implementations = tuple(component.implementation for component in run_config.components)
    if implementations != ("evolve_density", "evolve_pressure", "evolve_momentum"):
        return False
    if len({component.section for component in run_config.components}) != 1:
        return False
    section = run_config.components[0].section
    if run_config.mesh.nx != 1 or run_config.mesh.nz != 1:
        return False
    if run_config.solver.mms is not True:
        return False
    if bool(config.parsed(section, "thermal_conduction")) if config.has_option(section, "thermal_conduction") else True:
        return False
    if bool(config.parsed(section, "p_div_v")) if config.has_option(section, "p_div_v") else False:
        return False
    if not _uniform_identity_parallel_metric(mesh, metrics=metrics):
        return False
    return all(
        config.has_section(name) and config.has_option(name, "solution") and config.has_option(name, "source")
        for name in (f"N{section}", f"P{section}", f"NV{section}")
    )


def _is_supported_electrostatic_vorticity_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> bool:
    if tuple(component.implementation for component in run_config.components) != ("vorticity",):
        return False
    if run_config.mesh.ny != 1 or run_config.mesh.myg != 0:
        return False
    if mesh.mxg != 2:
        return False
    if not config.has_section("Vort") or not config.has_option("Vort", "function"):
        return False
    option_defaults = {
        "diamagnetic": False,
        "diamagnetic_polarisation": False,
        "bndry_flux": False,
        "poloidal_flows": False,
        "split_n0": False,
        "phi_dissipation": False,
        "vort_dissipation": False,
        "collisional_friction": False,
        "phi_boundary_relax": False,
        "phi_sheath_dissipation": False,
        "damp_core_vorticity": False,
    }
    for key, expected in option_defaults.items():
        value = bool(config.parsed("vorticity", key)) if config.has_option("vorticity", key) else (True if key == "phi_dissipation" else False)
        if value != expected:
            return False
    exb_advection = bool(config.parsed("vorticity", "exb_advection")) if config.has_option("vorticity", "exb_advection") else True
    exb_advection_simplified = (
        bool(config.parsed("vorticity", "exb_advection_simplified"))
        if config.has_option("vorticity", "exb_advection_simplified")
        else True
    )
    if not exb_advection or not exb_advection_simplified:
        return False
    if not np.allclose(np.asarray(metrics.g23), 0.0, rtol=1e-12, atol=1e-12):
        return False
    return True


def _is_supported_neutral_mixed_case(run_config: RunConfiguration) -> bool:
    implementations = tuple(component.implementation for component in run_config.components)
    if implementations != ("neutral_mixed",):
        return False
    return len({component.section for component in run_config.components}) == 1


def _execute_drift_wave_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    if parity_mode not in {"one_rhs", "one_step", "short_window"}:
        raise NotImplementedError(
            "Native drift-wave support currently covers one_rhs, one_step, and short_window parity only."
        )

    benchmark = build_drift_wave_benchmark(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    initial_state = (
        initialize_drift_wave_state(config, mesh=mesh)
        if restart_state is None
        else DriftWaveState(
            ion_density=np.asarray(restart_state.variables["Ni"], dtype=np.float64)[mesh.xstart, mesh.ystart : mesh.yend + 1, :],
            electron_momentum=np.asarray(restart_state.variables["NVe"], dtype=np.float64)[mesh.xstart, mesh.ystart : mesh.yend + 1, :],
            vorticity=np.asarray(restart_state.variables["Vort"], dtype=np.float64)[mesh.xstart, mesh.ystart : mesh.yend + 1, :],
        )
    )
    time_offset = 0.0 if restart_state is None else restart_state.time_offset

    if parity_mode == "one_rhs":
        rhs = compute_drift_wave_rhs(initial_state, mesh=mesh, benchmark=benchmark)
        density_field = _assemble_density_output(initial_state.ion_density, benchmark=benchmark, mesh=mesh)
        pressure_field = density_field * benchmark.electron_temperature
        return (0.0,), {
            "Ni": np.asarray(density_field[None, ...], dtype=np.float64),
            "Ne": np.asarray(density_field[None, ...], dtype=np.float64),
            "Pe": np.asarray(pressure_field[None, ...], dtype=np.float64),
            "ddt(Ni)": np.asarray(_assemble_density_output(rhs.density, benchmark=benchmark, mesh=mesh)[None, ...], dtype=np.float64),
            "ddt(NVe)": np.asarray(_assemble_zero_dirichlet_output(rhs.momentum, mesh=mesh)[None, ...], dtype=np.float64),
            "ddt(Vort)": np.asarray(_assemble_zero_dirichlet_output(rhs.vorticity, mesh=mesh)[None, ...], dtype=np.float64),
        }

    if parity_mode == "one_step":
        history = advance_drift_wave_history(
            initial_state,
            mesh=mesh,
            benchmark=benchmark,
            timestep=run_config.time.timestep,
            steps=1,
            substeps=10,
            start_time=time_offset,
        )
        return (time_offset, time_offset + run_config.time.timestep), {
            "Ni": np.asarray(history.ion_density_history, dtype=np.float64),
            "Ne": np.asarray(history.ion_density_history, dtype=np.float64),
            "NVe": np.asarray(history.electron_momentum_history, dtype=np.float64),
            "Vort": np.asarray(history.vorticity_history, dtype=np.float64),
            "phi": np.asarray(history.potential_history, dtype=np.float64),
        }

    steps = output_steps if output_steps is not None else _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
    history = advance_drift_wave_history_adaptive(
        initial_state,
        mesh=mesh,
        benchmark=benchmark,
        timestep=run_config.time.timestep,
        steps=steps,
        start_time=time_offset,
        rtol=1e-6,
        atol=1e-8,
        max_step=1.0,
        initial_step=0.25,
        include_parallel_transport=False,
        include_phi_dissipation=False,
    )
    time_points = tuple(time_offset + run_config.time.timestep * index for index in range(steps + 1))
    return time_points, {
        "Ni": np.asarray(history.ion_density_history, dtype=np.float64),
        "Ne": np.asarray(history.ion_density_history, dtype=np.float64),
        "NVe": np.asarray(history.electron_momentum_history, dtype=np.float64),
        "Vort": np.asarray(history.vorticity_history, dtype=np.float64),
        "phi": np.asarray(history.potential_history, dtype=np.float64),
    }


def _execute_blob2d_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    benchmark = build_blob2d_benchmark(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    initial_state = (
        initialize_blob2d_state(config, mesh=mesh)
        if restart_state is None
        else Blob2DState(
            electron_density=np.asarray(restart_state.variables["Ne"], dtype=np.float64),
            vorticity=np.asarray(restart_state.variables["Vort"], dtype=np.float64),
        )
    )
    time_offset = 0.0 if restart_state is None else restart_state.time_offset

    if parity_mode == "one_rhs":
        rhs = compute_blob2d_rhs(initial_state, mesh=mesh, benchmark=benchmark, operator=None)
        return (0.0,), {
            "Ne": np.asarray(rhs.electron_density[None, ...], dtype=np.float64),
            "Pe": np.asarray(rhs.electron_pressure[None, ...], dtype=np.float64),
            "phi": np.asarray(rhs.potential[None, ...], dtype=np.float64),
            "ddt(Ne)": np.asarray(rhs.density_rhs[None, ...], dtype=np.float64),
            "ddt(Vort)": np.asarray(rhs.vorticity_rhs[None, ...], dtype=np.float64),
        }

    if parity_mode not in {"one_step", "short_window"}:
        raise NotImplementedError("Native blob2d support currently covers one_rhs, one_step, and short_window parity only.")

    operator = build_blob2d_potential_operator(
        mesh=mesh,
        metrics=metrics,
        average_atomic_mass=NumericResolver(config).resolve("vorticity", "average_atomic_mass"),
    )
    steps = output_steps if output_steps is not None else _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
    substeps = 14 if parity_mode == "short_window" else 10
    history = advance_blob2d_history(
        initial_state,
        mesh=mesh,
        benchmark=benchmark,
        operator=operator,
        timestep=run_config.time.timestep,
        steps=steps,
        substeps=substeps,
        start_time=time_offset,
    )
    return tuple(time_offset + run_config.time.timestep * index for index in range(steps + 1)), {
        "Ne": np.asarray(history.electron_density_history, dtype=np.float64),
        "Pe": np.asarray(history.electron_pressure_history, dtype=np.float64),
        "Vort": np.asarray(history.vorticity_history, dtype=np.float64),
        "phi": np.asarray(history.potential_history, dtype=np.float64),
    }


def _is_supported_drift_wave_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> bool:
    implementations = tuple(component.implementation for component in run_config.components)
    expected = (
        "evolve_density",
        "fixed_velocity",
        "fixed_temperature",
        "quasineutral",
        "evolve_momentum",
        "fixed_temperature",
        "vorticity",
        "sound_speed",
        "braginskii_collisions",
        "braginskii_friction",
        "braginskii_heat_exchange",
    )
    if implementations != expected:
        return False
    if tuple(component.section for component in run_config.components[:3]) != ("i", "i", "i"):
        return False
    if tuple(component.section for component in run_config.components[3:6]) != ("e", "e", "e"):
        return False
    if mesh.mxg != 2 or mesh.myg != 2:
        return False
    if mesh.xend != mesh.xstart:
        return False
    if bool(config.parsed("vorticity", "diamagnetic")) if config.has_option("vorticity", "diamagnetic") else False:
        return False
    if bool(config.parsed("vorticity", "diamagnetic_polarisation")) if config.has_option("vorticity", "diamagnetic_polarisation") else False:
        return False
    if bool(config.parsed("vorticity", "bndry_flux")) if config.has_option("vorticity", "bndry_flux") else False:
        return False
    if bool(config.parsed("vorticity", "poloidal_flows")) if config.has_option("vorticity", "poloidal_flows") else False:
        return False
    if float(config.parsed("i", "charge")) != 1.0 or float(config.parsed("e", "charge")) != -1.0:
        return False
    if not config.has_option("Ni", "function"):
        return False
    return np.allclose(np.asarray(metrics.g23), 0.0, rtol=1e-12, atol=1e-12)


def _is_supported_blob2d_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> bool:
    implementations = tuple(component.implementation for component in run_config.components)
    if implementations != ("evolve_density", "isothermal", "vorticity", "sheath_closure"):
        return False
    if tuple(component.section for component in run_config.components[:2]) != ("e", "e"):
        return False
    if mesh.myg != 0:
        return False
    if mesh.nz <= 1:
        return False
    if not config.has_option("Ne", "function"):
        return False
    option_defaults = {
        "diamagnetic": True,
        "diamagnetic_polarisation": False,
        "bndry_flux": False,
        "poloidal_flows": False,
        "split_n0": False,
        "phi_dissipation": False,
    }
    for key, expected in option_defaults.items():
        value = bool(config.parsed("vorticity", key)) if config.has_option("vorticity", key) else False
        if value != expected:
            return False
    return np.allclose(np.asarray(metrics.g23), 0.0, rtol=1e-12, atol=1e-12)


def _assemble_density_output(interior: Any, *, benchmark: DriftWaveBenchmark, mesh: StructuredMesh) -> Any:
    return _assemble_density_field(interior, benchmark=benchmark, mesh=mesh)


def _assemble_zero_dirichlet_output(interior: Any, *, mesh: StructuredMesh) -> Any:
    return _assemble_zero_dirichlet_field(interior, mesh=mesh)


def _uniform_identity_parallel_metric(mesh: StructuredMesh, *, metrics: StructuredMetrics) -> bool:
    if not np.allclose(np.asarray(metrics.J), 1.0, rtol=1e-12, atol=1e-12):
        return False
    if not np.allclose(np.asarray(metrics.g22), 1.0, rtol=1e-12, atol=1e-12):
        return False
    if not np.allclose(np.asarray(metrics.g23), 0.0, rtol=1e-12, atol=1e-12):
        return False
    dy = np.asarray(metrics.dy[:, mesh.ystart : mesh.yend + 1, :], dtype=np.float64)
    return np.allclose(dy, dy[:, :1, :], rtol=1e-12, atol=1e-12)
