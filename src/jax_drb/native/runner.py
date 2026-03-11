from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config.boutinp import BoutConfig, load_bout_input
from ..config.normalization import ELEMENTARY_CHARGE, PROTON_MASS
from ..parity.portable import build_portable_summary_payload
from ..reference.cases import ReferenceCase
from ..runtime.run_config import RunConfiguration
from .expression import ArrayExpressionEvaluator
from .mesh import (
    StructuredMesh,
    apply_zero_dirichlet_x_guards,
    build_structured_mesh,
    communicate_y_guards,
    project_nonnegative_x_boundaries,
)


@dataclass(frozen=True)
class NativeRunResult:
    payload: Mapping[str, Any]
    variables: Mapping[str, Any]
    run_config: RunConfiguration
    mesh: StructuredMesh


def run_curated_case(
    case_name: str,
    *,
    reference_root: str | Path,
    manifest_path: str | Path | None = None,
) -> NativeRunResult:
    from ..parity.reference import resolve_reference_case

    case, input_path = resolve_reference_case(case_name, reference_root=reference_root, manifest_path=manifest_path)
    return run_input_case(
        input_path,
        case_name=case.name,
        parity_mode=case.parity_mode,
        compare_variables=case.compare_variables,
        reference_case=case,
    )


def run_input_case(
    input_path: str | Path,
    *,
    case_name: str | None = None,
    parity_mode: str = "manual",
    compare_variables: tuple[str, ...] = (),
    reference_case: ReferenceCase | None = None,
) -> NativeRunResult:
    config = load_bout_input(input_path)
    return run_config_case(
        config,
        case_name=case_name or Path(input_path).stem,
        parity_mode=parity_mode,
        compare_variables=compare_variables,
        reference_case=reference_case,
    )


def run_config_case(
    config: BoutConfig,
    *,
    case_name: str,
    parity_mode: str,
    compare_variables: tuple[str, ...] = (),
    reference_case: ReferenceCase | None = None,
) -> NativeRunResult:
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    variables = _execute_supported_case(config, run_config, mesh)
    compare_names = compare_variables or tuple(variables)
    dataset_scalars = _dataset_scalars(run_config)
    payload = build_portable_summary_payload(
        case_name=case_name,
        parity_mode=parity_mode,
        compare_variables=compare_names,
        component_labels=tuple(component.label for component in run_config.components),
        dimensions={"t": 1, "x": mesh.nx, "y": mesh.local_ny, "z": mesh.nz},
        time_points=(0.0,),
        dataset_scalars=dataset_scalars,
        variables={name: np.asarray(value[None, ...], dtype=np.float64) for name, value in variables.items()},
        overrides=("nout=0",) if parity_mode == "one_rhs" else (),
        configured_nout=run_config.time.nout,
        configured_timestep=run_config.time.timestep,
        producer="jax-drb",
    )
    return NativeRunResult(payload=payload, variables=variables, run_config=run_config, mesh=mesh)


def _execute_supported_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
) -> dict[str, Any]:
    if len(run_config.components) != 1:
        raise NotImplementedError("Native execution currently supports exactly one scheduled component.")
    component = run_config.components[0]
    if component.implementation != "evolve_density":
        raise NotImplementedError(
            f"Native execution for component {component.label!r} is not implemented yet."
        )

    variable_name = f"N{component.section}"
    if not config.has_section(variable_name) or not config.has_option(variable_name, "function"):
        raise KeyError(f"Missing initial condition function for {variable_name}.")

    field = _initialize_evolve_density(config, variable_name, mesh)
    return {variable_name: field}


def _initialize_evolve_density(config: BoutConfig, variable_name: str, mesh: StructuredMesh) -> Any:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    field = evaluator.evaluate(config.raw(variable_name, "function"), current_section=variable_name)
    field = apply_zero_dirichlet_x_guards(field, mesh)
    field = communicate_y_guards(field, mesh)
    field = project_nonnegative_x_boundaries(field, mesh)
    return field


def _dataset_scalars(run_config: RunConfiguration) -> dict[str, float]:
    if run_config.normalization is not None:
        normalization = run_config.normalization
        return {
            "Nnorm": normalization.Nnorm,
            "Tnorm": normalization.Tnorm,
            "Bnorm": normalization.Bnorm,
            "Cs0": normalization.Cs0,
            "Omega_ci": normalization.Omega_ci,
            "rho_s0": normalization.rho_s0,
        }

    Nnorm = float(run_config.model_scalars.get("Nnorm", 1.0e19))
    Tnorm = float(run_config.model_scalars.get("Tnorm", 100.0))
    Bnorm = float(run_config.model_scalars.get("Bnorm", 1.0))
    Cs0 = float((ELEMENTARY_CHARGE * Tnorm / PROTON_MASS) ** 0.5)
    Omega_ci = float(ELEMENTARY_CHARGE * Bnorm / PROTON_MASS)
    rho_s0 = float(Cs0 / Omega_ci)
    return {
        "Nnorm": Nnorm,
        "Tnorm": Tnorm,
        "Bnorm": Bnorm,
        "Cs0": Cs0,
        "Omega_ci": Omega_ci,
        "rho_s0": rho_s0,
    }
