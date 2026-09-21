#!/usr/bin/env python3
"""Operator-only HSX/RLP audit for the manufactured vorticity bracket.

The audit deliberately avoids the six-field RHS, phi reconstruction, time
integration, parallel operators, and curvature.  It consumes immutable
simulation-geometry artifacts, builds only the local Poisson-bracket support,
and writes one compact JSON result per process.  The ``campaign`` command
launches one process per resolution and merges those results after each child
has exited, so JAX and geometry allocations cannot accumulate across the
32/48/64 chain.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulate_hsx_blob as blob  # noqa: E402
import simulate_hsx_mms as mms  # noqa: E402
from hsx_mms_continuum_reference import (  # noqa: E402
    build_continuum_reference_from_artifact,
)
from drbx.geometry import (  # noqa: E402
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
)
from drbx.native.fci_boundaries import (  # noqa: E402
    build_local_boundary_face_trace_from_halo,
)
from drbx.native.fci_operators import (  # noqa: E402
    _third_order_scalar_face_states_from_halo,
    build_local_control_volume_direct_face_states,
    build_local_control_volume_poisson_face_stencil,
    local_control_volume_projected_fine_cell_volume,
    local_poisson_bracket_compatible_flux_op,
    restrict_local_control_volume_diffusion_average,
)


SCHEMA = "hsx-poisson-vorticity-suspect-audit-v2"
VARIANTS = (
    "phi_centered_action",
    "omega_centered_reverse_action",
    "current_centered_antisymmetric",
    "pure_scalar_upwind",
    "compatible_upwind",
)
OPERAND_CONTROLS = (
    "raw_phi_raw_omega",
    "H_phi_raw_omega",
    "raw_phi_H_omega",
    "H_phi_H_omega",
)
PRIMARY_REGIONS = (
    "global",
    "ordinary_bulk",
    "rlp_transition_rings",
    "physical_wall",
)
_FROZEN_RUNTIME_ENV = {
    "DRBX_PARALLEL_FLUX_PAIRING": "support-core",
    "DRBX_PARALLEL_BOUNDARY_PAIRING": "characteristic-sat",
    "DRBX_PARALLEL_SHORT_LEG_TREATMENT": "local-backward-euler",
    "DRBX_PARALLEL_SHORT_LEG_SELECTION": "all-physical-walls",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _artifact_directory(root: Path, resolution: int) -> Path:
    root = Path(root)
    if (root / "manifest.json").is_file():
        return root

    size = f"{int(resolution)}x{int(resolution)}x{int(resolution)}"
    candidates = (root / size, root / f"hsx_fci_{size}")
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    searched = ", ".join(str(path / "manifest.json") for path in candidates)
    raise FileNotFoundError(
        "cached simulation-geometry artifact is missing; this audit will not "
        f"rebuild it. Searched: {searched}"
    )


def _load_artifact(root: Path, resolution: int):
    path = _artifact_directory(root, resolution)
    artifact = blob.load_fci_simulation_geometry(path)
    expected = (int(resolution),) * 3
    if tuple(int(value) for value in artifact.global_geometry.shape) != expected:
        raise ValueError(
            f"artifact {path} has shape {artifact.global_geometry.shape}, "
            f"expected {expected}"
        )
    if artifact.owner_geometry is None:
        raise ValueError(f"artifact {path} has no producer-owned RLP geometry")
    return path, artifact


def _weighted_statistics(
    actual: np.ndarray,
    exact: np.ndarray,
    volume: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    actual = np.asarray(actual, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(selected))
    total_volume = float(np.sum(volume[selected]))
    if count == 0 or total_volume <= 0.0:
        return {
            "cell_count": count,
            "volume": total_volume,
            "absolute_l2": None,
            "relative_l2": None,
            "squared_error": None,
            "exact_rms": None,
        }
    difference = actual - exact
    squared_error = float(np.sum(volume[selected] * difference[selected] ** 2))
    exact_squared = float(np.sum(volume[selected] * exact[selected] ** 2))
    absolute = math.sqrt(squared_error / total_volume)
    exact_rms = math.sqrt(exact_squared / total_volume)
    relative = math.sqrt(squared_error / max(exact_squared, np.finfo(float).tiny))
    return {
        "cell_count": count,
        "volume": total_volume,
        "absolute_l2": absolute,
        "relative_l2": relative,
        "squared_error": squared_error,
        "exact_rms": exact_rms,
    }


def _orders(resolutions: Sequence[int], errors: Sequence[float | None]) -> list[float | None]:
    values: list[float | None] = []
    for coarse_n, fine_n, coarse, fine in zip(
        resolutions[:-1], resolutions[1:], errors[:-1], errors[1:]
    ):
        if coarse is None or fine is None or coarse <= 0.0 or fine <= 0.0:
            values.append(None)
        else:
            values.append(
                math.log(float(coarse) / float(fine))
                / math.log(float(fine_n) / float(coarse_n))
            )
    return values


def _variant_statistics(
    actual: np.ndarray,
    exact: np.ndarray,
    host: Any,
    masks: Mapping[str, np.ndarray],
    *,
    include_rings: bool,
) -> dict[str, Any]:
    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    volume = np.asarray(host.aggregate_chart_volume, dtype=np.float64)
    result: dict[str, Any] = {
        "global": _weighted_statistics(actual, exact, volume, active)
    }
    for name in PRIMARY_REGIONS[1:]:
        result[name] = _weighted_statistics(actual, exact, volume, masks[name])
    global_sse = result["global"]["squared_error"]
    transition_sse = result["rlp_transition_rings"]["squared_error"]
    result["rlp_transition_squared_error_fraction"] = (
        None
        if global_sse in (None, 0.0) or transition_sse is None
        else float(transition_sse) / float(global_sse)
    )
    if include_rings:
        rings = []
        for index in range(active.shape[0]):
            ring = np.zeros(active.shape, dtype=bool)
            ring[index] = active[index]
            rings.append(
                {
                    "radial_index": index,
                    "angular_group_size": int(host.angular_group_size[index]),
                    **_weighted_statistics(actual, exact, volume, ring),
                }
            )
        result["radial_rings"] = rings
    return result


def _region_masks(geometry: Any, host: Any) -> dict[str, np.ndarray]:
    # Stage 7 selects all physical walls for its short-leg treatment.  Passing
    # the complete wall mask reproduces that disjoint partition without
    # evaluating a parallel or implicit operator merely to rediscover it.
    selected_wall = np.asarray(geometry.maps.forward_boundary, dtype=bool) | np.asarray(
        geometry.maps.backward_boundary, dtype=bool
    )
    return mms._region_masks(geometry, host, selected_wall)


def _poisson_reference_fields(
    projector: Any,
    time: float,
    *,
    rho_star: float,
) -> tuple[Any, np.ndarray]:
    """Project MMS state values and only the vorticity Poisson lane.

    ``_QuadratureProjector`` supplies the structured fourth-order vorticity
    gradient required by the frozen MMS.  Evaluating the bracket directly
    from those prepared chunks avoids constructing the unrelated continuum
    RHS lanes, sources, Hessian consumers, and parallel/curvature references.
    """

    cells = int(np.prod(projector.shape))
    values = {name: np.empty(cells, dtype=np.float64) for name in mms.FIELDS}
    exact = np.empty(cells, dtype=np.float64)
    for first, last, points, prepared, weighted in projector.chunks:
        data = projector.reference.evaluate(points, time, prepared=prepared)
        denominator = np.sum(weighted, axis=1)

        def project(array: Any) -> np.ndarray:
            shaped = np.asarray(array, dtype=np.float64).reshape(
                (last - first, projector.nq)
            )
            return np.sum(weighted * shaped, axis=1) / np.maximum(
                denominator, 1.0e-30
            )

        for name in mms.FIELDS:
            values[name][first:last] = project(data.values[name])
        continuum = -np.sum(
            np.asarray(prepared.bcov, dtype=np.float64)
            * np.cross(
                np.asarray(data.gradients["phi"], dtype=np.float64),
                np.asarray(data.gradients["vorticity"], dtype=np.float64),
            ),
            axis=-1,
        )
        continuum /= (
            float(rho_star)
            * np.maximum(np.abs(np.asarray(prepared.J, dtype=np.float64)), 1.0e-30)
            * np.maximum(np.asarray(prepared.B, dtype=np.float64), 1.0e-30)
        )
        exact[first:last] = project(continuum)
    state = blob.FciDrbEBState(
        **{
            name: values[name].reshape(projector.shape)
            for name in mms.FIELDS
        }
    )
    return state, exact.reshape(projector.shape)


@dataclass(frozen=True)
class _PreparedOperands:
    phi_stencil: Any
    omega_stencil: Any
    phi_trace: Any
    omega_trace: Any
    omega_halo: Any
    omega_direct_states: tuple[Any, Any]
    raw_phi_stencil: Any
    raw_omega_stencil: Any
    raw_phi_trace: Any
    raw_omega_trace: Any
    raw_omega_halo: Any
    omega_owner: Any


def _prepare_operands(model: Any, owner_state: Any, raw_state: Any) -> _PreparedOperands:
    geometry = model.geometry
    domain = model.domain
    context = StencilBuilderContext(layout=domain.layout, domain=domain)
    face_bc = model._face_bcs(owner_state)
    owner_halos = model._prepare_state_halo(owner_state, face_bc)
    phi_halo = model._prepare_phi_halo(owner_state.phi, face_bc.phi)
    owner_halos = owner_halos.replace(phi=phi_halo)
    operator_boundary = blob.build_local_fci_drb_eb_operator_boundary_bundle(
        owner_halos,
        geometry,
        domain,
        face_bc,
        tau=model.parameters.tau,
    )

    phi_pb_halo = model._prepare_poisson_bracket_halo(owner_state.phi, face_bc.phi)
    omega_halo = model._prepare_poisson_bracket_support_halo(
        owner_state.vorticity, face_bc.vorticity
    )
    phi_stencil = build_local_conservative_stencil_from_field(
        phi_pb_halo, geometry, context
    )
    omega_stencil, omega_left, omega_right = (
        build_local_control_volume_poisson_face_stencil(
            omega_halo,
            geometry,
            domain,
            context,
            model.control_volume_geometry,
            model.control_volume_boundary_bc,
            owner_values_owned=model._owner_field(owner_state.vorticity),
            regular_face_bc=face_bc.vorticity,
            boundary_trace=operator_boundary.vorticity,
            halo_exchange=model.halo_exchange,
            topology_filler=model.topology_filler,
        )
    )

    raw_phi_halo = model._prepare_raw_poisson_bracket_halo(
        raw_state.phi, face_bc.phi
    )
    raw_omega_support_bc = model._prepare_poisson_bracket_support_face_bc(
        face_bc.vorticity
    )
    raw_omega_halo = model._prepare_raw_poisson_bracket_halo(
        raw_state.vorticity, raw_omega_support_bc
    )
    raw_omega_normal_halo = model._prepare_raw_poisson_bracket_halo(
        raw_state.vorticity, face_bc.vorticity
    )
    raw_phi_trace = build_local_boundary_face_trace_from_halo(
        raw_phi_halo, geometry, domain, face_bc.phi
    )
    raw_omega_trace = build_local_boundary_face_trace_from_halo(
        raw_omega_normal_halo, geometry, domain, face_bc.vorticity
    )
    return _PreparedOperands(
        phi_stencil=phi_stencil,
        omega_stencil=omega_stencil,
        phi_trace=operator_boundary.phi,
        omega_trace=operator_boundary.vorticity,
        omega_halo=omega_halo,
        omega_direct_states=(omega_left, omega_right),
        raw_phi_stencil=build_local_conservative_stencil_from_field(
            raw_phi_halo, geometry, context
        ),
        raw_omega_stencil=build_local_conservative_stencil_from_field(
            raw_omega_halo, geometry, context
        ),
        raw_phi_trace=raw_phi_trace,
        raw_omega_trace=raw_omega_trace,
        raw_omega_halo=raw_omega_halo,
        omega_owner=model._owner_field(owner_state.vorticity),
    )


def _operator_call(
    model: Any,
    f_stencil: Any,
    g_stencil: Any,
    *,
    f_trace: Any,
    g_trace: Any,
    characteristic_scheme: str,
    g_halo: Any | None = None,
    g_direct_states: tuple[Any, Any] | None = None,
) -> Any:
    return local_poisson_bracket_compatible_flux_op(
        f_stencil,
        g_stencil,
        model.geometry,
        domain=model.domain,
        axis_regular_axes=model.axis_regular_axes,
        f_boundary_trace=f_trace,
        g_boundary_trace=g_trace,
        characteristic_scheme=characteristic_scheme,
        g_field_halo=g_halo,
        g_direct_face_states=g_direct_states,
        cell_volume=local_control_volume_projected_fine_cell_volume(
            model.geometry, model.control_volume_geometry
        ),
    )


def _evaluate_variants(
    model: Any, operands: _PreparedOperands
) -> dict[str, dict[str, np.ndarray]]:
    def evaluate():
        phi_centered = _operator_call(
            model,
            operands.phi_stencil,
            operands.omega_stencil,
            f_trace=operands.phi_trace,
            g_trace=operands.omega_trace,
            characteristic_scheme="scalar-centered",
        )
        omega_centered = _operator_call(
            model,
            operands.omega_stencil,
            operands.phi_stencil,
            f_trace=operands.omega_trace,
            g_trace=operands.phi_trace,
            characteristic_scheme="scalar-centered",
        )
        centered = _operator_call(
            model,
            operands.phi_stencil,
            operands.omega_stencil,
            f_trace=operands.phi_trace,
            g_trace=operands.omega_trace,
            characteristic_scheme="centered",
        )
        centered_swapped = _operator_call(
            model,
            operands.omega_stencil,
            operands.phi_stencil,
            f_trace=operands.omega_trace,
            g_trace=operands.phi_trace,
            characteristic_scheme="centered",
        )
        pure_upwind = _operator_call(
            model,
            operands.phi_stencil,
            operands.omega_stencil,
            f_trace=operands.phi_trace,
            g_trace=operands.omega_trace,
            characteristic_scheme="scalar-third-order-upwind",
            g_halo=operands.omega_halo,
            g_direct_states=operands.omega_direct_states,
        )
        compatible_upwind = _operator_call(
            model,
            operands.phi_stencil,
            operands.omega_stencil,
            f_trace=operands.phi_trace,
            g_trace=operands.omega_trace,
            characteristic_scheme="compatible-third-order-upwind",
            g_halo=operands.omega_halo,
            g_direct_states=operands.omega_direct_states,
        )
        fine_actions = (
            phi_centered,
            omega_centered,
            centered,
            centered_swapped,
            pure_upwind,
            compatible_upwind,
        )
        return _return_actions_to_owner(model, fine_actions)

    values = jax.jit(evaluate)()
    values = jax.block_until_ready(values)
    names = (
        "phi_centered",
        "omega_centered",
        "centered",
        "centered_swapped",
        "pure_upwind",
        "compatible_upwind",
    )
    production, mass_weighted_h_transpose = values
    return {
        "production_R": {
            name: np.asarray(value)
            for name, value in zip(names, production)
        },
        "mass_weighted_H_transpose": {
            name: np.asarray(value)
            for name, value in zip(names, mass_weighted_h_transpose)
        },
    }


def _return_actions_to_owner(model: Any, fine_actions: Sequence[Any]):
    """Apply the two owner test maps being compared by this audit.

    The production Poisson bracket completes its fine-grid action and then
    applies the physical-volume aggregate average ``R``.  The diagnostic
    alternative uses the reconstruction-matched physical adjoint

    ``M_owner**-1 H.T M_raw``.

    Calling the latter ``H.T`` in reports is shorthand only: the two mass
    matrices are required to map a fine cell-average residual back to an
    owner cell-average residual with the correct physical units.
    """

    stacked = jnp.stack(tuple(fine_actions), axis=0)
    production = jax.vmap(model._restrict_fine_field)(stacked)
    mass_weighted_h_transpose = jax.vmap(
        lambda value: restrict_local_control_volume_diffusion_average(
            value,
            model.control_volume_geometry,
            model.domain,
        )
    )(stacked)
    return production, mass_weighted_h_transpose


def _evaluate_operand_controls(
    model: Any, operands: _PreparedOperands
) -> dict[str, np.ndarray]:
    configurations = (
        (
            "raw_phi_raw_omega",
            operands.raw_phi_stencil,
            operands.raw_omega_stencil,
            operands.raw_phi_trace,
            operands.raw_omega_trace,
            operands.raw_omega_halo,
            None,
        ),
        (
            "H_phi_raw_omega",
            operands.phi_stencil,
            operands.raw_omega_stencil,
            operands.phi_trace,
            operands.raw_omega_trace,
            operands.raw_omega_halo,
            None,
        ),
        (
            "raw_phi_H_omega",
            operands.raw_phi_stencil,
            operands.omega_stencil,
            operands.raw_phi_trace,
            operands.omega_trace,
            operands.omega_halo,
            operands.omega_direct_states,
        ),
        (
            "H_phi_H_omega",
            operands.phi_stencil,
            operands.omega_stencil,
            operands.phi_trace,
            operands.omega_trace,
            operands.omega_halo,
            operands.omega_direct_states,
        ),
    )

    def evaluate():
        fine_actions = tuple(
            _operator_call(
                model,
                f_stencil,
                g_stencil,
                f_trace=f_trace,
                g_trace=g_trace,
                characteristic_scheme="scalar-third-order-upwind",
                g_halo=g_halo,
                g_direct_states=direct,
            )
            for (
                _name,
                f_stencil,
                g_stencil,
                f_trace,
                g_trace,
                g_halo,
                direct,
            ) in configurations
        )
        # Operand controls diagnose H on the input side while retaining the
        # production Poisson return map.  The H.T comparison is reported
        # separately for the five primary bracket formulations.
        return tuple(model._restrict_fine_field(value) for value in fine_actions)

    values = jax.jit(evaluate)()
    values = jax.block_until_ready(values)
    return {
        configuration[0]: np.asarray(value)
        for configuration, value in zip(configurations, values)
    }


def _constant_and_fallback_contracts(
    model: Any, operands: _PreparedOperands
) -> dict[str, Any]:
    geometry = model.geometry
    domain = model.domain
    constant_owner = jnp.where(
        model.control_volume_geometry.cells.is_active_owner,
        1.0,
        0.0,
    )
    face_bc = model._face_bcs(
        blob.FciDrbEBState(
            density=constant_owner,
            phi=constant_owner,
            Te=constant_owner,
            Ti=constant_owner,
            Vi=constant_owner,
            Ve=constant_owner,
            vorticity=constant_owner,
        )
    )
    constant_halo = model._prepare_poisson_bracket_support_halo(
        constant_owner, face_bc.vorticity
    )
    constant_trace_halo = model._prepare_poisson_bracket_halo(
        constant_owner, face_bc.vorticity
    )
    constant_trace = build_local_boundary_face_trace_from_halo(
        constant_trace_halo, geometry, domain, face_bc.vorticity
    )
    context = StencilBuilderContext(layout=domain.layout, domain=domain)
    constant_stencil, constant_left, constant_right = (
        build_local_control_volume_poisson_face_stencil(
            constant_halo,
            geometry,
            domain,
            context,
            model.control_volume_geometry,
            model.control_volume_boundary_bc,
            owner_values_owned=constant_owner,
            regular_face_bc=face_bc.vorticity,
            boundary_trace=constant_trace,
            halo_exchange=model.halo_exchange,
            topology_filler=model.topology_filler,
        )
    )
    constant_actions = jax.jit(
        lambda: _return_actions_to_owner(
            model,
            (_operator_call(
            model,
            operands.phi_stencil,
            constant_stencil,
            f_trace=operands.phi_trace,
            g_trace=constant_trace,
            characteristic_scheme="scalar-third-order-upwind",
            g_halo=constant_halo,
            g_direct_states=(constant_left, constant_right),
            ),),
        )
    )()
    constant_actions = jax.block_until_ready(constant_actions)
    constant_production = np.asarray(constant_actions[0][0])
    constant_h_transpose = np.asarray(constant_actions[1][0])

    _left, _right, fallback = _third_order_scalar_face_states_from_halo(
        operands.omega_halo,
        geometry,
        boundary_trace=operands.omega_trace,
        axis_regular_axes=model.axis_regular_axes,
        positivity_floor=None,
    )
    fallback_axes = [
        np.array(np.asarray(value), copy=True)
        for value in (fallback.x, fallback.y, fallback.z)
    ]
    direct = build_local_control_volume_direct_face_states(
        operands.omega_owner,
        geometry,
        domain,
        model.control_volume_geometry,
        halo_exchange=model.halo_exchange,
        topology_filler=model.topology_filler,
    )
    faces = model.control_volume_geometry.irregular_faces
    rows = model.control_volume_geometry.face_functionals
    quadrature_active = np.asarray(faces.quadrature_active, dtype=bool)
    measure = np.where(
        quadrature_active,
        np.linalg.norm(np.asarray(faces.area_covector_weight), axis=-1),
        0.0,
    )
    measure_sum = np.sum(measure, axis=(1, 2))
    fi = np.asarray(faces.logical_face_i, dtype=np.int64)
    profile = np.asarray(model.control_volume_geometry.angular_group_sizes)
    safe = np.clip(fi, 1, profile.size - 1)
    transition = (
        (fi > 0)
        & (fi < profile.size)
        & (profile[safe - 1] != profile[safe])
    )
    row_valid = (
        np.asarray(rows.active, dtype=bool)
        & np.asarray(faces.active, dtype=bool)
        & (np.asarray(faces.logical_axis) == 0)
        & (measure_sum > 0.0)
        & np.all(
            np.asarray(direct.valid, dtype=bool) | ~quadrature_active,
            axis=(1, 2),
        )
        & transition
    )
    selected_rows = (
        np.asarray(rows.active, dtype=bool)
        & np.asarray(faces.active, dtype=bool)
        & (np.asarray(faces.logical_axis) == 0)
    )
    face_index = (
        np.asarray(faces.logical_face_i, dtype=np.int64)[selected_rows],
        np.asarray(faces.logical_face_j, dtype=np.int64)[selected_rows],
        np.asarray(faces.logical_face_k, dtype=np.int64)[selected_rows],
    )
    fallback_axes[0][face_index] = ~row_valid[selected_rows]
    fallback_count = sum(int(np.count_nonzero(value)) for value in fallback_axes)
    face_count = sum(int(value.size) for value in fallback_axes)
    transition_rows = transition & selected_rows
    transition_count = int(np.count_nonzero(transition_rows))
    return {
        "constant_advected_state_max_abs": float(
            np.max(np.abs(constant_production))
        ),
        "mass_weighted_H_transpose_constant_advected_state_max_abs": float(
            np.max(np.abs(constant_h_transpose))
        ),
        "upwind_fallback_fraction": float(fallback_count / max(face_count, 1)),
        "transition_direct_invalid_fraction": (
            0.0
            if transition_count == 0
            else float(
                np.count_nonzero(transition_rows & ~row_valid) / transition_count
            )
        ),
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    resolution = int(args.resolution)
    artifact_path, artifact = _load_artifact(args.geometry, resolution)
    _reference_path, reference_artifact = _load_artifact(
        args.geometry, int(args.reference_resolution)
    )
    reference = build_continuum_reference_from_artifact(
        reference_artifact,
        tau=mms.PHYSICAL_PARAMETERS["tau"],
        mi_over_me=mms.PHYSICAL_PARAMETERS["mi_over_me"],
        rho_star=mms.PHYSICAL_PARAMETERS["rho_star"],
        Ve_nu=mms.PHYSICAL_PARAMETERS["Ve_nu"],
        perp_diffusion=mms.PHYSICAL_PARAMETERS["density_D_perp"],
        enable_generalized_potential=True,
    )
    runtime_args = SimpleNamespace(
        shard_counts=(1, 1, 1),
        curvature_edge_one_form=False,
        reference=reference,
        metric_context=SimpleNamespace(
            metric_evaluator=reference.metric_evaluator,
            bfield=reference.bfield_evaluator,
            nfp=int(reference_artifact.nfp),
        ),
    )
    geometry = artifact.global_geometry
    host = artifact.owner_geometry
    # LocalFciDrbEBRhs reads these static construction choices from the
    # environment.  Bind the frozen Stage-7 contract only while constructing
    # the support object, then restore the caller's environment.  None of the
    # associated parallel or short-leg operators are evaluated by this audit.
    with patch.dict(os.environ, _FROZEN_RUNTIME_ENV, clear=False):
        runtime = mms._runtime(geometry, host, runtime_args)
    if runtime.model is None:
        raise RuntimeError("operator audit failed to construct its single-device model")
    projector = mms._QuadratureProjector(reference, geometry, host)
    raw_state, raw_exact = _poisson_reference_fields(
        projector,
        float(args.time),
        rho_star=float(runtime.model.parameters.rho_star),
    )
    owner_state = mms._owner_project(raw_state, host)
    exact = mms._owner_project_array(raw_exact, host)
    operands = _prepare_operands(runtime.model, owner_state, raw_state)
    returned_variants = _evaluate_variants(runtime.model, operands)
    production_raw = returned_variants["production_R"]
    adjoint_raw = returned_variants["mass_weighted_H_transpose"]
    rho_star = float(runtime.model.parameters.rho_star)

    def effective(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "phi_centered_action": -raw["phi_centered"] / rho_star,
            "omega_centered_reverse_action": raw["omega_centered"] / rho_star,
            "current_centered_antisymmetric": -raw["centered"] / rho_star,
            "pure_scalar_upwind": -raw["pure_upwind"] / rho_star,
            "compatible_upwind": -raw["compatible_upwind"] / rho_star,
        }

    production_effective = effective(production_raw)
    adjoint_effective = effective(adjoint_raw)
    masks = _region_masks(geometry, host)
    include_rings = bool(args.radial_rings)
    variants = {
        name: _variant_statistics(
            value, exact, host, masks, include_rings=include_rings
        )
        for name, value in production_effective.items()
    }
    adjoint_variants = {
        name: _variant_statistics(
            value, exact, host, masks, include_rings=include_rings
        )
        for name, value in adjoint_effective.items()
    }
    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    volume = np.asarray(host.aggregate_chart_volume, dtype=np.float64)
    total_volume = float(np.sum(volume[active]))
    contracts = _constant_and_fallback_contracts(runtime.model, operands)
    contracts.update(
        {
            "centered_antisymmetry_max_abs": float(
                np.max(
                    np.abs(
                        production_raw["centered"]
                        + production_raw["centered_swapped"]
                    )
                )
            ),
            "centered_decomposition_max_abs": float(
                np.max(
                    np.abs(
                        production_raw["centered"]
                        - 0.5
                        * (
                            production_raw["phi_centered"]
                            - production_raw["omega_centered"]
                        )
                    )
                )
            ),
            "volume_integrals": {
                name: float(np.sum(volume[active] * value[active]))
                for name, value in production_effective.items()
            },
            "volume_integrated_errors": {
                name: float(np.sum(volume[active] * (value - exact)[active]))
                for name, value in production_effective.items()
            },
            "mass_weighted_H_transpose_centered_antisymmetry_max_abs": float(
                np.max(
                    np.abs(
                        adjoint_raw["centered"]
                        + adjoint_raw["centered_swapped"]
                    )
                )
            ),
            "mass_weighted_H_transpose_centered_decomposition_max_abs": float(
                np.max(
                    np.abs(
                        adjoint_raw["centered"]
                        - 0.5
                        * (
                            adjoint_raw["phi_centered"]
                            - adjoint_raw["omega_centered"]
                        )
                    )
                )
            ),
            "mass_weighted_H_transpose_volume_integrals": {
                name: float(np.sum(volume[active] * value[active]))
                for name, value in adjoint_effective.items()
            },
            "mass_weighted_H_transpose_volume_integrated_errors": {
                name: float(np.sum(volume[active] * (value - exact)[active]))
                for name, value in adjoint_effective.items()
            },
            "total_active_volume": total_volume,
        }
    )
    operand_controls = None
    if bool(args.include_operand_controls):
        controls = _evaluate_operand_controls(runtime.model, operands)
        operand_controls = {
            name: _variant_statistics(
                -value / rho_star,
                exact,
                host,
                masks,
                include_rings=include_rings,
            )
            for name, value in controls.items()
        }
    payload = {
        "schema": SCHEMA,
        "resolution": resolution,
        "reference_resolution": int(args.reference_resolution),
        "time": float(args.time),
        "geometry_artifact": str(artifact_path),
        "execution": "single-device-operator-only",
        "operator_actions": {
            "production": "R A_f(H phi, H omega)",
            "diagnostic": "M_owner^-1 H.T M_raw A_f(H phi, H omega)",
        },
        "angular_group_profile": np.asarray(host.angular_group_size),
        "variants": variants,
        "mass_weighted_H_transpose_variants": adjoint_variants,
        "contracts": contracts,
        "operand_controls": operand_controls,
    }
    _write_json(args.output, payload)
    return payload


def _convergence_table(
    ordered: Sequence[Mapping[str, Any]],
    resolutions: Sequence[int],
    *,
    variants_key: str,
) -> dict[str, Any]:
    convergence: dict[str, Any] = {}
    for variant in VARIANTS:
        regions = {}
        for region in PRIMARY_REGIONS:
            absolute = [
                item[variants_key][variant][region]["absolute_l2"]
                for item in ordered
            ]
            relative = [
                item[variants_key][variant][region]["relative_l2"]
                for item in ordered
            ]
            regions[region] = {
                "absolute_l2": absolute,
                "relative_l2": relative,
                "absolute_order": _orders(resolutions, absolute),
                "relative_order": _orders(resolutions, relative),
            }
        convergence[variant] = regions
    return convergence


def summarize_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(cases, key=lambda item: int(item["resolution"]))
    if not ordered:
        raise ValueError("at least one case is required")
    if any(item.get("schema") != SCHEMA for item in ordered):
        raise ValueError("all case files must use the current audit schema")
    resolutions = [int(item["resolution"]) for item in ordered]
    convergence = _convergence_table(
        ordered,
        resolutions,
        variants_key="variants",
    )
    adjoint_convergence = _convergence_table(
        ordered,
        resolutions,
        variants_key="mass_weighted_H_transpose_variants",
    )

    pure_transition = convergence["pure_scalar_upwind"][
        "rlp_transition_rings"
    ]["relative_order"]
    pure_ordinary = convergence["pure_scalar_upwind"]["ordinary_bulk"][
        "relative_order"
    ]
    constant_max = max(
        float(item["contracts"]["constant_advected_state_max_abs"])
        for item in ordered
    )
    transition_pass = len(pure_transition) >= 2 and all(
        value is not None and value >= 1.8 for value in pure_transition
    )
    ordinary_pass = bool(pure_ordinary) and pure_ordinary[-1] is not None and (
        pure_ordinary[-1] >= 1.8
    )
    constant_pass = constant_max <= 1.0e-11
    adjoint_transition = adjoint_convergence["pure_scalar_upwind"][
        "rlp_transition_rings"
    ]["relative_order"]
    adjoint_ordinary = adjoint_convergence["pure_scalar_upwind"][
        "ordinary_bulk"
    ]["relative_order"]
    adjoint_constant_max = max(
        float(
            item["contracts"][
                "mass_weighted_H_transpose_constant_advected_state_max_abs"
            ]
        )
        for item in ordered
    )
    adjoint_transition_pass = len(adjoint_transition) >= 2 and all(
        value is not None and value >= 1.8 for value in adjoint_transition
    )
    adjoint_ordinary_pass = (
        bool(adjoint_ordinary)
        and adjoint_ordinary[-1] is not None
        and adjoint_ordinary[-1] >= 1.8
    )
    adjoint_constant_pass = adjoint_constant_max <= 1.0e-11
    operand_controls_required = not transition_pass
    controls_present = {
        int(item["resolution"]): item.get("operand_controls") is not None
        for item in ordered
    }
    control_cases = [
        item for item in ordered if item.get("operand_controls") is not None
    ]
    operand_control_convergence: dict[str, Any] = {}
    if len(control_cases) >= 2:
        control_resolutions = [int(item["resolution"]) for item in control_cases]
        for control in OPERAND_CONTROLS:
            regions = {}
            for region in PRIMARY_REGIONS:
                relative = [
                    item["operand_controls"][control][region]["relative_l2"]
                    for item in control_cases
                ]
                regions[region] = {
                    "relative_l2": relative,
                    "relative_order": _orders(control_resolutions, relative),
                }
            operand_control_convergence[control] = regions
    controls_complete = (
        not operand_controls_required
        or (
            bool(controls_present.get(min(resolutions), False))
            and bool(controls_present.get(max(resolutions), False))
        )
    )

    diagnoses = []
    if not transition_pass and adjoint_transition_pass:
        diagnoses.append(
            "the mass-weighted H-transpose return map recovers RLP-transition "
            "order while production R does not; the owner test/restriction "
            "map is the leading suspect"
        )
    if transition_pass and not adjoint_transition_pass:
        diagnoses.append(
            "production R passes the RLP-transition screen while the "
            "mass-weighted H-transpose return map does not"
        )
    compatible_transition = convergence["compatible_upwind"][
        "rlp_transition_rings"
    ]["relative_order"]
    compatible_pass = len(compatible_transition) >= 2 and all(
        value is not None and value >= 1.8 for value in compatible_transition
    )
    if transition_pass and not compatible_pass:
        diagnoses.append(
            "pure scalar upwind passes while compatible upwind fails; the "
            "retained centered antisymmetric core is the leading suspect"
        )
    phi_transition = convergence["phi_centered_action"][
        "rlp_transition_rings"
    ]["relative_order"]
    omega_transition = convergence["omega_centered_reverse_action"][
        "rlp_transition_rings"
    ]["relative_order"]
    phi_pass = len(phi_transition) >= 2 and all(
        value is not None and value >= 1.8 for value in phi_transition
    )
    omega_pass = len(omega_transition) >= 2 and all(
        value is not None and value >= 1.8 for value in omega_transition
    )
    if phi_pass and not omega_pass:
        diagnoses.append(
            "A_phi(omega) passes while A_omega(phi) fails; vorticity used as "
            "the reconstructed generator is the leading suspect"
        )
    if not transition_pass and not phi_pass:
        diagnoses.append(
            "centered and upwind A_phi(omega) remain transition-limited; "
            "inspect the phi-generated face velocity with operand controls"
        )
    if operand_control_convergence:
        def control_pass(name: str) -> bool:
            values = operand_control_convergence[name][
                "rlp_transition_rings"
            ]["relative_order"]
            return bool(values) and all(
                value is not None and value >= 1.8 for value in values
            )

        raw_raw_pass = control_pass("raw_phi_raw_omega")
        h_h_pass = control_pass("H_phi_H_omega")
        h_phi_raw_omega_pass = control_pass("H_phi_raw_omega")
        raw_phi_h_omega_pass = control_pass("raw_phi_H_omega")
        if raw_raw_pass and not h_h_pass:
            diagnoses.append(
                "raw operands recover transition order while H/H does not; "
                "the RLP operand representation is responsible"
            )
        if raw_phi_h_omega_pass and not h_phi_raw_omega_pass:
            diagnoses.append(
                "raw phi with reconstructed omega recovers order while the "
                "opposite control does not; the phi generator representation "
                "is the leading suspect"
            )
        if h_phi_raw_omega_pass and not raw_phi_h_omega_pass:
            diagnoses.append(
                "reconstructed phi with raw omega recovers order while the "
                "opposite control does not; the omega advected-state "
                "representation is the leading suspect"
            )
        if not raw_raw_pass:
            diagnoses.append(
                "raw/raw remains below order in the transition footprint; "
                "the next diagnostic is an exact face-quadrature oracle for "
                "geometry/incidence and continuum-reference consistency"
            )

    return {
        "schema": SCHEMA,
        "resolutions": resolutions,
        "convergence": convergence,
        "mass_weighted_H_transpose_convergence": adjoint_convergence,
        "qualification": {
            "pure_scalar_upwind_transition_pass": transition_pass,
            "pure_scalar_upwind_asymptotic_ordinary_pass": ordinary_pass,
            "constant_preservation_pass": constant_pass,
            "constant_advected_state_max_abs": constant_max,
            "overall_pass": transition_pass and ordinary_pass and constant_pass,
            "mass_weighted_H_transpose": {
                "pure_scalar_upwind_transition_pass": adjoint_transition_pass,
                "pure_scalar_upwind_asymptotic_ordinary_pass": (
                    adjoint_ordinary_pass
                ),
                "constant_preservation_pass": adjoint_constant_pass,
                "constant_advected_state_max_abs": adjoint_constant_max,
                "overall_pass": (
                    adjoint_transition_pass
                    and adjoint_ordinary_pass
                    and adjoint_constant_pass
                ),
            },
        },
        "operand_controls_required": operand_controls_required,
        "operand_controls_complete": controls_complete,
        "operand_controls_present": controls_present,
        "operand_control_convergence": operand_control_convergence,
        "diagnoses": diagnoses,
        "cases": ordered,
    }


def merge_cases(args: argparse.Namespace) -> dict[str, Any]:
    summary = summarize_cases([_load_json(path) for path in args.inputs])
    _write_json(args.output, summary)
    return summary


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    resolutions = tuple(int(value) for value in args.resolutions)

    def run_one(resolution: int, *, controls: bool) -> Path:
        suffix = ".controls" if controls else ""
        path = output / f"N{resolution}{suffix}.json"
        command = [
            sys.executable,
            str(script),
            "case",
            "--geometry",
            str(args.geometry),
            "--resolution",
            str(resolution),
            "--reference-resolution",
            str(args.reference_resolution),
            "--time",
            str(args.time),
            "--output",
            str(path),
        ]
        if resolution == max(resolutions):
            command.append("--radial-rings")
        if controls:
            command.append("--include-operand-controls")
        subprocess.run(command, check=True)
        return path

    paths = [run_one(resolution, controls=False) for resolution in resolutions]
    cases = [_load_json(path) for path in paths]
    provisional = summarize_cases(cases)
    controls_mode = str(args.operand_controls)
    need_controls = controls_mode == "always" or (
        controls_mode == "auto" and provisional["operand_controls_required"]
    )
    if need_controls:
        by_resolution = {int(case["resolution"]): case for case in cases}
        for resolution in (min(resolutions), max(resolutions)):
            control_path = run_one(resolution, controls=True)
            by_resolution[resolution] = _load_json(control_path)
        cases = [by_resolution[resolution] for resolution in resolutions]
    summary = summarize_cases(cases)
    summary["campaign"] = {
        "operand_controls_mode": controls_mode,
        "operand_controls_executed": need_controls,
        "one_resolution_per_process": True,
        "geometry_policy": "artifact-only-no-rebuild",
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    print(summary_path)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    case = subparsers.add_parser("case", help="run one resolution")
    case.add_argument("--geometry", type=Path, required=True)
    case.add_argument("--resolution", type=int, required=True)
    case.add_argument("--reference-resolution", type=int, default=64)
    case.add_argument("--time", type=float, default=1.0e-6)
    case.add_argument("--output", type=Path, required=True)
    case.add_argument("--radial-rings", action="store_true")
    case.add_argument("--include-operand-controls", action="store_true")
    case.set_defaults(handler=run_case)

    merge = subparsers.add_parser("merge", help="merge completed case JSON files")
    merge.add_argument("inputs", type=Path, nargs="+")
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(handler=merge_cases)

    campaign = subparsers.add_parser(
        "campaign", help="run isolated resolution processes and merge them"
    )
    campaign.add_argument("--geometry", type=Path, required=True)
    campaign.add_argument("--resolutions", type=int, nargs="+", default=(32, 48, 64))
    campaign.add_argument("--reference-resolution", type=int, default=64)
    campaign.add_argument("--time", type=float, default=1.0e-6)
    campaign.add_argument("--output", type=Path, required=True)
    campaign.add_argument(
        "--operand-controls",
        choices=("auto", "always", "never"),
        default="auto",
    )
    campaign.set_defaults(handler=run_campaign)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
