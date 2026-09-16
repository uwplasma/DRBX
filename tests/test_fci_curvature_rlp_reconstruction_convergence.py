"""Controlled P/H transition-face comparisons for production curvature."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drbx.geometry import (
    LocalCurvatureFaceCoefficients3D,
    build_local_conservative_stencil_from_field,
)
from drbx.native.fci_boundaries import LocalControlVolumeBoundaryBC3D
from drbx.native.fci_drb_EB_rhs import LocalFciDrbEBRhs
from drbx.native.fci_curvature_production_flux import (
    curvature_face_linearized_fluctuations,
)
from drbx.native.fci_model import inject_owned_field_to_halo
from drbx.native.fci_operators import (
    _third_order_scalar_face_states_from_halo,
    aggregate_local_control_volume_average,
    build_local_control_volume_poisson_face_stencil,
    build_local_control_volume_direct_face_states,
    build_local_radial_curvature_face_states,
    patch_local_control_volume_direct_face_values,
    patch_local_radial_curvature_face_values,
    expand_local_control_volume_owner_field,
    local_curvature_conservative_op,
    local_curvature_production_path_op,
)
from test_fci_poisson_rlp_reconstruction_convergence import _setup


def _polynomial_average(centroid, second_moment, coefficients):
    """Return the exact cell average of a quadratic chart-space field."""

    base, ax, ay, axx, axy, ayy = coefficients
    x = centroid[..., 0]
    y = centroid[..., 1]
    return (
        base
        + ax * x
        + ay * y
        + axx * (x * x + second_moment[..., 0, 0])
        + axy * (x * y + second_moment[..., 0, 1])
        + ayy * (y * y + second_moment[..., 1, 1])
    )


_STATE_COEFFICIENTS = (
    (1.0, 0.05, -0.03, 0.02, 0.01, -0.015),
    (1.0, -0.04, 0.025, 0.012, -0.008, 0.01),
    (1.0, 0.03, 0.02, -0.01, 0.014, 0.009),
    (0.0, 0.02, -0.015, 0.006, 0.004, -0.005),
)


def _smooth_state(cells, *, aggregate):
    centroid = cells.centroid if aggregate else cells.raw_centroid
    second = cells.second_moment if aggregate else cells.raw_second_moment
    values = tuple(
        _polynomial_average(centroid, second, value)
        for value in _STATE_COEFFICIENTS
    )
    if aggregate:
        values = tuple(
            jnp.where(cells.is_active_owner, value, 0.0) for value in values
        )
    return values


def _exact_radial_face_average(geometry, coefficients):
    """High-order transverse average of one chart polynomial on x faces."""

    radial = np.asarray(geometry.grid.x.faces_owned)
    theta = np.asarray(geometry.grid.y.centers_owned)
    dtheta = float(geometry.grid.y.faces_owned[1] - geometry.grid.y.faces_owned[0])
    nodes, weights = np.polynomial.legendre.leggauss(8)
    result = np.zeros((radial.size, theta.size), dtype=np.float64)
    for node, weight in zip(nodes, weights):
        angle = theta[None, :] + 0.5 * dtheta * node
        x = radial[:, None] * np.cos(angle)
        y = radial[:, None] * np.sin(angle)
        point = np.stack((x, y, np.zeros_like(x)), axis=-1)
        result += 0.5 * weight * np.asarray(
            _polynomial_average(
                jnp.asarray(point),
                jnp.zeros(point.shape[:-1] + (3, 3), dtype=jnp.float64),
                coefficients,
            )
        )
    return jnp.broadcast_to(
        jnp.asarray(result)[..., None], geometry.layout.face_control_shape(0)
    )


def _radial_curvature_coefficients(geometry):
    layout = geometry.layout
    radial = jnp.asarray(geometry.grid.x.faces_owned, dtype=jnp.float64)
    radial = jnp.broadcast_to(
        radial[:, None, None], layout.face_control_shape(0)
    )
    return LocalCurvatureFaceCoefficients3D(
        layout=layout,
        # A regular polar radial flux must vanish at the axis.  Q^r=r is the
        # simplest nontrivial smooth coefficient with that property.
        x=radial,
        y=jnp.zeros(layout.face_control_shape(1), dtype=jnp.float64),
        z=jnp.zeros(layout.face_control_shape(2), dtype=jnp.float64),
    )


def _relative_l2(actual, reference, weights, active):
    numerator = jnp.sum(
        jnp.where(
            active[..., None],
            weights[..., None] * (actual - reference) ** 2,
            0.0,
        )
    )
    denominator = jnp.sum(
        jnp.where(active[..., None], weights[..., None] * reference**2, 0.0)
    )
    return float(jnp.sqrt(numerator / jnp.maximum(denominator, 1.0e-30)))


def _scalar_relative_l2(actual, reference, weights, active):
    numerator = jnp.sum(jnp.where(active, weights * (actual - reference) ** 2, 0.0))
    denominator = jnp.sum(jnp.where(active, weights * reference**2, 0.0))
    return float(jnp.sqrt(numerator / jnp.maximum(denominator, 1.0e-30)))


def _transition_owner_mask(control_volume):
    cells = control_volume.cells
    faces = control_volume.irregular_faces
    face_i = np.asarray(faces.logical_face_i)
    group_size = np.asarray(control_volume.angular_group_sizes)
    active = (
        np.asarray(faces.active, dtype=bool)
        & (face_i > 0)
        & (face_i < group_size.size)
    )
    active &= group_size[np.clip(face_i - 1, 0, group_size.size - 1)] != group_size[
        np.clip(face_i, 0, group_size.size - 1)
    ]
    mask = np.zeros(cells.shape, dtype=bool)
    for prefix in ("minus", "plus"):
        mask[
            np.asarray(getattr(faces, f"{prefix}_owner_i"))[active],
            np.asarray(getattr(faces, f"{prefix}_owner_j"))[active],
            np.asarray(getattr(faces, f"{prefix}_owner_k"))[active],
        ] = True
    return jnp.asarray(mask) & cells.is_active_owner


def _compact_owner_mask(control_volume):
    """Owners touched by any active radial compact row, including the band."""

    cells = control_volume.cells
    faces = control_volume.irregular_faces
    active = np.asarray(faces.active, dtype=bool) & (
        np.asarray(faces.logical_axis, dtype=np.int32) == 0
    )
    mask = np.zeros(cells.shape, dtype=bool)
    for prefix in ("minus", "plus"):
        mask[
            np.asarray(getattr(faces, f"{prefix}_owner_i"))[active],
            np.asarray(getattr(faces, f"{prefix}_owner_j"))[active],
            np.asarray(getattr(faces, f"{prefix}_owner_k"))[active],
        ] = True
    return jnp.asarray(mask) & cells.is_active_owner


def _two_face_compact_owner_mask(control_volume):
    """Owners whose raw cells have both radial compact bounding faces."""

    cells = control_volume.cells
    faces = control_volume.irregular_faces
    logical_axis = np.asarray(faces.logical_axis, dtype=np.int32)
    active = np.asarray(faces.active, dtype=bool) & (logical_axis == 0)
    face_active = np.zeros(
        (cells.shape[0] + 1, cells.shape[1], cells.shape[2]), dtype=bool
    )
    for i, j, k in zip(
        np.asarray(faces.logical_face_i)[active],
        np.asarray(faces.logical_face_j)[active],
        np.asarray(faces.logical_face_k)[active],
    ):
        face_active[i, j, k] = True
    raw_valid = face_active[:-1] & face_active[1:]
    owner_mask = np.zeros(cells.shape, dtype=bool)
    owner_mask[
        np.asarray(cells.owner_i)[raw_valid],
        np.asarray(cells.owner_j)[raw_valid],
        np.asarray(cells.owner_k)[raw_valid],
    ] = True
    return jnp.asarray(owner_mask) & cells.is_active_owner


def _radial_direct_coverage(control_volume):
    """Summarize radial rows and expose any interior one-face seam owners."""

    cells = control_volume.cells
    lean_rows = control_volume.radial_curvature_faces
    if lean_rows is not None:
        rows = lean_rows
        active = np.asarray(rows.active, dtype=bool)
        logical_face_i = np.asarray(rows.logical_face_i)
        logical_face_j = np.asarray(rows.logical_face_j)
        logical_face_k = np.asarray(rows.logical_face_k)
        weights = (
            rows.centered_weights,
            rows.minus_weights,
            rows.plus_weights,
        )
    else:
        faces = control_volume.irregular_faces
        rows = control_volume.face_functionals
        active = np.asarray(faces.active, dtype=bool) & (
            np.asarray(faces.logical_axis, dtype=np.int32) == 0
        )
        logical_face_i = np.asarray(faces.logical_face_i)
        logical_face_j = np.asarray(faces.logical_face_j)
        logical_face_k = np.asarray(faces.logical_face_k)
        weights = (
            rows.value_weights,
            rows.upwind_minus_value_weights,
            rows.upwind_plus_value_weights,
        )
    face_active = np.zeros(
        (cells.shape[0] + 1, cells.shape[1], cells.shape[2]), dtype=bool
    )
    face_active[
        logical_face_i[active],
        logical_face_j[active],
        logical_face_k[active],
    ] = True
    face_count = face_active[:-1].astype(np.int32) + face_active[1:]
    one_face = np.zeros(cells.shape, dtype=bool)
    one_face[1:-1] = face_count[1:-1] == 1
    missing = np.zeros(cells.shape, dtype=bool)
    missing[1:-1] = face_count[1:-1] < 2
    owner_mask = np.zeros(cells.shape, dtype=bool)
    owner_mask[
        np.asarray(cells.owner_i)[one_face],
        np.asarray(cells.owner_j)[one_face],
        np.asarray(cells.owner_k)[one_face],
    ] = True
    weight_entries = sum(np.asarray(value).size for value in weights)
    weight_l1_max = max(
        float(np.max(np.sum(np.abs(np.asarray(value)), axis=-1)))
        for value in weights
    )
    return {
        "rows": int(rows.max_rows),
        "active_observations": int(np.sum(np.asarray(rows.observation_active))),
        "weight_entries": int(weight_entries),
        "weight_l1_max": weight_l1_max,
        "interior_one_face_raw": int(np.sum(one_face)),
        "interior_missing_raw": int(np.sum(missing)),
        "interior_one_face_owners": int(
            np.sum(owner_mask & np.asarray(cells.is_active_owner))
        ),
        "one_face_owner_mask": jnp.asarray(owner_mask) & cells.is_active_owner,
    }


def test_radial_direct_coverage_accepts_lean_only_descriptor():
    cv = _setup(8, radial_curvature_faces=True)[4]
    lean = cv.radial_curvature_faces

    assert cv.face_functionals is None
    assert lean is not None
    coverage = _radial_direct_coverage(cv)
    assert coverage["rows"] == lean.max_rows
    assert coverage["active_observations"] == int(
        np.sum(np.asarray(lean.observation_active))
    )
    assert coverage["weight_entries"] == sum(
        np.asarray(value).size
        for value in (
            lean.centered_weights,
            lean.minus_weights,
            lean.plus_weights,
        )
    )
    assert coverage["weight_l1_max"] <= 8.0
    assert coverage["interior_one_face_raw"] == 0
    assert coverage["interior_missing_raw"] == 0
    assert coverage["interior_one_face_owners"] == 0


def _direct_quadrature_states(control_volume, owner_values):
    """Evaluate centered/minus/plus compact-face traces without averaging."""

    rows = control_volume.face_functionals
    observations = jnp.stack(
        tuple(
            value[
                jnp.clip(rows.owned_i, 0, value.shape[0] - 1),
                jnp.clip(rows.owned_j, 0, value.shape[1] - 1),
                jnp.clip(rows.owned_k, 0, value.shape[2] - 1),
            ]
            for value in owner_values
        ),
        axis=-1,
    )
    observations = jnp.where(
        rows.observation_active[..., None], observations, 0.0
    )
    return tuple(
        jnp.einsum("rpqe,res->rpqs", weights, observations)
        for weights in (
            rows.value_weights,
            rows.upwind_minus_value_weights,
            rows.upwind_plus_value_weights,
        )
    )


def _quadrature_material_action(control_volume, minus, plus, *, tau):
    """Apply the radial within-cell material path at face quadrature."""

    cells = control_volume.cells
    faces = control_volume.irregular_faces
    face_shape = cells.layout.face_control_shape(0)
    row_for_face = jnp.full(face_shape, -1, dtype=jnp.int32).at[
        faces.logical_face_i,
        faces.logical_face_j,
        faces.logical_face_k,
    ].set(jnp.arange(faces.max_rows, dtype=jnp.int32))
    left_row = row_for_face[:-1]
    right_row = row_for_face[1:]
    left_clipped = jnp.clip(left_row, 0, max(faces.max_rows - 1, 0))
    right_clipped = jnp.clip(right_row, 0, max(faces.max_rows - 1, 0))

    # At a cell's lower radial face its interior trace is the plus-side fit;
    # at its upper face it is the minus-side fit.  Rows use the same tensor
    # Gauss nodes in theta/eta, so corresponding quadrature points define the
    # transverse path through the raw cell.
    cell_left = plus[left_clipped]
    cell_right = minus[right_clipped]
    cell_i = jnp.broadcast_to(
        jnp.arange(cells.shape[0], dtype=jnp.int32)[:, None, None], cells.shape
    )
    axis_cell = (cell_i == 0) & (left_row < 0) & (right_row >= 0)
    equilibrium = jnp.asarray(
        tuple(value[0] for value in _STATE_COEFFICIENTS), dtype=jnp.float64
    )
    cell_left = jnp.where(
        axis_cell[..., None, None, None], equilibrium, cell_left
    )
    left_measure = jnp.where(
        (left_row >= 0)[..., None, None]
        & faces.quadrature_active[left_clipped],
        jnp.linalg.norm(faces.area_covector_weight[left_clipped], axis=-1),
        0.0,
    )
    right_measure = jnp.where(
        faces.quadrature_active[right_clipped],
        jnp.linalg.norm(faces.area_covector_weight[right_clipped], axis=-1),
        0.0,
    )
    measure = 0.5 * (left_measure + right_measure)
    radial = jnp.linalg.norm(
        faces.quadrature_points[..., :2], axis=-1
    )
    left_radial = jnp.where(
        (left_row >= 0)[..., None, None], radial[left_clipped], 0.0
    )
    cell_normal = 0.5 * (
        left_radial + radial[right_clipped]
    )
    valid = (
        (((left_row >= 0) & (right_row >= 0)) | axis_cell)
        & jnp.any(measure > 0.0, axis=(-2, -1))
    )
    dplus, dminus = curvature_face_linearized_fluctuations(
        cell_left,
        cell_right,
        0.5 * (cell_left + cell_right),
        1.0,
        tau,
        normal=cell_normal,
    )
    raw_integrated = -jnp.sum(
        measure[..., None] * (dplus + dminus), axis=(-3, -2)
    )
    raw_integrated = jnp.where(valid[..., None], raw_integrated, 0.0)
    owner_integrated = jnp.zeros(cells.shape + (4,), dtype=jnp.float64).at[
        cells.owner_i, cells.owner_j, cells.owner_k
    ].add(raw_integrated)
    owner_update = jnp.where(
        cells.is_active_owner[..., None],
        owner_integrated
        / jnp.maximum(cells.aggregate_volume[..., None], 1.0e-30),
        0.0,
    )
    return owner_update[cells.owner_i, cells.owner_j, cells.owner_k]


def _fitted_quadrature_material_action(control_volume, owner_values, *, tau):
    _centered, minus, plus = _direct_quadrature_states(
        control_volume, owner_values
    )
    return _quadrature_material_action(
        control_volume, minus, plus, tau=tau
    )


def _exact_quadrature_material_action(control_volume, *, tau):
    points = control_volume.irregular_faces.quadrature_points
    exact = jnp.stack(
        tuple(
            _polynomial_average(
                points,
                jnp.zeros(points.shape[:-1] + (3, 3), dtype=jnp.float64),
                coefficients,
            )
            for coefficients in _STATE_COEFFICIENTS
        ),
        axis=-1,
    )
    return _quadrature_material_action(
        control_volume, exact, exact, tau=tau
    )


def _curvature_case(ntheta, representation, direct_face_band_radius=1):
    geometry, domain, context, _coordinates, cv, face_bc, closure = _setup(
        ntheta, True, direct_face_band_radius, True
    )
    cells = cv.cells
    exact_owned = _smooth_state(cells, aggregate=False)
    owner_values = _smooth_state(cells, aggregate=True)

    harness = SimpleNamespace(
        geometry=geometry,
        domain=domain,
        control_volume_geometry=cv,
        halo_exchange=closure.halo_exchange,
        topology_filler=closure.topology_filler,
        physical_ghost_filler=closure.physical_ghost_filler,
    )
    harness._owner_field = LocalFciDrbEBRhs._owner_field.__get__(harness)
    harness._prepare_rlp_reconstructed_halo = (
        LocalFciDrbEBRhs._prepare_rlp_reconstructed_halo.__get__(harness)
    )

    def close_fine(values):
        return closure(
            inject_owned_field_to_halo(values, geometry.layout),
            domain,
            face_bc,
        )

    def prepare(owner):
        if representation == "H":
            return LocalFciDrbEBRhs._prepare_poisson_bracket_halo(
                harness, owner, face_bc
            )
        if representation == "P":
            return close_fine(expand_local_control_volume_owner_field(owner, cells))
        raise ValueError(representation)

    coefficients = _radial_curvature_coefficients(geometry)
    # A face-averaged state is not a sufficient reference for this nonlinear
    # material path.  Evaluate the state-dependent action at the transverse
    # quadrature points, then integrate it over every raw cell owned by the
    # transition-adjacent aggregate.
    material_reference = _exact_quadrature_material_action(cv, tau=0.7)

    empty_cv_bc = LocalControlVolumeBoundaryBC3D.empty(max_rows=0)
    baseline_stencils = []
    direct_stencils = []
    direct_states = []
    for owner in owner_values:
        prepared = prepare(owner)
        baseline_stencils.append(
            build_local_conservative_stencil_from_field(
                prepared, geometry, context
            )
        )
        stencil, left, right = build_local_control_volume_poisson_face_stencil(
            prepared,
            geometry,
            domain,
            context,
            cv,
            empty_cv_bc,
            owner_values_owned=owner,
            regular_face_bc=face_bc,
            patch_centered_face_values=True,
        )
        direct_stencils.append(stencil)
        direct_states.append((left, right))
    psi_owner = owner_values[0] + 0.7 * owner_values[2]
    radial_curvature_states = build_local_radial_curvature_face_states(
        jnp.stack((*owner_values, psi_owner), axis=-1),
        geometry,
        domain,
        cv,
        positive_field_count=3,
        halo_exchange=closure.halo_exchange,
        topology_filler=closure.topology_filler,
    )

    baseline = local_curvature_production_path_op(
        tuple(baseline_stencils),
        geometry,
        coefficients,
        tau=0.7,
        domain=domain,
        control_volume_geometry=cv,
    )
    direct_traces = local_curvature_production_path_op(
        tuple(baseline_stencils),
        geometry,
        coefficients,
        tau=0.7,
        domain=domain,
        control_volume_geometry=cv,
        direct_transition_face_states=tuple(direct_states),
    )
    direct = local_curvature_production_path_op(
        tuple(direct_stencils),
        geometry,
        coefficients,
        tau=0.7,
        domain=domain,
        control_volume_geometry=cv,
        direct_transition_face_states=tuple(direct_states),
    )
    direct_quadrature_total, direct_quadrature_diagnostics = local_curvature_production_path_op(
        tuple(baseline_stencils),
        geometry,
        coefficients,
        tau=0.7,
        domain=domain,
        control_volume_geometry=cv,
        radial_curvature_face_states=radial_curvature_states,
        return_diagnostics=True,
    )
    # The controlled reference isolates the radial material path.  Use the
    # production operator's radial directional residual, while retaining the
    # independent quadrature helper solely as the exact reference.
    direct_quadrature = direct_quadrature_diagnostics["directional_residual"][0]
    direct_quadrature_consistency = float(
        jnp.max(
            jnp.abs(
                direct_quadrature_total
                - jnp.sum(
                    direct_quadrature_diagnostics["directional_residual"], axis=0
                )
            )
        )
    )

    psi_exact = exact_owned[0] + 0.7 * exact_owned[2]
    psi_reference_stencil = build_local_conservative_stencil_from_field(
        close_fine(psi_exact), geometry, context
    )
    psi_coefficients = tuple(
        left + 0.7 * right
        for left, right in zip(_STATE_COEFFICIENTS[0], _STATE_COEFFICIENTS[2])
    )
    psi_reference_stencil = replace(
        psi_reference_stencil,
        face_values=replace(
            psi_reference_stencil.face_values,
            x=_exact_radial_face_average(geometry, psi_coefficients),
        ),
    )
    psi_reference_fine = local_curvature_conservative_op(
        psi_reference_stencil,
        geometry,
        coefficients,
        domain=domain,
        axis_regular_axes=(True, False, False),
    )
    psi_reference = aggregate_local_control_volume_average(
        psi_reference_fine, cells, domain
    )
    psi_prepared = prepare(psi_owner)
    psi_baseline_stencil = build_local_conservative_stencil_from_field(
        psi_prepared, geometry, context
    )
    psi_direct_stencil = patch_local_radial_curvature_face_values(
        psi_baseline_stencil,
        radial_curvature_states,
        geometry,
        cv,
        field_index=4,
    )
    psi_baseline_fine = local_curvature_conservative_op(
        psi_baseline_stencil,
        geometry,
        coefficients,
        domain=domain,
        axis_regular_axes=(True, False, False),
    )
    psi_direct_fine = local_curvature_conservative_op(
        psi_direct_stencil,
        geometry,
        coefficients,
        domain=domain,
        axis_regular_axes=(True, False, False),
    )
    psi_baseline = aggregate_local_control_volume_average(
        psi_baseline_fine, cells, domain
    )
    psi_direct = aggregate_local_control_volume_average(
        psi_direct_fine, cells, domain
    )

    transition = _transition_owner_mask(cv)
    non_axis_transition = (
        transition
        & (cells.owner_i > 0)
        & _two_face_compact_owner_mask(cv)
    )
    axis_transition = transition & (cells.owner_i == 0)
    full_band = _two_face_compact_owner_mask(cv)
    active_non_axis = (
        cells.is_active_owner
        & (cells.owner_i > 0)
        & (cells.owner_i < cells.shape[0] - 1)
        & full_band
    )
    radius_one_cv = _setup(ntheta, True, 1)[4]
    former_band_edge = _radial_direct_coverage(radius_one_cv)[
        "one_face_owner_mask"
    ]
    comparison_masks = {
        "all_non_axis": active_non_axis,
        "transition": non_axis_transition,
        "former_band_edge": former_band_edge,
    }
    weights = cells.aggregate_volume
    result = {
        "coupled_baseline": _relative_l2(
            baseline, material_reference, weights, transition
        ),
        "coupled_direct_traces": _relative_l2(
            direct_traces, material_reference, weights, transition
        ),
        "coupled_direct": _relative_l2(
            direct, material_reference, weights, transition
        ),
        "coupled_direct_quadrature": _relative_l2(
            direct_quadrature, material_reference, weights, non_axis_transition
        ),
        "coupled_direct_quadrature_axis": _relative_l2(
            direct_quadrature, material_reference, weights, axis_transition
        ),
        "coupled_direct_quadrature_consistency": direct_quadrature_consistency,
        "coupled_direct_quadrature_non_axis_max": float(
            jnp.max(
                jnp.where(
                    non_axis_transition[..., None],
                    jnp.abs(direct_quadrature - material_reference),
                    0.0,
                )
            )
        ),
        "centered_baseline": _scalar_relative_l2(
            psi_baseline, psi_reference, weights, full_band
        ),
        "centered_direct": _scalar_relative_l2(
            psi_direct, psi_reference, weights, full_band
        ),
    }
    for name, mask in comparison_masks.items():
        result[f"coupled_direct_quadrature_{name}"] = _relative_l2(
            direct_quadrature, material_reference, weights, mask
        )
        result[f"centered_direct_{name}"] = _scalar_relative_l2(
            psi_direct, psi_reference, weights, mask
        )
    result["coverage"] = {
        key: value
        for key, value in _radial_direct_coverage(cv).items()
        if key != "one_face_owner_mask"
    }
    return result


def _orders(errors, resolutions):
    return [
        math.log(coarse / fine) / math.log(nfine / ncoarse)
        for coarse, fine, ncoarse, nfine in zip(
            errors[:-1], errors[1:], resolutions[:-1], resolutions[1:]
        )
    ]


def test_transition_face_targets_against_exact_quadratic_average():
    geometry, domain, context, _coordinates, cv, face_bc, closure = _setup(
        32, True, 1
    )
    cells = cv.cells
    owner = _smooth_state(cells, aggregate=True)[0]
    harness = SimpleNamespace(
        geometry=geometry,
        domain=domain,
        control_volume_geometry=cv,
        halo_exchange=closure.halo_exchange,
        topology_filler=closure.topology_filler,
        physical_ghost_filler=closure.physical_ghost_filler,
    )
    harness._owner_field = LocalFciDrbEBRhs._owner_field.__get__(harness)
    harness._prepare_rlp_reconstructed_halo = (
        LocalFciDrbEBRhs._prepare_rlp_reconstructed_halo.__get__(harness)
    )
    prepared = LocalFciDrbEBRhs._prepare_poisson_bracket_halo(
        harness, owner, face_bc
    )
    _stencil, fitted_left, fitted_right = (
        build_local_control_volume_poisson_face_stencil(
            prepared,
            geometry,
            domain,
            context,
            cv,
            LocalControlVolumeBoundaryBC3D.empty(max_rows=0),
            owner_values_owned=owner,
            regular_face_bc=face_bc,
            patch_centered_face_values=True,
        )
    )
    ordinary_left, ordinary_right, _ = _third_order_scalar_face_states_from_halo(
        prepared,
        geometry,
        boundary_trace=None,
        axis_regular_axes=(True, False, False),
        positivity_floor=None,
    )
    faces = cv.irregular_faces
    index = (
        faces.logical_face_i,
        faces.logical_face_j,
        faces.logical_face_k,
    )
    q = faces.quadrature_points
    expected_q = _polynomial_average(
        q,
        jnp.zeros(q.shape[:-1] + (3, 3), dtype=jnp.float64),
        _STATE_COEFFICIENTS[0],
    )
    measure = jnp.where(
        faces.quadrature_active,
        jnp.linalg.norm(faces.area_covector_weight, axis=-1),
        0.0,
    )
    expected = jnp.sum(measure * expected_q, axis=(1, 2)) / jnp.sum(
        measure, axis=(1, 2)
    )
    face_i = jnp.asarray(faces.logical_face_i, dtype=jnp.int32)
    group_size = jnp.asarray(cv.angular_group_sizes, dtype=jnp.int32)
    safe_i = jnp.clip(face_i, 1, group_size.shape[0] - 1)
    transition = (
        faces.active
        & (face_i > 0)
        & (face_i < group_size.shape[0])
        & (group_size[safe_i - 1] != group_size[safe_i])
    )
    fitted_error = max(
        float(jnp.max(jnp.where(transition, jnp.abs(fitted_left.x[index] - expected), 0.0))),
        float(jnp.max(jnp.where(transition, jnp.abs(fitted_right.x[index] - expected), 0.0))),
    )
    ordinary_error = max(
        float(jnp.max(jnp.abs(ordinary_left.x[index] - expected))),
        float(jnp.max(jnp.abs(ordinary_right.x[index] - expected))),
    )
    print({"fitted_error": fitted_error, "ordinary_H_error": ordinary_error})
    assert fitted_error < 1.0e-10


def test_direct_centered_face_patch_is_logical_and_transition_scoped():
    geometry, domain, context, _coordinates, cv, _face_bc, closure = _setup(
        32, True, 1
    )
    owner = _smooth_state(cv.cells, aggregate=True)[0]
    prepared = closure(
        inject_owned_field_to_halo(owner, geometry.layout), domain, _face_bc
    )
    baseline = build_local_conservative_stencil_from_field(
        prepared, geometry, context
    )
    direct = build_local_control_volume_direct_face_states(
        owner,
        geometry,
        domain,
        cv,
        halo_exchange=closure.halo_exchange,
        topology_filler=closure.topology_filler,
    )
    patched = patch_local_control_volume_direct_face_values(
        baseline, direct, geometry, cv, transition_only=True
    )
    faces = cv.irregular_faces
    index = (
        faces.logical_face_i,
        faces.logical_face_j,
        faces.logical_face_k,
    )
    measure = jnp.where(
        faces.quadrature_active,
        jnp.linalg.norm(faces.area_covector_weight, axis=-1),
        0.0,
    )
    expected = jnp.sum(measure * direct.centered, axis=(1, 2)) / jnp.maximum(
        jnp.sum(measure, axis=(1, 2)), 1.0e-30
    )
    fi = jnp.asarray(faces.logical_face_i, dtype=jnp.int32)
    profile = jnp.asarray(cv.angular_group_sizes, dtype=jnp.int32)
    safe = jnp.clip(fi, 1, profile.shape[0] - 1)
    transition = (
        faces.active
        & (fi > 0)
        & (fi < profile.shape[0])
        & (profile[safe - 1] != profile[safe])
    )
    np.testing.assert_allclose(
        patched.face_values.x[index][transition], expected[transition],
        rtol=1.0e-12, atol=1.0e-12,
    )
    np.testing.assert_allclose(
        patched.face_values.x[index][~transition],
        baseline.face_values.x[index][~transition],
        rtol=0.0, atol=0.0,
    )


def test_poisson_direct_collapse_ignores_compact_jacobian_factor():
    geometry, domain, context, _coordinates, cv, face_bc, closure = _setup(
        32, True, 1
    )
    owner = _smooth_state(cv.cells, aggregate=True)[0]
    harness = SimpleNamespace(
        geometry=geometry,
        domain=domain,
        control_volume_geometry=cv,
        halo_exchange=closure.halo_exchange,
        topology_filler=closure.topology_filler,
        physical_ghost_filler=closure.physical_ghost_filler,
    )
    harness._owner_field = LocalFciDrbEBRhs._owner_field.__get__(harness)
    harness._prepare_rlp_reconstructed_halo = (
        LocalFciDrbEBRhs._prepare_rlp_reconstructed_halo.__get__(harness)
    )
    prepared = LocalFciDrbEBRhs._prepare_poisson_bracket_halo(
        harness, owner, face_bc
    )
    altered_faces = replace(
        cv.irregular_faces,
        J=jnp.where(
            cv.irregular_faces.active[:, None, None],
            7.0 * cv.irregular_faces.J + 0.123,
            cv.irregular_faces.J,
        ),
    )
    altered_cv = replace(cv, irregular_faces=altered_faces)

    def build(control_volume, *, patch_centered_face_values):
        return build_local_control_volume_poisson_face_stencil(
            prepared,
            geometry,
            domain,
            context,
            control_volume,
            LocalControlVolumeBoundaryBC3D.empty(max_rows=0),
            owner_values_owned=owner,
            regular_face_bc=face_bc,
            patch_centered_face_values=patch_centered_face_values,
        )

    reference = build(cv, patch_centered_face_values=False)
    altered = build(altered_cv, patch_centered_face_values=False)
    for reference_face, altered_face in zip(reference[1:], altered[1:]):
        np.testing.assert_allclose(
            reference_face.x, altered_face.x, rtol=0.0, atol=1.0e-14
        )
    reference_centered = build(cv, patch_centered_face_values=True)
    altered_centered = build(altered_cv, patch_centered_face_values=True)
    np.testing.assert_allclose(
        reference_centered[0].face_values.x,
        altered_centered[0].face_values.x,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_p_and_h_full_radial_direct_curvature_comparison():
    resolutions = (32, 48)
    cases = {
        representation: [
            _curvature_case(
                resolution,
                representation,
                direct_face_band_radius=resolution,
            )
            for resolution in resolutions
        ]
        for representation in ("P", "H")
    }
    print(cases)
    for representation, rows in cases.items():
        coupled_baseline = [row["coupled_baseline"] for row in rows]
        coupled_direct = [row["coupled_direct"] for row in rows]
        coupled_quadrature = [
            row["coupled_direct_quadrature"] for row in rows
        ]
        centered_baseline = [row["centered_baseline"] for row in rows]
        centered_direct = [row["centered_direct"] for row in rows]

        assert max(
            row["coupled_direct_quadrature_consistency"] for row in rows
        ) < 1.0e-12
        assert max(coupled_quadrature) < 1.0e-9
        assert max(
            row["coupled_direct_quadrature_non_axis_max"] for row in rows
        ) < 1.0e-9
        for scope in ("all_non_axis", "transition", "former_band_edge"):
            scoped_quadrature = [
                row[f"coupled_direct_quadrature_{scope}"] for row in rows
            ]
            scoped_centered = [
                row[f"centered_direct_{scope}"] for row in rows
            ]
            assert max(scoped_quadrature) < 2.0e-9
            assert min(_orders(scoped_centered, resolutions)) > 2.0
        for row in rows:
            assert row["coverage"]["interior_one_face_raw"] == 0
            assert row["coverage"]["interior_missing_raw"] == 0
            assert row["coverage"]["interior_one_face_owners"] == 0
        assert centered_direct[-1] < centered_baseline[-1]
        assert min(_orders(centered_direct, resolutions)) > 2.0

    # Face-local operations lose all dependence on whether their off-face
    # representation is P or H.  The current averaged material path does not:
    # at the axis-adjacent aggregate it still uses the unpatched cell state.
    # H preserves that subcell structure; P does not.  Once the material path
    # itself is evaluated from direct quadrature traces, the distinction also
    # disappears.
    for key in (
        "coupled_direct_quadrature",
        "centered_direct",
        "coupled_direct_quadrature_all_non_axis",
        "coupled_direct_quadrature_transition",
        "coupled_direct_quadrature_former_band_edge",
        "centered_direct_all_non_axis",
        "centered_direct_transition",
        "centered_direct_former_band_edge",
    ):
        np.testing.assert_allclose(
            [row[key] for row in cases["P"]],
            [row[key] for row in cases["H"]],
            rtol=1.0e-10,
            atol=1.0e-14,
        )
    h_axis = [row["coupled_direct_quadrature_axis"] for row in cases["H"]]
    p_axis = [row["coupled_direct_quadrature_axis"] for row in cases["P"]]
    assert all(h_error < p_error for h_error, p_error in zip(h_axis, p_axis))
    h_coupled = [row["coupled_direct"] for row in cases["H"]]
    p_coupled = [row["coupled_direct"] for row in cases["P"]]
    assert min(_orders(h_coupled, resolutions)) > 1.0
    assert all(
        h_error < p_error
        for h_error, p_error in zip(h_coupled, p_coupled)
    )
