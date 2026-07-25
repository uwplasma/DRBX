"""Shifted-torus four-field MMS tests with a closed embedded cut-wall box."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from unittest import mock

from drbx.runtime import configure_jax_runtime

_JAX_COMPILATION_CACHE_DIR = configure_jax_runtime(precision="float64")

import jax
import numpy as np
import pytest

from drbx.geometry.fci_control_volumes import compile_local_control_volume_geometry

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))
import shifted_torus_4field_mms_helpers as shifted_mms  # noqa: E402
import shifted_torus_4field_cutwall_geometry as cutwall_geometry  # noqa: E402
from shifted_torus_4field_cutwall_geometry import (  # noqa: E402
    BOX_THETA_CENTER,
    BOX_THETA_HALF_WIDTH,
    BOX_X_FRACTION_RANGE,
    BOX_ZETA_RANGE,
    CV_RECONSTRUCTION_EQUATION_CELL,
    CV_RECONSTRUCTION_EQUATION_DIRICHLET,
    CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
    MESH_AXIS_NAMES,
    _GAUSS2_NODES,
    _box_bounds,
    _build_closed_box_control_volume_cells,
    _build_closed_box_control_volume_faces,
    _build_closed_box_embedded_control_volume_geometry,
    _build_global_coordinate_face_open_measures,
    _build_global_closed_box_control_volume_topology,
    _build_shifted_torus_regular_boundary_closure,
    _build_stacked_embedded_control_volume_geometry,
    _closed_box_fluid_moments_3point,
    _closed_box_irregular_storage_mask,
    _dilate_reconstruction_owner_mask,
    _face_patch_quadrature_numpy,
    _integrate_shifted_torus_rectangular_moments,
    _intrinsic_reconstruction_owner_mask,
    _open_face_rectangles_numpy,
    _pad_control_volume_face_rows,
    _pad_embedded_control_volume_geometry,
    _pad_quadratic_reconstruction,
    _replace_canonical_compact_rows,
    _sanitize_centroid_metric_points,
    _select_closed_box_control_volume_owners,
    _shape_from_resolution,
    _shifted_torus_cartesian_from_logical,
    _shifted_torus_curvature_at_logical_points,
    _shifted_torus_metric_payload_numpy,
    _validate_face_functional_boundary_weight_scale,
    _validate_face_functional_cell_radius,
    _validate_reconstruction_boundary_weight_scale,
    _with_embedded_control_volume_geometry,
)
from shifted_torus_4field_cutwall_mms import (  # noqa: E402
    _agglomerate_control_volume_average,
    _assemble_global_control_volume_cell_data,
    _control_volume_exact_boundary_bc,
    _expand_control_volume_owner_values,
    _integrate_local_exact_state_over_fluid,
    _integrate_local_four_field_over_fluid,
    _integrate_local_scalar_over_fluid,
    _multiply_local_dirichlet_face_bc,
    _project_global_exact_state_to_control_volumes,
    _project_local_exact_time_derivative_to_control_volumes,
    _project_local_mms_source_to_control_volumes,
    _shifted_torus_analytic_rhs_at_logical_points,
    _shifted_torus_exact_time_derivative_at_logical_points,
    _shifted_torus_mms_source_at_logical_points,
    _shifted_torus_operator_reference_at_logical_points,
    _shifted_torus_regular_radial_face_average,
    _with_shifted_torus_regular_radial_face_averages,
)
from shifted_torus_4field_cutwall_runtime import (  # noqa: E402
    LocalShiftedTorus4FieldCutWallRhs,
    _make_gmres_config,
    _make_parameters,
    _mask_4field_state_inactive,
    simulate_mms_shifted_torus_4field_cutwall,
)
from shifted_torus_4field_cutwall_convergence import (  # noqa: E402
    _POISSON_OMEGA_DIAGNOSTIC_WIDTH,
    _POISSON_OMEGA_EXACT_CENTROID,
    _POISSON_OMEGA_EXACT_PHI_RECONSTRUCTED_OMEGA,
    _POISSON_OMEGA_RECONSTRUCTED_PHI_EXACT_OMEGA,
    _clear_resolution_level_caches,
    _control_volume_operator_category_masks,
    _effective_box_translation,
    _fit_operator_order,
    _masked_field_error_statistics,
    _masked_state_error_statistics,
    _operator_category_statistics,
    _poisson_omega_component_diagnostic_statistics,
    _poisson_radial_plane_category_masks,
    _print_control_volume_geometry_summary,
    _print_state_error_statistics,
    _resolution_step_count,
    _state_error_statistics,
    _volume_weighted_field_error_statistics,
    _volume_weighted_state_error_statistics,
    run_shifted_torus_direct_face_closure_diagnostic,
    run_shifted_torus_4field_cutwall_convergence,
    run_shifted_torus_control_volume_operator_convergence,
)


def assert_shape_divisible_by_shards(*args, **kwargs):
    from mms_domain_decomp_helpers import assert_shape_divisible_by_shards as impl

    return impl(*args, **kwargs)


def build_shifted_torus_local_domain(*args, **kwargs):
    from mms_domain_decomp_helpers import build_shifted_torus_local_domain as impl

    return impl(*args, **kwargs)


def build_shifted_torus_local_geometry(*args, **kwargs):
    from mms_domain_decomp_helpers import build_shifted_torus_local_geometry as impl

    return impl(*args, **kwargs)


def expand_local_shard_pytree(*args, **kwargs):
    from mms_domain_decomp_helpers import expand_local_shard_pytree as impl

    return impl(*args, **kwargs)


def extract_local_shard_pytree(*args, **kwargs):
    from mms_domain_decomp_helpers import extract_local_shard_pytree as impl

    return impl(*args, **kwargs)


def local_shard_pytree_partition_spec(*args, **kwargs):
    from mms_domain_decomp_helpers import local_shard_pytree_partition_spec as impl

    return impl(*args, **kwargs)


def stack_local_shard_pytree(*args, **kwargs):
    from mms_domain_decomp_helpers import stack_local_shard_pytree as impl

    return impl(*args, **kwargs)


def make_mesh_for_shard_counts(*args, **kwargs):
    from mms_domain_decomp_helpers import make_mesh_for_shard_counts as impl

    return impl(*args, **kwargs)


def test_shifted_torus_global_compact_face_ids_are_unique_across_shards() -> None:
    """Step-2A decomposition characterization against an N=6 baseline."""
    global_shape = (6, 6, 6)
    topology, _ = _build_global_closed_box_control_volume_topology(
        global_shape=global_shape, halo_width=2, enable_merging=True,
    )
    baseline_ids: set[int] | None = None
    baseline_wall_rows: int | None = None
    baseline_functionals: dict[int, tuple[np.ndarray, ...]] | None = None
    for shard_counts in ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2)):
        geometry = _build_stacked_embedded_control_volume_geometry(
            global_shape=global_shape,
            shard_counts=shard_counts,
            halo_width=2 if shard_counts == (1, 1, 1) else 3,
            enable_merging=True,
        )
        rows = geometry.irregular_faces
        active = np.asarray(rows.active, dtype=bool)
        face_id = np.asarray(rows.global_face_id, dtype=np.int64)
        logical = face_id[active & (face_id >= 0)]
        assert logical.size == np.unique(logical).size
        current_ids = set(int(value) for value in logical)
        wall_rows = int(np.count_nonzero(active & (face_id < 0)))
        functionals = geometry.face_functionals
        current_functionals: dict[int, tuple[np.ndarray, ...]] = {}
        for shard_index in np.ndindex(*shard_counts):
            for row in np.flatnonzero(active[shard_index]):
                current_face_id = int(face_id[shard_index][row])
                equation_active = np.asarray(
                    functionals.observation_active[shard_index][row],
                    dtype=bool,
                )
                payload = (
                    np.asarray(functionals.projected_flux_weights[shard_index][row])[equation_active],
                    np.asarray(functionals.parallel_flux_weights[shard_index][row])[equation_active],
                    np.asarray(functionals.parallel_gradient_flux_weights[shard_index][row])[equation_active],
                    np.asarray((
                        functionals.condition_number[shard_index][row],
                        functionals.reproduction_residual[shard_index][row],
                        functionals.normalized_projected_weight_norm[shard_index][row],
                        functionals.normalized_parallel_weight_norm[shard_index][row],
                        functionals.normalized_parallel_gradient_weight_norm[shard_index][row],
                    )),
                )
                assert current_face_id not in current_functionals
                current_functionals[current_face_id] = payload
        if baseline_ids is None:
            baseline_ids, baseline_wall_rows = current_ids, wall_rows
            baseline_functionals = current_functionals
        else:
            assert current_ids == baseline_ids
            assert wall_rows == baseline_wall_rows
            assert baseline_functionals is not None
            assert current_functionals.keys() == baseline_functionals.keys()
            for current_face_id, payload in current_functionals.items():
                for actual, expected in zip(
                    payload, baseline_functionals[current_face_id]
                ):
                    np.testing.assert_array_equal(actual, expected)
        # Validate every row against the evaluator IDs for its actual shard.
        for shard_index in np.ndindex(*shard_counts):
            local = compile_local_control_volume_geometry(
                topology, shard_index=shard_index, shard_counts=shard_counts,
            )
            shard_active = active[shard_index]
            shard_ids = face_id[shard_index][shard_active]
            assert set(int(value) for value in shard_ids if value >= 0).issubset(
                set(int(value) for value in local.local_face_id)
            )
            remote = np.asarray(rows.has_remote_residual[shard_index], dtype=bool)
            remote_owner = np.asarray(rows.has_remote_owner[shard_index], dtype=bool)
            assert np.all(~remote | remote_owner)
            np.testing.assert_array_equal(
                np.asarray(rows.remote_residual_halo_i[shard_index])[remote],
                np.asarray(rows.remote_halo_i[shard_index])[remote],
            )
            np.testing.assert_array_equal(
                np.asarray(rows.remote_residual_halo_j[shard_index])[remote],
                np.asarray(rows.remote_halo_j[shard_index])[remote],
            )
            np.testing.assert_array_equal(
                np.asarray(rows.remote_residual_halo_k[shard_index])[remote],
                np.asarray(rows.remote_halo_k[shard_index])[remote],
            )
        if shard_counts != (1, 1, 1):
            # The y interface is evaluated once, and its plus residual has a
            # precomputed halo destination for the forthcoming reverse exchange.
            assert int(np.sum(np.asarray(rows.has_remote_residual, dtype=bool))) > 0
        del geometry
        jax.clear_caches()


def test_canonical_remote_residual_audit_routes_paired_rows() -> None:
    """Canonical oriented rows need routing, not a direct signed-row sum."""
    row_flux = np.asarray([3.0, -1.0])
    canonical_remote = np.asarray([True, True])

    # A canonical oriented-row sum is allowed to be nonzero: these are two
    # evaluator-side contributions, not mirrored copies to sum directly.
    canonical_row_sum = np.sum(np.where(canonical_remote, row_flux, 0.0))
    assert canonical_row_sum == 2.0

    # The reverse residual destinations receive the opposite contributions.
    routed_residual = np.where(canonical_remote, -row_flux, 0.0)
    assert canonical_row_sum + np.sum(routed_residual) == 0.0
    assert 2.0 * np.sum(np.abs(row_flux)) == 8.0


def test_decomposed_face_functionals_require_radius_plus_one_halo() -> None:
    with np.testing.assert_raises_regex(
        ValueError,
        "halo_width >= face_functional_cell_radius \\+ 1",
    ):
        _build_stacked_embedded_control_volume_geometry(
            global_shape=(8, 8, 8),
            shard_counts=(1, 1, 2),
            halo_width=2,
            enable_merging=True,
            face_functional_cell_radius=2,
        )


def test_decomposed_agglomerated_boundary_observations_have_remote_sources() -> None:
    """Agglomerated direct rows may gather a wall trace from another shard."""
    geometry = _build_stacked_embedded_control_volume_geometry(
        global_shape=(10, 10, 10),
        shard_counts=(1, 1, 2),
        halo_width=3,
        enable_merging=True,
        face_functional_all_owner_boundary_observations=True,
    )
    rows = geometry.face_functionals
    active = np.asarray(rows.observation_active, dtype=bool)
    kinds = np.asarray(rows.observation_kind, dtype=np.int32)
    sources = np.asarray(rows.boundary_source_shard, dtype=np.int32)
    evaluator_sources = np.broadcast_to(
        np.arange(2, dtype=np.int32)[None, None, :, None, None],
        sources.shape,
    )
    dirichlet = active & (kinds == CV_RECONSTRUCTION_EQUATION_DIRICHLET)
    assert np.any(dirichlet)
    assert np.any(sources[dirichlet] != evaluator_sources[dirichlet])


def test_n16_translated_canonical_compact_faces_repartition_across_z_shards() -> None:
    """Tangential owner migration must not drop canonical coordinate faces."""
    effective_translation = _effective_box_translation(
        16,
        box_translation=(0.0, 0.0, 0.0),
        box_cell_translation=(0.0, 0.5, 0.5),
    )
    one_shard = _build_stacked_embedded_control_volume_geometry(
        global_shape=(16, 16, 16),
        shard_counts=(1, 1, 1),
        halo_width=3,
        enable_merging=True,
        box_translation=effective_translation,
    )
    geometry = _build_stacked_embedded_control_volume_geometry(
        global_shape=(16, 16, 16),
        shard_counts=(1, 1, 4),
        halo_width=3,
        enable_merging=True,
        box_translation=effective_translation,
    )
    assert int(np.count_nonzero(np.asarray(one_shard.reconstruction.active))) == int(
        np.count_nonzero(np.asarray(geometry.reconstruction.active))
    )
    rows = geometry.irregular_faces
    active = np.asarray(rows.active, dtype=bool)
    face_ids = np.asarray(rows.global_face_id, dtype=np.int64)
    active_ids = face_ids[active]
    assert active_ids.size == np.unique(active_ids).size
    assert set((1479, 1735, 1991, 2247)).issubset(set(int(value) for value in active_ids))
    assert np.any(np.asarray(rows.has_remote_owner, dtype=bool))
    one_regular = one_shard.regular_faces
    four_regular = geometry.regular_faces
    for axis in range(3):
        mask_name = ("x_open_mask", "y_open_mask", "z_open_mask")[axis]
        fraction_name = (
            "x_area_fraction",
            "y_area_fraction",
            "z_area_fraction",
        )[axis]
        one_mask = np.asarray(getattr(one_regular, mask_name), dtype=bool)[0, 0, 0]
        one_fraction = np.asarray(
            getattr(one_regular, fraction_name), dtype=np.float64
        )[0, 0, 0]
        four_mask = np.asarray(getattr(four_regular, mask_name), dtype=bool)
        four_fraction = np.asarray(
            getattr(four_regular, fraction_name), dtype=np.float64
        )
        for shard_k in range(4):
            start = 4 * shard_k
            stop = start + 4 + (1 if axis == 2 else 0)
            one_slice = [slice(None), slice(None), slice(start, stop)]
            four_slice = (0, 0, shard_k)
            np.testing.assert_array_equal(
                one_mask[tuple(one_slice)], four_mask[four_slice]
            )
            np.testing.assert_allclose(
                one_fraction[tuple(one_slice)], four_fraction[four_slice],
                rtol=0.0, atol=2.0e-14,
            )

    one_reconstruction = one_shard.reconstruction
    four_reconstruction = geometry.reconstruction
    one_faces = one_shard.irregular_faces
    four_faces = geometry.irregular_faces
    one_target_rows = {}
    one_active = np.asarray(one_reconstruction.active[0, 0, 0], dtype=bool)
    one_targets = np.stack(
        (
            np.asarray(one_reconstruction.target_i[0, 0, 0]),
            np.asarray(one_reconstruction.target_j[0, 0, 0]),
            np.asarray(one_reconstruction.target_k[0, 0, 0]),
        ),
        axis=-1,
    )
    one_kinds = np.asarray(one_reconstruction.equation_kind[0, 0, 0])
    one_equation_active = np.asarray(one_reconstruction.equation_active[0, 0, 0])
    one_sample = np.stack(
        (
            np.asarray(one_reconstruction.sample_i[0, 0, 0]),
            np.asarray(one_reconstruction.sample_j[0, 0, 0]),
            np.asarray(one_reconstruction.sample_k[0, 0, 0]),
        ),
        axis=-1,
    )
    one_boundary_rows = np.asarray(one_reconstruction.boundary_face_row[0, 0, 0])
    one_boundary_patch = np.asarray(one_reconstruction.boundary_patch[0, 0, 0])
    one_boundary_quad = np.asarray(one_reconstruction.boundary_quadrature[0, 0, 0])
    one_face_ids = np.asarray(one_faces.global_face_id[0, 0, 0])
    one_transforms = np.asarray(one_reconstruction.rhs_transform[0, 0, 0])
    for row in np.flatnonzero(one_active):
        one_target_rows[tuple(int(value) for value in one_targets[row])] = row

    four_active = np.asarray(four_reconstruction.active, dtype=bool)
    four_targets = np.stack(
        (
            np.asarray(four_reconstruction.target_i),
            np.asarray(four_reconstruction.target_j),
            np.asarray(four_reconstruction.target_k),
        ),
        axis=-1,
    )
    four_kinds = np.asarray(four_reconstruction.equation_kind)
    four_equation_active = np.asarray(four_reconstruction.equation_active)
    four_sample = np.stack(
        (
            np.asarray(four_reconstruction.sample_i),
            np.asarray(four_reconstruction.sample_j),
            np.asarray(four_reconstruction.sample_k),
        ),
        axis=-1,
    )
    four_boundary_rows = np.asarray(four_reconstruction.boundary_face_row)
    four_boundary_patch = np.asarray(four_reconstruction.boundary_patch)
    four_boundary_quad = np.asarray(four_reconstruction.boundary_quadrature)
    four_transforms = np.asarray(four_reconstruction.rhs_transform)
    for shard_k in range(4):
        shard_start = np.asarray((0, 0, 4 * shard_k), dtype=np.int32)
        local_faces = np.asarray(four_faces.global_face_id[0, 0, shard_k])
        for row in np.flatnonzero(four_active[0, 0, shard_k]):
            target = tuple(
                int(value)
                for value in four_targets[0, 0, shard_k, row] + shard_start
            )
            assert target in one_target_rows
            one_row = one_target_rows[target]
            assert np.array_equal(
                one_equation_active[one_row],
                four_equation_active[0, 0, shard_k, row],
            )
            one_normalized = np.where(
                one_kinds[one_row] == CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
                CV_RECONSTRUCTION_EQUATION_CELL,
                one_kinds[one_row],
            )
            four_normalized = np.where(
                four_kinds[0, 0, shard_k, row] == CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
                CV_RECONSTRUCTION_EQUATION_CELL,
                four_kinds[0, 0, shard_k, row],
            )
            np.testing.assert_array_equal(one_normalized, four_normalized)
            np.testing.assert_allclose(
                one_transforms[one_row],
                four_transforms[0, 0, shard_k, row],
                rtol=0.0,
                atol=1.0e-13,
            )
            assert int(one_reconstruction.rank[0, 0, 0, one_row]) == int(
                four_reconstruction.rank[0, 0, shard_k, row]
            )
            np.testing.assert_allclose(
                one_reconstruction.condition_number[0, 0, 0, one_row],
                four_reconstruction.condition_number[0, 0, shard_k, row],
                rtol=0.0,
                atol=1.0e-10,
            )
            for equation in np.flatnonzero(one_equation_active[one_row]):
                one_kind = int(one_kinds[one_row, equation])
                four_kind = int(four_kinds[0, 0, shard_k, row, equation])
                if one_kind in (
                    CV_RECONSTRUCTION_EQUATION_CELL,
                    CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
                ):
                    assert four_kind in (
                        CV_RECONSTRUCTION_EQUATION_CELL,
                        CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
                    )
                    one_sample_global = tuple(int(value) for value in one_sample[one_row, equation])
                    if four_kind == CV_RECONSTRUCTION_EQUATION_CELL:
                        four_sample_global = tuple(
                            int(value) + int(shard_start[axis])
                            for axis, value in enumerate(four_sample[0, 0, shard_k, row, equation])
                        )
                    else:
                        four_sample_global = tuple(
                            int(value) + int(shard_start[axis]) - 3
                            for axis, value in enumerate(four_sample[0, 0, shard_k, row, equation])
                        )
                    four_sample_global = tuple(
                        value % 16 if axis in (1, 2) else value
                        for axis, value in enumerate(four_sample_global)
                    )
                    assert one_sample_global == four_sample_global
                elif one_kind == CV_RECONSTRUCTION_EQUATION_DIRICHLET:
                    one_face_id = int(one_face_ids[one_boundary_rows[one_row, equation]])
                    four_face_id = int(local_faces[four_boundary_rows[0, 0, shard_k, row, equation]])
                    assert one_face_id == four_face_id
                    assert int(one_boundary_patch[one_row, equation]) == int(
                        four_boundary_patch[0, 0, shard_k, row, equation]
                    )
                    assert int(one_boundary_quad[one_row, equation]) == int(
                        four_boundary_quad[0, 0, shard_k, row, equation]
                    )


def test_n16_translated_projected_exact_state_is_shard_invariant() -> None:
    """Raw exact integrals are assembled before cross-shard aggregation."""
    shape = (16, 16, 16)
    translation = _effective_box_translation(
        16,
        box_translation=(0.0, 0.0, 0.0),
        box_cell_translation=(0.0, 0.5, 0.5),
    )
    bundles = {}
    for shard_counts in ((1, 1, 1), (1, 1, 4)):
        bundles[shard_counts] = _build_stacked_embedded_control_volume_geometry(
            global_shape=shape,
            shard_counts=shard_counts,
            halo_width=3,
            enable_merging=True,
            box_translation=translation,
        )
    projected = {
        shard_counts: _project_global_exact_state_to_control_volumes(
            shifted_mms.build_shifted_torus_4field_geometry(shape),
            bundle,
            shard_counts=shard_counts,
            halo_width=3,
            time=0.0,
            box_translation=translation,
        )
        for shard_counts, bundle in bundles.items()
    }
    one_state, one_phi = projected[(1, 1, 1)]
    four_state, four_phi = projected[(1, 1, 4)]
    one_cells = _assemble_global_control_volume_cell_data(
        shape, bundles[(1, 1, 1)], shard_counts=(1, 1, 1)
    )
    four_cells = _assemble_global_control_volume_cell_data(
        shape, bundles[(1, 1, 4)], shard_counts=(1, 1, 4)
    )
    one_active = np.asarray(one_cells["is_active_owner"], dtype=bool)
    four_active = np.asarray(four_cells["is_active_owner"], dtype=bool)
    np.testing.assert_array_equal(one_active, four_active)
    assert np.any(np.asarray(four_cells["is_aggregate_target"], dtype=bool))
    for one_value, four_value in (
        (one_state.density, four_state.density),
        (one_state.omega, four_state.omega),
        (one_state.v_ion_parallel, four_state.v_ion_parallel),
        (one_state.v_electron_parallel, four_state.v_electron_parallel),
        (one_phi, four_phi),
    ):
        np.testing.assert_allclose(
            np.asarray(one_value)[one_active],
            np.asarray(four_value)[four_active],
            rtol=2.0e-12,
            atol=2.0e-13,
        )


def test_n16_explicit_divergence_payload_diagnostic() -> None:
    if os.environ.get("DRBX_RUN_EXPLICIT_DIAGNOSTICS") != "1":
        pytest.skip("explicit-only diagnostic; set DRBX_RUN_EXPLICIT_DIAGNOSTICS=1")
    one = run_shifted_torus_direct_face_closure_diagnostic(
        resolution=16, shard_counts=(1, 1, 1), halo_width=3
    )
    four = run_shifted_torus_direct_face_closure_diagnostic(
        resolution=16, shard_counts=(1, 1, 4), halo_width=3
    )

    def gather_owned(value):
        value = np.asarray(value)
        if value.shape[0] == 1:
            return value[0]
        return np.concatenate([value[shard] for shard in range(4)], axis=2)

    active = np.asarray(one["active_owner"], dtype=bool)
    operator_delta = np.abs(
        np.asarray(one["operator"]) - np.asarray(four["operator"])
    )
    operator_location = np.unravel_index(
        int(np.argmax(operator_delta)), operator_delta.shape
    )
    print(
        "operator equivalence: "
        f"max_abs={operator_delta[operator_location]:.16e} "
        f"location={operator_location} one={np.asarray(one['operator'])[operator_location]:.16e} "
        f"four={np.asarray(four['operator'])[operator_location]:.16e}"
    )
    np.testing.assert_allclose(
        np.asarray(one["operator"])[active],
        np.asarray(four["operator"])[active],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    one_owner_sum = gather_owned(one["payload"]["owner_sum"])
    four_owner_sum = gather_owned(four["payload"]["owner_sum"])
    owner_sum_delta = np.abs(one_owner_sum - four_owner_sum)
    owner_sum_location = np.unravel_index(
        int(np.argmax(owner_sum_delta)), owner_sum_delta.shape
    )
    print(
        "owner sum comparison: "
        f"max_abs={owner_sum_delta[owner_sum_location]:.16e} "
        f"location={owner_sum_location} one={one_owner_sum[owner_sum_location]:.16e} "
        f"four={four_owner_sum[owner_sum_location]:.16e}"
    )
    for name in (
        "owner_sum_regular",
        "accumulated_owner_sum",
        "aggregate_volume",
        "final_divergence",
    ):
        left = gather_owned(one["payload"][name])
        right = gather_owned(four["payload"][name])
        delta = np.abs(left - right)
        mask = active & (delta > 2.0e-12)
        if np.any(mask):
            location = np.unravel_index(
                int(np.argmax(np.where(mask, delta, 0.0))), delta.shape
            )
            print(
                f"divergence payload mismatch {name}: location={location} "
                f"delta={delta[location]:.16e} one={left[location]:.16e} "
                f"four={right[location]:.16e}"
            )
        else:
            print(f"divergence payload match: {name}")

    def reconstruct_remote_halo(payload_name):
        """Route z-only halo payloads back to the global owned-cell array."""
        payload = np.asarray(four["payload"][payload_name])
        assert payload.shape[0] == 4
        h = 3
        owned_z = 4
        global_payload = np.zeros((16, 16, 16), dtype=payload.dtype)
        outside_xy = 0.0
        for shard in range(4):
            local = payload[shard]
            outside_xy = max(
                outside_xy,
                float(np.max(np.abs(local[:h, :, :]))) if h else 0.0,
                float(np.max(np.abs(local[h + 16 :, :, :]))) if h else 0.0,
                float(np.max(np.abs(local[:, :h, :]))) if h else 0.0,
                float(np.max(np.abs(local[:, h + 16 :, :]))) if h else 0.0,
            )
            lower_receiver = (shard - 1) % 4
            upper_receiver = (shard + 1) % 4
            global_payload[:, :, lower_receiver * owned_z + owned_z - h : lower_receiver * owned_z + owned_z] += local[
                h : h + 16, h : h + 16, :h
            ]
            global_payload[:, :, upper_receiver * owned_z : upper_receiver * owned_z + h] += local[
                h : h + 16, h : h + 16, h + owned_z : h + owned_z + h
            ]
        print(
            f"{payload_name} manual halo routing: shape={payload.shape} "
            f"max_outside_owned_xy={outside_xy:.16e}"
        )
        return global_payload

    one_remote_regular = np.zeros((16, 16, 16), dtype=np.float64)
    one_remote = np.zeros((16, 16, 16), dtype=np.float64)
    four_remote_regular = reconstruct_remote_halo("remote_halo_regular")
    four_remote = reconstruct_remote_halo("remote_halo")
    for name, left, right in (
        ("remote_halo_regular", one_remote_regular, four_remote_regular),
        ("remote_halo", one_remote, four_remote),
    ):
        delta = np.abs(left - right)
        location = np.unravel_index(int(np.argmax(delta)), delta.shape)
        print(
            f"remote payload comparison {name}: max_abs={delta[location]:.16e} "
            f"location={location} one={left[location]:.16e} "
            f"four={right[location]:.16e}"
        )

    remote = np.asarray(four["payload"]["remote_halo"])
    h = 3
    owned_z = 4
    reverse = np.zeros((16, 16, 16), dtype=np.float64)
    for shard in range(4):
        lower_receiver = (shard - 1) % 4
        upper_receiver = (shard + 1) % 4
        reverse[:, :, lower_receiver * owned_z + owned_z - h : lower_receiver * owned_z + owned_z] += remote[
            shard, h : h + 16, h : h + 16, :h
        ]
        reverse[:, :, upper_receiver * owned_z : upper_receiver * owned_z + h] += remote[
            shard, h : h + 16, h : h + 16, h + owned_z : h + owned_z + h
        ]
    expected_reverse = gather_owned(four["payload"]["accumulated_owner_sum"]) - gather_owned(
        four["payload"]["owner_sum"]
    )
    delta = np.abs(reverse - expected_reverse)
    location = np.unravel_index(int(np.argmax(delta)), delta.shape)
    print(
        "reverse residual routing check: "
        f"max_abs={delta[location]:.16e} location={location} "
        f"manual={reverse[location]:.16e} actual={expected_reverse[location]:.16e}"
    )
    regular_one = np.asarray(one["regular_divergence"])
    regular_four = np.asarray(four["regular_divergence"])
    regular_delta = np.abs(regular_one - regular_four)
    regular_location = np.unravel_index(
        int(np.argmax(regular_delta)), regular_delta.shape
    )
    print(
        "regular divergence comparison: "
        f"max_abs={regular_delta[regular_location]:.16e} "
        f"location={regular_location} one={regular_one[regular_location]:.16e} "
        f"four={regular_four[regular_location]:.16e}"
    )
    one_rows = one["rows"][(1, 1, 1)]
    four_rows = four["rows"][(1, 1, 4)]
    common_ids = sorted(set(one_rows) & set(four_rows))
    row_deltas = np.asarray(
        [abs(float(one_rows[face_id]["flux"]) - float(four_rows[face_id]["flux"])) for face_id in common_ids]
    )
    row_index = int(np.argmax(row_deltas))
    print(
        "compact row flux comparison: "
        f"count={len(common_ids)} max_abs={row_deltas[row_index]:.16e} "
        f"face_id={common_ids[row_index]} one={one_rows[common_ids[row_index]]['flux']:.16e} "
        f"four={four_rows[common_ids[row_index]]['flux']:.16e}"
    )
    metadata_mismatches = []
    for face_id in common_ids:
        one_row = one_rows[face_id]
        four_row = four_rows[face_id]
        fields = (
            "minus_global",
            "plus_global",
            "has_remote_residual",
            "remote_residual_global",
        )
        if any(one_row[field] != four_row[field] for field in fields):
            metadata_mismatches.append((face_id, one_row, four_row))
    print(
        f"compact row ownership metadata mismatches: {len(metadata_mismatches)}"
    )
    if metadata_mismatches:
        face_id, one_row, four_row = metadata_mismatches[0]
        print(
            f"first compact row metadata mismatch: face_id={face_id} "
            f"one_minus={one_row['minus_global']} four_minus={four_row['minus_global']} "
            f"one_plus={one_row['plus_global']} four_plus={four_row['plus_global']} "
            f"one_remote_residual={one_row['has_remote_residual']} "
            f"four_remote_residual={four_row['has_remote_residual']} "
            f"one_remote_target={one_row['remote_residual_global']} "
            f"four_remote_target={four_row['remote_residual_global']}"
        )

    def reconstruct_compact_sum(row_map):
        result = np.zeros((16, 16, 16), dtype=np.float64)
        for row in row_map.values():
            flux_value = float(row["flux"])
            result[row["minus_global"]] += flux_value
            if row["plus_global"] is not None:
                result[row["plus_global"]] -= flux_value
            if row["remote_residual_global"] is not None:
                result[row["remote_residual_global"]] -= flux_value
        return result

    one_compact = reconstruct_compact_sum(one_rows)
    four_compact = reconstruct_compact_sum(four_rows)
    compact_delta = np.abs(one_compact - four_compact)
    compact_location = np.unravel_index(
        int(np.argmax(compact_delta)), compact_delta.shape
    )
    print(
        "manual compact global sum comparison: "
        f"max_abs={compact_delta[compact_location]:.16e} "
        f"location={compact_location} one={one_compact[compact_location]:.16e} "
        f"four={four_compact[compact_location]:.16e}"
    )
    for label, row_map in (("one", one_rows), ("four", four_rows)):
        hits = []
        for face_id, row in row_map.items():
            if (
                row["minus_global"] == compact_location
                or row["plus_global"] == compact_location
                or row["remote_residual_global"] == compact_location
            ):
                hits.append(
                    (
                        face_id,
                        row["flux"],
                        row["minus_global"],
                        row["plus_global"],
                        row["remote_residual_global"],
                    )
                )
        print(f"compact rows touching {compact_location} ({label}): {hits}")
    print(
        "face 10488 metadata: "
        f"one={one_rows.get(10488)} four={four_rows.get(10488)}"
    )


def test_canonical_compact_rows_replace_same_id_local_discovery() -> None:
    local = [
        {"global_face_id": 10488, "minus": (8, 14, 3), "plus": (8, 13, 4)},
        {"global_face_id": 7, "minus": (1, 1, 1)},
    ]
    canonical = [
        {
            "global_face_id": 10488,
            "minus": (8, 14, 3),
            "plus": None,
            "remote_residual_halo": (3, 17, 7),
        }
    ]
    rows = _replace_canonical_compact_rows(local, canonical)
    assert [row["global_face_id"] for row in rows] == [7, 10488]
    assert rows[-1] is canonical[0]
    assert rows[-1]["plus"] is None
    assert rows[-1]["remote_residual_halo"] == (3, 17, 7)


def test_global_coordinate_face_measure_batches_match_scalar_quadrature() -> None:
    """Batched open-face measures equal the reference scalar evaluation."""
    global_shape = (6, 5, 4)
    geometry = build_shifted_torus_local_geometry(
        global_shape,
        halo_width=2,
        global_shape=global_shape,
        shard_index=(0, 0, 0),
    )
    axis_faces = tuple(
        np.asarray(grid.faces_owned, dtype=np.float64)
        for grid in (geometry.grid.x, geometry.grid.y, geometry.grid.z)
    )
    translation = (0.0, 0.13, -0.17)

    for axis in range(3):
        face_shape = list(global_shape)
        face_shape[axis] += 1
        face_shape = tuple(face_shape)
        actual = _build_global_coordinate_face_open_measures(
            axis=axis,
            axis_faces=axis_faces,
            face_shape=face_shape,
            box_translation=translation,
            chunk_size=3,
        )
        expected = np.zeros(face_shape, dtype=np.float64)
        for face in np.ndindex(*face_shape):
            tangential = tuple(
                (
                    float(axis_faces[t][face[t]]),
                    float(axis_faces[t][face[t] + 1]),
                )
                for t in range(3) if t != axis
            )
            for rectangle in _open_face_rectangles_numpy(
                axis=axis,
                face_coordinate=float(axis_faces[axis][face[axis]]),
                tangential_bounds=tangential,
                box_translation=translation,
            ):
                points, area_weight = _face_patch_quadrature_numpy(
                    axis=axis,
                    face_coordinate=float(axis_faces[axis][face[axis]]),
                    rectangle=rectangle,
                    orientation=1.0,
                )
                expected[face] += float(
                    np.sum(
                        _shifted_torus_metric_payload_numpy(points)[0]
                        * np.linalg.norm(area_weight, axis=-1)
                    )
                )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)


def test_shifted_torus_one_shard_reuses_unsplit_bundle_and_sanitizes_metrics(
    tmp_path: Path,
) -> None:
    """The one-shard fast path stacks the already-compiled global bundle."""
    captured_bundles = []
    compile_calls = 0
    original_stack = cutwall_geometry.stack_local_shard_pytree
    original_compile = cutwall_geometry._compile_global_cubic_face_functional_records

    def capture_stack(shard_counts, builder):
        assert shard_counts == (1, 1, 1)
        bundle = builder((0, 0, 0))
        captured_bundles.append(bundle)
        return original_stack(shard_counts, lambda _index: bundle)

    def count_compile(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(*args, **kwargs)

    with mock.patch.object(
        cutwall_geometry, "stack_local_shard_pytree", capture_stack
    ), mock.patch.object(
        cutwall_geometry,
        "_compile_global_cubic_face_functional_records",
        count_compile,
    ):
        stacked = _build_stacked_embedded_control_volume_geometry(
            global_shape=(6, 6, 6),
            shard_counts=(1, 1, 1),
            halo_width=2,
            enable_merging=True,
            geometry_cache_dir=tmp_path,
        )

    assert compile_calls == 1
    assert len(captured_bundles) == 1
    unsplit = captured_bundles[0]
    # Compare matching pytrees after removing only the one-shard dimensions.
    stacked_leaves = jax.tree_util.tree_leaves(extract_local_shard_pytree(stacked))
    unsplit_leaves = jax.tree_util.tree_leaves(unsplit)
    assert len(stacked_leaves) == len(unsplit_leaves)
    for stacked_leaf, unsplit_leaf in zip(stacked_leaves, unsplit_leaves):
        np.testing.assert_array_equal(
            np.asarray(stacked_leaf), np.asarray(unsplit_leaf)
        )

    assert np.all(np.isfinite(np.asarray(unsplit.centroid_J)))
    cache_files = list(tmp_path.glob("global_face_functionals_*.npz"))
    assert len(cache_files) == 1

    with mock.patch.object(
        cutwall_geometry,
        "_compile_global_cubic_face_functional_records",
        side_effect=AssertionError("cache hit must skip functional compilation"),
    ):
        cached = _build_stacked_embedded_control_volume_geometry(
            global_shape=(6, 6, 6),
            shard_counts=(1, 1, 1),
            halo_width=2,
            enable_merging=True,
            geometry_cache_dir=tmp_path,
        )
    cached_leaves = jax.tree_util.tree_leaves(
        extract_local_shard_pytree(cached)
    )
    assert len(cached_leaves) == len(stacked_leaves)
    for cached_leaf, stacked_leaf in zip(cached_leaves, stacked_leaves):
        np.testing.assert_array_equal(
            np.asarray(cached_leaf),
            np.asarray(stacked_leaf),
        )


def test_shifted_torus_decomposed_lowering_skips_local_reconstruction_fit(
    tmp_path: Path,
) -> None:
    """Decomposed shards lower the canonical fit instead of refitting locally."""
    original = cutwall_geometry.precompute_local_moment_reconstruction
    calls = 0

    def count_calls(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    with mock.patch.object(
        cutwall_geometry,
        "precompute_local_moment_reconstruction",
        side_effect=count_calls,
    ):
        _build_stacked_embedded_control_volume_geometry(
            global_shape=(6, 6, 6),
            shard_counts=(1, 1, 1),
            halo_width=2,
            enable_merging=True,
            geometry_cache_dir=tmp_path / "one",
        )
    assert calls == 1

    calls = 0
    with mock.patch.object(
        cutwall_geometry,
        "precompute_local_moment_reconstruction",
        side_effect=count_calls,
    ):
        _build_stacked_embedded_control_volume_geometry(
            global_shape=(6, 6, 6),
            shard_counts=(1, 1, 2),
            halo_width=3,
            enable_merging=True,
            geometry_cache_dir=tmp_path / "two",
        )
    assert calls == 1


def test_inactive_centroid_metric_points_are_replaced_in_domain() -> None:
    points = np.asarray(
        (((0.4, 1.0, 2.0), (np.nan, 0.0, 0.0)), ((0.0, 0.0, 0.0), (np.inf, 0.0, 0.0))),
        dtype=np.float64,
    )
    active = np.asarray(((True, True), (False, True)), dtype=bool)
    sanitized = _sanitize_centroid_metric_points(points, active)
    reference = np.asarray(
        (0.5 * (shifted_mms.x_min + shifted_mms.x_max), 0.0, 0.0),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(sanitized[0, 0], points[0, 0])
    np.testing.assert_array_equal(sanitized[0, 1], reference)
    np.testing.assert_array_equal(sanitized[1, 0], reference)
    np.testing.assert_array_equal(sanitized[1, 1], reference)
    assert np.all(np.isfinite(sanitized))


def test_resolution_level_cleanup_clears_jax_and_collects(
    capsys,
) -> None:
    with mock.patch(
        "shifted_torus_4field_cutwall_convergence.jax.clear_caches"
    ) as clear_caches, mock.patch(
        "shifted_torus_4field_cutwall_convergence.gc.collect",
        return_value=17,
    ) as collect:
        assert _clear_resolution_level_caches(80) == 17
    clear_caches.assert_called_once_with()
    collect.assert_called_once_with()
    assert (
        "Released shifted_torus resolution payloads: N=80, collected=17"
        in capsys.readouterr().out
    )


def test_poisson_radial_plane_diagnostics_partition_active_owners() -> None:
    active = np.ones((8, 2, 2), dtype=bool)
    active[0, 0, 0] = False
    active[3, 1, 1] = False
    masks = _poisson_radial_plane_category_masks(
        {"is_active_owner": active}
    )

    assert tuple(masks) == (
        "radial_lower_x0",
        "radial_lower_x1",
        "radial_lower_x2",
        "radial_core",
        "radial_upper_xm2",
        "radial_upper_xm1",
    )
    membership = np.sum(
        np.stack([np.asarray(mask, dtype=np.int32) for mask in masks.values()]),
        axis=0,
    )
    np.testing.assert_array_equal(membership, active.astype(np.int32))
    assert np.all(np.asarray(masks["radial_lower_x0"])[1:] == 0)
    assert np.all(np.asarray(masks["radial_lower_x1"])[0] == 0)
    assert np.all(np.asarray(masks["radial_lower_x1"])[2:] == 0)
    assert np.all(np.asarray(masks["radial_lower_x2"])[2])
    assert np.all(np.asarray(masks["radial_upper_xm2"])[-2])
    assert np.all(np.asarray(masks["radial_upper_xm1"])[-1])


def test_poisson_radial_plane_diagnostics_reject_too_few_planes() -> None:
    try:
        _poisson_radial_plane_category_masks(
            {"is_active_owner": np.ones((5, 2, 2), dtype=bool)}
        )
    except ValueError as error:
        assert "at least six radial owner planes" in str(error)
    else:
        raise AssertionError("undersized radial diagnostics must be rejected")


def test_poisson_omega_component_diagnostics_compare_integrated_reference() -> None:
    shape = (6, 1, 1)
    active = np.ones(shape, dtype=bool)
    categories = {
        "all_active": active,
        **_poisson_radial_plane_category_masks(
            {"is_active_owner": active}
        ),
    }
    actual = np.full(shape, 2.0, dtype=np.float64)
    reference = np.zeros(shape, dtype=np.float64)
    payload = np.zeros(
        shape + (_POISSON_OMEGA_DIAGNOSTIC_WIDTH,),
        dtype=np.float64,
    )
    payload[
        ..., _POISSON_OMEGA_EXACT_PHI_RECONSTRUCTED_OMEGA
    ] = 1.5
    payload[
        ..., _POISSON_OMEGA_RECONSTRUCTED_PHI_EXACT_OMEGA
    ] = 1.0
    payload[..., _POISSON_OMEGA_EXACT_CENTROID] = 1.0

    statistics = _poisson_omega_component_diagnostic_statistics(
        actual,
        reference,
        payload,
        np.ones(shape, dtype=np.float64),
        categories,
    )

    assert statistics["reconstructed"]["all_active"][1] == 2.0
    assert (
        statistics["exact_phi_reconstructed_omega"]["radial_lower_x0"][1]
        == 1.5
    )
    assert (
        statistics["reconstructed_phi_exact_omega"]["radial_lower_x1"][1]
        == 1.0
    )
    assert statistics["exact_centroid"]["radial_upper_xm1"][1] == 1.0
    assert statistics["exact_centroid"]["radial_upper_xm1"][3] == 1

    centroid_statistics = _poisson_omega_component_diagnostic_statistics(
        actual,
        reference,
        payload,
        np.ones(shape, dtype=np.float64),
        categories,
        comparison_target="exact_centroid",
    )
    assert centroid_statistics["reconstructed"]["all_active"][1] == 1.0
    assert (
        centroid_statistics[
            "exact_phi_reconstructed_omega"
        ]["radial_lower_x0"][1]
        == 0.5
    )
    assert (
        centroid_statistics[
            "reconstructed_phi_exact_omega"
        ]["radial_lower_x1"][1]
        == 0.0
    )
    assert centroid_statistics["exact_centroid"]["all_active"][1] == 0.0

    try:
        _poisson_omega_component_diagnostic_statistics(
            actual,
            reference,
            payload[..., :-1],
            np.ones(shape, dtype=np.float64),
            categories,
        )
    except ValueError as error:
        assert "diagnostic payload must have shape" in str(error)
    else:
        raise AssertionError("incorrect diagnostic width must be rejected")

    try:
        _poisson_omega_component_diagnostic_statistics(
            actual,
            reference,
            payload,
            np.ones(shape, dtype=np.float64),
            categories,
            comparison_target="cell_corner",
        )
    except ValueError as error:
        assert "comparison_target must be" in str(error)
    else:
        raise AssertionError("unknown comparison target must be rejected")


def test_targeted_operator_selection_validates_names_without_running_geometry() -> None:
    """Targeted mode accepts known names and rejects unknown names up front."""
    result = run_shifted_torus_control_volume_operator_convergence(
        resolutions=[],
        shard_counts=(1, 1, 1),
        halo_width=2,
        selected_operators=[
            "perp_laplacian_phi",
            "parallel_density_flux_divergence",
        ],
    )
    assert result == {"records": {}, "orders": {}, "phi_residuals": []}
    try:
        run_shifted_torus_control_volume_operator_convergence(
            resolutions=[],
            shard_counts=(1, 1, 1),
            halo_width=2,
            selected_operators=["not_an_operator"],
        )
    except ValueError as error:
        assert "unknown control-volume operators" in str(error)
    else:
        raise AssertionError("unknown targeted operator must be rejected")


def test_face_functional_boundary_weight_scale_validates_without_geometry() -> None:
    result = run_shifted_torus_control_volume_operator_convergence(
        resolutions=[],
        shard_counts=(1, 1, 1),
        halo_width=2,
        face_functional_boundary_weight_scale=1.0,
    )
    assert result == {"records": {}, "orders": {}, "phi_residuals": []}
    for invalid in (0.0, np.nan):
        try:
            run_shifted_torus_control_volume_operator_convergence(
                resolutions=[],
                shard_counts=(1, 1, 1),
                halo_width=2,
                face_functional_boundary_weight_scale=invalid,
            )
        except ValueError as error:
            assert "finite and positive" in str(error)
        else:
            raise AssertionError("invalid boundary weight scale must be rejected")

    assert _validate_face_functional_boundary_weight_scale(1.0) == 1.0


def test_reconstruction_boundary_weight_scale_validates_without_geometry() -> None:
    result = run_shifted_torus_control_volume_operator_convergence(
        resolutions=[],
        shard_counts=(1, 1, 1),
        halo_width=2,
        reconstruction_boundary_weight_scale=1.0,
    )
    assert result == {"records": {}, "orders": {}, "phi_residuals": []}
    for invalid in (0.0, np.nan):
        try:
            run_shifted_torus_control_volume_operator_convergence(
                resolutions=[],
                shard_counts=(1, 1, 1),
                halo_width=2,
                reconstruction_boundary_weight_scale=invalid,
            )
        except ValueError as error:
            assert "finite and positive" in str(error)
        else:
            raise AssertionError(
                "invalid reconstruction boundary weight scale must be rejected"
            )

    assert _validate_reconstruction_boundary_weight_scale(1.0) == 1.0


def test_reconstruction_distance_row_weight_exponent_validates_without_geometry() -> None:
    result = run_shifted_torus_control_volume_operator_convergence(
        resolutions=[],
        shard_counts=(1, 1, 1),
        halo_width=2,
        reconstruction_distance_row_weight_exponent=0.0,
    )
    assert result == {"records": {}, "orders": {}, "phi_residuals": []}
    for valid in (0.0, 1.0, 4.0):
        result = run_shifted_torus_control_volume_operator_convergence(
            resolutions=[],
            shard_counts=(1, 1, 1),
            halo_width=2,
            reconstruction_distance_row_weight_exponent=valid,
        )
        assert result == {"records": {}, "orders": {}, "phi_residuals": []}
    for invalid in (-1.0, np.nan, np.inf, -np.inf):
        try:
            run_shifted_torus_control_volume_operator_convergence(
                resolutions=[],
                shard_counts=(1, 1, 1),
                halo_width=2,
                reconstruction_distance_row_weight_exponent=invalid,
            )
        except ValueError as error:
            assert "finite and nonnegative" in str(error)
        else:
            raise AssertionError(
                "invalid reconstruction distance-row weight exponent must be rejected"
            )


def test_face_functional_cell_radius_validates_without_geometry() -> None:
    result = run_shifted_torus_control_volume_operator_convergence(
        resolutions=[],
        shard_counts=(1, 1, 1),
        halo_width=2,
        face_functional_cell_radius=2,
    )
    assert result == {"records": {}, "orders": {}, "phi_residuals": []}
    for invalid in (0, 3, 1.0, True):
        try:
            run_shifted_torus_control_volume_operator_convergence(
                resolutions=[],
                shard_counts=(1, 1, 1),
                halo_width=2,
                face_functional_cell_radius=invalid,
            )
        except ValueError as error:
            assert "integer 1 or 2" in str(error)
        else:
            raise AssertionError("invalid face-functional cell radius must be rejected")

    assert _validate_face_functional_cell_radius(1) == 1
    assert _validate_face_functional_cell_radius(2) == 2


def test_all_owner_boundary_observations_validates_without_geometry() -> None:
    result = run_shifted_torus_control_volume_operator_convergence(
        resolutions=[],
        shard_counts=(1, 1, 1),
        halo_width=2,
        face_functional_all_owner_boundary_observations=True,
    )
    assert result == {"records": {}, "orders": {}, "phi_residuals": []}
    try:
        run_shifted_torus_control_volume_operator_convergence(
            resolutions=[],
            shard_counts=(1, 2, 1),
            halo_width=2,
            face_functional_all_owner_boundary_observations=True,
        )
    except ValueError as error:
        assert "requires one shard" in str(error)
    else:
        raise AssertionError(
            "all-owner boundary observations must be rejected for multiple shards"
        )


def test_box_translation_defaults_without_geometry() -> None:
    operator_result = run_shifted_torus_control_volume_operator_convergence(
        resolutions=[],
        shard_counts=(1, 1, 1),
        halo_width=2,
    )
    full_result = run_shifted_torus_4field_cutwall_convergence(
        resolutions=[],
        shard_counts=(1, 1, 1),
        halo_width=2,
    )
    assert operator_result == {"records": {}, "orders": {}, "phi_residuals": []}
    assert full_result["resolutions"] == []


def test_effective_box_translation_n10_quarter_cell_is_exact_and_additive() -> None:
    np.testing.assert_allclose(
        _effective_box_translation(
            10,
            box_translation=(0.5, -0.25, 1.0),
            box_cell_translation=(0.25, 0.25, 0.25),
        ),
        (0.52, -0.25 + np.pi / 20.0, 1.0 + np.pi / 20.0),
    )
    for invalid in ((0.0, 0.0), (0.0, np.inf, 0.0), None):
        try:
            _effective_box_translation(10, box_cell_translation=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "box_cell_translation must reject nonfinite or non-three-value inputs"
            )


def test_box_translation_shifts_every_bound_and_rejects_invalid_values() -> None:
    baseline = np.asarray(_box_bounds(), dtype=np.float64)
    translation = np.asarray((0.125, -0.25, 0.375), dtype=np.float64)
    translated = np.asarray(
        _box_bounds(box_translation=tuple(translation)),
        dtype=np.float64,
    )
    np.testing.assert_allclose(translated, baseline + translation[:, None])

    for invalid in ((0.0, 0.0), (0.0, np.inf, 0.0), None):
        try:
            _box_bounds(box_translation=invalid)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(
                "box_translation must reject nonfinite or non-three-value inputs"
            )


def test_box_translation_cli_reaches_both_runners_without_geometry() -> None:
    operator_kwargs = {}
    full_kwargs = {}

    def capture_operator(**kwargs):
        operator_kwargs.update(kwargs)
        return {"records": {}, "orders": {}, "phi_residuals": []}

    def capture_full(**kwargs):
        full_kwargs.update(kwargs)
        return {"resolutions": []}

    with mock.patch(
        __name__ + ".run_shifted_torus_control_volume_operator_convergence",
        side_effect=capture_operator,
    ), mock.patch(
        __name__ + ".run_shifted_torus_4field_cutwall_convergence",
        side_effect=capture_full,
    ):
        with mock.patch.object(
            sys,
            "argv",
            [
                "test_fci_cutwall_shifted_torus_4field.py",
                "--operator-convergence-only",
                "--resolutions",
                "10",
                "--skip-runtime-info",
            ],
        ):
            main()
        assert operator_kwargs["box_translation"] == (0.0, 0.0, 0.0)
        assert operator_kwargs["box_cell_translation"] == (0.0, 0.0, 0.0)
        assert operator_kwargs["geometry_cache_dir"] is None

        with mock.patch.object(
            sys,
            "argv",
            [
                "test_fci_cutwall_shifted_torus_4field.py",
                "--resolutions",
                "10",
                "--skip-runtime-info",
                "--box-translation",
                "0.125",
                "-0.25",
                "0.375",
                "--box-cell-translation",
                "0.25",
                "0.5",
                "-0.25",
                "--geometry-cache-dir",
                "/tmp/drbx-cutwall-test-cache",
            ],
        ):
            main()
        assert full_kwargs["box_translation"] == (0.125, -0.25, 0.375)
        assert full_kwargs["box_cell_translation"] == (0.25, 0.5, -0.25)
        assert full_kwargs["geometry_cache_dir"] == Path(
            "/tmp/drbx-cutwall-test-cache"
        )








def _print_runtime_info() -> None:
    print("=" * 80)
    print("JAX runtime")
    print("=" * 80)
    print(f"default backend: {jax.default_backend()}")
    print(f"local_device_count: {jax.local_device_count()}")
    print(f"compilation_cache_dir: {_JAX_COMPILATION_CACHE_DIR}")
    print("devices:")
    for index, device in enumerate(jax.local_devices()):
        print(f"  [{index}] {device}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shifted-torus 4-field cut-wall MMS convergence harness")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[10, 14])
    parser.add_argument("--shard-counts", nargs=3, type=int, metavar=("PX", "PY", "PZ"), default=(1, 1, 1))
    parser.add_argument("--halo-width", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=shifted_mms.tf)
    parser.add_argument("--base-steps", type=int, default=shifted_mms.num_steps)
    parser.add_argument("--rho-star", type=float, default=shifted_mms.rho_star)
    parser.add_argument(
        "--box-translation",
        nargs=3,
        type=float,
        metavar=("DX", "DTHETA", "DZ"),
        default=(0.0, 0.0, 0.0),
        help="Translate the closed reference box by (DX, DTHETA, DZ).",
    )
    parser.add_argument(
        "--box-cell-translation",
        nargs=3,
        type=float,
        metavar=("FX", "FTHETA", "FZETA"),
        default=(0.0, 0.0, 0.0),
        help=(
            "Translate the closed reference box by fractions of one cell "
            "(FX, FTHETA, FZETA) at each resolution."
        ),
    )
    parser.add_argument(
        "--geometry-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for versioned global face-functional caches. "
            "Cache files contain numeric NPZ data only; no pickle is used."
        ),
    )
    parser.add_argument(
        "--minimum-order",
        type=float,
        default=1.8,
        help=(
            "Minimum accepted per-field volume-L2 and active-owner Linf "
            "order for operator and full convergence sweeps."
        ),
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--operator-convergence-only",
        action="store_true",
        help=(
            "Run separate unified control-volume spatial operator kernels and "
            "skip the RK convergence sweep."
        ),
    )
    parser.add_argument(
        "--reconstruction-boundary-weight-scale",
        type=float,
        default=1.0,
        help=(
            "Test-only reconstruction scale for boundary observation fitting "
            "weights."
        ),
    )
    parser.add_argument(
        "--reconstruction-distance-row-weight-exponent",
        type=float,
        default=0.0,
        help=(
            "Test-only distance-row weight exponent for reconstruction; "
            "zero preserves legacy behavior."
        ),
    )
    parser.add_argument(
        "--face-functional-boundary-weight-scale",
        type=float,
        default=1.0,
        help=(
            "Diagnostic-only operator-convergence scale for Dirichlet "
            "observation fitting weights."
        ),
    )
    parser.add_argument(
        "--face-functional-all-owner-boundary-observations",
        action="store_true",
        help=(
            "Diagnostic-only operator-convergence option: include Dirichlet "
            "observations from every local compact-face owner (one shard only)."
        ),
    )
    parser.add_argument(
        "--face-functional-cell-radius",
        type=int,
        default=2,
        help=(
            "Diagnostic-only operator-convergence candidate-cell radius; "
            "must be 1 or 2."
        ),
    )
    parser.add_argument(
        "--skip-operator-phi-solve",
        action="store_true",
        help=(
            "Diagnostic-only: use projected exact phi in the full-RHS kernel "
            "and skip the separate phi inversion check while retaining all "
            "spatial operator kernels."
        ),
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        default=None,
        help=(
            "Run only the named scalar operators in operator-convergence "
            "mode. Targeted mode skips the full RHS and phi solve."
        ),
    )
    parser.add_argument(
        "--face-audit",
        action="store_true",
        help=(
            "Print the compact physical faces attached to each operator's "
            "worst aggregate, including functional conditioning diagnostics."
        ),
    )
    parser.add_argument(
        "--enable-agglomeration",
        action="store_true",
        help=(
            "Merge sub-threshold fluid cut cells into a face-connected "
            "control-volume owner."
        ),
    )
    parser.add_argument("--skip-runtime-info", action="store_true")
    args = parser.parse_args()

    if (
        bool(args.face_functional_all_owner_boundary_observations)
        and not bool(args.operator_convergence_only)
    ):
        parser.error(
            "--face-functional-all-owner-boundary-observations is only available "
            "with --operator-convergence-only"
        )

    if not args.skip_runtime_info:
        _print_runtime_info()
    if bool(args.operator_convergence_only):
        run_shifted_torus_control_volume_operator_convergence(
            resolutions=[int(value) for value in args.resolutions],
            shard_counts=tuple(int(value) for value in args.shard_counts),
            halo_width=int(args.halo_width),
            box_translation=tuple(float(value) for value in args.box_translation),
            box_cell_translation=tuple(
                float(value) for value in args.box_cell_translation
            ),
            geometry_cache_dir=args.geometry_cache_dir,
            rho_star_value=float(args.rho_star),
            enable_agglomeration=bool(args.enable_agglomeration),
            minimum_order=float(args.minimum_order),
            check_phi_solve=not bool(args.skip_operator_phi_solve),
            selected_operators=(
                None
                if args.operators is None
                else [str(value) for value in args.operators]
            ),
            face_audit=bool(args.face_audit),
            reconstruction_boundary_weight_scale=(
                float(args.reconstruction_boundary_weight_scale)
            ),
            reconstruction_distance_row_weight_exponent=(
                float(args.reconstruction_distance_row_weight_exponent)
            ),
            face_functional_boundary_weight_scale=(
                float(args.face_functional_boundary_weight_scale)
            ),
            face_functional_all_owner_boundary_observations=bool(
                args.face_functional_all_owner_boundary_observations
            ),
            face_functional_cell_radius=int(args.face_functional_cell_radius),
        )
        return
    run_shifted_torus_4field_cutwall_convergence(
        resolutions=[int(value) for value in args.resolutions],
        shard_counts=tuple(int(value) for value in args.shard_counts),
        halo_width=int(args.halo_width),
        box_translation=tuple(float(value) for value in args.box_translation),
        box_cell_translation=tuple(
            float(value) for value in args.box_cell_translation
        ),
        geometry_cache_dir=args.geometry_cache_dir,
        final_time=float(args.final_time),
        base_steps=int(args.base_steps),
        rho_star_value=float(args.rho_star),
        plot=bool(args.plot),
        plot_path=args.plot_path,
        show_progress=bool(args.show_progress),
        enable_agglomeration=bool(args.enable_agglomeration),
        minimum_order=float(args.minimum_order),
        reconstruction_boundary_weight_scale=(
            float(args.reconstruction_boundary_weight_scale)
        ),
        reconstruction_distance_row_weight_exponent=(
            float(args.reconstruction_distance_row_weight_exponent)
        ),
    )


if __name__ == "__main__":
    main()
