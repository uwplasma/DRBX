"""Shifted-torus four-field MMS tests with a closed embedded cut-wall box."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from drbx.runtime import configure_jax_runtime

_JAX_COMPILATION_CACHE_DIR = configure_jax_runtime(precision="float64")

import jax
import numpy as np

from drbx.geometry.fci_control_volumes import compile_local_control_volume_geometry

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))
import shifted_torus_4field_mms_helpers as shifted_mms  # noqa: E402
from shifted_torus_4field_cutwall_geometry import (  # noqa: E402
    BOX_THETA_CENTER,
    BOX_THETA_HALF_WIDTH,
    BOX_X_FRACTION_RANGE,
    BOX_ZETA_RANGE,
    MESH_AXIS_NAMES,
    _GAUSS2_NODES,
    _GAUSS3_NODES,
    _GAUSS3_WEIGHTS,
    _box_bounds,
    _build_closed_box_control_volume_cells,
    _build_closed_box_control_volume_faces,
    _build_closed_box_embedded_control_volume_geometry,
    _build_global_closed_box_control_volume_topology,
    _build_shifted_torus_regular_boundary_closure,
    _build_stacked_embedded_control_volume_geometry,
    _closed_box_fluid_moments_3point,
    _closed_box_irregular_storage_mask,
    _compact_face_reconstruction_owner_mask,
    _dilate_reconstruction_owner_mask,
    _face_patch_quadrature_numpy,
    _integrate_shifted_torus_rectangular_moments,
    _intrinsic_reconstruction_owner_mask,
    _open_face_rectangles_numpy,
    _pad_control_volume_face_rows,
    _pad_embedded_control_volume_geometry,
    _pad_quadratic_reconstruction,
    _print_shifted_torus_radial_moment_reproduction,
    _select_closed_box_control_volume_owners,
    _shape_from_resolution,
    _shifted_torus_cartesian_from_logical,
    _shifted_torus_curvature_at_logical_points,
    _shifted_torus_metric_payload_jax,
    _shifted_torus_metric_payload_numpy,
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
    _shifted_torus_exact_field_and_gradient_at_logical_points,
    _shifted_torus_exact_field_at_logical_points,
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
    _control_volume_operator_category_masks,
    _fit_operator_order,
    _masked_field_error_statistics,
    _masked_state_error_statistics,
    _operator_category_statistics,
    _print_control_volume_geometry_summary,
    _print_state_error_statistics,
    _resolution_step_count,
    _state_error_statistics,
    _volume_weighted_field_error_statistics,
    _volume_weighted_state_error_statistics,
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
        global_shape=global_shape, halo_width=2,
    )
    baseline_ids: set[int] | None = None
    baseline_wall_rows: int | None = None
    for shard_counts in ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2)):
        geometry = _build_stacked_embedded_control_volume_geometry(
            global_shape=global_shape,
            shard_counts=shard_counts,
            halo_width=2,
            enable_merging=True,
        )
        rows = geometry.irregular_faces
        active = np.asarray(rows.active, dtype=bool)
        face_id = np.asarray(rows.global_face_id, dtype=np.int64)
        logical = face_id[active & (face_id >= 0)]
        assert logical.size == np.unique(logical).size
        current_ids = set(int(value) for value in logical)
        wall_rows = int(np.count_nonzero(active & (face_id < 0)))
        if baseline_ids is None:
            baseline_ids, baseline_wall_rows = current_ids, wall_rows
        else:
            assert current_ids == baseline_ids
            assert wall_rows == baseline_wall_rows
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
        "--skip-operator-phi-solve",
        action="store_true",
        help=(
            "Diagnostic-only: use projected exact phi in the full-RHS kernel "
            "and skip the separate phi inversion check while retaining all "
            "spatial operator kernels."
        ),
    )
    parser.add_argument(
        "--debug-operator-failures",
        action="store_true",
        help=(
            "Print focused radial-boundary, compact-face, and reconstruction "
            "failure attribution diagnostics during --operator-convergence-only."
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

    if not args.skip_runtime_info:
        _print_runtime_info()
    if bool(args.operator_convergence_only):
        run_shifted_torus_control_volume_operator_convergence(
            resolutions=[int(value) for value in args.resolutions],
            shard_counts=tuple(int(value) for value in args.shard_counts),
            halo_width=int(args.halo_width),
            rho_star_value=float(args.rho_star),
            enable_agglomeration=bool(args.enable_agglomeration),
            minimum_order=float(args.minimum_order),
            check_phi_solve=not bool(args.skip_operator_phi_solve),
            debug_operator_failures=bool(args.debug_operator_failures),
        )
        return
    if bool(args.debug_operator_failures):
        parser.error(
            "--debug-operator-failures requires --operator-convergence-only"
        )
    run_shifted_torus_4field_cutwall_convergence(
        resolutions=[int(value) for value in args.resolutions],
        shard_counts=tuple(int(value) for value in args.shard_counts),
        halo_width=int(args.halo_width),
        final_time=float(args.final_time),
        base_steps=int(args.base_steps),
        rho_star_value=float(args.rho_star),
        plot=bool(args.plot),
        plot_path=args.plot_path,
        show_progress=bool(args.show_progress),
        enable_agglomeration=bool(args.enable_agglomeration),
        minimum_order=float(args.minimum_order),
    )


if __name__ == "__main__":
    main()
