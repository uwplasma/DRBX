from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt

from jax_drb.native.fci_drb_EB_rhs import FciDrbEBState
from jax_drb.native.fci_boundaries import (  # noqa: E402
    BoundaryConditionBuilder,
    CoordinateFaceValueReconstructor3D,
    CoordinateNormalDerivativeConstructor3D,
    CutWallBC3D,
    CutWallGeometry3D,
)
from test_shifted_torus_EB_blob import _build_eb_blob_geometry, _build_eb_blob_parameters, _build_eb_boundary_conditions  # noqa: E402

import analyze_ve


def _copy_step_subset(source_dir: Path, target_dir: Path, count: int = 3) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    step_files = sorted(source_dir.glob("step_*.npz"))
    if len(step_files) < count:
        raise RuntimeError(f"need at least {count} step dumps in {source_dir}")
    for source_file in step_files[:count]:
        shutil.copy2(source_file, target_dir / source_file.name)
    return target_dir


def _build_boundary_condition_builder(geometry):
    coordinate_face_reconstructor = CoordinateFaceValueReconstructor3D()
    coordinate_normal_derivative_constructor = CoordinateNormalDerivativeConstructor3D.from_geometry(geometry)
    return BoundaryConditionBuilder(
        lambda state, geom, periodic_axes, cut_wall_geometry, cut_wall_bc: _build_eb_boundary_conditions(
            state,
            geom,
            periodic_axes,
            cut_wall_geometry,
            cut_wall_bc,
            face_reconstructor=coordinate_face_reconstructor,
            normal_derivative_constructor=coordinate_normal_derivative_constructor,
        )
    )


def test_ve_cell_selection_uses_nearest_periodic_indices() -> None:
    geometry = SimpleNamespace(
        grid=SimpleNamespace(
            x=SimpleNamespace(centers=np.array([0.10, 0.41, 0.53, 0.81], dtype=np.float64)),
            y=SimpleNamespace(centers=np.array([0.50, 2.20, 3.05, 3.90, 5.80], dtype=np.float64)),
            z=SimpleNamespace(centers=np.array([0.20, 1.10, 2.40, 3.20], dtype=np.float64)),
        )
    )

    selections = analyze_ve._build_ve_cell_selections(geometry)

    assert [selection.name for selection in selections] == [
        "midplane",
        "plus_45",
        "minus_45",
        "plus_90",
        "minus_90",
    ]
    assert all(selection.x_index == 2 for selection in selections)
    assert all(selection.z_index == 3 for selection in selections)
    assert [selection.theta_index for selection in selections] == [2, 3, 1, 3, 1]
    assert np.isclose(selections[0].x_value, 0.53)
    assert np.isclose(selections[0].theta_value, 3.05)


def test_ve_term_specs_follow_rhs_order() -> None:
    assert [spec.key for spec in analyze_ve.VE_RHS_TERM_SPECS] == [
        "exb",
        "parallel_advection",
        "ve_nu_j_parallel",
        "parallel_phi",
        "parallel_pe_over_density",
        "parallel_te",
        "parallel_diffusion",
        "perp_diffusion",
    ]
    assert [spec.label for spec in analyze_ve.VE_RHS_TERM_SPECS] == [
        "ExB",
        "-Ve d_parallel Ve",
        "mi_over_me * Ve_nu j_parallel",
        "mi_over_me * d_parallel phi",
        "mi_over_me * -(d_parallel Pe)/n",
        "mi_over_me * -0.71 d_parallel Te",
        "parallel diffusion",
        "perp diffusion",
    ]


def test_ve_rhs_terms_plot_smoke(tmp_path: Path) -> None:
    source_step_dir = Path(__file__).resolve().parent / "EB_long" / "step_dumps"
    step_dump_dir = _copy_step_subset(source_step_dir, tmp_path / "ve_rhs_smoke")

    times, density_history, phi_history, te_history, ti_history, vi_history, ve_history, vorticity_history = analyze_ve._load_eb_blob_step_history(
        step_dump_dir
    )
    geometry = _build_eb_blob_geometry(tuple(int(size) for size in density_history.shape[1:]), construct_fci_maps=False)
    parameters = _build_eb_blob_parameters(1.0e-5)
    boundary_condition_builder = _build_boundary_condition_builder(geometry)
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()
    curvature_coefficients = analyze_ve.build_curvature_coefficients(
        geometry,
        periodic_axes=analyze_ve.PERIODIC_AXES,
        axis_regular_axes=analyze_ve.AXIS_REGULAR_AXES,
    )
    cell_selections = analyze_ve._build_ve_cell_selections(geometry)

    state = FciDrbEBState(
        density=np.asarray(density_history[0], dtype=np.float64),
        phi=np.asarray(phi_history[0], dtype=np.float64),
        Te=np.asarray(te_history[0], dtype=np.float64),
        Ti=np.asarray(ti_history[0], dtype=np.float64),
        Vi=np.asarray(vi_history[0], dtype=np.float64),
        Ve=np.asarray(ve_history[0], dtype=np.float64),
        vorticity=np.asarray(vorticity_history[0], dtype=np.float64),
    )
    term_fields = analyze_ve._ve_rhs_term_fields(
        state,
        geometry=geometry,
        parameters=parameters,
        boundary_condition_builder=boundary_condition_builder,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        curvature_coefficients=curvature_coefficients,
    )
    term_arrays = [np.asarray(term_fields[spec.key], dtype=np.float64) for spec in analyze_ve.VE_RHS_TERM_SPECS]
    total_from_terms = np.sum(term_arrays, axis=0)
    np.testing.assert_allclose(total_from_terms, np.asarray(term_fields["total_rhs"], dtype=np.float64))

    cell_histories = analyze_ve._collect_ve_rhs_histories(
        np.asarray(times, dtype=np.float64),
        np.asarray(density_history, dtype=np.float64),
        np.asarray(phi_history, dtype=np.float64),
        np.asarray(te_history, dtype=np.float64),
        np.asarray(ti_history, dtype=np.float64),
        np.asarray(vi_history, dtype=np.float64),
        np.asarray(ve_history, dtype=np.float64),
        np.asarray(vorticity_history, dtype=np.float64),
        geometry=geometry,
        parameters=parameters,
        boundary_condition_builder=boundary_condition_builder,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        curvature_coefficients=curvature_coefficients,
        cell_selections=cell_selections,
    )

    fig, trace_axes, cross_ax, colorbar_ax = analyze_ve._build_ve_rhs_terms_figure(
        np.asarray(times, dtype=np.float64),
        cell_histories,
        np.asarray(density_history[0, :, :, cell_selections[0].z_index], dtype=np.float64),
        geometry,
        cell_selections,
        cell_selections[0].z_index,
        title="smoke test",
    )
    output_path = step_dump_dir / "ve_rhs_smoke.png"
    fig.savefig(output_path, dpi=170)
    plt.close(fig)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert len(trace_axes) == 5
    assert cross_ax.name == "polar"
    assert cross_ax.get_subplotspec().colspan.start == 2
    assert cross_ax.get_subplotspec().rowspan.start == 0
    assert cross_ax.get_subplotspec().rowspan.stop == 3
    assert trace_axes[0].get_subplotspec().colspan.start == 0
    assert trace_axes[3].get_subplotspec().colspan.start == 1
    assert trace_axes[4].get_subplotspec().rowspan.start == 1
    assert colorbar_ax is not None
    assert colorbar_ax != cross_ax
