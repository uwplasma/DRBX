from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib import pyplot as plt

import analyze_density
import analyze_rhs_common
import analyze_te
import analyze_ti
import analyze_vi
import analyze_vorticity
import plot_density_3D
from jax_drb.native.fci_drb_EB_rhs import FciDrbEBState


FIELD_CASES = (
    (
        analyze_density,
        [
            "ExB bracket",
            "parallel Ve compression",
            "n * parallel Ve compression",
            "Ve * parallel density gradient",
            "curvature",
            "parallel diffusion",
            "perp diffusion",
        ],
        [
            "ExB bracket",
            "parallel Ve compression",
            "n * parallel Ve compression",
            "Ve * parallel density gradient",
            "curvature",
            "parallel diffusion",
            "perp diffusion",
        ],
        ("ExB bracket", "parallel Ve compression", "curvature", "parallel diffusion", "perp diffusion"),
    ),
    (
        analyze_te,
        [
            "exb",
            "parallel_advection",
            "curvature_pressure",
            "curvature_temperature",
            "curvature_potential",
            "parallel_current_density",
            "parallel_ve",
            "parallel_diffusion",
            "perp_diffusion",
        ],
        [
            "ExB",
            "-Ve d_parallel Te",
            "curvature pressure",
            "curvature temperature",
            "curvature potential",
            "0.71 parallel current",
            "-n * parallel Ve",
            "parallel diffusion",
            "perp diffusion",
        ],
        (
            "exb",
            "parallel_advection",
            "curvature_pressure",
            "curvature_temperature",
            "curvature_potential",
            "parallel_current_density",
            "parallel_ve",
            "parallel_diffusion",
            "perp_diffusion",
        ),
    ),
    (
        analyze_ti,
        [
            "exb",
            "parallel_advection",
            "curvature_pressure",
            "curvature_temperature",
            "curvature_potential",
            "parallel_current_density",
            "parallel_vi",
            "parallel_diffusion",
            "perp_diffusion",
        ],
        [
            "ExB",
            "-Vi d_parallel Ti",
            "curvature pressure",
            "curvature temperature",
            "curvature potential",
            "parallel current",
            "-n * parallel Vi",
            "parallel diffusion",
            "perp diffusion",
        ],
        (
            "exb",
            "parallel_advection",
            "curvature_pressure",
            "curvature_temperature",
            "curvature_potential",
            "parallel_current_density",
            "parallel_vi",
            "parallel_diffusion",
            "perp_diffusion",
        ),
    ),
    (
        analyze_vi,
        ["exb", "parallel_advection", "parallel_pressure", "parallel_diffusion", "perp_diffusion"],
        ["ExB", "-Vi d_parallel Vi", "pressure gradient", "parallel diffusion", "perp diffusion"],
        ("exb", "parallel_advection", "parallel_pressure", "parallel_diffusion", "perp_diffusion"),
    ),
    (
        analyze_vorticity,
        [
            "exb",
            "parallel_advection",
            "parallel_current_density",
            "curvature_pressure",
            "parallel_diffusion",
            "perp_diffusion",
        ],
        [
            "ExB",
            "-Vi d_parallel vorticity",
            "parallel current",
            "curvature pressure",
            "parallel diffusion",
            "perp diffusion",
        ],
        (
            "exb",
            "parallel_advection",
            "parallel_current_density",
            "curvature_pressure",
            "parallel_diffusion",
            "perp_diffusion",
        ),
    ),
)


def _copy_step_subset(source_dir: Path, target_dir: Path, count: int = 3) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    step_files = sorted(source_dir.glob("step_*.npz"))
    if len(step_files) < count:
        raise RuntimeError(f"need at least {count} step dumps in {source_dir}")
    for source_file in step_files[:count]:
        shutil.copy2(source_file, target_dir / source_file.name)
    return target_dir


def test_resolve_step_dump_dir_prefers_nested_step_dumps(tmp_path: Path) -> None:
    source_step_dir = Path(__file__).resolve().parent / "EB_long" / "step_dumps"
    nested_step_dir = tmp_path / "EB_outputs" / "step_dumps"
    nested_step_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sorted(source_step_dir.glob("step_*.npz"))[0], nested_step_dir / "step_00000.npz")

    resolved = analyze_rhs_common.resolve_step_dump_dir("EB_test", tmp_path / "EB_outputs")

    assert resolved == nested_step_dir

    resolved_from_file = analyze_rhs_common.resolve_step_dump_dir(
        "EB_test",
        nested_step_dir / "step_00000.npz",
    )
    assert resolved_from_file == nested_step_dir


def _build_state(history_index: int, histories) -> FciDrbEBState:
    density_history, phi_history, te_history, ti_history, vi_history, ve_history, vorticity_history = histories
    return FciDrbEBState(
        density=np.asarray(density_history[history_index], dtype=np.float64),
        phi=np.asarray(phi_history[history_index], dtype=np.float64),
        Te=np.asarray(te_history[history_index], dtype=np.float64),
        Ti=np.asarray(ti_history[history_index], dtype=np.float64),
        Vi=np.asarray(vi_history[history_index], dtype=np.float64),
        Ve=np.asarray(ve_history[history_index], dtype=np.float64),
        vorticity=np.asarray(vorticity_history[history_index], dtype=np.float64),
    )


def _build_history_context(step_dump_dir: Path):
    (
        times,
        density_history,
        phi_history,
        te_history,
        ti_history,
        vi_history,
        ve_history,
        vorticity_history,
    ) = analyze_rhs_common.load_eb_blob_step_history(step_dump_dir)
    context = analyze_rhs_common.build_eb_blob_context(density_history, 1.0e-5)
    histories = (density_history, phi_history, te_history, ti_history, vi_history, ve_history, vorticity_history)
    return times, histories, context


def test_shared_cell_selection_helper_returns_expected_cells() -> None:
    geometry = SimpleNamespace(
        grid=SimpleNamespace(
            x=SimpleNamespace(centers=np.array([0.10, 0.41, 0.53, 0.81], dtype=np.float64)),
            y=SimpleNamespace(centers=np.array([0.50, 2.20, 3.05, 3.90, 5.80], dtype=np.float64)),
            z=SimpleNamespace(centers=np.array([0.20, 1.10, 2.40, 3.20], dtype=np.float64)),
        )
    )

    selections = analyze_rhs_common.build_cell_selections(geometry)

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


@pytest.mark.parametrize("module, expected_keys, expected_labels, expected_sum_keys", FIELD_CASES)
def test_field_term_specs_follow_rhs_order(module, expected_keys, expected_labels, expected_sum_keys) -> None:
    assert [spec.key for spec in module.FIELD_SPEC.term_specs] == expected_keys
    assert [spec.label for spec in module.FIELD_SPEC.term_specs] == expected_labels
    assert module.FIELD_SPEC.sum_keys == expected_sum_keys


@pytest.mark.parametrize("module, expected_keys, expected_labels, expected_sum_keys", FIELD_CASES)
def test_field_component_sums_match_total_rhs(module, expected_keys, expected_labels, expected_sum_keys, tmp_path: Path) -> None:
    del expected_keys, expected_labels, expected_sum_keys
    source_step_dir = Path(__file__).resolve().parent / "EB_long" / "step_dumps"
    step_dump_dir = _copy_step_subset(source_step_dir, tmp_path / f"{module.FIELD_SPEC.output_suffix}_sums")
    times, histories, context = _build_history_context(step_dump_dir)
    cell_selections = analyze_rhs_common.build_cell_selections(context.geometry)
    state = _build_state(0, histories)

    term_fields = module.FIELD_SPEC.evaluate_terms(
        state,
        geometry=context.geometry,
        parameters=context.parameters,
        boundary_condition_builder=context.boundary_condition_builder,
        cut_wall_geometry=context.cut_wall_geometry,
        cut_wall_bc=context.cut_wall_bc,
        curvature_coefficients=context.curvature_coefficients,
    )
    total_from_terms = np.sum([np.asarray(term_fields[key], dtype=np.float64) for key in module.FIELD_SPEC.sum_keys], axis=0)
    np.testing.assert_allclose(total_from_terms, np.asarray(term_fields["total_rhs"], dtype=np.float64))
    assert len(times) == histories[0].shape[0]
    assert len(cell_selections) == 5


def test_shared_rhs_layout_builder_smoke(tmp_path: Path) -> None:
    source_step_dir = Path(__file__).resolve().parent / "EB_long" / "step_dumps"
    step_dump_dir = _copy_step_subset(source_step_dir, tmp_path / "layout_smoke")
    times, histories, context = _build_history_context(step_dump_dir)
    cell_selections = analyze_rhs_common.build_cell_selections(context.geometry)
    state = _build_state(0, histories)
    term_fields = analyze_te.FIELD_SPEC.evaluate_terms(
        state,
        geometry=context.geometry,
        parameters=context.parameters,
        boundary_condition_builder=context.boundary_condition_builder,
        cut_wall_geometry=context.cut_wall_geometry,
        cut_wall_bc=context.cut_wall_bc,
        curvature_coefficients=context.curvature_coefficients,
    )
    total_from_terms = np.sum([np.asarray(term_fields[key], dtype=np.float64) for key in analyze_te.FIELD_SPEC.sum_keys], axis=0)
    np.testing.assert_allclose(total_from_terms, np.asarray(term_fields["total_rhs"], dtype=np.float64))

    cell_histories = analyze_rhs_common.collect_rhs_histories(
        np.asarray(times, dtype=np.float64),
        np.asarray(histories[0], dtype=np.float64),
        np.asarray(histories[1], dtype=np.float64),
        np.asarray(histories[2], dtype=np.float64),
        np.asarray(histories[3], dtype=np.float64),
        np.asarray(histories[4], dtype=np.float64),
        np.asarray(histories[5], dtype=np.float64),
        np.asarray(histories[6], dtype=np.float64),
        field_spec=analyze_te.FIELD_SPEC,
        geometry=context.geometry,
        parameters=context.parameters,
        boundary_condition_builder=context.boundary_condition_builder,
        cut_wall_geometry=context.cut_wall_geometry,
        cut_wall_bc=context.cut_wall_bc,
        curvature_coefficients=context.curvature_coefficients,
        cell_selections=cell_selections,
    )

    fig, trace_axes, cross_ax, colorbar_ax = analyze_rhs_common.build_rhs_terms_figure(
        np.asarray(times, dtype=np.float64),
        cell_histories,
        np.asarray(histories[0][0, :, :, cell_selections[0].z_index], dtype=np.float64),
        context.geometry,
        cell_selections,
        cell_selections[0].z_index,
        field_spec=analyze_te.FIELD_SPEC,
        title="layout smoke test",
    )
    output_path = step_dump_dir / "layout_smoke.png"
    fig.savefig(output_path, dpi=170)
    plt.close(fig)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert len(trace_axes) == 5
    assert trace_axes[0].get_subplotspec().colspan.start == 0
    assert trace_axes[3].get_subplotspec().colspan.start == 1
    assert trace_axes[2].get_xlabel() == "time"
    assert trace_axes[4].get_xlabel() == "time"
    assert cross_ax.name == "polar"
    assert cross_ax.get_subplotspec().colspan.start == 2
    assert colorbar_ax is not None
    assert colorbar_ax != cross_ax


@pytest.mark.parametrize("module, expected_keys, expected_labels, expected_sum_keys", FIELD_CASES)
def test_field_wrapper_smoke_writes_png(module, expected_keys, expected_labels, expected_sum_keys, tmp_path: Path) -> None:
    del expected_keys, expected_labels, expected_sum_keys
    source_step_dir = Path(__file__).resolve().parent / "EB_long" / "step_dumps"
    step_dump_dir = _copy_step_subset(source_step_dir, tmp_path / f"{module.FIELD_SPEC.output_suffix}_wrapper")
    extra_args = ["--frame-stride", "5"] if module.FIELD_SPEC.output_suffix in {"density", "te", "ti", "ve"} else []
    output_path = module.main(
        [
            "--run-name",
            "EB_perp_diffusion",
            "--output-path",
            str(step_dump_dir),
            "--perp-diffusion",
            "1.0e-5",
            *extra_args,
        ]
    )
    assert output_path.exists()
    assert output_path.name == f"EB_perp_diffusion_{module.FIELD_SPEC.output_suffix}_rhs_terms.png"
    if module.FIELD_SPEC.output_suffix in {"density", "te", "ti", "ve"}:
        profile_path = output_path.parent / output_path.name.replace("_rhs_terms.png", "_profile.gif")
        assert profile_path.exists()


def test_plot_density_3d_smoke_writes_gif(tmp_path: Path) -> None:
    source_step_dir = Path(__file__).resolve().parent / "EB_long" / "step_dumps"
    step_dump_dir = _copy_step_subset(source_step_dir, tmp_path / "density_3d")

    output_path = plot_density_3D.main(
        [
            "--run-name",
            "EB_perp_diffusion",
            "--output-path",
            str(step_dump_dir),
            "--frame-stride",
            "5",
        ]
    )

    assert output_path.exists()
    assert output_path.name == "EB_perp_diffusion_density_volume.gif"
    assert output_path.parent == step_dump_dir
    assert output_path.stat().st_size > 0
