"""Fast contracts for the standalone HSX parallel heat-spot driver."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_hsx_sanity as sanity
import run_hsx_perpendicular_convergence as convergence


def _geometry():
    return SimpleNamespace(
        shape=(5, 8, 7),
        grid=SimpleNamespace(
            x=SimpleNamespace(centers=np.linspace(0.1, 0.9, 5)),
            y=SimpleNamespace(centers=np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)),
            z=SimpleNamespace(centers=np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)),
        ),
    )


def test_defaults_are_the_production_32_cubed_contract():
    args = sanity.build_parser().parse_args([])
    contract = sanity.production_geometry_kwargs(args)
    assert args.resolution == (32, 32, 32)
    assert args.fit_sample_shape == (32, 32, 32)
    assert args.metric_mesh_shape == (32, 32, 32)
    assert args.halo_width == 2
    assert args.metric_spline_degree == 1
    assert args.makegrid_currents == (10722.0,) * 6 + (0.0,) * 6
    assert contract["makegrid_currents"] == args.makegrid_currents
    assert contract["construct_fci_maps"] is True
    assert contract["topology"] == "toroidal"
    assert sanity.effective_geometry_kwargs(args)["return_metric_evaluator"] is True


def test_heat_spot_is_positive_and_uses_both_perpendicular_widths():
    geometry = _geometry()
    common = dict(
        amplitude=0.1, center_u=0.35, center_v=np.pi, center_eta=np.pi,
        width_u=0.08, width_v=0.23, width_eta=np.pi,
    )
    state = sanity.make_heat_spot_state(geometry, **common)
    narrow_theta = sanity.make_heat_spot_state(
        geometry, **(common | {"width_v": 0.05})
    )
    assert tuple(state.Te.shape) == geometry.shape
    assert float(np.min(np.asarray(state.Te))) >= 1.0
    assert np.max(np.abs(np.asarray(state.Te) - np.asarray(narrow_theta.Te))) > 1.0e-4
    assert np.max(np.asarray(state.Te) - 1.0) > 0.01


def test_flux_function_initial_condition_is_positive_and_surface_constant():
    geometry = _geometry()
    flux_label = np.broadcast_to(
        np.linspace(0.0, 1.0, geometry.shape[0])[:, None, None],
        geometry.shape,
    )
    state = sanity.make_flux_function_state(
        geometry, flux_label, amplitude=0.1, center=0.0, width=0.1
    )
    values = np.asarray(state.Te)
    assert values.shape == geometry.shape
    assert np.all(values >= 1.0)
    assert np.allclose(values[:, 0, 0], values[:, -1, -1])


def test_convergence_launcher_refines_only_perpendicular_dimensions(tmp_path):
    args = convergence.build_parser().parse_args(
        ["--series-dir", str(tmp_path), "--perpendicular-resolutions", "32,40,48"]
    )
    command = convergence.command_for_case(args, 40)
    resolution = command[command.index("--resolution") + 1]
    fit_shape = command[command.index("--fit-sample-shape") + 1]
    metric_shape = command[command.index("--metric-mesh-shape") + 1]
    assert resolution == fit_shape == metric_shape == "40,40,32"
    assert command[command.index("--initial-condition") + 1] == "flux-function"
    assert convergence.case_directory(tmp_path, 40, 32).name == "nperp_040_neta_032"


def test_every_state_axpy_call_has_an_explicit_scale():
    """Keep the RK4 composition compatible with FciModelState.axpy's API."""

    tree = ast.parse((ROOT / "run_hsx_sanity.py").read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "axpy"
    ]
    assert calls
    assert all(any(keyword.arg == "scale" for keyword in call.keywords) for call in calls)


def test_plotting_is_isolated_in_drb_notebook():
    driver = (ROOT / "run_hsx_sanity.py").read_text(encoding="utf-8")
    assert "matplotlib" not in driver
    assert "savefig" not in driver
    assert "analyze_existing" not in driver

    notebook_path = ROOT / "hsx_sanity_plots.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["kernelspec"]["name"] == "drb"
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "history.npz" in sources
    assert "poincare_theta_u.npz" in sources
    assert "fourier_poincare_band" in sources
    assert "draw_logical_panel" in sources
    assert "theta_edges_normalized" in sources
    assert "widgets.SelectionSlider" in sources
    assert "widgets.Play" in sources
    assert "update_heat_time" in sources
    assert "u * np.cos" not in sources
    assert "vmec_surface_rz" not in sources
    assert "run_heat_spot" not in sources

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(notebook_path), "exec")
