from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import create_tcv_x21_scaffold_package


def _write_reference_tree(root: Path) -> None:
    target = root / "examples" / "tokamak-3D" / "tcv-x21" / "data"
    target.mkdir(parents=True, exist_ok=True)
    (target / "BOUT.inp").write_text(
        "\n".join(
            [
                "nout = 4",
                "timestep = 0.25",
                "MZ = 32",
                "zperiod = 1.0",
                "",
                "[mesh]",
                "file = tokamak.nc",
                "nx = 32",
                "ny = 64",
                "nz = 32",
                "",
                "[solver]",
                "type = cvode",
                "rtol = 1e-6",
                "atol = 1e-9",
                "",
                "[model]",
                'components = ("e", "i", "vorticity")',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_tcv_x21_scaffold_preview_generates_artifacts(tmp_path: Path) -> None:
    artifacts = create_tcv_x21_scaffold_package(
        reference_root=tmp_path,
        output_root=tmp_path / "output",
    )

    for path in (
        artifacts.manifest_json_path,
        artifacts.input_report_json_path,
        artifacts.validation_contract_json_path,
        artifacts.observable_report_json_path,
        artifacts.profile_report_json_path,
        artifacts.profile_arrays_npz_path,
        artifacts.profile_plot_png_path,
        artifacts.arrays_npz_path,
        artifacts.analysis_json_path,
        artifacts.snapshots_png_path,
        artifacts.poster_png_path,
        artifacts.movie_gif_path,
    ):
        assert path.exists()

    manifest = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
    assert manifest["case_name"] == "tokamak_tcv_x21_escalation"
    assert manifest["geometry_family"] == "diverted_tokamak_3d"
    assert manifest["benchmark_adapter"] == "tcv_x21"
    assert manifest["capability_tier"] == "scaffolded_reference_backed"
    assert manifest["preview_mode"] is True
    assert manifest["workdir_mode"] == "synthetic_preview"
    assert manifest["reference_exists"] is False
    assert manifest["artifacts"]["input_report_json"].endswith("data/tokamak_tcv_x21_scaffold_input_report.json")
    assert manifest["artifacts"]["validation_contract_json"].endswith(
        "data/tokamak_tcv_x21_scaffold_validation_contract.json"
    )
    assert manifest["artifacts"]["observable_report_json"].endswith(
        "data/tokamak_tcv_x21_scaffold_observable_report.json"
    )
    assert manifest["artifacts"]["profile_report_json"].endswith(
        "data/tokamak_tcv_x21_scaffold_profile_report.json"
    )
    assert manifest["artifacts"]["profile_arrays_npz"].endswith(
        "data/tokamak_tcv_x21_scaffold_profile_arrays.npz"
    )
    assert manifest["artifacts"]["profile_plot_png"].endswith(
        "images/tokamak_tcv_x21_scaffold_profiles.png"
    )
    assert manifest["artifacts"]["movie_gif"].endswith("movies/tokamak_tcv_x21_scaffold.gif")

    input_report = json.loads(artifacts.input_report_json_path.read_text(encoding="utf-8"))
    assert input_report["available"] is False
    assert input_report["parse_status"] == "missing_input"
    assert input_report["compare_variables"] == ["Ne", "Pe", "Pi", "NVi", "phi"]

    validation_contract = json.loads(artifacts.validation_contract_json_path.read_text(encoding="utf-8"))
    assert validation_contract["geometry_family"] == "diverted_tokamak_3d"
    assert validation_contract["benchmark_adapter"] == "tcv_x21"
    assert validation_contract["diagnostic_layer"] == "benchmark_adapter_on_general_3d_geometry"
    assert validation_contract["benchmark"]["name"] == "TCV-X21 diverted L-mode reference case"
    assert validation_contract["promotion_gates"][0]["name"] == "scaffold_gate"
    assert validation_contract["diagnostic_sets"][0]["name"] == "FHRP"
    assert "density" in validation_contract["diagnostic_sets"][0]["observables"]
    observable_report = json.loads(artifacts.observable_report_json_path.read_text(encoding="utf-8"))
    assert observable_report["geometry_family"] == "diverted_tokamak_3d"
    assert observable_report["observable_groups"][0]["families"][0]["kind"] == "profile"

    profile_report = json.loads(artifacts.profile_report_json_path.read_text(encoding="utf-8"))
    assert profile_report["available"] is True
    assert profile_report["parse_status"] == "ok"
    assert profile_report["diagnostics"]["FHRP"]["density"]["units"] == "1/m^3"
    assert len(profile_report["diagnostics"]["LFS-LP"]["current"]["mean"]) == 4


def test_tcv_x21_scaffold_marks_reference_tree_when_present(tmp_path: Path) -> None:
    _write_reference_tree(tmp_path)
    artifacts = create_tcv_x21_scaffold_package(
        reference_root=tmp_path,
        output_root=tmp_path / "output",
    )

    manifest = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
    assert manifest["reference_exists"] is True
    assert manifest["reference_input_path"].endswith("examples/tokamak-3D/tcv-x21/data/BOUT.inp")

    input_report = json.loads(artifacts.input_report_json_path.read_text(encoding="utf-8"))
    assert input_report["parse_status"] == "ok"
    assert input_report["run_config_status"] == "ok"
    assert input_report["time"] == {"nout": 4, "timestep": 0.25}
    assert input_report["mesh"]["file"] == "tokamak.nc"
    assert input_report["mesh"]["mz"] == 32
    assert input_report["solver"]["type"] == "cvode"
    assert input_report["components"]["labels"] == ["e", "i", "vorticity"]
    assert input_report["declared_components"] == ["e", "i", "vorticity"]

    validation_contract = json.loads(artifacts.validation_contract_json_path.read_text(encoding="utf-8"))
    assert validation_contract["geometry_family"] == "diverted_tokamak_3d"
    assert validation_contract["reference_inputs"]["input_exists"] is True
    assert validation_contract["reference_inputs"]["reference_helper_scripts"] == [
        "examples/tokamak-3D/tcv-x21/gather_data.py",
        "examples/tokamak-3D/tcv-x21/convert_to_tcvx21.py",
        "examples/tokamak-3D/tcv-x21/make_tcvx21_plots.py",
    ]

    profile_report = json.loads(artifacts.profile_report_json_path.read_text(encoding="utf-8"))
    assert profile_report["normalization"]["status"] == "physical_units"
    assert sorted(profile_report["diagnostics"]) == ["FHRP", "HFS-LP", "LFS-LP"]
