from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

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


def _write_public_benchmark_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_public_benchmark_record(root / "TCV_forward_field.nc")
    _write_public_benchmark_sample_geometry(root / "TCV_ortho.nc")
    _write_public_benchmark_vgrid(root / "vgrid.nc")
    _write_public_benchmark_snapshot(root / "snaps00000.nc")


def _write_public_benchmark_record(path: Path) -> None:
    with Dataset(path, "w") as dataset:
        for diagnostic_name in ("FHRP", "LFS-LP", "HFS-LP"):
            diagnostic = dataset.createGroup(diagnostic_name)
            observables = diagnostic.createGroup("observables")
            for observable_name, base in (
                ("density", 1.0e19),
                ("electron_temp", 15.0),
                ("ion_temp", 12.0),
                ("potential", 8.0),
                ("current", 2.0),
                ("vfloat", 5.0),
            ):
                observable = observables.createGroup(observable_name)
                observable.createDimension("points", 4)
                value = observable.createVariable("value", "f8", ("points",))
                error = observable.createVariable("error", "f8", ("points",))
                position = observable.createVariable("Rsep_omp", "f8", ("points",))
                value.units = "arb"
                error.units = "arb"
                position.units = "cm"
                value[:] = np.asarray([base, 0.8 * base, 0.65 * base, 0.5 * base], dtype=np.float64)
                error[:] = np.asarray([0.08 * base, 0.06 * base, 0.05 * base, 0.04 * base], dtype=np.float64)
                position[:] = np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float64)


def _write_public_benchmark_sample_geometry(path: Path) -> None:
    with Dataset(path, "w") as dataset:
        magnetic = dataset.createGroup("Magnetic_geometry")
        magnetic.createDimension("R", 12)
        magnetic.createDimension("Z", 16)
        r_values = np.linspace(0.7, 1.2, 12, dtype=np.float64)
        z_values = np.linspace(-0.6, 0.6, 16, dtype=np.float64)
        rr, zz = np.meshgrid(r_values, z_values)
        magnetic.createVariable("R", "f8", ("R",))[:] = r_values
        magnetic.createVariable("Z", "f8", ("Z",))[:] = z_values
        magnetic.createVariable("psi", "f8", ("Z", "R"))[:] = (rr - 0.95) ** 2 + 0.6 * zz**2
        magnetic.createVariable("btor", "f8", ("Z", "R"))[:] = 1.0 + 0.2 * rr

        divertor = dataset.createGroup("divertor_polygon")
        divertor.createDimension("N_points", 5)
        divertor.createVariable("R_points", "f8", ("N_points",))[:] = np.asarray([0.72, 1.18, 1.18, 0.72, 0.72])
        divertor.createVariable("Z_points", "f8", ("N_points",))[:] = np.asarray([-0.52, -0.52, 0.52, 0.52, -0.52])

        exclusion = dataset.createGroup("exclusion_polygon")
        exclusion.createDimension("N_points", 5)
        exclusion.createVariable("R_points", "f8", ("N_points",))[:] = np.asarray([0.82, 1.08, 1.08, 0.82, 0.82])
        exclusion.createVariable("Z_points", "f8", ("N_points",))[:] = np.asarray([-0.25, -0.25, 0.25, 0.25, -0.25])


def _write_public_benchmark_vgrid(path: Path) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("dim_nl", 12)
        dataset.createDimension("dim_ngb", 3)
        dataset.createDimension("dim_infosize", 2)
        li = dataset.createVariable("li", "i4", ("dim_nl",))
        lj = dataset.createVariable("lj", "i4", ("dim_nl",))
        lperp = dataset.createVariable("lperp", "f8", ("dim_nl", "dim_ngb", "dim_ngb"))
        dst = dataset.createVariable("dst", "f8", ("dim_nl", "dim_ngb", "dim_ngb"))
        info = dataset.createVariable("info", "i4", ("dim_infosize", "dim_nl"))
        x_index = np.repeat(np.arange(3, dtype=np.int32), 4)
        y_index = np.tile(np.arange(4, dtype=np.int32), 3)
        li[:] = 10 + x_index
        lj[:] = 40 + y_index
        lperp[:] = 0.0
        dst[:] = 0.0
        info[:] = 0


def _write_public_benchmark_snapshot(path: Path) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("dim_tau", 3)
        dataset.createDimension("dim_vgrid", 12)
        dataset.createDimension("dim_perpghost", 1)
        tau = dataset.createVariable("tau", "f8", ("dim_tau",))
        tau[:] = np.asarray([0.0, 0.25, 0.5], dtype=np.float64)
        base = np.arange(12, dtype=np.float64)[None, :]
        for name, shift in (("logne", 0.2), ("logte", 0.4), ("logti", 0.5), ("uparx", 0.1), ("potxx", 0.3)):
            variable = dataset.createVariable(name, "f8", ("dim_tau", "dim_vgrid"))
            variable[:] = shift + base + np.asarray([[0.0], [0.1], [0.2]])
            ghost = dataset.createVariable(f"{name}_perpghost", "f8", ("dim_tau", "dim_perpghost"))
            ghost[:] = 0.0


def test_tcv_x21_scaffold_preview_generates_artifacts(tmp_path: Path) -> None:
    artifacts = create_tcv_x21_scaffold_package(
        reference_root=tmp_path,
        output_root=tmp_path / "output",
    )

    for path in (
        artifacts.manifest_json_path,
        artifacts.input_report_json_path,
        artifacts.validation_contract_json_path,
        artifacts.benchmark_data_report_json_path,
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
        if path is None:
            continue
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
    assert artifacts.benchmark_data_report_json_path is None

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


def test_tcv_x21_scaffold_public_benchmark_mode_generates_real_benchmark_bundle(tmp_path: Path) -> None:
    _write_reference_tree(tmp_path)
    benchmark_root = tmp_path / "benchmark"
    _write_public_benchmark_root(benchmark_root)
    artifacts = create_tcv_x21_scaffold_package(
        reference_root=tmp_path,
        output_root=tmp_path / "output",
        benchmark_data_root=benchmark_root,
    )

    assert artifacts.benchmark_data_report_json_path is not None
    assert artifacts.benchmark_data_report_json_path.exists()
    manifest = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
    assert manifest["preview_mode"] is False
    assert manifest["workdir_mode"] == "external_benchmark_data"
    assert manifest["artifacts"]["benchmark_data_report_json"].endswith(
        "data/tokamak_tcv_x21_scaffold_benchmark_data_report.json"
    )

    input_report = json.loads(artifacts.input_report_json_path.read_text(encoding="utf-8"))
    assert input_report["benchmark_data_mode"] == "external_public_tcv_x21_sample"

    benchmark_report = json.loads(artifacts.benchmark_data_report_json_path.read_text(encoding="utf-8"))
    assert benchmark_report["resolved_sample_field_name"] == "potxx"
    assert benchmark_report["present_files"]["TCV_forward_field.nc"] is True

    profile_report = json.loads(artifacts.profile_report_json_path.read_text(encoding="utf-8"))
    assert profile_report["parse_status"] == "ok"
    assert profile_report["diagnostics"]["FHRP"]["density"]["positions"] == [0.0, 0.5, 1.0, 1.5]

    analysis = json.loads(artifacts.analysis_json_path.read_text(encoding="utf-8"))
    assert analysis["field_name"] == "potxx"
