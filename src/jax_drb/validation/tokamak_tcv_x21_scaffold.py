from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from ..config.boutinp import BoutConfig, load_bout_input
from ..config.model import has_model_section, locate_model_section
from ..reference.cases import ReferenceCase, load_reference_cases
from ..runtime.run_config import RunConfiguration
from .diverted_tokamak_movie import (
    DivertedTokamakMovieArtifacts,
    DivertedTokamakGeometry,
    DivertedTokamakFieldHistory,
    assemble_tokamak_rank_history,
    create_diverted_tokamak_movie_package,
    load_diverted_tokamak_geometry,
)
from .geometry_adapter import build_geometry_adapter_contract, build_geometry_adapter_manifest
from .geometry_observables import (
    build_geometry_observable_report,
    profile_group_from_report,
    write_geometry_observable_report,
)
from .geometry_profiles import (
    build_diagnostic_profile_report,
    save_diagnostic_profile_summary_plot,
    write_diagnostic_profile_arrays_npz,
)

DEFAULT_TCV_X21_CASE_NAME = "tokamak_tcv_x21_escalation"


@dataclass(frozen=True)
class TcvX21ScaffoldArtifacts:
    manifest_json_path: Path
    input_report_json_path: Path
    validation_contract_json_path: Path
    observable_report_json_path: Path
    profile_report_json_path: Path
    profile_arrays_npz_path: Path
    profile_plot_png_path: Path
    arrays_npz_path: Path
    analysis_json_path: Path
    snapshots_png_path: Path
    poster_png_path: Path
    movie_gif_path: Path


@dataclass(frozen=True)
class TcvX21ReferenceStatus:
    case: ReferenceCase
    input_path: Path
    exists: bool


def resolve_tcv_x21_reference_case(
    reference_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    case_name: str = DEFAULT_TCV_X21_CASE_NAME,
) -> TcvX21ReferenceStatus:
    for case in load_reference_cases(manifest_path):
        if case.name == case_name:
            input_path = case.input_path(reference_root)
            return TcvX21ReferenceStatus(case=case, input_path=input_path, exists=input_path.exists())
    raise KeyError(f"Unknown reference case {case_name!r}")


def create_tcv_x21_scaffold_package(
    *,
    reference_root: str | Path,
    output_root: str | Path,
    case_name: str = DEFAULT_TCV_X21_CASE_NAME,
    case_label: str = "tokamak_tcv_x21_scaffold",
    field_name: str = "phi",
    workdir_in: str | Path | None = None,
    mesh_path: str | Path | None = None,
    fps: int = 10,
    frames_per_interval: int = 8,
) -> TcvX21ScaffoldArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    movies_dir = root / "movies"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    movies_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_tcv_x21_reference_case(reference_root, case_name=case_name)
    resolved_workdir = Path(workdir_in) if workdir_in is not None else None
    resolved_mesh = Path(mesh_path) if mesh_path is not None else None
    if resolved_workdir is not None and resolved_mesh is None:
        inferred_mesh = resolved_workdir / "tokamak.nc"
        if inferred_mesh.exists():
            resolved_mesh = inferred_mesh

    preview_mode = resolved_workdir is None or resolved_mesh is None

    if preview_mode:
        with tempfile.TemporaryDirectory(prefix="jax_drb_tcv_x21_preview_") as temp_dir:
            preview_root = Path(temp_dir)
            workdir = _write_synthetic_preview_workdir(preview_root, field_name=field_name)
            preview_artifacts = create_diverted_tokamak_movie_package(
                workdir=workdir.workdir,
                mesh_path=workdir.mesh_path,
                output_root=root,
                field_name=field_name,
                case_label=case_label,
                fps=fps,
                frames_per_interval=frames_per_interval,
            )
            return _finalize_scaffold_artifacts(
                preview_artifacts,
                workdir=workdir.workdir,
                mesh_path=workdir.mesh_path,
                output_root=root,
                data_dir=data_dir,
                case_label=case_label,
                field_name=field_name,
                reference_status=resolved,
                preview_mode=True,
                workdir_mode="synthetic_preview",
            )

    workdir = resolved_workdir
    if workdir is None or resolved_mesh is None:
        raise ValueError("workdir_in must resolve to a workdir and mesh_path must be available")
    preview_artifacts = create_diverted_tokamak_movie_package(
        workdir=workdir,
        mesh_path=resolved_mesh,
        output_root=root,
        field_name=field_name,
        case_label=case_label,
        fps=fps,
        frames_per_interval=frames_per_interval,
    )
    return _finalize_scaffold_artifacts(
        preview_artifacts,
        workdir=workdir,
        mesh_path=resolved_mesh,
        output_root=root,
        data_dir=data_dir,
        case_label=case_label,
        field_name=field_name,
        reference_status=resolved,
        preview_mode=False,
        workdir_mode="external_workdir",
    )


def _finalize_scaffold_artifacts(
    artifacts: DivertedTokamakMovieArtifacts,
    *,
    workdir: Path,
    mesh_path: Path,
    output_root: Path,
    data_dir: Path,
    case_label: str,
    field_name: str,
    reference_status: TcvX21ReferenceStatus,
    preview_mode: bool,
    workdir_mode: str,
) -> TcvX21ScaffoldArtifacts:
    input_report = _build_input_report(reference_status)
    input_report_json_path = data_dir / f"{case_label}_input_report.json"
    input_report_json_path.write_text(json.dumps(input_report, indent=2, sort_keys=True), encoding="utf-8")
    validation_contract = _build_validation_contract(reference_status)
    validation_contract_json_path = data_dir / f"{case_label}_validation_contract.json"
    validation_contract_json_path.write_text(
        json.dumps(validation_contract, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    profile_bundle = _build_profile_bundle(
        workdir=workdir,
        mesh_path=mesh_path,
        case_label=case_label,
        output_root=output_root,
        data_dir=data_dir,
        images_dir=output_root / "images",
    )
    observable_report = build_geometry_observable_report(
        geometry_family="diverted_tokamak_3d",
        benchmark_adapter="tcv_x21",
        observable_groups=(
            profile_group_from_report(
                profile_bundle["profile_report"],
                name="benchmark_profiles",
                description="Named TCV-X21 benchmark probe and divertor profile families.",
            ),
        ),
        metadata={"case_name": reference_status.case.name},
    )
    observable_report_json_path = write_geometry_observable_report(
        observable_report,
        data_dir / f"{case_label}_observable_report.json",
    )
    report = build_geometry_adapter_manifest(
        case_label=case_label,
        geometry_family="diverted_tokamak_3d",
        benchmark_adapter="tcv_x21",
        preview_mode=preview_mode,
        artifacts={
            "input_report_json": str(input_report_json_path.relative_to(output_root)),
            "validation_contract_json": str(validation_contract_json_path.relative_to(output_root)),
            "observable_report_json": str(observable_report_json_path.relative_to(output_root)),
            "profile_report_json": str(profile_bundle["profile_report_json_path"].relative_to(output_root)),
            "profile_arrays_npz": str(profile_bundle["profile_arrays_npz_path"].relative_to(output_root)),
            "profile_plot_png": str(profile_bundle["profile_plot_png_path"].relative_to(output_root)),
            "arrays_npz": str(artifacts.arrays_npz_path.relative_to(output_root)),
            "analysis_json": str(artifacts.analysis_json_path.relative_to(output_root)),
            "snapshots_png": str(artifacts.snapshots_png_path.relative_to(output_root)),
            "poster_png": str(artifacts.poster_png_path.relative_to(output_root)),
            "movie_gif": str(artifacts.movie_gif_path.relative_to(output_root)),
        },
        metadata={
            "case_name": reference_status.case.name,
            "field_name": field_name,
            "capability_tier": reference_status.case.capability_tier,
            "workdir_mode": workdir_mode,
            "reference_input_path": reference_status.case.reference_path,
            "reference_exists": reference_status.exists,
        },
    )
    manifest_json_path = data_dir / f"{case_label}_manifest.json"
    manifest_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return TcvX21ScaffoldArtifacts(
        manifest_json_path=manifest_json_path,
        input_report_json_path=input_report_json_path,
        validation_contract_json_path=validation_contract_json_path,
        observable_report_json_path=observable_report_json_path,
        profile_report_json_path=profile_bundle["profile_report_json_path"],
        profile_arrays_npz_path=profile_bundle["profile_arrays_npz_path"],
        profile_plot_png_path=profile_bundle["profile_plot_png_path"],
        arrays_npz_path=artifacts.arrays_npz_path,
        analysis_json_path=artifacts.analysis_json_path,
        snapshots_png_path=artifacts.snapshots_png_path,
        poster_png_path=artifacts.poster_png_path,
        movie_gif_path=artifacts.movie_gif_path,
    )


@dataclass(frozen=True)
class _SyntheticPreviewPaths:
    workdir: Path
    mesh_path: Path


def _write_synthetic_preview_workdir(root: Path, *, field_name: str) -> _SyntheticPreviewPaths:
    workdir = root / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    mesh_path = workdir / "tokamak.nc"
    _write_synthetic_tokamak_mesh(mesh_path)
    for pe_yind in (0, 1):
        _write_synthetic_dump(workdir / f"BOUT.dmp.{pe_yind}.nc", field_name=field_name, pe_yind=pe_yind)
    return _SyntheticPreviewPaths(workdir=workdir, mesh_path=mesh_path)


def _write_synthetic_tokamak_mesh(path: Path) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 6)
        dataset.createDimension("y", 6)
        rxy = dataset.createVariable("Rxy", "f8", ("x", "y"))
        zxy = dataset.createVariable("Zxy", "f8", ("x", "y"))
        psixy = dataset.createVariable("psixy", "f8", ("x", "y"))
        xx = np.linspace(1.15, 2.25, 6)[:, None]
        yy = np.linspace(-0.9, 0.9, 6)[None, :]
        rxy[:] = xx + 0.06 * np.sin(np.pi * yy)
        zxy[:] = yy + 0.12 * (xx - 1.65) ** 2 - 0.15
        psixy[:] = xx - 1.70 + 0.14 * (yy**2 - 0.2)


def _write_synthetic_dump(path: Path, *, field_name: str, pe_yind: int) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 6)
        dataset.createDimension("y", 5)
        dataset.createDimension("z", 4)
        dataset.createDimension("t", 3)
        for name, value in {
            "MYPE": pe_yind,
            "PE_XIND": 0,
            "PE_YIND": pe_yind,
            "NXPE": 1,
            "NYPE": 2,
            "MXSUB": 4,
            "MYSUB": 3,
            "MXG": 1,
            "MYG": 1,
        }.items():
            variable = dataset.createVariable(name, "i4")
            variable.assignValue(value)
        t_array = dataset.createVariable("t_array", "f8", ("t",))
        t_array[:] = np.array([0.0, 0.25, 0.5], dtype=np.float64)
        for name, value in {
            "Nnorm": 2.0e19,
            "Tnorm": 80.0,
            "Omega_ci": 1.0e6,
            "Cs0": 1.5e5,
        }.items():
            variable = dataset.createVariable(name, "f8")
            variable.assignValue(value)
        values = np.zeros((3, 6, 5, 4), dtype=np.float64)
        for time_index in range(3):
            for x_index in range(4):
                for y_index in range(3):
                    for z_index in range(4):
                        values[time_index, 1 + x_index, 1 + y_index, z_index] = (
                            0.2 * time_index
                            + 0.3 * pe_yind
                            + 0.05 * x_index
                            + 0.03 * y_index
                            + 0.02 * z_index
                        )
        field_specs = {
            "Ne": 1.0 + values,
            "Pe": 0.6 + 0.5 * values,
            "Pi": 0.7 + 0.45 * values,
            "phi": -0.05 + 0.2 * values,
            "NVi": 0.12 + 0.04 * values,
            "NVe": -0.08 + 0.03 * values,
        }
        if field_name not in field_specs:
            field_specs[field_name] = values
        for name, field_values in field_specs.items():
            field = dataset.createVariable(name, "f8", ("t", "x", "y", "z"))
            field[:] = field_values


def _build_profile_bundle(
    *,
    workdir: Path,
    mesh_path: Path,
    case_label: str,
    output_root: Path,
    data_dir: Path,
    images_dir: Path,
) -> dict[str, Path]:
    profile_report = _extract_tcv_x21_profile_report(workdir=workdir, mesh_path=mesh_path)
    profile_report_json_path = data_dir / f"{case_label}_profile_report.json"
    profile_report_json_path.write_text(json.dumps(profile_report, indent=2, sort_keys=True), encoding="utf-8")
    profile_arrays_npz_path = write_tcv_x21_profile_arrays_npz(
        profile_report,
        data_dir / f"{case_label}_profile_arrays.npz",
    )
    profile_plot_png_path = save_tcv_x21_profile_summary_plot(
        profile_report,
        images_dir / f"{case_label}_profiles.png",
    )
    return {
        "profile_report": profile_report,
        "profile_report_json_path": profile_report_json_path,
        "profile_arrays_npz_path": profile_arrays_npz_path,
        "profile_plot_png_path": profile_plot_png_path,
    }


def _extract_tcv_x21_profile_report(*, workdir: Path, mesh_path: Path) -> dict[str, object]:
    required_fields = ("Ne", "Pe", "Pi", "phi", "NVi", "NVe")
    first_dump = next(iter(sorted(workdir.glob("BOUT.dmp.*.nc"))), None)
    if first_dump is None:
        return {
            "available": False,
            "parse_status": "missing_workdir",
            "required_fields": list(required_fields),
        }
    missing_fields = _missing_dump_fields(first_dump, required_fields)
    if missing_fields:
        return {
            "available": False,
            "parse_status": "missing_fields",
            "required_fields": list(required_fields),
            "missing_fields": missing_fields,
        }

    histories = {
        name: assemble_tokamak_rank_history(workdir, field_name=name)
        for name in required_fields
    }
    geometry = load_diverted_tokamak_geometry(mesh_path, active_nx=histories["Ne"].history_4d.shape[1])
    norms = _load_profile_normalization(first_dump)
    return build_diagnostic_profile_report(
        diagnostic_positions=_build_tcv_x21_diagnostic_positions(geometry),
        derived_histories=_build_tcv_x21_derived_histories(histories=histories, norms=norms),
        time_points=np.asarray(histories["Ne"].time_points, dtype=np.float64),
        normalization=norms,
    )


def _missing_dump_fields(dump_path: Path, field_names: tuple[str, ...]) -> list[str]:
    with Dataset(dump_path) as dataset:
        return [name for name in field_names if name not in dataset.variables]


def _load_profile_normalization(dump_path: Path) -> dict[str, float | str]:
    with Dataset(dump_path) as dataset:
        nnorm = _read_optional_scalar(dataset, "Nnorm", default=1.0)
        tnorm = _read_optional_scalar(dataset, "Tnorm", default=1.0)
        cs0 = _read_optional_scalar(dataset, "Cs0", default=1.0)
    return {
        "status": "physical_units" if (nnorm, tnorm, cs0) != (1.0, 1.0, 1.0) else "code_units",
        "Nnorm": float(nnorm),
        "Tnorm_eV": float(tnorm),
        "Cs0_m_per_s": float(cs0),
    }


def _read_optional_scalar(dataset: Dataset, name: str, *, default: float) -> float:
    if name not in dataset.variables:
        return default
    return float(np.asarray(dataset.variables[name][:]).item())


def _build_tcv_x21_derived_histories(
    *,
    histories: dict[str, DivertedTokamakFieldHistory],
    norms: dict[str, float | str],
) -> dict[str, tuple[str, np.ndarray]]:
    qe = 1.602e-19
    aa_ion = 2.0
    aa_me = 1.0 / 1836.0
    nnorm = float(norms["Nnorm"])
    tnorm = float(norms["Tnorm_eV"])
    cs0 = float(norms["Cs0_m_per_s"])
    ne = np.asarray(histories["Ne"].history_4d, dtype=np.float64)
    pe = np.asarray(histories["Pe"].history_4d, dtype=np.float64)
    pi = np.asarray(histories["Pi"].history_4d, dtype=np.float64)
    phi = np.asarray(histories["phi"].history_4d, dtype=np.float64)
    nvi = np.asarray(histories["NVi"].history_4d, dtype=np.float64)
    nve = np.asarray(histories["NVe"].history_4d, dtype=np.float64)
    ne_floor = np.clip(ne, 1.0e-8, None)

    return {
        "density": ("1/m^3", ne * nnorm),
        "electron_temp": ("eV", pe / ne_floor * tnorm),
        "ion_temp": ("eV", pi / ne_floor * tnorm),
        "potential": ("V", phi * tnorm),
        "vfloat": ("V", (phi - 3.0 * pe / ne_floor) * tnorm),
        "current": ("A/m^2", (nvi / aa_ion - nve / aa_me) * qe * nnorm * cs0),
    }


def _build_tcv_x21_diagnostic_positions(
    geometry: DivertedTokamakGeometry,
) -> dict[str, tuple[int, np.ndarray]]:
    ymid = geometry.rxy.shape[1] // 2
    psi_mid = np.asarray(geometry.psixy[:, ymid], dtype=np.float64)
    r_mid = np.asarray(geometry.rxy[:, ymid], dtype=np.float64)
    order = np.argsort(psi_mid)
    psi_sorted = psi_mid[order]
    r_sorted = r_mid[order]
    r_sep = float(np.interp(0.0, psi_sorted, r_sorted))

    def map_psi_to_cm(psi_values: np.ndarray) -> np.ndarray:
        mapped_r = np.interp(np.asarray(psi_values, dtype=np.float64), psi_sorted, r_sorted)
        return (mapped_r - r_sep) * 100.0

    return {
        "FHRP": (ymid, (r_mid - r_sep) * 100.0),
        "HFS-LP": (0, map_psi_to_cm(geometry.psixy[:, 0])),
        "LFS-LP": (-1, map_psi_to_cm(geometry.psixy[:, -1])),
    }


def write_tcv_x21_profile_arrays_npz(profile_report: dict[str, object], path: str | Path) -> Path:
    return write_diagnostic_profile_arrays_npz(profile_report, path)


def save_tcv_x21_profile_summary_plot(profile_report: dict[str, object], path: str | Path) -> Path:
    return save_diagnostic_profile_summary_plot(
        profile_report,
        path,
        diagnostic_order=("FHRP", "LFS-LP", "HFS-LP"),
        observable_order=(
            ("density", "Density"),
            ("electron_temp", "Electron Temp"),
            ("ion_temp", "Ion Temp"),
            ("potential", "Potential"),
            ("current", "Current"),
        ),
        title="TCV-X21 Scaffold Profile Families",
    )


def _build_input_report(reference_status: TcvX21ReferenceStatus) -> dict[str, object]:
    case = reference_status.case
    report: dict[str, object] = {
        "available": reference_status.exists,
        "case_name": case.name,
        "reference_input_path": case.reference_path,
        "capability_tier": case.capability_tier,
        "parity_mode": case.parity_mode,
        "stage": case.stage,
        "rationale": case.rationale,
        "compare_variables": list(case.compare_variables),
        "extra_overrides": list(case.extra_overrides),
        "process_count": case.process_count,
        "artifact_bundle_files": list(case.artifact_bundle_files),
    }
    if not reference_status.exists:
        report["parse_status"] = "missing_input"
        return report

    try:
        config = load_bout_input(reference_status.input_path)
    except Exception as exc:
        report["parse_status"] = "parse_failed"
        report["parse_error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["parse_status"] = "ok"
    report["section_names"] = list(config.section_names())
    report["model_section"] = _resolve_model_section(config)
    report["declared_components"] = _resolve_declared_components(config)

    try:
        run_config = RunConfiguration.from_config(config)
    except Exception as exc:
        report["run_config_status"] = "partial"
        report["run_config_error"] = f"{type(exc).__name__}: {exc}"
        run_config = None
    else:
        report["run_config_status"] = "ok"

    report["time"] = _resolve_time_summary(config, run_config)
    report["mesh"] = _resolve_mesh_summary(config, run_config)
    report["solver"] = _resolve_solver_summary(config, run_config)
    report["components"] = _resolve_component_summary(run_config)
    return report


def _resolve_model_section(config: BoutConfig) -> str | None:
    if not has_model_section(config):
        return None
    try:
        return locate_model_section(config)
    except KeyError:
        return None


def _resolve_declared_components(config: BoutConfig) -> list[str]:
    model_section = _resolve_model_section(config)
    if model_section is None or not config.has_option(model_section, "components"):
        return []
    value = config.parsed(model_section, "components")
    if isinstance(value, tuple):
        return [_normalize_component_name(item) for item in value]
    return [_normalize_component_name(value)]


def _resolve_component_summary(run_config: RunConfiguration | None) -> dict[str, object]:
    if run_config is None:
        return {"count": 0, "labels": [], "sections": []}
    sections = sorted({_normalize_component_name(component.section) for component in run_config.components})
    return {
        "count": len(run_config.components),
        "labels": [_normalize_component_name(component.label) for component in run_config.components],
        "sections": sections,
    }


def _resolve_time_summary(
    config: BoutConfig,
    run_config: RunConfiguration | None,
) -> dict[str, object]:
    if run_config is not None:
        return {
            "nout": run_config.time.nout,
            "timestep": run_config.time.timestep,
        }
    return {
        "nout": _parsed_option(config, "__root__", "nout"),
        "timestep": _parsed_option(config, "__root__", "timestep"),
    }


def _resolve_mesh_summary(
    config: BoutConfig,
    run_config: RunConfiguration | None,
) -> dict[str, object]:
    if run_config is not None:
        mesh = run_config.mesh
        return {
            "nx": mesh.nx,
            "ny": mesh.ny,
            "nz": mesh.nz,
            "mxg": mesh.mxg,
            "myg": mesh.myg,
            "mz": mesh.mz,
            "zperiod": mesh.zperiod,
            "file": mesh.file,
            "extrapolate_y": mesh.extrapolate_y,
            "parallel_transform_type": mesh.parallel_transform.type,
        }
    return {
        "nx": _parsed_option(config, "mesh", "nx"),
        "ny": _parsed_option(config, "mesh", "ny"),
        "nz": _parsed_option(config, "mesh", "nz"),
        "mxg": _parsed_option(config, "__root__", "MXG"),
        "myg": _parsed_option(config, "__root__", "MYG"),
        "mz": _parsed_option(config, "__root__", "MZ"),
        "zperiod": _parsed_option(config, "__root__", "zperiod"),
        "file": _parsed_option(config, "mesh", "file"),
        "extrapolate_y": _parsed_option(config, "mesh", "extrapolate_y"),
        "parallel_transform_type": _parsed_option(config, "mesh:paralleltransform", "type"),
    }


def _resolve_solver_summary(
    config: BoutConfig,
    run_config: RunConfiguration | None,
) -> dict[str, object]:
    if run_config is not None:
        solver = run_config.solver
        return {
            "type": solver.type,
            "mxstep": solver.mxstep,
            "rtol": solver.rtol,
            "atol": solver.atol,
            "use_precon": solver.use_precon,
            "cvode_max_order": solver.cvode_max_order,
            "mms": solver.mms,
        }
    return {
        "type": _parsed_option(config, "solver", "type"),
        "mxstep": _parsed_option(config, "solver", "mxstep"),
        "rtol": _parsed_option(config, "solver", "rtol"),
        "atol": _parsed_option(config, "solver", "atol"),
        "use_precon": _parsed_option(config, "solver", "use_precon"),
        "cvode_max_order": _parsed_option(config, "solver", "cvode_max_order"),
        "mms": _parsed_option(config, "solver", "mms"),
    }


def _parsed_option(config: BoutConfig, section: str, key: str) -> bool | int | float | str | tuple[str, ...] | None:
    if not config.has_option(section, key):
        return None
    return config.parsed(section, key)


def _normalize_component_name(value: object) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _build_validation_contract(reference_status: TcvX21ReferenceStatus) -> dict[str, object]:
    case = reference_status.case
    return build_geometry_adapter_contract(
        geometry_family="diverted_tokamak_3d",
        benchmark_adapter="tcv_x21",
        diagnostic_layer="benchmark_adapter_on_general_3d_geometry",
        references=[
            {
                "label": "TCV-X21 validation benchmark",
                "url": "https://arxiv.org/abs/2109.01618",
            },
            {
                "label": "TCV-X21 turbulence validation follow-up",
                "url": "https://arxiv.org/abs/2506.12180",
            },
            {
                "label": "GBS code verification and MMS reference",
                "url": "https://arxiv.org/abs/2112.03573",
            },
            {
                "label": "Detachment-scaling 1D SOL benchmark",
                "url": "https://arxiv.org/abs/2406.16375",
            },
        ],
        promotion_gates=[
            "scaffold_gate",
            "external_workdir_gate",
            "selected_field_parity_gate",
            "benchmark_validation_gate",
        ],
        metadata={
            "case_name": case.name,
            "capability_tier": case.capability_tier,
            "benchmark": {
                "name": "TCV-X21 diverted L-mode reference case",
                "dataset_summary": "45 one- and two-dimensional observables across two toroidal-field directions.",
                "references": [
                    {
                        "label": "TCV-X21 validation benchmark",
                        "url": "https://arxiv.org/abs/2109.01618",
                    },
                    {
                        "label": "TCV-X21 turbulence validation follow-up",
                        "url": "https://arxiv.org/abs/2506.12180",
                    },
                    {
                        "label": "GBS code verification and MMS reference",
                        "url": "https://arxiv.org/abs/2112.03573",
                    },
                    {
                        "label": "Detachment-scaling 1D SOL benchmark",
                        "url": "https://arxiv.org/abs/2406.16375",
                    },
                ],
            },
            "reference_inputs": {
                "input_exists": reference_status.exists,
                "reference_input_path": case.reference_path,
                "reference_helper_scripts": [
                    "examples/tokamak-3D/tcv-x21/gather_data.py",
                    "examples/tokamak-3D/tcv-x21/convert_to_tcvx21.py",
                    "examples/tokamak-3D/tcv-x21/make_tcvx21_plots.py",
                ],
            },
            "diagnostic_sets": [
            {
                "name": "FHRP",
                "description": "Outboard midplane reciprocating probe profiles.",
                "observables": ["density", "electron_temp", "ion_temp", "potential", "vfloat"],
            },
            {
                "name": "LFS-LP",
                "description": "Low-field-side target Langmuir probe profiles.",
                "observables": ["density", "electron_temp", "ion_temp", "potential", "current", "vfloat"],
            },
            {
                "name": "HFS-LP",
                "description": "High-field-side target Langmuir probe profiles.",
                "observables": ["density", "electron_temp", "ion_temp", "potential", "current", "vfloat"],
            },
            ],
            "profile_metrics": [
                {
                    "name": "absolute_profile_agreement",
                    "description": "Match absolute magnitude and profile shape at OMP and divertor targets.",
                },
                {
                    "name": "omp_length_scales",
                    "description": "Fit density and temperature decay lengths on the outboard midplane profile window.",
                },
                {
                    "name": "target_peak_location",
                    "description": "Track LFS/HFS target peak position shifts under field reversal when the benchmark bundle is extended.",
                },
            ],
            "verification_requirements": [
                {
                    "name": "manufactured_solution_convergence",
                    "status": "required_before_native_3d_promotion",
                    "description": "Require MMS/order-of-accuracy evidence on the promoted operators before claiming native 3D closure.",
                },
                {
                    "name": "field_parity_surface",
                    "status": "required_before_selected_field_rung",
                    "description": "Require a reduced selected-field compare surface before broad benchmark claims.",
                },
                {
                    "name": "restart_and_provenance",
                    "status": "required_before_public_3d_release",
                    "description": "Require restart equivalence and artifact provenance alongside benchmark plots.",
                },
            ],
            "promotion_gates": [
                {
                    "name": "scaffold_gate",
                    "status": "landed",
                    "required_artifacts": [
                        "manifest_json",
                        "input_report_json",
                        "validation_contract_json",
                        "snapshots_png",
                        "poster_png",
                        "movie_gif",
                    ],
                },
                {
                    "name": "external_workdir_gate",
                    "status": "next",
                    "required_artifacts": [
                        "real_workdir_movie_bundle",
                        "real_input_report_json",
                        "profile_summary_json",
                    ],
                },
                {
                    "name": "selected_field_parity_gate",
                    "status": "planned",
                    "required_artifacts": [
                        "compact_reference_baseline_json",
                        "compact_reference_arrays_npz",
                        "bounded_parity_test",
                    ],
                },
                {
                    "name": "benchmark_validation_gate",
                    "status": "planned",
                    "required_artifacts": [
                        "tcv_x21_observable_package",
                        "publication_ready_profile_plots",
                        "methods_note",
                    ],
                },
            ],
        },
    )
