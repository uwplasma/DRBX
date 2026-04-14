from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from ..reference.cases import ReferenceCase, load_reference_cases
from .diverted_tokamak_movie import (
    DivertedTokamakMovieArtifacts,
    create_diverted_tokamak_movie_package,
)

DEFAULT_TCV_X21_CASE_NAME = "tokamak_tcv_x21_escalation"


@dataclass(frozen=True)
class TcvX21ScaffoldArtifacts:
    manifest_json_path: Path
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
    reference_input_path = str(resolved.case.reference_path)
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
                output_root=root,
                data_dir=data_dir,
                case_name=case_name,
                case_label=case_label,
                field_name=field_name,
                reference_input_path=reference_input_path,
                reference_exists=resolved.exists,
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
        output_root=root,
        data_dir=data_dir,
        case_name=case_name,
        case_label=case_label,
        field_name=field_name,
        reference_input_path=reference_input_path,
        reference_exists=resolved.exists,
        preview_mode=False,
        workdir_mode="external_workdir",
    )


def _finalize_scaffold_artifacts(
    artifacts: DivertedTokamakMovieArtifacts,
    *,
    output_root: Path,
    data_dir: Path,
    case_name: str,
    case_label: str,
    field_name: str,
    reference_input_path: str,
    reference_exists: bool,
    preview_mode: bool,
    workdir_mode: str,
) -> TcvX21ScaffoldArtifacts:
    report = {
        "case_name": case_name,
        "case_label": case_label,
        "field_name": field_name,
        "capability_tier": "scaffolded_reference_backed",
        "preview_mode": preview_mode,
        "workdir_mode": workdir_mode,
        "reference_input_path": reference_input_path,
        "reference_exists": reference_exists,
        "artifacts": {
            "arrays_npz": str(artifacts.arrays_npz_path.relative_to(output_root)),
            "analysis_json": str(artifacts.analysis_json_path.relative_to(output_root)),
            "snapshots_png": str(artifacts.snapshots_png_path.relative_to(output_root)),
            "poster_png": str(artifacts.poster_png_path.relative_to(output_root)),
            "movie_gif": str(artifacts.movie_gif_path.relative_to(output_root)),
        },
    }
    manifest_json_path = data_dir / f"{case_label}_manifest.json"
    manifest_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return TcvX21ScaffoldArtifacts(
        manifest_json_path=manifest_json_path,
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
        field = dataset.createVariable(field_name, "f8", ("t", "x", "y", "z"))
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
        field[:] = values
