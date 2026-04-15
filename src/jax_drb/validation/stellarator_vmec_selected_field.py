from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from netCDF4 import Dataset
import numpy as np

from .geometry_observables import build_geometry_observable_report, write_geometry_observable_report
from .geometry_selected_field import (
    GeometrySelectedFieldParityResult,
    compare_geometry_selected_fields,
    save_geometry_selected_field_parity_plot,
    write_geometry_selected_field_parity_arrays,
    write_geometry_selected_field_parity_json,
)
from .stellarator_vmec_scaffold import _load_vmec_wout, _write_synthetic_vmec_wout


@dataclass(frozen=True)
class StellaratorVmecSelectedFieldParityArtifacts:
    parity_json_path: Path
    parity_arrays_npz_path: Path
    parity_plot_png_path: Path
    observable_report_json_path: Path
    source_report_json_path: Path


def compare_stellarator_vmec_selected_fields(
    *,
    reference_equilibrium_path: str | Path,
    candidate_equilibrium_path: str | Path,
    field_names: tuple[str, ...] = ("iota", "pressure", "toroidal_flux"),
) -> GeometrySelectedFieldParityResult:
    reference = _load_vmec_selected_fields(reference_equilibrium_path)
    candidate = _load_vmec_selected_fields(candidate_equilibrium_path)
    return compare_geometry_selected_fields(
        reference_fields=reference,
        candidate_fields=candidate,
        field_names=field_names,
    )


def create_stellarator_vmec_selected_field_parity_package(
    *,
    reference_equilibrium_path: str | Path | None,
    candidate_equilibrium_path: str | Path | None,
    output_root: str | Path,
    case_label: str = "stellarator_vmec_selected_field_parity",
    field_names: tuple[str, ...] = ("iota", "pressure", "toroidal_flux"),
) -> StellaratorVmecSelectedFieldParityArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    source_mode = "explicit_pair"
    candidate_origin = "provided"
    reference_name = "<synthetic preview>"
    candidate_name = "<synthetic preview>"
    if reference_equilibrium_path is None or candidate_equilibrium_path is None:
        with tempfile.TemporaryDirectory(prefix="jax_drb_stellarator_vmec_selected_") as temp_dir:
            temp_root = Path(temp_dir)
            if reference_equilibrium_path is None:
                source_mode = "synthetic_preview"
                reference_path = temp_root / "reference_wout.nc"
                candidate_path = temp_root / "candidate_wout.nc"
                _write_synthetic_vmec_wout(reference_path)
                _write_candidate_from_reference_vmec(reference_path, candidate_path)
                candidate_origin = "synthetic_preview_pair"
                reference_name = reference_path.name
                candidate_name = candidate_path.name
            else:
                source_mode = "external_explicit_pair"
                reference_path = Path(reference_equilibrium_path)
                candidate_path = temp_root / f"candidate{reference_path.suffix}"
                _write_candidate_from_reference_vmec(reference_path, candidate_path)
                candidate_origin = "materialized_from_reference_input"
                reference_name = reference_path.name
                candidate_name = candidate_path.name
            result = compare_stellarator_vmec_selected_fields(
                reference_equilibrium_path=reference_path,
                candidate_equilibrium_path=candidate_path,
                field_names=field_names,
            )
    else:
        reference_name = Path(reference_equilibrium_path).name
        candidate_name = Path(candidate_equilibrium_path).name
        result = compare_stellarator_vmec_selected_fields(
            reference_equilibrium_path=reference_equilibrium_path,
            candidate_equilibrium_path=candidate_equilibrium_path,
            field_names=field_names,
        )

    parity_json_path = write_geometry_selected_field_parity_json(result, data_dir / f"{case_label}.json")
    parity_arrays_npz_path = write_geometry_selected_field_parity_arrays(result, data_dir / f"{case_label}.npz")
    parity_plot_png_path = save_geometry_selected_field_parity_plot(
        result,
        images_dir / f"{case_label}.png",
        title="Stellarator VMEC selected-field parity",
    )
    observable_report = build_geometry_observable_report(
        geometry_family="stellarator_vmec_3d",
        benchmark_adapter="stellarator_vmec_selected_field",
        observable_groups=(
            {
                "name": "selected_equilibrium_parity",
                "description": "Compact selected-field parity surface on stellarator VMEC equilibrium profiles.",
                "families": [
                    {
                        "name": "selected_equilibrium_fields",
                        "kind": "selected_field_parity",
                        "coordinate_name": "full_domain",
                        "field_names": list(result.field_names),
                    }
                ],
            },
        ),
        metadata={
            "compare_surface": "vmec_profile_bundle",
            "source_mode": source_mode,
            "candidate_origin": candidate_origin,
        },
    )
    observable_report_json_path = write_geometry_observable_report(
        observable_report,
        data_dir / f"{case_label}_observable_report.json",
    )
    source_report = {
        "available": True,
        "parse_status": "ok",
        "source_mode": source_mode,
        "candidate_origin": candidate_origin,
        "reference_input_name": reference_name,
        "candidate_input_name": candidate_name,
    }
    source_report_json_path = data_dir / f"{case_label}_source_report.json"
    source_report_json_path.write_text(json.dumps(source_report, indent=2, sort_keys=True), encoding="utf-8")
    return StellaratorVmecSelectedFieldParityArtifacts(
        parity_json_path=parity_json_path,
        parity_arrays_npz_path=parity_arrays_npz_path,
        parity_plot_png_path=parity_plot_png_path,
        observable_report_json_path=observable_report_json_path,
        source_report_json_path=source_report_json_path,
    )


def _load_vmec_selected_fields(path: str | Path) -> dict[str, np.ndarray]:
    payload = _load_vmec_wout(Path(path))
    return {
        "iota": np.asarray(payload["iota"], dtype=np.float64),
        "pressure": np.asarray(payload["pressure"], dtype=np.float64),
        "toroidal_flux": np.asarray(payload["toroidal_flux"], dtype=np.float64),
    }


def _write_candidate_from_reference_vmec(reference_path: Path, candidate_path: Path) -> None:
    shutil.copy2(reference_path, candidate_path)
    with Dataset(candidate_path, "r+") as dataset:
        for field_name, delta in (("iotaf", 0.01), ("presf", -0.02), ("phi", 0.015)):
            if field_name not in dataset.variables:
                continue
            values = np.asarray(dataset.variables[field_name][:], dtype=np.float64)
            dataset.variables[field_name][:] = values * (1.0 + delta)
