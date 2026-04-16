from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
from netCDF4 import Dataset
import numpy as np

from ..config.boutinp import load_bout_input
from ..parity.reference import discover_reference_binary
from ..reference.paths import require_reference_root


@dataclass(frozen=True)
class TemperatureFeedbackCampaignMetric:
    name: str
    kind: str
    value: float
    target: float
    passed: bool
    notes: str


@dataclass(frozen=True)
class TemperatureFeedbackCampaignArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


@dataclass(frozen=True)
class _TemperatureFeedbackSeries:
    time_points: np.ndarray
    target_temperature: np.ndarray
    setpoint: float
    reference_multiplier: np.ndarray
    reconstructed_multiplier: np.ndarray
    reference_proportional: np.ndarray
    reconstructed_proportional: np.ndarray
    reference_integral: np.ndarray
    reconstructed_integral: np.ndarray
    reference_integral_state: np.ndarray
    reconstructed_integral_state: np.ndarray
    reference_energy_source: np.ndarray
    reconstructed_energy_source: np.ndarray


def create_temperature_feedback_campaign_package(
    *,
    output_root: str | Path,
    reference_root: str | Path | None = None,
    reference_binary: str | Path | None = None,
    case_label: str = "temperature_feedback_campaign",
    nout: int = 4,
    timestep: float = 100.0,
    ny: int = 80,
    timeout_seconds: int = 180,
) -> TemperatureFeedbackCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_temperature_feedback_campaign(
        reference_root=reference_root,
        reference_binary=reference_binary,
        nout=nout,
        timestep=timestep,
        ny=ny,
        timeout_seconds=timeout_seconds,
    )
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report["summary"], indent=2, sort_keys=True), encoding="utf-8")

    series = report["series"]
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(
        arrays_npz_path,
        time_points=series.time_points,
        target_temperature=series.target_temperature,
        setpoint=np.asarray([series.setpoint], dtype=np.float64),
        reference_multiplier=series.reference_multiplier,
        reconstructed_multiplier=series.reconstructed_multiplier,
        reference_proportional=series.reference_proportional,
        reconstructed_proportional=series.reconstructed_proportional,
        reference_integral=series.reference_integral,
        reconstructed_integral=series.reconstructed_integral,
        reference_integral_state=series.reference_integral_state,
        reconstructed_integral_state=series.reconstructed_integral_state,
        reference_energy_source=series.reference_energy_source,
        reconstructed_energy_source=series.reconstructed_energy_source,
    )

    plot_png_path = images_dir / f"{case_label}.png"
    _save_temperature_feedback_plot(series, plot_png_path)
    return TemperatureFeedbackCampaignArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_temperature_feedback_campaign(
    *,
    reference_root: str | Path | None = None,
    reference_binary: str | Path | None = None,
    nout: int = 4,
    timestep: float = 100.0,
    ny: int = 80,
    timeout_seconds: int = 180,
) -> dict[str, object]:
    resolved_reference_root = Path(reference_root) if reference_root is not None else require_reference_root()
    series = _build_temperature_feedback_series(
        reference_root=resolved_reference_root,
        reference_binary=Path(reference_binary) if reference_binary is not None else None,
        nout=nout,
        timestep=timestep,
        ny=ny,
        timeout_seconds=timeout_seconds,
    )
    metrics = (
        TemperatureFeedbackCampaignMetric(
            name="temperature_feedback_src_mult_e_exact",
            kind="max_abs_error",
            value=float(np.max(np.abs(series.reference_multiplier - series.reconstructed_multiplier))),
            target=1.0e-12,
            passed=bool(np.max(np.abs(series.reference_multiplier - series.reconstructed_multiplier)) <= 1.0e-12),
            notes="Reconstructed PI controller multiplier matches the Hermes diagnostic exactly on the bounded Tt-control example.",
        ),
        TemperatureFeedbackCampaignMetric(
            name="temperature_feedback_src_p_e_exact",
            kind="max_abs_error",
            value=float(np.max(np.abs(series.reference_proportional - series.reconstructed_proportional))),
            target=1.0e-12,
            passed=bool(np.max(np.abs(series.reference_proportional - series.reconstructed_proportional)) <= 1.0e-12),
            notes="Reconstructed proportional term matches the Hermes diagnostic exactly.",
        ),
        TemperatureFeedbackCampaignMetric(
            name="temperature_feedback_src_i_e_exact",
            kind="max_abs_error",
            value=float(np.max(np.abs(series.reference_integral - series.reconstructed_integral))),
            target=1.0e-12,
            passed=bool(np.max(np.abs(series.reference_integral - series.reconstructed_integral)) <= 1.0e-12),
            notes="Reconstructed integral term matches the Hermes diagnostic exactly.",
        ),
        TemperatureFeedbackCampaignMetric(
            name="e_temperature_error_integral_exact",
            kind="max_abs_error",
            value=float(np.max(np.abs(series.reference_integral_state - series.reconstructed_integral_state))),
            target=1.0e-12,
            passed=bool(np.max(np.abs(series.reference_integral_state - series.reconstructed_integral_state)) <= 1.0e-12),
            notes="Reconstructed PI error integral matches the restart/state diagnostic exactly.",
        ),
        TemperatureFeedbackCampaignMetric(
            name="SPe_feedback_exact",
            kind="max_abs_error",
            value=float(np.max(np.abs(series.reference_energy_source - series.reconstructed_energy_source))),
            target=1.0e-12,
            passed=bool(np.max(np.abs(series.reference_energy_source - series.reconstructed_energy_source)) <= 1.0e-12),
            notes="Reconstructed electron energy-source history matches the Hermes diagnostic exactly.",
        ),
        TemperatureFeedbackCampaignMetric(
            name="target_temperature_error_ratio",
            kind="ratio",
            value=float(
                abs(series.setpoint - series.target_temperature[-1]) / max(abs(series.setpoint - series.target_temperature[0]), 1.0e-12)
            ),
            target=1.0,
            passed=bool(
                abs(series.setpoint - series.target_temperature[-1])
                <= abs(series.setpoint - series.target_temperature[0])
            ),
            notes="The bounded target-temperature run reduces the target-temperature error over the short validation window.",
        ),
    )
    summary = {
        "family": "temperature_feedback",
        "reference_mode": "external_example_cvode_patch",
        "example": "tokamak-1D/extra/1D-recycling-with-Tt-control",
        "nout": int(nout),
        "timestep": float(timestep),
        "ny": int(ny),
        "metric_count": len(metrics),
        "passed_metric_count": sum(1 for metric in metrics if metric.passed),
        "metrics": [
            {
                "name": metric.name,
                "kind": metric.kind,
                "value": float(metric.value),
                "target": float(metric.target),
                "passed": bool(metric.passed),
                "notes": metric.notes,
            }
            for metric in metrics
        ],
    }
    return {"summary": summary, "series": series}


def _build_temperature_feedback_series(
    *,
    reference_root: Path,
    reference_binary: Path | None,
    nout: int,
    timestep: float,
    ny: int,
    timeout_seconds: int,
) -> _TemperatureFeedbackSeries:
    binary = discover_reference_binary(reference_binary=reference_binary, reference_root=reference_root)
    example_dir = reference_root / "examples" / "tokamak-1D" / "extra" / "1D-recycling-with-Tt-control"
    if not example_dir.exists():
        raise FileNotFoundError(f"Temperature-feedback example not found: {example_dir}")

    with tempfile.TemporaryDirectory(prefix="jax_drb_temperature_feedback_") as temp_dir:
        workdir = Path(temp_dir)
        _stage_temperature_feedback_example(example_dir, workdir=workdir, nout=nout, timestep=timestep, ny=ny)
        _run_temperature_feedback_example(binary=binary, workdir=workdir, timeout_seconds=timeout_seconds)

        config = load_bout_input(workdir / "BOUT.inp")
        tnorm = float(config.parsed("hermes", "Tnorm"))
        setpoint = float(config.parsed("e", "temperature_setpoint")) / tnorm
        p_gain = float(config.parsed("e", "temperature_controller_p"))
        i_gain = float(config.parsed("e", "temperature_controller_i"))
        control_target = bool(config.parsed("e", "control_target_temperature"))
        integral_positive = bool(config.parsed("e", "temperature_integral_positive")) if config.has_option("e", "temperature_integral_positive") else False
        source_positive = bool(config.parsed("e", "temperature_source_positive")) if config.has_option("e", "temperature_source_positive") else True

        with Dataset(workdir / "BOUT.dmp.0.nc") as dataset:
            time_points = _extract_time_points(dataset)
            target_temperature = _extract_target_temperature(dataset, control_target=control_target)
            error = setpoint - target_temperature
            reconstructed_integral_state, reconstructed_proportional, reconstructed_integral, reconstructed_multiplier = (
                _reconstruct_temperature_controller(
                    time_points=time_points,
                    error=error,
                    proportional_gain=p_gain,
                    integral_gain=i_gain,
                    integral_positive=integral_positive,
                    source_positive=source_positive,
                )
            )
            reference_multiplier = _extract_scalar_series(dataset, "temperature_feedback_src_mult_e")
            reference_proportional = _extract_scalar_series(dataset, "temperature_feedback_src_p_e")
            reference_integral = _extract_scalar_series(dataset, "temperature_feedback_src_i_e")
            reference_integral_state = _extract_scalar_series(dataset, "e_temperature_error_integral")
            source_shape = _extract_spatial_series(dataset, "temperature_feedback_src_shape_e", time_count=time_points.size)
            reference_energy_source = _extract_spatial_series(dataset, "SPe_feedback", time_count=time_points.size)
            reconstructed_energy_source = source_shape * reconstructed_multiplier[:, None, None, None]

    return _TemperatureFeedbackSeries(
        time_points=time_points,
        target_temperature=target_temperature,
        setpoint=float(setpoint),
        reference_multiplier=reference_multiplier,
        reconstructed_multiplier=reconstructed_multiplier,
        reference_proportional=reference_proportional,
        reconstructed_proportional=reconstructed_proportional,
        reference_integral=reference_integral,
        reconstructed_integral=reconstructed_integral,
        reference_integral_state=reference_integral_state,
        reconstructed_integral_state=reconstructed_integral_state,
        reference_energy_source=reference_energy_source,
        reconstructed_energy_source=reconstructed_energy_source,
    )


def _stage_temperature_feedback_example(
    example_dir: Path,
    *,
    workdir: Path,
    nout: int,
    timestep: float,
    ny: int,
) -> None:
    for child in example_dir.iterdir():
        if child.is_file():
            shutil.copy2(child, workdir / child.name)
    input_path = workdir / "BOUT.inp"
    text = input_path.read_text(encoding="utf-8")
    text = _replace_bout_setting(text, "nout", str(int(nout)))
    text = _replace_bout_setting(text, "timestep", f"{float(timestep):g}")
    text = _replace_bout_setting(text, "ny", str(int(ny)))
    text = _replace_bout_setting(text, "type", "cvode")
    input_path.write_text(text, encoding="utf-8")


def _replace_bout_setting(text: str, key: str, value: str) -> str:
    pattern = rf"(?m)^({re.escape(key)}\s*=\s*).*$"
    replaced, count = re.subn(pattern, rf"\1{value}", text, count=1)
    if count != 1:
        raise ValueError(f"Could not replace {key!r} in patched BOUT.inp")
    return replaced


def _run_temperature_feedback_example(
    *,
    binary: Path,
    workdir: Path,
    timeout_seconds: int,
) -> None:
    stdout_path = workdir / "run.stdout"
    try:
        result = subprocess.run(
            [str(binary), "-d", "."],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Temperature-feedback reference run did not finish within {timeout_seconds}s in {workdir}"
        ) from exc
    stdout_path.write_text(result.stdout + ("\n" if result.stdout and not result.stdout.endswith("\n") else "") + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Temperature-feedback reference run failed with exit code {result.returncode}; see {stdout_path}"
        )
    if not (workdir / "BOUT.dmp.0.nc").exists():
        raise FileNotFoundError(f"Temperature-feedback reference run did not produce BOUT.dmp.0.nc in {workdir}")


def _extract_time_points(dataset: Dataset) -> np.ndarray:
    if "t_array" in dataset.variables:
        return np.asarray(dataset.variables["t_array"][:], dtype=np.float64).reshape(-1)
    if "t" in dataset.variables:
        return np.asarray(dataset.variables["t"][:], dtype=np.float64).reshape(-1)
    raise KeyError("missing time coordinate in temperature-feedback dataset")


def _extract_target_temperature(dataset: Dataset, *, control_target: bool) -> np.ndarray:
    time_points = _extract_time_points(dataset)
    field = _extract_spatial_series(dataset, "Te", time_count=time_points.size)
    y_index = -1 if control_target else 0
    return np.asarray(field[:, 0, y_index, 0], dtype=np.float64)


def _extract_scalar_series(dataset: Dataset, name: str) -> np.ndarray:
    values = np.asarray(dataset.variables[name][:], dtype=np.float64)
    if values.ndim == 1:
        return values
    return values.reshape(values.shape[0], -1)[:, 0]


def _extract_spatial_series(dataset: Dataset, name: str, *, time_count: int) -> np.ndarray:
    variable = dataset.variables[name]
    values = np.asarray(variable[:], dtype=np.float64)
    dimensions = tuple(variable.dimensions)
    has_time = "t" in dimensions
    if has_time:
        if values.ndim == 4:
            return values
        if values.ndim == 3:
            return values[:, None, :, :]
        if values.ndim == 2:
            return values[:, None, :, None]
        if values.ndim == 1:
            return values[:, None, None, None]
    else:
        if values.ndim == 3:
            return np.broadcast_to(values[None, ...], (time_count, *values.shape))
        if values.ndim == 2:
            return np.broadcast_to(values[None, None, :, :], (time_count, 1, *values.shape))
        if values.ndim == 1:
            return np.broadcast_to(values[None, None, :, None], (time_count, 1, values.shape[0], 1))
        if values.ndim == 0:
            return np.full((time_count, 1, 1, 1), float(values), dtype=np.float64)
    raise ValueError(f"Unsupported variable shape for {name}: {values.shape!r} dims={dimensions!r}")


def _reconstruct_temperature_controller(
    *,
    time_points: np.ndarray,
    error: np.ndarray,
    proportional_gain: float,
    integral_gain: float,
    integral_positive: bool,
    source_positive: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_points = np.asarray(time_points, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    if time_points.shape != error.shape:
        raise ValueError("time_points and error must have matching shape")

    integral_state = np.zeros_like(error)
    proportional_term = np.zeros_like(error)
    integral_term = np.zeros_like(error)
    multiplier = np.zeros_like(error)
    last_time = -1.0
    last_error = 0.0
    running_integral = 0.0
    for index, (time_value, error_value) in enumerate(zip(time_points, error, strict=True)):
        if last_time < 0.0:
            last_time = float(time_value)
            last_error = float(error_value)
        if time_value > last_time:
            running_integral += (float(time_value) - last_time) * 0.5 * (float(error_value) + last_error)
        if running_integral < 0.0 and integral_positive:
            running_integral = 0.0
        proportional = proportional_gain * float(error_value)
        integral = integral_gain * running_integral
        total = proportional + integral
        if total < 0.0 and source_positive:
            total = 0.0
        integral_state[index] = running_integral
        proportional_term[index] = proportional
        integral_term[index] = integral
        multiplier[index] = total
        last_time = float(time_value)
        last_error = float(error_value)
    return integral_state, proportional_term, integral_term, multiplier


def _save_temperature_feedback_plot(series: _TemperatureFeedbackSeries, path: Path) -> None:
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.5, 8.0), constrained_layout=True)

    axes[0, 0].plot(series.time_points, series.target_temperature, color="#0a9396", linewidth=2.0, label="Hermes target Te")
    axes[0, 0].axhline(series.setpoint, color="#ae2012", linestyle="--", linewidth=1.5, label="setpoint")
    axes[0, 0].set_title("Target temperature control")
    axes[0, 0].set_ylabel("normalized Te")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(series.time_points, series.reference_multiplier, color="#005f73", linewidth=2.0, label="Hermes")
    axes[0, 1].plot(series.time_points, series.reconstructed_multiplier, color="#ee9b00", linestyle="--", linewidth=1.8, label="reconstructed")
    axes[0, 1].set_title("Source multiplier")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(series.time_points, series.reference_proportional, color="#0a9396", linewidth=2.0, label="Hermes P")
    axes[1, 0].plot(series.time_points, series.reconstructed_proportional, color="#94d2bd", linestyle="--", linewidth=1.8, label="reconstructed P")
    axes[1, 0].plot(series.time_points, series.reference_integral, color="#bb3e03", linewidth=2.0, label="Hermes I")
    axes[1, 0].plot(series.time_points, series.reconstructed_integral, color="#ee9b00", linestyle="--", linewidth=1.8, label="reconstructed I")
    axes[1, 0].set_title("PI controller terms")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("controller term")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False, ncol=2)

    reference_source = series.reference_energy_source.reshape(series.reference_energy_source.shape[0], -1).max(axis=1)
    reconstructed_source = series.reconstructed_energy_source.reshape(series.reconstructed_energy_source.shape[0], -1).max(axis=1)
    axes[1, 1].plot(series.time_points, reference_source, color="#9b2226", linewidth=2.0, label="Hermes SPe")
    axes[1, 1].plot(series.time_points, reconstructed_source, color="#ca6702", linestyle="--", linewidth=1.8, label="reconstructed SPe")
    axes[1, 1].set_title("Electron energy source")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)

    figure.savefig(path, dpi=180)
    plt.close(figure)
