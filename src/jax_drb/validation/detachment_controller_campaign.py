from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from matplotlib import pyplot as plt
from netCDF4 import Dataset
import numpy as np

from ..parity.reference import discover_reference_binary
from ..reference.paths import require_reference_root


_INCOMPATIBLE_IMPLICIT_SOLVER_OPTIONS = (
    "snes_type",
    "ksp_type",
    "max_nonlinear_iterations",
    "lag_jacobian",
)


@dataclass(frozen=True)
class DetachmentControllerCampaignMetric:
    name: str
    kind: str
    value: float
    target: float
    passed: bool
    notes: str


@dataclass(frozen=True)
class DetachmentControllerCampaignArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


@dataclass(frozen=True)
class _DetachmentControllerSeries:
    time_points: np.ndarray
    front_location: np.ndarray
    setpoint: float
    source_multiplier: np.ndarray
    reconstructed_multiplier: np.ndarray
    proportional_term: np.ndarray
    reconstructed_proportional_term: np.ndarray
    integral_term: np.ndarray
    derivative_term: np.ndarray
    source_feedback: np.ndarray
    reconstructed_source_feedback: np.ndarray


def create_detachment_controller_campaign_package(
    *,
    output_root: str | Path,
    reference_root: str | Path | None = None,
    reference_binary: str | Path | None = None,
    case_label: str = "detachment_controller_campaign",
    nout: int = 10,
    timestep: float = 100.0,
    ny: int = 16,
    solver_type: str = "cvode",
    settling_time: float = 0.0,
    timeout_seconds: int = 180,
) -> DetachmentControllerCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_detachment_controller_campaign(
        reference_root=reference_root,
        reference_binary=reference_binary,
        nout=nout,
        timestep=timestep,
        ny=ny,
        solver_type=solver_type,
        settling_time=settling_time,
        timeout_seconds=timeout_seconds,
    )
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report["summary"], indent=2, sort_keys=True), encoding="utf-8")

    series = report["series"]
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(
        arrays_npz_path,
        time_points=series.time_points,
        front_location=series.front_location,
        setpoint=np.asarray([series.setpoint], dtype=np.float64),
        source_multiplier=series.source_multiplier,
        reconstructed_multiplier=series.reconstructed_multiplier,
        proportional_term=series.proportional_term,
        reconstructed_proportional_term=series.reconstructed_proportional_term,
        integral_term=series.integral_term,
        derivative_term=series.derivative_term,
        source_feedback=series.source_feedback,
        reconstructed_source_feedback=series.reconstructed_source_feedback,
    )

    plot_png_path = images_dir / f"{case_label}.png"
    _save_detachment_controller_plot(series, plot_png_path)
    return DetachmentControllerCampaignArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_detachment_controller_campaign(
    *,
    reference_root: str | Path | None = None,
    reference_binary: str | Path | None = None,
    nout: int = 10,
    timestep: float = 100.0,
    ny: int = 16,
    solver_type: str = "cvode",
    settling_time: float = 0.0,
    timeout_seconds: int = 180,
) -> dict[str, object]:
    resolved_reference_root = Path(reference_root) if reference_root is not None else require_reference_root()
    series, timing_seconds = _build_detachment_controller_series(
        reference_root=resolved_reference_root,
        reference_binary=Path(reference_binary) if reference_binary is not None else None,
        nout=nout,
        timestep=timestep,
        ny=ny,
        solver_type=solver_type,
        settling_time=settling_time,
        timeout_seconds=timeout_seconds,
    )
    response_span = float(np.max(series.source_multiplier) - np.min(series.source_multiplier))
    integral_backslide = float(max(0.0, np.max(-np.diff(series.integral_term)))) if series.integral_term.size > 1 else 0.0
    derivative_span = float(np.max(np.abs(series.derivative_term)))
    source_balance = float(np.max(np.abs(series.source_feedback - series.reconstructed_source_feedback)))
    proportional_balance = float(np.max(np.abs(series.proportional_term - series.reconstructed_proportional_term)))
    multiplier_balance = float(np.max(np.abs(series.source_multiplier - series.reconstructed_multiplier)))
    metrics = (
        DetachmentControllerCampaignMetric(
            name="detachment_control_proportional_term_exact",
            kind="max_abs_error",
            value=proportional_balance,
            target=1.0e-12,
            passed=proportional_balance <= 1.0e-12,
            notes="The bounded reduced detachment-controller lane reproduces the proportional term directly from the front-location error.",
        ),
        DetachmentControllerCampaignMetric(
            name="detachment_control_src_mult_balance_exact",
            kind="max_abs_error",
            value=multiplier_balance,
            target=1.0e-12,
            passed=multiplier_balance <= 1.0e-12,
            notes="The saved source multiplier matches the reduced position-form controller balance `offset + P + I + D` exactly.",
        ),
        DetachmentControllerCampaignMetric(
            name="detachment_source_feedback_exact",
            kind="max_abs_error",
            value=source_balance,
            target=1.0e-12,
            passed=source_balance <= 1.0e-12,
            notes="The detachment control source stays equal to `source_multiplier * source_shape` on the bounded reduced lane.",
        ),
        DetachmentControllerCampaignMetric(
            name="detachment_control_response_span",
            kind="min_span",
            value=response_span,
            target=2.0e-2,
            passed=response_span >= 2.0e-2,
            notes="The reduced detachment-controller probe shows a nontrivial control response instead of a frozen multiplier.",
        ),
        DetachmentControllerCampaignMetric(
            name="detachment_control_integral_monotone",
            kind="max_backslide",
            value=integral_backslide,
            target=1.0e-12,
            passed=integral_backslide <= 1.0e-12,
            notes="With constant positive error on the reduced particles-actuator lane, the integral term should not decrease.",
        ),
        DetachmentControllerCampaignMetric(
            name="detachment_control_derivative_near_zero",
            kind="max_abs_value",
            value=derivative_span,
            target=1.0e-10,
            passed=derivative_span <= 1.0e-10,
            notes="The reduced lane keeps the detachment front stationary enough that the derivative term remains numerically negligible.",
        ),
    )
    summary = {
        "family": "impurity_radiation_and_detachment_control",
        "reference_mode": "external_example_solver_patch_reduced",
        "example": "tokamak-1D/extra/1D-recycling-with-detachment-control",
        "nout": int(nout),
        "timestep": float(timestep),
        "ny": int(ny),
        "solver_type": solver_type,
        "settling_time": float(settling_time),
        "metric_count": len(metrics),
        "passed_metric_count": sum(1 for metric in metrics if metric.passed),
        "timing_seconds": timing_seconds,
        "stripped_solver_options": list(_INCOMPATIBLE_IMPLICIT_SOLVER_OPTIONS if solver_type != "beuler" else ()),
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


def _build_detachment_controller_series(
    *,
    reference_root: Path,
    reference_binary: Path | None,
    nout: int,
    timestep: float,
    ny: int,
    solver_type: str,
    settling_time: float,
    timeout_seconds: int,
) -> tuple[_DetachmentControllerSeries, dict[str, float]]:
    overall_start = perf_counter()
    binary = discover_reference_binary(reference_binary=reference_binary, reference_root=reference_root)
    example_dir = reference_root / "examples" / "tokamak-1D" / "extra" / "1D-recycling-with-detachment-control"
    if not example_dir.exists():
        raise FileNotFoundError(f"Detachment-controller example not found: {example_dir}")

    with tempfile.TemporaryDirectory(prefix="jax_drb_detachment_controller_") as temp_dir:
        workdir = Path(temp_dir)
        stage_start = perf_counter()
        _stage_detachment_controller_example(
            example_dir,
            workdir=workdir,
            nout=nout,
            timestep=timestep,
            ny=ny,
            solver_type=solver_type,
            settling_time=settling_time,
        )
        stage_elapsed = perf_counter() - stage_start

        run_start = perf_counter()
        _run_detachment_controller_example(binary=binary, workdir=workdir, timeout_seconds=timeout_seconds)
        run_elapsed = perf_counter() - run_start

        load_start = perf_counter()
        controller_options = _read_bout_section_options(workdir / "BOUT.inp", section="detachment_controller")
        setpoint = float(controller_options["detachment_front_setpoint"])
        velocity_form = _parse_bool(controller_options["velocity_form"])
        min_time_for_change = float(controller_options["min_time_for_change"])
        min_error_for_change = float(controller_options["min_error_for_change"])
        minval_for_source_multiplier = float(controller_options["minval_for_source_multiplier"])
        maxval_for_source_multiplier = float(controller_options["maxval_for_source_multiplier"])
        actuator = _strip_quotes(controller_options["actuator"])
        initial_control = float(controller_options["initial_control"])
        control_offset = float(controller_options["control_offset"])
        reset_integral_on_first_crossing = _parse_bool(controller_options["reset_integral_on_first_crossing"])
        controller_gain = float(controller_options["controller_gain"])
        integral_time = float(controller_options["integral_time"])
        derivative_time = float(controller_options["derivative_time"])
        buffer_size = int(controller_options["buffer_size"])

        with Dataset(workdir / "BOUT.dmp.0.nc") as dataset:
            time_points = _extract_time_points(dataset)
            omega_ci = float(np.asarray(dataset.variables["Omega_ci"][:], dtype=np.float64).reshape(-1)[0])
            front_location = _extract_scalar_series(dataset, "detachment_front_location")
            source_multiplier = _extract_scalar_series(dataset, "detachment_control_src_mult")
            proportional_term = _extract_scalar_series(dataset, "detachment_control_proportional_term")
            integral_term = _extract_scalar_series(dataset, "detachment_control_integral_term")
            derivative_term = _extract_scalar_series(dataset, "detachment_control_derivative_term")
            source_feedback = _extract_spatial_series(dataset, "detachment_source_feedback", time_count=time_points.size)
            source_shape = _extract_spatial_series(dataset, "detachment_control_src_shape", time_count=time_points.size)
        load_elapsed = perf_counter() - load_start

        reconstruct_start = perf_counter()
        time_seconds = np.asarray(time_points, dtype=np.float64) / omega_ci
        response_sign = -1.0 if actuator == "power" else 1.0
        reconstructed_multiplier, reconstructed_proportional, _, _ = _reconstruct_detachment_controller(
            time_seconds=time_seconds,
            front_location=front_location,
            setpoint=setpoint,
            velocity_form=velocity_form,
            min_time_for_change=min_time_for_change,
            min_error_for_change=min_error_for_change,
            minval_for_source_multiplier=minval_for_source_multiplier,
            maxval_for_source_multiplier=maxval_for_source_multiplier,
            control_offset=control_offset,
            initial_control=initial_control,
            controller_gain=controller_gain,
            integral_time=integral_time,
            derivative_time=derivative_time,
            buffer_size=buffer_size,
            response_sign=response_sign,
            reset_integral_on_first_crossing=reset_integral_on_first_crossing,
            settling_time=settling_time,
        )
        balance_multiplier = (
            np.asarray([initial_control], dtype=np.float64)
            if source_multiplier.size == 1
            else np.concatenate(
                (
                    np.asarray([initial_control], dtype=np.float64),
                    np.asarray(control_offset + proportional_term[1:] + integral_term[1:] + derivative_term[1:], dtype=np.float64),
                )
            )
        )
        reconstructed_source_feedback = source_shape * balance_multiplier[:, None, None, None]
        reconstruct_elapsed = perf_counter() - reconstruct_start

    return _DetachmentControllerSeries(
        time_points=time_points,
        front_location=front_location,
        setpoint=float(setpoint),
        source_multiplier=source_multiplier,
        reconstructed_multiplier=balance_multiplier,
        proportional_term=proportional_term,
        reconstructed_proportional_term=reconstructed_proportional,
        integral_term=integral_term,
        derivative_term=derivative_term,
        source_feedback=source_feedback,
        reconstructed_source_feedback=reconstructed_source_feedback,
    ), {
        "stage_input": float(stage_elapsed),
        "reference_run": float(run_elapsed),
        "load_dataset": float(load_elapsed),
        "reconstruct_controller": float(reconstruct_elapsed),
        "total": float(perf_counter() - overall_start),
    }


def _strip_quotes(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _parse_bool(value: str) -> bool:
    normalized = _strip_quotes(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Could not parse boolean value {value!r}")


def _read_bout_section_options(path: Path, *, section: str) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    active_section: str | None = None
    options: dict[str, str] = {}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(";"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            active_section = stripped[1:-1].strip()
            continue
        if active_section != section or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        options[key.strip()] = value.strip()
    if not options:
        raise KeyError(f"Missing section [{section}] in {path}")
    return options


def _stage_detachment_controller_example(
    example_dir: Path,
    *,
    workdir: Path,
    nout: int,
    timestep: float,
    ny: int,
    solver_type: str,
    settling_time: float,
) -> None:
    for child in example_dir.iterdir():
        if child.is_file():
            shutil.copy2(child, workdir / child.name)
    input_path = workdir / "BOUT.inp"
    text = input_path.read_text(encoding="utf-8")
    text = _replace_bout_setting(text, "nout", str(int(nout)))
    text = _replace_bout_setting(text, "timestep", f"{float(timestep):g}")
    text = _replace_bout_setting(text, "ny", str(int(ny)))
    text = _replace_bout_setting(text, "type", solver_type)
    text = _replace_bout_setting(text, "settling_time", f"{float(settling_time):g}")
    if solver_type != "beuler":
        text = _strip_solver_option_lines(text, _INCOMPATIBLE_IMPLICIT_SOLVER_OPTIONS)
    input_path.write_text(text, encoding="utf-8")


def _replace_bout_setting(text: str, key: str, value: str) -> str:
    pattern = rf"(?m)^({re.escape(key)}\s*=\s*).*$"
    replaced, count = re.subn(pattern, lambda match: f"{match.group(1)}{value}", text, count=1)
    if count != 1:
        raise ValueError(f"Could not replace {key!r} in patched BOUT.inp")
    return replaced


def _strip_solver_option_lines(text: str, keys: tuple[str, ...]) -> str:
    updated = text
    for key in keys:
        updated = re.sub(rf"(?m)^.*\b{re.escape(key)}\b.*\n", "", updated)
    return updated


def _run_detachment_controller_example(
    *,
    binary: Path,
    workdir: Path,
    timeout_seconds: int,
) -> None:
    stdout_path = workdir / "run.stdout"
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_stream:
            result = subprocess.run(
                [str(binary), "-d", "."],
                cwd=workdir,
                stdout=stdout_stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Detachment-controller reference run did not finish within {timeout_seconds}s in {workdir}"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"Detachment-controller reference run failed with exit code {result.returncode}; see {stdout_path}"
        )
    if not (workdir / "BOUT.dmp.0.nc").exists():
        raise FileNotFoundError(f"Detachment-controller reference run did not produce BOUT.dmp.0.nc in {workdir}")


def _extract_time_points(dataset: Dataset) -> np.ndarray:
    if "t_array" in dataset.variables:
        return np.asarray(dataset.variables["t_array"][:], dtype=np.float64).reshape(-1)
    if "t" in dataset.variables:
        return np.asarray(dataset.variables["t"][:], dtype=np.float64).reshape(-1)
    raise KeyError("missing time coordinate in detachment-controller dataset")


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


def _reconstruct_detachment_proportional_term(
    *,
    front_location: np.ndarray,
    setpoint: float,
    controller_gain: float,
    response_sign: float,
) -> np.ndarray:
    error = float(setpoint) - np.asarray(front_location, dtype=np.float64).reshape(-1)
    return response_sign * controller_gain * error


def _calculate_gradient(x: list[float], y: list[float]) -> float:
    if not x or len(x) != len(y):
        return 0.0
    sum_x = float(np.sum(x))
    sum_y = float(np.sum(y))
    sum_xy = float(np.sum(np.asarray(x) * np.asarray(y)))
    sum_x_squared = float(np.sum(np.asarray(x) * np.asarray(x)))
    n_real = float(len(x))
    numerator = (n_real * sum_xy) - (sum_x * sum_y)
    denominator = (n_real * sum_x_squared) - (sum_x * sum_x)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _reconstruct_detachment_controller(
    *,
    time_seconds: np.ndarray,
    front_location: np.ndarray,
    setpoint: float,
    velocity_form: bool,
    min_time_for_change: float,
    min_error_for_change: float,
    minval_for_source_multiplier: float,
    maxval_for_source_multiplier: float,
    control_offset: float,
    initial_control: float,
    controller_gain: float,
    integral_time: float,
    derivative_time: float,
    buffer_size: int,
    response_sign: float,
    reset_integral_on_first_crossing: bool,
    settling_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_seconds = np.asarray(time_seconds, dtype=np.float64).reshape(-1)
    front_location = np.asarray(front_location, dtype=np.float64).reshape(-1)
    if time_seconds.shape != front_location.shape:
        raise ValueError("time_seconds and front_location must have matching shape")

    control = np.zeros_like(front_location)
    proportional_term = np.zeros_like(front_location)
    integral_term = np.zeros_like(front_location)
    derivative_term = np.zeros_like(front_location)

    error_integral = 0.0
    previous_time = 0.0
    previous_error = 0.0
    previous_control = float(initial_control)
    previous_derivative = 0.0
    first_step = True
    number_of_crossings = 0.0
    time_buffer: list[float] = []
    error_buffer: list[float] = []
    current_control = float(initial_control)
    current_p = 0.0
    current_i = 0.0
    current_d = 0.0

    for index, (time_value, location_value) in enumerate(zip(time_seconds, front_location, strict=True)):
        error = float(setpoint) - float(location_value)
        if (time_value > settling_time) and ((time_value - previous_time) >= min_time_for_change) and (abs(error - previous_error) >= min_error_for_change):
            change_in_time = float(time_value) - previous_time
            if len(time_buffer) >= buffer_size:
                time_buffer.pop(0)
                error_buffer.pop(0)
            time_buffer.append(float(time_value))
            error_buffer.append(error)
            derivative = _calculate_gradient(time_buffer, error_buffer)
            change_in_error = 0.0 if first_step else error - previous_error
            change_in_derivative = 0.0 if first_step else derivative - previous_derivative
            error_integral = 0.0 if first_step else error_integral + change_in_time * 0.5 * (error + previous_error)

            if velocity_form:
                current_p = response_sign * controller_gain * change_in_error
                current_i = 0.0 if np.isinf(integral_time) else response_sign * controller_gain * (change_in_time / integral_time) * error
                current_d = response_sign * controller_gain * derivative_time * change_in_derivative
                current_control = previous_control + current_p + current_i + current_d
            else:
                current_p = response_sign * controller_gain * error
                current_i = 0.0 if np.isinf(integral_time) else response_sign * controller_gain * error_integral / integral_time
                current_d = response_sign * controller_gain * derivative_time * derivative
                current_control = control_offset + current_p + current_i + current_d

            current_control = min(max(current_control, minval_for_source_multiplier), maxval_for_source_multiplier)

            if (((error < 0.0) and (previous_error > 0.0)) or ((error > 0.0) and (previous_error < 0.0))):
                if (number_of_crossings < 1.0) and reset_integral_on_first_crossing:
                    error_integral = 0.0
                number_of_crossings += 1.0

            previous_time = float(time_value)
            previous_error = error
            previous_control = current_control
            previous_derivative = derivative
            first_step = False

        control[index] = current_control
        proportional_term[index] = current_p
        integral_term[index] = current_i
        derivative_term[index] = current_d
    return control, proportional_term, integral_term, derivative_term


def _save_detachment_controller_plot(series: _DetachmentControllerSeries, path: Path) -> None:
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.5, 8.0), constrained_layout=True)

    axes[0, 0].plot(series.time_points, series.front_location, color="#0a9396", linewidth=2.0, label="front")
    axes[0, 0].axhline(series.setpoint, color="#ae2012", linestyle="--", linewidth=1.5, label="setpoint")
    axes[0, 0].set_title("Detachment front location")
    axes[0, 0].set_ylabel("distance from target [m]")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(series.time_points, series.source_multiplier, color="#005f73", linewidth=2.0, label="Hermes")
    axes[0, 1].plot(series.time_points, series.reconstructed_multiplier, color="#ee9b00", linestyle="--", linewidth=1.8, label="balance")
    axes[0, 1].set_title("Controller multiplier")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(series.time_points, series.proportional_term, color="#0a9396", linewidth=2.0, label="Hermes P")
    axes[1, 0].plot(series.time_points, series.reconstructed_proportional_term, color="#94d2bd", linestyle="--", linewidth=1.8, label="reconstructed P")
    axes[1, 0].plot(series.time_points, series.integral_term, color="#bb3e03", linewidth=2.0, label="Hermes I")
    axes[1, 0].plot(series.time_points, series.derivative_term, color="#ee9b00", linewidth=1.8, label="Hermes D")
    axes[1, 0].set_title("Controller terms")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("term")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False, ncol=2)

    reference_source = series.source_feedback.reshape(series.source_feedback.shape[0], -1).max(axis=1)
    reconstructed_source = series.reconstructed_source_feedback.reshape(series.reconstructed_source_feedback.shape[0], -1).max(axis=1)
    axes[1, 1].plot(series.time_points, reference_source, color="#9b2226", linewidth=2.0, label="Hermes source")
    axes[1, 1].plot(series.time_points, reconstructed_source, color="#ca6702", linestyle="--", linewidth=1.8, label="mult * shape")
    axes[1, 1].set_title("Feedback source")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)

    figure.savefig(path, dpi=180)
    plt.close(figure)
