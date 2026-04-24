from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import threading
from time import perf_counter
from collections.abc import Callable
from typing import TypeVar

from matplotlib import pyplot as plt
import numpy as np

from ..native import run_curated_case
from ..parity.arrays import build_array_payload_from_summary_payload, build_dataset_array_payload
from ..parity.diff import build_scaled_array_diff_entries
from ..parity.reference import discover_reference_binary, resolve_reference_case, run_reference_case
from .publication_plotting import save_publication_figure, style_axis

T = TypeVar("T")


@dataclass(frozen=True)
class HermesLiveRerunCaseSpec:
    case_name: str
    display_label: str
    family: str


@dataclass(frozen=True)
class HermesLiveRerunCampaignArtifacts:
    report_json_path: Path
    report_npz_path: Path
    report_plot_png_path: Path


@dataclass(frozen=True)
class PeakRssMeasurement:
    start_rss_bytes: int | None
    end_rss_bytes: int | None
    peak_rss_bytes: int | None
    peak_rss_delta_bytes: int | None
    sample_count: int
    sampling_interval_seconds: float
    status: str


DEFAULT_HERMES_LIVE_RERUN_CASE_SPECS = (
    HermesLiveRerunCaseSpec("neutral_mixed_one_step", "neutral\nmixed", "1D neutral"),
    HermesLiveRerunCaseSpec("recycling_1d_one_step", "1D\nrecycling", "1D recycling"),
    HermesLiveRerunCaseSpec("recycling_dthe_one_step", "1D\nD/T/He\nrecycling", "1D multispecies recycling"),
    HermesLiveRerunCaseSpec("integrated_2d_recycling_one_step", "integrated\n2D\nrecycling", "2D integrated recycling"),
    HermesLiveRerunCaseSpec("tokamak_recycling_one_step", "tokamak\nrecycling", "2D diverted tokamak recycling"),
    HermesLiveRerunCaseSpec("tokamak_isothermal_one_step", "tokamak\nisothermal", "2D tokamak transport"),
    HermesLiveRerunCaseSpec("tokamak_turbulence_one_step", "tokamak\nturbulence", "2D tokamak turbulence"),
    HermesLiveRerunCaseSpec("tokamak_diffusion_transport_short_window", "tokamak\ntransport", "2D tokamak transport"),
    HermesLiveRerunCaseSpec("annulus_he_emag_one_step", "annulus\nEM", "annulus electromagnetic"),
    HermesLiveRerunCaseSpec("alfven_wave_one_step", "Alfven\nwave", "electromagnetic wave"),
)

NORMALIZATION_SENSITIVE_ABS_TOL = 1.0e-6
NORMALIZATION_SENSITIVE_REL_TOL = 1.0e-1


def create_hermes_live_rerun_campaign_package(
    *,
    reference_root: str | Path,
    output_root: str | Path,
    case_specs: tuple[HermesLiveRerunCaseSpec, ...] = DEFAULT_HERMES_LIVE_RERUN_CASE_SPECS,
    case_label: str = "hermes_live_rerun_campaign",
) -> HermesLiveRerunCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_hermes_live_rerun_campaign_report(
        reference_root=reference_root,
        case_specs=case_specs,
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report_npz_path = _write_hermes_live_rerun_campaign_arrays(report, data_dir / f"{case_label}.npz")
    report_plot_png_path = save_hermes_live_rerun_campaign_plot(report, images_dir / f"{case_label}.png")
    return HermesLiveRerunCampaignArtifacts(
        report_json_path=report_json_path,
        report_npz_path=report_npz_path,
        report_plot_png_path=report_plot_png_path,
    )


def build_hermes_live_rerun_campaign_report(
    *,
    reference_root: str | Path,
    case_specs: tuple[HermesLiveRerunCaseSpec, ...] = DEFAULT_HERMES_LIVE_RERUN_CASE_SPECS,
) -> dict[str, object]:
    reference_root_path = Path(reference_root).expanduser().resolve()
    reference_binary = discover_reference_binary(reference_root=reference_root_path)
    cases = [
        _run_hermes_live_rerun_case(spec, reference_root=reference_root_path)
        for spec in case_specs
    ]
    return {
        "reference_code": "hermes-3",
        "reference_root": _sanitize_public_path(reference_root_path),
        "reference_binary": _sanitize_public_path(reference_binary),
        "case_count": len(cases),
        "cases": cases,
        "summaries": _build_hermes_live_rerun_summary(cases),
        "notes": {
            "comparison_surface": "live_native_vs_live_reference_curated_cases",
            "runtime_note": "Runtime ratios below one indicate faster native execution on this machine for the selected parity rung.",
            "memory_note": (
                "Peak RSS is sampled from the Python process tree during each native and reference run. "
                "It is an operating-system level validation metric, not a device allocator trace."
            ),
            "normalization_note": (
                "Cases flagged as normalization-sensitive have small absolute max-error on the guarded compare "
                "surface but moderate relative error because the dominant field is near zero in the reference output."
            ),
            "three_d_status": "This live rerun matrix currently covers 1D and 2D Hermès-backed lanes. The current 3D evidence remains the selected-field reference-backed packages.",
        },
    }


def save_hermes_live_rerun_campaign_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cases = list(report["cases"])
    labels = [str(entry["display_label"]).replace("\n", " ") for entry in cases]
    families = [str(entry["family"]) for entry in cases]

    worst_rms_rel = np.asarray([float(entry["worst_relative_rms_error"]) for entry in cases], dtype=np.float64)
    worst_relmax = np.asarray(
        [
            float(entry["worst_relative_to_expected_max"])
            if entry.get("worst_relative_to_expected_max") is not None
            else 0.0
            for entry in cases
        ],
        dtype=np.float64,
    )
    worst_abs = np.asarray([float(entry["worst_max_abs_diff"]) for entry in cases], dtype=np.float64)
    runtime_ratio = np.asarray([float(entry["native_to_reference_runtime_ratio"]) for entry in cases], dtype=np.float64)
    normalization_sensitive = np.asarray(
        [bool(entry.get("normalization_sensitive", False)) for entry in cases],
        dtype=bool,
    )

    floor = 1.0e-16
    rms_plot = np.maximum(worst_rms_rel, floor)
    relmax_plot = np.maximum(worst_relmax, floor)
    abs_plot = np.maximum(worst_abs, floor)
    ratio_plot = np.maximum(runtime_ratio, floor)

    family_colors = _family_color_map(families)
    bar_colors = [family_colors[family] for family in families]

    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.8))

    _plot_live_horizontal_bars(
        axes[0, 0],
        labels=labels,
        values=rms_plot,
        raw_values=worst_rms_rel,
        bar_colors=bar_colors,
        normalization_sensitive=normalization_sensitive,
        title="Worst RMS error on the guarded compare surface",
        xlabel="rms |Δ| / max |reference|",
        exact_zero=True,
    )

    _plot_live_horizontal_bars(
        axes[0, 1],
        labels=labels,
        values=relmax_plot,
        raw_values=worst_relmax,
        bar_colors=bar_colors,
        normalization_sensitive=normalization_sensitive,
        title="Worst max-error normalized by reference amplitude",
        xlabel="max |Δ| / max |reference|",
        exact_zero=True,
    )

    _plot_live_horizontal_bars(
        axes[1, 0],
        labels=labels,
        values=ratio_plot,
        raw_values=runtime_ratio,
        bar_colors=bar_colors,
        normalization_sensitive=normalization_sensitive,
        title="Native to reference wall-time ratio",
        xlabel="native / Hermès wall time",
        threshold=1.0,
        value_suffix="x",
    )

    _plot_live_horizontal_bars(
        axes[1, 1],
        labels=labels,
        values=abs_plot,
        raw_values=worst_abs,
        bar_colors=bar_colors,
        normalization_sensitive=normalization_sensitive,
        title="Worst absolute max-error on the compare surface",
        xlabel="max |Δ|",
        exact_zero=True,
    )

    figure.suptitle(
        "Live JAX-DRB versus live Hermès-3 rerun matrix across curated verification and validation lanes",
        fontsize=13.0,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.02,
        "Hatched bars mark cases where the dominant relative mismatch is driven by a near-zero reference field; use the absolute-error panel to judge physical significance.",
        ha="center",
        va="center",
        fontsize=8.8,
    )
    figure.subplots_adjust(left=0.19, right=0.985, bottom=0.12, top=0.90, wspace=0.42, hspace=0.38)
    save_publication_figure(figure, target)
    return target


def _run_hermes_live_rerun_case(
    spec: HermesLiveRerunCaseSpec,
    *,
    reference_root: Path,
) -> dict[str, object]:
    reference_case, _ = resolve_reference_case(spec.case_name, reference_root=reference_root)
    native_started_at = perf_counter()
    native_result, native_memory = _measure_peak_rss(
        lambda: run_curated_case(spec.case_name, reference_root=reference_root)
    )
    native_elapsed_seconds = perf_counter() - native_started_at

    with tempfile.TemporaryDirectory(prefix=f"jaxdrb-{spec.case_name}-") as workdir_name:
        workdir = Path(workdir_name)
        reference_started_at = perf_counter()
        reference_execution, reference_memory = _measure_peak_rss(
            lambda: run_reference_case(
                spec.case_name,
                reference_root=reference_root,
                workdir=workdir,
            )
        )
        reference_elapsed_seconds = perf_counter() - reference_started_at
        dataset_path = workdir / "BOUT.dmp.0.nc"
        reference_payload = build_dataset_array_payload(
            dataset_path,
            case_name=reference_execution.summary.case_name,
            parity_mode=reference_execution.summary.parity_mode,
            capability_tier=reference_execution.summary.capability_tier,
            compare_variables=reference_execution.summary.compare_variables,
            component_labels=reference_execution.summary.component_labels,
            overrides=reference_execution.summary.overrides,
            trim_x_guards=reference_case.trim_x_guards,
            x_guards=native_result.run_config.mesh.mxg,
            trim_y_guards=reference_case.trim_y_guards,
            y_guards=native_result.run_config.mesh.myg,
            configured_nout=reference_execution.summary.nout,
            configured_timestep=reference_execution.summary.timestep,
            producer="external-reference-rerun",
        )

    native_payload = build_array_payload_from_summary_payload(native_result.payload, native_result.variables)
    compare_variables = tuple(reference_execution.summary.compare_variables)
    scaled_entries = build_scaled_array_diff_entries(
        reference_payload["variables"],
        native_payload["variables"],
        compare_variables=compare_variables,
    )
    field_error_metrics = _compute_field_error_metrics(
        reference_payload["variables"],
        native_payload["variables"],
        compare_variables=compare_variables,
    )
    worst_l2_field, worst_l2_error = max(
        ((name, values["relative_l2_error"]) for name, values in field_error_metrics.items()),
        key=lambda item: item[1],
        default=("", 0.0),
    )
    worst_rms_field, worst_rms_error = max(
        ((name, values["relative_rms_to_expected_max"]) for name, values in field_error_metrics.items()),
        key=lambda item: item[1],
        default=("", 0.0),
    )
    worst_scaled_entry = max(
        scaled_entries,
        key=lambda entry: (
            -1.0 if entry.relative_to_expected_max is None else entry.relative_to_expected_max,
            entry.max_abs_diff,
        ),
        default=None,
    )
    return {
        "case_name": spec.case_name,
        "display_label": spec.display_label,
        "family": spec.family,
        "reference_path": reference_case.reference_path,
        "parity_mode": reference_case.parity_mode,
        "capability_tier": reference_case.capability_tier,
        "compare_variable_count": len(compare_variables),
        "native_elapsed_seconds": float(native_elapsed_seconds),
        "reference_elapsed_seconds": float(reference_elapsed_seconds),
        **_memory_report_fields("native", native_memory),
        **_memory_report_fields("reference", reference_memory),
        "native_to_reference_runtime_ratio": (
            float(native_elapsed_seconds / reference_elapsed_seconds)
            if reference_elapsed_seconds > 0.0
            else None
        ),
        "reference_to_native_speedup": (
            float(reference_elapsed_seconds / native_elapsed_seconds)
            if native_elapsed_seconds > 0.0
            else None
        ),
        "native_to_reference_peak_rss_ratio": _safe_ratio(
            native_memory.peak_rss_bytes,
            reference_memory.peak_rss_bytes,
        ),
        "native_to_reference_peak_rss_delta_ratio": _safe_ratio(
            native_memory.peak_rss_delta_bytes,
            reference_memory.peak_rss_delta_bytes,
        ),
        "worst_relative_l2_field": worst_l2_field,
        "worst_relative_l2_error": float(worst_l2_error),
        "worst_relative_rms_field": worst_rms_field,
        "worst_relative_rms_error": float(worst_rms_error),
        "worst_max_abs_field": "" if worst_scaled_entry is None else worst_scaled_entry.field,
        "worst_max_abs_diff": 0.0 if worst_scaled_entry is None else float(worst_scaled_entry.max_abs_diff),
        "worst_relative_to_expected_max": (
            None
            if worst_scaled_entry is None or worst_scaled_entry.relative_to_expected_max is None
            else float(worst_scaled_entry.relative_to_expected_max)
        ),
        "normalization_sensitive": _is_normalization_sensitive_case(worst_scaled_entry),
        "exact_match": bool(
            worst_scaled_entry is not None
            and worst_scaled_entry.max_abs_diff == 0.0
            and worst_l2_error == 0.0
        ),
    }


def _compute_field_error_metrics(
    reference_fields: dict[str, np.ndarray],
    candidate_fields: dict[str, np.ndarray],
    *,
    compare_variables: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    errors: dict[str, dict[str, float]] = {}
    tiny = np.finfo(np.float64).tiny
    for name in compare_variables:
        if name not in reference_fields or name not in candidate_fields:
            continue
        reference = np.asarray(reference_fields[name], dtype=np.float64)
        candidate = np.asarray(candidate_fields[name], dtype=np.float64)
        valid = np.isfinite(reference) & np.isfinite(candidate)
        if not np.any(valid):
            errors[name] = {
                "relative_l2_error": float("inf"),
                "relative_rms_to_expected_max": float("inf"),
            }
            continue
        reference_values = reference[valid]
        diff_values = candidate[valid] - reference_values
        reference_abs_max = max(float(np.max(np.abs(reference_values))), tiny)
        errors[name] = {
            "relative_l2_error": float(np.linalg.norm(diff_values) / max(np.linalg.norm(reference_values), tiny)),
            "relative_rms_to_expected_max": float(np.sqrt(np.mean(np.square(diff_values))) / reference_abs_max),
        }
    return errors


def _build_hermes_live_rerun_summary(cases: list[dict[str, object]]) -> dict[str, object]:
    if not cases:
        return {
            "exact_match_case_count": 0,
            "worst_runtime_ratio_case": None,
            "best_runtime_ratio_case": None,
            "worst_relative_l2_case": None,
            "worst_relative_l2_error": None,
            "worst_relative_rms_case": None,
            "worst_relative_rms_error": None,
            "worst_native_peak_rss_case": None,
            "worst_native_peak_rss_mebibytes": None,
            "worst_native_to_reference_peak_rss_ratio_case": None,
            "worst_native_to_reference_peak_rss_ratio": None,
        }
    exact_match_case_count = sum(bool(entry["exact_match"]) for entry in cases)
    runtime_sorted = sorted(cases, key=lambda entry: float(entry["native_to_reference_runtime_ratio"]))
    worst_l2_entry = max(cases, key=lambda entry: float(entry["worst_relative_l2_error"]))
    worst_rms_entry = max(cases, key=lambda entry: float(entry["worst_relative_rms_error"]))
    memory_measured_cases = [
        entry for entry in cases
        if entry.get("native_peak_rss_bytes") is not None
    ]
    memory_ratio_cases = [
        entry for entry in cases
        if entry.get("native_to_reference_peak_rss_ratio") is not None
    ]
    worst_native_memory_entry = (
        None
        if not memory_measured_cases
        else max(memory_measured_cases, key=lambda entry: float(entry["native_peak_rss_bytes"]))
    )
    worst_memory_ratio_entry = (
        None
        if not memory_ratio_cases
        else max(memory_ratio_cases, key=lambda entry: float(entry["native_to_reference_peak_rss_ratio"]))
    )
    normalization_sensitive_cases = [
        str(entry["case_name"])
        for entry in cases
        if bool(entry.get("normalization_sensitive", False))
    ]
    return {
        "exact_match_case_count": int(exact_match_case_count),
        "best_runtime_ratio_case": runtime_sorted[0]["case_name"],
        "best_runtime_ratio": float(runtime_sorted[0]["native_to_reference_runtime_ratio"]),
        "worst_runtime_ratio_case": runtime_sorted[-1]["case_name"],
        "worst_runtime_ratio": float(runtime_sorted[-1]["native_to_reference_runtime_ratio"]),
        "worst_relative_l2_case": worst_l2_entry["case_name"],
        "worst_relative_l2_error": float(worst_l2_entry["worst_relative_l2_error"]),
        "worst_relative_rms_case": worst_rms_entry["case_name"],
        "worst_relative_rms_error": float(worst_rms_entry["worst_relative_rms_error"]),
        "worst_native_peak_rss_case": None if worst_native_memory_entry is None else worst_native_memory_entry["case_name"],
        "worst_native_peak_rss_mebibytes": (
            None
            if worst_native_memory_entry is None
            else float(worst_native_memory_entry["native_peak_rss_mebibytes"])
        ),
        "worst_native_to_reference_peak_rss_ratio_case": (
            None if worst_memory_ratio_entry is None else worst_memory_ratio_entry["case_name"]
        ),
        "worst_native_to_reference_peak_rss_ratio": (
            None
            if worst_memory_ratio_entry is None
            else float(worst_memory_ratio_entry["native_to_reference_peak_rss_ratio"])
        ),
        "normalization_sensitive_case_count": len(normalization_sensitive_cases),
        "normalization_sensitive_cases": normalization_sensitive_cases,
    }


def _write_hermes_live_rerun_campaign_arrays(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cases = list(report["cases"])
    np.savez_compressed(
        target,
        case_names=np.asarray([str(entry["case_name"]) for entry in cases], dtype=object),
        display_labels=np.asarray([str(entry["display_label"]) for entry in cases], dtype=object),
        families=np.asarray([str(entry["family"]) for entry in cases], dtype=object),
        native_elapsed_seconds=np.asarray([float(entry["native_elapsed_seconds"]) for entry in cases], dtype=np.float64),
        reference_elapsed_seconds=np.asarray([float(entry["reference_elapsed_seconds"]) for entry in cases], dtype=np.float64),
        native_peak_rss_bytes=np.asarray(
            [_nan_if_none(entry.get("native_peak_rss_bytes")) for entry in cases],
            dtype=np.float64,
        ),
        reference_peak_rss_bytes=np.asarray(
            [_nan_if_none(entry.get("reference_peak_rss_bytes")) for entry in cases],
            dtype=np.float64,
        ),
        native_to_reference_peak_rss_ratio=np.asarray(
            [_nan_if_none(entry.get("native_to_reference_peak_rss_ratio")) for entry in cases],
            dtype=np.float64,
        ),
        native_to_reference_runtime_ratio=np.asarray(
            [float(entry["native_to_reference_runtime_ratio"]) for entry in cases],
            dtype=np.float64,
        ),
        worst_relative_l2_error=np.asarray([float(entry["worst_relative_l2_error"]) for entry in cases], dtype=np.float64),
        worst_relative_rms_error=np.asarray([float(entry["worst_relative_rms_error"]) for entry in cases], dtype=np.float64),
        worst_max_abs_diff=np.asarray([float(entry["worst_max_abs_diff"]) for entry in cases], dtype=np.float64),
        worst_relative_to_expected_max=np.asarray(
            [
                0.0 if entry.get("worst_relative_to_expected_max") is None else float(entry["worst_relative_to_expected_max"])
                for entry in cases
            ],
            dtype=np.float64,
        ),
        normalization_sensitive=np.asarray(
            [bool(entry.get("normalization_sensitive", False)) for entry in cases],
            dtype=bool,
        ),
    )
    return target


def _measure_peak_rss(
    function: Callable[[], T],
    *,
    sampling_interval_seconds: float = 0.10,
) -> tuple[T, PeakRssMeasurement]:
    samples: list[int] = []
    sample_failures = 0
    stop_event = threading.Event()
    root_pid = os.getpid()

    def append_sample() -> None:
        nonlocal sample_failures
        value = _process_tree_rss_bytes(root_pid)
        if value is None:
            sample_failures += 1
            return
        samples.append(value)

    append_sample()

    def sample_loop() -> None:
        while not stop_event.wait(sampling_interval_seconds):
            append_sample()

    sampler = threading.Thread(target=sample_loop, name="jaxdrb-rss-sampler", daemon=True)
    sampler.start()
    try:
        result = function()
    finally:
        stop_event.set()
        sampler.join(timeout=max(1.0, 4.0 * sampling_interval_seconds))
    append_sample()
    start_rss_bytes = samples[0] if samples else None
    end_rss_bytes = samples[-1] if samples else None
    peak_rss_bytes = max(samples) if samples else None
    peak_rss_delta_bytes = (
        None
        if peak_rss_bytes is None or start_rss_bytes is None
        else max(int(peak_rss_bytes - start_rss_bytes), 0)
    )
    status = "sampled_process_tree_rss"
    if not samples:
        status = "unavailable"
    elif sample_failures:
        status = "sampled_process_tree_rss_with_partial_failures"
    return result, PeakRssMeasurement(
        start_rss_bytes=start_rss_bytes,
        end_rss_bytes=end_rss_bytes,
        peak_rss_bytes=peak_rss_bytes,
        peak_rss_delta_bytes=peak_rss_delta_bytes,
        sample_count=len(samples),
        sampling_interval_seconds=float(sampling_interval_seconds),
        status=status,
    )


def _process_tree_rss_bytes(root_pid: int) -> int | None:
    pids = _process_tree_pids(root_pid)
    rss_kib = 0
    observed = False
    for pid in pids:
        pid_rss = _process_rss_kib(pid)
        if pid_rss is None:
            continue
        rss_kib += pid_rss
        observed = True
    return int(rss_kib * 1024) if observed else None


def _process_tree_pids(root_pid: int) -> list[int]:
    pids: list[int] = [int(root_pid)]
    frontier: list[int] = [int(root_pid)]
    while frontier:
        parent = frontier.pop()
        try:
            completed = subprocess.run(
                ["pgrep", "-P", str(parent)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            continue
        if completed.returncode not in (0, 1):
            continue
        children = [int(value) for value in completed.stdout.split() if value.isdigit()]
        for child in children:
            if child not in pids:
                pids.append(child)
                frontier.append(child)
    return pids


def _process_rss_kib(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    if not text:
        return None
    try:
        return int(float(text.splitlines()[-1].strip()))
    except ValueError:
        return None


def _memory_report_fields(prefix: str, measurement: PeakRssMeasurement) -> dict[str, object]:
    return {
        f"{prefix}_memory_measurement_status": measurement.status,
        f"{prefix}_memory_sample_count": int(measurement.sample_count),
        f"{prefix}_memory_sampling_interval_seconds": float(measurement.sampling_interval_seconds),
        f"{prefix}_start_rss_bytes": measurement.start_rss_bytes,
        f"{prefix}_end_rss_bytes": measurement.end_rss_bytes,
        f"{prefix}_peak_rss_bytes": measurement.peak_rss_bytes,
        f"{prefix}_peak_rss_delta_bytes": measurement.peak_rss_delta_bytes,
        f"{prefix}_peak_rss_mebibytes": _bytes_to_mebibytes(measurement.peak_rss_bytes),
        f"{prefix}_peak_rss_delta_mebibytes": _bytes_to_mebibytes(measurement.peak_rss_delta_bytes),
    }


def _safe_ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator)


def _bytes_to_mebibytes(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value / (1024.0 * 1024.0))


def _nan_if_none(value: object) -> float:
    return float("nan") if value is None else float(value)


def _family_color_map(families: list[str]) -> dict[str, str]:
    palette = (
        "#005f73",
        "#0a9396",
        "#94d2bd",
        "#ee9b00",
        "#ca6702",
        "#bb3e03",
        "#ae2012",
        "#9b2226",
    )
    ordered_families = list(dict.fromkeys(families))
    return {
        family: palette[index % len(palette)]
        for index, family in enumerate(ordered_families)
    }


def _plot_live_horizontal_bars(
    axis,
    *,
    labels: list[str],
    values: np.ndarray,
    raw_values: np.ndarray,
    bar_colors: list[str],
    normalization_sensitive: np.ndarray,
    title: str,
    xlabel: str,
    threshold: float | None = None,
    exact_zero: bool = False,
    value_suffix: str = "",
) -> None:
    y = np.arange(len(labels), dtype=np.float64)
    bars = axis.barh(y, values, color=bar_colors, height=0.68)
    _apply_normalization_sensitive_hatch(bars, normalization_sensitive)
    if threshold is not None:
        axis.axvline(float(threshold), color="#6c757d", linestyle="--", linewidth=1.1)
    style_axis(axis, title=title, xlabel=xlabel, xscale="log", grid="x")
    positive = values[values > 0.0]
    lower = max(float(np.min(positive)) * 0.45, 1.0e-17) if positive.size else 1.0e-17
    upper = float(np.max(values)) * 7.0 if values.size else 1.0
    axis.set_xlim(lower, upper)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.tick_params(axis="y", labelsize=8.8)
    for yi, plotted, raw in zip(y, values, raw_values, strict=True):
        if exact_zero and raw == 0.0:
            label = "exact"
        elif value_suffix == "x":
            label = f"{raw:.2f}x" if raw >= 1.0e-2 else f"{raw:.2e}x"
        else:
            label = f"{raw:.2e}"
        axis.text(float(plotted * 1.18), float(yi), label, ha="left", va="center", fontsize=8.2)


def _annotate_runtime_ratio_bars(axis, x: np.ndarray, runtime_ratio: np.ndarray) -> None:
    values = np.asarray(runtime_ratio, dtype=np.float64)
    positive = np.abs(values[np.nonzero(values)])
    scale = float(np.max(positive)) if positive.size else 1.0
    offset = 0.03 * scale
    for xi, value in zip(np.asarray(x, dtype=np.float64), values, strict=True):
        if value >= 1.0e-2:
            label = f"{value:.2f}x"
        else:
            label = f"{value:.2e}x"
        axis.text(
            float(xi),
            float(value + offset),
            label,
            ha="center",
            va="bottom",
            fontsize=8.4,
        )


def _annotate_fidelity_bars(axis, x: np.ndarray, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 0.0]
    scale = float(np.max(positive)) if positive.size else 1.0
    offset = 0.03 * scale
    for xi, value in zip(np.asarray(x, dtype=np.float64), values, strict=True):
        label = "exact" if value == 0.0 else f"{value:.2e}"
        anchor = float(max(value, np.finfo(np.float64).tiny) + offset)
        axis.text(
            float(xi),
            anchor,
            label,
            ha="center",
            va="bottom",
            fontsize=8.4,
        )


def _apply_normalization_sensitive_hatch(bar_container, normalization_sensitive: np.ndarray) -> None:
    for bar, flagged in zip(bar_container.patches, normalization_sensitive, strict=True):
        if bool(flagged):
            bar.set_hatch("//")
            bar.set_edgecolor("black")
            bar.set_linewidth(0.7)


def _is_normalization_sensitive_case(worst_scaled_entry) -> bool:
    if worst_scaled_entry is None or worst_scaled_entry.relative_to_expected_max is None:
        return False
    return bool(
        worst_scaled_entry.max_abs_diff <= NORMALIZATION_SENSITIVE_ABS_TOL
        and worst_scaled_entry.relative_to_expected_max >= NORMALIZATION_SENSITIVE_REL_TOL
    )


def _sanitize_public_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    home = Path.home().resolve()
    try:
        return f"~/{resolved.relative_to(home).as_posix()}"
    except ValueError:
        return resolved.as_posix()
