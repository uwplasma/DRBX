from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import jax
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import numpy as np

from .native_3d_runtime_campaign import _write_metric_grid, _write_vmec_case
from .publication_plotting import annotate_bars, save_publication_figure, style_axis
from .stellarator_vmec_native_selected_field import _native_vmec_profile_batch
from .stellarator_vmec_selected_field import _load_vmec_selected_fields
from .traced_field_line_native_selected_field import _native_radial_profile_batch
from .traced_field_line_selected_field import _load_metric_fields


@dataclass(frozen=True)
class JaxNativeProfileAuditArtifacts:
    summary_json_path: Path
    summary_plot_png_path: Path


def create_jax_native_profile_audit_package(
    *,
    output_root: str | Path,
    case_label: str = "jax_native_profile_audit",
) -> JaxNativeProfileAuditArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    traces_dir = root / "traces"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    report = build_jax_native_profile_audit_report(traces_root=traces_dir)
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary_plot_png_path = save_jax_native_profile_audit_plot(report, images_dir / f"{case_label}.png")
    return JaxNativeProfileAuditArtifacts(
        summary_json_path=summary_json_path,
        summary_plot_png_path=summary_plot_png_path,
    )


def build_jax_native_profile_audit_report(
    *,
    traces_root: str | Path,
) -> dict[str, object]:
    trace_root = Path(traces_root)
    trace_root.mkdir(parents=True, exist_ok=True)
    traced_report = _profile_traced_field_line_kernel(trace_root / "traced_field_line")
    stellarator_report = _profile_stellarator_vmec_kernel(trace_root / "stellarator_vmec")
    return {
        "case": "jax_native_profile_audit",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "lanes": [traced_report, stellarator_report],
        "recommendations": [
            "batch same-shape selected-field reductions before entering jitted kernels to avoid repeated tiny dispatches",
            "warm jitted reduced operators once before timing or summary runtime summaries",
            "keep solver-mode and geometry metadata out of static jit arguments to avoid avoidable recompilation",
        ],
    }


def save_jax_native_profile_audit_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lanes = list(report["lanes"])
    labels = ["traced-field-line\nreduced", "stellarator VMEC\nreduced"]
    compile_seconds = np.asarray([float(entry["compile_seconds"]) for entry in lanes], dtype=np.float64)
    first_execute_seconds = np.asarray([float(entry["first_execute_seconds"]) for entry in lanes], dtype=np.float64)
    warm_execute_seconds = np.asarray([float(entry["warm_execute_seconds"]) for entry in lanes], dtype=np.float64)

    x = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    axis.bar(x - width, compile_seconds, width=width, color="#bb3e03", label="compile")
    axis.bar(x, first_execute_seconds, width=width, color="#0a9396", label="first execute")
    axis.bar(x + width, warm_execute_seconds, width=width, color="#3a86ff", label="warm execute")
    axis.set_xticks(x, labels)
    style_axis(
        axis,
        title="Reduced native JAX kernel profile audit",
        ylabel="seconds",
        yscale="log",
    )
    axis.legend(frameon=False, ncol=3)
    annotate_bars(axis, x - width, compile_seconds, fmt="{:.2e}", fontsize=8.4)
    annotate_bars(axis, x, first_execute_seconds, fmt="{:.2e}", fontsize=8.4)
    annotate_bars(axis, x + width, warm_execute_seconds, fmt="{:.2e}", fontsize=8.4)
    save_publication_figure(figure, target)
    return target


def _profile_traced_field_line_kernel(trace_root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="jax_drb_jax_profile_traced_") as temp_dir:
        temp_root = Path(temp_dir)
        reference_path = temp_root / "reference.fci.nc"
        candidate_path = temp_root / "candidate.fci.nc"
        _write_metric_grid(reference_path, nx=32, ny=4, nz=3, offset=0.0)
        _write_metric_grid(candidate_path, nx=32, ny=4, nz=3, offset=0.25)
        reference_fields = _load_metric_fields(reference_path)
        candidate_fields = _load_metric_fields(candidate_path)
        field_names = ("g11", "g33")
        reference_batch = _stack_traced_metric_fields(reference_fields, field_names)
        candidate_batch = _stack_traced_metric_fields(candidate_fields, field_names)
        return _profile_batched_kernel(
            lane_name="traced_field_line_native_selected_field",
            trace_root=trace_root,
            kernel=_native_radial_profile_batch,
            reference_batch=reference_batch,
            candidate_batch=candidate_batch,
            field_names=field_names,
        )


def _profile_stellarator_vmec_kernel(trace_root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="jax_drb_jax_profile_stellarator_") as temp_dir:
        temp_root = Path(temp_dir)
        reference_path = temp_root / "reference.nc"
        candidate_path = temp_root / "candidate.nc"
        _write_vmec_case(reference_path, ns=24, scale=1.0)
        _write_vmec_case(candidate_path, ns=24, scale=1.1)
        reference_fields = _load_vmec_selected_fields(reference_path)
        candidate_fields = _load_vmec_selected_fields(candidate_path)
        field_names = ("iota", "pressure", "toroidal_flux")
        reference_batch = _stack_vmec_fields(reference_fields, field_names)
        candidate_batch = _stack_vmec_fields(candidate_fields, field_names)
        return _profile_batched_kernel(
            lane_name="stellarator_vmec_native_selected_field",
            trace_root=trace_root,
            kernel=_native_vmec_profile_batch,
            reference_batch=reference_batch,
            candidate_batch=candidate_batch,
            field_names=field_names,
        )


def _profile_batched_kernel(
    *,
    lane_name: str,
    trace_root: Path,
    kernel,
    reference_batch: jax.Array,
    candidate_batch: jax.Array,
    field_names: tuple[str, ...],
) -> dict[str, object]:
    lowered = kernel.lower(reference_batch)
    compile_start = perf_counter()
    compiled = lowered.compile()
    compile_seconds = perf_counter() - compile_start

    first_execute_seconds = _execute_compiled_pair(compiled, reference_batch, candidate_batch)
    warm_execute_seconds = _execute_compiled_pair(compiled, reference_batch, candidate_batch)
    perfetto_files = _capture_kernel_trace(
        trace_root=trace_root,
        compiled=compiled,
        reference_batch=reference_batch,
        candidate_batch=candidate_batch,
    )

    return {
        "lane_name": lane_name,
        "field_names": list(field_names),
        "compile_seconds": float(compile_seconds),
        "first_execute_seconds": float(first_execute_seconds),
        "warm_execute_seconds": float(warm_execute_seconds),
        "batched_shape": list(reference_batch.shape),
        "trace_files": perfetto_files,
    }


def _execute_compiled_pair(compiled, reference_batch: jax.Array, candidate_batch: jax.Array) -> float:
    start = perf_counter()
    jax.block_until_ready(compiled(reference_batch))
    jax.block_until_ready(compiled(candidate_batch))
    return perf_counter() - start


def _capture_kernel_trace(
    *,
    trace_root: Path,
    compiled,
    reference_batch: jax.Array,
    candidate_batch: jax.Array,
) -> list[str]:
    if trace_root.exists():
        shutil.rmtree(trace_root)
    trace_root.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_root), create_perfetto_trace=True):
        jax.block_until_ready(compiled(reference_batch))
        jax.block_until_ready(compiled(candidate_batch))
    normalized_paths: list[str] = []
    file_map = {
        ".trace.json.gz": "runtime.trace.json.gz",
        ".xplane.pb": "runtime.xplane.pb",
        "perfetto_trace.json.gz": "perfetto_trace.json.gz",
    }
    staged_root = trace_root / "normalized"
    staged_root.mkdir(parents=True, exist_ok=True)
    for source in sorted(path for path in trace_root.rglob("*") if path.is_file()):
        destination_name = None
        for suffix, normalized_name in file_map.items():
            if source.name.endswith(suffix):
                destination_name = normalized_name
                break
        if destination_name is None:
            continue
        destination = staged_root / destination_name
        shutil.copy2(source, destination)
        normalized_paths.append(str(destination.relative_to(trace_root)))
    plugins_root = trace_root / "plugins"
    if plugins_root.exists():
        shutil.rmtree(plugins_root)
    return sorted(normalized_paths)


def _stack_traced_metric_fields(fields: dict[str, np.ndarray], field_names: tuple[str, ...]) -> jax.Array:
    return jax.numpy.stack([jax.numpy.asarray(fields[field_name], dtype=jax.numpy.float64) for field_name in field_names], axis=0)


def _stack_vmec_fields(fields: dict[str, np.ndarray], field_names: tuple[str, ...]) -> jax.Array:
    return jax.numpy.stack([jax.numpy.asarray(fields[field_name], dtype=jax.numpy.float64) for field_name in field_names], axis=0)
