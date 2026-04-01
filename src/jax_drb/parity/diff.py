from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .arrays import compare_array_payloads, load_portable_array_payload
from .compare import ComparisonResult, compare_summary_payloads, load_summary_json


@dataclass(frozen=True)
class ArrayDiffEntry:
    field: str
    shape: tuple[int, ...]
    max_abs_diff: float
    max_abs_location: tuple[int, ...]
    expected_value: float
    actual_value: float


@dataclass(frozen=True)
class ArrayDiffReport:
    compared_fields: tuple[str, ...]
    missing_expected_fields: tuple[str, ...]
    missing_actual_fields: tuple[str, ...]
    entries: tuple[ArrayDiffEntry, ...]

    @property
    def ok(self) -> bool:
        return (
            not self.missing_expected_fields
            and not self.missing_actual_fields
            and all(entry.max_abs_diff == 0.0 for entry in self.entries)
        )

    @property
    def max_abs_diff(self) -> float:
        if not self.entries:
            return 0.0
        return max(entry.max_abs_diff for entry in self.entries)


@dataclass(frozen=True)
class ScaledArrayDiffEntry:
    field: str
    shape: tuple[int, ...]
    max_abs_diff: float
    max_abs_location: tuple[int, ...]
    expected_value: float
    actual_value: float
    expected_abs_max: float
    actual_abs_max: float
    relative_to_expected_max: float | None
    near_zero_expected: bool


@dataclass(frozen=True)
class ArrayTimeTrace:
    field: str
    spatial_location: tuple[int, ...]
    expected_series: tuple[float, ...]
    actual_series: tuple[float, ...]
    abs_diff_series: tuple[float, ...]


@dataclass(frozen=True)
class RecyclingArtifactDiffReport:
    expected_path: Path
    actual_path: Path
    artifact_kind: str
    summary_result: ComparisonResult | None = None
    metadata_result: ComparisonResult | None = None
    array_report: ArrayDiffReport | None = None

    @property
    def ok(self) -> bool:
        summary_ok = True if self.summary_result is None else self.summary_result.ok
        metadata_ok = True if self.metadata_result is None else self.metadata_result.ok
        array_ok = True if self.array_report is None else self.array_report.ok
        return summary_ok and metadata_ok and array_ok

    @property
    def max_abs_diff(self) -> float:
        if self.array_report is not None:
            return self.array_report.max_abs_diff
        return 0.0

    @property
    def worst_field(self) -> str | None:
        if self.ok:
            return None
        if self.array_report is not None and self.array_report.max_abs_diff > 0.0 and self.array_report.entries:
            return self._worst_array_entry().field
        for result in (self.summary_result, self.metadata_result):
            if result is None or result.ok or not result.issues:
                continue
            return result.issues[0].field
        return None

    @property
    def worst_variable(self) -> str | None:
        if self.ok:
            return None
        if self.array_report is not None and self.array_report.max_abs_diff > 0.0 and self.array_report.entries:
            return self._worst_array_entry().field
        for result in (self.summary_result, self.metadata_result):
            if result is None or result.ok:
                continue
            for issue in result.issues:
                variable = _extract_summary_variable_name(issue.field)
                if variable is not None:
                    return variable
        return None

    @property
    def worst_location(self) -> tuple[int, ...] | None:
        if self.ok:
            return None
        if self.array_report is None or self.array_report.max_abs_diff <= 0.0 or not self.array_report.entries:
            return None
        return self._worst_array_entry().max_abs_location

    def _worst_array_entry(self) -> ArrayDiffEntry:
        assert self.array_report is not None
        return max(self.array_report.entries, key=lambda entry: entry.max_abs_diff)


def build_array_diff_report(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    compare_variables: tuple[str, ...] | None = None,
) -> ArrayDiffReport:
    if compare_variables is None:
        compared_fields = tuple(sorted(set(expected) & set(actual)))
    else:
        compared_fields = tuple(compare_variables)

    missing_expected_fields = tuple(name for name in compared_fields if name not in expected)
    missing_actual_fields = tuple(name for name in compared_fields if name not in actual)

    entries: list[ArrayDiffEntry] = []
    for name in compared_fields:
        if name not in expected or name not in actual:
            continue
        expected_array = np.asarray(expected[name], dtype=np.float64)
        actual_array = np.asarray(actual[name], dtype=np.float64)
        if expected_array.shape != actual_array.shape:
            entries.append(
                ArrayDiffEntry(
                    field=name,
                    shape=tuple(int(value) for value in actual_array.shape),
                    max_abs_diff=float("inf"),
                    max_abs_location=(),
                    expected_value=float("nan"),
                    actual_value=float("nan"),
                )
            )
            continue

        delta = np.abs(actual_array - expected_array)
        if delta.size == 0:
            location: tuple[int, ...] = ()
            max_abs_diff = 0.0
            expected_value = 0.0
            actual_value = 0.0
        else:
            flat_index = int(np.argmax(delta))
            max_abs_diff = float(np.reshape(delta, (-1,))[flat_index])
            location = _unravel_location(delta.shape, flat_index)
            expected_value = float(np.asarray(expected_array[location], dtype=np.float64))
            actual_value = float(np.asarray(actual_array[location], dtype=np.float64))
        entries.append(
            ArrayDiffEntry(
                field=name,
                shape=tuple(int(value) for value in actual_array.shape),
                max_abs_diff=max_abs_diff,
                max_abs_location=location,
                expected_value=expected_value,
                actual_value=actual_value,
            )
        )

    return ArrayDiffReport(
        compared_fields=compared_fields,
        missing_expected_fields=missing_expected_fields,
        missing_actual_fields=missing_actual_fields,
        entries=tuple(entries),
    )


def format_array_diff_report(report: ArrayDiffReport) -> str:
    lines: list[str] = []
    if report.missing_expected_fields:
        lines.append(f"missing expected fields: {', '.join(report.missing_expected_fields)}")
    if report.missing_actual_fields:
        lines.append(f"missing actual fields: {', '.join(report.missing_actual_fields)}")
    for entry in report.entries:
        location = f"@{entry.max_abs_location}" if entry.max_abs_location else ""
        lines.append(
            f"{entry.field}: max_abs_diff={entry.max_abs_diff:.8e} {location} "
            f"expected={entry.expected_value:.8e} actual={entry.actual_value:.8e}"
        )
    if not lines:
        lines.append("comparison: ok")
    return "\n".join(lines)


def build_scaled_array_diff_entries(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    compare_variables: tuple[str, ...] | None = None,
    near_zero_atol: float = 1e-12,
) -> tuple[ScaledArrayDiffEntry, ...]:
    report = build_array_diff_report(expected, actual, compare_variables=compare_variables)
    entries: list[ScaledArrayDiffEntry] = []
    for entry in report.entries:
        if entry.field not in expected or entry.field not in actual:
            continue
        expected_array = np.asarray(expected[entry.field], dtype=np.float64)
        actual_array = np.asarray(actual[entry.field], dtype=np.float64)
        expected_abs_max = float(np.max(np.abs(expected_array))) if expected_array.size else 0.0
        actual_abs_max = float(np.max(np.abs(actual_array))) if actual_array.size else 0.0
        near_zero_expected = expected_abs_max <= near_zero_atol
        relative_to_expected_max = None
        if not near_zero_expected:
            relative_to_expected_max = float(entry.max_abs_diff / expected_abs_max)
        entries.append(
            ScaledArrayDiffEntry(
                field=entry.field,
                shape=entry.shape,
                max_abs_diff=entry.max_abs_diff,
                max_abs_location=entry.max_abs_location,
                expected_value=entry.expected_value,
                actual_value=entry.actual_value,
                expected_abs_max=expected_abs_max,
                actual_abs_max=actual_abs_max,
                relative_to_expected_max=relative_to_expected_max,
                near_zero_expected=near_zero_expected,
            )
        )
    return tuple(entries)


def filter_scaled_array_diff_entries_to_band(
    entries: tuple[ScaledArrayDiffEntry, ...] | list[ScaledArrayDiffEntry],
    *,
    axis: int,
) -> tuple[ScaledArrayDiffEntry, ...]:
    filtered: list[ScaledArrayDiffEntry] = []
    for entry in entries:
        if axis < 0 or axis >= len(entry.shape) or axis >= len(entry.max_abs_location):
            continue
        index = entry.max_abs_location[axis]
        if index == 0 or index == entry.shape[axis] - 1:
            filtered.append(entry)
    return tuple(filtered)


def build_array_time_trace(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    field: str,
    spatial_location: tuple[int, ...],
    time_axis: int = 0,
) -> ArrayTimeTrace:
    expected_array = np.asarray(expected[field], dtype=np.float64)
    actual_array = np.asarray(actual[field], dtype=np.float64)
    if expected_array.shape != actual_array.shape:
        raise ValueError(f"shape mismatch for field {field}: {expected_array.shape} != {actual_array.shape}")
    if expected_array.ndim == 0:
        expected_series = (float(expected_array),)
        actual_series = (float(actual_array),)
        abs_diff_series = (abs(expected_series[0] - actual_series[0]),)
        return ArrayTimeTrace(
            field=field,
            spatial_location=(),
            expected_series=expected_series,
            actual_series=actual_series,
            abs_diff_series=abs_diff_series,
        )
    if time_axis < 0 or time_axis >= expected_array.ndim:
        raise ValueError(f"time_axis {time_axis} is out of bounds for field {field} with ndim {expected_array.ndim}")
    if expected_array.ndim - 1 != len(spatial_location):
        raise ValueError(
            f"spatial_location {spatial_location} does not match field {field} shape {expected_array.shape} "
            f"with time_axis {time_axis}"
        )
    slicer = [slice(None)] * expected_array.ndim
    spatial_index = 0
    for axis in range(expected_array.ndim):
        if axis == time_axis:
            continue
        index = spatial_location[spatial_index]
        if index < 0 or index >= expected_array.shape[axis]:
            raise ValueError(
                f"spatial index {index} out of bounds for axis {axis} in field {field} shape {expected_array.shape}"
            )
        slicer[axis] = index
        spatial_index += 1
    expected_series_array = np.asarray(expected_array[tuple(slicer)], dtype=np.float64)
    actual_series_array = np.asarray(actual_array[tuple(slicer)], dtype=np.float64)
    abs_diff_series_array = np.abs(actual_series_array - expected_series_array)
    return ArrayTimeTrace(
        field=field,
        spatial_location=spatial_location,
        expected_series=tuple(float(value) for value in expected_series_array.tolist()),
        actual_series=tuple(float(value) for value in actual_series_array.tolist()),
        abs_diff_series=tuple(float(value) for value in abs_diff_series_array.tolist()),
    )


def compare_recycling_artifacts(
    expected_artifact: str | Path,
    actual_artifact: str | Path,
    *,
    artifact_kind: str = "auto",
    scalar_rtol: float = 1e-10,
    scalar_atol: float = 1e-12,
    array_rtol: float = 1e-10,
    array_atol: float = 1e-12,
) -> RecyclingArtifactDiffReport:
    expected_path = Path(expected_artifact)
    actual_path = Path(actual_artifact)
    resolved_kind = _resolve_recycling_artifact_kind(expected_path, actual_path, artifact_kind)

    if resolved_kind == "summary":
        expected = load_summary_json(expected_path)
        actual = load_summary_json(actual_path)
        return RecyclingArtifactDiffReport(
            expected_path=expected_path,
            actual_path=actual_path,
            artifact_kind=resolved_kind,
            summary_result=compare_summary_payloads(
                expected,
                actual,
                scalar_rtol=scalar_rtol,
                scalar_atol=scalar_atol,
            ),
        )

    expected = load_portable_array_payload(expected_path)
    actual = load_portable_array_payload(actual_path)
    metadata_result = compare_array_payloads(
        expected,
        actual,
        scalar_rtol=scalar_rtol,
        scalar_atol=scalar_atol,
        array_rtol=array_rtol,
        array_atol=array_atol,
    )
    compare_variables = tuple(expected.get("compare_variables", [])) or None
    array_report = build_array_diff_report(
        expected.get("variables", {}),
        actual.get("variables", {}),
        compare_variables=compare_variables,
    )
    return RecyclingArtifactDiffReport(
        expected_path=expected_path,
        actual_path=actual_path,
        artifact_kind=resolved_kind,
        metadata_result=metadata_result,
        array_report=array_report,
    )


def format_recycling_diff_report(report: RecyclingArtifactDiffReport) -> str:
    lines = [
        f"artifact_kind: {report.artifact_kind}",
        f"expected: {report.expected_path}",
        f"actual: {report.actual_path}",
    ]
    if report.worst_field is not None:
        lines.append(f"worst_field: {report.worst_field}")
    if report.worst_variable is not None:
        lines.append(f"worst_variable: {report.worst_variable}")
    if report.worst_location is not None:
        lines.append(f"worst_location: {report.worst_location}")
    if report.summary_result is not None:
        lines.append(f"summary: {'ok' if report.summary_result.ok else 'mismatch'}")
        for issue in report.summary_result.issues:
            lines.append(f"  {issue.field}: {issue.message}")
    if report.metadata_result is not None:
        lines.append(f"metadata: {'ok' if report.metadata_result.ok else 'mismatch'}")
        for issue in report.metadata_result.issues:
            lines.append(f"  {issue.field}: {issue.message}")
    if report.array_report is not None:
        lines.append("arrays:")
        if report.array_report.ok:
            lines.append("  comparison: ok")
        else:
            formatted = format_array_diff_report(report.array_report)
            lines.extend(f"  {line}" for line in formatted.splitlines())
        if report.array_report.entries and not report.array_report.ok:
            worst = report._worst_array_entry()
            lines.append(
                f"worst: {worst.field} @ {worst.max_abs_location} diff={worst.max_abs_diff:.8e}"
            )
    return "\n".join(lines)


def _unravel_location(shape: tuple[int, ...], flat_index: int) -> tuple[int, ...]:
    if not shape:
        return ()
    return tuple(int(value) for value in np.unravel_index(flat_index, shape))


def _resolve_recycling_artifact_kind(
    expected_path: Path,
    actual_path: Path,
    artifact_kind: str,
) -> str:
    if artifact_kind != "auto":
        if artifact_kind not in {"summary", "arrays"}:
            raise ValueError(f"unsupported artifact_kind: {artifact_kind!r}")
        return artifact_kind
    suffixes = {expected_path.suffix.lower(), actual_path.suffix.lower()}
    if suffixes == {".json"}:
        return "summary"
    if suffixes == {".npz"}:
        return "arrays"
    raise ValueError(
        "cannot infer recycling artifact kind from mismatched suffixes "
        f"{expected_path.suffix!r} and {actual_path.suffix!r}"
    )


def _extract_summary_variable_name(field: str) -> str | None:
    prefix = "variable_summaries."
    if not field.startswith(prefix):
        return None
    remainder = field.removeprefix(prefix)
    return remainder.split(".", 1)[0] if remainder else None
