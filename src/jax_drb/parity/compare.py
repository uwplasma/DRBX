from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ComparisonIssue:
    field: str
    message: str


@dataclass(frozen=True)
class ComparisonResult:
    ok: bool
    issues: tuple[ComparisonIssue, ...]


def load_summary_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_summary_payloads(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    scalar_rtol: float = 1e-10,
    scalar_atol: float = 1e-12,
    time_rtol: float = 1e-12,
    time_atol: float = 1e-12,
) -> ComparisonResult:
    issues: list[ComparisonIssue] = []

    for field in ("case_name", "parity_mode", "compare_variables", "component_labels", "dimensions"):
        if _normalize_summary_field(field, actual.get(field)) != _normalize_summary_field(field, expected.get(field)):
            issues.append(ComparisonIssue(field=field, message=f"expected {expected.get(field)!r}, got {actual.get(field)!r}"))

    _compare_float_sequences(
        issues,
        field="time_points",
        expected=expected.get("time_points", []),
        actual=actual.get("time_points", []),
        rtol=time_rtol,
        atol=time_atol,
    )
    _compare_float_mapping(
        issues,
        field="dataset_scalars",
        expected=expected.get("dataset_scalars", {}),
        actual=actual.get("dataset_scalars", {}),
        rtol=scalar_rtol,
        atol=scalar_atol,
    )
    _compare_variable_summaries(
        issues,
        expected=expected.get("variable_summaries", {}),
        actual=actual.get("variable_summaries", {}),
        rtol=scalar_rtol,
        atol=scalar_atol,
    )

    return ComparisonResult(ok=not issues, issues=tuple(issues))


def _compare_float_sequences(
    issues: list[ComparisonIssue],
    *,
    field: str,
    expected: list[Any],
    actual: list[Any],
    rtol: float,
    atol: float,
) -> None:
    if len(expected) != len(actual):
        issues.append(ComparisonIssue(field=field, message=f"expected length {len(expected)}, got {len(actual)}"))
        return
    for index, (expected_value, actual_value) in enumerate(zip(expected, actual, strict=True)):
        if not math.isclose(float(actual_value), float(expected_value), rel_tol=rtol, abs_tol=atol):
            issues.append(
                ComparisonIssue(
                    field=f"{field}[{index}]",
                    message=f"expected {expected_value!r}, got {actual_value!r}",
                )
            )


def _compare_float_mapping(
    issues: list[ComparisonIssue],
    *,
    field: str,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    rtol: float,
    atol: float,
) -> None:
    for key in sorted(set(expected) | set(actual)):
        if key not in expected:
            issues.append(ComparisonIssue(field=f"{field}.{key}", message="unexpected field"))
            continue
        if key not in actual:
            issues.append(ComparisonIssue(field=f"{field}.{key}", message="missing field"))
            continue
        if not math.isclose(float(actual[key]), float(expected[key]), rel_tol=rtol, abs_tol=atol):
            issues.append(
                ComparisonIssue(
                    field=f"{field}.{key}",
                    message=f"expected {expected[key]!r}, got {actual[key]!r}",
                )
            )


def _compare_variable_summaries(
    issues: list[ComparisonIssue],
    *,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    rtol: float,
    atol: float,
) -> None:
    for name in sorted(set(expected) | set(actual)):
        if name not in expected:
            issues.append(ComparisonIssue(field=f"variable_summaries.{name}", message="unexpected variable"))
            continue
        if name not in actual:
            issues.append(ComparisonIssue(field=f"variable_summaries.{name}", message="missing variable"))
            continue

        expected_summary = expected[name]
        actual_summary = actual[name]
        for field in ("dimensions", "shape", "name"):
            if _normalize_variable_field(field, actual_summary.get(field)) != _normalize_variable_field(field, expected_summary.get(field)):
                issues.append(
                    ComparisonIssue(
                        field=f"variable_summaries.{name}.{field}",
                        message=f"expected {expected_summary.get(field)!r}, got {actual_summary.get(field)!r}",
                    )
                )

        for field in ("minimum", "maximum", "mean"):
            if not math.isclose(
                float(actual_summary[field]),
                float(expected_summary[field]),
                rel_tol=rtol,
                abs_tol=atol,
            ):
                issues.append(
                    ComparisonIssue(
                        field=f"variable_summaries.{name}.{field}",
                        message=f"expected {expected_summary[field]!r}, got {actual_summary[field]!r}",
                    )
                )

        expected_delta = expected_summary.get("max_abs_delta_last_first")
        actual_delta = actual_summary.get("max_abs_delta_last_first")
        if expected_delta is None or actual_delta is None:
            if actual_delta != expected_delta:
                issues.append(
                    ComparisonIssue(
                        field=f"variable_summaries.{name}.max_abs_delta_last_first",
                        message=f"expected {expected_delta!r}, got {actual_delta!r}",
                    )
                )
        elif not math.isclose(float(actual_delta), float(expected_delta), rel_tol=rtol, abs_tol=atol):
            issues.append(
                ComparisonIssue(
                    field=f"variable_summaries.{name}.max_abs_delta_last_first",
                    message=f"expected {expected_delta!r}, got {actual_delta!r}",
                )
            )


def _normalize_summary_field(field: str, value: Any) -> Any:
    if field in {"compare_variables", "component_labels"}:
        return tuple(value) if isinstance(value, (list, tuple)) else value
    return value


def _normalize_variable_field(field: str, value: Any) -> Any:
    if field in {"dimensions", "shape"} and isinstance(value, (list, tuple)):
        return tuple(value)
    return value
