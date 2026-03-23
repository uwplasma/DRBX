from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


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


def _unravel_location(shape: tuple[int, ...], flat_index: int) -> tuple[int, ...]:
    if not shape:
        return ()
    return tuple(int(value) for value in np.unravel_index(flat_index, shape))
