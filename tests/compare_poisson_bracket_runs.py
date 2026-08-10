#!/usr/bin/env python3
"""Compare direct and compatible-flux Poisson-bracket run artifacts.

This is intentionally a small, dependency-light prototype diagnostic.  It
does not import the simulation or operator stack, so it can also be used when
one run stopped before writing its final output.  Snapshot arrays are assumed
to be ordered ``(u, theta, eta)`` as in the toroidal run artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_PROGRESS_RE = re.compile(
    r"\b(?P<step>\d+)\s*/\s*(?P<total>\d+)\s+t\s*=\s*"
    r"(?P<time>[+\-0-9.eE]+)\s+step\s*=\s*(?P<seconds>[+\-0-9.eE]+)s"
)
_COMPILE_RE = re.compile(
    r"compiled\s+sharded\s+(?P<kind>.+?)\s+in\s+(?P<seconds>[0-9.eE+\-]+)\s*s",
    re.IGNORECASE,
)
_FAILURE_STEP_RE = re.compile(
    r"\[diagnostics\]\s+step=(?P<step>\d+)\s+positivity\s+failure",
    re.IGNORECASE,
)
_DT_RE = re.compile(r"\bdt=(?P<dt>[+\-0-9.eE]+)")
_SNAPSHOT_TIME_RE = re.compile(
    r"snapshot_t(?P<int>\d+)d(?P<frac>\d+)e(?P<sign>[mp])(?P<exp>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LogSummary:
    path: str
    latest_step: int | None
    latest_time: float | None
    failure_step: int | None
    failure_time: float | None
    total_steps: int | None
    compile_seconds: float | None
    compile_events: tuple[dict[str, Any], ...]
    runtime_step_seconds: dict[str, float | int | None]
    step_seconds_by_step: dict[str, float]
    status: str
    failure_reason: str | None


@dataclass(frozen=True)
class SnapshotRecord:
    path: str
    requested_time: float | None
    actual_time: float | None
    step: int | None


def _as_scalar(value: Any) -> Any:
    """Convert a zero-dimensional NumPy value to a Python value."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _metadata_from_npz(data: Mapping[str, Any]) -> dict[str, Any]:
    raw = data.get("run_metadata_json")
    if raw is None:
        return {}
    raw = _as_scalar(raw)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_log_text(text: str, path: str = "<memory>") -> LogSummary:
    """Parse progress, compile timing, and failure information from a log."""
    progress: list[tuple[int, int, float, float]] = []
    for match in _PROGRESS_RE.finditer(text):
        try:
            progress.append(
                (
                    int(match.group("step")),
                    int(match.group("total")),
                    float(match.group("time")),
                    float(match.group("seconds")),
                )
            )
        except ValueError:
            continue

    compile_events = tuple(
        {
            "kind": match.group("kind").strip(),
            "seconds": float(match.group("seconds")),
        }
        for match in _COMPILE_RE.finditer(text)
    )
    compile_seconds = (
        float(sum(event["seconds"] for event in compile_events))
        if compile_events
        else None
    )

    failure_reason = find_failure_reason(text)
    failure_matches = list(_FAILURE_STEP_RE.finditer(text))
    failure_step = (
        int(failure_matches[-1].group("step")) if failure_matches else None
    )
    dt_matches = list(_DT_RE.finditer(text))
    dt = float(dt_matches[-1].group("dt")) if dt_matches else None
    failure_time = (
        float(failure_step * dt)
        if failure_step is not None and dt is not None
        else None
    )
    if failure_reason is not None:
        status = "failed"
    elif progress and progress[-1][0] >= progress[-1][1]:
        status = "completed"
    elif progress:
        status = "incomplete"
    else:
        status = "no-progress"

    latest = progress[-1] if progress else None
    step_times = np.asarray([item[3] for item in progress], dtype=float)
    runtime = {
        "samples": int(step_times.size),
        "mean_step_seconds": float(step_times.mean()) if step_times.size else None,
        "median_step_seconds": float(np.median(step_times)) if step_times.size else None,
        "p90_step_seconds": float(np.percentile(step_times, 90.0)) if step_times.size else None,
        "min_step_seconds": float(step_times.min()) if step_times.size else None,
        "max_step_seconds": float(step_times.max()) if step_times.size else None,
        "estimated_advance_seconds": float(step_times.sum()) if step_times.size else None,
    }
    return LogSummary(
        path=str(path),
        latest_step=latest[0] if latest else None,
        latest_time=latest[2] if latest else None,
        failure_step=failure_step,
        failure_time=failure_time,
        total_steps=latest[1] if latest else None,
        compile_seconds=compile_seconds,
        compile_events=compile_events,
        runtime_step_seconds=runtime,
        step_seconds_by_step={str(item[0]): item[3] for item in progress},
        status=status,
        failure_reason=failure_reason,
    )


def find_failure_reason(text: str) -> str | None:
    """Return a concise final failure/traceback reason, if one is present."""
    lines = text.splitlines()
    traceback_start = next(
        (index for index, line in enumerate(lines) if line.strip() == "Traceback (most recent call last):"),
        None,
    )
    if traceback_start is not None:
        tail = [line.strip() for line in lines[traceback_start + 1 :] if line.strip()]
        for line in reversed(tail):
            if re.match(r"^[A-Za-z_][\w.]*Error\s*:", line) or re.match(
                r"^[A-Za-z_][\w.]*Exception\s*:", line
            ):
                return line
        if tail:
            return tail[-1]

    failure_patterns = (
        r"\b(?:fatal\s+)?error\b",
        r"\bfailed\b",
        r"\bfailure\b",
        r"\bexception\b",
        r"\bnon[- ]finite\b",
        r"\bblow[- ]?up\b",
    )
    for line in reversed(lines):
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in failure_patterns):
            return line.strip() or None
    return None


def parse_snapshot_time_from_name(path: str | Path) -> float | None:
    """Parse the driver's ``t5d000...em03`` requested-time filename token."""
    match = _SNAPSHOT_TIME_RE.search(Path(path).name)
    if match is None:
        return None
    sign = "-" if match.group("sign").lower() == "m" else "+"
    try:
        return float(f"{match.group('int')}.{match.group('frac')}e{sign}{match.group('exp')}")
    except ValueError:
        return None


def snapshot_record(path: str | Path) -> SnapshotRecord:
    """Read only scalar metadata needed to pair one snapshot."""
    requested = parse_snapshot_time_from_name(path)
    actual = None
    step = None
    try:
        with np.load(path, allow_pickle=False) as data:
            if "requested_snapshot_time" in data:
                requested = float(_as_scalar(data["requested_snapshot_time"]))
            if "time" in data:
                actual = float(_as_scalar(data["time"]))
            if "step" in data:
                step = int(_as_scalar(data["step"]))
    except (OSError, ValueError, TypeError):
        pass
    return SnapshotRecord(str(path), requested, actual, step)


def list_snapshots(directory: str | Path) -> list[SnapshotRecord]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    records = [snapshot_record(path) for path in sorted(directory.glob("*.npz"))]
    return sorted(records, key=lambda item: (float("inf") if item.requested_time is None else item.requested_time, item.path))


def pair_snapshots(
    direct: Sequence[SnapshotRecord], compatible: Sequence[SnapshotRecord],
    tolerance: float = 1.0e-10,
) -> list[tuple[SnapshotRecord, SnapshotRecord]]:
    """Pair snapshots by requested time, falling back to actual time."""
    def key(record: SnapshotRecord) -> float | None:
        return record.requested_time if record.requested_time is not None else record.actual_time

    compatible_by_time = [(record, key(record)) for record in compatible]
    pairs: list[tuple[SnapshotRecord, SnapshotRecord]] = []
    used: set[int] = set()
    for left in direct:
        left_key = key(left)
        if left_key is None:
            continue
        candidates = [
            (abs(left_key - right_key), index, right)
            for index, (right, right_key) in enumerate(compatible_by_time)
            if index not in used and right_key is not None and abs(left_key - right_key) <= tolerance
        ]
        if candidates:
            _, index, right = min(candidates, key=lambda item: item[0])
            used.add(index)
            pairs.append((left, right))
    return pairs


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Unweighted relative L2 norm, with a useful absolute fallback at zero."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    difference_norm = float(np.linalg.norm(candidate - reference))
    reference_norm = float(np.linalg.norm(reference))
    return difference_norm / reference_norm if reference_norm > 0.0 else difference_norm


def weighted_relative_l2(
    reference: np.ndarray, candidate: np.ndarray, weights: np.ndarray | None,
) -> float | None:
    if weights is None:
        return None
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    weights = np.broadcast_to(np.asarray(weights, dtype=float), reference.shape)
    difference_norm = float(np.sqrt(np.sum(weights * (candidate - reference) ** 2)))
    reference_norm = float(np.sqrt(np.sum(weights * reference**2)))
    return difference_norm / reference_norm if reference_norm > 0.0 else difference_norm


def volume_integral(field: np.ndarray, weights: np.ndarray | None) -> float | None:
    if weights is None:
        return None
    return float(np.sum(np.asarray(field, dtype=float) * np.asarray(weights, dtype=float)))


def relative_integral_drift(reference: np.ndarray, candidate: np.ndarray, weights: np.ndarray | None) -> float | None:
    left = volume_integral(reference, weights)
    right = volume_integral(candidate, weights)
    if left is None or right is None:
        return None
    scale = abs(left)
    return float((right - left) / scale) if scale > 0.0 else float(right - left)


def _relative_scalar_drift(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None:
        return None
    scale = abs(reference)
    return float((candidate - reference) / scale) if scale > 0.0 else float(candidate - reference)


def near_axis_fft_proxy(
    field: np.ndarray, first_rings: int = 3, high_mode_cutoff: int = 4,
) -> dict[str, Any]:
    """Summarize angular Fourier energy on the first radial rings.

    The proxy is intentionally geometry-agnostic: it measures high angular
    content in the sampled ``(theta, eta)`` plane and is not a claim that the
    field itself satisfies a particular polar regularity rule.
    """
    array = np.asarray(field, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"expected a (u, theta, eta) field, got shape {array.shape}")
    ring_count = min(max(int(first_rings), 1), array.shape[0])
    spectrum = np.fft.fftn(array[:ring_count], axes=(1, 2), norm="ortho")
    energy = np.abs(spectrum) ** 2
    theta_modes = np.fft.fftfreq(array.shape[1]) * array.shape[1]
    eta_modes = np.fft.fftfreq(array.shape[2]) * array.shape[2]
    mode_radius = np.maximum(np.abs(theta_modes)[:, None], np.abs(eta_modes)[None, :])
    high = mode_radius >= max(int(high_mode_cutoff), 1)
    total_energy = float(np.sum(energy))
    high_energy = float(np.sum(energy[:, high]))
    return {
        "rings": ring_count,
        "theta_modes": int(array.shape[1]),
        "eta_modes": int(array.shape[2]),
        "high_mode_cutoff": int(high_mode_cutoff),
        "total_energy": total_energy,
        "high_mode_energy": high_energy,
        "high_mode_fraction": high_energy / total_energy if total_energy > 0.0 else 0.0,
    }


def _field_names(data: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    names = metadata.get("field_names")
    if isinstance(names, list):
        return [name for name in names if isinstance(name, str) and name in data]
    return [
        key for key, value in data.items()
        if isinstance(key, str) and isinstance(value, np.ndarray) and value.ndim == 3
        and key not in {"jacobian", "metric_J"}
    ]


def _weights(data: Mapping[str, Any]) -> np.ndarray | None:
    for key in ("jacobian", "metric_J", "volume_weights", "weights"):
        if key in data:
            array = np.asarray(data[key], dtype=float)
            if array.ndim == 3 and np.all(np.isfinite(array)) and np.all(array >= 0.0):
                return array
    return None


def compare_snapshot_pair(
    direct_record: SnapshotRecord,
    compatible_record: SnapshotRecord,
    first_rings: int = 3,
    high_mode_cutoff: int = 4,
) -> dict[str, Any]:
    with np.load(direct_record.path, allow_pickle=False) as direct_data, np.load(
        compatible_record.path, allow_pickle=False
    ) as compatible_data:
        direct_meta = _metadata_from_npz(direct_data)
        compatible_meta = _metadata_from_npz(compatible_data)
        direct_fields = set(_field_names(direct_data, direct_meta))
        compatible_fields = set(_field_names(compatible_data, compatible_meta))
        fields = sorted(direct_fields & compatible_fields)
        direct_weights = _weights(direct_data)
        compatible_weights = _weights(compatible_data)
        weights = direct_weights if direct_weights is not None else compatible_weights
        if direct_weights is not None and compatible_weights is not None:
            if direct_weights.shape != compatible_weights.shape:
                weights = None

        field_results: dict[str, Any] = {}
        for name in fields:
            reference = np.asarray(direct_data[name], dtype=float)
            candidate = np.asarray(compatible_data[name], dtype=float)
            if reference.shape != candidate.shape or reference.ndim != 3:
                continue
            field_results[name] = {
                "shape": list(reference.shape),
                "direct_min": float(np.nanmin(reference)),
                "direct_max": float(np.nanmax(reference)),
                "compatible_min": float(np.nanmin(candidate)),
                "compatible_max": float(np.nanmax(candidate)),
                "unweighted_relative_l2": relative_l2(reference, candidate),
                "volume_weighted_relative_l2": weighted_relative_l2(reference, candidate, weights),
                "direct_integral": volume_integral(reference, weights),
                "compatible_integral": volume_integral(candidate, weights),
                # Cross-scheme difference at this checkpoint.  Temporal
                # drift from each scheme's first checkpoint is added by
                # ``build_report`` once all pairs are available.
                "relative_integral_drift": relative_integral_drift(reference, candidate, weights),
                "near_axis_direct": near_axis_fft_proxy(reference, first_rings, high_mode_cutoff),
                "near_axis_compatible": near_axis_fft_proxy(candidate, first_rings, high_mode_cutoff),
            }

    requested = direct_record.requested_time
    return {
        "requested_time": requested,
        "direct_time": direct_record.actual_time,
        "compatible_time": compatible_record.actual_time,
        "direct_step": direct_record.step,
        "compatible_step": compatible_record.step,
        "direct_snapshot": direct_record.path,
        "compatible_snapshot": compatible_record.path,
        "weights": "jacobian/volume" if weights is not None else None,
        "fields": field_results,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    def read_log(path: str | None) -> LogSummary | None:
        if path is None:
            return None
        log_path = Path(path)
        if not log_path.is_file():
            return LogSummary(
                str(log_path), None, None, None, None, None, None, (), {}, {},
                "missing", "log file not found",
            )
        return parse_log_text(log_path.read_text(encoding="utf-8", errors="replace"), str(log_path))

    direct_log = read_log(args.direct_log)
    compatible_log = read_log(args.compatible_log)
    direct_snapshots = list_snapshots(args.direct_snapshot_dir) if args.direct_snapshot_dir else []
    compatible_snapshots = list_snapshots(args.compatible_snapshot_dir) if args.compatible_snapshot_dir else []
    pairs = pair_snapshots(direct_snapshots, compatible_snapshots, args.time_tolerance)
    checkpoints = [
        compare_snapshot_pair(left, right, args.first_rings, args.high_mode_cutoff)
        for left, right in pairs
    ]
    if checkpoints:
        first_fields = checkpoints[0]["fields"]
        for checkpoint in checkpoints:
            for name, result in checkpoint["fields"].items():
                first = first_fields.get(name, {})
                result["direct_integral_drift_from_first"] = _relative_scalar_drift(
                    first.get("direct_integral"), result.get("direct_integral")
                )
                result["compatible_integral_drift_from_first"] = _relative_scalar_drift(
                    first.get("compatible_integral"), result.get("compatible_integral")
                )
    matched_runtime = None
    if direct_log is not None and compatible_log is not None:
        direct_times = direct_log.step_seconds_by_step
        compatible_times = compatible_log.step_seconds_by_step
        common_steps = sorted(set(direct_times) & set(compatible_times), key=int)
        if common_steps:
            direct_common = np.asarray([direct_times[step] for step in common_steps])
            compatible_common = np.asarray([compatible_times[step] for step in common_steps])
            ratios = compatible_common / np.maximum(direct_common, 1.0e-300)
            matched_runtime = {
                "sample_count": len(common_steps),
                "last_common_step": int(common_steps[-1]),
                "direct_mean_step_seconds": float(np.mean(direct_common)),
                "compatible_mean_step_seconds": float(np.mean(compatible_common)),
                "mean_time_overhead_fraction": float(
                    np.mean(compatible_common) / np.mean(direct_common) - 1.0
                ),
                "median_paired_time_ratio": float(np.median(ratios)),
            }
    return {
        "comparison": "poisson-bracket direct vs compatible-flux prototype",
        "direct_log": asdict(direct_log) if direct_log else None,
        "compatible_log": asdict(compatible_log) if compatible_log else None,
        "direct_snapshot_count": len(direct_snapshots),
        "compatible_snapshot_count": len(compatible_snapshots),
        "common_checkpoint_count": len(checkpoints),
        "matched_runtime_step_seconds": matched_runtime,
        "unpaired_direct_checkpoints": [record.requested_time for record in direct_snapshots if record not in {p[0] for p in pairs}],
        "unpaired_compatible_checkpoints": [record.requested_time for record in compatible_snapshots if record not in {p[1] for p in pairs}],
        "checkpoints": checkpoints,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}e}"
    return str(value)


def format_report(report: Mapping[str, Any]) -> str:
    lines = [str(report["comparison"])]
    for label in ("direct_log", "compatible_log"):
        summary = report.get(label)
        if summary is None:
            lines.append(f"{label.removesuffix('_log')}: not supplied")
        else:
            lines.append(
                f"{label.removesuffix('_log')}: status={summary['status']} "
                f"latest_step={summary['latest_step']}/{summary['total_steps']} "
                f"time={_fmt(summary['latest_time'])} compile={_fmt(summary['compile_seconds'])} "
                f"median_step={_fmt(summary['runtime_step_seconds'].get('median_step_seconds'))}"
            )
            if summary.get("failure_step") is not None:
                lines.append(
                    f"  failed_attempt=step {summary['failure_step']} "
                    f"time={_fmt(summary.get('failure_time'))}"
                )
            if summary.get("failure_reason"):
                lines.append(f"  failure: {summary['failure_reason']}")
    lines.append(
        f"snapshots: direct={report['direct_snapshot_count']} "
        f"compatible={report['compatible_snapshot_count']} "
        f"common={report['common_checkpoint_count']}"
    )
    timing = report.get("matched_runtime_step_seconds")
    if timing:
        lines.append(
            "matched runtime: "
            f"steps=1..{timing['last_common_step']} "
            f"mean=({_fmt(timing['direct_mean_step_seconds'])},"
            f"{_fmt(timing['compatible_mean_step_seconds'])}) "
            f"overhead={_fmt(timing['mean_time_overhead_fraction'])}"
        )
    for checkpoint in report["checkpoints"]:
        fields = checkpoint["fields"]
        worst = max(
            ((result["unweighted_relative_l2"], name) for name, result in fields.items()),
            default=(0.0, "n/a"),
        )
        axis = max(
            (
                abs(result["near_axis_compatible"]["high_mode_fraction"] - result["near_axis_direct"]["high_mode_fraction"]),
                name,
            )
            for name, result in fields.items()
        ) if fields else (0.0, "n/a")
        lines.append(
            f"  t_req={_fmt(checkpoint['requested_time'])} "
            f"t=({_fmt(checkpoint['direct_time'])},{_fmt(checkpoint['compatible_time'])}) "
            f"worst_rel_l2={_fmt(worst[0])}({worst[1]}) "
            f"max_axis_fraction_delta={_fmt(axis[0])}({axis[1]})"
        )
    if not report["checkpoints"]:
        lines.append("  no common checkpoints available yet")
    return "\n".join(lines)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-log")
    parser.add_argument("--compatible-log")
    parser.add_argument("--direct-snapshot-dir")
    parser.add_argument("--compatible-snapshot-dir")
    parser.add_argument("--output-json", help="write the complete report to this path")
    parser.add_argument("--time-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--first-rings", type=int, default=3)
    parser.add_argument("--high-mode-cutoff", type=int, default=4)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    report = build_report(args)
    print(format_report(report))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
