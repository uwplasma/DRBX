from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np


@dataclass(frozen=True)
class SliceSpec:
    name: str
    axis: int
    coordinate_name: str
    coordinate_values: np.ndarray


def build_slice_report(
    *,
    field_name: str,
    values: np.ndarray,
    spec: SliceSpec,
) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array for slice diagnostics, got shape {array.shape}")
    frames = []
    reference_plane = _extract_plane(array, spec.axis, 0)
    for index, coord in enumerate(np.asarray(spec.coordinate_values, dtype=np.float64)):
        plane = _extract_plane(array, spec.axis, index)
        delta = plane - reference_plane
        frames.append(
            {
                "index": int(index),
                "coordinate_value": float(coord),
                "minimum": float(np.min(plane)),
                "maximum": float(np.max(plane)),
                "mean": float(np.mean(plane)),
                "standard_deviation": float(np.std(plane)),
                "rms_delta_from_first": float(np.sqrt(np.mean(delta**2))),
            }
        )
    return {
        "available": True,
        "parse_status": "ok",
        "field_name": field_name,
        "slice_name": spec.name,
        "coordinate_name": spec.coordinate_name,
        "coordinate_values": np.asarray(spec.coordinate_values, dtype=np.float64).tolist(),
        "frames": frames,
    }


def write_slice_arrays_npz(
    *,
    field_name: str,
    values: np.ndarray,
    spec: SliceSpec,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float64)
    payload = {
        "field_name": np.asarray(field_name),
        "coordinate_name": np.asarray(spec.coordinate_name),
        "coordinate_values": np.asarray(spec.coordinate_values, dtype=np.float64),
        "values": array,
    }
    np.savez_compressed(target, **payload)
    return target


def save_slice_summary_plot(
    report: dict[str, object],
    path: str | Path,
    *,
    title: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = report.get("frames", [])
    coords = np.asarray([frame["coordinate_value"] for frame in frames], dtype=np.float64)
    mins = np.asarray([frame["minimum"] for frame in frames], dtype=np.float64)
    maxs = np.asarray([frame["maximum"] for frame in frames], dtype=np.float64)
    means = np.asarray([frame["mean"] for frame in frames], dtype=np.float64)
    stds = np.asarray([frame.get("standard_deviation", 0.0) for frame in frames], dtype=np.float64)
    rms_deltas = np.asarray([frame.get("rms_delta_from_first", 0.0) for frame in frames], dtype=np.float64)
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), constrained_layout=True, sharex=True)
    axes[0].plot(coords, means, color="#005f73", linewidth=2.0, label="mean")
    axes[0].fill_between(coords, mins, maxs, color="#94d2bd", alpha=0.35, label="min/max band")
    axes[0].set_ylabel(_display_label(str(report.get("field_name", "field"))))
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
    axes[1].plot(coords, stds, color="#0a9396", linewidth=1.9, label="plane std")
    axes[1].plot(coords, rms_deltas, color="#ae2012", linewidth=1.9, label="RMS delta from first")
    axes[1].set_xlabel(str(report.get("coordinate_name", "coord")))
    axes[1].set_ylabel("diagnostic amplitude")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def save_slice_gif(
    *,
    field_name: str,
    values: np.ndarray,
    spec: SliceSpec,
    path: str | Path,
    fps: int = 8,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float64)
    frames = [_extract_plane(array, spec.axis, index) for index in range(array.shape[spec.axis])]
    vmin = float(np.min(array))
    vmax = float(np.max(array))
    figure, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    image = axis.imshow(frames[0], origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    title = axis.set_title("")
    axis.set_xlabel("poloidal index")
    axis.set_ylabel("radial index")
    figure.colorbar(image, ax=axis, shrink=0.88, pad=0.02, label=_display_label(field_name))

    def _update(frame_index: int):
        image.set_data(frames[frame_index])
        coord = float(np.asarray(spec.coordinate_values, dtype=np.float64)[frame_index])
        title.set_text(f"{_display_label(field_name)} · {_display_label(spec.name)} · {spec.coordinate_name}={coord:.3f}")
        return (image, title)

    anim = animation.FuncAnimation(figure, _update, frames=len(frames), interval=max(1, int(1000 / fps)), blit=False)
    writer = animation.PillowWriter(fps=fps)
    anim.save(target, writer=writer)
    plt.close(figure)
    return target


def write_slice_report_json(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _extract_plane(values: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return np.asarray(values[index, :, :], dtype=np.float64)
    if axis == 1:
        return np.asarray(values[:, index, :], dtype=np.float64)
    if axis == 2:
        return np.asarray(values[:, :, index], dtype=np.float64)
    raise ValueError(f"Unsupported slice axis {axis}")


def _display_label(name: str) -> str:
    labels = {
        "Bmag": "|B|",
        "J": "Jacobian",
        "jacobian": "Jacobian",
        "g11": "g11",
        "g22": "g22",
        "g33": "g33",
        "g_11": "g^11",
        "g_22": "g^22",
        "g_33": "g^33",
        "radial_index_planes": "Radial slices",
        "toroidal_index_planes": "Toroidal slices",
        "poloidal_index_planes": "Poloidal slices",
    }
    return labels.get(name, name.replace("_", " "))
