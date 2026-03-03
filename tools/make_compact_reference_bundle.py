from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jaxdrb.benchmarking import BenchmarkBundle, load_bundle_npz, save_bundle_npz


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trim a full benchmark bundle into a compact reference fixture."
    )
    p.add_argument("--input", required=True, help="Input full benchmark bundle (.npz).")
    p.add_argument("--output", required=True, help="Output compact bundle (.npz).")
    p.add_argument(
        "--diagnostics",
        default="rms_n_fluct,rms_Te_fluct,rms_omega_fluct,rms_phi_fluct,freq_hz,psd_n_f,ky_m-1,psd_n_ky,pdf_n_x,pdf_n_y,coh_freq_hz,coh_n_phi,phase_n_phi,gamma_r_profile",
        help="Comma-separated diagnostic keys to keep.",
    )
    p.add_argument(
        "--snapshots",
        default="n_fluct_last,phi_fluct_last",
        help="Comma-separated snapshot keys to keep.",
    )
    p.add_argument(
        "--planes",
        default="xz,xy",
        help="Comma-separated planes to extract from 3D snapshots (choices: xz, xy).",
    )
    return p.parse_args()


def _as_plane(a: np.ndarray, plane: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Unsupported snapshot rank for plane extraction: shape={arr.shape}")
    if plane == "xy":
        return arr[:, :, arr.shape[2] // 2]
    if plane == "xz":
        return arr[:, arr.shape[1] // 2, :]
    raise ValueError(f"Unsupported plane '{plane}'")


def _compact_bundle(
    bundle: BenchmarkBundle,
    *,
    diagnostic_keys: tuple[str, ...],
    snapshot_keys: tuple[str, ...],
    planes: tuple[str, ...],
) -> BenchmarkBundle:
    axes = {
        key: np.asarray(val, dtype=np.float64)
        for key, val in bundle.axes.items()
        if key in ("x_index", "y_index", "z_index")
    }
    diagnostics = {
        key: np.asarray(bundle.diagnostics[key], dtype=np.float64)
        for key in diagnostic_keys
        if key in bundle.diagnostics
    }
    snapshots: dict[str, np.ndarray] = {}
    for key in snapshot_keys:
        if key not in bundle.snapshots:
            continue
        snap = np.asarray(bundle.snapshots[key], dtype=np.float64)
        if snap.ndim <= 2:
            snapshots[key] = snap
            continue
        for plane in planes:
            snapshots[f"{key}_{plane}"] = _as_plane(snap, plane)

    metadata = dict(bundle.metadata)
    metadata["reference_kind"] = "compact"
    metadata["source_bundle"] = f"{bundle.code}:{bundle.geometry}"
    metadata["planes"] = list(planes)

    return BenchmarkBundle(
        code=bundle.code,
        geometry=bundle.geometry,
        normalization=bundle.normalization,
        times_norm=np.asarray(bundle.times_norm, dtype=np.float64),
        times_si=np.asarray(bundle.times_si, dtype=np.float64),
        axes=axes,
        diagnostics=diagnostics,
        snapshots=snapshots,
        metadata=metadata,
    )


def main() -> None:
    args = _parse_args()
    bundle = load_bundle_npz(args.input)
    diagnostic_keys = tuple(k.strip() for k in args.diagnostics.split(",") if k.strip())
    snapshot_keys = tuple(k.strip() for k in args.snapshots.split(",") if k.strip())
    planes = tuple(k.strip() for k in args.planes.split(",") if k.strip())
    compact = _compact_bundle(
        bundle,
        diagnostic_keys=diagnostic_keys,
        snapshot_keys=snapshot_keys,
        planes=planes,
    )
    out = save_bundle_npz(compact, Path(args.output))
    print(out)


if __name__ == "__main__":
    main()
