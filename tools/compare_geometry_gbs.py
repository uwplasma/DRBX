#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jaxdrb.driver import build_system_from_config
from jaxdrb.io import load_config


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a**2))) if a.size else 0.0


def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(_rms(b), 1e-12)
    return _rms(a - b) / denom


def _theta_grid(n: int, theta_min: float, theta_max: float) -> np.ndarray:
    return np.linspace(theta_min, theta_max, n, endpoint=False)


def _interp_periodic(
    theta_src: np.ndarray, values: np.ndarray, theta_tgt: np.ndarray
) -> np.ndarray:
    # Ensure periodic coverage by duplicating endpoints
    order = np.argsort(theta_src)
    theta_src = theta_src[order]
    values = values[order]
    theta_src = np.concatenate([theta_src, theta_src[:1] + 2 * np.pi])
    values = np.concatenate([values, values[:1]])
    theta_tgt = np.mod(theta_tgt, 2 * np.pi)
    return np.interp(theta_tgt, theta_src, values)


def _mapping_defaults(name: str) -> dict[str, object]:
    name = name.lower()
    if name in (
        "canonical",
        "canonical_salpha",
        "canonical_salpha_logb",
        "salpha_logb",
        "logb_salpha",
    ):
        return {
            "theta_range": "0,6.283185307179586",
            "swap_xy": False,
            "curv_sign_x": 1.0,
            "curv_sign_y": 1.0,
            "normalize": True,
        }
    return {}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare analytic geometry to GBS equilibrium curvature arrays"
    )
    p.add_argument("--config", required=True, help="jax_drb TOML config (analytic geometry)")
    p.add_argument("--gbs-file", required=True, help="GBS output HDF5 file (results_*.h5)")
    p.add_argument(
        "--equil-group", default="equil/00", help="HDF5 group containing curvature arrays"
    )
    p.add_argument("--curx-var", default="cur_x")
    p.add_argument("--cury-var", default="cur_y")
    p.add_argument(
        "--mapping",
        default="default",
        help="Canonical mapping preset (canonical, canonical_salpha_logb, salpha_logb)",
    )
    p.add_argument("--theta-range", default=None, help="theta range in radians (min,max)")
    p.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize by RMS before comparison",
    )
    p.add_argument(
        "--swap-xy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Swap analytic curv_x/curv_y when comparing",
    )
    p.add_argument("--curv-sign-x", type=float, default=None)
    p.add_argument("--curv-sign-y", type=float, default=None)
    args = p.parse_args()

    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("h5py is required for GBS comparison") from exc

    mapping = _mapping_defaults(str(args.mapping))

    theta_range = args.theta_range or mapping.get("theta_range", "0,6.283185307179586")
    swap_xy = (
        bool(args.swap_xy) if args.swap_xy is not None else bool(mapping.get("swap_xy", False))
    )
    curv_sign_x = (
        float(args.curv_sign_x)
        if args.curv_sign_x is not None
        else float(mapping.get("curv_sign_x", 1.0))
    )
    curv_sign_y = (
        float(args.curv_sign_y)
        if args.curv_sign_y is not None
        else float(mapping.get("curv_sign_y", 1.0))
    )
    normalize = (
        bool(args.normalize)
        if args.normalize is not None
        else bool(mapping.get("normalize", False))
    )

    with h5py.File(str(Path(args.gbs_file)), "r") as f:
        group = f[args.equil_group]
        cur_x = np.asarray(group[args.curx_var][...]).reshape(-1)
        cur_y = np.asarray(group[args.cury_var][...]).reshape(-1)

    theta_min, theta_max = [float(x) for x in theta_range.split(",")]
    theta_gbs = _theta_grid(cur_x.size, theta_min, theta_max)

    cfg = load_config(args.config)
    built = build_system_from_config(cfg.data)
    geom = built.system.geom

    curv_x = np.asarray(getattr(geom, "curv_x")).reshape(-1)
    curv_y = np.asarray(getattr(geom, "curv_y")).reshape(-1)

    # Map analytic theta from geometry
    z = np.asarray(getattr(getattr(geom, "grid", None), "z", None))
    if z is None or z.size == 0:
        theta_analytic = _theta_grid(curv_x.size, theta_min, theta_max)
    else:
        theta_scale = float(cfg.data.get("geometry", {}).get("theta_scale", 1.0))
        theta_analytic = (z / max(theta_scale, 1e-8)).reshape(-1)
        theta_analytic = np.mod(theta_analytic, 2 * np.pi)

    curv_x_interp = _interp_periodic(theta_analytic, curv_x, theta_gbs)
    curv_y_interp = _interp_periodic(theta_analytic, curv_y, theta_gbs)

    if swap_xy:
        curv_x_interp, curv_y_interp = curv_y_interp, curv_x_interp

    curv_x_interp = curv_x_interp * curv_sign_x
    curv_y_interp = curv_y_interp * curv_sign_y

    ref_x = cur_x
    ref_y = cur_y

    if normalize:
        ref_x = ref_x / max(_rms(ref_x), 1e-12)
        ref_y = ref_y / max(_rms(ref_y), 1e-12)
        curv_x_interp = curv_x_interp / max(_rms(curv_x_interp), 1e-12)
        curv_y_interp = curv_y_interp / max(_rms(curv_y_interp), 1e-12)

    report = {
        "curv_x": {
            "rms_ref": _rms(ref_x),
            "rms_geom": _rms(curv_x_interp),
            "rel_error": _rel_error(curv_x_interp, ref_x),
        },
        "curv_y": {
            "rms_ref": _rms(ref_y),
            "rms_geom": _rms(curv_y_interp),
            "rel_error": _rel_error(curv_y_interp, ref_y),
        },
    }

    for key, stats in report.items():
        print(f"{key}: {stats}")


if __name__ == "__main__":
    main()
