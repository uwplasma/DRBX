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


def _slice_var(arr: np.ndarray, x_index: int | None) -> np.ndarray:
    if arr.ndim >= 2 and x_index is not None:
        return np.asarray(arr[x_index, ...])
    return np.asarray(arr)


def _reconstruct_logB(logB: np.ndarray, zeta: float) -> tuple[np.ndarray, np.ndarray]:
    if logB.ndim == 1:
        return logB, np.zeros_like(logB)
    if logB.ndim == 2 and logB.shape[-1] >= 3:
        logB0 = logB[..., 0]
        logB1 = logB[..., 1]
        logB2 = logB[..., 2]
        logB_z = logB0 + logB1 * np.cos(zeta) + logB2 * np.sin(zeta)
        dlogB_dz = -logB1 * np.sin(zeta) + logB2 * np.cos(zeta)
        return logB_z, dlogB_dz
    if logB.ndim == 2:
        return logB, np.zeros_like(logB)
    raise ValueError(f"Unsupported logB shape {logB.shape}")


def _coord_from_spacing(spacing: np.ndarray, axis: int) -> np.ndarray:
    if spacing.ndim == 2:
        if axis == 0:
            spacing_mean = np.mean(spacing, axis=1)
        else:
            spacing_mean = np.mean(spacing, axis=0)
    else:
        spacing_mean = np.asarray(spacing).reshape(-1)
    coord = np.zeros_like(spacing_mean)
    if spacing_mean.size > 1:
        coord[1:] = np.cumsum(spacing_mean[:-1])
    return coord


def _radial_from_dr(r0: float, dr: float, nx: int, mxg: int) -> np.ndarray:
    h = dr / max(nx - 2 * mxg, 1)
    rminor = np.linspace(
        r0 - 0.5 * dr - (mxg - 0.5) * h,
        r0 + 0.5 * dr + (mxg - 0.5) * h,
        nx,
    )
    return rminor


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
            "radial_coordinate": "physical",
            "radial_from": "dr",
            "curv_x_axis": "z",
            "curv_y_axis": "x",
            "curv_sign_x": 1.0,
            "curv_sign_y": 1.0,
            "use_metric": True,
        }
    return {}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare analytic geometry to BOUT++ metric-derived coefficients"
    )
    p.add_argument("--config", required=True, help="jax_drb TOML config (analytic geometry)")
    p.add_argument("--bout-grid", required=True, help="BOUT++ grid file (.nc)")
    p.add_argument(
        "--mapping",
        default="default",
        help="Canonical mapping preset (canonical, canonical_salpha_logb, salpha_logb)",
    )
    p.add_argument("--x-index", type=int, default=0, help="Radial index to extract")
    p.add_argument(
        "--zeta", type=float, default=0.0, help="Toroidal angle for logB Fourier reconstruction"
    )
    p.add_argument("--use-metric", action="store_true", help="Use gxx/gxy/gyy for scaling")
    p.add_argument(
        "--radial-coordinate",
        choices=("flux", "physical"),
        default=None,
        help="Use BOUT flux coordinate or physical minor radius for x-derivative",
    )
    p.add_argument(
        "--radial-from",
        choices=("auto", "dr", "dx_btor", "dx"),
        default=None,
        help="How to build physical radial coordinate when radial-coordinate=physical",
    )
    p.add_argument(
        "--curv-x-axis",
        choices=("x", "y", "z"),
        default=None,
        help="Which BOUT axis derivative defines curv_x",
    )
    p.add_argument(
        "--curv-y-axis",
        choices=("x", "y", "z"),
        default=None,
        help="Which BOUT axis derivative defines curv_y",
    )
    p.add_argument("--curv-sign-x", type=float, default=None, help="Optional sign flip for curv_x")
    p.add_argument("--curv-sign-y", type=float, default=None, help="Optional sign flip for curv_y")
    p.add_argument("--gxx-var", default="gxx_ballooning")
    p.add_argument("--gxy-var", default="gxy_ballooning")
    p.add_argument("--gyy-var", default="gyy_ballooning")
    p.add_argument("--logb-var", default="logB")
    p.add_argument("--hthe-var", default="hthe")
    p.add_argument("--bxy-var", default="Bxy")
    p.add_argument("--bpxy-var", default="Bpxy")
    p.add_argument("--dx-var", default="dx")
    p.add_argument("--dy-var", default="dy")
    p.add_argument("--mxg-var", default="mxg")
    args = p.parse_args()

    mapping = _mapping_defaults(str(args.mapping))

    radial_coordinate = args.radial_coordinate or mapping.get("radial_coordinate", "physical")
    radial_from = args.radial_from or mapping.get("radial_from", "auto")
    curv_x_axis = args.curv_x_axis or mapping.get("curv_x_axis", "z")
    curv_y_axis = args.curv_y_axis or mapping.get("curv_y_axis", "x")
    curv_sign_x = (
        args.curv_sign_x if args.curv_sign_x is not None else mapping.get("curv_sign_x", 1.0)
    )
    curv_sign_y = (
        args.curv_sign_y if args.curv_sign_y is not None else mapping.get("curv_sign_y", 1.0)
    )
    use_metric = args.use_metric or bool(mapping.get("use_metric", False))

    try:
        import netCDF4  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("netCDF4 is required for BOUT++ grid comparison") from exc

    with netCDF4.Dataset(str(Path(args.bout_grid)), "r") as ds:
        if args.logb_var not in ds.variables:
            raise ValueError("logB variable missing in BOUT grid")
        logB_raw = np.asarray(ds.variables[args.logb_var][:])
        logB_slice = _slice_var(logB_raw, args.x_index)
        logB_z, dlogB_dz = _reconstruct_logB(logB_slice, args.zeta)

        dx = np.asarray(ds.variables[args.dx_var][:])
        dy = np.asarray(ds.variables[args.dy_var][:]) if args.dy_var in ds.variables else None

        Bxy_full = np.asarray(ds.variables[args.bxy_var][:])
        Bpxy_full = np.asarray(ds.variables[args.bpxy_var][:])
        Rxy_full = np.asarray(ds.variables["Rxy"][:]) if "Rxy" in ds.variables else None

        # Build x-coordinate for derivative.
        if radial_coordinate == "flux":
            x_coord = _coord_from_spacing(dx, axis=0)
        else:
            mode = radial_from
            if mode == "auto":
                if "dr" in ds.variables and args.mxg_var in ds.variables and "r0" in ds.variables:
                    mode = "dr"
                elif Rxy_full is not None:
                    mode = "dx_btor"
                else:
                    mode = "dx"
            if mode == "dr":
                r0 = float(np.asarray(ds.variables["r0"][:]).ravel()[0])
                dr = float(np.asarray(ds.variables["dr"][:]).ravel()[0])
                mxg = int(np.asarray(ds.variables[args.mxg_var][:]).ravel()[0])
                nx = int(np.asarray(ds.variables["nx"][:]).ravel()[0])
                x_coord = _radial_from_dr(r0, dr, nx, mxg)
            elif mode == "dx_btor" and Rxy_full is not None:
                dr_est = dx / np.maximum(Rxy_full * Bpxy_full, 1e-12)
                x_coord = _coord_from_spacing(dr_est, axis=0)
            else:
                x_coord = _coord_from_spacing(dx, axis=0)

        # Reconstruct logB in real space (x,y at zeta)
        logB_xy = logB_raw
        if logB_xy.ndim >= 3 and logB_xy.shape[-1] >= 3:
            logB_xy = (
                logB_xy[..., 0]
                + logB_xy[..., 1] * np.cos(args.zeta)
                + logB_xy[..., 2] * np.sin(args.zeta)
            )

        dlogB_dx_full = np.gradient(logB_xy, x_coord, axis=0, edge_order=2)
        dlogB_dx = _slice_var(dlogB_dx_full, args.x_index)

        if dy is not None:
            y_coord = _coord_from_spacing(dy, axis=1)
            dlogB_dy_full = np.gradient(logB_xy, y_coord, axis=1, edge_order=2)
            dlogB_dy = _slice_var(dlogB_dy_full, args.x_index)
        else:
            dlogB_dy = np.zeros_like(dlogB_dx)

        Bxy = _slice_var(Bxy_full, args.x_index)
        Bpxy = _slice_var(Bpxy_full, args.x_index)
        hthe = _slice_var(np.asarray(ds.variables[args.hthe_var][:]), args.x_index)

        scale_x = 1.0
        scale_y = 1.0
        if use_metric:
            if args.gxx_var in ds.variables and args.gyy_var in ds.variables:
                gxx = _slice_var(np.asarray(ds.variables[args.gxx_var][:]), args.x_index)
                gyy = _slice_var(np.asarray(ds.variables[args.gyy_var][:]), args.x_index)
                if args.gxy_var in ds.variables:
                    gxy = _slice_var(np.asarray(ds.variables[args.gxy_var][:]), args.x_index)
                else:
                    gxy = np.zeros_like(gxx)
                gxx = np.maximum(gxx, 1e-12)
                gperp = np.maximum(gyy - (gxy**2) / gxx, 1e-12)
                scale_x = np.sqrt(gxx)
                scale_y = np.sqrt(gperp)

        derivs = {
            "x": dlogB_dx,
            "y": dlogB_dy,
            "z": dlogB_dz,
        }

        curv_x = -Bxy * scale_y * derivs[curv_x_axis]
        curv_y = Bxy * scale_x * derivs[curv_y_axis]
        dpar_factor = Bpxy / np.maximum(Bxy * hthe, 1e-12)

    cfg = load_config(args.config)
    built = build_system_from_config(cfg.data)
    geom = built.system.geom

    def grab(name: str) -> np.ndarray:
        val = getattr(geom, name, None)
        if val is None:
            raise ValueError(f"Geometry has no attribute '{name}'.")
        return np.asarray(val)

    ref = {
        "curv_x": curv_x * float(curv_sign_x),
        "curv_y": curv_y * float(curv_sign_y),
        "dpar_factor": dpar_factor,
        "B": Bxy,
    }

    report = {}
    for key in ("curv_x", "curv_y", "dpar_factor", "B"):
        ref_arr = np.asarray(ref[key]).reshape(-1)
        geom_arr = np.asarray(grab(key)).reshape(-1)
        n = min(ref_arr.size, geom_arr.size)
        if n == 0:
            report[key] = {"error": None}
            continue
        report[key] = {
            "rms_ref": _rms(ref_arr[:n]),
            "rms_geom": _rms(geom_arr[:n]),
            "rel_error": _rel_error(geom_arr[:n], ref_arr[:n]),
        }

    for key, stats in report.items():
        print(f"{key}: {stats}")


if __name__ == "__main__":
    main()
