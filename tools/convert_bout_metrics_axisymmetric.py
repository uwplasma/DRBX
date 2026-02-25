#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

E_CHARGE = 1.602176634e-19
M_PROTON = 1.67262192369e-27


def _require(module: str):
    try:
        return __import__(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency '{module}'. Install it to read BOUT++ grids."
        ) from exc


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
    return np.linspace(
        r0 - 0.5 * dr - (mxg - 0.5) * h,
        r0 + 0.5 * dr + (mxg - 0.5) * h,
        nx,
    )


def _scalar(ds, name: str, default: float) -> float:
    if name in ds.variables:
        arr = np.asarray(ds.variables[name][:])
        if arr.size == 1:
            return float(arr.ravel()[0])
        if arr.ndim == 0:
            return float(arr)
    return float(default)


def _compute_rho_s0(
    *,
    Te0_eV: float,
    B0_T: float,
    m_i_amu: float,
    Z_i: float,
) -> float:
    m_i = max(float(m_i_amu), 1e-30) * M_PROTON
    cs = np.sqrt(max(float(Te0_eV), 1e-30) * E_CHARGE / m_i)
    omega_ci = max(float(Z_i), 1e-30) * E_CHARGE * max(float(B0_T), 1e-30) / m_i
    return float(cs / max(omega_ci, 1e-30))


def _build_region_masks(ds, *, nx: int, ny: int) -> dict[str, np.ndarray]:
    mask_fields: dict[str, np.ndarray] = {}
    if "ixseps1" in ds.variables:
        ix1 = int(np.asarray(ds.variables["ixseps1"][:]).ravel()[0])
        ix2 = ix1
        if "ixseps2" in ds.variables:
            ix2 = int(np.asarray(ds.variables["ixseps2"][:]).ravel()[0])
        ixsep = int(round(0.5 * (ix1 + ix2)))
        ixsep = max(min(ixsep, nx - 1), 0)
        x_index = np.arange(nx, dtype=int)[:, None]
        mask_open = (x_index >= ixsep).astype(float)
        mask_closed = 1.0 - mask_open
        mask_fields["open"] = np.broadcast_to(mask_open, (nx, ny))
        mask_fields["closed"] = np.broadcast_to(mask_closed, (nx, ny))
        mask_fields["core"] = np.broadcast_to(mask_closed, (nx, ny))
        mask_fields["sol"] = np.broadcast_to(mask_open, (nx, ny))
    return mask_fields


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert BOUT++ metric grid to axisymmetric coefficients (.npz)"
    )
    p.add_argument("--grid", required=True, type=str, help="Path to BOUT++ grid (.nc)")
    p.add_argument("--out", required=True, type=str, help="Output .npz path")
    p.add_argument(
        "--x-index",
        type=int,
        default=0,
        help="Radial index to extract (set to -1 to keep full 2D fields).",
    )
    p.add_argument(
        "--zeta", type=float, default=0.0, help="Toroidal angle for logB Fourier reconstruction"
    )
    p.add_argument(
        "--radial-coordinate",
        choices=("flux", "physical"),
        default="physical",
        help="Use flux coordinate or physical minor radius for x-derivative",
    )
    p.add_argument(
        "--radial-from",
        choices=("auto", "dr", "dx_btor", "dx"),
        default="auto",
        help="How to build physical radial coordinate when radial-coordinate=physical",
    )
    p.add_argument(
        "--curv-x-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Which axis derivative defines curv_x",
    )
    p.add_argument(
        "--curv-y-axis",
        choices=("x", "y", "z"),
        default="x",
        help="Which axis derivative defines curv_y",
    )
    p.add_argument("--curv-sign-x", type=float, default=1.0, help="Optional sign flip for curv_x")
    p.add_argument("--curv-sign-y", type=float, default=1.0, help="Optional sign flip for curv_y")
    p.add_argument("--use-metric", action="store_true", help="Use gxx/gxy/gyy for scaling")
    p.add_argument("--gxx-var", default="gxx_ballooning")
    p.add_argument("--gxy-var", default="gxy_ballooning")
    p.add_argument("--gyy-var", default="gyy_ballooning")
    p.add_argument("--logb-var", default="logB")
    p.add_argument("--bxcv-var-x", default="bxcvx", help="BOUT++ bxcv x-component")
    p.add_argument("--bxcv-var-y", default="bxcvy", help="BOUT++ bxcv y-component")
    p.add_argument("--bxcv-var-z", default="bxcvz", help="BOUT++ bxcv z-component")
    p.add_argument(
        "--bxcv-curv-x",
        choices=("x", "y", "z"),
        default="x",
        help="Which bxcv component maps to curv_x",
    )
    p.add_argument(
        "--bxcv-curv-y",
        choices=("x", "y", "z"),
        default="y",
        help="Which bxcv component maps to curv_y",
    )
    p.add_argument(
        "--bxcv-use-2overB",
        action="store_true",
        help="Scale bxcv by 2/Bxy to match common drift-reduced curvature conventions",
    )
    p.add_argument(
        "--bxcv-normalization",
        choices=("none", "hermes"),
        default="none",
        help=(
            "Apply a normalization convention before mapping bxcv components. "
            "'hermes' uses x/Bnorm, y*Lnorm^2, z*Lnorm^2, then 2/B."
        ),
    )
    p.add_argument(
        "--norm-Te0-eV",
        type=float,
        default=50.0,
        help="Reference Te [eV] used to compute rho_s0 when --bxcv-normalization=hermes.",
    )
    p.add_argument(
        "--norm-B0-T",
        type=float,
        default=1.0,
        help="Reference B [T] used for Hermes-style bxcv normalization.",
    )
    p.add_argument(
        "--norm-mi-amu",
        type=float,
        default=2.0,
        help="Ion mass [amu] used to compute rho_s0 for Hermes-style bxcv normalization.",
    )
    p.add_argument(
        "--norm-Zi",
        type=float,
        default=1.0,
        help="Ion charge state used to compute rho_s0 for Hermes-style bxcv normalization.",
    )
    p.add_argument(
        "--norm-rho-s0",
        type=float,
        default=None,
        help=(
            "Explicit rho_s0 [m] for Hermes-style bxcv normalization. "
            "If omitted, computed from Te0/B0/mi/Zi."
        ),
    )
    p.add_argument(
        "--bxcv-shifted-sinty",
        action="store_true",
        help=(
            "Apply shifted-metric correction bxcv_z <- bxcv_z + sinty*bxcv_x before normalization."
        ),
    )
    p.add_argument(
        "--parallel-from-y",
        action="store_true",
        help=(
            "Treat BOUT++ y (poloidal) as the parallel coordinate: transpose"
            " curvature/dpar/B/metric arrays to (ny, nx) so they vary along z."
        ),
    )
    p.add_argument(
        "--bxcv-scale",
        type=float,
        default=1.0,
        help="Additional multiplicative scale for bxcv curvature coefficients",
    )
    p.add_argument("--hthe-var", default="hthe")
    p.add_argument("--bxy-var", default="Bxy")
    p.add_argument("--bpxy-var", default="Bpxy")
    p.add_argument("--dx-var", default="dx")
    p.add_argument("--dy-var", default="dy")
    p.add_argument("--mxg-var", default="mxg")
    p.add_argument("--theta-var", default="theta_ballooning")
    args = p.parse_args()

    netcdf4 = _require("netCDF4")
    ds = netcdf4.Dataset(str(Path(args.grid)), "r")

    x_index = int(args.x_index)
    if x_index < 0:
        x_index = None

    gxx_var = args.gxx_var
    gxy_var = args.gxy_var
    gyy_var = args.gyy_var
    if gxx_var not in ds.variables and "g11" in ds.variables:
        gxx_var = "g11"
        gxy_var = "g12" if "g12" in ds.variables else gxy_var
        gyy_var = "g22" if "g22" in ds.variables else gyy_var

    use_logb = args.logb_var in ds.variables
    use_bxcv = (
        args.bxcv_var_x in ds.variables
        and args.bxcv_var_y in ds.variables
        and args.bxcv_var_z in ds.variables
    )
    if not use_logb and not use_bxcv:
        raise ValueError("Missing logB and bxcv variables in BOUT grid.")

    logB_raw = None
    logB_slice = None
    logB_z = None
    dlogB_dz = None
    if use_logb:
        logB_raw = np.asarray(ds.variables[args.logb_var][:])
        logB_slice = _slice_var(logB_raw, x_index)
        logB_z, dlogB_dz = _reconstruct_logB(logB_slice, args.zeta)

    dx = np.asarray(ds.variables[args.dx_var][:])
    dy = np.asarray(ds.variables[args.dy_var][:]) if args.dy_var in ds.variables else None

    Bxy_full = np.asarray(ds.variables[args.bxy_var][:])
    Bpxy_full = np.asarray(ds.variables[args.bpxy_var][:])
    Rxy_full = np.asarray(ds.variables["Rxy"][:]) if "Rxy" in ds.variables else None
    Zxy_full = np.asarray(ds.variables["Zxy"][:]) if "Zxy" in ds.variables else None

    # Build x-coordinate for derivative.
    if args.radial_coordinate == "flux":
        x_coord = _coord_from_spacing(dx, axis=0)
    else:
        mode = args.radial_from
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

    if use_logb:
        logB_xy = logB_raw
        if logB_xy.ndim >= 3 and logB_xy.shape[-1] >= 3:
            logB_xy = (
                logB_xy[..., 0]
                + logB_xy[..., 1] * np.cos(args.zeta)
                + logB_xy[..., 2] * np.sin(args.zeta)
            )

        dlogB_dx_full = np.gradient(logB_xy, x_coord, axis=0, edge_order=2)
        dlogB_dx = _slice_var(dlogB_dx_full, x_index)

        if dy is not None:
            y_coord = _coord_from_spacing(dy, axis=1)
            dlogB_dy_full = np.gradient(logB_xy, y_coord, axis=1, edge_order=2)
            dlogB_dy = _slice_var(dlogB_dy_full, x_index)
        else:
            y_coord = np.arange(logB_z.shape[-1])
            dlogB_dy = np.zeros_like(dlogB_dx)
    else:
        if dy is not None:
            y_coord = _coord_from_spacing(dy, axis=1)
        else:
            y_coord = None

    Bxy = _slice_var(Bxy_full, x_index)
    Bpxy = _slice_var(Bpxy_full, x_index)
    hthe = _slice_var(np.asarray(ds.variables[args.hthe_var][:]), x_index)
    Rxy = None
    Zxy = None
    if Rxy_full is not None:
        Rxy = np.asarray(Rxy_full)
    if Zxy_full is not None:
        Zxy = np.asarray(Zxy_full)

    scale_x = 1.0
    scale_y = 1.0
    gxx = None
    gxy = None
    gyy = None
    if (gxx_var in ds.variables) and (gyy_var in ds.variables):
        gxx = _slice_var(np.asarray(ds.variables[gxx_var][:]), x_index)
        gyy = _slice_var(np.asarray(ds.variables[gyy_var][:]), x_index)
        if gxy_var in ds.variables:
            gxy = _slice_var(np.asarray(ds.variables[gxy_var][:]), x_index)
        else:
            gxy = np.zeros_like(gxx)
        if args.use_metric:
            gxx_safe = np.maximum(gxx, 1e-12)
            gperp = np.maximum(gyy - (gxy**2) / gxx_safe, 1e-12)
            scale_x = np.sqrt(gxx_safe)
            scale_y = np.sqrt(gperp)

    if use_logb:
        derivs = {"x": dlogB_dx, "y": dlogB_dy, "z": dlogB_dz}
        curv_x = -Bxy * scale_y * derivs[args.curv_x_axis] * float(args.curv_sign_x)
        curv_y = Bxy * scale_x * derivs[args.curv_y_axis] * float(args.curv_sign_y)
    else:
        bxcv_x = _slice_var(np.asarray(ds.variables[args.bxcv_var_x][:]), x_index)
        bxcv_y = _slice_var(np.asarray(ds.variables[args.bxcv_var_y][:]), x_index)
        bxcv_z = _slice_var(np.asarray(ds.variables[args.bxcv_var_z][:]), x_index)
        if args.bxcv_shifted_sinty and "sinty" in ds.variables:
            sinty = _slice_var(np.asarray(ds.variables["sinty"][:]), x_index)
            bxcv_z = bxcv_z + sinty * bxcv_x

        if args.bxcv_normalization == "hermes":
            rho_s0 = (
                float(args.norm_rho_s0)
                if args.norm_rho_s0 is not None
                else _compute_rho_s0(
                    Te0_eV=float(args.norm_Te0_eV),
                    B0_T=float(args.norm_B0_T),
                    m_i_amu=float(args.norm_mi_amu),
                    Z_i=float(args.norm_Zi),
                )
            )
            Bnorm = max(float(args.norm_B0_T), 1e-30)
            bxcv_x = bxcv_x / Bnorm
            bxcv_y = bxcv_y * (rho_s0**2)
            bxcv_z = bxcv_z * (rho_s0**2)
            bxcv_x = 2.0 * bxcv_x / np.maximum(Bxy, 1e-12)
            bxcv_y = 2.0 * bxcv_y / np.maximum(Bxy, 1e-12)
            bxcv_z = 2.0 * bxcv_z / np.maximum(Bxy, 1e-12)

        comp = {"x": bxcv_x, "y": bxcv_y, "z": bxcv_z}
        curv_x = comp[args.bxcv_curv_x]
        curv_y = comp[args.bxcv_curv_y]
        if args.bxcv_use_2overB and args.bxcv_normalization != "hermes":
            curv_x = 2.0 * curv_x / np.maximum(Bxy, 1e-12)
            curv_y = 2.0 * curv_y / np.maximum(Bxy, 1e-12)
        curv_x = curv_x * float(args.bxcv_scale) * float(args.curv_sign_x)
        curv_y = curv_y * float(args.bxcv_scale) * float(args.curv_sign_y)
    dpar_factor = Bpxy / np.maximum(Bxy * hthe, 1e-12)

    def _maybe_transpose(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        if args.parallel_from_y and arr.ndim >= 2:
            return np.swapaxes(arr, 0, 1)
        return arr

    curv_x = _maybe_transpose(curv_x)
    curv_y = _maybe_transpose(curv_y)
    dpar_factor = _maybe_transpose(dpar_factor)
    Bxy = _maybe_transpose(Bxy)
    gxx = _maybe_transpose(gxx)
    gxy = _maybe_transpose(gxy)
    gyy = _maybe_transpose(gyy)

    # z (field-aligned coordinate)
    if args.theta_var in ds.variables:
        theta_full = np.asarray(ds.variables[args.theta_var][:])
        if theta_full.ndim >= 2:
            if x_index is None:
                mid = int(theta_full.shape[0] // 2)
                z = np.asarray(theta_full[mid, :])
            else:
                z = np.asarray(theta_full[x_index, :])
        else:
            z = np.asarray(theta_full).reshape(-1)
    else:
        if y_coord is None:
            y_coord = np.arange(curv_x.shape[-1])
        z = np.asarray(y_coord).reshape(-1)

    # Lengths
    Lx = float(x_coord[-1] - x_coord[0] + (x_coord[1] - x_coord[0]) if x_coord.size > 1 else 1.0)
    Ly = float(y_coord[-1] - y_coord[0] + (y_coord[1] - y_coord[0]) if y_coord.size > 1 else 1.0)
    if curv_x.ndim >= 2:
        nx_guess, ny_guess = curv_x.shape[:2]
    else:
        nx_guess, ny_guess = curv_x.shape[-1], curv_x.shape[-1]
    nx = int(_scalar(ds, "nx", nx_guess))
    ny = int(_scalar(ds, "ny", ny_guess))

    mask_fields = _build_region_masks(ds, nx=nx, ny=ny)

    out = {
        "z": np.asarray(z).reshape(-1),
        "curv_x": np.asarray(curv_x),
        "curv_y": np.asarray(curv_y),
        "dpar_factor": np.asarray(dpar_factor),
        "B": np.asarray(Bxy),
        "Lx": Lx,
        "Ly": Ly,
        "nx": nx,
        "ny": ny,
    }
    if Rxy is not None:
        out["Rxy"] = np.asarray(Rxy)
    if Zxy is not None:
        out["Zxy"] = np.asarray(Zxy)

    if gxx is not None:
        out["gxx"] = np.asarray(gxx)
    if gxy is not None:
        out["gxy"] = np.asarray(gxy)
    if gyy is not None:
        out["gyy"] = np.asarray(gyy)
    for name, mask in mask_fields.items():
        out[f"mask_{name}"] = np.asarray(mask)

    np.savez(Path(args.out), **out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
