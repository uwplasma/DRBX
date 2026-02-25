from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jaxdrb.core.geometry_field_aligned import FieldAlignedGeometryAdapter, FieldAlignedGrid
from jaxdrb.core.region_bcs import parse_region_bcs
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.geometry.plane import Grid2D


def _apply_sheath_windows(
    z: np.ndarray,
    *,
    windows: list[tuple[float, float]] | None,
    sign: list[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not windows:
        return np.zeros_like(z, dtype=float), np.zeros_like(z, dtype=float)
    mask = np.zeros_like(z, dtype=float)
    sign_arr = np.zeros_like(z, dtype=float)
    signs = sign if sign is not None else [1.0] * len(windows)
    for (zmin, zmax), sgn in zip(windows, signs):
        in_window = (z >= float(zmin)) & (z <= float(zmax))
        mask = np.where(in_window, 1.0, mask)
        sign_arr = np.where(in_window, float(sgn), sign_arr)
    return mask, sign_arr


def _load_axisymmetric_npz(path: str | Path, *, keys: dict[str, str]) -> dict[str, Any]:
    data = np.load(path)

    def get_required(key: str) -> np.ndarray:
        if key not in data:
            raise KeyError(f"Missing '{key}' in axisymmetric coefficients file.")
        return np.asarray(data[key])

    def get_optional(key: str, default: float | None = 0.0) -> np.ndarray | None:
        if key in data:
            return np.asarray(data[key])
        if default is None:
            return None
        return np.asarray(default)

    z_key = keys.get("z", "z")
    if z_key not in data:
        for alt in ("l", "theta"):
            if alt in data:
                z_key = alt
                break
    z = get_required(z_key)

    mask_fields = {
        key[len("mask_") :]: np.asarray(data[key]) for key in data.files if key.startswith("mask_")
    }

    return {
        "z": z,
        "curv_x": get_required(keys.get("curv_x", "curv_x")),
        "curv_y": get_required(keys.get("curv_y", "curv_y")),
        "dpar_factor": get_required(keys.get("dpar_factor", "dpar_factor")),
        "B": get_optional(keys.get("B", "B"), default=1.0),
        "gxx": get_optional(keys.get("gxx", "gxx"), default=None),
        "gxy": get_optional(keys.get("gxy", "gxy"), default=None),
        "gyy": get_optional(keys.get("gyy", "gyy"), default=None),
        "sheath_mask": get_optional(keys.get("sheath_mask", "sheath_mask"), default=None),
        "sheath_sign": get_optional(keys.get("sheath_sign", "sheath_sign"), default=None),
        "nx": data.get("nx"),
        "ny": data.get("ny"),
        "Lx": data.get("Lx"),
        "Ly": data.get("Ly"),
        "x": data.get("x"),
        "y": data.get("y"),
        "mask_fields": mask_fields,
    }


def _load_axisymmetric_netcdf(path: str | Path, *, keys: dict[str, str]) -> dict[str, Any]:
    try:
        import netCDF4  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Reading axisymmetric netCDF files requires netCDF4. "
            "Install it or convert the file to .npz."
        ) from exc

    ds = netCDF4.Dataset(str(path), "r")

    def get_required(key: str) -> np.ndarray:
        if key not in ds.variables:
            raise KeyError(f"Missing '{key}' in axisymmetric netCDF file.")
        return np.asarray(ds.variables[key][:])

    def get_optional(key: str, default: float | None = 0.0) -> np.ndarray | None:
        if key in ds.variables:
            return np.asarray(ds.variables[key][:])
        if default is None:
            return None
        return np.asarray(default)

    z_key = keys.get("z", "z")
    if z_key not in ds.variables:
        for alt in ("l", "theta"):
            if alt in ds.variables:
                z_key = alt
                break
    z = get_required(z_key)

    def get_scalar(key: str):
        if key in ds.variables:
            arr = np.asarray(ds.variables[key][:])
            if arr.ndim == 0:
                return float(arr)
            if arr.size == 1:
                return float(arr.ravel()[0])
            return arr
        return None

    mask_fields = {
        key[len("mask_") :]: np.asarray(ds.variables[key][:])
        for key in ds.variables
        if key.startswith("mask_")
    }

    return {
        "z": z,
        "curv_x": get_required(keys.get("curv_x", "curv_x")),
        "curv_y": get_required(keys.get("curv_y", "curv_y")),
        "dpar_factor": get_required(keys.get("dpar_factor", "dpar_factor")),
        "B": get_optional(keys.get("B", "B"), default=1.0),
        "gxx": get_optional(keys.get("gxx", "gxx"), default=None),
        "gxy": get_optional(keys.get("gxy", "gxy"), default=None),
        "gyy": get_optional(keys.get("gyy", "gyy"), default=None),
        "sheath_mask": get_optional(keys.get("sheath_mask", "sheath_mask"), default=None),
        "sheath_sign": get_optional(keys.get("sheath_sign", "sheath_sign"), default=None),
        "nx": get_scalar("nx"),
        "ny": get_scalar("ny"),
        "Lx": get_scalar("Lx"),
        "Ly": get_scalar("Ly"),
        "x": get_optional("x", default=None),
        "y": get_optional("y", default=None),
        "mask_fields": mask_fields,
    }


def load_axisymmetric_coefficients(
    path: str | Path, *, keys: dict[str, str] | None = None
) -> dict[str, Any]:
    path = Path(path)
    keys = keys or {}
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_axisymmetric_npz(path, keys=keys)
    if suffix in {".nc", ".cdf", ".netcdf"}:
        return _load_axisymmetric_netcdf(path, keys=keys)
    raise ValueError("Axisymmetric coefficients must be .npz or .nc/.cdf files.")


def build_axisymmetric_field_aligned_adapter(
    *,
    params: DRBSystemParams,
    coeff_path: str | Path,
    nx: int | None,
    ny: int | None,
    Lx: float | None,
    Ly: float | None,
    bc_x: str,
    bc_y: str,
    dealias: bool,
    open_field_line: bool,
    bc_value_x: float,
    bc_value_y: float,
    bc_grad_x: float,
    bc_grad_y: float,
    keys: dict[str, str] | None = None,
    boundary_policy: dict[str, Any] | None = None,
) -> FieldAlignedGeometryAdapter:
    coeffs = load_axisymmetric_coefficients(coeff_path, keys=keys)
    z = coeffs["z"]

    if nx is None or ny is None:
        nx = int(coeffs.get("nx") or 0)
        ny = int(coeffs.get("ny") or 0)
    if Lx is None or Ly is None:
        Lx = coeffs.get("Lx")
        Ly = coeffs.get("Ly")

    if nx <= 0 or ny <= 0 or Lx is None or Ly is None:
        raise ValueError(
            "Axisymmetric adapter requires nx, ny, Lx, Ly (either in config or in file)."
        )

    perp = Grid2D.make(
        nx=int(nx),
        ny=int(ny),
        Lx=float(Lx),
        Ly=float(Ly),
        dealias=dealias,
        bc_x=bc_x,
        bc_y=bc_y,
        bc_value_x=bc_value_x,
        bc_value_y=bc_value_y,
        bc_grad_x=bc_grad_x,
        bc_grad_y=bc_grad_y,
    )
    policy = boundary_policy or {}
    region_masks = None
    region_bcs = None
    file_masks = coeffs.get("mask_fields") or {}
    if file_masks:
        region_masks = {str(k): np.asarray(v, dtype=float) for k, v in file_masks.items()}

    def _match_mask_shape(mask: np.ndarray) -> np.ndarray:
        if mask.ndim != 2:
            return mask
        if mask.shape == (perp.nx, perp.ny):
            return mask
        if mask.shape == (perp.ny, perp.nx):
            return mask.T
        if mask.shape[0] == perp.nx:
            # Collapse along the second axis and broadcast to binormal size.
            collapsed = np.mean(mask, axis=1)
            collapsed = (collapsed > 0.5).astype(float)[:, None]
            return np.broadcast_to(collapsed, (perp.nx, perp.ny))
        return mask

    if region_masks:
        region_masks = {name: _match_mask_shape(mask) for name, mask in region_masks.items()}
    regions = policy.get("regions", None)
    if regions:
        masks = {}
        for region in regions:
            name = str(region.get("name", "")).strip()
            if not name:
                continue
            windows = None
            if "theta" in region:
                windows = [region["theta"]]
            elif "theta_window" in region:
                windows = [region["theta_window"]]
            elif "theta_windows" in region:
                windows = region["theta_windows"]
            if not windows:
                continue
            mask = np.zeros_like(z, dtype=bool)
            for theta_min, theta_max in windows:
                mask |= (z >= float(theta_min)) & (z <= float(theta_max))
            masks[name] = mask.astype(float)
        if masks:
            if region_masks is None:
                region_masks = {}
            region_masks.update(masks)
    if region_masks:
        region_bcs = parse_region_bcs(policy, region_masks)

    sheath_mask = coeffs.get("sheath_mask")
    sheath_sign = coeffs.get("sheath_sign")
    if "sheath_windows" in policy:
        windows = [(float(a), float(b)) for a, b in policy.get("sheath_windows", [])]
        signs = None
        if "sheath_sign" in policy:
            signs = [float(s) for s in policy.get("sheath_sign", [])]
        sheath_mask, sheath_sign = _apply_sheath_windows(np.asarray(z), windows=windows, sign=signs)

    grid = FieldAlignedGrid.from_z(
        perp=perp,
        z=z,
        open_field_line=open_field_line,
        sheath_mask=sheath_mask,
        sheath_sign=sheath_sign,
        region_masks=region_masks,
        region_bcs=region_bcs,
    )

    return FieldAlignedGeometryAdapter.from_coefficients(
        params=params,
        grid=grid,
        curv_x=coeffs["curv_x"],
        curv_y=coeffs["curv_y"],
        dpar_factor=coeffs["dpar_factor"],
        B=coeffs.get("B", 1.0),
        gxx=coeffs.get("gxx") if params.poisson_metric_on else None,
        gxy=coeffs.get("gxy") if params.poisson_metric_on else None,
        gyy=coeffs.get("gyy") if params.poisson_metric_on else None,
    )
