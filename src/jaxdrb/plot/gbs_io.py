from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class GBSGrid:
    nx: int
    ny: int
    nz: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @property
    def Lx(self) -> float:
        return float(self.xmax - self.xmin)

    @property
    def Ly(self) -> float:
        return float(self.ymax - self.ymin)

    @property
    def Lz(self) -> float:
        return float(self.zmax - self.zmin)


GBS_LOG_FIELDS = {"theta", "temperature", "temperaturi"}
GBS_FIELD_ALIASES = {
    "ne": "theta",
    "ne_log": "theta",
    "te": "temperature",
    "ti": "temperaturi",
    "phi": "strmf",
    "vpare": "vpare",
    "vpari": "vpari",
    "omega": "omega",
    "strmf": "strmf",
    "theta": "theta",
    "temperature": "temperature",
    "temperaturi": "temperaturi",
}


def read_gbs_stdin_text(h5_path: str | Path) -> str | None:
    import h5py  # type: ignore

    path = Path(h5_path)
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        if "files" not in f:
            return None
        for key in f["files"].keys():
            if key.startswith("STDIN"):
                raw = f["files"][key][...]
                if raw.size == 0:
                    return None
                data = raw[0]
                if isinstance(data, bytes):
                    return data.decode("utf-8", errors="ignore")
                return str(data)
    return None


def parse_gbs_input_text(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not text:
        return out
    pat = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([-+0-9.eEdD]+)")
    for line in text.splitlines():
        line = line.split("!")[0].strip()
        if not line:
            continue
        for key, val in pat.findall(line):
            try:
                out[key.lower()] = float(val.replace("D", "E").replace("d", "e"))
            except ValueError:
                continue
    return out


def infer_gbs_grid(params: dict[str, float] | None) -> GBSGrid | None:
    if not params:
        return None
    nx = int(params.get("nx", 0))
    ny = int(params.get("ny", 0))
    nz = int(params.get("nz", 0))
    if nx <= 0 or ny <= 0 or nz <= 0:
        return None
    xmin = float(params.get("xmin", 0.0))
    xmax = float(params.get("xmax", float(nx)))
    ymin = float(params.get("ymin", 0.0))
    ymax = float(params.get("ymax", float(ny)))
    zmin = float(params.get("zmin", 0.0))
    zmax = float(params.get("zmax", float(nz)))
    return GBSGrid(nx=nx, ny=ny, nz=nz, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax)


def list_gbs_steps(h5_path: str | Path, var: str = "theta") -> list[str]:
    import h5py  # type: ignore

    path = Path(h5_path)
    if not path.exists():
        return []
    with h5py.File(path, "r") as f:
        grp = f.get(f"data/var3d/{var}")
        if grp is None:
            return []
        steps = sorted([k for k in grp.keys() if k.isdigit()])
    return steps


def _infer_ghosts(shape: tuple[int, int, int], grid: GBSGrid | None, axes: str) -> tuple[int, int]:
    if grid is None:
        return (0, 0)
    axes = axes.lower()
    if len(shape) != 3 or len(axes) != 3:
        return (0, 0)
    # shape order according to axes, ex: zxy
    ax_x = axes.index("x")
    ax_y = axes.index("y")
    size_x = shape[ax_x]
    size_y = shape[ax_y]
    gx = max((size_x - grid.nx) // 2, 0)
    gy = max((size_y - grid.ny) // 2, 0)
    return (gx, gy)


def _trim_ghosts(data: np.ndarray, gx: int, gy: int, axes: str) -> np.ndarray:
    if gx <= 0 and gy <= 0:
        return data
    axes = axes.lower()
    slices = [slice(None)] * data.ndim
    if gx > 0:
        ax_x = axes.index("x") + 1
        slices[ax_x] = slice(gx, -gx)
    if gy > 0:
        ax_y = axes.index("y") + 1
        slices[ax_y] = slice(gy, -gy)
    return data[tuple(slices)]


def _permute_to_zxy(data: np.ndarray, axes: str) -> np.ndarray:
    axes = axes.lower()
    if axes == "zxy":
        return data
    if sorted(axes) != ["x", "y", "z"]:
        raise ValueError("axes must be permutation of 'x', 'y', 'z'")
    perm = [axes.index("z"), axes.index("x"), axes.index("y")]
    return data.transpose(0, *(p + 1 for p in perm))


def load_gbs_var3d(
    h5_path: str | Path,
    var: str,
    *,
    step: int | str | None = None,
    steps: Sequence[int | str] | None = None,
    axes: str = "zxy",
    exp: bool | None = None,
    trim_ghosts: bool = True,
) -> np.ndarray:
    import h5py  # type: ignore

    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(path)

    text = read_gbs_stdin_text(path)
    params = parse_gbs_input_text(text or "")
    grid = infer_gbs_grid(params)

    with h5py.File(path, "r") as f:
        grp = f.get(f"data/var3d/{var}")
        if grp is None:
            raise KeyError(f"var3d/{var} not found")
        if steps is None:
            if step is None:
                steps = [sorted([k for k in grp.keys() if k.isdigit()])[-1]]
            else:
                steps = [step]
        data_list = []
        for s in steps:
            if isinstance(s, int):
                key = f"{s:06d}"
            else:
                key = str(s)
            if key not in grp:
                raise KeyError(f"step {key} not found for {var}")
            arr = np.array(grp[key][...])
            data_list.append(arr)

    data = np.stack(data_list, axis=0)

    if trim_ghosts:
        gx, gy = _infer_ghosts(data.shape[1:], grid, axes)
        data = _trim_ghosts(data, gx, gy, axes)

    data = _permute_to_zxy(data, axes)

    if exp is None:
        exp = var in GBS_LOG_FIELDS
    if exp:
        data = np.exp(data)

    return data


def load_gbs_field(
    h5_path: str | Path,
    field: str,
    *,
    step: int | str | None = None,
    steps: Sequence[int | str] | None = None,
    axes: str = "zxy",
    trim_ghosts: bool = True,
    periodic_y: bool = True,
) -> np.ndarray:
    name = field.lower()
    if name in ("ne", "te", "ti", "phi", "omega", "vpare", "vpari"):
        var = GBS_FIELD_ALIASES[name]
        exp = var in GBS_LOG_FIELDS
        return load_gbs_var3d(h5_path, var, step=step, steps=steps, axes=axes, exp=exp, trim_ghosts=trim_ghosts)

    # Derived fields
    if name in ("pe", "pi", "isat", "cur", "gamma", "ey", "vexby"):
        ne = load_gbs_field(h5_path, "ne", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts)
        if name == "pe":
            te = load_gbs_field(h5_path, "te", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts)
            return ne * te
        if name == "pi":
            ti = load_gbs_field(h5_path, "ti", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts)
            return ne * ti
        if name == "isat":
            te = load_gbs_field(h5_path, "te", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts)
            return ne * np.sqrt(np.maximum(te, 0.0))
        if name == "cur":
            vpari = load_gbs_field(h5_path, "vpari", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts)
            vpare = load_gbs_field(h5_path, "vpare", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts)
            return ne * (vpari - vpare)
        if name in ("gamma", "ey", "vexby"):
            phi = load_gbs_field(h5_path, "phi", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts)
            # derivative in y (last axis in zxy)
            dy = 1.0
            grid = infer_gbs_grid(parse_gbs_input_text(read_gbs_stdin_text(h5_path) or ""))
            if grid is not None and grid.ny > 0:
                dy = grid.Ly / grid.ny
            dphi_dy = _central_diff(phi, axis=3, dx=dy, periodic=periodic_y)
            if name == "gamma":
                return ne * dphi_dy
            return dphi_dy

    raise ValueError(f"Unknown field '{field}'")


def _central_diff(arr: np.ndarray, axis: int, dx: float, periodic: bool) -> np.ndarray:
    if periodic:
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2.0 * dx)
    slc = [slice(None)] * arr.ndim
    out = np.zeros_like(arr)
    slc[axis] = slice(1, -1)
    out[tuple(slc)] = (np.take(arr, range(2, arr.shape[axis]), axis=axis) - np.take(arr, range(0, arr.shape[axis]-2), axis=axis)) / (2.0 * dx)
    # forward/backward for edges
    slc[axis] = 0
    out[tuple(slc)] = (np.take(arr, 1, axis=axis) - np.take(arr, 0, axis=axis)) / dx
    slc[axis] = -1
    out[tuple(slc)] = (np.take(arr, -1, axis=axis) - np.take(arr, -2, axis=axis)) / dx
    return out
