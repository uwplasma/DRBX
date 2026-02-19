from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional
    jax = None
    jnp = None


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


# ---------------------------- IO helpers ----------------------------


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
    slc[axis] = 0
    out[tuple(slc)] = (np.take(arr, 1, axis=axis) - np.take(arr, 0, axis=axis)) / dx
    slc[axis] = -1
    out[tuple(slc)] = (np.take(arr, -1, axis=axis) - np.take(arr, -2, axis=axis)) / dx
    return out


# ---------------------------- Plot helpers ----------------------------


def _default_index(size: int) -> int:
    return max(size // 2, 0)


def slice_2d(data_zxy: np.ndarray, cut: Literal["pol", "tor", "rad"], index: int | None = None) -> np.ndarray:
    nz, nx, ny = data_zxy.shape
    if cut == "pol":
        iz = _default_index(nz) if index is None else int(index)
        return data_zxy[iz, :, :].T
    if cut == "tor":
        ix = _default_index(nx) if index is None else int(index)
        return data_zxy[:, ix, :].T
    if cut == "rad":
        iy = _default_index(ny) if index is None else int(index)
        return data_zxy[:, :, iy].T
    raise ValueError(f"Unknown cut '{cut}'")


def plot_snapshot(
    h5_path: str | Path,
    field: str,
    *,
    step: int | str | None = None,
    cut: Literal["pol", "tor", "rad"] = "pol",
    index: int | None = None,
    axes: str = "zxy",
    title: str | None = None,
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    output: str | Path | None = None,
):
    data = load_gbs_field(h5_path, field, step=step, axes=axes)
    data2d = slice_2d(data[0], cut=cut, index=index)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    im = ax.imshow(
        data2d,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title or f"{field} ({cut})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    if output is not None:
        fig.savefig(str(output), dpi=150)
    return fig


def _poloidal_coords(ny: int, nx: int, Lx: float, Ly: float, width_factor: float = 0.9, theta_window: float = 0.0):
    a = Ly / (2.0 * np.pi)
    width = width_factor * Lx
    theta = np.linspace(theta_window / 2.0, 2.0 * np.pi - theta_window / 2.0, ny)
    x = np.linspace(0.0, width, nx)
    xp = np.zeros((ny, nx))
    yp = np.zeros((ny, nx))
    for jj in range(ny):
        xp[jj, :] = (-x - a) * np.cos(theta[jj])
        yp[jj, :] = -(x + a) * np.sin(theta[jj])
    return xp, yp


def plot_poloidal(
    h5_path: str | Path,
    field: str,
    *,
    step: int | str | None = None,
    axes: str = "zxy",
    title: str | None = None,
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    output: str | Path | None = None,
    width_factor: float = 0.9,
    theta_window: float = 0.0,
):
    data = load_gbs_field(h5_path, field, step=step, axes=axes)
    data2d = slice_2d(data[0], cut="pol", index=None)

    text = read_gbs_stdin_text(h5_path) or ""
    params = parse_gbs_input_text(text)
    grid = infer_gbs_grid(params)
    Lx = grid.Lx if grid is not None else 1.0
    Ly = grid.Ly if grid is not None else 1.0

    xp, yp = _poloidal_coords(data2d.shape[0], data2d.shape[1], Lx=Lx, Ly=Ly, width_factor=width_factor, theta_window=theta_window)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    surf = ax.pcolormesh(xp, yp, data2d, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"{field} (poloidal)")
    fig.colorbar(surf, ax=ax, shrink=0.8)
    fig.tight_layout()
    if output is not None:
        fig.savefig(str(output), dpi=150)
    return fig


# ---------------------------- Spectra ----------------------------


def _fft_backend(name: str):
    if name == "jax" and jnp is not None:
        return jnp
    return np


def power_spectrum_1d(
    data: np.ndarray,
    *,
    axis: int,
    length: float,
    backend: Literal["numpy", "jax"] = "numpy",
) -> tuple[np.ndarray, np.ndarray]:
    xp = _fft_backend(backend)
    arr = xp.asarray(data)
    spec = xp.abs(xp.fft.rfft(arr, axis=axis)) ** 2
    axes = tuple(i for i in range(spec.ndim) if i != axis)
    spec = spec.mean(axis=axes)
    n = arr.shape[axis]
    k = 2.0 * np.pi * np.fft.rfftfreq(n, d=length / n)
    spec = np.asarray(spec)
    return k, spec


def plot_power_spectrum(
    h5_path: str | Path,
    field: str,
    *,
    step: int | str | None = None,
    axis: Literal["x", "y", "z"] = "y",
    axes: str = "zxy",
    backend: Literal["numpy", "jax"] = "numpy",
    output: str | Path | None = None,
):
    data = load_gbs_field(h5_path, field, step=step, axes=axes)
    snapshot = data[0]

    text = read_gbs_stdin_text(h5_path) or ""
    params = parse_gbs_input_text(text)
    grid = infer_gbs_grid(params)
    if grid is None:
        Lx = Ly = Lz = 1.0
    else:
        Lx, Ly, Lz = grid.Lx, grid.Ly, grid.Lz

    axis_map = {"z": 0, "x": 1, "y": 2}
    if axis not in axis_map:
        raise ValueError("axis must be x, y, or z")
    ax = axis_map[axis]
    length = {"x": Lx, "y": Ly, "z": Lz}[axis]

    k, spec = power_spectrum_1d(snapshot, axis=ax, length=length, backend=backend)

    fig, axp = plt.subplots(figsize=(5, 4), dpi=150)
    axp.loglog(k[1:], spec[1:] + 1e-30)
    axp.set_xlabel(f"k{axis}")
    axp.set_ylabel(f"FFT({field})")
    axp.set_title(f"Power spectrum {field} vs k{axis}")
    fig.tight_layout()
    if output is not None:
        fig.savefig(str(output), dpi=150)
    return fig


# ---------------------------- Movies ----------------------------


def _get_writer(output: Path, fps: int = 15):
    if output.suffix.lower() == ".mp4" and FFMpegWriter.isAvailable():
        return FFMpegWriter(fps=fps)
    if output.suffix.lower() not in (".gif", ".mp4"):
        output = output.with_suffix(".gif")
    return PillowWriter(fps=fps)


def make_movie_rect(
    h5_path: str | Path,
    field: str,
    *,
    steps: Iterable[int | str] | None = None,
    cut: Literal["pol", "tor", "rad"] = "pol",
    index: int | None = None,
    axes: str = "zxy",
    output: str | Path = "movie.gif",
    fps: int = 15,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "jet",
) -> Path:
    h5_path = Path(h5_path)
    output = Path(output)
    if steps is None:
        steps = list_gbs_steps(h5_path, var="theta")
    steps = list(steps)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    writer = _get_writer(output, fps=fps)

    with writer.saving(fig, str(output), dpi=150):
        for step in steps:
            data = load_gbs_field(h5_path, field, step=step, axes=axes)
            frame = slice_2d(data[0], cut=cut, index=index)
            ax.clear()
            im = ax.imshow(frame, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{field} {cut} step {step}")
            fig.colorbar(im, ax=ax, shrink=0.8)
            writer.grab_frame()
    plt.close(fig)
    return output


def make_movie_poloidal(
    h5_path: str | Path,
    field: str,
    *,
    steps: Iterable[int | str] | None = None,
    axes: str = "zxy",
    output: str | Path = "movie_poloidal.gif",
    fps: int = 15,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "jet",
    width_factor: float = 0.9,
    theta_window: float = 0.0,
) -> Path:
    h5_path = Path(h5_path)
    output = Path(output)
    if steps is None:
        steps = list_gbs_steps(h5_path, var="theta")
    steps = list(steps)

    text = read_gbs_stdin_text(h5_path) or ""
    params = parse_gbs_input_text(text)
    grid = infer_gbs_grid(params)
    Lx = grid.Lx if grid is not None else 1.0
    Ly = grid.Ly if grid is not None else 1.0

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    writer = _get_writer(output, fps=fps)

    with writer.saving(fig, str(output), dpi=150):
        for step in steps:
            data = load_gbs_field(h5_path, field, step=step, axes=axes)
            frame = slice_2d(data[0], cut="pol", index=None)
            xp, yp = _poloidal_coords(frame.shape[0], frame.shape[1], Lx=Lx, Ly=Ly, width_factor=width_factor, theta_window=theta_window)
            ax.clear()
            surf = ax.pcolormesh(xp, yp, frame, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{field} poloidal step {step}")
            fig.colorbar(surf, ax=ax, shrink=0.8)
            writer.grab_frame()
    plt.close(fig)
    return output


# ---------------------------- 0D diagnostics ----------------------------


def plot_0d_time_traces(
    h5_path: str | Path,
    *,
    fields: Iterable[str] = ("globtheta", "globtemperature", "globomega"),
    output: str | Path | None = None,
):
    import h5py  # type: ignore

    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        if "data/var0d/time" not in f:
            raise KeyError("/data/var0d/time not found")
        t = np.asarray(f["data/var0d/time"][...])
        series = {}
        for name in fields:
            key = f"data/var0d/{name}"
            if key in f:
                series[name] = np.asarray(f[key][...])

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for name, arr in series.items():
        ax.plot(t, arr, label=name)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("GBS 0D diagnostics")
    fig.tight_layout()
    if output is not None:
        fig.savefig(str(output), dpi=150)
    return fig
