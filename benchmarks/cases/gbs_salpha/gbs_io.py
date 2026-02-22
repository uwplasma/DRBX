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
    return GBSGrid(
        nx=nx, ny=ny, nz=nz, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax
    )


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
        return load_gbs_var3d(
            h5_path, var, step=step, steps=steps, axes=axes, exp=exp, trim_ghosts=trim_ghosts
        )

    if name in ("pe", "pi", "isat", "cur", "gamma", "ey", "vexby"):
        ne = load_gbs_field(
            h5_path, "ne", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts
        )
        if name == "pe":
            te = load_gbs_field(
                h5_path, "te", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts
            )
            return ne * te
        if name == "pi":
            ti = load_gbs_field(
                h5_path, "ti", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts
            )
            return ne * ti
        if name == "isat":
            te = load_gbs_field(
                h5_path, "te", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts
            )
            return ne * np.sqrt(np.maximum(te, 0.0))
        if name == "cur":
            vpari = load_gbs_field(
                h5_path, "vpari", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts
            )
            vpare = load_gbs_field(
                h5_path, "vpare", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts
            )
            return ne * (vpari - vpare)
        if name in ("gamma", "ey", "vexby"):
            phi = load_gbs_field(
                h5_path, "phi", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts
            )
            dy = 1.0
            grid = infer_gbs_grid(parse_gbs_input_text(read_gbs_stdin_text(h5_path) or ""))
            if grid is not None and grid.ny > 0:
                dy = grid.Ly / grid.ny
            dphi_dy = _central_diff(phi, axis=3, dx=dy, periodic=periodic_y)
            if name == "gamma":
                return ne * dphi_dy
            return dphi_dy

    if name in ("vf", "gammaexb"):
        if name == "vf":
            phi = load_gbs_field(
                h5_path, "phi", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts
            )
            te = load_gbs_field(
                h5_path, "te", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts
            )
            return phi - 3.0 * te
        if name == "gammaexb":
            return load_gbs_field(
                h5_path, "gamma", step=step, steps=steps, axes=axes, trim_ghosts=trim_ghosts
            )

    raise ValueError(f"Unknown field '{field}'")


def _central_diff(arr: np.ndarray, axis: int, dx: float, periodic: bool) -> np.ndarray:
    if periodic:
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2.0 * dx)
    slc = [slice(None)] * arr.ndim
    out = np.zeros_like(arr)
    slc[axis] = slice(1, -1)
    out[tuple(slc)] = (
        np.take(arr, range(2, arr.shape[axis]), axis=axis)
        - np.take(arr, range(0, arr.shape[axis] - 2), axis=axis)
    ) / (2.0 * dx)
    slc[axis] = 0
    out[tuple(slc)] = (np.take(arr, 1, axis=axis) - np.take(arr, 0, axis=axis)) / dx
    slc[axis] = -1
    out[tuple(slc)] = (np.take(arr, -1, axis=axis) - np.take(arr, -2, axis=axis)) / dx
    return out


# ---------------------------- Plot helpers ----------------------------


def _default_index(size: int) -> int:
    return max(size // 2 - 1, 0)


def slice_2d(
    data_zxy: np.ndarray, cut: Literal["pol", "tor", "rad"], index: int | None = None
) -> np.ndarray:
    nz, nx, ny = data_zxy.shape
    if cut == "pol":
        iz = 0 if index is None else int(index)
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
    frame = data2d[:-1, :-1] if data2d.shape[0] > 1 and data2d.shape[1] > 1 else data2d
    im = ax.pcolormesh(
        frame,
        shading="gouraud",
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


def _poloidal_coords(
    ny: int, nx: int, Lx: float, Ly: float, width_factor: float = 0.9, theta_window: float = 0.0
):
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
    nz = int(data.shape[1]) if data.shape[1] > 0 else 1
    iz_mid = max(nz // 2 - 1, 0)
    data2d = slice_2d(data[0], cut="pol", index=iz_mid)

    text = read_gbs_stdin_text(h5_path) or ""
    params = parse_gbs_input_text(text)
    grid = infer_gbs_grid(params)
    Lx = grid.Lx if grid is not None else 1.0
    Ly = grid.Ly if grid is not None else 1.0

    width = width_factor * Lx
    ixs = int(np.floor(data2d.shape[1] * (Lx - width) / Lx))
    if ixs < 0:
        ixs = 0
    data2d = data2d[:, ixs:]
    xp, yp = _poloidal_coords(
        data2d.shape[0],
        data2d.shape[1],
        Lx=Lx,
        Ly=Ly,
        width_factor=width_factor,
        theta_window=theta_window,
    )

    fig = plt.figure(figsize=(6, 6), dpi=150, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        xp, yp, data2d, cmap=cmap, linewidth=0, antialiased=False, vmin=vmin, vmax=vmax
    )
    ax.view_init(elev=30.0, azim=-37.5)
    ax.set_title(title or f"{field} (poloidal)")
    ax.set_axis_off()
    ax.set_xlim((-(Ly / (2.0 * np.pi) + width) * 1.05, (Ly / (2.0 * np.pi) + width) * 1.05))
    ax.set_ylim((-(Ly / (2.0 * np.pi) + width) * 1.05, (Ly / (2.0 * np.pi) + width) * 1.05))
    ax.set_zlim((np.nanmin(data2d), np.nanmax(data2d)))
    ax.set_box_aspect([1, 1, 0.3])
    fig.subplots_adjust(0, 0, 1, 1)
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
    double_sided: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    xp = _fft_backend(backend)
    arr = xp.asarray(data)
    n = arr.shape[axis]
    nuq = int(np.ceil((n + 1) / 2))
    fftd = xp.fft.fft(arr, axis=axis)
    slc = [slice(None)] * fftd.ndim
    slc[axis] = slice(0, nuq)
    fftc = fftd[tuple(slc)]
    spec = xp.abs(fftc) / length
    spec = spec**2
    if double_sided:
        spec = 2.0 * spec
    axes = tuple(i for i in range(spec.ndim) if i != axis)
    spec = spec.mean(axis=axes)
    k = 2.0 * np.pi * np.arange(nuq) / length
    return np.asarray(k), np.asarray(spec)


def plot_power_spectrum(
    h5_path: str | Path,
    field: str,
    *,
    step: int | str | None = None,
    steps: Iterable[int | str] | None = None,
    axis: Literal["x", "y", "z"] = "y",
    axes: str = "zxy",
    backend: Literal["numpy", "jax"] = "numpy",
    output: str | Path | None = None,
):
    if steps is None:
        if step is None:
            steps = [list_gbs_steps(h5_path, var="theta")[-1]]
        else:
            steps = [step]
    steps_list = select_steps(h5_path, steps=steps)
    if not steps_list:
        raise ValueError("No steps selected for spectrum")

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

    double_sided = axis in ("x", "y")
    spec_acc = None
    for st in steps_list:
        data = load_gbs_field(h5_path, field, step=st, axes=axes)
        snapshot = data[0]
        k, spec = power_spectrum_1d(
            snapshot, axis=ax, length=length, backend=backend, double_sided=double_sided
        )
        if spec_acc is None:
            spec_acc = np.zeros_like(spec)
        spec_acc += spec
    spec = spec_acc / float(len(steps_list))

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

    if vmin is None or vmax is None:
        first = load_gbs_field(h5_path, field, step=steps[0], axes=axes)
        mm = float(np.min(np.abs(first)))
        ll = float(np.max(np.abs(first)))
        if vmin is None:
            vmin = mm
        if vmax is None:
            vmax = ll

    with writer.saving(fig, str(output), dpi=150):
        for step in steps:
            data = load_gbs_field(h5_path, field, step=step, axes=axes)
            if index is None:
                if cut == "pol":
                    index = min(7, data.shape[1] - 1)
                elif cut == "tor":
                    index = min(7, data.shape[2] - 1)
                else:
                    index = min(7, data.shape[3] - 1)
            frame = slice_2d(data[0], cut=cut, index=index)
            if frame.shape[0] > 1 and frame.shape[1] > 1:
                frame = frame[:-1, :-1]
            ax.clear()
            im = ax.pcolormesh(frame, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
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

    fig = plt.figure(figsize=(6, 6), dpi=150, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    writer = _get_writer(output, fps=fps)

    with writer.saving(fig, str(output), dpi=150):
        for step in steps:
            data = load_gbs_field(h5_path, field, step=step, axes=axes)
            nz = int(data.shape[1]) if data.shape[1] > 0 else 1
            iz_mid = max(nz // 2 - 1, 0)
            frame = slice_2d(data[0], cut="pol", index=iz_mid)
            width = width_factor * Lx
            ixs = int(np.floor(frame.shape[1] * (Lx - width) / Lx))
            if ixs < 0:
                ixs = 0
            frame = frame[:, ixs:]
            xp, yp = _poloidal_coords(
                frame.shape[0],
                frame.shape[1],
                Lx=Lx,
                Ly=Ly,
                width_factor=width_factor,
                theta_window=theta_window,
            )
            ax.clear()
            surf = ax.plot_surface(
                xp, yp, frame, cmap=cmap, linewidth=0, antialiased=False, vmin=vmin, vmax=vmax
            )
            ax.view_init(elev=30.0, azim=-37.5)
            ax.set_axis_off()
            ax.set_xlim((-(Ly / (2.0 * np.pi) + width) * 1.05, (Ly / (2.0 * np.pi) + width) * 1.05))
            ax.set_ylim((-(Ly / (2.0 * np.pi) + width) * 1.05, (Ly / (2.0 * np.pi) + width) * 1.05))
            ax.set_zlim((np.nanmin(frame), np.nanmax(frame)))
            ax.set_box_aspect([1, 1, 0.3])
            ax.set_title(f"{field} poloidal step {step}")
            fig.subplots_adjust(0, 0, 1, 1)
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


# ---------------------------- Helpers for time/steps ----------------------------


def read_gbs_time(h5_path: str | Path, *, var: str = "theta") -> np.ndarray:
    import h5py  # type: ignore

    path = Path(h5_path)
    with h5py.File(path, "r") as f:
        key = f"data/var3d/{var}/time"
        if key not in f:
            raise KeyError(f"{key} not found")
        return np.asarray(f[key][...])


def _normalize_steps(steps: Iterable[int | str]) -> list[int]:
    out: list[int] = []
    for s in steps:
        if isinstance(s, int):
            out.append(s)
        else:
            out.append(int(str(s)))
    return out


def select_steps(
    h5_path: str | Path,
    *,
    var: str = "theta",
    start: int | None = None,
    end: int | None = None,
    stride: int = 1,
    steps: Iterable[int | str] | None = None,
) -> list[int]:
    if steps is None:
        steps = list_gbs_steps(h5_path, var=var)
    step_ints = _normalize_steps(steps)
    step_ints = sorted(step_ints)
    if start is not None:
        step_ints = [s for s in step_ints if s >= start]
    if end is not None:
        step_ints = [s for s in step_ints if s <= end]
    if stride > 1:
        step_ints = step_ints[::stride]
    return step_ints


def steps_to_time(
    h5_path: str | Path,
    steps: Iterable[int | str],
    *,
    var: str = "theta",
) -> np.ndarray:
    steps_sorted = select_steps(h5_path, var=var, steps=steps)
    all_steps = select_steps(h5_path, var=var)
    time = read_gbs_time(h5_path, var=var)
    if len(time) != len(all_steps):
        return np.arange(len(steps_sorted), dtype=float)
    step_to_idx = {s: i for i, s in enumerate(all_steps)}
    tvals = [time[step_to_idx[s]] for s in steps_sorted]
    return np.asarray(tvals)


def time_average_field(
    h5_path: str | Path,
    field: str,
    *,
    steps: Iterable[int | str],
    axes: str = "zxy",
) -> np.ndarray:
    steps_list = select_steps(h5_path, steps=steps)
    if not steps_list:
        raise ValueError("No steps selected for time average")
    acc: np.ndarray | None = None
    for step in steps_list:
        data = load_gbs_field(h5_path, field, step=step, axes=axes)
        snap = data[0].astype(np.float64)
        if acc is None:
            acc = np.zeros_like(snap, dtype=np.float64)
        acc += snap
    acc /= float(len(steps_list))
    return acc


def estimate_lfs_hfs_indices(
    h5_path: str | Path,
    *,
    steps: Iterable[int | str],
    axes: str = "zxy",
    lx_fit: float = 50.0,
) -> tuple[int, int]:
    pe_mean = time_average_field(h5_path, "pe", steps=steps, axes=axes)
    pe_mean_x_y = pe_mean.mean(axis=0)
    nx, ny = pe_mean_x_y.shape

    x_fit = np.linspace(0.0, lx_fit, max(nx - 4, 2))

    Lp = np.full(ny, np.nan)
    for iy in range(ny):
        prof = pe_mean_x_y[:, iy]
        idx_peak = int(np.argmax(prof))
        start = min(idx_peak + 2, nx - 2)
        end = max(start + 2, nx - 2)
        pfit = prof[start:end]
        if pfit.size < 2:
            continue
        xfit = x_fit[: pfit.size]
        mask = pfit > 0
        if np.count_nonzero(mask) >= 2:
            slope, _ = np.polyfit(xfit[mask], np.log(pfit[mask]), 1)
            if slope != 0:
                Lp[iy] = -1.0 / slope
    if np.all(np.isnan(Lp)):
        return ny // 2, max(ny // 4, 1)
    y_lfs = int(np.nanargmax(Lp))
    y_hfs = int(np.nanargmin(Lp))
    return y_lfs, y_hfs


def estimate_sol_x_index(
    h5_path: str | Path,
    *,
    steps: Iterable[int | str],
    field: str = "phi",
    axes: str = "zxy",
    offset: int = 5,
) -> int:
    mean_field = time_average_field(h5_path, field, steps=steps, axes=axes)
    prof_x = mean_field.mean(axis=(0, 2))
    idx = int(np.argmax(prof_x))
    return int(min(idx + offset, mean_field.shape[1] - 1))


def _slice_bounds(size: int, start: int | None, end: int | None) -> slice:
    if start is None:
        start = 0
    if end is None:
        end = size
    start = max(int(start), 0)
    end = min(int(end), size)
    return slice(start, end)


def _normalize_samples(data: np.ndarray) -> np.ndarray:
    mean = float(np.mean(data))
    std = float(np.std(data))
    if std <= 0:
        return data - mean
    return (data - mean) / std


def _normalize_samples_matlab(data: np.ndarray) -> np.ndarray:
    mean = float(np.mean(data))
    if data.size > 1:
        std = float(np.std(data, ddof=1))
    else:
        std = 0.0
    return (data - mean) / std


# ---------------------------- PDFs ----------------------------


def plot_pdf(
    h5_path: str | Path,
    field: str,
    *,
    steps: Iterable[int | str] | None = None,
    axes: str = "zxy",
    nbins: int = 100,
    value_range: tuple[float, float] | None = None,
    x_slice: tuple[int | None, int | None] | None = None,
    z_slice: tuple[int | None, int | None] | None = None,
    y_lfs: int | None = None,
    y_hfs: int | None = None,
    output: str | Path | None = None,
):
    if steps is None:
        steps = select_steps(h5_path)
    steps_list = select_steps(h5_path, steps=steps)
    if not steps_list:
        raise ValueError("No steps selected for PDF")

    data0 = load_gbs_field(h5_path, field, step=steps_list[0], axes=axes)
    nz, nx, ny = data0[0].shape

    if y_lfs is None or y_hfs is None:
        y_lfs, y_hfs = estimate_lfs_hfs_indices(h5_path, steps=steps_list, axes=axes)

    if x_slice is None:
        x_slice = (20, 22)
    if z_slice is None:
        z_mid = nz // 2
        z_slice = (z_mid - 3, z_mid + 3)

    x_sl = _slice_bounds(nx, x_slice[0] - 1 if x_slice[0] is not None else None, x_slice[1])
    z_sl = _slice_bounds(nz, z_slice[0] - 1 if z_slice[0] is not None else None, z_slice[1])

    vals_lfs: list[np.ndarray] = []
    vals_hfs: list[np.ndarray] = []
    for step in steps_list:
        data = load_gbs_field(h5_path, field, step=step, axes=axes)
        snap = data[0]
        vals_lfs.append(snap[z_sl, x_sl, y_lfs].ravel())
        vals_hfs.append(snap[z_sl, x_sl, y_hfs].ravel())
    data_lfs = _normalize_samples_matlab(np.concatenate(vals_lfs))
    data_hfs = _normalize_samples_matlab(np.concatenate(vals_hfs))

    if value_range is None:
        range_lfs = (float(np.min(data_lfs)), float(np.max(data_lfs)))
        range_hfs = (float(np.min(data_hfs)), float(np.max(data_hfs)))
    else:
        range_lfs = range_hfs = value_range

    hist_lfs, edges_lfs = np.histogram(data_lfs, bins=nbins, range=range_lfs)
    hist_hfs, edges_hfs = np.histogram(data_hfs, bins=nbins, range=range_hfs)
    centers_lfs = 0.5 * (edges_lfs[:-1] + edges_lfs[1:])
    centers_hfs = 0.5 * (edges_hfs[:-1] + edges_hfs[1:])
    hist_lfs = hist_lfs / np.trapz(hist_lfs, centers_lfs)
    hist_hfs = hist_hfs / np.trapz(hist_hfs, centers_hfs)

    skew_lfs = float(np.mean(hist_lfs**3))
    skew_hfs = float(np.mean(hist_hfs**3))
    kurt_lfs = float(np.mean(hist_lfs**4))
    kurt_hfs = float(np.mean(hist_hfs**4))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.semilogy(centers_lfs, hist_lfs, "b.", label="LFS")
    ax.semilogy(centers_hfs, hist_hfs, "r.", label="HFS")
    ax.set_xlabel(f"({field} - <{field}>)/sigma")
    ax.set_ylabel("PDF")
    if value_range is not None:
        ax.set_xlim(value_range)
    ax.set_ylim(1e-4, 2.0)
    ax.set_title(f"PDF {field}")
    ax.legend(loc="best", fontsize=8)
    ax.text(
        0.02,
        0.95,
        f"skew LFS={skew_lfs:.2f}, HFS={skew_hfs:.2f}\nkurt LFS={kurt_lfs:.2f}, HFS={kurt_hfs:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    fig.tight_layout()
    if output is not None:
        fig.savefig(str(output), dpi=150)
    return fig


# ---------------------------- Poloidal time traces ----------------------------


def extract_point_series(
    h5_path: str | Path,
    field: str,
    *,
    steps: Iterable[int | str],
    x_index: int,
    y_index: int,
    z_index: int,
    axes: str = "zxy",
) -> np.ndarray:
    steps_list = select_steps(h5_path, steps=steps)
    series = []
    for step in steps_list:
        data = load_gbs_field(h5_path, field, step=step, axes=axes)
        series.append(float(data[0][z_index, x_index, y_index]))
    return np.asarray(series)


def make_poloidal_time_trace_movie(
    h5_path: str | Path,
    field: str,
    *,
    steps: Iterable[int | str] | None = None,
    axes: str = "zxy",
    output: str | Path = "poloidal_time_trace.gif",
    fps: int = 15,
    x_index: int | None = None,
    y_index: int | None = None,
    z_index: int | None = None,
    width_factor: float = 0.8,
    theta_window: float = 0.0,
    cmap: str = "jet",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
) -> Path:
    h5_path = Path(h5_path)
    output = Path(output)
    if steps is None:
        steps = select_steps(h5_path)
    steps_list = select_steps(h5_path, steps=steps)
    if not steps_list:
        raise ValueError("No steps selected for poloidal time trace")

    data0 = load_gbs_field(h5_path, field, step=steps_list[0], axes=axes)
    nz, nx, ny = data0[0].shape
    iz_mid = max(nz // 2 - 1, 0)
    if z_index is None:
        z_index = iz_mid
    if x_index is None:
        x_index = max(nx // 3, 0)
    if y_index is None:
        y_index = max(ny // 2, 0)

    text = read_gbs_stdin_text(h5_path) or ""
    params = parse_gbs_input_text(text)
    grid = infer_gbs_grid(params)
    Lx = grid.Lx if grid is not None else 1.0
    Ly = grid.Ly if grid is not None else 1.0

    width = width_factor * Lx
    ixs = int(np.floor(nx * (Lx - width) / Lx))
    if ixs < 0:
        ixs = 0
    xp, yp = _poloidal_coords(
        ny, nx - ixs, Lx=Lx, Ly=Ly, width_factor=width_factor, theta_window=theta_window
    )

    time_arr = read_gbs_time(h5_path)
    if time_arr.size > 1:
        total_time = float(time_arr[-1] - time_arr[0])
    else:
        total_time = float(len(steps_list) - 1)
    times = np.linspace(0.0, total_time, len(steps_list))
    series = extract_point_series(
        h5_path,
        field,
        steps=steps_list,
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
        axes=axes,
    )

    fig = plt.figure(figsize=(6, 8), dpi=150)
    writer = _get_writer(output, fps=fps)
    with writer.saving(fig, str(output), dpi=150):
        for idx, step in enumerate(steps_list):
            data = load_gbs_field(h5_path, field, step=step, axes=axes)
            frame = slice_2d(data[0], cut="pol", index=z_index)
            frame = frame[:, ixs:]
            y0 = max(y_index - 2, 0)
            y1 = min(y_index + 3, frame.shape[0])
            x0 = max(x_index - 2 - ixs, 0)
            x1 = min(x_index + 3 - ixs, frame.shape[1])
            if x0 < x1 and y0 < y1:
                frame[y0:y1, x0:x1] = 0.0

            fig.clear()
            gs = fig.add_gridspec(4, 1)
            ax0 = fig.add_subplot(gs[0:3, 0])
            ax1 = fig.add_subplot(gs[3, 0])

            surf = ax0.pcolormesh(xp, yp, frame, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
            ax0.set_aspect("equal")
            ax0.axis("off")
            ax0.set_title(f"{field} poloidal step {step}")

            ax1.plot(times, series, "b-")
            ax1.plot(times[idx], series[idx], "rx", markersize=6, markeredgewidth=2)
            ax1.set_xlabel("time")
            ax1.set_ylabel(field)
            ax1.set_xlim(times[0], times[-1])

            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)
    return output


# ---------------------------- Cross coherence and phase ----------------------------


def plot_cross_coherence(
    h5_path: str | Path,
    field1: str,
    field2: str,
    *,
    steps: Iterable[int | str] | None = None,
    axes: str = "zxy",
    bins: int = 100,
    value_range: tuple[float, float] = (-4.0, 4.0),
    x_index: int | None = None,
    z_index: int | None = None,
    y_window: int = 25,
    output_prefix: str | Path | None = None,
):
    if steps is None:
        steps = select_steps(h5_path)
    steps_list = select_steps(h5_path, steps=steps)
    if not steps_list:
        raise ValueError("No steps selected for cross coherence")

    data0 = load_gbs_field(h5_path, field1, step=steps_list[0], axes=axes)
    nz, nx, ny = data0[0].shape
    if z_index is None:
        z_index = max(nz // 2 - 1, 0)

    if x_index is None:
        x_index = estimate_sol_x_index(h5_path, steps=steps_list, field=field1, axes=axes)

    y_lfs, y_hfs = estimate_lfs_hfs_indices(h5_path, steps=steps_list, axes=axes)
    y_lfs = int(np.clip(y_lfs, 0, ny - 1))
    y_hfs = int(np.clip(y_hfs, 0, ny - 1))

    y_lfs_start = max(y_lfs - y_window, 0)
    y_lfs_end = min(y_lfs + y_window, ny - 1)
    y_hfs_start = 1
    y_hfs_end = min(51, ny - 1)

    avg1 = time_average_field(h5_path, field1, steps=steps_list, axes=axes)
    avg2 = time_average_field(h5_path, field2, steps=steps_list, axes=axes)

    vals1_lfs: list[np.ndarray] = []
    vals2_lfs: list[np.ndarray] = []
    vals1_hfs: list[np.ndarray] = []
    vals2_hfs: list[np.ndarray] = []

    for step in steps_list:
        f1 = load_gbs_field(h5_path, field1, step=step, axes=axes)[0]
        f2 = load_gbs_field(h5_path, field2, step=step, axes=axes)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = (f1 - avg1) / avg1
            f2 = (f2 - avg2) / avg2

        seq1_lfs = f1[z_index, x_index, y_lfs_start : y_lfs_end + 1]
        seq2_lfs = f2[z_index, x_index, y_lfs_start : y_lfs_end + 1]
        seq1_hfs = f1[z_index, x_index, y_hfs_start : y_hfs_end + 1]
        seq2_hfs = f2[z_index, x_index, y_hfs_start : y_hfs_end + 1]

        seq1_lfs = _normalize_samples_matlab(seq1_lfs)
        seq2_lfs = _normalize_samples_matlab(seq2_lfs)
        seq1_hfs = _normalize_samples_matlab(seq1_hfs)
        seq2_hfs = _normalize_samples_matlab(seq2_hfs)

        vals1_lfs.append(seq1_lfs)
        vals2_lfs.append(seq2_lfs)
        vals1_hfs.append(seq1_hfs)
        vals2_hfs.append(seq2_hfs)

    x_lfs = np.concatenate(vals1_lfs)
    y_lfs_vals = np.concatenate(vals2_lfs)
    x_hfs = np.concatenate(vals1_hfs)
    y_hfs_vals = np.concatenate(vals2_hfs)

    low, high = value_range
    edges = np.linspace(low, high, bins + 1)
    h_lfs = np.zeros((bins, bins))
    h_hfs = np.zeros((bins, bins))

    mask_lfs = np.isfinite(x_lfs) & np.isfinite(y_lfs_vals)
    for xv, yv in zip(x_lfs[mask_lfs], y_lfs_vals[mask_lfs]):
        ix = np.searchsorted(edges, xv, side="right") - 1
        iy = np.searchsorted(edges, yv, side="right") - 1
        if 0 <= ix < bins and 0 <= iy < bins:
            h_lfs[ix, iy] += 1

    mask_hfs = np.isfinite(x_hfs) & np.isfinite(y_hfs_vals)
    for xv, yv in zip(x_hfs[mask_hfs], y_hfs_vals[mask_hfs]):
        ix = np.searchsorted(edges, xv, side="right") - 1
        iy = np.searchsorted(edges, yv, side="right") - 1
        if 0 <= ix < bins and 0 <= iy < bins:
            h_hfs[ix, iy] += 1

    h_lfs = h_lfs / np.sum(h_lfs)
    h_hfs = h_hfs / np.sum(h_hfs)

    xcent = 0.5 * (edges[:-1] + edges[1:])
    ycent = xcent

    fig1, ax1 = plt.subplots(figsize=(5, 4), dpi=150)
    c1 = ax1.contourf(xcent, ycent, h_hfs.T, levels=20)
    ax1.set_xlabel(f"{field1} / sigma")
    ax1.set_ylabel(f"{field2} / sigma")
    ax1.set_title("Cross coherence HFS")
    fig1.colorbar(c1, ax=ax1)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(5, 4), dpi=150)
    c2 = ax2.contourf(xcent, ycent, h_lfs.T, levels=20)
    ax2.set_xlabel(f"{field1} / sigma")
    ax2.set_ylabel(f"{field2} / sigma")
    ax2.set_title("Cross coherence LFS")
    fig2.colorbar(c2, ax=ax2)
    fig2.tight_layout()

    if output_prefix is not None:
        output_prefix = Path(output_prefix)
        fig1.savefig(str(output_prefix.with_name(output_prefix.name + "_hfs.png")), dpi=150)
        fig2.savefig(str(output_prefix.with_name(output_prefix.name + "_lfs.png")), dpi=150)
    return fig1, fig2


def plot_cross_phase(
    h5_path: str | Path,
    field1: str,
    field2: str,
    *,
    steps: Iterable[int | str] | None = None,
    axes: str = "zxy",
    x_index: int | None = None,
    output_prefix: str | Path | None = None,
    nypts: int = 8,
) -> tuple[plt.Figure, plt.Figure]:
    if steps is None:
        steps = select_steps(h5_path)
    steps_list = select_steps(h5_path, steps=steps)
    if not steps_list:
        raise ValueError("No steps selected for cross phase")

    data0 = load_gbs_field(h5_path, field1, step=steps_list[0], axes=axes)
    nz, nx, ny = data0[0].shape
    if x_index is None:
        x_index = 17
    x_index = int(np.clip(x_index, 0, nx - 1))

    avg1 = time_average_field(h5_path, field1, steps=steps_list, axes=axes)
    avg2 = time_average_field(h5_path, field2, steps=steps_list, axes=axes)

    y_lfs, _ = estimate_lfs_hfs_indices(h5_path, steps=steps_list, axes=axes)
    y_lfs_start = max(y_lfs - nypts // 2, 0)
    y_lfs_end = min(y_lfs_start + nypts, ny)
    y_hfs_start = 0
    y_hfs_end = min(y_hfs_start + nypts, ny)

    fft1_hfs = []
    fft2_hfs = []
    fft1_lfs = []
    fft2_lfs = []

    for step in steps_list:
        f1 = load_gbs_field(h5_path, field1, step=step, axes=axes)[0] - avg1
        f2 = load_gbs_field(h5_path, field2, step=step, axes=axes)[0] - avg2

        f1_hfs = f1[:, x_index, y_hfs_start:y_hfs_end].mean(axis=0)
        f2_hfs = f2[:, x_index, y_hfs_start:y_hfs_end].mean(axis=0)
        f1_lfs = f1[:, x_index, y_lfs_start:y_lfs_end].mean(axis=0)
        f2_lfs = f2[:, x_index, y_lfs_start:y_lfs_end].mean(axis=0)

        fft1_hfs.append(np.fft.fft(f1_hfs))
        fft2_hfs.append(np.fft.fft(f2_hfs))
        fft1_lfs.append(np.fft.fft(f1_lfs))
        fft2_lfs.append(np.fft.fft(f2_lfs))

    fft1_hfs = np.asarray(fft1_hfs)
    fft2_hfs = np.asarray(fft2_hfs)
    fft1_lfs = np.asarray(fft1_lfs)
    fft2_lfs = np.asarray(fft2_lfs)

    kmax = fft1_hfs.shape[1] // 2
    fft1_hfs = fft1_hfs[:, 1 : kmax + 1]
    fft2_hfs = fft2_hfs[:, 1 : kmax + 1]
    fft1_lfs = fft1_lfs[:, 1 : kmax + 1]
    fft2_lfs = fft2_lfs[:, 1 : kmax + 1]

    ph_hfs = np.angle(fft2_hfs) - np.angle(fft1_hfs)
    ph_lfs = np.angle(fft2_lfs) - np.angle(fft1_lfs)

    num_int = 50
    bins = np.linspace(-np.pi, np.pi, num_int + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    ph_hist_hfs = np.zeros((kmax, num_int))
    ph_hist_lfs = np.zeros((kmax, num_int))
    for isl in range(ph_hfs.shape[0]):
        for kk in range(kmax):
            val_hfs = ph_hfs[isl, kk]
            idx_hfs = np.searchsorted(bins, val_hfs, side="right") - 1
            if 0 <= idx_hfs < num_int:
                ph_hist_hfs[kk, idx_hfs] += 1
            val_lfs = ph_lfs[isl, kk]
            idx_lfs = np.searchsorted(bins, val_lfs, side="right") - 1
            if 0 <= idx_lfs < num_int:
                ph_hist_lfs[kk, idx_lfs] += 1

    tot_hfs = np.sum(ph_hist_hfs)
    tot_lfs = np.sum(ph_hist_lfs)
    norm_hfs = np.max(ph_hist_hfs / (tot_hfs + 1e-30))
    norm_lfs = np.max(ph_hist_lfs / (tot_lfs + 1e-30))

    power_hfs = np.abs(np.mean(fft1_hfs, axis=0)) ** 2
    power_lfs = np.abs(np.mean(fft1_lfs, axis=0)) ** 2
    power_hfs = np.tile(power_hfs[:, None], (1, num_int))
    power_lfs = np.tile(power_lfs[:, None], (1, num_int))

    ph_hist_hfs = ph_hist_hfs / (norm_hfs + 1e-30) / (power_hfs + 1e-30)
    ph_hist_lfs = ph_hist_lfs / (norm_lfs + 1e-30) / (power_lfs + 1e-30)

    grid = infer_gbs_grid(parse_gbs_input_text(read_gbs_stdin_text(h5_path) or ""))
    Ly = grid.Ly if grid is not None else float(ny)
    ky = (2.0 * np.pi / Ly) * np.linspace(1, kmax, kmax)
    phase_deg = centers * (180.0 / np.pi)

    fig_hfs, ax_hfs = plt.subplots(figsize=(6, 4), dpi=150)
    c1 = ax_hfs.contour(phase_deg, ky, ph_hist_hfs, levels=30)
    ax_hfs.set_yscale("log")
    ax_hfs.set_xlabel("phase shift (deg)")
    ax_hfs.set_ylabel("ky")
    ax_hfs.set_title(f"Phase shift HFS {field1} vs {field2}")
    fig_hfs.colorbar(c1, ax=ax_hfs)

    fig_lfs, ax_lfs = plt.subplots(figsize=(6, 4), dpi=150)
    c2 = ax_lfs.contour(phase_deg, ky, ph_hist_lfs, levels=30)
    ax_lfs.set_yscale("log")
    ax_lfs.set_xlabel("phase shift (deg)")
    ax_lfs.set_ylabel("ky")
    ax_lfs.set_title(f"Phase shift LFS {field1} vs {field2}")
    fig_lfs.colorbar(c2, ax=ax_lfs)

    if output_prefix is not None:
        output_prefix = Path(output_prefix)
        fig_hfs.savefig(str(output_prefix.with_name(output_prefix.name + "_hfs.png")), dpi=150)
        fig_lfs.savefig(str(output_prefix.with_name(output_prefix.name + "_lfs.png")), dpi=150)
    return fig_hfs, fig_lfs
