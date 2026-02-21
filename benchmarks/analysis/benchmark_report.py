#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _rms(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr**2)))


def _fit_growth_rate(t: np.ndarray, rms: np.ndarray, frac: float = 0.3) -> float:
    t = np.asarray(t)
    rms = np.asarray(rms)
    mask = (rms > 0) & np.isfinite(rms) & np.isfinite(t)
    t = t[mask]
    rms = rms[mask]
    if t.size < 3:
        return 0.0
    n = max(int(frac * t.size), 3)
    t = t[:n]
    y = np.log(rms[:n])
    coeff = np.polyfit(t, y, 1)
    return float(coeff[0])


def _trim_guard(arr: np.ndarray, axis: int, g: int) -> np.ndarray:
    if g <= 0:
        return arr
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(g, -g)
    return arr[tuple(sl)]


def _permute_spatial(data: np.ndarray, axes: str, target: str = "zxy") -> np.ndarray:
    axes = axes.lower()
    target = target.lower()
    if len(axes) != 3 or len(target) != 3:
        raise ValueError("axes and target must have length 3")
    if sorted(axes) != sorted(target):
        raise ValueError(f"axes {axes} and target {target} must contain same labels")
    perm = [axes.index(c) for c in target]
    return data.transpose(0, *(p + 1 for p in perm))


def _parse_gbs_input(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    pat = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([-+0-9.eEdD]+)")
    for line in path.read_text().splitlines():
        line = line.split("!")[0].strip()
        if not line:
            continue
        for key, val in pat.findall(line):
            try:
                out[key.lower()] = float(val.replace("D", "E").replace("d", "e"))
            except ValueError:
                continue
    return out


def _parse_time_log(path: Path | None) -> tuple[float, float] | None:
    if path is None or not path.exists():
        return None
    text = path.read_text()
    real_match = re.search(r"^\s*([0-9.]+)\s+real", text, re.MULTILINE)
    mem_match = re.search(r"^\s*([0-9]+)\s+maximum resident set size", text, re.MULTILINE)
    if real_match is None and mem_match is None:
        return None
    runtime = float(real_match.group(1)) if real_match else float("nan")
    mem_bytes = float(mem_match.group(1)) if mem_match else float("nan")
    mem_mb = mem_bytes / (1024.0 ** 2) if mem_bytes == mem_bytes else float("nan")
    return runtime, mem_mb


def _spectrum_ky(
    snapshot: np.ndarray, Ly: float, ky_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    # snapshot shape (z, x, y)
    spec = np.abs(np.fft.rfft(snapshot, axis=2)) ** 2
    spec = spec.mean(axis=(0, 1))
    ny = snapshot.shape[2]
    ky = 2.0 * np.pi * np.fft.rfftfreq(ny, d=Ly / ny)
    return ky * ky_scale, spec


def _spectrum_kx(snapshot: np.ndarray, Lx: float) -> tuple[np.ndarray, np.ndarray]:
    spec = np.abs(np.fft.rfft(snapshot, axis=1)) ** 2
    spec = spec.mean(axis=(0, 2))
    nx = snapshot.shape[1]
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=Lx / nx)
    return kx, spec


def _freq_spectrum(series: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series)
    t = np.asarray(t)
    if series.size < 4:
        return np.asarray([]), np.asarray([])
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return np.asarray([]), np.asarray([])
    series = series - np.mean(series)
    spec = np.abs(np.fft.rfft(series)) ** 2 / series.size
    freq = np.fft.rfftfreq(series.size, d=dt)
    return freq, spec


def _cross_phase_ky(
    n: np.ndarray, phi: np.ndarray, Ly: float, ky_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_hat = np.fft.rfft(n, axis=2)
    phi_hat = np.fft.rfft(phi, axis=2)
    cross = (n_hat * np.conj(phi_hat)).mean(axis=(0, 1))
    pn = (np.abs(n_hat) ** 2).mean(axis=(0, 1))
    pphi = (np.abs(phi_hat) ** 2).mean(axis=(0, 1))
    coh = np.abs(cross) / np.sqrt(np.maximum(pn * pphi, 1e-30))
    phase = np.angle(cross)
    ny = n.shape[2]
    ky = 2.0 * np.pi * np.fft.rfftfreq(ny, d=Ly / ny)
    return ky * ky_scale, phase, coh


def _particle_flux(n: np.ndarray, phi: np.ndarray, Ly: float) -> np.ndarray:
    dy = Ly / n.shape[2]
    dphi_dy = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2.0 * dy)
    v_ex = -dphi_dy
    flux = (n * v_ex).mean(axis=(0, 2))
    return flux


def _fit_lp(x: np.ndarray, profile: np.ndarray, frac=(0.3, 0.8)) -> float:
    if profile.size < 3:
        return 0.0
    xmin = x.min()
    xmax = x.max()
    lo = xmin + frac[0] * (xmax - xmin)
    hi = xmin + frac[1] * (xmax - xmin)
    mask = (x >= lo) & (x <= hi) & (profile > 0)
    if mask.sum() < 3:
        return 0.0
    coeff = np.polyfit(x[mask], np.log(profile[mask]), 1)
    if coeff[0] == 0:
        return 0.0
    return float(-1.0 / coeff[0])


@dataclass
class Dataset:
    name: str
    t: np.ndarray
    n: np.ndarray
    Te: np.ndarray | None
    phi: np.ndarray | None
    omega: np.ndarray | None
    Lx: float
    Ly: float
    Lz: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    point_idx: tuple[int, int, int]
    ky_scale: float = 1.0
    rms_n_series: np.ndarray | None = None
    rms_phi_series: np.ndarray | None = None
    rms_Te_series: np.ndarray | None = None
    point_n_series: np.ndarray | None = None
    point_phi_series: np.ndarray | None = None
    point_Te_series: np.ndarray | None = None


def _load_hermes(
    path: Path,
    axes: str = "xyz",
    hermes_grid: Path | None = None,
    ky_scale_mode: str = "metric_shift",
) -> Dataset:
    import netCDF4  # type: ignore

    with netCDF4.Dataset(path) as ds:
        t = np.asarray(ds.variables["t"][:]) if "t" in ds.variables else np.arange(ds.dimensions["t"].size)
        mxg = int(ds.variables["MXG"][:]) if "MXG" in ds.variables else 0
        myg = int(ds.variables["MYG"][:]) if "MYG" in ds.variables else 0
        mzg = int(ds.variables["MZG"][:]) if "MZG" in ds.variables else 0

        def load_var(name: str) -> np.ndarray | None:
            if name not in ds.variables:
                return None
            data = np.asarray(ds.variables[name][:])
            if data.ndim == 0:
                return None
            if data.ndim == 3:
                data = data[None, ...]
            if data.ndim != 4:
                return None
            # data shape (t,x,y,z)
            data = _trim_guard(data, 1, mxg)
            data = _trim_guard(data, 2, myg)
            data = _trim_guard(data, 3, mzg)
            # reorder to (t,z,x,y)
            return _permute_spatial(data, axes, target="zxy")

        n = load_var("Ne")
        if n is None:
            raise ValueError("Hermes Ne not found")
        Te = load_var("Te")
        if Te is None:
            Pe = load_var("Pe")
            if Pe is not None:
                Te = Pe / np.maximum(n, 1e-12)
        phi = load_var("phi")
        omega = load_var("omega")
        if omega is None:
            omega = load_var("Vort")

        dy = ds.variables["dy"][:] if "dy" in ds.variables else None
        dz = ds.variables["dz"][:] if "dz" in ds.variables else None
        dx = ds.variables["dx"][:] if "dx" in ds.variables else None

    ny = n.shape[3]
    nx = n.shape[2]
    nz = n.shape[1]
    if dy is not None:
        dy = _trim_guard(dy, 0, mxg)
        dy = _trim_guard(dy, 1, myg)
        Ly = float(np.mean(dy)) * ny
    else:
        Ly = float(ny)
    if dz is not None:
        dz = _trim_guard(dz, 0, mxg)
        dz = _trim_guard(dz, 1, myg)
        Lz = float(np.mean(dz)) * nz
    else:
        Lz = float(nz)
    if dx is not None:
        dx = _trim_guard(dx, 0, mxg)
        dx = _trim_guard(dx, 1, myg)
        Lx = float(np.mean(dx)) * nx
    else:
        Lx = float(nx)

    ky_scale = 1.0
    if hermes_grid is not None and hermes_grid.exists():
        try:
            with netCDF4.Dataset(str(hermes_grid)) as gds:
                mode = str(ky_scale_mode).lower()
                use_metric = mode in ("metric", "metric_shift")
                use_shift = mode in ("metric_shift", "shift", "shiftangle")

                if "dy" in gds.variables:
                    dy_g = np.asarray(gds.variables["dy"][:])
                    if dy_g.ndim == 2:
                        dy_g = _trim_guard(dy_g, 0, mxg)
                        dy_g = _trim_guard(dy_g, 1, myg)
                    dy_mean = float(np.mean(dy_g))
                    if "hthe" in gds.variables:
                        hthe = np.asarray(gds.variables["hthe"][:])
                        if hthe.ndim == 2:
                            hthe = _trim_guard(hthe, 0, mxg)
                            hthe = _trim_guard(hthe, 1, myg)
                        Ly = float(np.mean(hthe)) * dy_mean * ny
                    else:
                        Ly = dy_mean * ny
                if use_metric:
                    gyy_name = "gyy_ballooning" if "gyy_ballooning" in gds.variables else "gyy"
                    if gyy_name in gds.variables:
                        gyy = np.asarray(gds.variables[gyy_name][:])
                        if gyy.ndim == 2:
                            gyy = _trim_guard(gyy, 0, mxg)
                            gyy = _trim_guard(gyy, 1, myg)
                        gyy_mean = float(np.mean(gyy))
                        if gyy_mean > 0:
                            ky_scale *= float(np.sqrt(gyy_mean))
                if use_shift and "ShiftAngle" in gds.variables:
                    shift = np.asarray(gds.variables["ShiftAngle"][:])
                    if shift.size:
                        shift_mean = float(np.mean(shift))
                        ky_scale *= float(abs(np.cos(shift_mean)))
        except Exception:
            pass

    x = np.linspace(0.0, Lx, nx, endpoint=False)
    y = np.linspace(0.0, Ly, ny, endpoint=False)
    z = np.linspace(0.0, Lz, nz, endpoint=False)
    point_idx = (nz // 2, nx // 2, ny // 2)
    return Dataset(
        name="Hermes",
        t=t,
        n=n,
        Te=Te,
        phi=phi,
        omega=omega,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        x=x,
        y=y,
        z=z,
        point_idx=point_idx,
        ky_scale=ky_scale,
    )


def _load_gbs(path: Path, gbs_input: Path | None = None, axes: str = "zxy") -> Dataset:
    import h5py  # type: ignore

    with h5py.File(path, "r") as f:
        def load_var(name: str) -> np.ndarray | None:
            if f"data/var3d/{name}" not in f:
                return None
            grp = f[f"data/var3d/{name}"]
            steps = sorted([k for k in grp.keys() if k.isdigit()])
            data = np.array([grp[k][...] for k in steps])
            # data shape (t, z, x, y) in h5py (z, x, y) per step
            ax = axes.lower()
            if len(ax) != 3:
                raise ValueError("GBS axes must be length 3")
            if "x" not in ax or "y" not in ax or "z" not in ax:
                raise ValueError("GBS axes must include x,y,z")
            x_axis = ax.index("x")
            y_axis = ax.index("y")
            data = _trim_guard(data, x_axis + 1, 1)
            data = _trim_guard(data, y_axis + 1, 1)
            data = _permute_spatial(data, ax, target="zxy")
            return data

        t = None
        if "data/var3d/time" in f:
            t = np.asarray(f["data/var3d/time"][...])
        n = load_var("theta")
        if n is None:
            raise ValueError("GBS theta not found")
        Te = load_var("temperature")
        phi = load_var("strmf")
        omega = load_var("omega")

    # theta, temperature are log
    n = np.exp(n)
    if Te is not None:
        Te = np.exp(Te)

    if t is None:
        t = np.arange(n.shape[0])

    ny = n.shape[3]
    nx = n.shape[2]
    nz = n.shape[1]
    params = _parse_gbs_input(gbs_input) if gbs_input else {}
    Lx = float(params.get("xmax", nx) - params.get("xmin", 0.0))
    Ly = float(params.get("ymax", ny) - params.get("ymin", 0.0))
    Lz = float(params.get("zmax", nz) - params.get("zmin", 0.0))
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    y = np.linspace(0.0, Ly, ny, endpoint=False)
    z = np.linspace(0.0, Lz, nz, endpoint=False)
    point_idx = (nz // 2, nx // 2, ny // 2)
    return Dataset(
        name="GBS",
        t=t,
        n=n,
        Te=Te,
        phi=phi,
        omega=omega,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        x=x,
        y=y,
        z=z,
        point_idx=point_idx,
    )


def _load_jaxdrb(path: Path, config: Path | None = None, axes: str = "zxy") -> Dataset:
    data = np.load(path)
    t = data["times"]
    n = data["snapshot_n"][None, ...]
    Te = data["snapshot_Te"][None, ...] if "snapshot_Te" in data else None
    phi = data["snapshot_phi"][None, ...] if "snapshot_phi" in data else None
    omega = data["snapshot_omega"][None, ...] if "snapshot_omega" in data else None
    n = _permute_spatial(n, axes, target="zxy")
    if Te is not None:
        Te = _permute_spatial(Te, axes, target="zxy")
    if phi is not None:
        phi = _permute_spatial(phi, axes, target="zxy")
    if omega is not None:
        omega = _permute_spatial(omega, axes, target="zxy")
    point_idx = tuple(int(x) for x in data["point_idx"]) if "point_idx" in data else (0, 0, 0)
    rms_n_series = data["rms_n"] if "rms_n" in data else None
    rms_phi_series = data["rms_phi"] if "rms_phi" in data else None
    rms_Te_series = data["rms_Te"] if "rms_Te" in data else None
    point_n_series = data["point_n"] if "point_n" in data else None
    point_phi_series = data["point_phi"] if "point_phi" in data else None
    point_Te_series = data["point_Te"] if "point_Te" in data else None

    Lx = float(n.shape[2])
    Ly = float(n.shape[3])
    Lz = float(n.shape[1])
    if config and config.exists():
        try:
            import tomllib

            cfg = tomllib.loads(config.read_text())
            geom = cfg.get("geometry", {})
            Lx = float(geom.get("Lx", Lx))
            Ly = float(geom.get("Ly", Ly))
            Lz = float(geom.get("Lz", Lz))
        except Exception:
            pass

    nx = n.shape[2]
    ny = n.shape[3]
    nz = n.shape[1]
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    y = np.linspace(0.0, Ly, ny, endpoint=False)
    z = np.linspace(0.0, Lz, nz, endpoint=False)
    return Dataset(
        name="jax_drb",
        t=t,
        n=n,
        Te=Te,
        phi=phi,
        omega=omega,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        x=x,
        y=y,
        z=z,
        point_idx=point_idx,
        rms_n_series=rms_n_series,
        rms_phi_series=rms_phi_series,
        rms_Te_series=rms_Te_series,
        point_n_series=point_n_series,
        point_phi_series=point_phi_series,
        point_Te_series=point_Te_series,
    )


def _series_from_dataset(ds: Dataset) -> dict[str, np.ndarray]:
    if ds.rms_n_series is not None:
        rms_n = ds.rms_n_series
        rms_phi = ds.rms_phi_series
        rms_Te = ds.rms_Te_series
        point_n = ds.point_n_series
        point_phi = ds.point_phi_series
        point_Te = ds.point_Te_series
    else:
        n = ds.n
        phi = ds.phi
        Te = ds.Te
        rms_n = np.array([_rms(n[i]) for i in range(n.shape[0])])
        rms_phi = np.array([_rms(phi[i]) for i in range(phi.shape[0])]) if phi is not None else None
        rms_Te = np.array([_rms(Te[i]) for i in range(Te.shape[0])]) if Te is not None else None
        z0, x0, y0 = ds.point_idx
        point_n = n[:, z0, x0, y0]
        point_phi = phi[:, z0, x0, y0] if phi is not None else None
        point_Te = Te[:, z0, x0, y0] if Te is not None else None
    return {
        "rms_n": rms_n,
        "rms_phi": rms_phi,
        "rms_Te": rms_Te,
        "point_n": point_n,
        "point_phi": point_phi,
        "point_Te": point_Te,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate benchmark plots for Hermes/GBS/jax_drb.")
    p.add_argument("--hermes", required=True, help="Path to Hermes BOUT.dmp.0.nc")
    p.add_argument("--gbs", default=None, help="Path to GBS results_*.h5 (optional)")
    p.add_argument("--jaxdrb", required=True, help="Path to jax_drb .npz output")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--gbs-input", default=None, help="GBS input file for geometry lengths")
    p.add_argument("--jaxdrb-config", default=None, help="jax_drb TOML config for geometry lengths")
    p.add_argument("--hermes-grid", default=None, help="Optional Hermes grid file (salpha.nc)")
    p.add_argument("--hermes-axes", default="xyz", help="Hermes spatial axis order in file (default xyz)")
    p.add_argument(
        "--hermes-ky-scale",
        default="metric_shift",
        choices=("none", "metric", "metric_shift"),
        help="Hermes ky scaling: none, metric (gyy), or metric_shift (gyy + ShiftAngle).",
    )
    p.add_argument("--gbs-axes", default="zxy", help="GBS spatial axis order in file (default zxy)")
    p.add_argument("--jaxdrb-axes", default="zxy", help="jax_drb spatial axis order in file (default zxy)")
    p.add_argument("--growth-frac", type=float, default=0.3)
    p.add_argument("--hermes-time", default=None, help="Optional /usr/bin/time -l log for Hermes")
    p.add_argument("--jaxdrb-time", default=None, help="Optional /usr/bin/time -l log for jax_drb")
    p.add_argument("--gbs-time", default=None, help="Optional /usr/bin/time -l log for GBS")
    args = p.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    hermes = _load_hermes(
        Path(args.hermes),
        axes=args.hermes_axes,
        hermes_grid=Path(args.hermes_grid) if args.hermes_grid else None,
        ky_scale_mode=args.hermes_ky_scale,
    )
    jaxdrb = _load_jaxdrb(
        Path(args.jaxdrb),
        Path(args.jaxdrb_config) if args.jaxdrb_config else None,
        axes=args.jaxdrb_axes,
    )
    datasets = [hermes, jaxdrb]
    if args.gbs:
        gbs = _load_gbs(
            Path(args.gbs),
            Path(args.gbs_input) if args.gbs_input else None,
            axes=args.gbs_axes,
        )
        datasets = [hermes, gbs, jaxdrb]

    series = {ds.name: _series_from_dataset(ds) for ds in datasets}

    metrics: dict[str, Any] = {}
    for ds in datasets:
        s = series[ds.name]
        metrics[f"{ds.name}_growth_n"] = _fit_growth_rate(ds.t, s["rms_n"], frac=args.growth_frac)
        if s["rms_phi"] is not None:
            metrics[f"{ds.name}_growth_phi"] = _fit_growth_rate(ds.t, s["rms_phi"], frac=args.growth_frac)
        metrics[f"{ds.name}_rms_n_final"] = float(s["rms_n"][-1])
        if s["rms_phi"] is not None:
            metrics[f"{ds.name}_rms_phi_final"] = float(s["rms_phi"][-1])

    time_logs = {
        "Hermes": _parse_time_log(Path(args.hermes_time)) if args.hermes_time else None,
        "jax_drb": _parse_time_log(Path(args.jaxdrb_time)) if args.jaxdrb_time else None,
        "GBS": _parse_time_log(Path(args.gbs_time)) if args.gbs_time else None,
    }
    for name, stats in time_logs.items():
        if stats is None:
            continue
        runtime_s, mem_mb = stats
        metrics[f"{name}_runtime_s"] = float(runtime_s)
        metrics[f"{name}_mem_mb"] = float(mem_mb)

    # Snapshot-based metrics
    for ds in datasets:
        snap_n = ds.n[-1]
        profile_n = snap_n.mean(axis=(0, 2))
        lp_n = _fit_lp(ds.x, profile_n)
        metrics[f"{ds.name}_Lp_n"] = lp_n
        ky_n, spec_n = _spectrum_ky(snap_n, ds.Ly, ds.ky_scale)
        if ky_n.size > 1:
            idx_n = int(np.argmax(spec_n[1:])) + 1
            metrics[f"{ds.name}_ky_peak_n"] = float(ky_n[idx_n])
            metrics[f"{ds.name}_spec_peak_n"] = float(spec_n[idx_n])
        if ds.phi is not None:
            ky_p, spec_p = _spectrum_ky(ds.phi[-1], ds.Ly, ds.ky_scale)
            if ky_p.size > 1:
                idx_p = int(np.argmax(spec_p[1:])) + 1
                metrics[f"{ds.name}_ky_peak_phi"] = float(ky_p[idx_p])
                metrics[f"{ds.name}_spec_peak_phi"] = float(spec_p[idx_p])
            ky_cp, phase_cp, coh_cp = _cross_phase_ky(ds.n[-1], ds.phi[-1], ds.Ly, ds.ky_scale)
            if ky_cp.size > 1:
                idx_cp = int(np.argmax(spec_n[1:])) + 1 if spec_n.size == ky_cp.size else int(np.argmax(coh_cp[1:])) + 1
                metrics[f"{ds.name}_phase_ky_peak"] = float(phase_cp[idx_cp])
                metrics[f"{ds.name}_coh_ky_peak"] = float(coh_cp[idx_cp])
            flux = _particle_flux(ds.n[-1], ds.phi[-1], ds.Ly)
            metrics[f"{ds.name}_flux_mean"] = float(np.mean(flux))
            metrics[f"{ds.name}_flux_peak_abs"] = float(np.max(np.abs(flux)))

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    import matplotlib.pyplot as plt

    # RMS time series
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds in datasets:
        ax.plot(ds.t, series[ds.name]["rms_n"], label=f"{ds.name} n")
    ax.set_xlabel("t")
    ax.set_ylabel("RMS(n)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "rms_n_time_series.png", dpi=150)
    plt.close(fig)

    if all(series[ds.name]["rms_phi"] is not None for ds in datasets):
        fig, ax = plt.subplots(figsize=(8, 4))
        for ds in datasets:
            ax.plot(ds.t, series[ds.name]["rms_phi"], label=f"{ds.name} phi")
        ax.set_xlabel("t")
        ax.set_ylabel("RMS(phi)")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "rms_phi_time_series.png", dpi=150)
        plt.close(fig)

    # Frequency spectra at point
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds in datasets:
        freq, spec = _freq_spectrum(series[ds.name]["point_n"], ds.t)
        if freq.size:
            ax.plot(freq, spec + 1e-30, label=f"{ds.name} n")
    ax.set_xlabel("f")
    ax.set_ylabel("PSD(n)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "freq_spectrum_n.png", dpi=150)
    plt.close(fig)

    # ky spectra
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds in datasets:
        ky, spec = _spectrum_ky(ds.n[-1], ds.Ly, ds.ky_scale)
        ax.plot(ky, spec + 1e-30, label=ds.name)
    ax.set_xlabel("k_y")
    ax.set_ylabel("Spectrum(n)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "ky_spectrum_n.png", dpi=150)
    plt.close(fig)

    if all(ds.phi is not None for ds in datasets):
        fig, ax = plt.subplots(figsize=(8, 4))
        for ds in datasets:
            ky, spec = _spectrum_ky(ds.phi[-1], ds.Ly, ds.ky_scale)  # type: ignore[arg-type]
            ax.plot(ky, spec + 1e-30, label=ds.name)
        ax.set_xlabel("k_y")
        ax.set_ylabel("Spectrum(phi)")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "ky_spectrum_phi.png", dpi=150)
        plt.close(fig)

    # Mean profiles
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds in datasets:
        profile_n = ds.n[-1].mean(axis=(0, 2))
        ax.plot(ds.x, profile_n, label=ds.name)
    ax.set_xlabel("x")
    ax.set_ylabel("Mean n")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "mean_profile_n.png", dpi=150)
    plt.close(fig)

    # Particle flux
    if all(ds.phi is not None for ds in datasets):
        fig, ax = plt.subplots(figsize=(8, 4))
        for ds in datasets:
            flux = _particle_flux(ds.n[-1], ds.phi[-1], ds.Ly)  # type: ignore[arg-type]
            ax.plot(ds.x, flux, label=ds.name)
        ax.set_xlabel("x")
        ax.set_ylabel("Flux n v_E,x")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "particle_flux_profile.png", dpi=150)
        plt.close(fig)

    # Cross-phase and coherence
    if all(ds.phi is not None for ds in datasets):
        fig, ax = plt.subplots(figsize=(8, 4))
        for ds in datasets:
            ky, phase, _ = _cross_phase_ky(ds.n[-1], ds.phi[-1], ds.Ly, ds.ky_scale)  # type: ignore[arg-type]
            ax.plot(ky, phase, label=ds.name)
        ax.set_xlabel("k_y")
        ax.set_ylabel("Phase(n,phi)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "cross_phase_ky.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for ds in datasets:
            ky, _, coh = _cross_phase_ky(ds.n[-1], ds.phi[-1], ds.Ly, ds.ky_scale)  # type: ignore[arg-type]
            ax.plot(ky, coh, label=ds.name)
        ax.set_xlabel("k_y")
        ax.set_ylabel("Coherence(n,phi)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "cross_coherence_ky.png", dpi=150)
        plt.close(fig)

    # Snapshots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for col, ds in enumerate(datasets):
        im0 = axes[0, col].imshow(ds.n[-1][ds.n.shape[1] // 2], origin="lower", aspect="auto")
        axes[0, col].set_title(f"{ds.name} n")
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)
        if ds.phi is not None:
            im1 = axes[1, col].imshow(ds.phi[-1][ds.n.shape[1] // 2], origin="lower", aspect="auto")
            axes[1, col].set_title(f"{ds.name} phi")
            fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)
        else:
            axes[1, col].axis("off")
    fig.tight_layout()
    fig.savefig(outdir / "snapshots_n_phi.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
