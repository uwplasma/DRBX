from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from jaxdrb.benchmarking import (
    BenchmarkBundle,
    BenchmarkNormalization,
    compute_cross_coherence_phase,
    compute_fluctuation_rms,
    compute_frequency_psd,
    compute_ky_psd,
    compute_pdf,
    compute_radial_particle_flux_profile,
    compute_target_fluxes,
    finite_run_gate,
    save_bundle_npz,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a shared Hermes/jax_drb benchmark bundle (normalized + SI)."
    )
    p.add_argument("--code", choices=("jax", "hermes"), required=True)
    p.add_argument("--input", required=True, help="jax npz file or Hermes data directory")
    p.add_argument("--output", required=True, help="Output bundle npz path")
    p.add_argument("--geometry", default="tokamak_open_field")
    p.add_argument("--config", default=None, help="jax_drb TOML config (for spacing/normalization)")
    p.add_argument("--Nnorm", type=float, default=None)
    p.add_argument("--Tnorm-eV", type=float, default=None)
    p.add_argument("--Bnorm-T", type=float, default=None)
    p.add_argument("--m-i-amu", type=float, default=2.0)
    p.add_argument("--Z-i", type=float, default=1.0)
    p.add_argument("--nperseg", type=int, default=256)
    p.add_argument("--bins", type=int, default=120)
    p.add_argument("--max-growth-factor", type=float, default=None)
    p.add_argument("--max-rms-abs", type=float, default=None)
    return p.parse_args()


def _load_toml(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def _norm_from_cfg_or_args(cfg: dict[str, Any], args: argparse.Namespace) -> BenchmarkNormalization:
    nrm = cfg.get("normalization", {}) if isinstance(cfg, dict) else {}
    Nnorm = float(args.Nnorm if args.Nnorm is not None else nrm.get("n0", 1.0e19))
    Tnorm = float(args.Tnorm_eV if args.Tnorm_eV is not None else nrm.get("Te0_eV", 50.0))
    Bnorm = float(args.Bnorm_T if args.Bnorm_T is not None else nrm.get("B0", 1.0))
    m_i = float(args.m_i_amu if args.m_i_amu is not None else nrm.get("m_i_amu", 2.0))
    z_i = float(args.Z_i if args.Z_i is not None else nrm.get("Z_i", 1.0))
    return BenchmarkNormalization(Nnorm=Nnorm, Tnorm_eV=Tnorm, Bnorm_T=Bnorm, m_i_amu=m_i, Z_i=z_i)


def _to_time_series(a: np.ndarray, nt: int) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim >= 1 and arr.shape[0] == nt:
        return arr
    if arr.ndim >= 1 and arr.shape[0] == 1 and nt > 1:
        return np.repeat(arr, nt, axis=0)
    if nt == 1:
        return arr[None, ...] if arr.ndim >= 2 else arr
    # fallback: trim or pad by repeating last
    if arr.ndim == 1:
        if arr.size >= nt:
            return arr[:nt]
        out = np.empty(nt, dtype=np.float64)
        out[: arr.size] = arr
        out[arr.size :] = arr[-1]
        return out
    return np.repeat(arr[-1:, ...], nt, axis=0)


def _pick_probe_series(a_t: np.ndarray) -> np.ndarray:
    arr = np.asarray(a_t, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    idx = []
    for d in arr.shape[1:]:
        idx.append(max(0, d // 2))
    # bias probe toward outboard side in radial index if available
    if len(idx) >= 1:
        idx[-2 if len(idx) >= 2 else 0] = (
            int(0.75 * (arr.shape[-2] - 1)) if len(idx) >= 2 else idx[0]
        )
    return arr[(slice(None), *idx)]


def _pick_plane(a_t: np.ndarray) -> np.ndarray:
    arr = np.asarray(a_t, dtype=np.float64)
    if arr.ndim == 3:
        return arr[-1]
    if arr.ndim == 4:
        return arr[-1, arr.shape[1] // 2]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported field rank for plane extraction: {arr.ndim}")


def _gate(diagnostics: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    passed, reason, growth, peak = finite_run_gate(
        diagnostics,
        max_growth_factor=args.max_growth_factor,
        max_rms_abs=args.max_rms_abs,
    )
    if not passed:
        raise RuntimeError(
            f"Finite-run gate failed: reason={reason} growth={growth:.3e} peak={peak:.3e}"
        )


def _bundle_from_jax(
    path: Path,
    cfg: dict[str, Any],
    norm: BenchmarkNormalization,
    args: argparse.Namespace,
) -> BenchmarkBundle:
    raw = np.load(path, allow_pickle=True)
    times = np.asarray(raw["times"] if "times" in raw else raw["t"], dtype=np.float64)
    nt = int(times.size)
    times_si = times / max(norm.omega_ci_s, 1e-30)

    fields: dict[str, np.ndarray] = {}
    for name in ("n", "Te", "omega", "phi", "vpar_i", "Ti"):
        key_snapshots = f"snapshots_{name}"
        key_snapshot = f"snapshot_{name}"
        if key_snapshots in raw:
            fields[name] = _to_time_series(raw[key_snapshots], nt)
        elif key_snapshot in raw:
            fields[name] = _to_time_series(np.asarray(raw[key_snapshot])[None, ...], nt)

    diagnostics: dict[str, np.ndarray] = {}
    snapshots: dict[str, np.ndarray] = {}

    for name in ("n", "Te", "omega", "phi"):
        if name not in fields:
            continue
        rms_total, rms_fluct, eq = compute_fluctuation_rms(fields[name], equilibrium_mode="t0")
        diagnostics[f"rms_{name}"] = rms_total
        diagnostics[f"rms_{name}_fluct"] = rms_fluct
        snapshots[f"{name}_equilibrium"] = eq
        snapshots[f"{name}_fluct_last"] = _pick_plane(fields[name] - eq[None, ...])
        snapshots[f"{name}_last"] = _pick_plane(fields[name])

    # Probe and spectral diagnostics.
    if "n" in fields:
        n_eq = snapshots["n_equilibrium"]
        n_fluct = fields["n"] - n_eq[None, ...]
        probe_n = _pick_probe_series(n_fluct)
        dt = float(np.median(np.diff(times_si))) if nt > 1 else 1.0
        f, p = compute_frequency_psd(probe_n, dt=dt, nperseg=args.nperseg)
        diagnostics["freq_hz"] = f
        diagnostics["psd_n_f"] = p
        cfg_geom = cfg.get("geometry", {}) if isinstance(cfg, dict) else {}
        ny = int(cfg_geom.get("ny", _pick_plane(n_fluct).shape[-1]))
        Ly = float(cfg_geom.get("Ly", 1.0))
        dy = Ly / max(ny, 1)
        ky, pky = compute_ky_psd(_pick_plane(n_fluct), dy=dy, axis_y=-1)
        diagnostics["ky_m-1"] = ky
        diagnostics["psd_n_ky"] = pky
        pdf_x, pdf_y = compute_pdf(_pick_plane(n_fluct), bins=args.bins)
        diagnostics["pdf_n_x"] = pdf_x
        diagnostics["pdf_n_y"] = pdf_y

    if "Te" in fields:
        Te_eq = snapshots["Te_equilibrium"]
        Te_fluct = fields["Te"] - Te_eq[None, ...]
        pdf_x, pdf_y = compute_pdf(_pick_plane(Te_fluct), bins=args.bins)
        diagnostics["pdf_Te_x"] = pdf_x
        diagnostics["pdf_Te_y"] = pdf_y

    if "n" in fields and "phi" in fields:
        n_eq = snapshots["n_equilibrium"]
        phi_eq = snapshots["phi_equilibrium"]
        n_probe = _pick_probe_series(fields["n"] - n_eq[None, ...])
        phi_probe = _pick_probe_series(fields["phi"] - phi_eq[None, ...])
        dt = float(np.median(np.diff(times_si))) if nt > 1 else 1.0
        f, coh, phase = compute_cross_coherence_phase(
            n_probe, phi_probe, dt=dt, nperseg=args.nperseg
        )
        diagnostics["coh_freq_hz"] = f
        diagnostics["coh_n_phi"] = coh
        diagnostics["phase_n_phi"] = phase

    if "n" in fields and "phi" in fields:
        cfg_geom = cfg.get("geometry", {}) if isinstance(cfg, dict) else {}
        ny = int(cfg_geom.get("ny", _pick_plane(fields["n"]).shape[-1]))
        Ly = float(cfg_geom.get("Ly", 1.0))
        dy = Ly / max(ny, 1)
        gamma_r = compute_radial_particle_flux_profile(
            _pick_plane(fields["n"]),
            _pick_plane(fields["phi"]),
            dy=dy,
            B0=norm.Bnorm_T,
            axis_y=-1,
        )
        diagnostics["gamma_r_profile"] = gamma_r

    if "n" in fields and "vpar_i" in fields and "Te" in fields:
        ti_field = fields.get("Ti", None)
        if ti_field is not None:
            te_ndim = int(np.asarray(fields["Te"]).ndim)
            ti_ndim = int(np.asarray(ti_field).ndim)
            if ti_ndim != te_ndim:
                ti_field = None
        gamma_t, qe_t, qi_t = compute_target_fluxes(
            fields["n"],
            fields["vpar_i"],
            fields["Te"],
            Ti=ti_field,
            axis_par=1,
        )
        diagnostics["target_particle_flux"] = gamma_t
        diagnostics["target_heat_flux_e"] = qe_t
        diagnostics["target_heat_flux_i"] = qi_t

    _gate(diagnostics, args)

    return BenchmarkBundle(
        code="jax_drb",
        geometry=args.geometry,
        normalization=norm,
        times_norm=times,
        times_si=times_si,
        axes={"time_norm": times, "time_s": times_si},
        diagnostics=diagnostics,
        snapshots=snapshots,
        metadata={"input": str(path), "config": str(args.config) if args.config else ""},
    )


def _read_scalar(ds, names: tuple[str, ...], default: float | None = None) -> float:
    for n in names:
        if n in ds.variables:
            v = ds.variables[n][:]
            return float(np.asarray(v).reshape(-1)[0])
    if default is None:
        raise KeyError(f"Missing any of {names}")
    return float(default)


def _read_var_interior(ds, name: str, mxg: int, myg: int, mxsub: int, mysub: int) -> np.ndarray:
    arr = np.asarray(ds.variables[name][:], dtype=np.float64)
    if arr.ndim == 4:
        return arr[:, mxg : mxg + mxsub, myg : myg + mysub, :]
    if arr.ndim == 3:
        return arr[:, mxg : mxg + mxsub, myg : myg + mysub]
    raise ValueError(f"Unsupported rank for {name}: {arr.ndim}")


def _bundle_from_hermes(
    path: Path, norm_in: BenchmarkNormalization | None, args: argparse.Namespace
):
    try:
        from netCDF4 import Dataset
    except Exception as e:
        raise RuntimeError("netCDF4 is required to build Hermes bundles.") from e

    files = sorted(path.glob("BOUT.dmp.*.nc"))
    if not files:
        raise FileNotFoundError(f"No BOUT.dmp.*.nc files found in {path}")

    with Dataset(str(files[0])) as ds0:
        times = np.asarray(ds0.variables["t"][:], dtype=np.float64)
        mxg = int(_read_scalar(ds0, ("MXG",), 2))
        myg = int(_read_scalar(ds0, ("MYG",), 2))
        mxsub = int(_read_scalar(ds0, ("MXSUB",), ds0.variables["Ne"].shape[1] - 2 * mxg))
        mysub = int(_read_scalar(ds0, ("MYSUB",), ds0.variables["Ne"].shape[2] - 2 * myg))
        nxpe = int(_read_scalar(ds0, ("NXPE", "MYPE_X", "PE_NX"), 1))
        nype = int(
            _read_scalar(ds0, ("NYPE", "MYPE_Y", "PE_NY"), max(1, len(files) // max(nxpe, 1)))
        )
        nnorm = _read_scalar(ds0, ("Nnorm",), norm_in.Nnorm if norm_in else 1.0e19)
        tnorm = _read_scalar(ds0, ("Tnorm",), norm_in.Tnorm_eV if norm_in else 50.0)
        bnorm = _read_scalar(ds0, ("Bnorm",), norm_in.Bnorm_T if norm_in else 1.0)
        omega_ci = _read_scalar(ds0, ("Omega_ci",), None)
        dx_med = (
            float(np.nanmedian(np.asarray(ds0.variables["dx"][:])))
            if "dx" in ds0.variables
            else 1.0
        )
        dy_med = (
            float(np.nanmedian(np.asarray(ds0.variables["dy"][:])))
            if "dy" in ds0.variables
            else 1.0
        )
        var_names = set(ds0.variables.keys())

    norm = (
        norm_in
        if norm_in is not None
        else BenchmarkNormalization(
            Nnorm=float(nnorm),
            Tnorm_eV=float(tnorm),
            Bnorm_T=float(bnorm),
            m_i_amu=float(args.m_i_amu),
            Z_i=float(args.Z_i),
        )
    )
    if np.isfinite(omega_ci):
        times_si = times / max(float(omega_ci), 1e-30)
    else:
        times_si = times / max(norm.omega_ci_s, 1e-30)

    nt = int(times.size)
    nx = int(nxpe * mxsub)
    ny = int(nype * mysub)
    nz = None

    def alloc(name: str) -> np.ndarray:
        nonlocal nz
        with Dataset(str(files[0])) as ds:
            arr = _read_var_interior(ds, name, mxg, myg, mxsub, mysub)
            if arr.ndim == 4:
                nz = arr.shape[-1]
                return np.zeros((nt, nx, ny, nz), dtype=np.float64)
            return np.zeros((nt, nx, ny), dtype=np.float64)

    for required in ("Ne", "Te", "Vort", "phi"):
        if required not in var_names:
            raise KeyError(f"Hermes dump missing required variable '{required}'")

    fields_global: dict[str, np.ndarray] = {
        "n": alloc("Ne"),
        "Te": alloc("Te"),
        "omega": alloc("Vort"),
        "phi": alloc("phi"),
    }

    ion_density = None
    ion_momentum = None
    if "Nd+" in var_names and "NVd+" in var_names:
        ion_density, ion_momentum = "Nd+", "NVd+"
    else:
        nv_plus = sorted([v for v in var_names if v.startswith("NV") and v.endswith("+")])
        for nv in nv_plus:
            n_candidate = "N" + nv[2:]
            if n_candidate in var_names:
                ion_density, ion_momentum = n_candidate, nv
                break
    if ion_density and ion_momentum:
        fields_global["vpar_i"] = alloc(ion_momentum)
        fields_global["_ion_n"] = alloc(ion_density)

    for local_rank, fp in enumerate(files):
        with Dataset(str(fp)) as ds:
            pe_x = int(_read_scalar(ds, ("PE_XIND", "PE_XINDICES"), local_rank % max(nxpe, 1)))
            pe_y = int(_read_scalar(ds, ("PE_YIND", "PE_YINDICES"), local_rank // max(nxpe, 1)))
            x0 = pe_x * mxsub
            y0 = pe_y * mysub
            x1 = x0 + mxsub
            y1 = y0 + mysub
            fields_global["n"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                ds, "Ne", mxg, myg, mxsub, mysub
            )
            fields_global["Te"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                ds, "Te", mxg, myg, mxsub, mysub
            )
            fields_global["omega"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                ds, "Vort", mxg, myg, mxsub, mysub
            )
            fields_global["phi"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                ds, "phi", mxg, myg, mxsub, mysub
            )
            if ion_density and ion_momentum:
                fields_global["vpar_i"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                    ds, ion_momentum, mxg, myg, mxsub, mysub
                )
                fields_global["_ion_n"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                    ds, ion_density, mxg, myg, mxsub, mysub
                )

    if "vpar_i" in fields_global:
        fields_global["vpar_i"] = fields_global["vpar_i"] / np.maximum(
            fields_global["_ion_n"], 1e-12
        )
        fields_global.pop("_ion_n", None)

    diagnostics: dict[str, np.ndarray] = {}
    snapshots: dict[str, np.ndarray] = {}
    for name in ("n", "Te", "omega", "phi"):
        rms_total, rms_fluct, eq = compute_fluctuation_rms(
            fields_global[name], equilibrium_mode="t0"
        )
        diagnostics[f"rms_{name}"] = rms_total
        diagnostics[f"rms_{name}_fluct"] = rms_fluct
        snapshots[f"{name}_equilibrium"] = eq
        snapshots[f"{name}_last"] = fields_global[name][-1]
        snapshots[f"{name}_fluct_last"] = fields_global[name][-1] - eq

    n_fluct = fields_global["n"] - snapshots["n_equilibrium"][None, ...]
    probe_n = _pick_probe_series(n_fluct)
    dt = float(np.median(np.diff(times_si))) if nt > 1 else 1.0
    f, p = compute_frequency_psd(probe_n, dt=dt, nperseg=args.nperseg)
    diagnostics["freq_hz"] = f
    diagnostics["psd_n_f"] = p
    ky, pky = compute_ky_psd(_pick_plane(n_fluct), dy=dy_med, axis_y=-1)
    diagnostics["ky_m-1"] = ky
    diagnostics["psd_n_ky"] = pky
    pdf_x, pdf_y = compute_pdf(_pick_plane(n_fluct), bins=args.bins)
    diagnostics["pdf_n_x"] = pdf_x
    diagnostics["pdf_n_y"] = pdf_y
    Te_fluct = fields_global["Te"] - snapshots["Te_equilibrium"][None, ...]
    pdf_x, pdf_y = compute_pdf(_pick_plane(Te_fluct), bins=args.bins)
    diagnostics["pdf_Te_x"] = pdf_x
    diagnostics["pdf_Te_y"] = pdf_y
    n_probe = _pick_probe_series(n_fluct)
    phi_probe = _pick_probe_series(fields_global["phi"] - snapshots["phi_equilibrium"][None, ...])
    f, coh, phase = compute_cross_coherence_phase(n_probe, phi_probe, dt=dt, nperseg=args.nperseg)
    diagnostics["coh_freq_hz"] = f
    diagnostics["coh_n_phi"] = coh
    diagnostics["phase_n_phi"] = phase
    diagnostics["gamma_r_profile"] = compute_radial_particle_flux_profile(
        _pick_plane(fields_global["n"]),
        _pick_plane(fields_global["phi"]),
        dy=dy_med,
        B0=norm.Bnorm_T,
    )
    if "vpar_i" in fields_global:
        gamma_t, qe_t, qi_t = compute_target_fluxes(
            fields_global["n"],
            fields_global["vpar_i"],
            fields_global["Te"],
            Ti=None,
            axis_par=3 if fields_global["n"].ndim == 4 else 1,
        )
        diagnostics["target_particle_flux"] = gamma_t
        diagnostics["target_heat_flux_e"] = qe_t
        diagnostics["target_heat_flux_i"] = qi_t

    _gate(diagnostics, args)

    return BenchmarkBundle(
        code="hermes",
        geometry=args.geometry,
        normalization=norm,
        times_norm=times,
        times_si=times_si,
        axes={
            "time_norm": times,
            "time_s": times_si,
            "x_index": np.arange(nx, dtype=np.float64),
            "y_index": np.arange(ny, dtype=np.float64),
            "z_index": np.arange(int(nz if nz is not None else 1), dtype=np.float64),
        },
        diagnostics=diagnostics,
        snapshots=snapshots,
        metadata={"input": str(path), "dx": dx_med, "dy": dy_med},
    )


def main() -> None:
    args = _parse_args()
    cfg = _load_toml(args.config)
    norm = _norm_from_cfg_or_args(cfg, args)
    inp = Path(args.input).resolve()
    if args.code == "jax":
        bundle = _bundle_from_jax(inp, cfg, norm, args)
    else:
        bundle = _bundle_from_hermes(inp, norm, args)
    out = save_bundle_npz(bundle, args.output)
    print(f"Saved benchmark bundle: {out}")


if __name__ == "__main__":
    main()
