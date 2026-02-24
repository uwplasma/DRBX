from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.benchmarking import load_bundle_npz


def _as_plane(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[:, :, arr.shape[2] // 2]
    raise ValueError(f"Expected 2D/3D snapshot, got shape={arr.shape}")


def _pick_snapshot(bundle, field: str, use_fluct: bool) -> np.ndarray:
    key = f"{field}_fluct_last" if use_fluct else f"{field}_last"
    if key not in bundle.snapshots:
        alt = f"{field}_last"
        if alt not in bundle.snapshots:
            raise KeyError(f"Missing snapshot key '{key}' and '{alt}'")
        key = alt
    return _as_plane(bundle.snapshots[key])


def _shared_symmetric_range(a: np.ndarray, b: np.ndarray, q: float = 99.5) -> tuple[float, float]:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    v = np.concatenate([av, bv])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    s = float(np.percentile(np.abs(v), q))
    if s <= 0.0:
        s = 1.0
    return -s, s


def _line(ax, t: np.ndarray, y: np.ndarray, label: str, style: str, color: str) -> None:
    if t.size == 0 or y.size == 0:
        return
    ax.plot(t, y, style, lw=2.0, color=color, label=label)


def _diag(bundle, key: str) -> np.ndarray | None:
    if key not in bundle.diagnostics:
        return None
    return np.asarray(bundle.diagnostics[key], dtype=np.float64)


def _save_summary_csv(path: Path, rows: list[tuple[str, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "hermes_end", "jax_end", "rel_end_error"])
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical Hermes vs jax_drb benchmark panel.")
    parser.add_argument("--hermes", required=True, help="Hermes benchmark bundle (.npz).")
    parser.add_argument("--jax", required=True, help="jax_drb benchmark bundle (.npz).")
    parser.add_argument("--out", required=True, help="Output PNG panel path.")
    parser.add_argument("--summary-csv", default="", help="Optional summary CSV path.")
    parser.add_argument("--field", default="n", choices=("n", "Te", "omega", "phi"))
    parser.add_argument(
        "--snapshot-mode",
        choices=("fluct", "total"),
        default="fluct",
        help="Use fluctuation snapshot (default) or total field snapshot.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    hermes = load_bundle_npz(args.hermes)
    jax = load_bundle_npz(args.jax)

    use_fluct = args.snapshot_mode == "fluct"
    h_snap = _pick_snapshot(hermes, args.field, use_fluct)
    j_snap = _pick_snapshot(jax, args.field, use_fluct)

    h_times = np.asarray(hermes.times_norm, dtype=np.float64)
    j_times = np.asarray(jax.times_norm, dtype=np.float64)

    fig = plt.figure(figsize=(16.0, 12.0))
    gs = fig.add_gridspec(3, 3)

    # Row 1: side-by-side snapshot + difference with shared range.
    vmin, vmax = _shared_symmetric_range(h_snap, j_snap)
    cmap = "coolwarm" if use_fluct else "viridis"
    suffix = "'" if use_fluct else ""

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(h_snap.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(f"Hermes {args.field}{suffix}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(j_snap.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(f"jax_drb {args.field}{suffix}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 2])
    if h_snap.shape == j_snap.shape:
        diff = j_snap - h_snap
        dvmin, dvmax = _shared_symmetric_range(diff, diff)
        im = ax.imshow(
            diff.T,
            origin="lower",
            cmap="coolwarm",
            vmin=dvmin,
            vmax=dvmax,
            aspect="auto",
        )
        ax.set_title("jax_drb - Hermes")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.axis("off")
        ax.set_title("difference unavailable\n(shape mismatch)")

    # Row 2: RMS fluct, ky PSD, f PSD
    ax = fig.add_subplot(gs[1, 0])
    colors = {
        "n": "#1f77b4",
        "Te": "#ff7f0e",
        "omega": "#2ca02c",
        "phi": "#d62728",
    }
    summary_rows: list[tuple[str, float, float, float]] = []
    for field in ("n", "Te", "omega", "phi"):
        key = f"rms_{field}_fluct"
        hy = _diag(hermes, key)
        jy = _diag(jax, key)
        if hy is None or jy is None:
            continue
        _line(ax, h_times, hy, f"Hermes {field}", "--", colors[field])
        _line(ax, j_times, jy, f"jax_drb {field}", "-", colors[field])
        jy_interp = np.interp(h_times, j_times, jy)
        rel = float((jy_interp[-1] - hy[-1]) / (abs(hy[-1]) + 1e-12))
        summary_rows.append((field, float(hy[-1]), float(jy_interp[-1]), rel))
    ax.set_title("Fluctuation RMS")
    ax.set_xlabel("t (normalized)")
    ax.set_ylabel("RMS")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = fig.add_subplot(gs[1, 1])
    h_ky = _diag(hermes, "ky_m-1")
    h_pky = _diag(hermes, "psd_n_ky")
    j_ky = _diag(jax, "ky_m-1")
    j_pky = _diag(jax, "psd_n_ky")
    if h_ky is not None and h_pky is not None and j_ky is not None and j_pky is not None:
        ax.loglog(h_ky[1:], np.maximum(h_pky[1:], 1e-30), "--", lw=2, label="Hermes")
        ax.loglog(j_ky[1:], np.maximum(j_pky[1:], 1e-30), "-", lw=1.8, label="jax_drb")
    ax.set_title(r"$k_y$ PSD (n')")
    ax.set_xlabel(r"$k_y$ [m$^{-1}$]")
    ax.set_ylabel("PSD")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    h_f = _diag(hermes, "freq_hz")
    h_pf = _diag(hermes, "psd_n_f")
    j_f = _diag(jax, "freq_hz")
    j_pf = _diag(jax, "psd_n_f")
    if h_f is not None and h_pf is not None and j_f is not None and j_pf is not None:
        ax.loglog(h_f[1:], np.maximum(h_pf[1:], 1e-30), "--", lw=2, label="Hermes")
        ax.loglog(j_f[1:], np.maximum(j_pf[1:], 1e-30), "-", lw=1.8, label="jax_drb")
    ax.set_title("Frequency PSD (n')")
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("PSD")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8)

    # Row 3: PDFs, coherence/phase, radial flux profile.
    ax = fig.add_subplot(gs[2, 0])
    h_px = _diag(hermes, "pdf_n_x")
    h_py = _diag(hermes, "pdf_n_y")
    j_px = _diag(jax, "pdf_n_x")
    j_py = _diag(jax, "pdf_n_y")
    if h_px is not None and h_py is not None and j_px is not None and j_py is not None:
        ax.plot(h_px, h_py, "--", lw=2, label="Hermes")
        ax.plot(j_px, j_py, "-", lw=1.8, label="jax_drb")
    ax.set_title("PDF(n')")
    ax.set_xlabel("n'")
    ax.set_ylabel("PDF")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    h_cf = _diag(hermes, "coh_freq_hz")
    h_coh = _diag(hermes, "coh_n_phi")
    h_phase = _diag(hermes, "phase_n_phi")
    j_cf = _diag(jax, "coh_freq_hz")
    j_coh = _diag(jax, "coh_n_phi")
    j_phase = _diag(jax, "phase_n_phi")
    if h_cf is not None and h_coh is not None and j_cf is not None and j_coh is not None:
        ax.semilogx(h_cf[1:], h_coh[1:], "--", lw=2, label="Hermes coherence")
        ax.semilogx(j_cf[1:], j_coh[1:], "-", lw=1.8, label="jax_drb coherence")
        ax2 = ax.twinx()
        if h_phase is not None and j_phase is not None:
            ax2.semilogx(h_cf[1:], h_phase[1:], "--", lw=1.5, color="#9467bd", alpha=0.8)
            ax2.semilogx(j_cf[1:], j_phase[1:], "-", lw=1.2, color="#9467bd", alpha=0.8)
            ax2.set_ylabel("phase [rad]", color="#9467bd")
    ax.set_title("Cross coherence/phase (n',phi')")
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("coherence")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[2, 2])
    h_g = _diag(hermes, "gamma_r_profile")
    j_g = _diag(jax, "gamma_r_profile")
    if h_g is not None and j_g is not None:
        ax.plot(np.arange(h_g.size), h_g, "--", lw=2, label="Hermes")
        ax.plot(np.arange(j_g.size), j_g, "-", lw=1.8, label="jax_drb")
    ax.set_title(r"Radial flux $\Gamma_r(x)$")
    ax.set_xlabel("radial index")
    ax.set_ylabel(r"$\Gamma_r$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Canonical Hermes vs jax_drb benchmark panel", fontsize=15)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi))
    print(f"Saved benchmark panel: {out}")

    if args.summary_csv:
        csv_path = Path(args.summary_csv)
        _save_summary_csv(csv_path, summary_rows)
        print(f"Saved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
