from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from jaxdrb.analysis.lp import LpFixedPointResult, solve_lp_fixed_point
from jaxdrb.analysis.scan import Scan1DResult, scan_ky
from jaxdrb.geometry.tabulated import TabulatedGeometry
from jaxdrb.models.params import DRBParams


def _require_pyqsc():
    try:
        from qsc import Qsc  # type: ignore

        return Qsc
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "This example requires pyQSC.\n\n"
            "Options:\n"
            "  - If you have a local checkout (as in this repo), run with:\n"
            "      PYTHONPATH=../pyQSC-main python examples/run_pyqsc_stellarator.py\n"
            "  - Or install pyQSC (package name may be `qsc`) and try again.\n\n"
            f"Import error: {exc}"
        ) from exc


def _central_diff_periodic(x: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(x, -1, axis=0) - np.roll(x, 1, axis=0)) / (2.0 * dx)


@dataclass
class NearAxisGeomConfig:
    qsc_config: str = "r1 section 5.1"
    r: float = 0.1
    alpha: float = 0.0
    nl: int = 256
    dr: float = 1e-3
    dalpha: float = 2e-3
    max_turns: int = 24
    turns: int | None = None


def build_tabulated_geometry_from_pyqsc(
    cfg: NearAxisGeomConfig, out_npz: Path
) -> TabulatedGeometry:
    """Build a TabulatedGeometry by finite-differencing the near-axis mapping from pyQSC.

    Notes:
      - We compute the mapping x(r, theta, phi) using pyQSC's Fourier-based surface builder
        (`get_boundary`) and evaluate points with smooth bivariate splines. This is much faster
        than calling `to_RZ` in a Python loop.
      - We select a toroidal domain length L = 2π * turns, choosing `turns` so that iota*turns ≈ integer,
        which makes the field-line mapping approximately periodic and therefore compatible with the
        periodic parallel derivative used in v1 of `jaxdrb`.
    """

    Qsc = _require_pyqsc()
    qsc = Qsc.from_paper(cfg.qsc_config)

    from scipy.interpolate import RectBivariateSpline  # type: ignore

    iota = float(qsc.iota)
    if cfg.turns is None:
        # Choose a number of toroidal turns that approximately closes the field line:
        # want iota * turns ~ integer.
        best_turns = None
        for turns in range(1, int(cfg.max_turns) + 1):
            mismatch = abs(iota * turns - round(iota * turns))
            cand = (mismatch, turns)
            if best_turns is None or cand < best_turns:
                best_turns = cand
        assert best_turns is not None
        turns = best_turns[1]
    else:
        turns = int(cfg.turns)

    L = 2.0 * np.pi * turns
    l = np.linspace(0.0, L, cfg.nl, endpoint=False)
    dl = float(l[1] - l[0])
    phi = np.mod(l, 2.0 * np.pi)
    varphi = l + qsc.nu_spline(phi)

    theta0 = cfg.alpha + qsc.iota * varphi
    theta_p = theta0 + cfg.dalpha
    theta_m = theta0 - cfg.dalpha

    def surface_splines(r: float):
        x2d, y2d, z2d, _R2d = qsc.get_boundary(r=r, ntheta=96, nphi=144, ntheta_fourier=24)
        # get_boundary returns arrays shaped (theta, phi) on [0,2π)×[0,2π). We extend with periodic
        # endpoints so spline evaluation near 2π is well-behaved.
        x2d = np.concatenate([x2d, x2d[:1, :]], axis=0)
        y2d = np.concatenate([y2d, y2d[:1, :]], axis=0)
        z2d = np.concatenate([z2d, z2d[:1, :]], axis=0)
        x2d = np.concatenate([x2d, x2d[:, :1]], axis=1)
        y2d = np.concatenate([y2d, y2d[:, :1]], axis=1)
        z2d = np.concatenate([z2d, z2d[:, :1]], axis=1)

        theta_grid = np.linspace(0.0, 2.0 * np.pi, x2d.shape[0], endpoint=True)
        phi_grid = np.linspace(0.0, 2.0 * np.pi, x2d.shape[1], endpoint=True)

        sx = RectBivariateSpline(phi_grid, theta_grid, x2d.T)
        sy = RectBivariateSpline(phi_grid, theta_grid, y2d.T)
        sz = RectBivariateSpline(phi_grid, theta_grid, z2d.T)
        return sx, sy, sz

    sx0, sy0, sz0 = surface_splines(cfg.r)
    sxp, syp, szp = surface_splines(cfg.r + cfg.dr)
    sxm, sym, szm = surface_splines(cfg.r - cfg.dr)

    def eval_xyz(sx, sy, sz, theta: np.ndarray) -> np.ndarray:
        th = np.mod(theta, 2.0 * np.pi)
        X = sx.ev(phi, th)
        Y = sy.ev(phi, th)
        Z = sz.ev(phi, th)
        return np.stack([X, Y, Z], axis=1)

    x0 = eval_xyz(sx0, sy0, sz0, theta0)
    x_rp = eval_xyz(sxp, syp, szp, theta0)
    x_rm = eval_xyz(sxm, sym, szm, theta0)
    x_ap = eval_xyz(sx0, sy0, sz0, theta_p)
    x_am = eval_xyz(sx0, sy0, sz0, theta_m)

    e_r = (x_rp - x_rm) / (2.0 * cfg.dr)
    e_a = (x_ap - x_am) / (2.0 * cfg.dalpha)
    e_l = _central_diff_periodic(x0, dl)

    J = np.einsum("ij,ij->i", e_r, np.cross(e_a, e_l))
    grad_r = np.cross(e_a, e_l) / J[:, None]
    grad_a = np.cross(e_l, e_r) / J[:, None]
    grad_l = np.cross(e_r, e_a) / J[:, None]

    gxx = np.einsum("ij,ij->i", grad_r, grad_r)
    gxy = np.einsum("ij,ij->i", grad_r, grad_a)
    gyy = np.einsum("ij,ij->i", grad_a, grad_a)

    def Bmag(r: float, theta: np.ndarray) -> np.ndarray:
        return np.asarray(qsc.B_mag(r, theta, varphi, Boozer_toroidal=True), dtype=float)

    B0 = Bmag(cfg.r, theta0)
    dB_dr = (Bmag(cfg.r + cfg.dr, theta0) - Bmag(cfg.r - cfg.dr, theta0)) / (2.0 * cfg.dr)
    dB_da = (Bmag(cfg.r, theta_p) - Bmag(cfg.r, theta_m)) / (2.0 * cfg.dalpha)
    dB_dl = _central_diff_periodic(B0, dl)

    gradB = dB_dr[:, None] * grad_r + dB_da[:, None] * grad_a + dB_dl[:, None] * grad_l
    b = e_l / np.linalg.norm(e_l, axis=1)[:, None]
    b_x_gradB = np.cross(b, gradB)

    curv_x = 2.0 * np.einsum("ij,ij->i", b_x_gradB, grad_r) / (B0**2)
    curv_y = 2.0 * np.einsum("ij,ij->i", b_x_gradB, grad_a) / (B0**2)

    dpar_factor = 1.0 / np.linalg.norm(e_l, axis=1)

    np.savez(
        out_npz,
        l=l,
        gxx=gxx,
        gxy=gxy,
        gyy=gyy,
        curv_x=curv_x,
        curv_y=curv_y,
        dpar_factor=dpar_factor,
        B=B0,
        meta=json.dumps(
            {
                "qsc_config": cfg.qsc_config,
                "r": cfg.r,
                "alpha": cfg.alpha,
                "nl": cfg.nl,
                "dr": cfg.dr,
                "dalpha": cfg.dalpha,
                "turns": turns,
                "iota": iota,
            },
            indent=2,
            sort_keys=True,
        ),
    )
    return TabulatedGeometry.from_npz(out_npz)


def _save_scan(out_dir: Path, name: str, scan: Scan1DResult) -> None:
    np.savez(
        out_dir / f"{name}.npz",
        ky=scan.ky,
        gamma_eigs=scan.gamma_eigs,
        omega_eigs=scan.omega_eigs,
        gamma_iv=scan.gamma_iv if scan.gamma_iv is not None else np.nan,
        omega_iv=scan.omega_iv if scan.omega_iv is not None else np.nan,
        eigs=scan.eigs,
    )


def _save_lp(out_dir: Path, name: str, res: LpFixedPointResult) -> None:
    np.savez(out_dir / f"{name}.npz", **asdict(res))


def main() -> None:
    out_dir = Path("out_pyqsc_stellarator_example")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))

    cfg = NearAxisGeomConfig(qsc_config="r1 section 5.1", r=0.1, alpha=0.0, nl=192)
    geom_npz = out_dir / "pyqsc_geom.npz"
    geom = build_tabulated_geometry_from_pyqsc(cfg, geom_npz)

    # --- Parameter choices (dimensionless / qualitative) ---
    # The absolute scaling in this v1 example is not meant to be quantitatively predictive;
    # it demonstrates how to ingest near-axis geometry and run the same linear workflows.
    params_resistive = DRBParams(
        omega_n=0.6,
        omega_Te=0.0,
        eta=2.0,
        me_hat=2e-3,  # small inertia -> resistive-like branch
        curvature_on=True,
        Dn=0.02,
        DOmega=0.02,
        DTe=0.02,
    )
    params_inertial = DRBParams(
        omega_n=0.6,
        omega_Te=0.0,
        eta=0.05,
        me_hat=0.5,  # larger inertia -> inertial-like branch
        curvature_on=True,
        Dn=0.02,
        DOmega=0.02,
        DTe=0.02,
    )

    ky = np.linspace(0.03, 0.20, 16)

    scan_res = scan_ky(
        params_resistive,
        geom,
        ky=ky,
        kx=0.0,
        arnoldi_m=30,
        nev=4,
        do_initial_value=False,
        seed=0,
    )
    scan_in = scan_ky(
        params_inertial,
        geom,
        ky=ky,
        kx=0.0,
        arnoldi_m=30,
        nev=4,
        do_initial_value=False,
        seed=1,
    )

    _save_scan(out_dir, "scan_resistive", scan_res)
    _save_scan(out_dir, "scan_inertial", scan_in)

    # Estimate Lp via the Halpern fixed-point rule Lp = q (gamma/ky)_max.
    # We map omega_n ~ 1/Lp (see docs/literature/sol_width.md).
    lp_res = solve_lp_fixed_point(
        params_resistive,
        geom,
        q=3.0,  # an effective safety factor-like parameter for comparison
        ky=ky,
        Lp0=20.0,
        omega_n_scale=1.0,
        relax=0.6,
        max_iter=15,
        arnoldi_m=30,
        nev=4,
        seed=2,
    )
    _save_lp(out_dir, "lp_fixed_point", lp_res)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ky, scan_res.gamma_eigs, "o-", label="resistive-like")
    ax.plot(ky, scan_in.gamma_eigs, "s--", label="inertial-like")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_ky_branches.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ratio = scan_res.gamma_eigs / ky
    ax.plot(ky, ratio, "o-")
    ax.axvline(lp_res.ky_star, color="k", alpha=0.35, linestyle="--")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma/k_y$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_over_ky.png", dpi=160)
    plt.close(fig)

    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "near_axis_geom": asdict(cfg),
                "params_resistive": asdict(params_resistive),
                "params_inertial": asdict(params_inertial),
                "Lp_fixed_point": {
                    "Lp": lp_res.Lp,
                    "ky_star": lp_res.ky_star,
                    "gamma_star": lp_res.gamma_star,
                    "gamma_over_ky_star": lp_res.gamma_over_ky_star,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"Wrote {out_dir / 'pyqsc_geom.npz'}")
    print(f"Wrote {out_dir / 'gamma_ky_branches.png'}")
    print(f"Wrote {out_dir / 'gamma_over_ky.png'}")
    print(f"Wrote {out_dir / 'lp_fixed_point.npz'}")


if __name__ == "__main__":
    main()
