"""FCI DRB3D full model on an ESSOS Biot-Savart field (LandremanPaulQA coils).

This example demonstrates an end-to-end workflow:

1) load a Biot-Savart magnetic field from ESSOS coils,
2) estimate a magnetic-axis location in the phi=0 plane (coarse BR/BZ root proxy),
3) build toroidal-plane FCI maps in a local edge/SOL patch,
4) run the full 3D DRB milestone model with sheath closure + hot-ion + EM + neutrals,
5) generate a diagnostics panel and save metrics.

The plasma boundary is not explicitly provided by the coils-only model. We therefore use a
pragmatic local patch around the estimated axis and interpret the patch edges as open targets
for this demonstration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.bc import BC2D
from jaxdrb.fci.builder import (
    EssosToroidalFCIConfig,
    build_fci_maps_essos_toroidal_planes,
)
from jaxdrb.fci.drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps
from jaxdrb.nonlinear.neutrals import NeutralParams


def _default_coils_file() -> Path | None:
    candidates = [
        Path(
            "/Users/rogerio/local/ESSOS/examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json"
        ),
    ]
    try:
        import essos  # type: ignore

        roots = list(getattr(essos, "__path__", []))
        for root in roots:
            p = Path(root).resolve()
            candidates.append(
                p / "examples" / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json"
            )
            candidates.append(
                p.parent / "examples" / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json"
            )
    except Exception:
        pass
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _cylindrical_B(field, R: float, Z: float, phi: float = 0.0) -> tuple[float, float, float]:
    xyz = np.asarray([R * np.cos(phi), R * np.sin(phi), Z], dtype=float)
    Bx, By, Bz = np.asarray(field.B(xyz), dtype=float)
    BR = Bx * np.cos(phi) + By * np.sin(phi)
    Bphi = -Bx * np.sin(phi) + By * np.cos(phi)
    return float(BR), float(Bphi), float(Bz)


def _estimate_axis(
    field, *, R_min: float, R_max: float, Z_min: float, Z_max: float
) -> tuple[float, float]:
    """Coarse axis estimate via minimization of BR^2+BZ^2 at phi=0."""
    Rs = np.linspace(R_min, R_max, 80)
    Zs = np.linspace(Z_min, Z_max, 80)
    best = (1e30, 0.5 * (R_min + R_max), 0.5 * (Z_min + Z_max))
    for R in Rs:
        for Z in Zs:
            BR, _, BZ = _cylindrical_B(field, float(R), float(Z), phi=0.0)
            score = BR * BR + BZ * BZ
            if score < best[0]:
                best = (score, float(R), float(Z))
    return best[1], best[2]


def make_model_from_essos(
    *,
    coils_json: Path,
    nphi: int,
    nR: int,
    nZ: int,
    dphi: float,
    radial_offset: float,
    radial_width: float,
    vertical_halfwidth: float,
    dl_min: float,
    sheath_model: str,
    stable_gate: bool = False,
) -> tuple[FCIDRB3DFullModel, FCIDRB3DFullState, dict[str, float]]:
    from essos.coils import Coils_from_json  # type: ignore
    from essos.fields import BiotSavart  # type: ignore

    field = BiotSavart(Coils_from_json(str(coils_json)))
    R_axis, Z_axis = _estimate_axis(field, R_min=1.05, R_max=1.45, Z_min=-0.35, Z_max=0.35)

    R_min = R_axis + radial_offset
    R_max = R_min + radial_width
    Z_min = Z_axis - vertical_halfwidth
    Z_max = Z_axis + vertical_halfwidth
    dR = (R_max - R_min) / float(max(nR - 1, 1))
    dZ = (Z_max - Z_min) / float(max(nZ - 1, 1))

    cfg = EssosToroidalFCIConfig(
        R0=float(R_min),
        Z0=float(Z_min),
        dR=float(dR),
        dZ=float(dZ),
        nR=int(nR),
        nZ=int(nZ),
        phi0=0.0,
        dphi=float(dphi),
        nphi=int(nphi),
        open_field_line=True,
        cell_centered=True,
        periodic_R=False,
        periodic_Z=False,
        periodic_phi=True,
        R_min=float(R_min),
        R_max=float(R_max),
        Z_min=float(Z_min),
        Z_max=float(Z_max),
    )
    map_fwd, map_bwd, map_meta = build_fci_maps_essos_toroidal_planes(
        cfg, field=field, nsub=10, dl_min=float(dl_min)
    )
    l = cfg.phi0 + cfg.dphi * jnp.arange(cfg.nphi)
    grid = FCISlabGrid.from_maps(
        x0=cfg.R0,
        y0=cfg.Z0,
        dx=cfg.dR,
        dy=cfg.dZ,
        nx=cfg.nR,
        ny=cfg.nZ,
        l=l,
        map_fwd=map_fwd,
        map_bwd=map_bwd,
        open_field_line=True,
        cell_centered=True,
    )

    if stable_gate:
        params = FCIDRB3DFullParams(
            omega_n=0.0,
            omega_Te=0.0,
            omega_Ti=0.0,
            kappa=0.0,
            alpha=0.0,
            eta_par=0.0,
            me_hat=0.5,
            Dn=1.0e-2,
            DOmega=1.0e-2,
            Dvpar=1.0e-2,
            DTe=1.0e-2,
            DTi=1.0e-2,
            Dpsi=1.0e-2,
            chi_par=1.0e-2,
            hot_ion_on=False,
            tau_i=0.6,
            em_on=False,
            beta=0.0,
            neutrals_on=False,
            neutrals=NeutralParams(enabled=False),
            sheath_on=True,
            sheath_bc_model=sheath_model,
            sheath_nu_mom=0.3,
            sheath_nu_particle=0.12,
            sheath_nu_energy=0.08,
            sheath_gamma_e=3.2,
            sheath_gamma_i=3.0,
            bracket="arakawa",
            perp_operator="fd",
            perp_bc=BC2D.dirichlet(),
            perp_bc_nu=0.1,
        )
        amp = 2.0e-5
    else:
        params = FCIDRB3DFullParams(
            omega_n=0.06,
            omega_Te=0.03,
            omega_Ti=0.02,
            kappa=0.08,
            alpha=0.03,
            eta_par=0.03,
            me_hat=0.5,
            Dn=1.2e-3,
            DOmega=1.6e-3,
            Dvpar=1.2e-3,
            DTe=1.2e-3,
            DTi=1.2e-3,
            Dpsi=1.0e-3,
            chi_par=1.2e-3,
            hot_ion_on=True,
            tau_i=0.6,
            em_on=True,
            beta=0.02,
            neutrals_on=True,
            neutrals=NeutralParams(
                enabled=True,
                Dn0=6e-4,
                nu_ion=1.0e-3,
                nu_rec=8.0e-4,
                n_background=0.2,
                nu_cx_omega=1.0e-3,
            ),
            sheath_on=True,
            sheath_bc_model=sheath_model,
            sheath_nu_mom=0.22,
            sheath_nu_particle=0.08,
            sheath_nu_energy=0.05,
            sheath_gamma_e=3.2,
            sheath_gamma_i=3.0,
            bracket="arakawa",
            perp_operator="fd",
        )
        amp = 4.0e-4
    model = FCIDRB3DFullModel(params=params, grid=grid)

    shape = (grid.nz, grid.nx, grid.ny)
    key = jax.random.key(503)
    k = jax.random.split(key, 8)
    hot_on = bool(params.hot_ion_on)
    em_on = bool(params.em_on)
    neut_on = bool(params.neutrals_on and params.neutrals.enabled)
    y0 = FCIDRB3DFullState(
        n=amp * jax.random.normal(k[0], shape),
        omega=amp * jax.random.normal(k[1], shape),
        vpar_e=amp * jax.random.normal(k[2], shape),
        vpar_i=amp * jax.random.normal(k[3], shape),
        Te=amp * jax.random.normal(k[4], shape),
        Ti=amp * jax.random.normal(k[5], shape) if hot_on else None,
        psi=amp * jax.random.normal(k[6], shape) if em_on else None,
        N=0.02 + amp * jax.random.normal(k[7], shape) if neut_on else None,
    )
    meta = {
        "R_axis_est": float(R_axis),
        "Z_axis_est": float(Z_axis),
        "R_patch_min": float(R_min),
        "R_patch_max": float(R_max),
        "Z_patch_min": float(Z_min),
        "Z_patch_max": float(Z_max),
        **{
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in map_meta.items()
        },
        "dl_min": float(dl_min),
    }
    return model, y0, meta


def main() -> None:
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coils-json", type=str, default="")
    parser.add_argument("--out", type=str, default="out_fci_essos_biotsavart")
    parser.add_argument("--nphi", type=int, default=10)
    parser.add_argument("--nR", type=int, default=12)
    parser.add_argument("--nZ", type=int, default=12)
    parser.add_argument("--dphi", type=float, default=0.10)
    parser.add_argument("--radial-offset", type=float, default=0.08)
    parser.add_argument("--radial-width", type=float, default=0.18)
    parser.add_argument("--vertical-halfwidth", type=float, default=0.14)
    parser.add_argument("--dl-min", type=float, default=5e-2)
    parser.add_argument("--sheath-model", choices=["simple", "loizu_linear"], default="simple")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--nsteps", type=int, default=600)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    coils_json = Path(args.coils_json) if args.coils_json else _default_coils_file()
    if coils_json is None or not coils_json.exists():
        raise FileNotFoundError(
            "Could not locate ESSOS_biot_savart_LandremanPaulQA.json. Set --coils-json explicitly."
        )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fci-essos] using coils file: {coils_json}")
    model, y0, map_meta = make_model_from_essos(
        coils_json=coils_json,
        nphi=int(args.nphi),
        nR=int(args.nR),
        nZ=int(args.nZ),
        dphi=float(args.dphi),
        radial_offset=float(args.radial_offset),
        radial_width=float(args.radial_width),
        vertical_halfwidth=float(args.vertical_halfwidth),
        dl_min=float(args.dl_min),
        sheath_model=str(args.sheath_model),
    )
    print(
        "[fci-essos] map built:",
        f"hits_fwd={map_meta.get('n_hit_fwd', 0.0):.0f},",
        f"hits_bwd={map_meta.get('n_hit_bwd', 0.0):.0f},",
        f"axis≈({map_meta['R_axis_est']:.3f}, {map_meta['Z_axis_est']:.3f}),",
        f"sheath_model={args.sheath_model}, dl_min={args.dl_min:.2e}",
    )

    ys, y_end = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=float(args.dt),
        nsteps=int(args.nsteps),
        save_every=int(args.save_every),
        solver="dopri5",
    )
    ts = float(args.dt) * jnp.arange(
        int(args.save_every), int(args.nsteps) + 1, int(args.save_every)
    )

    finite_mask = jnp.isfinite(ys.n).all(axis=(1, 2, 3)) & jnp.isfinite(ys.omega).all(
        axis=(1, 2, 3)
    )
    finite_mask = finite_mask & jnp.isfinite(ys.Te).all(axis=(1, 2, 3))
    if ys.Ti is not None:
        finite_mask = finite_mask & jnp.isfinite(ys.Ti).all(axis=(1, 2, 3))
    if ys.psi is not None:
        finite_mask = finite_mask & jnp.isfinite(ys.psi).all(axis=(1, 2, 3))
    if ys.N is not None:
        finite_mask = finite_mask & jnp.isfinite(ys.N).all(axis=(1, 2, 3))
    finite_idx = jnp.where(finite_mask)[0]
    if int(finite_idx.size) == 0:
        raise RuntimeError("No finite snapshots were produced in the ESSOS Biot-Savart run.")
    n_keep = int(finite_idx[-1]) + 1
    if n_keep < ys.n.shape[0]:
        print(
            f"[fci-essos] warning: non-finite state reached after saved frame {n_keep}; "
            "diagnostics truncated to finite prefix."
        )
    ys = FCIDRB3DFullState(
        n=ys.n[:n_keep],
        omega=ys.omega[:n_keep],
        vpar_e=ys.vpar_e[:n_keep],
        vpar_i=ys.vpar_i[:n_keep],
        Te=ys.Te[:n_keep],
        Ti=None if ys.Ti is None else ys.Ti[:n_keep],
        psi=None if ys.psi is None else ys.psi[:n_keep],
        N=None if ys.N is None else ys.N[:n_keep],
    )
    ts = ts[:n_keep]
    y_end = FCIDRB3DFullState(
        n=ys.n[-1],
        omega=ys.omega[-1],
        vpar_e=ys.vpar_e[-1],
        vpar_i=ys.vpar_i[-1],
        Te=ys.Te[-1],
        Ti=None if ys.Ti is None else ys.Ti[-1],
        psi=None if ys.psi is None else ys.psi[-1],
        N=None if ys.N is None else ys.N[-1],
    )

    energy = []
    n_rms = []
    sheath_pr = []
    sheath_er = []
    parallel_pr = []
    for i, t in enumerate(ts):
        yi = FCIDRB3DFullState(
            n=ys.n[i],
            omega=ys.omega[i],
            vpar_e=ys.vpar_e[i],
            vpar_i=ys.vpar_i[i],
            Te=ys.Te[i],
            Ti=None if ys.Ti is None else ys.Ti[i],
            psi=None if ys.psi is None else ys.psi[i],
            N=None if ys.N is None else ys.N[i],
        )
        pb = model.particle_budget_terms(yi)
        energy.append(float(model.energy(yi)))
        n_fluct = yi.n - jnp.mean(yi.n)
        n_rms.append(float(jnp.sqrt(jnp.mean(n_fluct**2))))
        sheath_pr.append(float(pb["sheath"]))
        parallel_pr.append(float(pb["parallel"]))
        _, esh = model.sheath_budget_rates(yi)
        sheath_er.append(float(esh))
    energy = jnp.asarray(energy)
    n_rms = jnp.asarray(n_rms)
    sheath_pr = jnp.asarray(sheath_pr)
    sheath_er = jnp.asarray(sheath_er)
    parallel_pr = jnp.asarray(parallel_pr)

    kz = y_end.n.shape[0] // 2
    n_mid = y_end.n[kz]
    omega_mid = y_end.omega[kz]

    hit_fwd = jnp.asarray(model.grid.map_fwd.hit, dtype=jnp.float64)
    if hit_fwd.ndim == 2:
        hit_fwd = hit_fwd[None, ...]
    hit_bwd = jnp.asarray(model.grid.map_bwd.hit, dtype=jnp.float64)
    if hit_bwd.ndim == 2:
        hit_bwd = hit_bwd[None, ...]
    hit_frac = jnp.mean(jnp.clip(hit_fwd + hit_bwd, 0.0, 1.0), axis=(1, 2))

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.6))
    ax = axes[0, 0]
    ax.plot(jnp.arange(hit_frac.size), hit_frac, "-o", ms=3)
    ax.set_xlabel("plane index")
    ax.set_ylabel("hit fraction")
    ax.set_title("Target-intersection fraction per plane")
    ax.grid(True, alpha=0.3)

    e_norm = jnp.maximum(jnp.abs(energy[0]), 1.0)
    ax = axes[0, 1]
    ax.plot(
        ts,
        (energy - energy[0]) / e_norm,
        lw=2.0,
        label=r"$(E-E_0)/\max(|E_0|,1)$",
    )
    ax.plot(ts, n_rms, lw=2.0, label=r"$\mathrm{rms}(n-\langle n \rangle)$")
    ax.set_xlabel("t")
    ax.set_title("Energy and fluctuation level")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 2]
    ax.plot(ts, parallel_pr, lw=2.0, label="parallel particle rate")
    ax.plot(ts, sheath_pr, lw=2.0, label="sheath particle rate")
    ax.plot(ts, sheath_er, lw=2.0, label="sheath energy rate")
    ax.set_xlabel("t")
    ax.set_ylabel("rate")
    ax.set_title("Target/sheath rate channels")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    im = ax.imshow(n_mid, origin="lower", cmap="coolwarm", aspect="equal")
    ax.set_title("final n (mid toroidal plane)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1, 1]
    im = ax.imshow(omega_mid, origin="lower", cmap="coolwarm", aspect="equal")
    ax.set_title(r"final $\Omega$ (mid toroidal plane)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1, 2]
    ax.axis("off")
    text = "\n".join(
        [
            f"axis estimate: R={map_meta['R_axis_est']:.3f}, Z={map_meta['Z_axis_est']:.3f}",
            f"R patch: [{map_meta['R_patch_min']:.3f}, {map_meta['R_patch_max']:.3f}]",
            f"Z patch: [{map_meta['Z_patch_min']:.3f}, {map_meta['Z_patch_max']:.3f}]",
            f"hit count fwd: {int(map_meta.get('n_hit_fwd', 0.0))}",
            f"hit count bwd: {int(map_meta.get('n_hit_bwd', 0.0))}",
            f"final rel energy drift: {float((energy[-1] - energy[0]) / e_norm):.3e}",
            f"final n rms: {float(n_rms[-1]):.3e}",
        ]
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle("FCI DRB3D on ESSOS Biot-Savart field (local edge/SOL patch)", fontsize=14)
    fig.tight_layout()
    out_png = out_dir / "fci_drb3d_full_essos_biotsavart.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    metrics = {
        "coils_json": str(coils_json),
        "out_figure": str(out_png),
        **map_meta,
        "final_rel_energy_drift": float((energy[-1] - energy[0]) / e_norm),
        "final_n_rms": float(n_rms[-1]),
        "median_parallel_particle_rate": float(jnp.median(parallel_pr)),
        "median_sheath_particle_rate": float(jnp.median(sheath_pr)),
    }
    out_json = out_dir / "fci_drb3d_full_essos_biotsavart_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[fci-essos] wrote {out_png}")
    print(f"[fci-essos] wrote {out_json}")


if __name__ == "__main__":
    main()
