#!/usr/bin/env python3
"""FCI DRB3D ESSOS Biot-Savart stability gate (O(1) time)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "09_fci"))

try:
    from fci_drb3d_full_essos_biotsavart import (  # type: ignore  # noqa: E402
        _default_coils_file,
        make_model_from_essos,
    )
except Exception as exc:  # pragma: no cover - import check
    raise SystemExit(f"Failed to import ESSOS example helpers: {exc}") from exc

from jaxdrb.fci.drb3d_full import FCIDRB3DFullState  # noqa: E402
from jaxdrb.nonlinear.integrate import diffeqsolve  # noqa: E402


def _rms(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.mean(x**2))


def _state_norms(state: FCIDRB3DFullState) -> dict[str, float]:
    out = {
        "n": float(_rms(state.n)),
        "omega": float(_rms(state.omega)),
        "vpar_e": float(_rms(state.vpar_e)),
        "vpar_i": float(_rms(state.vpar_i)),
        "Te": float(_rms(state.Te)),
    }
    if state.Ti is not None:
        out["Ti"] = float(_rms(state.Ti))
    if state.psi is not None:
        out["psi"] = float(_rms(state.psi))
    if state.N is not None:
        out["N"] = float(_rms(state.N))
    return out


def _state_finite(state: FCIDRB3DFullState) -> bool:
    finite = bool(jnp.isfinite(state.n).all())
    finite = finite and bool(jnp.isfinite(state.omega).all())
    finite = finite and bool(jnp.isfinite(state.vpar_e).all())
    finite = finite and bool(jnp.isfinite(state.vpar_i).all())
    finite = finite and bool(jnp.isfinite(state.Te).all())
    if state.Ti is not None:
        finite = finite and bool(jnp.isfinite(state.Ti).all())
    if state.psi is not None:
        finite = finite and bool(jnp.isfinite(state.psi).all())
    if state.N is not None:
        finite = finite and bool(jnp.isfinite(state.N).all())
    return finite


def _split_norms(split) -> dict[str, dict[str, float]]:
    return {
        "conservative": _state_norms(split.conservative),
        "source": _state_norms(split.source),
        "dissipative": _state_norms(split.dissipative),
        "total": _state_norms(split.total()),
    }


def _finite_stats(arr: jnp.ndarray) -> dict[str, float | int | None]:
    arr = jnp.asarray(arr)
    finite = jnp.isfinite(arr)
    n_finite = int(jnp.sum(finite))
    n_total = int(arr.size)
    if n_finite == 0:
        return {"min": None, "max": None, "finite_frac": 0.0, "finite_count": 0, "total": n_total}
    minv = float(jnp.min(jnp.where(finite, arr, jnp.inf)))
    maxv = float(jnp.max(jnp.where(finite, arr, -jnp.inf)))
    return {
        "min": minv,
        "max": maxv,
        "finite_frac": float(n_finite) / float(n_total),
        "finite_count": n_finite,
        "total": n_total,
    }


def _fmt_stat(val: float | None) -> str:
    if val is None:
        return "nan"
    return f"{val:.3e}"


def _map_stats(model) -> dict[str, dict[str, float | int | None]]:
    fwd = model.grid.map_fwd
    bwd = model.grid.map_bwd
    stats = {
        "dl_fwd": _finite_stats(fwd.dl),
        "dl_bwd": _finite_stats(bwd.dl),
    }
    if fwd.dl_hit is not None:
        stats["dl_hit_fwd"] = _finite_stats(fwd.dl_hit)
    if bwd.dl_hit is not None:
        stats["dl_hit_bwd"] = _finite_stats(bwd.dl_hit)
    return stats


def _hit_stats(model) -> dict[str, float]:
    hit_fwd = jnp.asarray(model.grid.map_fwd.hit, dtype=bool)
    hit_bwd = jnp.asarray(model.grid.map_bwd.hit, dtype=bool)
    if hit_fwd.ndim == 2:
        hit_fwd = hit_fwd[None, ...]
    if hit_bwd.ndim == 2:
        hit_bwd = hit_bwd[None, ...]
    hit_any = hit_fwd | hit_bwd
    hit_both = hit_fwd & hit_bwd
    return {
        "hit_frac": float(jnp.mean(hit_any)),
        "xpoint_frac": float(jnp.mean(hit_both)),
    }


def _sheath_stats(model) -> dict[str, float]:
    mask, sign = model._sheath_mask_sign()
    return {
        "sheath_mask_frac": float(jnp.mean(mask)),
        "sheath_sign_mean": float(jnp.mean(sign)),
    }


def _n_eff_stats(model, y: FCIDRB3DFullState) -> dict[str, float]:
    if model.params.boussinesq:
        n_eff = jnp.full_like(y.n, float(model.params.n0))
    else:
        n_eff = jnp.asarray(float(model.params.n0), dtype=y.n.dtype)
        if model.params.non_boussinesq_perturbed_density_on:
            n_eff = n_eff + y.n
        n_eff = jnp.maximum(n_eff, jnp.asarray(float(model.params.n0_min), dtype=y.n.dtype))
    return {
        "n_eff_min": float(jnp.min(n_eff)),
        "n_eff_max": float(jnp.max(n_eff)),
    }


def _state_from_ys(ys, idx: int) -> FCIDRB3DFullState:
    return FCIDRB3DFullState(
        n=ys.n[idx],
        omega=ys.omega[idx],
        vpar_e=ys.vpar_e[idx],
        vpar_i=ys.vpar_i[idx],
        Te=ys.Te[idx],
        Ti=None if ys.Ti is None else ys.Ti[idx],
        psi=None if ys.psi is None else ys.psi[idx],
        N=None if ys.N is None else ys.N[idx],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coils-json", type=str, default="")
    p.add_argument("--nphi", type=int, default=8)
    p.add_argument("--nR", type=int, default=8)
    p.add_argument("--nZ", type=int, default=8)
    p.add_argument("--dphi", type=float, default=0.12)
    p.add_argument("--radial-offset", type=float, default=0.08)
    p.add_argument("--radial-width", type=float, default=0.18)
    p.add_argument("--vertical-halfwidth", type=float, default=0.14)
    p.add_argument("--dl-min", type=float, default=6e-2)
    p.add_argument("--sheath-model", choices=["simple", "loizu_linear"], default="simple")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--nsteps", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--max-rel-energy-drift", type=float, default=1.0)
    p.add_argument("--max-rel-particle-drift", type=float, default=1.0)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", True)

    coils_json = Path(args.coils_json) if args.coils_json else _default_coils_file()
    if coils_json is None or not coils_json.exists():
        raise SystemExit("ESSOS Biot-Savart coils JSON not found.")

    model, y0, _ = make_model_from_essos(
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
        stable_gate=True,
    )

    split0 = model.rhs_decomposed(0.0, y0)
    rhs0_finite = _state_finite(split0.total())
    rhs0_norms = _split_norms(split0)
    map_stats = _map_stats(model)
    hit_stats = _hit_stats(model)
    sheath_stats = _sheath_stats(model)
    n_eff_stats = _n_eff_stats(model, y0)

    print(
        "[fci-essos-gate] "
        f"rhs0_finite={rhs0_finite} "
        f"hit_frac={hit_stats['hit_frac']:.3f} "
        f"xpoint_frac={hit_stats['xpoint_frac']:.3f} "
        f"sheath_mask_frac={sheath_stats['sheath_mask_frac']:.3f} "
        f"dl_fwd_min={_fmt_stat(map_stats['dl_fwd']['min'])} "
        f"dl_fwd_max={_fmt_stat(map_stats['dl_fwd']['max'])} "
        f"dl_bwd_min={_fmt_stat(map_stats['dl_bwd']['min'])} "
        f"dl_bwd_max={_fmt_stat(map_stats['dl_bwd']['max'])} "
        f"n_eff=[{n_eff_stats['n_eff_min']:.3e},{n_eff_stats['n_eff_max']:.3e}]"
    )
    if not rhs0_finite:
        raise SystemExit("RHS contains NaN/Inf at t=0.")

    dt = float(args.dt)
    nsteps = int(args.nsteps)
    t1 = dt * nsteps
    save_ts = dt * jnp.arange(int(args.save_every), nsteps + 1, int(args.save_every))
    sol = diffeqsolve(
        model.rhs,
        y0=y0,
        t0=0.0,
        t1=float(t1),
        dt0=dt,
        save_ts=save_ts,
        solver="kvaerno4",
        adaptive=True,
        rtol=1e-5,
        atol=1e-8,
        max_steps=400_000,
        progress=False,
    )
    ys = sol.ys

    finite_mask = jnp.isfinite(ys.n).all(axis=(1, 2, 3))
    finite_mask = finite_mask & jnp.isfinite(ys.omega).all(axis=(1, 2, 3))
    finite_mask = finite_mask & jnp.isfinite(ys.Te).all(axis=(1, 2, 3))
    if ys.Ti is not None:
        finite_mask = finite_mask & jnp.isfinite(ys.Ti).all(axis=(1, 2, 3))
    if ys.psi is not None:
        finite_mask = finite_mask & jnp.isfinite(ys.psi).all(axis=(1, 2, 3))
    if ys.N is not None:
        finite_mask = finite_mask & jnp.isfinite(ys.N).all(axis=(1, 2, 3))
    finite_ok = bool(jnp.all(finite_mask))
    bad_idx = None
    if not finite_ok:
        bad_idx = int(jnp.where(~finite_mask)[0][0])
        print(f"[fci-essos-gate] warning: first non-finite snapshot at index {bad_idx}.")

    n_keep = int(ys.n.shape[0] if bad_idx is None else max(bad_idx, 0))
    if n_keep == 0:
        raise SystemExit("No finite snapshots were produced in the ESSOS Biot-Savart gate run.")
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

    energy = []
    particles = []
    for i in range(ys.n.shape[0]):
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
        energy.append(float(model.energy(yi)))
        particles.append(float(model.total_particle_content(yi)))
    energy = jnp.asarray(energy)
    particles = jnp.asarray(particles)
    energy_scale = jnp.maximum(jnp.maximum(jnp.abs(energy[0]), jnp.abs(energy[-1])), 1e-12)
    particle_scale = jnp.maximum(jnp.maximum(jnp.abs(particles[0]), jnp.abs(particles[-1])), 1e-6)
    rel_energy_drift = float(jnp.abs((energy[-1] - energy[0]) / energy_scale))
    rel_particle_drift = float(jnp.abs((particles[-1] - particles[0]) / particle_scale))
    rhs_last_norms = None
    if n_keep > 0:
        y_last = _state_from_ys(ys, n_keep - 1)
        t_last = float(dt * int(args.save_every) * n_keep)
        rhs_last = model.rhs_decomposed(t_last, y_last)
        rhs_last_norms = _split_norms(rhs_last)

    metrics = {
        "finite_ok": finite_ok,
        "rhs0_finite": rhs0_finite,
        "rel_energy_drift": rel_energy_drift,
        "rel_particle_drift": rel_particle_drift,
        "map_stats": map_stats,
        "hit_stats": hit_stats,
        "sheath_stats": sheath_stats,
        "n_eff_stats": n_eff_stats,
        "rhs0_norms": rhs0_norms,
        "rhs_last_norms": rhs_last_norms,
        "first_bad_index": bad_idx,
    }
    print(
        "[fci-essos-gate] "
        f"finite={metrics['finite_ok']} "
        f"rel_energy_drift={metrics['rel_energy_drift']:.3e} "
        f"rel_particle_drift={metrics['rel_particle_drift']:.3e}"
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    failures: list[str] = []
    if not metrics["finite_ok"]:
        failures.append("solution contains NaN/Inf")
    if metrics["rel_energy_drift"] > float(args.max_rel_energy_drift):
        failures.append(
            f"rel_energy_drift {metrics['rel_energy_drift']:.3e} > {args.max_rel_energy_drift:.3e}"
        )
    if metrics["rel_particle_drift"] > float(args.max_rel_particle_drift):
        failures.append(
            f"rel_particle_drift {metrics['rel_particle_drift']:.3e} > {args.max_rel_particle_drift:.3e}"
        )

    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
