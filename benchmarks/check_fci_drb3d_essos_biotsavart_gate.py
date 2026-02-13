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
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps  # noqa: E402


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
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--nsteps", type=int, default=600)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--max-rel-energy-drift", type=float, default=0.5)
    p.add_argument("--max-rel-particle-drift", type=float, default=0.5)
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
    )

    ys, _ = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=float(args.dt),
        nsteps=int(args.nsteps),
        save_every=int(args.save_every),
        solver="dopri5",
    )

    finite_ok = bool(
        jnp.isfinite(ys.n).all()
        and jnp.isfinite(ys.omega).all()
        and jnp.isfinite(ys.Te).all()
    )
    if ys.Ti is not None:
        finite_ok = finite_ok and bool(jnp.isfinite(ys.Ti).all())
    if ys.psi is not None:
        finite_ok = finite_ok and bool(jnp.isfinite(ys.psi).all())
    if ys.N is not None:
        finite_ok = finite_ok and bool(jnp.isfinite(ys.N).all())

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
    rel_energy_drift = float(jnp.abs((energy[-1] - energy[0]) / jnp.maximum(energy[0], 1e-12)))
    rel_particle_drift = float(
        jnp.abs((particles[-1] - particles[0]) / jnp.maximum(particles[0], 1e-12))
    )

    metrics = {
        "finite_ok": finite_ok,
        "rel_energy_drift": rel_energy_drift,
        "rel_particle_drift": rel_particle_drift,
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
