"""Strict conservative-operator benchmark gate for cold-ion DRB.

This script enforces physics-regression thresholds on the periodic conservative subset of the
actual field-line cold-ion DRB branch:

1) instantaneous operator residuals (invariant rates from RHS),
2) finite-time invariant drifts under fixed-step RK4 evolution.

It is designed for CI gating and writes optional JSON metrics for artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.geometry.slab import SlabGeometry  # noqa: E402
from jaxdrb.models.cold_ion_drb import Equilibrium, State, rhs_nonlinear  # noqa: E402
from jaxdrb.models.invariants import (  # noqa: E402
    cold_ion_invariant_rates_from_rhs,
    cold_ion_invariants,
)
from jaxdrb.models.params import DRBParams  # noqa: E402
from jaxdrb.nonlinear.stepper import rk4_step  # noqa: E402


def _build_conservative_setup(*, nl: int) -> tuple[SlabGeometry, Equilibrium, DRBParams]:
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.2,
        alpha_Te_ohm=1.71,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        chi_par_Te=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        boussinesq=True,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )
    return geom, eq, params


def _operator_gate_metrics(
    *,
    nl: int,
    amplitude: float,
    ky_values: np.ndarray,
    nseeds: int,
) -> dict[str, float]:
    geom, eq, params = _build_conservative_setup(nl=nl)
    key = jax.random.key(0)
    keys = jax.random.split(key, int(nseeds))

    maxima = np.zeros(5, dtype=float)  # dE, dmass, dcharge, dcurrent, dmomentum
    for seed_key in keys:
        y = State.random(seed_key, nl, amplitude=amplitude)
        for ky in ky_values:
            dy = rhs_nonlinear(0.0, y, params, geom, kx=0.0, ky=float(ky), eq=eq)
            rates = cold_ion_invariant_rates_from_rhs(
                y, dy, params=params, geom=geom, kx=0.0, ky=float(ky), eq=eq
            )
            vals = np.asarray(
                [
                    jnp.abs(rates["denergy_dt"]),
                    jnp.abs(rates["dmass_dt"]),
                    jnp.abs(rates["dcharge_dt"]),
                    jnp.abs(rates["dcurrent_dt"]),
                    jnp.abs(rates["dmomentum_dt"]),
                ],
                dtype=float,
            )
            maxima = np.maximum(maxima, vals)

    return {
        "max_abs_denergy_dt": float(maxima[0]),
        "max_abs_dmass_dt": float(maxima[1]),
        "max_abs_dcharge_dt": float(maxima[2]),
        "max_abs_dcurrent_dt": float(maxima[3]),
        "max_abs_dmomentum_dt": float(maxima[4]),
    }


def _time_drift_metrics(
    *,
    nl: int,
    dt: float,
    nsteps: int,
    ky: float,
    amplitude: float,
) -> dict[str, float]:
    geom, eq, params = _build_conservative_setup(nl=nl)
    y0 = State.random(jax.random.key(123), nl, amplitude=amplitude)
    keys = ("energy", "mass", "charge", "current", "momentum")

    def rhs_local(t, y):
        return rhs_nonlinear(t, y, params, geom, kx=0.0, ky=ky, eq=eq)

    @jax.jit
    def evolve(y_init):
        inv0 = cold_ion_invariants(y_init, params=params, geom=geom, kx=0.0, ky=ky, eq=eq)
        vec0 = jnp.asarray([inv0[k] for k in keys], dtype=jnp.float64)

        def step(carry, _):
            t, y = carry
            y_next = rk4_step(y, t, dt, rhs_local)
            inv = cold_ion_invariants(y_next, params=params, geom=geom, kx=0.0, ky=ky, eq=eq)
            vec = jnp.asarray([inv[k] for k in keys], dtype=jnp.float64)
            return (t + dt, y_next), vec

        (_, _), hist = jax.lax.scan(
            step, (jnp.asarray(0.0, dtype=jnp.float64), y_init), xs=None, length=int(nsteps)
        )
        return jnp.vstack([vec0[None, :], hist])

    H = np.asarray(evolve(y0))
    E = H[:, 0]
    E0 = float(E[0])
    rel_span_E = float((np.max(E) - np.min(E)) / max(abs(E0), 1e-30))
    rel_end_E = float(abs(E[-1] - E0) / max(abs(E0), 1e-30))
    mean_drift = np.max(np.abs(H[:, 1:] - H[0:1, 1:]), axis=0)
    return {
        "rel_energy_span": rel_span_E,
        "rel_energy_end": rel_end_E,
        "max_abs_mass_drift": float(mean_drift[0]),
        "max_abs_charge_drift": float(mean_drift[1]),
        "max_abs_current_drift": float(mean_drift[2]),
        "max_abs_momentum_drift": float(mean_drift[3]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nl", type=int, default=64)
    parser.add_argument("--amplitude", type=float, default=1.0e-2)
    parser.add_argument("--nseeds", type=int, default=4)
    parser.add_argument("--ky-min", type=float, default=0.12)
    parser.add_argument("--ky-max", type=float, default=0.72)
    parser.add_argument("--nky", type=int, default=5)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--nsteps", type=int, default=1500)
    parser.add_argument("--ky-time", type=float, default=0.35)
    parser.add_argument("--max-abs-denergy-dt", type=float, default=5e-8)
    parser.add_argument("--max-abs-dmean-dt", type=float, default=1e-11)
    parser.add_argument("--max-rel-energy-span", type=float, default=3e-4)
    parser.add_argument("--max-rel-energy-end", type=float, default=3e-4)
    parser.add_argument("--max-abs-mean-drift", type=float, default=5e-10)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", True)

    ky_values = np.linspace(float(args.ky_min), float(args.ky_max), int(args.nky))
    operator = _operator_gate_metrics(
        nl=int(args.nl),
        amplitude=float(args.amplitude),
        ky_values=ky_values,
        nseeds=int(args.nseeds),
    )
    drift = _time_drift_metrics(
        nl=int(args.nl),
        dt=float(args.dt),
        nsteps=int(args.nsteps),
        ky=float(args.ky_time),
        amplitude=float(args.amplitude),
    )
    metrics = {"operator_gate": operator, "time_gate": drift}

    print(
        "[drb-conservative-gate] "
        f"max|dE/dt|={operator['max_abs_denergy_dt']:.3e}, "
        f"max|dmean/dt|={max(operator[k] for k in ('max_abs_dmass_dt', 'max_abs_dcharge_dt', 'max_abs_dcurrent_dt', 'max_abs_dmomentum_dt')):.3e}, "
        f"relE_span={drift['rel_energy_span']:.3e}, relE_end={drift['rel_energy_end']:.3e}",
        flush=True,
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2))

    failures: list[str] = []
    if operator["max_abs_denergy_dt"] > float(args.max_abs_denergy_dt):
        failures.append(
            f"operator gate failed: max|dE/dt|={operator['max_abs_denergy_dt']:.3e} > {float(args.max_abs_denergy_dt):.3e}"
        )
    for name in (
        "max_abs_dmass_dt",
        "max_abs_dcharge_dt",
        "max_abs_dcurrent_dt",
        "max_abs_dmomentum_dt",
    ):
        if operator[name] > float(args.max_abs_dmean_dt):
            failures.append(
                f"operator gate failed: {name}={operator[name]:.3e} > {float(args.max_abs_dmean_dt):.3e}"
            )
    if drift["rel_energy_span"] > float(args.max_rel_energy_span):
        failures.append(
            f"time gate failed: rel_energy_span={drift['rel_energy_span']:.3e} > {float(args.max_rel_energy_span):.3e}"
        )
    if drift["rel_energy_end"] > float(args.max_rel_energy_end):
        failures.append(
            f"time gate failed: rel_energy_end={drift['rel_energy_end']:.3e} > {float(args.max_rel_energy_end):.3e}"
        )
    for name in (
        "max_abs_mass_drift",
        "max_abs_charge_drift",
        "max_abs_current_drift",
        "max_abs_momentum_drift",
    ):
        if drift[name] > float(args.max_abs_mean_drift):
            failures.append(
                f"time gate failed: {name}={drift[name]:.3e} > {float(args.max_abs_mean_drift):.3e}"
            )
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
